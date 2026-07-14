# fm-index-lite

A small, fast C++17 FM-index for **exact substring search over large text**. Given
a body of documents, it answers, for any pattern — *does it occur, how many times,
in which documents, at what offsets* — exactly (no false positives), and in bulk.

It is built for three concrete jobs:

1. **Training-data decontamination.** "Does this benchmark question (or 13-gram)
   appear in my training corpus, how many times, and in which documents?" Score
   thousands of n-grams against a multi-GB corpus and get exact counts + the
   documents to remove.
2. **Exact log / text search.** `contains(text, 'error-9F2')` over large VARCHAR/
   TEXT columns — exact, complete, with counts and positions, not a sampled
   approximation.
3. **Sequence / dedup matching.** Any "find every occurrence of this needle in
   that haystack" over bytes (e.g. genomic reads).

On general text it is **faster than the reference FM-index (sdsl-lite)** on build,
count, and locate — up to several× on batched count — at equal index size. See
[`BENCHMARK.md`](BENCHMARK.md). Internals and design decisions are in
[`DESIGN.md`](DESIGN.md).

---

## Modeling your business data

You feed the index your documents **already split** — a list of byte strings. It
concatenates them internally and injects a separator **symbol** after each, so
every result is **document-scoped (row semantics)**: no query can ever match
across a document boundary.

```
docs = [ "doc A text...", "doc B text...", "doc C text..." ]

internal = "doc A text..." ⎵ "doc B text..." ⎵ "doc C text..." ⎵   // ⎵ = separator symbol (not a byte)
```

- A "document" is whatever unit you want results attributed to and to act on — a
  training document, a table row, a log line, a paragraph, a chunk, a DNA read.
  Document `i` (0-based, in the order you pass) is what every result maps back to.
- **You own nothing but the split.** `LocateDocs` returns `(doc_id,
  offset_within_doc)`; `MatchingDocs`/`FuzzyMatchingDocs` return doc ids; `Extract`
  takes `(doc_id, offset, len)`. No global offsets to manage.
- **Contents may be ANY bytes — `\0` included.** The separator is a symbol
  *outside* the byte alphabet (an internal dense id), so no byte value is reserved:
  raw logs, `{A,C,G,T}`, binary, embedded NULs all index and match byte-exactly
  (optionally ASCII-case-insensitive).
- **No cross-document matches — ever, in any operation.** No query byte maps to the
  separator symbol, so a substring that would straddle two documents (e.g. docs
  `"ab"`,`"cd"` — `"bc"`) simply does not exist in the index — and this holds even
  when the boundary byte would be `\0`. It applies to *every* primitive, `Count`
  included, so `Count` is per-document exact **and** still O(1) in the match interval.
- One index handles corpora up to **2^63 bytes**: the suffix array is built with
  32-bit indices under 2 GiB (compact, less build memory) and 64-bit indices at or
  above it — chosen automatically. Sampled positions likewise serialize 4 bytes
  wide under 4 GiB, 8 above, so a small corpus keeps a small index. Going past
  2 GiB in one index is mostly limited by **build memory** (~10× the corpus with
  the int32 symbol-text SA path), so for large data prefer one index per
  shard/segment (this is how it plugs into Milvus, one index per sealed segment).

## Quick start

```cpp
#include "index/fmindex/FMIndex.h"
using namespace milvus::index::fmindex;

// 1. build straight from your split documents — the index concatenates them and
//    injects the separator symbols for you.
//    Build(docs, sa_sample_rate = 32, case_insensitive = false)
//      docs           your documents, in order (document i gets doc id i);
//                     contents may be any bytes, '\0' included
//      sa_sample_rate SA sampling rate: space/locate trade-off, no effect on
//                     Count — default 32 is balanced; 4-8 = faster LocateDocs,
//                     bigger index; 64+ = smaller index if you only Count
std::vector<std::string_view> docs = { "doc A text...", "doc B text...", ... };
FMIndex fm;
fm.Build(docs);                        // sa_sample_rate defaults to 32

// 2. query — everything is per-document
auto n    = fm.Count(P, plen);         // per-document exact occurrence count
auto hit  = fm.LocateDocs(P, plen);    // sorted {doc_id, offset} per occurrence
auto ids  = fm.MatchingDocs(P, plen);  // sorted doc ids containing P (LIKE '%P%')
```

## Workflow: decontamination (the primary use case)

```cpp
// index the training corpus once — one document per training example
FMIndex fm;
fm.Build(training_docs);  // std::vector<std::string_view>

// extract n-grams (e.g. 13-grams) from every benchmark item, then score them all
// at once — batched is far faster than one-by-one (see below)
std::vector<std::pair<const uint8_t*, size_t>> ngrams = extract_ngrams(benchmarks);
std::vector<size_t> counts = fm.CountBatch(ngrams);

// any n-gram with count > 0 is a leak; MatchingDocs tells you which documents to drop
for (size_t i = 0; i < ngrams.size(); ++i)
    if (counts[i] > 0)
        for (uint64_t doc_id : fm.MatchingDocs(ngrams[i].first, ngrams[i].second))
            contaminated_docs.insert(doc_id);
```

**Use `CountBatch` for bulk work.** Scoring many patterns at once runs their
searches in lock-step so memory latency overlaps — several× the throughput of
calling `Count` in a loop. It is the right primitive for decontamination
(thousands of n-grams) and batch log scans.

## API

| Call | Returns |
|---|---|
| `Build(docs, sa_sample_rate=32, case_insensitive=false)` | build over a list of documents (`std::vector<std::string_view>`) |
| `Count(pat, len)` | per-document exact occurrence count |
| `CountBatch(patterns)` | counts for many patterns, high throughput |
| `LocateDocs(pat, len)` | sorted `(doc_id, offset_within_doc)` of every occurrence |
| `MatchingDocs(pat, len)` | sorted, unique **doc ids** containing `pat` (exact `LIKE '%pat%'`) |
| `FuzzyMatchingDocs(pat, len, k)` | sorted, unique **doc ids** containing a substring within **edit distance ≤ k** of `pat` |
| `LocatePrefixDocs(pat, len)` / `CountPrefixDocs` | documents that **begin** with `pat` |
| `LocateSuffixDocs(pat, len)` / `CountSuffixDocs` | documents that **end** with `pat` |
| `Extract(doc_id, offset, len)` | recover the original bytes of a document (match context) |
| `LongestMatch(query, qlen)` | longest substring of `query` present in the corpus (fuzzy overlap) |
| `NextTokenCounts(pat, len)` | distribution of the byte following `pat` (n-gram model) |
| `Serialize()` / `SerializeToFile(path)` | persist the index |
| `Deserialize(blob)` / `LoadView(base, size)` | load (copy / zero-copy mmap) |

**Anchored matching.** `Count`/`LocateDocs`/`MatchingDocs` find `pat` *anywhere* in a
document (substring). To match only at document boundaries — "log lines starting
with `ERROR`", "files ending in `.log`" — use `LocatePrefixDocs` /
`LocateSuffixDocs`; they return the sorted, unique document ids.

**Case-insensitive.** Pass `case_insensitive=true` to `Build`: ASCII `A-Z` are folded
to `a-z` at build time, so every query matches case-insensitively at **zero query-time
cost** (`LocateDocs` offsets still point into the original text). Only ASCII case is
folded — non-ASCII / UTF-8 bytes stay exact.

**Document-granularity filtering (for a scalar index).** A columnar filter wants a
per-row boolean — *which rows contain this pattern* — not positions. `MatchingDocs(pat)`
returns the sorted, unique doc ids that contain `pat` exactly (`col LIKE '%pat%'`), and
`FuzzyMatchingDocs(pat, k)` returns the doc ids that contain a substring within **edit
distance ≤ k** of `pat` (typo / variant tolerant — names, domains, codes). Both are
document-scoped by construction — a match always lies inside one document, since the
injected separator symbols keep any substring from spanning two. `FuzzyMatchingDocs` is a
backtracking backward search (substitution / insertion / deletion against an error
budget) that never steps through a separator, so its cost grows fast with `k` and the
alphabet — `k` is meant to be small (1–2) over short patterns; `k == 0` is exactly
`MatchingDocs`.

**Fuzzy contamination & n-gram stats.** Exact-13-gram matching misses paraphrased or
truncated contamination. `LongestMatch(query)` answers "how long a contiguous span of
this benchmark item appears in the corpus" — a partial-overlap signal that survives
edits. `NextTokenCounts(P)` gives the distribution of the byte that follows `P`,
turning the corpus into an unbounded-context n-gram model — `P(next=b | P) = count_b /
sum` — for probabilistic contamination scoring or corpus statistics. Both use only
backward search, no extra structure.

**Match context.** `Extract(doc_id, offset, len)` rebuilds the original bytes of a
document from the index (no need to keep the source text around) — e.g. to show the
line around a hit from `LocateDocs`. It clamps to the document's end (never crosses
into the next one). Costs `O(len + sa_sample_rate)`. On a `case_insensitive` index,
ASCII letters come back lowercased (original case isn't stored).

**Concurrency:** all queries are `const` and lock-free — many threads may query one
index at once (verified: an 8-thread stress test matches single-threaded results and
is ThreadSanitizer-clean). **Immutable:** build-once; to update, rebuild (or, at
scale, add a new shard and mask deletes at query time). **Serving:** `LoadView` maps
the index zero-copy from an mmap'd file — warm query throughput equals in-RAM. A
truncated or corrupt blob is rejected: `Deserialize`/`LoadView` yield an index
whose `valid()` is false (check it), and `MappedFMIndex::Open` returns `nullptr`.

Not supported (by design): regex, arbitrary lexicographic range between two different
strings (needs forward navigation), incremental update, Unicode-aware case folding.

**Parallel build.** The index is immutable and each `Build` is independent, so the
natural way to use many cores is to **build shards concurrently** — one thread per
segment index (measured ~4.4× on 8 shards, no extra dependency; queries then union
across shards). To speed up a *single* large index instead, configure with
`-DFMINDEX_OPENMP=ON` to parallelize the suffix-array construction via libsais
OpenMP (~2× on that phase, ~1.4× on the whole build). It is **off by default**, so
the default build has no OpenMP dependency and is single-threaded.

## Build & test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
./build/test_fmindex     # unit tests: fuzz vs brute force, concurrency, Extract, 32/64-bit, corrupt-blob
./build/demo             # tiny walkthrough of the anchored + case-insensitive API
./build/bench_fmindex    # build / count / batch / locate / size (4 MB sweep)
./build/bench_mmap       # in-RAM vs mmap query throughput
./build/bench_gig [GiB]  # 1 GiB document-scoped build + query, peak RSS
```

Measured numbers — the 4 MB sweep vs sdsl-lite and a **1 GiB document-scoped run**
(build 44.6 s, index 1.01× corpus, Count 137 K qps / batched ~1 M qps) — are in
[`BENCHMARK.md`](BENCHMARK.md).

## Layout

```
src/index/fmindex/   the library (headers + FMIndex.cpp); ports into Milvus core
third_party/libsais/ vendored suffix-array builder, 32- and 64-bit (Apache-2.0)
test/                dependency-free brute-force-oracle tests
bench/               self-contained benchmark harnesses (4 MB sweep, mmap, 1 GiB)
DESIGN.md            architecture, algorithms, correctness, serialization format
BENCHMARK.md         measured numbers: 4 MB vs sdsl-lite/Lance, and a 1 GiB run
```

## Licensing

Apache-2.0. Structures are clean-room from Apache-2.0 / MIT sources and papers
(libsais vendored; rank9 / wavelet-matrix / quad-matrix clean-room). No GPL code
is copied; sdsl-lite is used only as a benchmark/correctness reference. See
[`DESIGN.md`](DESIGN.md) → Provenance.
