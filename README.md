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
concatenates them internally and injects a `\0` separator after each, so every
result is **document-scoped (row semantics)**: no query can ever match across a
document boundary.

```
docs = [ "doc A text...", "doc B text...", "doc C text..." ]

internal = "doc A text...\0doc B text...\0doc C text...\0"   // separators injected for you
```

- A "document" is whatever unit you want results attributed to and to act on — a
  training document, a table row, a log line, a paragraph, a chunk, a DNA read.
  Document `i` (0-based, in the order you pass) is what every result maps back to.
- **You own nothing but the split.** `LocateDocs` returns `(doc_id,
  offset_within_doc)`; `MatchingDocs`/`FuzzyMatchingDocs` return doc ids; `Extract`
  takes `(doc_id, offset, len)`. No global offsets to manage.
- **Contents must not contain `\0`** — that byte is the separator. All valid UTF-8
  text qualifies (UTF-8 never encodes a NUL except for U+0000 itself). Everything
  else is arbitrary bytes — raw logs, `{A,C,G,T}`, binary. Matching is byte-exact
  (optionally ASCII-case-insensitive).
- **No cross-document matches — ever, in any operation.** Because a pattern free of
  `\0` cannot span a separator, a substring that would straddle two documents (e.g.
  docs `"ab"`,`"cd"` — `"bc"`) simply does not exist in the index. This holds for
  *every* primitive, `Count` included — not just the document-attributed ones — so
  `Count` is per-document exact **and** still O(1) in the match interval.
- One index handles corpora up to **2^63 bytes**: the suffix array is built with
  32-bit indices under 2 GiB (compact, less build memory) and 64-bit indices at or
  above it — chosen automatically. Sampled positions likewise serialize 4 bytes
  wide under 4 GiB, 8 above, so a small corpus keeps a small index. Going past
  2 GiB in one index is mostly limited by **build memory** (~8× the corpus), so for
  large data prefer one index per shard/segment (this is how it plugs into Milvus,
  one index per sealed segment).

## Quick start

```cpp
#include "index/fmindex/FMIndex.h"
using namespace milvus::index::fmindex;

// 1. build straight from your split documents — the index concatenates them and
//    injects the '\0' separators for you.
//    Build(docs, sa_sample_rate = 32, case_insensitive = false)
//      docs           your documents, in order (document i gets doc id i);
//                     contents must not contain '\0'
//      sa_sample_rate SA sampling rate: space/locate trade-off, no effect on
//                     Count — default 32 is balanced; 4-8 = faster Locate,
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
| `Locate(pat, len)` | sorted positions (separator-free global offsets) |
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

**Anchored matching.** `Count`/`Locate`/`MatchingDocs` find `pat` *anywhere* in a
document (substring). To match only at document boundaries — "log lines starting
with `ERROR`", "files ending in `.log`" — use `LocatePrefixDocs` /
`LocateSuffixDocs`; they return the sorted, unique document ids.

**Case-insensitive.** Pass `case_insensitive=true` to `Build`: ASCII `A-Z` are folded
to `a-z` at build time, so every query matches case-insensitively at **zero query-time
cost** (`Locate` offsets still point into the original text). Only ASCII case is
folded — non-ASCII / UTF-8 bytes stay exact.

**Document-granularity filtering (for a scalar index).** A columnar filter wants a
per-row boolean — *which rows contain this pattern* — not positions. `MatchingDocs(pat)`
returns the sorted, unique doc ids that contain `pat` exactly (`col LIKE '%pat%'`), and
`FuzzyMatchingDocs(pat, k)` returns the doc ids that contain a substring within **edit
distance ≤ k** of `pat` (typo / variant tolerant — names, domains, codes). Both are
document-scoped by construction — a match always lies inside one document, since the
injected `\0` separators keep any substring from spanning two. `FuzzyMatchingDocs` is a
backtracking backward search (substitution / insertion / deletion against an error
budget) that never steps through a separator, so its cost grows fast with `k` and the
alphabet — `k` is meant to be small (1–2) over short patterns; `k == 0` is exactly
`MatchingDocs`.

**Token-level index (`TokenFMIndex`).** The byte `FMIndex` matches byte substrings.
For LLM decontamination the unit is *N consecutive tokens* (e.g. a 13-token overlap,
GPT-3 style), so `TokenFMIndex` builds the same structure over a `uint32` token-id
array instead of bytes — same methods (`Count`/`Locate`/`LocateDocs`/`LongestMatch`/
`NextTokenCounts`/`Extract`), but lengths and positions are in tokens, `Extract`
returns token ids, and `NextTokenCounts` returns `(token_id, count)` — a real
unbounded-context **token n-gram model** (Infini-gram-style). You tokenize the corpus
and queries with your own tokenizer; the index has no tokenizer dependency. (In-memory
build + query today; serialization and a shared templated core with the byte index are
follow-ups.)

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
./build/bench_fmindex    # build / count / batch / locate / size
./build/bench_mmap       # in-RAM vs mmap query throughput
```

## Layout

```
src/index/fmindex/   the library (headers + FMIndex.cpp); ports into Milvus core
                     (TokenFMIndex.h = token-level variant over uint32 token ids)
third_party/libsais/ vendored suffix-array builder, 32- and 64-bit (Apache-2.0)
test/                dependency-free brute-force-oracle tests
bench/               self-contained benchmark harnesses (build / count / mmap)
DESIGN.md            architecture, algorithms, correctness, serialization format
BENCHMARK.md         measured numbers vs sdsl-lite and Lance
```

## Licensing

Apache-2.0. Structures are clean-room from Apache-2.0 / MIT sources and papers
(libsais vendored; rank9 / wavelet-matrix / quad-matrix clean-room). No GPL code
is copied; sdsl-lite is used only as a benchmark/correctness reference. See
[`DESIGN.md`](DESIGN.md) → Provenance.
