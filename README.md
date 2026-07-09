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

The index works over **one contiguous byte buffer** — the concatenation of all
your documents — plus the **document boundaries**, so every match maps back to the
document it came from.

```
docs = [ "doc A text...", "doc B text...", "doc C text..." ]

concat      = "doc A text...doc B text...doc C text..."
doc_start   = [ 0,           13,           26 ]   // byte offset where each doc begins
```

- A "document" is whatever unit you want results attributed to and to remove/act
  on — a training document, a log line, a paragraph, a chunk, a DNA read.
- `doc_start[i]` is the byte offset where document `i` begins in `concat` (sorted,
  starts at 0). Pass it via `SetDocStarts`; then `LocateDocs` returns
  `(doc_id, offset_within_doc)` for every hit. If you don't set it, the whole
  buffer is treated as one document.
- Bytes are arbitrary — UTF-8 text, raw logs, `{A,C,G,T}`, anything. Matching is
  byte-exact and case-sensitive.
- One index handles corpora up to **2^63 bytes**: the suffix array is built with
  32-bit indices under 2 GiB (compact, less build memory) and 64-bit indices at or
  above it — chosen automatically from `len`. Sampled positions likewise serialize
  4 bytes wide under 4 GiB, 8 above, so a small corpus keeps a small index. Going
  past 2 GiB in one index is mostly limited by **build memory** (~12× the corpus),
  so for large data prefer one index per shard/segment (this is how it plugs into
  Milvus, one index per sealed segment).
- **Document boundaries are respected on attribution.** Because documents are
  concatenated, a pattern can straddle a seam (e.g. docs `"ab"`,`"cd"` — `"bc"`
  exists in the buffer but in no document). Raw `Count`/`Locate` work over the
  concatenated buffer and *do* see such matches; the document-attributed queries
  (`LocateDocs`, `LocatePrefixDocs`, `LocateSuffixDocs`) **exclude** them, so they
  never report a match that no single document contains. If you need `Count`
  itself to be per-document exact, separate documents with a byte your queries
  never use.

## Quick start

```cpp
#include "index/fmindex/FMIndex.h"
using namespace milvus::index::fmindex;

// 1. concatenate your documents and record their offsets
std::string concat;
std::vector<uint64_t> doc_start;
for (const std::string& d : docs) { doc_start.push_back(concat.size()); concat += d; }

// 2. build
//    Build(data, len, sa_sample_rate = 32)
//      data           first byte of the concatenated buffer
//      len            its length in BYTES (< 2^31; shard larger corpora)
//      sa_sample_rate SA sampling rate: space/locate trade-off, no effect on
//                     Count — default 32 is balanced; 4-8 = faster Locate,
//                     bigger index; 64+ = smaller index if you only Count
FMIndex fm;
fm.Build(reinterpret_cast<const uint8_t*>(concat.data()), concat.size());  // sa_sample_rate defaults to 32
fm.SetDocStarts(doc_start);

// 3. query
auto n   = fm.Count (P, plen);         // how many times P occurs
auto pos = fm.Locate(P, plen);         // sorted text positions
auto hit = fm.LocateDocs(P, plen);     // sorted {doc_id, offset} per occurrence
```

## Workflow: decontamination (the primary use case)

```cpp
// index the training corpus once
FMIndex fm;
fm.Build(corpus_ptr, corpus_len, 32);
fm.SetDocStarts(doc_start);

// extract n-grams (e.g. 13-grams) from every benchmark item, then score them all
// at once — batched is far faster than one-by-one (see below)
std::vector<std::pair<const uint8_t*, size_t>> ngrams = extract_ngrams(benchmarks);
std::vector<size_t> counts = fm.CountBatch(ngrams);

// any n-gram with count > 0 is a leak; LocateDocs tells you which documents to drop
for (size_t i = 0; i < ngrams.size(); ++i)
    if (counts[i] > 0)
        for (auto [doc_id, off] : fm.LocateDocs(ngrams[i].first, ngrams[i].second))
            contaminated_docs.insert(doc_id);
```

**Use `CountBatch` for bulk work.** Scoring many patterns at once runs their
searches in lock-step so memory latency overlaps — several× the throughput of
calling `Count` in a loop. It is the right primitive for decontamination
(thousands of n-grams) and batch log scans.

## API

| Call | Returns |
|---|---|
| `Build(data, len, sa_sample_rate=32, case_insensitive=false)` | build the index over a byte buffer |
| `SetDocStarts(offsets)` | document boundaries (optional; default = one doc) |
| `Count(pat, len)` | exact occurrence count |
| `CountBatch(patterns)` | counts for many patterns, high throughput |
| `Locate(pat, len)` | sorted text positions of every occurrence |
| `LocateDocs(pat, len)` | sorted `(doc_id, offset)` of every occurrence |
| `LocatePrefixDocs(pat, len)` / `CountPrefixDocs` | documents that **begin** with `pat` |
| `LocateSuffixDocs(pat, len)` / `CountSuffixDocs` | documents that **end** with `pat` |
| `Extract(pos, len)` | recover the original bytes `T[pos, pos+len)` (match context) |
| `LongestMatch(query, qlen)` | longest substring of `query` present in the corpus (fuzzy overlap) |
| `NextTokenCounts(pat, len)` | distribution of the byte following `pat` (n-gram model) |
| `Serialize()` / `SerializeToFile(path)` | persist the index |
| `Deserialize(blob)` / `LoadView(base, size)` | load (copy / zero-copy mmap) |

**Anchored matching.** `Count`/`Locate` find `pat` *anywhere* (substring). To match
only at document boundaries — "log lines starting with `ERROR`", "files ending in
`.log`" — use `LocatePrefixDocs` / `LocateSuffixDocs`; they return the sorted, unique
document ids. Both need `SetDocStarts` to be meaningful.

**Case-insensitive.** Pass `case_insensitive=true` to `Build`: ASCII `A-Z` are folded
to `a-z` at build time, so every query matches case-insensitively at **zero query-time
cost** (`Locate` offsets still point into the original text). Only ASCII case is
folded — non-ASCII / UTF-8 bytes stay exact.

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

**Match context.** `Extract(pos, len)` rebuilds the original bytes from the index
(no need to keep the source text around) — e.g. to show the line around a hit from
`Locate`. Costs `O(len + sa_sample_rate)`. On a `case_insensitive` index, ASCII
letters come back lowercased (original case isn't stored).

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
