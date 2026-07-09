# fm-index-lite — Design

A self-contained C++17 FM-index for **exact substring search** over byte data:
occurrence **count**, **locate** (positions), and **(document, offset)** mapping,
with batched throughput for bulk workloads. Built as the prototype for a Milvus
`FMINDEX` scalar index (see `../milvus/docs/design-docs/design_docs/20260708-fm_index_scalar_index.md`),
but usable standalone for decontamination, log search, or sequence matching.

Everything is translated clean-room from permissively-licensed sources
(Apache-2.0 / MIT / paper descriptions); no GPL code is copied. See "Provenance".

## 1. What it computes

Given a byte buffer `T` of length `n` (the concatenation of many documents), the
index answers, for any pattern `P`:

- `Count(P)` — number of occurrences of `P` in `T` (exact, no false positives).
- `Locate(P)` — the sorted text positions of every occurrence.
- `LocateDocs(P)` — sorted `(doc_id, offset_within_doc)` of every occurrence.
- `CountBatch({P_i})` — counts for many patterns at once, at much higher
  throughput (the bulk-decontamination path).

Matching is **byte-exact and case-sensitive**. Regex/prefix/range are out of
scope (a different index serves those).

## 2. Encoding

- **Dense alphabet.** The distinct bytes of `T` are mapped, in ascending byte
  order, to ids `1..σ-1`; id `0` is a **sentinel**. So `σ = (#distinct bytes) + 1`.
  Order-preservation keeps the suffix/BWT ordering correct. A byte absent from
  `T` maps to `-1`, so a pattern containing it counts 0 immediately.
- **Sentinel.** One sentinel `0` is appended: `t = map(T) · [0]`, length `m = n+1`.
  Being the unique smallest symbol, it makes all suffixes distinct and anchors the
  BWT.
- **Why dense:** the wavelet structure's depth is `⌈log₂ σ⌉` (binary) or
  `⌈log₂ σ / 2⌉` (quad). Remapping to the *actual* alphabet means DNA (5 symbols)
  uses 3 binary / 2 quad levels instead of 9 / 5 — fewer levels ⇒ fewer dependent
  cache misses per query and a smaller index.

## 3. Components

```
FMIndex
 ├─ SuffixArray   (libsais_int)                 build the SA of t
 ├─ WaveletMatrix4(QuadVector per level)         rank over the BWT   ← query hot path
 │    └─ QuadVector (2-bit packed + SWAR rank + 2-level directory)
 ├─ BitVector     (rank9)                        the sampled-row bitmap
 ├─ C-table       (uint32[σ])                    LF-mapping base
 ├─ first_[σ]     (per-symbol map_zero baseline) 2 ranks/level, not 3
 ├─ sampled SA    (uint32[], by row)             locate
 └─ doc_start[]   (uint64[], sorted)             (pk, offset) mapping
```

### 3.1 SuffixArray — `SuffixArray.h`
Thin wrapper over **libsais** (`libsais_int`, SA-IS, linear time, Apache-2.0,
vendored in `third_party/libsais`). Input is the int32 symbol buffer `t`
(alphabet size σ); output is the standard suffix array. `int32` indices bound
the corpus to `< 2^31` bytes (a 64-bit path is a later change).

### 3.2 BitVector — `BitVector.h` (rank9)
Plain bit vector with `rank1`/`rank0`. Directory is **rank9** (Vigna, WEA 2008):
per 512-bit block, one 128-bit interleaved entry `[abs, l2]` where `abs` is the
cumulative popcount before the block and `l2` packs the 7×9-bit sub-block (64-bit)
offsets. A `rank1` is: read one directory entry (one cache line) + one partial-word
popcount — no loop. A trailing **sentinel block** makes `rank1(n)` at an exact
block boundary valid without a hot-path branch. Used for the sampled-row bitmap
(and by the binary `WaveletMatrix`, kept only for its own unit test).

### 3.3 QuadVector — `QuadVector.h` (the heart of the query path)
A vector of `n` 2-bit symbols with `rank(v, i)` for each `v ∈ {0,1,2,3}`.

- **Storage:** 32 symbols per 64-bit word.
- **In-block count (SWAR):** to count lanes equal to `v` in the low `k` lanes of
  a word,
  `t = word ^ (v·0x5555…); eq = (~t) & (~(t>>1)) & 0x5555…; popcount(eq & mask_k)`.
  A single in-register vectorized popcount — this is where per-query SIMD actually
  pays (see §5).
- **Directory (two-level):** per 2048-symbol superblock, 4 absolute `uint64`
  counts; per 32-symbol block, 4 `uint16` counts relative to the superblock. A
  `rank` is: superblock abs + block rel + one SWAR popcount over one word. `uint16`
  is safe because a symbol's count within a 2048-symbol superblock is `< 2048`.
- **Serialization:** only the 2-bit words travel; the directory is rebuilt on load.

### 3.4 WaveletMatrix4 — `WaveletMatrix4.h` (4-ary quad wavelet matrix)
Pointerless wavelet matrix consuming **2 bits per level** (Ceregini-Kurpicz-
Venturini, DCC 2024). `qlevels = ⌈⌈log₂σ⌉ / 2⌉`. Per level: one `QuadVector` of
the level's 2-bit digit for each symbol, plus `start[4]` = the cumulative digit
histogram (group-start offsets). Halving the levels vs a binary matrix halves the
dependent cache misses on the backward-search hot path.

- `rank(c, i)` = descend both `0` and `i` following `c`'s digits;
  `mapped_i − mapped_0`.
- `map2(c, lo, hi)` fuses the `lo`/`hi` descent (the backward-search primitive).
- `map_zero(c)` = descent of `0` (a per-symbol constant, precomputed into `first_`
  so backward search needs only 2 descents, not 3).
- `map_batch(...)` = level-major descent across many queries with prefetch (§5).

### 3.5 FMIndex — `FMIndex.h` / `FMIndex.cpp`
Ties it together: dense-alphabet remap, SA → BWT (of dense ids) → quad wavelet
matrix, C-table, `first_`, sampled SA + sampled-row `BitVector`, `doc_start`.

## 4. Query algorithms

**Backward search / Count.** Maintain a suffix-array interval `[lo, hi)`
(initially `[0, m)`). For each pattern byte, back to front, with id `c`:
`base = C[c] − first_[c]; (plo, phi) = map2(c, lo, hi); lo = base+plo; hi = base+phi`.
If `lo ≥ hi`, no match. After the last byte, **`hi − lo` is the occurrence count**
(the interval width). `first_[c]` cancels the wavelet-matrix group-start baseline.

**Locate.** For each row `i ∈ [lo, hi)`, walk LF-mapping
(`LF(i) = C[BWT[i]] + rank(BWT[i], i)`, with `BWT[i] = wm.access(i)`) until a
**sampled** row is reached (the sampled-row `BitVector`'s rank gives the index into
`sa_sample_vals`), then `pos = sampled_value + steps`. Positions ≥ `n` (the
sentinel seat) are dropped. Sampling every `k`-th SA value trades index size for
locate latency.

**LocateDocs.** `upper_bound(doc_start, pos) − 1` gives the document; offset is
`pos − doc_start[doc_id]`.

**CountBatch (throughput).** The workload is latency-bound: each `rank` is a
data-dependent cache miss and the core would otherwise idle. `CountBatch` runs
patterns in **tiles of 32**, advancing all active patterns one character per round
in lock-step; within a round the level-major `map_batch` issues many independent
rank loads that **overlap in the memory system** (memory-level parallelism), with
explicit prefetch. ~2× over calling `Count` in a loop, and it composes with the
quad matrix. This is the primitive for bulk decontamination (score thousands of
n-grams at once).

## 5. Where SIMD lives (and where it doesn't)

Per-query, a backward search is a **serial chain of dependent cache misses** (each
level's position needs the previous level's rank); SIMD cannot shorten a
dependency chain, and rank9 is already one popcount, so per-query SIMD-within-a-
level buys ~1% — deliberately not done. The two real levers:

1. **SWAR quad-rank** (single query): counting a 2-bit symbol over a packed word
   is an in-register vectorized popcount, and the quad matrix halves the levels —
   together they make single-query count 1.7× the binary/reference baseline.
2. **Batched MLP** (throughput): overlap independent queries' misses. ~2× more.

## 6. Correctness invariants

- One sentinel, unique-smallest ⇒ all suffixes distinct ⇒ well-defined BWT/LF.
- Dense-id order preserves byte order ⇒ SA/BWT/C-table consistent.
- A match interval `[lo, hi)` always satisfies `0 ≤ lo ≤ hi ≤ m`.
- `rank(v, i)` valid for `i ∈ [0, n]` (directory padded with a sentinel entry).
- QuadVector never counts trailing lanes past `n` (build counts only valid lanes).
- Verified continuously: every `Count` is checked byte-identical against sdsl-lite
  on 120k queries, plus 200-trial randomized fuzz vs a brute-force oracle, plus
  `CountBatch == Count`, across sample rates and alphabets (empty, single,
  all-same, full 256-byte, binary). See `test/test_fmindex.cpp` (25 tests).

## 7. Serialization format

A flat little-endian blob (format v3): a header — magic `"FMIX"`, version,
`sa_sample_rate`, `σ`, `qlevels`, `text_len`, the 256-entry `byte→id` map, the
C-table, per-level `start[4]`, then the section sizes (per-level quad word counts,
sampled word count, sample count, doc count) — followed by the large payload
arrays, **each padded to an 8-byte boundary**: per quad level the 2-bit words,
then the sampled-row bit words, the `uint32` sampled-SA values, and the `uint64`
`doc_start`. The 8-byte alignment is what lets `LoadView` point the quad/sampled
words at mmap'd memory (zero-copy). Derived structures (wavelet/quad rank
directories, `first_`) are **recomputed on load**, so they cost nothing on disk —
they are in-RAM only. `sa_sample_vals` is `uint32` (corpus `< 2^32`), halving the
sample array vs `uint64` at no query cost.

## 8. Performance summary (4 MB, arm64)

Versus sdsl-lite (`csa_wt`), the reference FM-index, on general text: build 1.6×,
single-query count 1.7×, batched count 5.7×, locate 1.6–3× — all faster — at a
tied index size (1.00× vs 1.005×). Only DNA index size trails (0.75× vs sdsl's
entropy-coded 0.56×), not pursued. Full tables in `BENCHMARK.md`.

## 8b. Operational properties (mutability, concurrency, mmap)

**Immutable / build-once.** The index is not incrementally updatable: the BWT,
wavelet matrix, and sampled SA all derive from the *global* suffix ordering, so
inserting or deleting a document requires rebuilding the SA/BWT. This is inherent
to FM-indexes (sdsl, Lance, Infini-gram are the same). In Milvus this maps onto
the per-segment lifecycle: **inserts** go to new segments (new indexes);
**deletes** are applied at query time via the segment delete-bitmap masking the
returned rows/positions (the index is untouched); **reclamation** happens through
segment compaction rebuilding the index — all existing Milvus machinery.

**Concurrency: lock-free reads.** Every query method (`Count`, `BackwardSearch`,
`Locate`, `LocateDocs`, `CountBatch`) is `const` and pure-read; `CountBatch`'s
scratch buffers are function-local, so it is re-entrant. There is no mutable
cache, lazy init, or shared state, so **many threads may query concurrently
without locks**, provided no thread is building/deserializing the same instance
(guaranteed by the build-then-serve lifecycle).

**mmap (zero-copy load).** The serialization format (v3) lays each large payload
array — the quad wavelet words and the sampled-row bit words — at an 8-byte
boundary, so they can be viewed in place from a memory mapping. `BitVector` and
`QuadVector` store their words as a pointer + length that is either **owned**
(built in RAM) or a **view** into external memory; `WaveletMatrix4` and the
sampled bitvector view the mapped bytes directly. Only the rank directories (and
`first_`, and the small sample/doc arrays) are rebuilt/copied in RAM on load, so
serving RAM is a fraction of the index. Two entry points:
- `FMIndex::LoadView(base, size)` — zero-copy over caller-owned memory (e.g. an
  mmap). The caller keeps the mapping alive for the index's lifetime.
- `FMIndex::Deserialize(blob)` — copies the blob into an internal buffer and
  views that (owns its bytes).

`MmapFmIndex.h` provides a POSIX `SaveToFile` + `MappedFMIndex::Open` convenience
(mmap a file, view it). The Milvus port would instead call `Serialize` /
`LoadView` against Milvus's own storage + mmap layer. Move semantics are handled:
the word pointers stay valid across moves (a `std::vector` move preserves its
buffer address; owning structures re-point to it, views keep the external
pointer). Verified by `LoadViewZeroCopyMatchesInRam` and `MmapFileRoundTrip`.

**Copy budget.** Load: `LoadView` copies **nothing** of the large word arrays
(views mmap); `Deserialize` copies the blob **once** into an owned buffer then
views it (not one copy per array). Save: `SerializeToFile` streams the arrays
straight to the file (only a tiny header is buffered), so it avoids the
intermediate full-index blob that `Serialize()`→write would build. Only the small
sampled-SA / doc-start arrays and the rank directories are (re)built in RAM on
load, regardless of path.

**Measured (4 MB text index, arm64).** Load latency: `Deserialize` ~0.8 ms,
mmap `Open`+`LoadView` ~2 ms (both dominated by directory rebuild, not data copy).
Query throughput: **warm mmap equals in-RAM** (within noise, often slightly
faster — the serialized layout is one contiguous block, better for cache/TLB than
the fragmented freshly-built structures); the only mmap cost is a one-time
page-fault sweep on the cold first pass.

## 9. Porting into Milvus

`src/index/fmindex/*` moves to `internal/core/src/index/` unchanged (same
`milvus::index::fmindex` namespace, `#include "index/fmindex/..."`). `libsais`
(Apache-2.0) is vendored. The Milvus scalar-index wrapper registers
`ScalarIndexType::FMINDEX`, serves `InnerMatch` (`LIKE '%x%'`) via `Count`/`Locate`,
and exposes `CountBatch` for bulk decontamination. Per-segment build over the
column's concatenated row values + `doc_start`; queries fan out and aggregate.

## 10. Limitations / future

- **Corpus `< 2^31` bytes** per index (int32 SA + uint32 samples). Per-segment in
  Milvus this is fine; a 64-bit path (`libsais64`) is a later change.
- **DNA size** is behind sdsl; closing it (sentinel removal → 2 quad levels, or a
  bwa-mem2-style 2-bit occ) costs effort for a non-target workload.
- **Repetitive corpora**: an r-index (RLBWT) could be ~10× smaller on heavily
  deduplicated training data — worth measuring BWT-run density on real data first.

## Provenance (all permissive)

- **libsais** — Ilya Grebnov, Apache-2.0 (vendored).
- **rank9** — Vigna, WEA 2008 (clean-room; sux is GPL, not used).
- **wavelet matrix** — Claude & Navarro, SPIRE 2012 (clean-room).
- **quad wavelet matrix** — Ceregini, Kurpicz, Venturini, DCC 2024 (clean-room;
  `rossanoventurini/qwt` MIT for reference).
- **FM-index / backward search** — Ferragina & Manzini, JACM 2005.
- **batched/MLP backward search** — bwa-mem2 (Vasimuddin et al., IPDPS 2019, MIT).
- **FM-index shape** — following Lance `lance_index::scalar::fmindex` (Apache-2.0)
  and Infini-gram Mini (arXiv:2506.12229).
- sdsl-lite is used only as a benchmark/correctness reference (GPL — never copied).
