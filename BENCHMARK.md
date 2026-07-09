# FM-Index Benchmark: ours vs sdsl vs Lance

Apple Silicon (arm64), Apple clang 21 `-O2`, Rust 1.92 `--release`. Synthetic DNA
(`{A,C,G,T}`) and word-like English text. Numbers below are the 4 MB corpus.

## What each column measures (read this first)

| impl | what it is | query measured |
|---|---|---|
| **ours** | this library (`src/index/fmindex`) — pure in-memory structure | **occurrence count** and **locate** (raw substring search) |
| **sdsl** | `sdsl::csa_wt<>` (Huffman wavelet tree), the reference FM-index | occurrence count / locate |
| **Lance** | `lancedb` 0.31 `Index::Fm` — FM index **through a Dataset + `contains()` scan** | **matching rows** via the full async query engine |

ours vs sdsl is apples-to-apples (both pure FM structures, same corpus/queries;
their 120,000 occurrence counts are **byte-identical** — correctness proven).
**Lance is not directly comparable on query**: its FM internals are crate-private,
so the only public path is a Dataset scan with a `contains()` filter that
materializes matching *rows* through query planning + async IO. That measures a
different operation (row retrieval, not counting) with large per-query overhead.

## Build time (ms, lower is better)

| corpus | ours | Lance | sdsl |
|---|---|---|---|
| DNA 4 MB | **108** | 133 | 219 |
| text 4 MB | **145** | 177 | 233 |

**Ours is fastest** (libsais SA-IS vs sdsl's divsufsort; Lance in between).

## Query throughput (queries/sec)

Structure: **4-ary quad wavelet matrix** (QWT) + compact SWAR quad-rank.

| corpus | ours count | ours count (batch, tile=32) | sdsl count | ours locate | sdsl locate | Lance contains |
|---|---|---|---|---|---|---|
| DNA 4 MB | **2.6 M** | **7.1 M** | 2.4 M | 0.83–1.62 M | 0.64 M | 32 † |
| text 4 MB | **1.5 M** | **5.1 M** | 0.89 M | 0.53–1.05 M | 0.33 M | 2,674 † |

- **count (single query): ours beats sdsl on BOTH** — text 1.5M vs 0.89M (1.7×),
  DNA 2.6M vs 2.4M (1.1×). The 4-ary quad wavelet matrix halves the levels (text:
  5 binary → 3 quad) → half the dependent cache misses, and its quad-rank is a
  single SWAR "count 2-bit lanes == v" popcount. **This is where per-query SIMD
  actually pays** — in the binary matrix a query is a serial miss-chain SIMD
  can't touch; the quad structure is what makes single-query vectorization real.
- **count (batched, `CountBatch`): ours beats sdsl by 3–6×** — text 5.1M vs
  0.89M, DNA 7.1M vs 2.4M. Batching runs many patterns' backward searches in
  lock-step so their rank cache misses overlap (memory-level parallelism);
  ~2× over single-query, stacks with QWT. Matches the real bulk-decontamination
  workload (score thousands of n-grams at once).
- **locate: ours ~1.6–3× sdsl** (quad + fewer levels helps LF too).
- † **Lance contains/sec is not comparable** — it retrieves and materializes all
  matching rows through the async query engine, not a raw count. DNA is
  pathological (a 6-char DNA substring matches almost every row → ~all rows
  materialized → 32 q/s). This reflects query-engine overhead, not the FM
  structure. For the decontamination workload (count + positions of an n-gram),
  the pure-structure API (ours/sdsl) is the right tool and is orders of magnitude
  faster.

## Index size (ratio to corpus, lower is better)

| corpus | ours | sdsl | Lance |
|---|---|---|---|
| DNA 4 MB | 0.75× | **0.56×** | 1.26× ‡ |
| text 4 MB | **1.00×** | 1.005× | 1.99× ‡ |

(QWT costs ~1 extra bit/symbol vs the binary matrix, but storing the sampled-SA
values as uint32 instead of uint64 — valid for corpora < 2^32 bytes, zero query
cost — halves the sample array and nets text back to **1.00×, matching sdsl**.
DNA is 0.75× vs sdsl's entropy-coded 0.56×; further DNA shrink would cost query
speed and isn't pursued — the target is general text.)

- ours ties sdsl on text; **sdsl wins DNA** (Huffman/entropy coding → ~n·H₀; ours
  uses a balanced 3-level wavelet matrix = 3 bits/symbol, plus 8-byte sample
  values).
- ‡ **Lance's ratio includes the stored dataset**, not just the index (its index
  files can't be cleanly isolated from the temp dir), so it overstates index size
  — treat as an upper bound, not a like-for-like index measurement.
- Note: our rank9 directory (~25% overhead) lives **in RAM only** — it is rebuilt
  on load and not serialized, so it does not affect the on-disk ratios above.

## Scorecard vs sdsl (the apples-to-apples reference)

| axis | verdict |
|---|---|
| build | ✅ **ours ~1.6× faster** |
| locate | ✅ **ours ~1.6–3× faster** |
| count, single query | ✅ **ours faster on both** (text 1.7×, DNA 1.1×) |
| count, batched (`CountBatch`) | ✅ **ours 3–6× faster** (text 5.7×, DNA 3×) |
| size (text / general) | ➖ **tie** (1.00× vs 1.005×) |
| size (DNA) | ❌ sdsl smaller (0.56× vs 0.75×) |

**On general text — the target — we match or beat sdsl on every axis:** build
1.6×, single-query count 1.7×, batched count 5.7×, locate 1.6–3×, size a tie.
The only remaining sdsl lead is **DNA index size** (entropy-coded Huffman tree),
which we do not pursue — it costs query speed and genomic data isn't the target.
For bulk decontamination (batched substring count + locate over general text) we
are **3–6× faster than the reference at equal size**.

## Optimization history (ours)

| step | change | effect (DNA 4 MB) |
|---|---|---|
| baseline | prefix-doubling SA, pointer wavelet tree, size_t/word rank | build 660 ms, size 5.25×, count 0.49 M |
| +libsais | SA-IS construction (Apache-2.0) | build → 108 ms (**beats sdsl**) |
| +dense alphabet | levels = ⌈log₂σ⌉ (3 for DNA, not 9) | size 5.25× → 0.75×, count → 1.5 M |
| +wavelet matrix | pointerless, 1 rank/level, cache-friendly | count → 1.6 M, locate **beats sdsl** |
| +rank9 | interleaved rank directory (RAM-only) | count → 1.9 M |
| +first_ precompute | 2 ranks/level in backward search, not 3 | marginal |
| +CountBatch (MLP) | lock-step batched backward search + prefetch, tile 32 | **batch count 2× single** |
| +4-ary QWT | quad wavelet matrix, half the levels, SWAR quad-rank | text single 1.0→1.5 M, batch → 5.1 M |

The full unit-test suite passes; counts were byte-identical to sdsl across 120k
queries throughout development.

## Reproduce

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
./build/bench_fmindex     # ours: build / count / batch / locate / size
./build/bench_mmap        # in-RAM vs mmap query throughput
```

The sdsl-lite (`csa_wt`) and Lance (`lancedb` 0.31, `Index::Fm`) numbers above were
measured with small external harnesses that depend on those projects; they are not
shipped here to keep this repository self-contained (and GPL-free — sdsl-lite is
GPLv3 and used only as an external reference). The ours-vs-sdsl correctness
cross-check compared 120,000 per-query counts and found them byte-identical.
