# fm-index-lite v2 — Symbol-Domain Redesign

- **Date:** 2026-07-14
- **Status:** Design (approved scope: see Requirements; implementation not started)
- **Supersedes:** the byte-`\0`-separator scheme of v1 (`DESIGN.md` §1–§2) and the
  two v1 future-list items it absorbs (`\0` relaxation, token-level mode).

## Requirements (from the Milvus FMINDEX integration)

1. **`\0` must be a first-class content byte** — never rejected at `Build`, and
   **queryable in patterns**. Consequence: the separator cannot be the byte
   `0x00`; it must live outside the byte alphabet.
2. **Token-level matching** — index sequences of integer token ids (tokenizer
   vocabularies, "complex encodings") with the same primitive set: count,
   locate, per-document attribution, anchored prefix/suffix, batch.
3. **Close the remaining API gaps** the Milvus integration needs (streaming
   serialization sink, interval reuse, bitmap-friendly output, equality,
   empty-document semantics) — inventoried in §6.

Requirements 1 and 2 converge on one refactor: once the separator is its own
symbol outside the content alphabet, a "byte document" is just a symbol
sequence with σ ≤ 258 — the same machinery as token sequences. v2 makes the
core **symbol-domain generic** and derives both modes from it.

## 1. What stays (the v1 assets)

The query-path components are already symbol-generic and carry over unchanged:

- `QuadVector` (2-bit packed, SWAR rank, two-level directory) — operates on
  2-bit digits, no alphabet assumption.
- `WaveletMatrix4` — `qlevels = ⌈⌈log₂σ⌉/2⌉` derived from σ; `rank`/`map2`/
  `map_zero`/`map_batch` already take `uint32_t` symbols.
- `BitVector` (rank9), C-table, `first_`, sampled SA + sampled-row bitmap,
  `doc_start` mapping, backward search / LF / locate algorithms, `CountBatch`
  tiling, serialization framework (8-byte-aligned sections, `LoadView`
  zero-copy, directory rebuild on load), the whole test/oracle methodology.

What changes is the **encoding front-end** (alphabet, separator, SA
construction, staging widths) and the **API surface** (token entry points,
alignment additions).

## 2. Encoding: dense symbol alphabet with an out-of-alphabet separator

```
dense id:   0          1           2 .. σ-1
symbol:     sentinel   separator   content symbols (ascending original order)
```

- **Content symbols**: byte mode — the distinct bytes of the corpus, including
  `0x00`, mapped in ascending byte order; token mode — the distinct token ids
  present in the corpus, mapped in ascending token order. Order-preserving
  remap keeps SA/BWT ordering semantics identical to v1.
- **Separator** is dense id 1: smaller than every content symbol, larger than
  the sentinel — it can never collide with content, so *any* byte / token
  value is storable and queryable. The v1 `Build` throw on `\0` is deleted.
- **Sentinel** stays dense id 0, appended once.
- Backward search no longer needs a pattern-side separator rejection: no
  byte/token maps to id 1, so patterns simply cannot address the separator.
  `FuzzyMatchingDocs` (substitution enumeration), `LongestMatch`, and
  `NextTokenCounts` keep their explicit "skip id 1" guards — unchanged logic,
  now watertight rather than convention-dependent.
- σ: byte mode ≤ 258 (256 bytes + separator + sentinel). Typical text has
  ~100 distinct bytes → σ ≈ 102 → 4 quad levels, **same as v1**; the full-byte
  worst case is 258 → 9 bits → 5 levels, also same as v1's 257. Query-path
  depth is unchanged in every case except σ crossing a power-of-two boundary
  (rare and marginal).

## 3. Suffix-array construction (the real cost of the redesign)

v1's zero-copy trick — `libsais` directly over the raw byte buffer, `\0` as
separator, end-of-string as sentinel — is what reserved `\0`. With 257+
distinct symbol values the SA input cannot be `uint8_t`; v2 materializes an
**integer symbol text** and selects the SA routine by corpus size. **Both
integer-alphabet routines are already vendored** — no new libsais file to add:

| mode | σ | m (symbols incl. sentinel) | routine | staging | vendoring |
|---|---|---|---|---|---|
| byte / token, compact | ≤ 2³¹ | < 2³¹ | `libsais_int` | int32 (4 B/sym) | **already vendored** |
| byte / token, large | ≤ 2⁶³ | ≥ 2³¹, < 2⁶³ | `libsais64_long` | int64 (8 B/sym) | **already vendored** |

`libsais_int` / `libsais64_long` take an integer text over alphabet `[0, k)`
with `k` up to 2³¹ / 2⁶³ — one path serves **both** byte mode (k = σ ≤ 258) and
token mode (k = distinct token count), so token support needs no new SA code,
only a wider symbol table. Both **restore the input array on success**
(documented), so the same `t` is reused to compute the BWT — no second copy.

Implemented as `build_suffix_array_symbols32` / `_64` in `SuffixArray.h`. The
symbol text `t` carries the trailing sentinel (id 0, unique-smallest) so the
routine returns the length-m sentinel SA directly (no shift-trick). Case folding
(byte mode) is baked into the byte→id map — uppercase aliases lowercase's id —
so `t` is folded implicitly, with no separate folded buffer.

Notes:

- **Chosen over `libsais16`.** An upstream `libsais16` (u16 text, 2 B/sym) would
  cut staging memory but must be vendored fresh and would still fall back to the
  int paths for token σ > 65 535. `libsais_int` is already in-tree, one path for
  both modes, and correctness-first; the u16 staging optimization is a
  deferrable follow-up (halves the text buffer only), not a prerequisite.
- **Build memory.** The symbol text is built directly from the caller's
  documents (no intermediate byte concat buffer), int32 for the compact path.
  Peak during construction ≈ text(4m) + SA(4m) then + BWT(2m) ≈ **~10× corpus**
  for the 32-bit path (v1 was ~8×; the delta is the int32 text vs v1's byte
  buffer). The 64-bit path (≥ 2 GiB, shard territory) doubles the text/SA to
  ~16m. Re-benchmark and update `BENCHMARK.md`. The u16 staging follow-up would
  bring the 32-bit peak back toward ~8×.
- **Query path and index size are unchanged** — same qlevels (σ 257→258 does not
  cross a power-of-two boundary), same BWT/wavelet/sampled structures.

## 4. Structure changes downstream of the alphabet

- **BWT staging**: u16 today, stays u16 for σ ≤ 65 536; the huge-vocab token
  path needs a u32 staging variant — `WaveletMatrix4` gains a constructor
  overload taking `std::vector<uint32_t>` (the level-partition loop is width-
  agnostic; the QuadVectors it emits are identical).
- **C-table / `first_`**: sized σ — unchanged logic, token mode just makes σ
  larger (128 K entries = 1 MB of u64, negligible).
- **`byte_to_id_[256]`** generalizes to a **symbol map**: byte mode keeps the
  flat 256-entry array (hot path untouched — one load per pattern byte); token
  mode uses a sorted `id_to_token` array + binary search (or hash) for
  pattern-side token→id, and the inverse array for `Extract`.
- **Serialization format v7**: header gains a mode flag (byte/token) and, for
  token mode, the `id_to_token` table (u32 × σ) replacing the 256-entry byte
  map. Everything else (per-level quad words, sampled bitmap, sampled SA,
  doc_start, 8-byte alignment, `LoadView` zero-copy, load-time validation and
  byte-flip fuzz) carries over. v6 blobs: **not migrated** — the index is a
  derived structure; rebuild (Milvus rebuilds per-segment indexes on version
  bump as a matter of course).

## 5. API

Byte-mode signatures are unchanged (`Build(vector<string_view>, ...)`,
patterns as `const uint8_t*` + length) — only semantics widen: `\0` in content
indexes normally, `\0` in patterns matches normally, the
`std::invalid_argument` throw disappears.

Token mode adds parallel entry points (C++17, no `std::span`):

```cpp
// documents are sequences of token ids; document i = attribution unit
void BuildTokens(const std::vector<std::pair<const uint32_t*, size_t>>& docs,
                 uint32_t sa_sample_rate = 32);

size_t CountTokens(const uint32_t* pattern, size_t plen) const;
std::vector<size_t> CountTokensBatch(
    const std::vector<std::pair<const uint32_t*, size_t>>& patterns) const;
std::vector<uint64_t> MatchingDocsTokens(const uint32_t* p, size_t n) const;
std::vector<std::pair<uint64_t, uint64_t>> LocateDocsTokens(...) const;   // (doc, token_offset)
std::vector<uint64_t> LocatePrefixDocsTokens(...) const;                  // doc begins with sequence
std::vector<uint64_t> LocateSuffixDocsTokens(...) const;                  // doc ends with sequence
std::vector<uint32_t> ExtractTokens(uint64_t doc, uint64_t off, size_t n) const;
std::vector<std::pair<uint32_t, size_t>> NextTokenCounts(...) const;      // now literally next-*token* (∞-gram LM)
```

Internally both modes run the same engine; whether to express this as
`template <class Domain> class BasicFMIndex` with two aliases or one class
with a mode discriminant is an implementation choice — the deciding constraint
is that the byte hot path must keep the flat-array symbol lookup (no branch,
no virtual dispatch). Offsets in token mode are **token offsets**, not byte
offsets (callers own the mapping if they need bytes).

## 6. Milvus-alignment additions (requirement 3 inventory)

Audit of the Milvus FMINDEX design (20260708 MEP) against the current API:

| gap | addition | consumer |
|---|---|---|
| serialization sink is POSIX-file or full-blob only | `SerializeToSink(const std::function<void(const void*, size_t)>&)` — streaming, same layout as `SerializeToFile` | Milvus V3 `WriteEntries(IndexEntryWriter*)` |
| count-then-enumerate re-runs backward search | expose interval reuse: `MatchingDocsFromInterval(lo, hi, plen)`, `LocateDocsFromInterval(...)` (interval from public `BackwardSearch`) | `CanAccelerate` count-first guard → enumeration without a second `O(\|P\|)` search |
| doc-id results materialize `std::vector` | visitor overloads: `MatchingDocs(P, plen, const std::function<void(uint64_t)>&)` (sorted, unique callbacks) | filling `TargetBitmap` directly, no intermediate vector |
| equality needs caller-side length math | `EqualsDocs(P, plen)` — prefix-anchored hits filtered by `doc_start[d+1] − doc_start[d] − 1 == plen` internally | `Equal` / `In` / `NotIn` |
| empty documents (Milvus NULL rows) untested | explicit tests: empty docs at head/middle/tail, all-empty corpus, doc ids stay aligned with row offsets | NULL-row indexing |
| batch doc-membership for factor AND | `MatchingDocsBatch(patterns)` — thin composition over `LocateDocsBatch` + dedup, keeps the MLP win | `Match`/`RegexMatch` phase-1 factor evaluation |

Explicitly **not** in v2 (unchanged decisions): Damerau transpositions and all
fuzzy work (deferred with the Milvus fuzzy phase), char-level edit semantics,
r-index / doc-listing, `\0`-pattern *decline* logic in Milvus (now deletable —
patterns with `\0` just work).

## 7. Expected numbers (to verify, updating BENCHMARK.md)

| axis | v1 | v2 expected |
|---|---|---|
| byte build peak RSS | ~8× corpus (9.1× measured incl. caller docs) | **~10×** (int32 symbol text; u16 staging follow-up → ~9×) |
| byte build time (1 GiB) | 44.6 s | to measure (`libsais_int` vs byte `libsais`) |
| byte query (count/locate/match) | — | **unchanged** (same qlevels, same structures) |
| byte index size | 1.01× corpus | unchanged (BWT symbols, samples identical) |
| token index size (English, ~4 B/token) | n/a | ~0.5–0.6× of the *byte* corpus (n/4 symbols × ~18 bits + samples) |
| token query, 13-token pattern | n/a | ~2× faster than the 50-byte equivalent (117 vs ~260 level-steps) |

## 8. Test plan

- **Byte-mode differential oracles re-run** (the existing 200-trial fuzz +
  120 K sdsl cross-check corpus) — results must be identical for `\0`-free
  data; new oracle classes: content containing `\0` (head/middle/tail/
  adjacent-to-separator), patterns containing `\0`, `\0`-only documents.
- **Anchored/equality invariants with `\0`**: doc `"a\0"` — `EqualsDocs("a")`
  false, `EqualsDocs("a\0")` true; prefix/suffix boundary cases.
- **Token-mode oracles**: brute-force subsequence search over random token
  corpora (vocab sizes straddling 65 535 to exercise both SA paths), plus the
  same doc-attribution / anchored / batch invariants as byte mode.
- **Format v7**: round-trip, `LoadView` parity, byte-flip load fuzz (ASan),
  v6-blob rejection with a clear error.
- **Alignment APIs**: sink-serialization byte-identical to `SerializeToFile`;
  interval-reuse ≡ pattern re-search; visitor ≡ vector results;
  `MatchingDocsBatch[i]` ≡ `MatchingDocs(patterns[i])`.
- **Benchmarks**: full `bench_fmindex` + `bench_gig` sweep; assert the "byte
  query unchanged" expectation, record the build deltas.

## 9. Implementation order (each lands green)

1. **✅ Symbol-domain core (DONE).** `SuffixArray.h` gains
   `build_suffix_array_symbols32/64` over the already-vendored `libsais_int` /
   `libsais64_long`; `FMIndex::Build` builds the dense-id symbol text directly
   (id 0 sentinel, id 1 separator, ids 2.. content), SA over it, BWT from it;
   `\0` throw deleted; byte-driven query paths drop the now-dead separator check;
   format v7 + validation. All 60 prior oracles stay byte-identical on `\0`-free
   data, plus new `\0`-content / `\0`-pattern / cross-document / fuzz oracles,
   ASan+UBSan clean including the corrupted-blob load fuzz. *(Requirement 1.)*
2. **Token mode**: `BuildTokens` + token query APIs, u32 BWT/wavelet staging
   overload (σ > 65 535), token oracles. Reuses the same `libsais_int` path.
   *(Requirement 2.)*
3. **Alignment APIs** (§6) + tests. *(Requirement 3.)*
4. **Benchmark sweep + BENCHMARK.md / DESIGN.md / README.md updates.**
5. **(Optional) u16 staging** via a vendored `libsais16` — memory-only, halves
   the symbol-text buffer; deferrable.

Steps 2–3 are independently shippable; **Milvus vendoring can start now** (step 1
done — the library fully supports `\0`), with token mode and alignment APIs
landing additively.
