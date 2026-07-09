// Licensed to the LF AI & Data foundation under Apache-2.0.
// Portions translated from Lance (lance_index::scalar::fmindex), Apache-2.0.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "index/fmindex/BitVector.h"
#include "index/fmindex/WaveletMatrix4.h"

namespace milvus::index::fmindex {

// FM-index over a single concatenated byte buffer. Bytes are remapped to a
// dense alphabet [1, sigma) (0 = sentinel), so the wavelet matrix uses only
// ceil(log2(sigma)) levels — e.g. 3 for DNA, not a fixed 9. Exact substring
// count / locate / (pk, offset) via BWT backward search; no false positives.
class FMIndex {
 public:
    FMIndex() = default;

    // Build the index over one contiguous byte buffer (the concatenation of all
    // your documents).
    //
    //   data            pointer to the first byte of the buffer.
    //   len             its length in BYTES (not documents, not characters;
    //                   a UTF-8 CJK char counts as its 3 bytes). Must be
    //                   < 2^31; for larger corpora, shard and build one index
    //                   per shard. Count/Locate are exact over [data, data+len).
    //   sa_sample_rate  suffix-array sampling rate R (>= 1): store one SA value
    //                   every R text positions, recovering the rest via LF-
    //                   mapping. Pure space/locate trade-off, zero effect on
    //                   Count or correctness: larger R = smaller index but
    //                   slower Locate/LocateDocs (up to R-1 extra LF steps per
    //                   hit); smaller R = faster Locate, larger sample array
    //                   (~len/R * 4 bytes). Default 32 is balanced; use 4-8 if
    //                   you Locate often, 64+ if you only ever Count.
    //   case_insensitive  when true, ASCII letters A-Z are folded to a-z at
    //                   build time (both cases share one symbol), so all queries
    //                   match case-insensitively with zero query-time cost. Only
    //                   ASCII case is folded; non-ASCII / UTF-8 bytes are left
    //                   exact. Locate offsets still refer to the original text.
    void
    Build(const uint8_t* data, size_t len, uint32_t sa_sample_rate = 32,
          bool case_insensitive = false);

    // Half-open SA interval [lo, hi) for pattern; lo==hi means no match.
    std::pair<size_t, size_t>
    BackwardSearch(const uint8_t* pattern, size_t plen) const;

    size_t
    Count(const uint8_t* pattern, size_t plen) const {
        auto r = BackwardSearch(pattern, plen);
        return r.second - r.first;
    }

    // Occurrence counts for many patterns at once. Runs all patterns' backward
    // searches in lock-step (one character each per round) so their rank cache
    // misses overlap (memory-level parallelism) — far higher throughput than
    // calling Count() in a loop. Result[i] corresponds to patterns[i].
    std::vector<size_t>
    CountBatch(const std::vector<std::pair<const uint8_t*, size_t>>& patterns)
        const;

    std::vector<uint64_t>
    Locate(const uint8_t* pattern, size_t plen) const;

    // Document boundaries (offsets into the concatenated buffer), ascending and
    // beginning with 0. If empty, the whole corpus is treated as one document.
    void
    SetDocStarts(std::vector<uint64_t> doc_start) {
        doc_start_ = std::move(doc_start);
        if (doc_start_.empty() || doc_start_[0] != 0) {
            doc_start_.insert(doc_start_.begin(), 0);
        }
    }

    std::vector<std::pair<uint64_t, uint64_t>>
    LocateDocs(const uint8_t* pattern, size_t plen) const;

    // Documents that BEGIN with the pattern (anchored prefix match), as sorted,
    // unique document ids. A hit is an occurrence whose text position is exactly
    // a document boundary. Requires SetDocStarts to be meaningful; with the
    // default single document it reports {0} iff the whole corpus starts with P.
    std::vector<uint64_t>
    LocatePrefixDocs(const uint8_t* pattern, size_t plen) const;
    // Number of documents that begin with the pattern.
    size_t
    CountPrefixDocs(const uint8_t* pattern, size_t plen) const {
        return LocatePrefixDocs(pattern, plen).size();
    }

    // Documents that END with the pattern (anchored suffix match), as sorted,
    // unique document ids. A hit is an occurrence whose end (pos + plen) lands
    // exactly on the document's end boundary.
    std::vector<uint64_t>
    LocateSuffixDocs(const uint8_t* pattern, size_t plen) const;
    // Number of documents that end with the pattern.
    size_t
    CountSuffixDocs(const uint8_t* pattern, size_t plen) const {
        return LocateSuffixDocs(pattern, plen).size();
    }

    // Recover the original bytes T[pos, pos+len) from the index — e.g. to show
    // the context around a match found via Locate. O(len + sa_sample_rate) LF
    // steps; no forward navigation needed. If the index was built with
    // case_insensitive=true, ASCII letters come back lowercased (original case
    // is not stored). Returns fewer than len bytes if pos+len exceeds the text.
    std::string
    Extract(uint64_t pos, size_t len) const;

    // --- accessors used by tests ---
    size_t
    bwt_size() const {
        return text_len_ + 1;
    }
    uint32_t
    alphabet() const {
        return sigma_;
    }
    uint32_t
    qlevels() const {
        return qlevels_;
    }
    uint32_t
    c_at(uint32_t c) const {
        return c_[c];
    }
    bool
    case_insensitive() const {
        return case_fold_;
    }

    // --- serialization ---
    // Serialize to a flat, 8-byte-aligned blob (payload arrays aligned so they
    // can be viewed zero-copy from an mmap).
    std::string
    Serialize() const;
    // Stream the serialized bytes straight to a file, without materializing a
    // full-index blob first (only a small header is buffered) — one fewer copy
    // than Serialize()+write on the save path.
    bool
    SerializeToFile(const std::string& path) const;

    // Copy the blob and load (owns its bytes).
    static FMIndex
    Deserialize(const std::string& blob);
    // Zero-copy load: view serialized bytes at [base, base+size) (must be
    // 8-byte aligned, e.g. mmap). The caller keeps that memory alive for the
    // index's lifetime; the large word arrays are not copied.
    static FMIndex
    LoadView(const uint8_t* base, size_t size);

 private:
    size_t
    LF(size_t i) const;
    // Rebuild the in-RAM-only structures derived from the serialized fields:
    // id_to_byte_ (from byte_to_id_) and isa_sample_ (from sampled_bv_ +
    // sa_sample_vals_). Called after Build and after a load.
    void
    buildDerived();
    // Fill *this by viewing serialized bytes at base; the wavelet/sampled word
    // arrays are viewed, the small sample/doc arrays are copied. Returns false
    // on a truncated/corrupt/incompatible blob.
    bool
    parseView(const uint8_t* base, size_t size);
    // Append the fixed header (everything before the aligned payload arrays).
    void
    writeHeader(std::string& s) const;

    uint32_t sa_sample_rate_ = 1;
    bool case_fold_ = false;             // ASCII A-Z folded to a-z at build time
    uint64_t text_len_ = 0;              // original byte count (no sentinel)
    uint32_t sigma_ = 1;                 // dense alphabet size incl sentinel
    uint32_t qlevels_ = 0;              // ceil(ceil(log2(sigma_)) / 2)
    std::array<int32_t, 256> byte_to_id_{};  // byte -> dense id, -1 if absent
    WaveletMatrix4 wm_;                 // 4-ary quad matrix over the BWT
    std::vector<uint32_t> c_;           // C-table, size sigma_
    std::vector<size_t> first_;         // per-symbol map_zero (derived, not serialized)
    BitVector sampled_bv_;              // row is SA-sampled? rank = sample index
    // SA values of sampled rows, in row order. uint32 (text < 2^32) halves the
    // sample storage vs uint64 with no query cost — plain array load either way.
    std::vector<uint32_t> sa_sample_vals_;
    std::vector<uint64_t> doc_start_;  // document boundaries
    // Derived, in-RAM only (rebuilt on load, never serialized):
    std::vector<uint8_t> id_to_byte_;    // dense id -> byte (inverse of byte_to_id_)
    std::vector<uint32_t> isa_sample_;   // isa_sample_[k] = row whose SA value = k*rate
    std::vector<uint8_t> owned_blob_;  // backs the views when Deserialized by copy
};

}  // namespace milvus::index::fmindex
