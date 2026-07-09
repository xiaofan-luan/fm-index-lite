// Licensed to the LF AI & Data foundation under Apache-2.0.
// Wavelet matrix (Claude & Navarro, SPIRE 2012) — pointerless, one contiguous
// bitvector per level + a per-level zero-count. Faster and more cache-friendly
// than a pointer-based balanced wavelet tree: one rank per level, hot top
// levels stay resident. Clean-room from the paper.
#pragma once
#include <cstdint>
#include <utility>
#include <vector>
#include "index/fmindex/BitVector.h"

namespace milvus::index::fmindex {

class WaveletMatrix {
 public:
    WaveletMatrix() = default;

    // seq holds symbols in [0, 2^levels). levels = ceil(log2(alphabet_size)).
    WaveletMatrix(const std::vector<uint32_t>& seq, uint32_t levels)
        : n_(seq.size()), levels_(levels) {
        bv_.reserve(levels_);
        z_.assign(levels_, 0);
        std::vector<uint32_t> cur = seq;
        std::vector<uint32_t> zeros, ones;
        for (uint32_t l = 0; l < levels_; ++l) {
            uint32_t bitpos = levels_ - 1 - l;  // MSB first
            BitVector bv(n_);
            zeros.clear();
            ones.clear();
            for (size_t i = 0; i < n_; ++i) {
                if ((cur[i] >> bitpos) & 1u) {
                    bv.set(i);
                    ones.push_back(cur[i]);
                } else {
                    zeros.push_back(cur[i]);
                }
            }
            bv.build_rank();
            z_[l] = zeros.size();
            bv_.push_back(std::move(bv));
            cur.swap(zeros);
            cur.insert(cur.end(), ones.begin(), ones.end());
        }
    }

    uint32_t
    access(size_t i) const {
        uint32_t sym = 0;
        size_t pos = i;
        for (uint32_t l = 0; l < levels_; ++l) {
            if (bv_[l].get(pos)) {
                sym = (sym << 1) | 1u;
                pos = z_[l] + bv_[l].rank1(pos);
            } else {
                sym <<= 1;
                pos = bv_[l].rank0(pos);
            }
        }
        return sym;
    }

    // count of symbol c in [0, i)
    size_t
    rank(uint32_t c, size_t i) const {
        size_t first = 0, last = i;
        for (uint32_t l = 0; l < levels_; ++l) {
            if ((c >> (levels_ - 1 - l)) & 1u) {
                first = z_[l] + bv_[l].rank1(first);
                last = z_[l] + bv_[l].rank1(last);
            } else {
                first = bv_[l].rank0(first);
                last = bv_[l].rank0(last);
            }
        }
        return last - first;
    }

    // (rank(c, lo), rank(c, hi)) in a single descent — the backward-search hot
    // path. Tracks the group start plus both endpoints.
    std::pair<size_t, size_t>
    rank2(uint32_t c, size_t lo, size_t hi) const {
        size_t first = 0, plo = lo, phi = hi;
        for (uint32_t l = 0; l < levels_; ++l) {
            if ((c >> (levels_ - 1 - l)) & 1u) {
                first = z_[l] + bv_[l].rank1(first);
                plo = z_[l] + bv_[l].rank1(plo);
                phi = z_[l] + bv_[l].rank1(phi);
            } else {
                first = bv_[l].rank0(first);
                plo = bv_[l].rank0(plo);
                phi = bv_[l].rank0(phi);
            }
        }
        return {plo - first, phi - first};
    }

    // Map lo and hi down following c's bit path (no group-start baseline).
    // With a precomputed map_zero(c), rank(c,x) = mapped_x - map_zero(c), so
    // backward search needs only 2 ranks/level instead of rank2's 3.
    std::pair<size_t, size_t>
    map2(uint32_t c, size_t lo, size_t hi) const {
        for (uint32_t l = 0; l < levels_; ++l) {
            if ((c >> (levels_ - 1 - l)) & 1u) {
                lo = z_[l] + bv_[l].rank1(lo);
                hi = z_[l] + bv_[l].rank1(hi);
            } else {
                lo = bv_[l].rank0(lo);
                hi = bv_[l].rank0(hi);
            }
        }
        return {lo, hi};
    }

    // Batched level-major descent: map lo[q],hi[q] down following c[q]'s bits,
    // for all n queries, level by level. Processing a whole level across queries
    // issues n independent rank loads that overlap in the memory system (MLP);
    // an explicit prefetch pass warms them. In-place on lo/hi.
    void
    map_batch(const uint32_t* c, size_t* lo, size_t* hi, size_t n) const {
        for (uint32_t l = 0; l < levels_; ++l) {
            const BitVector& bv = bv_[l];
            const size_t z = z_[l];
            for (size_t q = 0; q < n; ++q) {
                bv.prefetch(lo[q]);
                bv.prefetch(hi[q]);
            }
            for (size_t q = 0; q < n; ++q) {
                if ((c[q] >> (levels_ - 1 - l)) & 1u) {
                    lo[q] = z + bv.rank1(lo[q]);
                    hi[q] = z + bv.rank1(hi[q]);
                } else {
                    lo[q] = bv.rank0(lo[q]);
                    hi[q] = bv.rank0(hi[q]);
                }
            }
        }
    }

    // Descended position of 0 following c's bit path — a per-symbol constant.
    size_t
    map_zero(uint32_t c) const {
        size_t p = 0;
        for (uint32_t l = 0; l < levels_; ++l) {
            if ((c >> (levels_ - 1 - l)) & 1u) {
                p = z_[l] + bv_[l].rank1(p);
            } else {
                p = bv_[l].rank0(p);
            }
        }
        return p;
    }

    size_t
    size() const {
        return n_;
    }
    uint32_t
    levels() const {
        return levels_;
    }

    // --- serialization ---
    const std::vector<BitVector>&
    levels_bv() const {
        return bv_;
    }
    const std::vector<size_t>&
    zeros() const {
        return z_;
    }
    static WaveletMatrix
    from_parts(size_t n, uint32_t levels, std::vector<BitVector> bv,
               std::vector<size_t> z) {
        WaveletMatrix wm;
        wm.n_ = n;
        wm.levels_ = levels;
        wm.bv_ = std::move(bv);
        wm.z_ = std::move(z);
        return wm;
    }

 private:
    size_t n_ = 0;
    uint32_t levels_ = 0;
    std::vector<BitVector> bv_;
    std::vector<size_t> z_;
};

}  // namespace milvus::index::fmindex
