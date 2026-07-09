// Licensed to the LF AI & Data foundation under Apache-2.0.
// 4-ary quad wavelet matrix (Ceregini-Kurpicz-Venturini, "Faster Wavelet Trees
// with Quad Vectors," DCC 2024). Two bits consumed per level => half the levels
// of a binary wavelet matrix => half the dependent cache misses on the
// backward-search hot path. Same interface as WaveletMatrix so FMIndex can swap.
#pragma once
#include <array>
#include <cstdint>
#include <utility>
#include <vector>
#include "index/fmindex/QuadVector.h"

namespace milvus::index::fmindex {

class WaveletMatrix4 {
 public:
    WaveletMatrix4() = default;

    // seq holds symbols in [0, 4^qlevels). qlevels = ceil(log2(sigma)/2).
    WaveletMatrix4(const std::vector<uint32_t>& seq, uint32_t qlevels)
        : n_(seq.size()), qlevels_(qlevels) {
        qv_.reserve(qlevels_);
        start_.assign(qlevels_, {0, 0, 0, 0});
        std::vector<uint32_t> cur = seq;
        std::vector<uint8_t> digits(n_);
        std::array<std::vector<uint32_t>, 4> bucket;
        for (uint32_t l = 0; l < qlevels_; ++l) {
            uint32_t shift = 2 * (qlevels_ - 1 - l);
            std::array<size_t, 4> hist{0, 0, 0, 0};
            for (size_t i = 0; i < n_; ++i) {
                uint8_t d = (cur[i] >> shift) & 3u;
                digits[i] = d;
                ++hist[d];
            }
            qv_.emplace_back(digits);
            start_[l][0] = 0;
            start_[l][1] = hist[0];
            start_[l][2] = hist[0] + hist[1];
            start_[l][3] = hist[0] + hist[1] + hist[2];
            // stable partition cur into the 4 digit groups, order preserved
            for (auto& b : bucket) {
                b.clear();
            }
            for (size_t i = 0; i < n_; ++i) {
                bucket[digits[i]].push_back(cur[i]);
            }
            size_t o = 0;
            for (auto& b : bucket) {
                for (uint32_t x : b) {
                    cur[o++] = x;
                }
            }
        }
    }

    // count of symbol c in [0, i)
    size_t
    rank(uint32_t c, size_t i) const {
        size_t first = 0, last = i;
        for (uint32_t l = 0; l < qlevels_; ++l) {
            uint8_t d = (c >> (2 * (qlevels_ - 1 - l))) & 3u;
            first = start_[l][d] + qv_[l].rank(d, first);
            last = start_[l][d] + qv_[l].rank(d, last);
        }
        return last - first;
    }

    uint32_t
    access(size_t i) const {
        uint32_t sym = 0;
        size_t pos = i;
        for (uint32_t l = 0; l < qlevels_; ++l) {
            uint8_t d = qv_[l].at(pos);
            sym = (sym << 2) | d;
            pos = start_[l][d] + qv_[l].rank(d, pos);
        }
        return sym;
    }

    // descend lo and hi following c's digit path (no group-start baseline)
    std::pair<size_t, size_t>
    map2(uint32_t c, size_t lo, size_t hi) const {
        for (uint32_t l = 0; l < qlevels_; ++l) {
            uint8_t d = (c >> (2 * (qlevels_ - 1 - l))) & 3u;
            lo = start_[l][d] + qv_[l].rank(d, lo);
            hi = start_[l][d] + qv_[l].rank(d, hi);
        }
        return {lo, hi};
    }

    size_t
    map_zero(uint32_t c) const {
        size_t p = 0;
        for (uint32_t l = 0; l < qlevels_; ++l) {
            uint8_t d = (c >> (2 * (qlevels_ - 1 - l))) & 3u;
            p = start_[l][d] + qv_[l].rank(d, p);
        }
        return p;
    }

    // batched level-major descent with prefetch (memory-level parallelism)
    void
    map_batch(const uint32_t* c, size_t* lo, size_t* hi, size_t n) const {
        for (uint32_t l = 0; l < qlevels_; ++l) {
            const QuadVector& q = qv_[l];
            uint32_t shift = 2 * (qlevels_ - 1 - l);
            for (size_t k = 0; k < n; ++k) {
                q.prefetch(lo[k]);
                q.prefetch(hi[k]);
            }
            for (size_t k = 0; k < n; ++k) {
                uint8_t d = (c[k] >> shift) & 3u;
                lo[k] = start_[l][d] + q.rank(d, lo[k]);
                hi[k] = start_[l][d] + q.rank(d, hi[k]);
            }
        }
    }

    size_t
    size() const {
        return n_;
    }
    uint32_t
    qlevels() const {
        return qlevels_;
    }

    // --- serialization ---
    const std::vector<QuadVector>&
    levels_qv() const {
        return qv_;
    }
    const std::vector<std::array<size_t, 4>>&
    starts() const {
        return start_;
    }
    static WaveletMatrix4
    from_parts(size_t n, uint32_t qlevels, std::vector<QuadVector> qv,
               std::vector<std::array<size_t, 4>> start) {
        WaveletMatrix4 wm;
        wm.n_ = n;
        wm.qlevels_ = qlevels;
        wm.qv_ = std::move(qv);
        wm.start_ = std::move(start);
        return wm;
    }

 private:
    size_t n_ = 0;
    uint32_t qlevels_ = 0;
    std::vector<QuadVector> qv_;
    std::vector<std::array<size_t, 4>> start_;
};

}  // namespace milvus::index::fmindex
