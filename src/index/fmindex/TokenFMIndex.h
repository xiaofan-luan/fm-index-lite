// Licensed to the LF AI & Data foundation under Apache-2.0.
// Token-level FM-index: same machinery as the byte FMIndex, but the symbols are
// 32-bit token ids (from a tokenizer's vocabulary) instead of bytes. This makes
// substring / n-gram queries operate at the granularity of "N consecutive
// tokens" — the unit LLM decontamination actually uses (e.g. a 13-token overlap,
// GPT-3 style) — and turns NextTokenCounts into a real token n-gram model (an
// Infini-gram-style unbounded-context LM over the corpus).
//
// The caller owns tokenization: pass token ids in, get token positions/ids out.
// No tokenizer dependency lives here. In-memory build + query; serialization and
// unification with the byte index (a common templated core) are follow-ups.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include "index/fmindex/BitVector.h"
#include "index/fmindex/SuffixArray.h"
#include "index/fmindex/WaveletMatrix4.h"

namespace milvus::index::fmindex {

class TokenFMIndex {
 public:
    TokenFMIndex() = default;

    // Build over a contiguous array of `ntok` token ids. `len` semantics are in
    // TOKENS (not bytes); positions returned by Locate are token positions.
    void
    Build(const uint32_t* tokens, size_t ntok, uint32_t sa_sample_rate = 32) {
        sa_sample_rate_ = sa_sample_rate == 0 ? 1 : sa_sample_rate;
        text_len_ = ntok;
        doc_start_ = {0};

        // Dense, order-preserving alphabet: sorted distinct tokens -> ids 1..sigma
        // (0 = sentinel), so the wavelet depth is ceil(log2(sigma)) not 32.
        vocab_.assign(tokens, tokens + ntok);
        std::sort(vocab_.begin(), vocab_.end());
        vocab_.erase(std::unique(vocab_.begin(), vocab_.end()), vocab_.end());
        sigma_ = static_cast<uint32_t>(vocab_.size()) + 1;
        uint32_t bits = 1;
        while ((1u << bits) < sigma_) {
            ++bits;
        }
        qlevels_ = (bits + 1) / 2;

        // remapped token ids + sentinel 0
        std::vector<uint32_t> t(ntok + 1);
        for (size_t i = 0; i < ntok; ++i) {
            t[i] = static_cast<uint32_t>(id_of(tokens[i]));  // >= 1
        }
        t[ntok] = 0;

        std::vector<uint32_t> sa = build_suffix_array(t);  // libsais_int
        const size_t m = t.size();

        std::vector<uint32_t> bwt(m);
        for (size_t i = 0; i < m; ++i) {
            bwt[i] = (sa[i] == 0) ? t[m - 1] : t[sa[i] - 1];
        }

        c_.assign(sigma_, 0);
        for (uint32_t sym : bwt) {
            c_[sym]++;
        }
        uint64_t acc = 0;
        for (uint32_t c = 0; c < sigma_; ++c) {
            uint64_t cnt = c_[c];
            c_[c] = acc;
            acc += cnt;
        }

        wm_ = WaveletMatrix4(std::move(bwt), qlevels_);  // uint32 symbols
        first_.resize(sigma_);
        for (uint32_t c = 0; c < sigma_; ++c) {
            first_[c] = wm_.map_zero(c);
        }

        BitVector samp(m);
        sa_sample_vals_.clear();
        sa_sample_vals_.reserve(m / sa_sample_rate_ + 1);
        for (size_t i = 0; i < m; ++i) {
            if (sa[i] % sa_sample_rate_ == 0) {
                samp.set(i);
                sa_sample_vals_.push_back(sa[i]);
            }
        }
        samp.build_rank();
        sampled_bv_ = std::move(samp);
        buildIsa();
    }

    void
    SetDocStarts(std::vector<uint64_t> doc_start) {
        doc_start_ = std::move(doc_start);
        if (doc_start_.empty() || doc_start_[0] != 0) {
            doc_start_.insert(doc_start_.begin(), 0);
        }
    }

    // Half-open SA interval for a token pattern; lo==hi means no match.
    std::pair<size_t, size_t>
    BackwardSearch(const uint32_t* pat, size_t plen) const {
        if (c_.empty()) {
            return {0, 0};
        }
        const size_t m = text_len_ + 1;
        if (plen == 0) {
            return {0, m};
        }
        size_t lo = 0, hi = m;
        for (size_t k = plen; k-- > 0;) {
            int64_t id = id_of(pat[k]);
            if (id < 0) {
                return {0, 0};
            }
            auto pp = wm_.map2(static_cast<uint32_t>(id), lo, hi);
            size_t base = c_[id] - first_[id];
            lo = base + pp.first;
            hi = base + pp.second;
            if (lo >= hi) {
                return {0, 0};
            }
        }
        return {lo, hi};
    }

    size_t
    Count(const uint32_t* pat, size_t plen) const {
        auto r = BackwardSearch(pat, plen);
        return r.second - r.first;
    }

    // Sorted token positions of every occurrence.
    std::vector<uint64_t>
    Locate(const uint32_t* pat, size_t plen) const {
        auto r = BackwardSearch(pat, plen);
        std::vector<uint64_t> out;
        for (size_t i = r.first; i < r.second; ++i) {
            size_t row = i;
            uint64_t steps = 0;
            while (!sampled_bv_.get(row)) {
                row = LF(row);
                ++steps;
            }
            uint64_t pos = sa_sample_vals_[sampled_bv_.rank1(row)] + steps;
            if (pos < text_len_) {
                out.push_back(pos);
            }
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    // Sorted (doc_id, token_offset) of every occurrence fully inside a document.
    std::vector<std::pair<uint64_t, uint64_t>>
    LocateDocs(const uint32_t* pat, size_t plen) const {
        auto pos = Locate(pat, plen);
        std::vector<std::pair<uint64_t, uint64_t>> out;
        for (uint64_t p : pos) {
            auto it = std::upper_bound(doc_start_.begin(), doc_start_.end(), p);
            uint64_t doc = static_cast<uint64_t>(it - doc_start_.begin()) - 1;
            uint64_t doc_end = (doc + 1 < doc_start_.size()) ? doc_start_[doc + 1]
                                                             : text_len_;
            if (p + plen > doc_end) {
                continue;  // spills across a document boundary
            }
            out.emplace_back(doc, p - doc_start_[doc]);
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    struct LongestMatchResult {
        size_t length;
        size_t query_pos;
        size_t count;
    };
    // Longest contiguous run of `query` tokens that occurs in the corpus.
    LongestMatchResult
    LongestMatch(const uint32_t* query, size_t qlen) const {
        LongestMatchResult best{0, 0, 0};
        if (c_.empty()) {
            return best;
        }
        const size_t m = text_len_ + 1;
        for (size_t e = 0; e < qlen; ++e) {
            size_t lo = 0, hi = m, len = 0;
            for (size_t k = e + 1; k-- > 0;) {
                int64_t id = id_of(query[k]);
                if (id < 0) {
                    break;
                }
                auto pp = wm_.map2(static_cast<uint32_t>(id), lo, hi);
                size_t base = c_[id] - first_[id];
                size_t nlo = base + pp.first, nhi = base + pp.second;
                if (nlo >= nhi) {
                    break;
                }
                lo = nlo;
                hi = nhi;
                ++len;
                if (len > best.length) {
                    best.length = len;
                    best.query_pos = k;
                    best.count = hi - lo;
                }
            }
        }
        return best;
    }

    // Distribution of the token that FOLLOWS context P: (token, count) for every
    // token that occurs after P, sorted by token id. The corpus n-gram model:
    // P(next=tok | P) = count / sum. Sum can be < Count(P) (P at corpus end).
    std::vector<std::pair<uint32_t, size_t>>
    NextTokenCounts(const uint32_t* pat, size_t plen) const {
        std::vector<std::pair<uint32_t, size_t>> out;
        if (c_.empty()) {
            return out;
        }
        auto pr = BackwardSearch(pat, plen);
        if (pr.first >= pr.second) {
            return out;
        }
        std::vector<uint32_t> buf(pat, pat + plen);
        buf.push_back(0);
        for (uint32_t tok : vocab_) {  // vocab_ is sorted, so output is sorted
            buf[plen] = tok;
            auto r = BackwardSearch(buf.data(), plen + 1);
            size_t cnt = r.second - r.first;
            if (cnt) {
                out.emplace_back(tok, cnt);
            }
        }
        return out;
    }

    // Recover the original token ids T[pos, pos+len).
    std::vector<uint32_t>
    Extract(uint64_t pos, size_t len) const {
        std::vector<uint32_t> out;
        if (c_.empty() || pos >= text_len_) {
            return out;
        }
        uint64_t end = pos + std::min<uint64_t>(len, text_len_ - pos);
        uint64_t k = (end + sa_sample_rate_ - 1) / sa_sample_rate_;
        size_t row;
        uint64_t p;
        if (k < isa_sample_.size()) {
            row = isa_sample_[k];
            p = k * sa_sample_rate_;
        } else {
            row = 0;
            p = text_len_;
        }
        out.assign(static_cast<size_t>(end - pos), 0);
        while (p > pos) {
            uint32_t sym = wm_.access(row);  // dense id of T[p-1]
            if (p - 1 < end) {
                out[static_cast<size_t>(p - 1 - pos)] =
                    sym == 0 ? 0 : vocab_[sym - 1];  // dense id -> token
            }
            row = LF(row);
            --p;
        }
        return out;
    }

    bool
    valid() const {
        return !c_.empty();
    }
    uint32_t
    alphabet() const {
        return sigma_;
    }
    uint64_t
    length() const {
        return text_len_;
    }

 private:
    int64_t
    id_of(uint32_t tok) const {
        auto it = std::lower_bound(vocab_.begin(), vocab_.end(), tok);
        if (it == vocab_.end() || *it != tok) {
            return -1;
        }
        return (it - vocab_.begin()) + 1;  // dense id; 0 is the sentinel
    }
    size_t
    LF(size_t i) const {
        uint32_t sym = wm_.access(i);
        return c_[sym] + wm_.rank(sym, i);
    }
    void
    buildIsa() {
        const size_t m = text_len_ + 1;
        isa_sample_.assign(text_len_ / sa_sample_rate_ + 1, 0);
        for (size_t r = 0; r < m; ++r) {
            if (sampled_bv_.get(r)) {
                uint64_t val = sa_sample_vals_[sampled_bv_.rank1(r)];
                isa_sample_[val / sa_sample_rate_] = static_cast<uint64_t>(r);
            }
        }
    }

    uint32_t sa_sample_rate_ = 1;
    uint64_t text_len_ = 0;
    uint32_t sigma_ = 1;
    uint32_t qlevels_ = 0;
    std::vector<uint32_t> vocab_;  // sorted distinct tokens; dense id d -> vocab_[d-1]
    WaveletMatrix4 wm_;
    std::vector<uint64_t> c_;
    std::vector<size_t> first_;
    BitVector sampled_bv_;
    std::vector<uint64_t> sa_sample_vals_;
    std::vector<uint64_t> doc_start_;
    std::vector<uint64_t> isa_sample_;
};

}  // namespace milvus::index::fmindex
