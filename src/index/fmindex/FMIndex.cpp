// Licensed to the LF AI & Data foundation under Apache-2.0.
// Portions translated from Lance (lance_index::scalar::fmindex), Apache-2.0.
#include "index/fmindex/FMIndex.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include "index/fmindex/SuffixArray.h"

#ifdef FMIX_BUILD_MEM_PROFILE
#include <sys/resource.h>
#define FMIX_MEM(tag)                                                        \
    do {                                                                     \
        struct rusage ru;                                                    \
        getrusage(RUSAGE_SELF, &ru);                                         \
        std::fprintf(stderr, "  [mem] %-20s peak=%ld MB\n", tag,             \
                     ru.ru_maxrss / (1024 * 1024));                          \
    } while (0)
#else
#define FMIX_MEM(tag) ((void)0)
#endif

namespace milvus::index::fmindex {

void
FMIndex::Build(const std::vector<std::string_view>& docs, uint32_t sa_sample_rate,
               bool case_insensitive, bool force_wide) {
    // Concatenate the documents into one internal buffer, injecting a '\0'
    // separator after each. That separator is what makes every query
    // document-scoped: no pattern free of '\0' can match a substring that
    // straddles it, so cross-document matches simply do not exist in the index.
    uint64_t internal_len = 0;
    for (const auto& d : docs) {
        internal_len += d.size() + 1;  // + separator
    }
    std::vector<uint8_t> buf;
    buf.reserve(internal_len);
    doc_start_.clear();
    doc_start_.reserve(docs.size() + 1);
    for (const auto& d : docs) {
        // '\0' is the reserved document separator; a document that contains it
        // would forge a boundary and break the no-cross-document guarantee.
        if (std::memchr(d.data(), '\0', d.size()) != nullptr) {
            throw std::invalid_argument(
                "FMIndex::Build: document contains a '\\0' byte (reserved as "
                "the document separator)");
        }
        doc_start_.push_back(buf.size());
        buf.insert(buf.end(), d.begin(), d.end());
        buf.push_back('\0');  // document separator
    }
    doc_start_.push_back(buf.size());  // end boundary (== internal_len)
    const uint8_t* data = buf.data();
    const size_t len = static_cast<size_t>(internal_len);

    // Suffix array is built with 32-bit indices under 2 GiB (compact) and 64-bit
    // indices at or above it (libsais64). Only the astronomically large 2^63
    // ceiling of the 64-bit path is a hard error.
    const bool use64 = force_wide || len >= (static_cast<size_t>(INT32_MAX));
    if (len >= (static_cast<size_t>(1) << 63)) {
        throw std::length_error("FMIndex: corpus too large (>= 2^63 bytes)");
    }
    sa_sample_rate_ = sa_sample_rate == 0 ? 1 : sa_sample_rate;
    case_fold_ = case_insensitive;
    // Store positions 8 bytes wide once they can exceed uint32 (>= 4 GiB), or
    // when forced (lets tests exercise the wide path on small inputs).
    wide_storage_ = force_wide || len >= (static_cast<size_t>(1) << 32);
    text_len_ = len;

    // ASCII case fold: A-Z -> a-z when case_insensitive, identity otherwise.
    auto fold = [cf = case_fold_](uint8_t b) -> uint8_t {
        return (cf && b >= 'A' && b <= 'Z') ? static_cast<uint8_t>(b + 32) : b;
    };

    // 1. dense alphabet: distinct (folded) bytes -> ids 1..sigma-1 (0 =
    //    sentinel), order-preserving so lexicographic order of ids matches
    //    bytes. With case folding, both cases of a letter alias to one id, so
    //    the query hot path stays a plain byte_to_id_ lookup (no fold branch).
    std::array<bool, 256> present{};
    for (size_t i = 0; i < len; ++i) {
        present[fold(data[i])] = true;
    }
    byte_to_id_.fill(-1);
    uint32_t id = 1;
    for (int b = 0; b < 256; ++b) {
        if (present[b]) {
            byte_to_id_[b] = static_cast<int32_t>(id++);
        }
    }
    if (case_fold_) {
        for (int b = 'A'; b <= 'Z'; ++b) {
            byte_to_id_[b] = byte_to_id_[b + 32];  // uppercase aliases lowercase
        }
    }
    sigma_ = id;  // sentinel + number of distinct bytes
    uint32_t bits = 1;
    while ((1u << bits) < sigma_) {
        ++bits;
    }
    qlevels_ = (bits + 1) / 2;  // 2 bits per quad level

    // 2. suffix array — built directly over bytes (see build_suffix_array_bytes),
    //    so no int32 dense-id copy of the whole text is materialized. For a
    //    case-insensitive index the SA must sort by the folded bytes, so feed a
    //    folded copy; otherwise libsais reads the caller's buffer in place.
    const size_t m = len + 1;
    // 32-bit SA under 2 GiB, 64-bit at/above it (or when forced). Only one of
    // the two buffers is allocated; sa_at() reads whichever, as a 64-bit value.
    std::vector<int32_t> sa32;
    std::vector<int64_t> sa64;
    {
        std::vector<uint8_t> folded;
        const uint8_t* sa_input = data;
        if (case_fold_) {
            folded.resize(len);
            for (size_t i = 0; i < len; ++i) {
                folded[i] = fold(data[i]);
            }
            sa_input = folded.data();
        }
        if (use64) {
            sa64 = build_suffix_array_bytes64(sa_input, len);
        } else {
            sa32 = build_suffix_array_bytes(sa_input, len);
        }
    }  // folded buffer freed here
    FMIX_MEM("after SA");
    auto sa_at = [&](size_t i) -> uint64_t {
        return use64 ? static_cast<uint64_t>(sa64[i])
                     : static_cast<uint64_t>(sa32[i]);
    };

    // 3. BWT of dense ids, computed inline from data + sa (no dense-id text
    //    array). bwt[i] is the symbol preceding row i's suffix; row 0's suffix
    //    starts at position 0 so its predecessor wraps to the sentinel (0). The
    //    aliased byte_to_id_ maps both letter cases to one id in folded mode.
    // BWT symbols are dense ids in [0, sigma) with sigma <= 257, so uint16 holds
    // them and halves this buffer (and the wavelet partition buffers it feeds).
    std::vector<uint16_t> bwt(m);
    for (size_t i = 0; i < m; ++i) {
        uint64_t v = sa_at(i);
        bwt[i] = (v == 0) ? uint16_t{0}
                          : static_cast<uint16_t>(byte_to_id_[data[v - 1]]);
    }
    std::vector<uint8_t>().swap(buf);  // internal text no longer needed
    data = nullptr;
    FMIX_MEM("after BWT");

    // 4. C-table over sigma_ (cumulative counts can reach m, so 64-bit)
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

    // 5. sampled SA (needs the SA, not the BWT): do it before the wavelet so the
    //    SA can be freed and does not stack onto the wavelet's peak memory.
    BitVector samp(m);
    sa_sample_vals_.clear();
    sa_sample_vals_.reserve(m / sa_sample_rate_ + 1);
    for (size_t i = 0; i < m; ++i) {
        uint64_t v = sa_at(i);
        if (v % sa_sample_rate_ == 0) {
            samp.set(i);
            sa_sample_vals_.push_back(v);
        }
    }
    samp.build_rank();
    sampled_bv_ = std::move(samp);
    std::vector<int32_t>().swap(sa32);  // SA no longer needed — free before wavelet
    std::vector<int64_t>().swap(sa64);
    FMIX_MEM("after sampling");

    // 6. quad wavelet matrix over the BWT (moved in — the BWT buffer becomes the
    //    wavelet's working array, no copy) + per-symbol map_zero baseline.
    wm_ = WaveletMatrix4(std::move(bwt), qlevels_);
    FMIX_MEM("after wavelet");
    first_.resize(sigma_);
    for (uint32_t c = 0; c < sigma_; ++c) {
        first_[c] = wm_.map_zero(c);
    }

    buildDerived();
}

void
FMIndex::buildDerived() {
    // id -> byte (inverse of byte_to_id_). For a case-folded index both 'A' and
    // 'a' map to one id; ascending iteration lets lowercase win, so Extract
    // returns the canonical lowercase byte.
    id_to_byte_.assign(sigma_, 0);
    for (int b = 0; b < 256; ++b) {
        if (byte_to_id_[b] > 0) {
            id_to_byte_[byte_to_id_[b]] = static_cast<uint8_t>(b);
        }
    }
    sep_id_ = byte_to_id_[0];  // dense id of the '\0' separator (-1 if none)
    // isa_sample_[k] = the BWT row whose suffix starts at text position k*rate.
    // Every multiple of rate in [0, text_len_] appears exactly once in the SA,
    // so this array is fully populated. Gives Extract an anchor within `rate`
    // steps of any position without needing select/psi.
    const size_t m = text_len_ + 1;
    isa_sample_.assign(text_len_ / sa_sample_rate_ + 1, 0);
    for (size_t r = 0; r < m; ++r) {
        if (sampled_bv_.get(r)) {
            uint64_t val = sa_sample_vals_[sampled_bv_.rank1(r)];
            isa_sample_[val / sa_sample_rate_] = static_cast<uint64_t>(r);
        }
    }
}

std::string
FMIndex::Extract(uint64_t doc_id, uint64_t offset, size_t len) const {
    if (c_.empty() || doc_id + 1 >= doc_start_.size()) {
        return {};  // empty index or doc_id out of range
    }
    // Translate (doc, offset) to internal coordinates and clamp to the document's
    // content end (the byte before its '\0'), so Extract never crosses into the
    // separator or the next document.
    uint64_t dstart = doc_start_[doc_id];
    uint64_t dlen = doc_start_[doc_id + 1] - 1 - dstart;  // content length
    if (offset >= dlen) {
        return {};
    }
    uint64_t pos = dstart + offset;
    uint64_t end = pos + std::min<uint64_t>(len, dlen - offset);
    // Anchor: a row whose suffix starts at a sampled position >= end, then walk
    // backward (LF) collecting BWT chars, which are the text bytes before each
    // row's suffix. Prefer the nearest sample above `end`; fall back to row 0
    // (the sentinel suffix, which starts at text_len_) when none exists.
    uint64_t k = (end + sa_sample_rate_ - 1) / sa_sample_rate_;  // ceil(end/rate)
    size_t row;
    uint64_t p;
    if (k < isa_sample_.size()) {
        row = isa_sample_[k];
        p = k * sa_sample_rate_;
    } else {
        row = 0;  // SA[0] is the sentinel-only suffix, starting at text_len_
        p = text_len_;
    }
    std::string out(static_cast<size_t>(end - pos), '\0');
    while (p > pos) {
        uint32_t sym = wm_.access(row);  // BWT[row] = T[p-1]
        if (p - 1 < end) {
            out[static_cast<size_t>(p - 1 - pos)] =
                static_cast<char>(id_to_byte_[sym]);
        }
        row = LF(row);
        --p;
    }
    return out;
}

FMIndex::LongestMatchResult
FMIndex::LongestMatch(const uint8_t* query, size_t qlen) const {
    LongestMatchResult best{0, 0, 0};
    if (c_.empty()) {
        return best;
    }
    const size_t m = text_len_ + 1;
    // For each end position e, extend the match backward (query[k..e]) as far as
    // the SA interval stays non-empty; track the global longest across all e.
    for (size_t e = 0; e < qlen; ++e) {
        size_t lo = 0, hi = m, len = 0;
        for (size_t k = e + 1; k-- > 0;) {
            int32_t id = byte_to_id_[query[k]];
            if (id < 0 || id == sep_id_) {
                break;  // byte absent, or the separator — match cannot extend
            }
            auto pp = wm_.map2(static_cast<uint32_t>(id), lo, hi);
            size_t base = c_[id] - first_[id];
            size_t nlo = base + pp.first, nhi = base + pp.second;
            if (nlo >= nhi) {
                break;  // query[k..e] does not occur — stop extending
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

std::vector<std::pair<uint8_t, size_t>>
FMIndex::NextTokenCounts(const uint8_t* pattern, size_t plen) const {
    std::vector<std::pair<uint8_t, size_t>> out;
    if (c_.empty()) {
        return out;
    }
    // No continuations if P itself does not occur.
    auto pr = BackwardSearch(pattern, plen);
    if (pr.first >= pr.second) {
        return out;
    }
    // count(P·b) = number of occurrences of the (plen+1)-length string P then b.
    std::vector<uint8_t> buf(pattern, pattern + plen);
    buf.push_back(0);
    for (int b = 1; b < 256; ++b) {  // skip b==0: the '\0' separator, not a real byte
        if (byte_to_id_[b] < 0) {
            continue;  // byte never appears in the corpus
        }
        buf[plen] = static_cast<uint8_t>(b);
        auto r = BackwardSearch(buf.data(), plen + 1);
        size_t cnt = r.second - r.first;
        if (cnt) {
            out.emplace_back(static_cast<uint8_t>(b), cnt);
        }
    }
    return out;
}

std::vector<uint64_t>
FMIndex::MatchingDocs(const uint8_t* pat, size_t plen) const {
    auto hits = LocateDocs(pat, plen);  // (doc, offset), cross-doc seam filtered
    std::vector<uint64_t> docs;
    docs.reserve(hits.size());
    for (auto& h : hits) {
        docs.push_back(h.first);
    }
    docs.erase(std::unique(docs.begin(), docs.end()), docs.end());  // sorted
    return docs;
}

std::vector<uint64_t>
FMIndex::FuzzyMatchingDocs(const uint8_t* pat, size_t plen, uint32_t k) const {
    if (c_.empty() || plen == 0) {
        return {};
    }
    const size_t m = text_len_ + 1;
    // Backtracking backward search: DFS over SA intervals, spending an error
    // budget on substitution / insertion (extra text char) / deletion (skipped
    // pattern char). The DFS never steps onto the separator id, so every matched
    // string T' lies inside one document — no length/boundary bookkeeping needed.
    // At i==0 record T''s SA interval; memoize (lo,hi,i)->min errors to prune.
    std::vector<std::pair<size_t, size_t>> hits;
    std::map<std::tuple<size_t, size_t, size_t>, uint32_t> visited;
    std::function<void(size_t, size_t, size_t, uint32_t)> dfs =
        [&](size_t lo, size_t hi, size_t i, uint32_t e) {
            if (i == 0) {
                hits.emplace_back(lo, hi);
                return;
            }
            auto key = std::make_tuple(lo, hi, i);
            auto v = visited.find(key);
            if (v != visited.end() && v->second <= e) {
                return;
            }
            visited[key] = e;
            int32_t tid = byte_to_id_[pat[i - 1]];
            if (tid == sep_id_) {
                tid = -1;  // a '\0' in the pattern never matches a real byte
            }
            auto step = [&](uint32_t d, size_t& nlo, size_t& nhi) {
                auto pp = wm_.map2(d, lo, hi);
                size_t base = c_[d] - first_[d];
                nlo = base + pp.first;
                nhi = base + pp.second;
            };
            if (e == k) {  // budget spent: only exact matches from here on
                if (tid >= 0) {
                    size_t nlo, nhi;
                    step(static_cast<uint32_t>(tid), nlo, nhi);
                    if (nlo < nhi) {
                        dfs(nlo, nhi, i - 1, e);
                    }
                }
                return;
            }
            dfs(lo, hi, i - 1, e + 1);  // deletion: skip pat[i-1], no text char
            for (uint32_t d = 1; d < sigma_; ++d) {
                if (static_cast<int32_t>(d) == sep_id_) {
                    continue;  // never edit toward the separator (stays in-doc)
                }
                size_t nlo, nhi;
                step(d, nlo, nhi);
                if (nlo >= nhi) {
                    continue;
                }
                if (tid >= 0 && static_cast<uint32_t>(tid) == d) {
                    dfs(nlo, nhi, i - 1, e);  // match, no cost
                } else {
                    dfs(nlo, nhi, i - 1, e + 1);  // substitution
                }
                dfs(nlo, nhi, i, e + 1);  // insertion: extra text char
            }
        };
    dfs(0, m, plen, 0);

    // Union the documents of every matched interval (each match is in-document).
    std::set<uint64_t> docset;
    std::set<size_t> seen_rows;
    for (auto& h : hits) {
        for (size_t r = h.first; r < h.second; ++r) {
            if (!seen_rows.insert(r).second) {
                continue;  // this row already resolved by another interval
            }
            size_t row = r;
            uint64_t steps = 0;
            while (!sampled_bv_.get(row)) {
                row = LF(row);
                ++steps;
            }
            uint64_t pos = sa_sample_vals_[sampled_bv_.rank1(row)] + steps;
            if (pos < text_len_) {
                docset.insert(docOf(pos));
            }
        }
    }
    return {docset.begin(), docset.end()};
}

size_t
FMIndex::LF(size_t i) const {
    uint32_t sym = wm_.access(i);
    return c_[sym] + wm_.rank(sym, i);
}

std::pair<size_t, size_t>
FMIndex::BackwardSearch(const uint8_t* pattern, size_t plen) const {
    if (c_.empty()) {
        return {0, 0};  // default-constructed / failed-Deserialize index
    }
    const size_t m = text_len_ + 1;
    if (plen == 0) {
        return {0, m};
    }
    size_t lo = 0, hi = m;
    for (size_t k = plen; k-- > 0;) {
        int32_t id = byte_to_id_[pattern[k]];
        if (id < 0 || id == sep_id_) {
            return {0, 0};  // byte absent, or the '\0' separator (never queryable)
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

std::vector<size_t>
FMIndex::CountBatch(
    const std::vector<std::pair<const uint8_t*, size_t>>& patterns) const {
    const size_t m = text_len_ + 1;
    const size_t B = patterns.size();
    std::vector<size_t> result(B, 0);
    if (c_.empty()) {
        return result;  // default-constructed / failed-Deserialize index
    }

    // Process in small tiles so the set of in-flight positions stays L1-resident
    // while still giving enough memory-level parallelism to overlap misses.
    constexpr size_t kTile = 32;

    std::vector<uint32_t> ids(kTile);
    std::vector<size_t> blo(kTile), bhi(kTile);
    std::vector<size_t> qslot(kTile);
    std::vector<size_t> lo(kTile), hi(kTile);
    std::vector<int64_t> posn(kTile);
    std::vector<uint8_t> active(kTile);

    for (size_t t0 = 0; t0 < B; t0 += kTile) {
        size_t t1 = std::min(t0 + kTile, B);
        size_t tn = t1 - t0;
        for (size_t s = 0; s < tn; ++s) {
            size_t len = patterns[t0 + s].second;
            if (len == 0) {
                result[t0 + s] = m;
                active[s] = 0;
                continue;
            }
            lo[s] = 0;
            hi[s] = m;
            posn[s] = static_cast<int64_t>(len) - 1;
            active[s] = 1;
        }
        for (;;) {
            size_t k = 0;  // active-in-tile count this round
            for (size_t s = 0; s < tn; ++s) {
                if (!active[s]) {
                    continue;
                }
                int32_t id = byte_to_id_[patterns[t0 + s].first[posn[s]]];
                if (id < 0 || id == sep_id_) {
                    active[s] = 0;
                    continue;
                }
                ids[k] = static_cast<uint32_t>(id);
                blo[k] = lo[s];
                bhi[k] = hi[s];
                qslot[k] = s;
                ++k;
            }
            if (k == 0) {
                break;
            }
            wm_.map_batch(ids.data(), blo.data(), bhi.data(), k);
            for (size_t j = 0; j < k; ++j) {
                size_t s = qslot[j];
                uint32_t id = ids[j];
                size_t base = c_[id] - first_[id];
                size_t nlo = base + blo[j];
                size_t nhi = base + bhi[j];
                if (nlo >= nhi) {
                    active[s] = 0;
                    continue;
                }
                lo[s] = nlo;
                hi[s] = nhi;
                if (--posn[s] < 0) {
                    result[t0 + s] = nhi - nlo;
                    active[s] = 0;
                }
            }
        }
    }
    return result;
}

uint64_t
FMIndex::docOf(uint64_t internal_pos) const {
    auto it = std::upper_bound(doc_start_.begin(), doc_start_.end(), internal_pos);
    return static_cast<uint64_t>(it - doc_start_.begin()) - 1;
}

std::vector<uint64_t>
FMIndex::locateInternal(const uint8_t* pattern, size_t plen) const {
    if (plen == 0) {
        return {};  // empty pattern: no document-scoped hits (matches
                    // FuzzyMatchingDocs); a raw Count still reports m positions.
    }
    auto r = BackwardSearch(pattern, plen);
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
            out.push_back(pos);  // internal coordinate (separators included)
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<std::pair<uint64_t, uint64_t>>
FMIndex::LocateDocs(const uint8_t* pattern, size_t plen) const {
    std::vector<uint64_t> positions = locateInternal(pattern, plen);
    std::vector<std::pair<uint64_t, uint64_t>> out;
    out.reserve(positions.size());
    for (uint64_t pos : positions) {
        // No occurrence can span a document (the pattern holds no '\0'), so every
        // hit is fully inside doc docOf(pos); the offset is measured from its
        // internal start.
        uint64_t doc = docOf(pos);
        out.emplace_back(doc, pos - doc_start_[doc]);
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<uint64_t>
FMIndex::LocatePrefixDocs(const uint8_t* pattern, size_t plen) const {
    std::vector<uint64_t> positions = locateInternal(pattern, plen);
    std::vector<uint64_t> docs;
    for (uint64_t pos : positions) {
        // A document begins with the pattern iff the occurrence sits exactly on
        // a document's start boundary. It cannot spill past the document (no
        // '\0' in the pattern), so no length check is needed.
        auto it = std::lower_bound(doc_start_.begin(), doc_start_.end(), pos);
        if (it != doc_start_.end() && *it == pos) {
            uint64_t doc_id = static_cast<uint64_t>(it - doc_start_.begin());
            if (doc_id + 1 < doc_start_.size()) {  // not the end sentinel
                docs.push_back(doc_id);
            }
        }
    }
    std::sort(docs.begin(), docs.end());
    docs.erase(std::unique(docs.begin(), docs.end()), docs.end());
    return docs;
}

std::vector<uint64_t>
FMIndex::LocateSuffixDocs(const uint8_t* pattern, size_t plen) const {
    std::vector<uint64_t> positions = locateInternal(pattern, plen);
    std::vector<uint64_t> docs;
    for (uint64_t pos : positions) {
        // A document ends with the pattern iff the occurrence ends exactly on
        // that document's content end (the byte just before its '\0' separator).
        uint64_t doc = docOf(pos);
        uint64_t content_end = doc_start_[doc + 1] - 1;
        if (pos + plen == content_end) {
            docs.push_back(doc);
        }
    }
    std::sort(docs.begin(), docs.end());
    docs.erase(std::unique(docs.begin(), docs.end()), docs.end());
    return docs;
}

// ------------------------- serialization -------------------------
// Format v3: a header of scalars/metadata/section-sizes, then the large payload
// arrays each padded to an 8-byte boundary so LoadView can point at them
// (zero-copy) from mmap'd memory. Only the wavelet/sampled word arrays are
// viewed; the small sample/doc arrays are copied on load either way.
namespace {
template <typename T>
void
put(std::string& s, const T& v) {
    s.append(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T>
T
get(const char*& p, const char* end) {
    if (p + sizeof(T) > end) {
        throw std::runtime_error("fmindex: truncated blob");
    }
    T v;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
}
constexpr uint32_t kMagic = 0x464D4958;  // "FMIX"
constexpr uint32_t kFormatVersion = 6;  // v6: docs '\0'-separated, doc_start_ internal
}  // namespace

void
FMIndex::writeHeader(std::string& s) const {
    put(s, kMagic);
    put(s, kFormatVersion);
    put(s, sa_sample_rate_);
    put(s, sigma_);
    put(s, qlevels_);
    // Flags live in the former 4-byte alignment pad (keeps text_len 8-aligned):
    // bit 0 = ASCII case-insensitive, bit 1 = 8-byte sampled-SA storage.
    put(s, static_cast<uint32_t>((case_fold_ ? 1u : 0u) |
                                 (wide_storage_ ? 2u : 0u)));
    put(s, text_len_);
    for (int32_t v : byte_to_id_) {
        put(s, v);
    }
    for (uint64_t c : c_) {
        put(s, c);
    }
    for (uint32_t l = 0; l < qlevels_; ++l) {
        for (int d = 0; d < 4; ++d) {
            put(s, static_cast<uint64_t>(wm_.starts()[l][d]));
        }
    }
    // section sizes (word counts / element counts)
    for (uint32_t l = 0; l < qlevels_; ++l) {
        put(s, static_cast<uint64_t>(wm_.levels_qv()[l].word_count()));
    }
    put(s, static_cast<uint64_t>(sampled_bv_.word_count()));
    put(s, static_cast<uint64_t>(sa_sample_vals_.size()));
    put(s, static_cast<uint64_t>(doc_start_.size()));
}

std::string
FMIndex::Serialize() const {
    std::string s;
    writeHeader(s);
    auto align8 = [&s] {
        while (s.size() & 7u) {
            s.push_back('\0');
        }
    };
    for (uint32_t l = 0; l < qlevels_; ++l) {
        align8();
        const QuadVector& q = wm_.levels_qv()[l];
        s.append(reinterpret_cast<const char*>(q.words()),
                 q.word_count() * sizeof(uint64_t));
    }
    align8();
    s.append(reinterpret_cast<const char*>(sampled_bv_.words()),
             sampled_bv_.word_count() * sizeof(uint64_t));
    align8();
    if (wide_storage_) {
        s.append(reinterpret_cast<const char*>(sa_sample_vals_.data()),
                 sa_sample_vals_.size() * sizeof(uint64_t));
    } else {
        for (uint64_t v : sa_sample_vals_) {
            uint32_t x = static_cast<uint32_t>(v);
            s.append(reinterpret_cast<const char*>(&x), sizeof(x));
        }
    }
    align8();
    s.append(reinterpret_cast<const char*>(doc_start_.data()),
             doc_start_.size() * sizeof(uint64_t));
    return s;
}

bool
FMIndex::SerializeToFile(const std::string& path) const {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    // Only the small header is buffered; the large arrays stream straight to
    // the file (no intermediate full-index copy).
    std::string header;
    writeHeader(header);
    size_t off = 0;
    bool ok = true;
    auto emit = [&](const void* p, size_t n) {
        if (n && std::fwrite(p, 1, n, f) != n) {
            ok = false;
        }
        off += n;
    };
    static const char kZero[8] = {0};
    auto pad8 = [&] { emit(kZero, (8 - (off & 7u)) & 7u); };
    emit(header.data(), header.size());
    for (uint32_t l = 0; l < qlevels_; ++l) {
        pad8();
        const QuadVector& q = wm_.levels_qv()[l];
        emit(q.words(), q.word_count() * sizeof(uint64_t));
    }
    pad8();
    emit(sampled_bv_.words(), sampled_bv_.word_count() * sizeof(uint64_t));
    pad8();
    if (wide_storage_) {
        emit(sa_sample_vals_.data(), sa_sample_vals_.size() * sizeof(uint64_t));
    } else {
        for (uint64_t v : sa_sample_vals_) {
            uint32_t x = static_cast<uint32_t>(v);
            emit(&x, sizeof(x));
        }
    }
    pad8();
    emit(doc_start_.data(), doc_start_.size() * sizeof(uint64_t));
    std::fclose(f);
    return ok;
}

bool
FMIndex::parseView(const uint8_t* base, size_t size) {
    const char* p = reinterpret_cast<const char*>(base);
    const char* end = p + size;
    try {
        if (get<uint32_t>(p, end) != kMagic ||
            get<uint32_t>(p, end) != kFormatVersion) {
            return false;
        }
        sa_sample_rate_ = get<uint32_t>(p, end);
        sigma_ = get<uint32_t>(p, end);
        qlevels_ = get<uint32_t>(p, end);
        uint32_t flags = get<uint32_t>(p, end);  // former pad, now flags
        case_fold_ = (flags & 1u) != 0;
        wide_storage_ = (flags & 2u) != 0;
        if (sigma_ == 0 || sigma_ > 257 || qlevels_ > 5) {
            return false;
        }
        // qlevels_ must be exactly what Build derives from sigma_ (2 bits/level
        // over ceil(log2(sigma_)) bits). A mismatched value would consume the
        // wrong number of bits per symbol and silently alias distinct ids onto
        // one wavelet path — wrong (but memory-safe) query answers.
        {
            uint32_t bits = 1;
            while ((1u << bits) < sigma_) {
                ++bits;
            }
            if (qlevels_ != (bits + 1) / 2) {
                return false;
            }
        }
        text_len_ = get<uint64_t>(p, end);
        for (int i = 0; i < 256; ++i) {
            byte_to_id_[i] = get<int32_t>(p, end);
            // A byte maps to -1 (absent) or a dense id in [1, sigma_). Anything
            // else would index c_/first_ (size sigma_) out of bounds in a query.
            if (byte_to_id_[i] < -1 ||
                byte_to_id_[i] >= static_cast<int32_t>(sigma_)) {
                return false;
            }
        }
        c_.resize(sigma_);
        for (uint32_t i = 0; i < sigma_; ++i) {
            c_[i] = get<uint64_t>(p, end);
        }
        std::vector<std::array<size_t, 4>> starts(qlevels_);
        for (uint32_t l = 0; l < qlevels_; ++l) {
            for (int d = 0; d < 4; ++d) {
                starts[l][d] = static_cast<size_t>(get<uint64_t>(p, end));
            }
        }
        std::vector<uint64_t> qnw(qlevels_);
        for (uint32_t l = 0; l < qlevels_; ++l) {
            qnw[l] = get<uint64_t>(p, end);
        }
        uint64_t sampled_nw = get<uint64_t>(p, end);
        uint64_t n_samples = get<uint64_t>(p, end);
        uint64_t n_docs = get<uint64_t>(p, end);

        const size_t m = text_len_ + 1;
        // The payload section sizes are fully determined by m; reject any blob
        // whose declared counts don't match (a mutated size would otherwise
        // drive align_view / from_view to read past the mapping).
        const uint64_t exp_qnw = (m + 31) / 32;   // 2-bit symbols, 32 per word
        const uint64_t exp_snw = (m + 63) / 64;   // 1-bit rows, 64 per word
        for (uint32_t l = 0; l < qlevels_; ++l) {
            if (qnw[l] != exp_qnw) {
                return false;
            }
        }
        if (sampled_nw != exp_snw || n_samples > m || n_docs == 0 ||
            n_docs > m + 1) {
            return false;
        }
        auto align_view = [&](uint64_t nbytes) -> const uint64_t* {
            size_t off = static_cast<size_t>(p - reinterpret_cast<const char*>(
                                                     base));
            size_t pad = (8 - (off & 7u)) & 7u;
            if (p + pad + nbytes > end) {
                throw std::runtime_error("fmindex: truncated payload");
            }
            p += pad;
            const uint64_t* view = reinterpret_cast<const uint64_t*>(p);
            p += nbytes;
            return view;
        };

        std::vector<QuadVector> qvs;
        qvs.reserve(qlevels_);
        for (uint32_t l = 0; l < qlevels_; ++l) {
            const uint64_t* w = align_view(qnw[l] * sizeof(uint64_t));
            qvs.push_back(QuadVector::from_view(m, w, qnw[l]));
        }
        // The group-start offsets are fully determined by the quad vectors: the
        // four per-level digit counts always sum to m, so the derived offsets are
        // in [0, m] and keep every map_zero/map2/rank descent in bounds. Reject a
        // blob whose stored `starts` disagree — otherwise a corrupt offset drives
        // the wavelet descent off the QuadVector directory (heap OOB) right here
        // in map_zero, before any query.
        for (uint32_t l = 0; l < qlevels_; ++l) {
            size_t c0 = qvs[l].rank(0, m), c1 = qvs[l].rank(1, m),
                   c2 = qvs[l].rank(2, m);
            std::array<size_t, 4> derived = {0, c0, c0 + c1, c0 + c1 + c2};
            if (starts[l] != derived) {
                return false;
            }
        }
        wm_ = WaveletMatrix4::from_parts(m, qlevels_, std::move(qvs),
                                         std::move(starts));
        first_.resize(sigma_);
        for (uint32_t c = 0; c < sigma_; ++c) {
            first_[c] = wm_.map_zero(c);
        }
        const uint64_t* sw = align_view(sampled_nw * sizeof(uint64_t));
        sampled_bv_ = BitVector::from_view(m, sw, sampled_nw);
        // The set-bit count must equal the number of sampled values; otherwise
        // sa_sample_vals_[sampled_bv_.rank1(row)] could index past the array.
        if (sampled_bv_.count_ones() != n_samples) {
            return false;
        }
        // small arrays copied (cheap, keeps them owned/aligned-independent)
        const size_t pw = wide_storage_ ? 8 : 4;
        const uint64_t* svp = align_view(n_samples * pw);
        sa_sample_vals_.resize(n_samples);
        if (pw == 8) {
            std::memcpy(sa_sample_vals_.data(), svp,
                        n_samples * sizeof(uint64_t));
        } else {
            const uint32_t* s32 = reinterpret_cast<const uint32_t*>(svp);
            for (uint64_t i = 0; i < n_samples; ++i) {
                sa_sample_vals_[i] = s32[i];
            }
        }
        // Sampled values are text positions in [0, text_len_]; a larger value
        // would drive buildDerived's isa_sample_[val/rate] write out of bounds.
        for (uint64_t v : sa_sample_vals_) {
            if (v > text_len_) {
                return false;
            }
        }
        const uint64_t* dsp = align_view(n_docs * sizeof(uint64_t));
        doc_start_.resize(n_docs);
        std::memcpy(doc_start_.data(), dsp, n_docs * sizeof(uint64_t));
        // doc_start_ must be a valid boundary list: start at 0, be strictly
        // increasing (each document adds at least its separator), and end at
        // text_len_. Otherwise docOf's upper_bound could underflow to a bogus
        // document id and read c_/doc_start_ out of bounds during a query.
        if (doc_start_.front() != 0 || doc_start_.back() != text_len_) {
            return false;
        }
        for (size_t i = 1; i < doc_start_.size(); ++i) {
            if (doc_start_[i] <= doc_start_[i - 1]) {
                return false;
            }
        }
        buildDerived();  // id_to_byte_, isa_sample_ (in-RAM only)
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

FMIndex
FMIndex::LoadView(const uint8_t* base, size_t size) {
    FMIndex fm;
    if (!fm.parseView(base, size)) {
        return {};
    }
    return fm;
}

FMIndex
FMIndex::Deserialize(const std::string& blob) {
    FMIndex fm;
    fm.owned_blob_.assign(blob.begin(), blob.end());
    if (!fm.parseView(fm.owned_blob_.data(), fm.owned_blob_.size())) {
        return {};
    }
    return fm;
}

}  // namespace milvus::index::fmindex
