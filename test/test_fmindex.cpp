// Licensed to the LF AI & Data foundation under Apache-2.0.
#include <algorithm>
#include <atomic>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <cstring>

#include "index/fmindex/BitVector.h"
#include "index/fmindex/FMIndex.h"
#include "index/fmindex/MmapFmIndex.h"
#include "index/fmindex/QuadVector.h"
#include "index/fmindex/SuffixArray.h"
#include "index/fmindex/WaveletMatrix.h"
#include "index/fmindex/WaveletMatrix4.h"
#include "simple_test.h"

using namespace milvus::index::fmindex;

static const uint8_t*
bytes(const std::string& s) {
    return reinterpret_cast<const uint8_t*>(s.data());
}

// Build a single-document index (the common case for the byte-level tests).
static void
build1(FMIndex& fm, const std::string& text, uint32_t rate = 32, bool ci = false,
       bool fw = false) {
    fm.Build(std::vector<std::string_view>{std::string_view(text)}, rate, ci, fw);
}

// Build a multi-document index from a list of documents.
static void
buildn(FMIndex& fm, const std::vector<std::string>& docs, uint32_t rate = 32,
       bool ci = false, bool fw = false) {
    std::vector<std::string_view> v;
    v.reserve(docs.size());
    for (const auto& d : docs) {
        v.emplace_back(d);
    }
    fm.Build(v, rate, ci, fw);
}

// Sorted doc-local offsets of every occurrence. For a single-document index this
// equals the plain text positions (every doc id is 0).
static std::vector<uint64_t>
offsets(const FMIndex& fm, const std::string& p) {
    std::vector<uint64_t> v;
    for (auto [doc, off] : fm.LocateDocs(bytes(p), p.size())) {
        (void)doc;
        v.push_back(off);
    }
    std::sort(v.begin(), v.end());
    return v;
}

// ---------- oracles ----------
static std::vector<uint32_t>
brute_sa(const std::vector<uint32_t>& s) {
    std::vector<uint32_t> sa(s.size());
    for (uint32_t i = 0; i < s.size(); ++i) {
        sa[i] = i;
    }
    std::sort(sa.begin(), sa.end(), [&](uint32_t a, uint32_t b) {
        while (a < s.size() && b < s.size()) {
            if (s[a] != s[b]) {
                return s[a] < s[b];
            }
            ++a;
            ++b;
        }
        return a >= s.size() && b < s.size();  // shorter suffix is smaller
    });
    return sa;
}

static size_t
brute_count(const std::string& text, const std::string& pat) {
    if (pat.empty()) {
        return text.size() + 1;
    }
    size_t c = 0, pos = 0;
    while ((pos = text.find(pat, pos)) != std::string::npos) {
        ++c;
        ++pos;
    }
    return c;
}

static std::vector<uint64_t>
brute_positions(const std::string& text, const std::string& pat) {
    std::vector<uint64_t> out;
    if (pat.empty()) {
        return out;
    }
    size_t pos = 0;
    while ((pos = text.find(pat, pos)) != std::string::npos) {
        out.push_back(pos);
        ++pos;
    }
    std::sort(out.begin(), out.end());
    return out;
}

// ---------- BitVector ----------
TEST(BitVector, RankMatchesBruteForce) {
    std::vector<bool> ref = {1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1};
    BitVector bv(ref.size());
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i]) {
            bv.set(i);
        }
    }
    bv.build_rank();
    size_t acc = 0;
    for (size_t i = 0; i <= ref.size(); ++i) {
        CHECK_EQ(bv.rank1(i), acc);
        CHECK_EQ(bv.rank0(i), i - acc);
        if (i < ref.size() && ref[i]) {
            ++acc;
        }
    }
    CHECK(bv.get(0));
    CHECK(!bv.get(1));
}

TEST(BitVector, CrossesWordBoundary) {
    BitVector bv(200);
    for (size_t i = 0; i < 200; i += 3) {
        bv.set(i);
    }
    bv.build_rank();
    size_t acc = 0;
    for (size_t i = 0; i <= 200; ++i) {
        CHECK_EQ(bv.rank1(i), acc);
        if (i < 200 && (i % 3 == 0)) {
            ++acc;
        }
    }
}

TEST(BitVector, CrossesBlockAndSuperblockBoundaries) {
    // Exercise sizes that land exactly on 64/256/4096-bit boundaries, where
    // rank1(i) for i == n (n % 64 == 0) must not read past the directory.
    for (size_t n : {size_t(256), size_t(512), size_t(4096), size_t(8192),
                     size_t(10000)}) {
        BitVector bv(n);
        std::vector<char> ref(n);
        for (size_t i = 0; i < n; ++i) {
            ref[i] = static_cast<char>((i * 2654435761u >> 13) & 1);
            if (ref[i]) {
                bv.set(i);
            }
        }
        bv.build_rank();
        size_t acc = 0;
        for (size_t i = 0; i <= n; ++i) {
            CHECK_EQ(bv.rank1(i), acc);
            if (i < n && ref[i]) {
                ++acc;
            }
        }
    }
}

// ---------- WaveletMatrix ----------
TEST(WaveletMatrix, AccessAndRankMatchBruteForce) {
    std::vector<uint32_t> seq = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 0, 256};
    WaveletMatrix wm(seq, 9);  // 2^9 = 512 > 256
    for (size_t i = 0; i < seq.size(); ++i) {
        CHECK_EQ(wm.access(i), seq[i]);
    }
    for (uint32_t c : {0u, 1u, 3u, 5u, 9u, 256u, 7u}) {
        size_t acc = 0;
        for (size_t i = 0; i <= seq.size(); ++i) {
            CHECK_EQ(wm.rank(c, i), acc);
            // rank2 endpoints must agree with rank
            auto rr = wm.rank2(c, 0, i);
            CHECK_EQ(rr.second, acc);
            if (i < seq.size() && seq[i] == c) {
                ++acc;
            }
        }
    }
}

TEST(WaveletMatrix, SingleSymbolAlphabet) {
    std::vector<uint32_t> seq = {0, 0, 0, 0};
    WaveletMatrix wm(seq, 1);
    CHECK_EQ(wm.access(2), 0u);
    CHECK_EQ(wm.rank(0, 4), 4u);
}

// ---------- SuffixArray ----------
TEST(SuffixArray, MatchesBruteForceSmall) {
    std::vector<uint32_t> s = {2, 1, 3, 1, 3, 1, 0};
    CHECK_EQ(build_suffix_array(s), brute_sa(s));
}

TEST(SuffixArray, MatchesBruteForceRandom) {
    std::mt19937 rng(123);
    for (int trial = 0; trial < 50; ++trial) {
        size_t n = 1 + rng() % 60;
        std::vector<uint32_t> s(n);
        for (auto& x : s) {
            x = 1 + rng() % 4;
        }
        s.push_back(0);
        CHECK_EQ(build_suffix_array(s), brute_sa(s));
    }
}

// ---------- FMIndex ----------
TEST(FMIndex, BuildInvariants) {
    std::string text = "banana";
    FMIndex fm;
    build1(fm, text, 1);
    // internal text is "banana" + one '\0' separator, so bwt_size == len + 2.
    CHECK_EQ(fm.bwt_size(), text.size() + 2);
    CHECK_EQ(fm.c_at(0), 0u);
    for (uint32_t c = 1; c < fm.alphabet(); ++c) {
        CHECK(fm.c_at(c - 1) <= fm.c_at(c));
    }
}

TEST(FMIndex, CountMatchesBruteForce) {
    std::string text = "mississippi";
    FMIndex fm;
    build1(fm, text, 1);
    for (std::string pat :
         {"i", "s", "ss", "issi", "ippi", "mississippi", "x", "ppi"}) {
        CHECK_EQ(fm.Count(bytes(pat), pat.size()), brute_count(text, pat));
    }
}

TEST(FMIndex, NoMatchReturnsZero) {
    std::string text = "abcabc";
    FMIndex fm;
    build1(fm, text, 1);
    std::string pat = "abcd";
    CHECK_EQ(fm.Count(bytes(pat), pat.size()), 0u);
}

TEST(FMIndex, LocateMatchesBruteForceAcrossSampleRates) {
    std::string text = "abracadabra_abracadabra";
    for (uint32_t rate : {1u, 2u, 3u, 8u, 32u}) {
        FMIndex fm;
        build1(fm, text, rate);
        for (std::string pat :
             {"a", "abra", "bra", "dab", "xyz", "abracadabra"}) {
            CHECK_EQ(offsets(fm, pat), brute_positions(text, pat));
        }
    }
}

TEST(FMIndex, DocMappingLocatesPkAndOffset) {
    FMIndex fm;
    buildn(fm, {"cat", "banana", "cataract"}, 2);

    std::string pat = "cat";
    auto hits = fm.LocateDocs(bytes(pat), pat.size());
    std::vector<std::pair<uint64_t, uint64_t>> expect = {{0, 0}, {2, 0}};
    CHECK_EQ(hits, expect);

    std::string p2 = "an";
    auto h2 = fm.LocateDocs(bytes(p2), p2.size());
    std::vector<std::pair<uint64_t, uint64_t>> e2 = {{1, 1}, {1, 3}};
    CHECK_EQ(h2, e2);
}

TEST(FMIndex, SerializeRoundTrip) {
    std::vector<std::string> docs = {"the quick brown fox", "the lazy dog",
                                     "quick quick"};
    FMIndex fm;
    buildn(fm, docs, 4);

    std::string blob = fm.Serialize();
    FMIndex fm2 = FMIndex::Deserialize(blob);

    for (std::string pat : {"quick", "the", "dog", "zzz", "o"}) {
        CHECK_EQ(fm.LocateDocs(bytes(pat), pat.size()),
                 fm2.LocateDocs(bytes(pat), pat.size()));
        CHECK_EQ(fm.Count(bytes(pat), pat.size()),
                 fm2.Count(bytes(pat), pat.size()));
    }
}

TEST(FMIndex, RandomizedFuzzVsBruteForce) {
    std::mt19937 rng(2026);
    for (int trial = 0; trial < 200; ++trial) {
        size_t n = 1 + rng() % 300;
        std::string text(n, 'a');
        for (auto& ch : text) {
            ch = 'a' + (rng() % 4);
        }
        uint32_t rate = 1 + rng() % 16;
        FMIndex fm;
        build1(fm, text, rate);
        for (int q = 0; q < 20; ++q) {
            size_t plen = 1 + rng() % 6;
            std::string pat(plen, 'a');
            for (auto& ch : pat) {
                ch = 'a' + (rng() % 5);  // 'e' never appears in text
            }
            CHECK_EQ(fm.Count(bytes(pat), plen), brute_count(text, pat));
            CHECK_EQ(offsets(fm, pat), brute_positions(text, pat));
        }
    }
}

TEST(FMIndex, UTF8AndBoundaryBytes) {
    // Multibyte UTF-8 plus boundary bytes 0xff / 0x01 / 0x00. v2 indexes '\0' as
    // ordinary content.
    std::string text = "caf\xc3\xa9 \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e ";
    text.push_back('\xff');
    text.push_back('\x01');
    text.push_back('\0');
    text += "test";
    FMIndex fm;
    build1(fm, text, 3);
    std::string pat = "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e";  // 日本語
    CHECK_EQ(fm.Count(bytes(pat), pat.size()), 1u);
    std::string pat2 = "test";
    CHECK_EQ(fm.Count(bytes(pat2), pat2.size()), 1u);
}

TEST(FMIndex, CountBatchMatchesPerQuery) {
    std::mt19937 rng(99);
    for (int trial = 0; trial < 30; ++trial) {
        size_t n = 10 + rng() % 500;
        std::string text(n, 'a');
        for (auto& ch : text) {
            ch = 'a' + (rng() % 4);
        }
        FMIndex fm;
        build1(fm, text, 8);
        std::vector<std::string> pats;
        for (int q = 0; q < 40; ++q) {
            size_t plen = rng() % 6;  // includes empty pattern
            std::string p(plen, 'a');
            for (auto& ch : p) {
                ch = 'a' + (rng() % 5);  // 'e' absent
            }
            pats.push_back(p);
        }
        std::vector<std::pair<const uint8_t*, size_t>> batch;
        for (auto& p : pats) {
            batch.emplace_back(bytes(p), p.size());
        }
        auto got = fm.CountBatch(batch);
        CHECK_EQ(got.size(), pats.size());
        for (size_t i = 0; i < pats.size(); ++i) {
            CHECK_EQ(got[i], fm.Count(bytes(pats[i]), pats[i].size()));
        }
    }
}

TEST(FMIndex, LocateDocsBatchMatchesPerQuery) {
    // LocateDocsBatch(patterns)[i] must equal LocateDocs(patterns[i]). Multi-doc
    // corpora exercise the per-occurrence LF-walk and the doc mapping.
    std::mt19937 rng(7);
    const char alpha[] = {'a', 'b', 'c', 'd'};
    for (int trial = 0; trial < 30; ++trial) {
        size_t ndocs = 1 + rng() % 5;
        std::vector<std::string> docs;
        for (size_t d = 0; d < ndocs; ++d) {
            std::string doc(rng() % 60, 'a');
            for (auto& ch : doc) {
                ch = alpha[rng() % 4];
            }
            docs.push_back(doc);
        }
        FMIndex fm;
        buildn(fm, docs, 1 + rng() % 8);
        std::vector<std::string> pats;
        for (int q = 0; q < 30; ++q) {
            std::string p(rng() % 5, 'a');  // includes empty pattern
            for (auto& ch : p) {
                ch = alpha[rng() % 4];
            }
            pats.push_back(p);
        }
        std::vector<std::pair<const uint8_t*, size_t>> batch;
        for (auto& p : pats) {
            batch.emplace_back(bytes(p), p.size());
        }
        auto got = fm.LocateDocsBatch(batch);
        CHECK_EQ(got.size(), pats.size());
        for (size_t i = 0; i < pats.size(); ++i) {
            CHECK_EQ(got[i], fm.LocateDocs(bytes(pats[i]), pats[i].size()));
        }
    }
}

TEST(FMIndex, DnaReadAlignment) {
    std::string ref = "ACGTTGCACGATTACAGGATCCACGTACGTTGCA";
    FMIndex fm;
    build1(fm, ref, 4);
    std::string read = "ACGT";
    auto pos = offsets(fm, read);
    CHECK_EQ(pos, brute_positions(ref, read));
    CHECK_EQ(pos.size(), 3u);
    std::string read2 = "TTGCA";
    CHECK_EQ(offsets(fm, read2), brute_positions(ref, read2));
}

// ---------- QuadVector direct ----------
TEST(QuadVector, RankMatchesBruteForceAllValues) {
    std::mt19937 rng(7);
    for (size_t n : {size_t(1), size_t(31), size_t(32), size_t(33), size_t(64),
                     size_t(2048), size_t(2049), size_t(5000)}) {
        std::vector<uint8_t> syms(n);
        for (auto& s : syms) {
            s = rng() % 4;
        }
        QuadVector qv(syms);
        for (uint8_t v = 0; v < 4; ++v) {
            size_t acc = 0;
            for (size_t i = 0; i <= n; ++i) {
                CHECK_EQ(qv.rank(v, i), acc);
                if (i < n && syms[i] == v) {
                    ++acc;
                }
            }
        }
        for (size_t i = 0; i < n; ++i) {
            CHECK_EQ(qv.at(i), syms[i]);
        }
    }
}

TEST(QuadVector, AllSameValueNoPhantomZeros) {
    // trailing lanes past n_ are zero; must not be counted as value 0
    for (uint8_t v : {uint8_t(0), uint8_t(3)}) {
        std::vector<uint8_t> syms(50, v);
        QuadVector qv(syms);
        CHECK_EQ(qv.rank(v, 50), 50u);
        CHECK_EQ(qv.rank(0, 50), v == 0 ? 50u : 0u);
    }
}

// ---------- WaveletMatrix4 direct ----------
TEST(WaveletMatrix4, AccessRankMap2Consistent) {
    std::mt19937 rng(11);
    // alphabet sizes around power-of-4 and power-of-2 boundaries
    for (uint32_t sigma : {2u, 3u, 4u, 5u, 16u, 17u, 27u, 256u, 257u}) {
        uint32_t bits = 1;
        while ((1u << bits) < sigma) {
            ++bits;
        }
        uint32_t qlevels = (bits + 1) / 2;
        size_t n = 500;
        std::vector<uint16_t> seq(n);
        for (auto& x : seq) {
            x = static_cast<uint16_t>(rng() % sigma);
        }
        WaveletMatrix4 wm(seq, qlevels);
        for (size_t i = 0; i < n; ++i) {
            CHECK_EQ(wm.access(i), seq[i]);
        }
        for (uint32_t c = 0; c < sigma; ++c) {
            size_t acc = 0;
            for (size_t i = 0; i <= n; ++i) {
                CHECK_EQ(wm.rank(c, i), acc);
                if (i < n && seq[i] == c) {
                    ++acc;
                }
            }
        }
    }
}

// ---------- FMIndex edge cases ----------
TEST(FMIndex, EmptyCorpus) {
    FMIndex fm;
    fm.Build(std::vector<std::string_view>{}, 4);  // no documents at all
    CHECK_EQ(fm.Count(bytes(std::string("a")), 1), 0u);
    CHECK_EQ(fm.Count(bytes(std::string()), 0), 1u);  // empty matches 1 position
    CHECK_EQ(fm.LocateDocs(bytes(std::string("a")), 1).size(), 0u);
    std::string blob = fm.Serialize();
    FMIndex fm2 = FMIndex::Deserialize(blob);
    CHECK_EQ(fm2.Count(bytes(std::string("a")), 1), 0u);
}

TEST(FMIndex, SingleCharAndAllSame) {
    for (std::string text : {std::string("a"), std::string("aaaaaaaa")}) {
        FMIndex fm;
        build1(fm, text, 2);
        CHECK_EQ(fm.Count(bytes(std::string("a")), 1), text.size());
        CHECK_EQ(fm.Count(bytes(std::string("aa")), 2),
                 text.size() >= 2 ? text.size() - 1 : 0u);
        CHECK_EQ(fm.Count(bytes(std::string("b")), 1), 0u);
        CHECK_EQ(fm.LocateDocs(bytes(std::string("a")), 1).size(), text.size());
    }
}

TEST(FMIndex, FullByteAlphabetAndBinary) {
    // Every byte value 0..255 present as content (v2: '\0' is a normal content
    // byte; the document separator is an out-of-byte-alphabet symbol), arbitrary
    // binary content. alphabet == 256 content bytes + separator + sentinel == 258.
    std::mt19937 rng(2024);
    std::string text;
    for (int i = 0; i < 256; ++i) {
        text.push_back(static_cast<char>(i));
    }
    for (int i = 0; i < 4000; ++i) {
        text.push_back(static_cast<char>(rng() % 256));  // any byte incl. 0
    }
    FMIndex fm;
    build1(fm, text, 8);
    CHECK_EQ(fm.alphabet(), 258u);  // 256 bytes + separator + sentinel
    // check a handful of substrings against brute force
    std::mt19937 rng2(5);
    for (int q = 0; q < 50; ++q) {
        size_t L = 1 + rng2() % 5;
        size_t pos = rng2() % (text.size() - L);
        std::string pat = text.substr(pos, L);
        CHECK_EQ(fm.Count(bytes(pat), pat.size()), brute_count(text, pat));
        CHECK_EQ(offsets(fm, pat), brute_positions(text, pat));
    }
}

TEST(FMIndex, PatternLongerThanText) {
    std::string text = "abc";
    FMIndex fm;
    build1(fm, text, 1);
    CHECK_EQ(fm.Count(bytes(std::string("abcd")), 4), 0u);
    CHECK_EQ(fm.Count(bytes(std::string("abcabc")), 6), 0u);
}

TEST(FMIndex, CountBatchEdgeCases) {
    std::string text = "abracadabra";
    FMIndex fm;
    build1(fm, text, 2);
    // empty list
    CHECK_EQ(fm.CountBatch({}).size(), 0u);
    // mix: normal, empty pattern, absent char, single
    std::vector<std::string> pats = {"abra", "", "xyz", "a", "abracadabra"};
    std::vector<std::pair<const uint8_t*, size_t>> batch;
    for (auto& p : pats) {
        batch.emplace_back(bytes(p), p.size());
    }
    auto got = fm.CountBatch(batch);
    CHECK_EQ(got.size(), pats.size());
    for (size_t i = 0; i < pats.size(); ++i) {
        CHECK_EQ(got[i], fm.Count(bytes(pats[i]), pats[i].size()));
    }
}

// ---------- empty structures (must not OOB the sentinel directories) ----------
TEST(Structures, EmptyDoNotCrash) {
    BitVector bv(0);
    bv.build_rank();
    CHECK_EQ(bv.rank1(0), 0u);
    CHECK_EQ(bv.count_ones(), 0u);

    QuadVector qv(std::vector<uint8_t>{});
    for (uint8_t v = 0; v < 4; ++v) {
        CHECK_EQ(qv.rank(v, 0), 0u);
    }

    WaveletMatrix wm(std::vector<uint32_t>{}, 3);
    CHECK_EQ(wm.rank(1, 0), 0u);
    WaveletMatrix4 wm4(std::vector<uint16_t>{}, 2);
    CHECK_EQ(wm4.rank(1, 0), 0u);
    CHECK_EQ(wm4.map_zero(1), 0u);
}

// ---------- QuadVector: symbol 0 across superblock boundary (padding lanes) --
TEST(QuadVector, AllSameAcrossSuperblockBoundaries) {
    for (uint8_t v : {uint8_t(0), uint8_t(1), uint8_t(3)}) {
        for (size_t n : {size_t(31), size_t(32), size_t(33), size_t(2047),
                         size_t(2048), size_t(2049), size_t(4096),
                         size_t(4097)}) {
            std::vector<uint8_t> syms(n, v);
            QuadVector qv(syms);
            CHECK_EQ(qv.rank(v, n), n);        // every symbol counted
            for (uint8_t w = 0; w < 4; ++w) {  // no phantom counts (esp. v==0)
                if (w != v) {
                    CHECK_EQ(qv.rank(w, n), 0u);
                }
            }
            // spot-check a mid position
            CHECK_EQ(qv.rank(v, n / 2), n / 2);
        }
    }
}

// ---------- WaveletMatrix4: qlevels ceil-log boundaries + invariants ----------
TEST(WaveletMatrix4, AlphabetBoundariesAndSymbolSum) {
    std::mt19937 rng(31);
    for (uint32_t sigma :
         {4u, 5u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 33u, 255u, 256u, 257u}) {
        uint32_t bits = 1;
        while ((1u << bits) < sigma) {
            ++bits;
        }
        uint32_t qlevels = (bits + 1) / 2;
        size_t n = 1025;
        std::vector<uint16_t> seq(n);
        for (auto& x : seq) {
            x = static_cast<uint16_t>(rng() % sigma);
        }
        WaveletMatrix4 wm(seq, qlevels);
        // symbol-sum invariant over a random range: sum_c rank(c,j)-rank(c,i)==j-i
        size_t i = rng() % n, j = i + rng() % (n - i + 1);
        size_t total = 0;
        for (uint32_t c = 0; c < sigma; ++c) {
            total += wm.rank(c, j) - wm.rank(c, i);
        }
        CHECK_EQ(total, j - i);
        // map2/map_zero consistency with rank, for a few symbols
        for (int t = 0; t < 8; ++t) {
            uint32_t c = rng() % sigma;
            size_t lo = rng() % n, hi = lo + rng() % (n - lo + 1);
            auto mp = wm.map2(c, lo, hi);
            size_t base = wm.map_zero(c);
            CHECK_EQ(mp.first - base, wm.rank(c, lo));
            CHECK_EQ(mp.second - base, wm.rank(c, hi));
        }
    }
}

// ---------- WaveletMatrix (binary) vs WaveletMatrix4 differential ----------
TEST(WaveletMatrix, BinaryVsQuadAgree) {
    std::mt19937 rng(77);
    uint32_t sigma = 200;
    uint32_t bits = 8;  // 2^8=256 > 200
    uint32_t qlevels = (bits + 1) / 2;
    size_t n = 3000;
    std::vector<uint32_t> seq(n);
    for (auto& x : seq) {
        x = rng() % sigma;
    }
    std::vector<uint16_t> seq16(seq.begin(), seq.end());  // WM4 takes uint16
    WaveletMatrix bwm(seq, bits);
    WaveletMatrix4 qwm(seq16, qlevels);
    for (size_t i = 0; i < n; ++i) {
        CHECK_EQ(bwm.access(i), qwm.access(i));
    }
    for (int t = 0; t < 200; ++t) {
        uint32_t c = rng() % sigma;
        size_t i = rng() % (n + 1);
        CHECK_EQ(bwm.rank(c, i), qwm.rank(c, i));
    }
}

// ---------- SuffixArray: permutation + textbook/edge corpora ----------
TEST(SuffixArray, PermutationAndTextbookCorpora) {
    auto remap = [](const std::string& s) {
        std::vector<uint32_t> t;
        for (unsigned char ch : s) {
            t.push_back(static_cast<uint32_t>(ch) + 1);
        }
        t.push_back(0);  // sentinel
        return t;
    };
    for (std::string s : {std::string("mississippi"), std::string("banana"),
                          std::string("abcabcabc"), std::string("aaaaaaaa"),
                          std::string("a")}) {
        auto t = remap(s);
        auto sa = build_suffix_array(t);
        CHECK_EQ(sa, brute_sa(t));
        auto sorted = sa;
        std::sort(sorted.begin(), sorted.end());
        for (uint32_t i = 0; i < sorted.size(); ++i) {
            CHECK_EQ(sorted[i], i);  // SA is a permutation of [0, n)
        }
    }
    CHECK_EQ(build_suffix_array({}).size(), 0u);
    CHECK_EQ(build_suffix_array({0}), std::vector<uint32_t>{0});
}

// ---------- robustness: bug-fix regressions ----------
TEST(FMIndex, LocateDocsSingleDoc) {
    // one document starting at 0 (no underflow computing the doc id)
    std::string text = "abcabc";
    FMIndex fm;
    build1(fm, text, 1);
    auto hits = fm.LocateDocs(bytes(std::string("abc")), 3);
    std::vector<std::pair<uint64_t, uint64_t>> expect = {{0, 0}, {0, 3}};
    CHECK_EQ(hits, expect);
}

TEST(FMIndex, DeserializeCorruptReturnsEmpty) {
    // wrong magic, truncated, and empty blobs must all fail gracefully
    CHECK_EQ(FMIndex::Deserialize(std::string()).Count(bytes(std::string("a")),
                                                       1),
             0u);
    CHECK_EQ(FMIndex::Deserialize(std::string("xx")).bwt_size(), 1u);
    std::string text = "hello world";
    FMIndex fm;
    build1(fm, text, 4);
    std::string blob = fm.Serialize();
    std::string truncated = blob.substr(0, blob.size() / 2);
    FMIndex bad = FMIndex::Deserialize(truncated);  // must not crash/OOB
    CHECK_EQ(bad.Count(bytes(std::string("hello")), 5), 0u);
    // full round-trip still works and re-serializes identically
    FMIndex ok = FMIndex::Deserialize(blob);
    CHECK_EQ(ok.Count(bytes(std::string("world")), 5), 1u);
    CHECK(ok.Serialize() == blob);
}

TEST(FMIndex, CountBatchAcrossTiles) {
    std::string text = "the quick brown fox jumps over the lazy dog the the";
    FMIndex fm;
    build1(fm, text, 4);
    for (size_t B : {size_t(1), size_t(32), size_t(33), size_t(64),
                     size_t(65), size_t(100)}) {
        std::vector<std::string> pats;
        std::mt19937 rng(B);
        for (size_t i = 0; i < B; ++i) {
            size_t L = rng() % 5;
            size_t pos = L && text.size() > L ? rng() % (text.size() - L) : 0;
            pats.push_back(text.substr(pos, L));
        }
        std::vector<std::pair<const uint8_t*, size_t>> batch;
        for (auto& p : pats) {
            batch.emplace_back(bytes(p), p.size());
        }
        auto got = fm.CountBatch(batch);
        CHECK_EQ(got.size(), B);
        for (size_t i = 0; i < B; ++i) {
            CHECK_EQ(got[i], fm.Count(bytes(pats[i]), pats[i].size()));
        }
    }
}

// ---------- mmap / zero-copy load ----------
TEST(FMIndex, LoadViewZeroCopyMatchesInRam) {
    std::vector<std::string> docs = {"the quick brown fox", " the lazy dog",
                                     " the the quick fox"};
    FMIndex fm;
    buildn(fm, docs, 4);
    std::string blob = fm.Serialize();

    // LoadView requires 8-byte-aligned bytes; a uint64 buffer guarantees it.
    std::vector<uint64_t> aligned((blob.size() + 7) / 8, 0);
    std::memcpy(aligned.data(), blob.data(), blob.size());
    FMIndex view = FMIndex::LoadView(
        reinterpret_cast<const uint8_t*>(aligned.data()), blob.size());

    for (std::string pat : {"the", "quick", "fox", "zzz", "the the"}) {
        CHECK_EQ(view.Count(bytes(pat), pat.size()),
                 fm.Count(bytes(pat), pat.size()));
        CHECK_EQ(view.LocateDocs(bytes(pat), pat.size()),
                 fm.LocateDocs(bytes(pat), pat.size()));
    }
    // views must survive a move (pointers into `aligned` stay valid)
    FMIndex moved = std::move(view);
    CHECK_EQ(moved.Count(bytes(std::string("quick")), 5),
             fm.Count(bytes(std::string("quick")), 5));
}

TEST(FMIndex, MmapFileRoundTrip) {
    std::string text =
        "mmap the quick brown fox jumps over the lazy dog the the";
    FMIndex fm;
    build1(fm, text, 4);
    std::string path = "/tmp/fmidx_mmap_test.fmix";
    CHECK(SaveToFile(fm, path));
    // streamed file bytes must be byte-identical to Serialize()
    {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        CHECK(f != nullptr);
        std::string on_disk;
        if (f) {
            char buf[4096];
            size_t r;
            while ((r = std::fread(buf, 1, sizeof(buf), f)) > 0) {
                on_disk.append(buf, r);
            }
            std::fclose(f);
        }
        CHECK(on_disk == fm.Serialize());
    }
    auto mapped = MappedFMIndex::Open(path);
    CHECK(mapped != nullptr);
    if (mapped) {
        for (std::string pat : {"the", "fox", "quick", "zzz", "the the"}) {
            CHECK_EQ(mapped->index().Count(bytes(pat), pat.size()),
                     fm.Count(bytes(pat), pat.size()));
            CHECK_EQ(mapped->index().LocateDocs(bytes(pat), pat.size()),
                     fm.LocateDocs(bytes(pat), pat.size()));
        }
    }
    std::remove(path.c_str());
}

TEST(FMIndex, PrefixDocsAnchored) {
    std::vector<std::string> docs = {"error: disk full", "warning: low mem",
                                     "error: timeout", "erratic value"};
    FMIndex fm;
    buildn(fm, docs, 4);

    // "error" begins docs 0 and 2 (not 3 "erratic", not 1).
    std::string p = "error";
    std::vector<uint64_t> expect = {0, 2};
    CHECK_EQ(fm.LocatePrefixDocs(bytes(p), p.size()), expect);
    CHECK_EQ(fm.CountPrefixDocs(bytes(p), p.size()), size_t(2));

    // "err" begins 0, 2, 3.
    std::string p2 = "err";
    std::vector<uint64_t> e2 = {0, 2, 3};
    CHECK_EQ(fm.LocatePrefixDocs(bytes(p2), p2.size()), e2);

    // substring that never sits at a doc start -> no prefix docs.
    std::string p3 = "disk";
    CHECK_EQ(fm.CountPrefixDocs(bytes(p3), p3.size()), size_t(0));
    CHECK(fm.Count(bytes(p3), p3.size()) == 1);  // but it does occur
}

TEST(FMIndex, SuffixDocsAnchored) {
    std::vector<std::string> docs = {"path/to/a.log", "path/to/b.txt", "c.log",
                                     "logistics"};
    FMIndex fm;
    buildn(fm, docs, 4);

    // ".log" ends docs 0 and 2 (not "logistics").
    std::string p = ".log";
    std::vector<uint64_t> expect = {0, 2};
    CHECK_EQ(fm.LocateSuffixDocs(bytes(p), p.size()), expect);
    CHECK_EQ(fm.CountSuffixDocs(bytes(p), p.size()), size_t(2));

    // "log" occurs in docs 0,2,3 but ends only 0 and 2.
    std::string p2 = "log";
    std::vector<uint64_t> e2 = {0, 2};
    CHECK_EQ(fm.LocateSuffixDocs(bytes(p2), p2.size()), e2);

    // whole last doc as suffix.
    std::string p3 = "logistics";
    std::vector<uint64_t> e3 = {3};
    CHECK_EQ(fm.LocateSuffixDocs(bytes(p3), p3.size()), e3);
}

// Randomized differential oracle for the anchored prefix/suffix doc queries and
// their counts. Small alphabet + short docs so that most patterns occur BOTH at
// document boundaries and mid-document — the case where a boundary-anchored
// result (Count/LocatePrefixDocs/SuffixDocs) legitimately differs from the raw
// occurrence set, which is exactly what the separator-rank fast paths must get
// right. Also mixes in patterns borrowed from real doc prefixes/suffixes (to
// force boundary hits) and patterns over a byte that never appears (empty).
TEST(FMIndex, PrefixSuffixDocsMatchBruteForce) {
    std::mt19937 rng(20260714);
    for (int trial = 0; trial < 200; ++trial) {
        size_t ndocs = 1 + rng() % 12;
        std::vector<std::string> docs;
        docs.reserve(ndocs);
        for (size_t d = 0; d < ndocs; ++d) {
            size_t len = rng() % 8;  // 0..7, deliberately includes empty docs
            std::string s(len, 'a');
            for (auto& ch : s) {
                ch = 'a' + (rng() % 3);  // {a,b,c}
            }
            docs.push_back(std::move(s));
        }
        uint32_t rate = 1 + rng() % 8;
        FMIndex fm;
        buildn(fm, docs, rate);

        for (int q = 0; q < 24; ++q) {
            std::string pat;
            if ((rng() % 3) == 0) {
                // Borrow a real prefix or suffix of a random doc to force hits.
                const std::string& src = docs[rng() % ndocs];
                if (!src.empty()) {
                    size_t l = 1 + rng() % src.size();
                    pat = (rng() & 1) ? src.substr(0, l)
                                      : src.substr(src.size() - l, l);
                }
            }
            if (pat.empty()) {
                size_t plen = 1 + rng() % 4;
                pat.assign(plen, 'a');
                for (auto& ch : pat) {
                    ch = 'a' + (rng() % 4);  // 'd' never appears in any doc
                }
            }
            std::vector<uint64_t> ep, es;  // brute-force prefix / suffix doc ids
            for (uint64_t d = 0; d < ndocs; ++d) {
                const std::string& s = docs[d];
                if (s.size() >= pat.size()) {
                    if (s.compare(0, pat.size(), pat) == 0) {
                        ep.push_back(d);
                    }
                    if (s.compare(s.size() - pat.size(), pat.size(), pat) == 0) {
                        es.push_back(d);
                    }
                }
            }
            CHECK_EQ(fm.LocatePrefixDocs(bytes(pat), pat.size()), ep);
            CHECK_EQ(fm.CountPrefixDocs(bytes(pat), pat.size()), ep.size());
            CHECK_EQ(fm.LocateSuffixDocs(bytes(pat), pat.size()), es);
            CHECK_EQ(fm.CountSuffixDocs(bytes(pat), pat.size()), es.size());
        }
    }
}

TEST(FMIndex, CaseInsensitiveMatch) {
    std::string text = "The Quick BROWN fox; the quick brown FOX.";
    FMIndex fm;
    build1(fm, text, 4, /*ci=*/true);
    CHECK(fm.case_insensitive());

    // all case variants of "quick" find both occurrences.
    for (std::string pat : {"quick", "QUICK", "QuIcK", "Quick"}) {
        CHECK_EQ(fm.Count(bytes(pat), pat.size()), size_t(2));
    }
    // "fox" (2 occurrences: "fox" and "FOX")
    CHECK_EQ(fm.Count(bytes("FOX"), 3), size_t(2));
    CHECK_EQ(fm.Count(bytes("fox"), 3), size_t(2));

    // Offsets still point into the ORIGINAL text.
    auto pos = offsets(fm, "the");
    std::vector<uint64_t> expect = {0, 21};  // "The" at 0, "the" at 21
    CHECK_EQ(pos, expect);

    // non-letters unaffected; digits/punctuation exact.
    CHECK_EQ(fm.Count(bytes(";"), 1), size_t(1));

    // case-sensitive index does NOT fold.
    FMIndex cs;
    build1(cs, text, 4);
    CHECK(!cs.case_insensitive());
    CHECK_EQ(cs.Count(bytes("QUICK"), 5), size_t(0));
    CHECK_EQ(cs.Count(bytes("quick"), 5), size_t(1));
}

TEST(FMIndex, CaseInsensitiveSurvivesRoundTrip) {
    std::string text = "Hello WORLD hello world";
    FMIndex fm;
    build1(fm, text, 4, /*ci=*/true);
    std::string blob = fm.Serialize();
    FMIndex fm2 = FMIndex::Deserialize(blob);
    CHECK(fm2.case_insensitive());
    for (std::string pat : {"hello", "HELLO", "World", "WORLD"}) {
        CHECK_EQ(fm2.Count(bytes(pat), pat.size()), size_t(2));
    }
}

TEST(FMIndex, ExtractMatchesOriginal) {
    std::mt19937 rng(99);
    for (int trial = 0; trial < 60; ++trial) {
        size_t n = 1 + rng() % 500;
        std::string text(n, 'a');
        for (auto& ch : text) ch = 'a' + (rng() % 6);  // small alphabet, repeats
        uint32_t rate = 1 + rng() % 20;
        FMIndex fm;
        build1(fm, text, rate);
        // random windows within the single document, including edges/over-long
        for (int q = 0; q < 30; ++q) {
            uint64_t pos = rng() % n;
            size_t len = rng() % (n + 5);
            std::string got = fm.Extract(0, pos, len);
            size_t want_len = std::min<size_t>(len, n - pos);
            CHECK_EQ(got, text.substr(pos, want_len));
        }
        // whole-document extraction
        CHECK_EQ(fm.Extract(0, 0, n), text);
        // offset past end, and out-of-range document
        CHECK(fm.Extract(0, n, 5).empty());
        CHECK(fm.Extract(1, 0, 5).empty());
    }
}

TEST(FMIndex, ExtractContextAroundMatchAfterLoad) {
    std::string text = "the quick brown fox jumps over the lazy dog";
    FMIndex fm;
    build1(fm, text, 4);
    std::string blob = fm.Serialize();
    FMIndex fm2 = FMIndex::Deserialize(blob);  // derived structs rebuilt on load

    for (std::string pat : {"quick", "fox", "the", "dog"}) {
        auto hits = fm2.LocateDocs(bytes(pat), pat.size());
        CHECK(!hits.empty());
        for (auto [doc, off] : hits) {
            // the extracted window at the match equals the pattern
            CHECK_EQ(fm2.Extract(doc, off, pat.size()), pat);
            // and a little context on each side matches the original text
            uint64_t start = off >= 3 ? off - 3 : 0;
            size_t clen = std::min<size_t>(pat.size() + 6, text.size() - start);
            CHECK_EQ(fm2.Extract(doc, start, clen), text.substr(start, clen));
        }
    }
}

TEST(FMIndex, ConcurrentQueriesMatchSingleThreaded) {
    // Build one immutable index, hammer it from many threads, and check every
    // thread's answers equal the single-threaded ground truth (verifies the
    // const/lock-free query claim: no data races, no shared mutable state).
    std::mt19937 rng(7);
    std::vector<std::string> docs;
    for (int d = 0; d < 400; ++d) {
        std::string doc;
        size_t dl = 5 + rng() % 40;
        for (size_t i = 0; i < dl; ++i) doc += char('a' + rng() % 8);
        docs.push_back(doc);
    }
    FMIndex fm;
    buildn(fm, docs, 8);
    const size_t nd = docs.size();

    std::vector<std::string> pats;
    for (int i = 0; i < 64; ++i) {
        size_t pl = 1 + rng() % 5;
        std::string p;
        for (size_t j = 0; j < pl; ++j) p += char('a' + rng() % 8);
        pats.push_back(p);
    }
    // ground truth (single-threaded)
    std::vector<size_t> truth_count(pats.size());
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> truth_loc(pats.size());
    std::vector<std::string> truth_ex(pats.size());
    for (size_t i = 0; i < pats.size(); ++i) {
        truth_count[i] = fm.Count(bytes(pats[i]), pats[i].size());
        truth_loc[i] = fm.LocateDocs(bytes(pats[i]), pats[i].size());
        truth_ex[i] = fm.Extract(i % nd, 0, 10);
    }

    const int kThreads = 8;
    std::atomic<int> mismatches{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < kThreads; ++t) {
        ts.emplace_back([&, t]() {
            std::mt19937 lr(1000 + t);
            for (int iter = 0; iter < 300; ++iter) {
                size_t i = lr() % pats.size();
                if (fm.Count(bytes(pats[i]), pats[i].size()) != truth_count[i])
                    mismatches++;
                if (fm.LocateDocs(bytes(pats[i]), pats[i].size()) != truth_loc[i])
                    mismatches++;
                if (fm.Extract(i % nd, 0, 10) != truth_ex[i])
                    mismatches++;
            }
        });
    }
    for (auto& th : ts) th.join();
    CHECK_EQ(mismatches.load(), 0);
}

TEST(FMIndex, NoCrossDocumentMatches) {
    // Documents are '\0'-separated internally, so a pattern that would straddle
    // a document seam does not occur AT ALL — not even raw Count sees it.
    FMIndex fm;
    buildn(fm, {"ab", "cd", "ef"}, 1);

    // "bc" would span the ab|cd seam: it simply does not exist in the index.
    CHECK_EQ(fm.Count(bytes("bc"), 2), size_t(0));
    CHECK(fm.LocateDocs(bytes("bc"), 2).empty());
    CHECK(fm.MatchingDocs(bytes("bc"), 2).empty());

    // "abc" would spill doc 0 into doc 1: also absent.
    CHECK_EQ(fm.Count(bytes("abc"), 3), size_t(0));
    CHECK(fm.LocatePrefixDocs(bytes("abc"), 3).empty());

    // genuine within-document matches still work.
    std::vector<std::pair<uint64_t, uint64_t>> cd = {{1, 0}};
    CHECK_EQ(fm.LocateDocs(bytes("cd"), 2), cd);
    std::vector<uint64_t> pfx = {0};
    CHECK_EQ(fm.LocatePrefixDocs(bytes("ab"), 2), pfx);
    std::vector<uint64_t> md = {2};
    CHECK_EQ(fm.MatchingDocs(bytes("ef"), 2), md);
}

TEST(FMIndex, DocScopedFuzzVsPerDocOracle) {
    // Count / LocateDocs / MatchingDocs over random documents must equal a naive
    // per-document oracle — a cross-seam occurrence never exists, so Count is
    // exactly the sum of in-document occurrences.
    std::mt19937 rng(20260709);
    for (int trial = 0; trial < 60; ++trial) {
        int nd = 1 + rng() % 12;
        std::vector<std::string> docs(nd);
        for (auto& d : docs) {
            size_t dl = rng() % 12;  // some empty documents too
            for (size_t i = 0; i < dl; ++i) d += char('a' + rng() % 3);
        }
        FMIndex fm;
        buildn(fm, docs, 1 + rng() % 8);
        for (int q = 0; q < 15; ++q) {
            size_t pl = 1 + rng() % 4;
            std::string p;
            for (size_t j = 0; j < pl; ++j) p += char('a' + rng() % 4);
            size_t want_count = 0;
            std::vector<std::pair<uint64_t, uint64_t>> want_ld;
            std::vector<uint64_t> want_md;
            for (size_t d = 0; d < docs.size(); ++d) {
                auto occ = brute_positions(docs[d], p);
                for (uint64_t pos : occ) {
                    want_count++;
                    want_ld.emplace_back(d, pos);
                }
                if (!occ.empty()) want_md.push_back(d);
            }
            std::sort(want_ld.begin(), want_ld.end());
            CHECK_EQ(fm.Count(bytes(p), p.size()), want_count);
            CHECK_EQ(fm.LocateDocs(bytes(p), p.size()), want_ld);
            CHECK_EQ(fm.MatchingDocs(bytes(p), p.size()), want_md);
        }
    }
}

TEST(FMIndex, LongestMatchStaysInDocument) {
    // "abc" | "XYZ": the query "abcXYZ" would match length 6 across the seam in a
    // naive concatenation, but each document only offers length 3.
    FMIndex fm;
    buildn(fm, {"abc", "XYZ"}, 1);
    std::string q = "abcXYZ";
    auto r = fm.LongestMatch(bytes(q), q.size());
    CHECK_EQ(r.length, size_t(3));  // never the 6-long cross-seam span
    CHECK_EQ(r.count, size_t(1));
}

TEST(FMIndex, ExtractStaysInDocument) {
    std::vector<std::string> docs = {"hello", "world!!", "x"};
    FMIndex fm;
    buildn(fm, docs, 2);
    for (size_t d = 0; d < docs.size(); ++d) {
        CHECK_EQ(fm.Extract(d, 0, docs[d].size()), docs[d]);       // whole doc
        CHECK_EQ(fm.Extract(d, 0, docs[d].size() + 10), docs[d]);  // clamped
        if (!docs[d].empty())
            CHECK_EQ(fm.Extract(d, 1, 100), docs[d].substr(1));    // mid to end
    }
    CHECK(fm.Extract(docs.size(), 0, 5).empty());     // out-of-range document
    CHECK(fm.Extract(0, docs[0].size(), 5).empty());  // offset past end
}

TEST(FMIndex, NextTokenCountsExcludesSeparator) {
    // "ab" ends doc 0 (follower is the separator, which must NOT be reported) and
    // in doc 1 "ab" is followed by 'c'.
    FMIndex fm;
    buildn(fm, {"ab", "abc"}, 1);
    auto got = fm.NextTokenCounts(bytes("ab"), 2);
    std::vector<std::pair<uint8_t, size_t>> want = {{'c', 1}};
    CHECK_EQ(got, want);
    for (auto& [b, c] : got) CHECK(b != 0);  // separator never appears
}

TEST(FMIndex, DeserializeRejectsCorruptByteMap) {
    std::string text = "banana";
    FMIndex fm;
    build1(fm, text, 4);
    std::string blob = fm.Serialize();

    // Header layout: magic,ver,rate,sigma,qlevels,flags (6*uint32=24) then
    // text_len (uint64) => byte_to_id_ starts at offset 32, entry b at 32+4b.
    // Point byte 'z' (normally -1, absent) at a bogus id well past sigma_.
    int32_t bogus = 100;
    size_t off = 32 + size_t('z') * 4;
    CHECK(off + 4 <= blob.size());
    std::memcpy(&blob[off], &bogus, sizeof(bogus));

    FMIndex bad = FMIndex::Deserialize(blob);
    CHECK(!bad.valid());  // rejected, not a live index
    // Querying the (empty) index must not read out of bounds — just returns 0.
    CHECK_EQ(bad.Count(bytes("z"), 1), size_t(0));
    CHECK_EQ(bad.Count(bytes("banana"), 6), size_t(0));
}

TEST(FMIndex, DeserializeRejectsCorruptDocStart) {
    FMIndex fm;
    buildn(fm, {"alpha", "beta", "gamma"}, 4);
    std::string blob = fm.Serialize();
    // doc_start_ is the final section; its last uint64 is the end boundary
    // (== internal text_len_). Corrupting it must be rejected, not read OOB via
    // a bogus document id at query time.
    CHECK(blob.size() >= 8);
    uint64_t bogus = 0;  // != text_len_, and breaks the strictly-increasing rule
    std::memcpy(&blob[blob.size() - 8], &bogus, sizeof(bogus));
    FMIndex bad = FMIndex::Deserialize(blob);
    CHECK(!bad.valid());
    CHECK_EQ(bad.Count(bytes("alpha"), 5), size_t(0));
}

TEST(FMIndex, MappedOpenFailsOnCorruptFile) {
    std::string text = "the quick brown fox";
    FMIndex fm;
    build1(fm, text, 4);
    std::string path = "/tmp/fmix_corrupt_test.fmix";
    CHECK(SaveToFile(fm, path));

    // Sanity: a good file opens.
    auto ok = MappedFMIndex::Open(path);
    CHECK(ok != nullptr);
    ok.reset();

    // Corrupt the magic; Open must now fail (nullptr), not hand back a dud.
    {
        FILE* f = std::fopen(path.c_str(), "r+b");
        CHECK(f != nullptr);
        if (f) {
            const char junk[4] = {'X', 'X', 'X', 'X'};
            std::fwrite(junk, 1, 4, f);
            std::fclose(f);
        }
    }
    auto bad = MappedFMIndex::Open(path);
    CHECK(bad == nullptr);
    std::remove(path.c_str());
}

TEST(SuffixArray, Bytes32And64Agree) {
    std::mt19937 rng(2027);
    for (int trial = 0; trial < 40; ++trial) {
        size_t n = 1 + rng() % 400;
        std::vector<uint8_t> text(n);
        for (auto& c : text) c = static_cast<uint8_t>(rng() % 5);
        auto sa32 = build_suffix_array_bytes(text.data(), n);
        auto sa64 = build_suffix_array_bytes64(text.data(), n);
        CHECK_EQ(sa32.size(), sa64.size());
        for (size_t i = 0; i < sa32.size(); ++i) {
            CHECK(static_cast<int64_t>(sa32[i]) == sa64[i]);
        }
    }
}

TEST(FMIndex, WideAndNarrowBuildsAgree) {
    // force_wide builds the 64-bit SA path (and 8-byte position storage) on
    // small inputs; it must produce identical query results to the 32-bit path.
    std::mt19937 rng(31);
    std::vector<std::string> docs;
    for (int d = 0; d < 200; ++d) {
        std::string doc;
        size_t dl = 3 + rng() % 30;
        for (size_t i = 0; i < dl; ++i) doc += char('a' + rng() % 6);
        docs.push_back(doc);
    }
    FMIndex narrow, wide;
    buildn(narrow, docs, 8, /*ci=*/false, /*fw=*/false);
    buildn(wide, docs, 8, /*ci=*/false, /*fw=*/true);
    CHECK(!narrow.wide());
    CHECK(wide.wide());

    for (int q = 0; q < 200; ++q) {
        size_t pl = 1 + rng() % 5;
        std::string p;
        for (size_t j = 0; j < pl; ++j) p += char('a' + rng() % 6);
        CHECK_EQ(narrow.Count(bytes(p), p.size()),
                 wide.Count(bytes(p), p.size()));
        CHECK_EQ(narrow.LocateDocs(bytes(p), p.size()),
                 wide.LocateDocs(bytes(p), p.size()));
    }
    // Extract agrees too.
    CHECK_EQ(narrow.Extract(0, 0, 50), wide.Extract(0, 0, 50));
}

TEST(FMIndex, WideStorageSurvivesRoundTrip) {
    // Exercises the 8-byte sampled-SA serialization path on small data.
    std::string text = "the quick brown fox jumps over the lazy dog again";
    FMIndex fm;
    build1(fm, text, 4, /*ci=*/false, /*fw=*/true);
    CHECK(fm.wide());
    std::string blob = fm.Serialize();
    FMIndex fm2 = FMIndex::Deserialize(blob);
    CHECK(fm2.valid());
    CHECK(fm2.wide());  // flag round-tripped, read back 8-byte samples
    for (std::string p : {"the", "quick", "dog", "zzz", "o", "again"}) {
        CHECK_EQ(fm.Count(bytes(p), p.size()), fm2.Count(bytes(p), p.size()));
        CHECK_EQ(fm.LocateDocs(bytes(p), p.size()),
                 fm2.LocateDocs(bytes(p), p.size()));
    }
    // Byte-identical re-serialization (deterministic wide layout).
    CHECK(fm2.Serialize() == blob);
}

TEST(FMIndex, MegabyteScaleVsNaiveOracle) {
    // Correctness at a scale the tiny fuzz tests don't reach (~1 MB), against an
    // independent naive std::string::find oracle, exercising BOTH the 32-bit and
    // 64-bit (force_wide) build paths on the same single-document data.
    std::mt19937 rng(4242);
    const size_t N = 1u << 20;  // 1 MiB
    std::string text(N, 'a');
    for (auto& c : text) c = char('a' + rng() % 6);  // small alphabet: real hits
    // inject a few known needles so long patterns match too
    const char* needles[] = {"zebra", "qwerty", "needle123", "FoxJumps"};
    for (const char* nd : needles)
        for (int k = 0; k < 20; ++k)
            text.replace(rng() % (N - 16), strlen(nd), nd);

    FMIndex narrow, wide;
    build1(narrow, text, 16);
    build1(wide, text, 16, /*ci=*/false, /*fw=*/true);

    auto naive_count = [&](const std::string& p) {
        size_t c = 0, pos = 0;
        while ((pos = text.find(p, pos)) != std::string::npos) { ++c; ++pos; }
        return c;
    };
    auto naive_pos = [&](const std::string& p) {
        std::vector<uint64_t> v;
        size_t pos = 0;
        while ((pos = text.find(p, pos)) != std::string::npos) {
            v.push_back(pos);
            ++pos;
        }
        return v;
    };

    // build a mix of patterns: known needles, likely-present short strings,
    // likely-absent strings
    std::vector<std::string> pats;
    for (const char* nd : needles) pats.push_back(nd);
    for (int i = 0; i < 80; ++i) {
        size_t pl = 3 + rng() % 8;
        std::string p;
        for (size_t j = 0; j < pl; ++j) p += char('a' + rng() % 6);
        pats.push_back(p);
    }
    for (int i = 0; i < 20; ++i)  // absent (uses 'x','y','z' rarely in text)
        pats.push_back(std::string(4 + rng() % 4, 'x'));

    std::vector<std::pair<const uint8_t*, size_t>> batch;
    for (auto& p : pats) batch.emplace_back(bytes(p), p.size());
    auto counts32 = narrow.CountBatch(batch);

    for (size_t i = 0; i < pats.size(); ++i) {
        const std::string& p = pats[i];
        size_t want = naive_count(p);
        CHECK_EQ(narrow.Count(bytes(p), p.size()), want);
        CHECK_EQ(wide.Count(bytes(p), p.size()), want);   // 64-bit path agrees
        CHECK_EQ(counts32[i], want);                      // batched agrees
        auto want_pos = naive_pos(p);
        std::sort(want_pos.begin(), want_pos.end());
        CHECK_EQ(offsets(narrow, p), want_pos);           // single doc: offset==pos
        CHECK_EQ(offsets(wide, p), want_pos);
    }
    // Extract random windows against the source text (both paths).
    for (int q = 0; q < 200; ++q) {
        uint64_t pos = rng() % N;
        size_t len = rng() % 64;
        std::string want = text.substr(pos, std::min<size_t>(len, N - pos));
        CHECK_EQ(narrow.Extract(0, pos, len), want);
        CHECK_EQ(wide.Extract(0, pos, len), want);
    }
}

TEST(FMIndex, LongestMatchVsOracle) {
    std::string text = "the quick brown fox jumps over the lazy dog";
    FMIndex fm;
    build1(fm, text, 4);

    auto naive_count = [&](const std::string& p) {
        if (p.empty()) return size_t(0);
        size_t c = 0, pos = 0;
        while ((pos = text.find(p, pos)) != std::string::npos) { ++c; ++pos; }
        return c;
    };
    // brute-force longest substring of query that occurs in text
    auto oracle = [&](const std::string& q) {
        size_t best = 0, bpos = 0;
        for (size_t i = 0; i < q.size(); ++i)
            for (size_t len = q.size() - i; len > best; --len)
                if (text.find(q.substr(i, len)) != std::string::npos) {
                    best = len; bpos = i; break;
                }
        return std::make_pair(best, bpos);
    };

    for (std::string q : {"a quick brown cat", "the lazy dog", "xyz not here",
                          "jumps over", "q", "", "brown fox jumps over the l"}) {
        auto want = oracle(q);
        auto got = fm.LongestMatch(bytes(q), q.size());
        CHECK_EQ(got.length, want.first);
        if (got.length > 0) {
            // the reported span actually occurs, `count` times
            std::string sub = q.substr(got.query_pos, got.length);
            CHECK_EQ(got.count, naive_count(sub));
            CHECK(text.find(sub) != std::string::npos);
        }
    }
    // fully-contained query -> match is the whole query
    std::string whole = "quick brown";
    CHECK_EQ(fm.LongestMatch(bytes(whole), whole.size()).length, whole.size());
}

TEST(FMIndex, NextTokenCountsVsOracle) {
    std::string text = "abcabdabxabc";  // "ab" -> c,d,x,c ; "abc" -> (end?) etc.
    FMIndex fm;
    build1(fm, text, 4);

    auto check = [&](const std::string& ctx) {
        // oracle: scan text for ctx, tally the following byte
        std::map<uint8_t, size_t> want;
        size_t pos = 0;
        while ((pos = text.find(ctx, pos)) != std::string::npos) {
            if (pos + ctx.size() < text.size())
                want[(uint8_t)text[pos + ctx.size()]]++;
            ++pos;
        }
        auto got = fm.NextTokenCounts(bytes(ctx), ctx.size());
        // same set of (byte,count), sorted by byte
        std::vector<std::pair<uint8_t, size_t>> wv(want.begin(), want.end());
        CHECK_EQ(got, wv);
        // each entry matches Count(ctx+byte); sum <= Count(ctx)
        size_t sum = 0;
        for (auto& [b, cnt] : got) {
            std::string pc = ctx + char(b);
            CHECK_EQ(cnt, fm.Count(bytes(pc), pc.size()));
            sum += cnt;
        }
        CHECK(sum <= fm.Count(bytes(ctx), ctx.size()));
    };
    check("ab");   // -> c,d,x,c
    check("abc");
    check("a");
    check("z");    // absent -> empty
}

TEST(FMIndex, MatchingDocsAndFuzzyVsOracle) {
    std::vector<std::string> docs = {"paypal login here", "paypa1 secure page",
                                     "visit the paypall site", "unrelated content",
                                     "pay pal support",       "totally different"};
    FMIndex fm;
    buildn(fm, docs, 4);

    // exact: MatchingDocs == docs whose text contains P as a substring
    auto exact_oracle = [&](const std::string& p) {
        std::vector<uint64_t> v;
        for (size_t d = 0; d < docs.size(); ++d)
            if (docs[d].find(p) != std::string::npos) v.push_back(d);
        return v;
    };
    // fuzzy: a doc matches iff its OWN text contains a substring within edit
    // distance k of P (Sellers approximate substring matching, run per doc, so a
    // match never spans a document boundary). D[0][j] = 0 (empty pattern matches
    // ending anywhere at 0 cost); doc hit iff min_j D[pl][j] <= k.
    auto approx_in = [](const std::string& p, const std::string& w, uint32_t k) {
        const size_t pl = p.size(), nd = w.size();
        std::vector<int> prev(nd + 1, 0), cur(nd + 1, 0);
        for (size_t i = 1; i <= pl; ++i) {
            cur[0] = static_cast<int>(i);
            for (size_t j = 1; j <= nd; ++j) {
                int sub = prev[j - 1] + (p[i - 1] == w[j - 1] ? 0 : 1);
                cur[j] = std::min({sub, prev[j] + 1, cur[j - 1] + 1});
            }
            std::swap(prev, cur);
        }
        return *std::min_element(prev.begin(), prev.end()) <= static_cast<int>(k);
    };
    auto fuzzy_oracle = [&](const std::string& p, uint32_t k) {
        std::vector<uint64_t> ds;
        for (size_t d = 0; d < docs.size(); ++d)
            if (approx_in(p, docs[d], k)) ds.push_back(d);
        return ds;
    };

    for (std::string p : {"paypal", "secure", "pal", "xyz", "different"}) {
        CHECK_EQ(fm.MatchingDocs(bytes(p), p.size()), exact_oracle(p));
        CHECK_EQ(fm.FuzzyMatchingDocs(bytes(p), p.size(), 0), fuzzy_oracle(p, 0));
        for (uint32_t k = 1; k <= 2; ++k)
            CHECK_EQ(fm.FuzzyMatchingDocs(bytes(p), p.size(), k),
                     fuzzy_oracle(p, k));
    }

    // randomized: small alphabet so edits actually land on real substrings
    std::mt19937 rng(909);
    std::vector<std::string> rdocs;
    for (int d = 0; d < 30; ++d) {
        std::string doc;
        size_t dl = 5 + rng() % 15;
        for (size_t i = 0; i < dl; ++i) doc += char('a' + rng() % 4);
        rdocs.push_back(doc);
    }
    FMIndex rfm;
    buildn(rfm, rdocs, 4);
    auto rfuzzy = [&](const std::string& p, uint32_t k) {
        std::vector<uint64_t> ds;
        for (size_t d = 0; d < rdocs.size(); ++d)
            if (approx_in(p, rdocs[d], k)) ds.push_back(d);
        return ds;
    };
    for (int q = 0; q < 60; ++q) {
        size_t pl = 2 + rng() % 4;
        std::string p;
        for (size_t j = 0; j < pl; ++j) p += char('a' + rng() % 4);
        for (uint32_t k = 0; k <= 2; ++k)
            CHECK_EQ(rfm.FuzzyMatchingDocs(bytes(p), p.size(), k), rfuzzy(p, k));
    }
}

TEST(FMIndex, EmptyPatternDocApisReturnEmpty) {
    FMIndex fm;
    buildn(fm, {"error one", "error two", "warn"}, 2);
    std::string empty;
    const uint8_t* e = bytes(empty);
    // The document-scoped queries all return no hits for an empty pattern
    // (consistent with each other; a match at a separator is never reported).
    CHECK(fm.LocateDocs(e, 0).empty());
    CHECK(fm.MatchingDocs(e, 0).empty());
    CHECK(fm.LocatePrefixDocs(e, 0).empty());
    CHECK(fm.LocateSuffixDocs(e, 0).empty());
    CHECK(fm.FuzzyMatchingDocs(e, 0, 1).empty());
    // The raw Count of the empty pattern is still the position count (m).
    CHECK_EQ(fm.Count(e, 0), fm.bwt_size());
}

TEST(FMIndex, NulByteInContentIsIndexed) {
    // v2: '\0' is a first-class content byte, not the document separator. A
    // single document containing '\0' indexes normally and matches byte-exactly,
    // including patterns that contain '\0'.
    std::string text = "ab";
    text.push_back('\0');
    text += "cd";
    text.push_back('\0');
    text += "ab";  // "ab\0cd\0ab"
    FMIndex fm;
    build1(fm, text, 1);
    for (std::string pat : {std::string("ab"), std::string("cd"),
                            std::string("b\0c", 3), std::string("\0", 1),
                            std::string("ab\0cd", 5), std::string("d\0a", 3),
                            std::string("x")}) {
        CHECK_EQ(fm.Count(bytes(pat), pat.size()), brute_count(text, pat));
        CHECK_EQ(offsets(fm, pat), brute_positions(text, pat));
    }
}

TEST(FMIndex, NulByteContentDoesNotCrossDocuments) {
    // '\0' as content (mapped to a real dense id) is distinct from the internal
    // document separator (an out-of-byte-alphabet symbol), so a query can never
    // match across a document boundary even when the boundary byte would be '\0'.
    std::vector<std::string> docs = {std::string("a\0b", 3),
                                     std::string("c\0d", 3)};
    FMIndex fm;
    buildn(fm, docs, 1);
    // "\0" occurs once in each document.
    std::string nul("\0", 1);
    std::vector<uint64_t> both = {0, 1};
    CHECK_EQ(fm.MatchingDocs(bytes(nul), 1), both);
    CHECK_EQ(fm.Count(bytes(nul), 1), size_t(2));
    // Anchored + full-document matches see '\0' as content.
    CHECK_EQ(fm.MatchingDocs(bytes(std::string("a\0b", 3)), 3),
             std::vector<uint64_t>{0});
    // A pattern that would only match by spanning the doc separator finds nothing.
    CHECK(fm.MatchingDocs(bytes(std::string("b\0c", 3)), 3).empty());
    CHECK(fm.MatchingDocs(bytes(std::string("bc", 2)), 2).empty());
}

TEST(FMIndex, RandomizedFuzzWithNulBytes) {
    // Same differential-oracle fuzz as RandomizedFuzzVsBruteForce, but the
    // alphabet includes '\0' so content and patterns exercise the byte-0 path.
    std::mt19937 rng(4242);
    const char alpha[] = {'\0', '\1', 'a', 'b'};  // small alphabet, '\0' frequent
    for (int trial = 0; trial < 200; ++trial) {
        size_t n = 1 + rng() % 300;
        std::string text(n, 'a');
        for (auto& ch : text) {
            ch = alpha[rng() % 4];
        }
        uint32_t rate = 1 + rng() % 16;
        FMIndex fm;
        build1(fm, text, rate);
        for (int q = 0; q < 20; ++q) {
            size_t plen = 1 + rng() % 6;
            std::string pat(plen, 'a');
            for (auto& ch : pat) {
                ch = alpha[rng() % 4];
            }
            CHECK_EQ(fm.Count(bytes(pat), plen), brute_count(text, pat));
            CHECK_EQ(offsets(fm, pat), brute_positions(text, pat));
        }
    }
}

TEST(FMIndex, EmptyDocumentAmongNonEmpty) {
    std::vector<std::string> docs = {"", "abc", "", "abcd", ""};
    FMIndex fm;
    buildn(fm, docs, 2);
    std::vector<uint64_t> md = {1, 3};
    CHECK_EQ(fm.MatchingDocs(bytes("abc"), 3), md);
    CHECK_EQ(fm.LocatePrefixDocs(bytes("abc"), 3), md);
    std::vector<uint64_t> sd = {1};  // "abc" ends doc 1, not doc 3 ("abcd")
    CHECK_EQ(fm.LocateSuffixDocs(bytes("abc"), 3), sd);
    // Extract on empty documents is empty; content docs come back intact.
    CHECK(fm.Extract(0, 0, 5).empty());
    CHECK(fm.Extract(2, 0, 5).empty());
    CHECK(fm.Extract(4, 0, 5).empty());
    CHECK_EQ(fm.Extract(1, 0, 5), std::string("abc"));
    CHECK_EQ(fm.Extract(3, 1, 2), std::string("bc"));
    // serialize round-trip preserves all of it (empty-doc boundaries included).
    FMIndex fm2 = FMIndex::Deserialize(fm.Serialize());
    CHECK(fm2.valid());
    CHECK_EQ(fm2.MatchingDocs(bytes("abc"), 3), md);
    CHECK_EQ(fm2.LocateSuffixDocs(bytes("abc"), 3), sd);
    CHECK_EQ(fm2.Extract(1, 0, 5), std::string("abc"));
}

TEST(FMIndex, DeserializeRandomCorruptionNeverCrashes) {
    // Contract: Deserialize / LoadView must never crash or read out of bounds on
    // ANY byte sequence, and must reject a structurally-malformed blob. (A blob
    // that passes validation but has corrupted PAYLOAD WORDS yields a live index
    // whose answers may be wrong — byte integrity is the caller's job, e.g. a
    // storage checksum — so we do not query the survivors here.) This exercises
    // the parseView gauntlet: magic/version, byte_to_id_ range, qlevels vs sigma,
    // section sizes, derived wavelet starts, sampled popcount, sampled-SA range,
    // and doc_start monotonicity. Run under ASan to prove the no-OOB claim.
    FMIndex fm;
    buildn(fm, {"the quick brown", "fox jumps", "over the lazy dog"}, 4);
    std::string good = fm.Serialize();
    std::mt19937 rng(12345);
    size_t survivors = 0;
    for (int trial = 0; trial < 8000; ++trial) {
        std::string blob = good;
        int flips = 1 + rng() % 6;
        for (int f = 0; f < flips; ++f) {
            blob[rng() % blob.size()] ^= static_cast<char>(1 + rng() % 255);
        }
        FMIndex bad = FMIndex::Deserialize(blob);  // must never OOB/crash
        if (bad.valid()) ++survivors;
    }
    CHECK(survivors <= 8000);  // reached here without a crash / sanitizer trip
}

int
main() {
    setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered: survive a crash
    return simple_test::run_all();
}
