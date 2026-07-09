// Licensed to the LF AI & Data foundation under Apache-2.0.
#include <algorithm>
#include <atomic>
#include <random>
#include <string>
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
    fm.Build(bytes(text), text.size(), 1);
    CHECK_EQ(fm.bwt_size(), text.size() + 1);
    CHECK_EQ(fm.c_at(0), 0u);
    for (uint32_t c = 1; c < fm.alphabet(); ++c) {
        CHECK(fm.c_at(c - 1) <= fm.c_at(c));
    }
}

TEST(FMIndex, CountMatchesBruteForce) {
    std::string text = "mississippi";
    FMIndex fm;
    fm.Build(bytes(text), text.size(), 1);
    for (std::string pat :
         {"i", "s", "ss", "issi", "ippi", "mississippi", "x", "ppi"}) {
        CHECK_EQ(fm.Count(bytes(pat), pat.size()), brute_count(text, pat));
    }
}

TEST(FMIndex, NoMatchReturnsZero) {
    std::string text = "abcabc";
    FMIndex fm;
    fm.Build(bytes(text), text.size(), 1);
    std::string pat = "abcd";
    CHECK_EQ(fm.Count(bytes(pat), pat.size()), 0u);
}

TEST(FMIndex, LocateMatchesBruteForceAcrossSampleRates) {
    std::string text = "abracadabra_abracadabra";
    for (uint32_t rate : {1u, 2u, 3u, 8u, 32u}) {
        FMIndex fm;
        fm.Build(bytes(text), text.size(), rate);
        for (std::string pat :
             {"a", "abra", "bra", "dab", "xyz", "abracadabra"}) {
            CHECK_EQ(fm.Locate(bytes(pat), pat.size()),
                     brute_positions(text, pat));
        }
    }
}

TEST(FMIndex, DocMappingLocatesPkAndOffset) {
    std::vector<std::string> docs = {"cat", "banana", "cataract"};
    std::string concat;
    std::vector<uint64_t> starts;
    for (auto& d : docs) {
        starts.push_back(concat.size());
        concat += d;
    }
    FMIndex fm;
    fm.Build(bytes(concat), concat.size(), 2);
    fm.SetDocStarts(starts);

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
    std::string concat;
    std::vector<uint64_t> starts;
    for (auto& d : docs) {
        starts.push_back(concat.size());
        concat += d;
    }
    FMIndex fm;
    fm.Build(bytes(concat), concat.size(), 4);
    fm.SetDocStarts(starts);

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
        fm.Build(bytes(text), text.size(), rate);
        for (int q = 0; q < 20; ++q) {
            size_t plen = 1 + rng() % 6;
            std::string pat(plen, 'a');
            for (auto& ch : pat) {
                ch = 'a' + (rng() % 5);  // 'e' never appears in text
            }
            CHECK_EQ(fm.Count(bytes(pat), plen), brute_count(text, pat));
            CHECK_EQ(fm.Locate(bytes(pat), plen), brute_positions(text, pat));
        }
    }
}

TEST(FMIndex, UTF8AndBoundaryBytes) {
    // Build explicitly: a string literal with an embedded \x00 would be
    // truncated by std::string's const char* constructor.
    std::string text = "caf\xc3\xa9 \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e ";
    text.push_back('\xff');
    text.push_back('\x00');
    text.push_back('\x01');
    text += "test";
    text.push_back('\x00');
    FMIndex fm;
    fm.Build(bytes(text), text.size(), 3);
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
        fm.Build(bytes(text), text.size(), 8);
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

TEST(FMIndex, DnaReadAlignment) {
    std::string ref = "ACGTTGCACGATTACAGGATCCACGTACGTTGCA";
    FMIndex fm;
    fm.Build(bytes(ref), ref.size(), 4);
    std::string read = "ACGT";
    auto pos = fm.Locate(bytes(read), read.size());
    CHECK_EQ(pos, brute_positions(ref, read));
    CHECK_EQ(pos.size(), 3u);
    std::string read2 = "TTGCA";
    CHECK_EQ(fm.Locate(bytes(read2), read2.size()),
             brute_positions(ref, read2));
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
        std::vector<uint32_t> seq(n);
        for (auto& x : seq) {
            x = rng() % sigma;
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
    fm.Build(bytes(std::string()), 0, 4);
    CHECK_EQ(fm.Count(bytes(std::string("a")), 1), 0u);
    CHECK_EQ(fm.Count(bytes(std::string()), 0), 1u);  // empty matches 1 position
    CHECK_EQ(fm.Locate(bytes(std::string("a")), 1).size(), 0u);
    std::string blob = fm.Serialize();
    FMIndex fm2 = FMIndex::Deserialize(blob);
    CHECK_EQ(fm2.Count(bytes(std::string("a")), 1), 0u);
}

TEST(FMIndex, SingleCharAndAllSame) {
    for (std::string text : {std::string("a"), std::string("aaaaaaaa")}) {
        FMIndex fm;
        fm.Build(bytes(text), text.size(), 2);
        CHECK_EQ(fm.Count(bytes(std::string("a")), 1), text.size());
        CHECK_EQ(fm.Count(bytes(std::string("aa")), 2),
                 text.size() >= 2 ? text.size() - 1 : 0u);
        CHECK_EQ(fm.Count(bytes(std::string("b")), 1), 0u);
        CHECK_EQ(fm.Locate(bytes(std::string("a")), 1).size(), text.size());
    }
}

TEST(FMIndex, FullByteAlphabetAndBinary) {
    // every byte value 0..255 present, arbitrary binary (not UTF-8)
    std::mt19937 rng(2024);
    std::string text;
    for (int i = 0; i < 256; ++i) {
        text.push_back(static_cast<char>(i));
    }
    for (int i = 0; i < 4000; ++i) {
        text.push_back(static_cast<char>(rng() & 0xff));
    }
    FMIndex fm;
    fm.Build(bytes(text), text.size(), 8);
    CHECK_EQ(fm.alphabet(), 257u);  // 256 bytes + sentinel
    // check a handful of substrings against brute force
    std::mt19937 rng2(5);
    for (int q = 0; q < 50; ++q) {
        size_t L = 1 + rng2() % 5;
        size_t pos = rng2() % (text.size() - L);
        std::string pat = text.substr(pos, L);
        CHECK_EQ(fm.Count(bytes(pat), pat.size()), brute_count(text, pat));
        CHECK_EQ(fm.Locate(bytes(pat), pat.size()),
                 brute_positions(text, pat));
    }
}

TEST(FMIndex, PatternLongerThanText) {
    std::string text = "abc";
    FMIndex fm;
    fm.Build(bytes(text), text.size(), 1);
    CHECK_EQ(fm.Count(bytes(std::string("abcd")), 4), 0u);
    CHECK_EQ(fm.Count(bytes(std::string("abcabc")), 6), 0u);
}

TEST(FMIndex, CountBatchEdgeCases) {
    std::string text = "abracadabra";
    FMIndex fm;
    fm.Build(bytes(text), text.size(), 2);
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
    WaveletMatrix4 wm4(std::vector<uint32_t>{}, 2);
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
        std::vector<uint32_t> seq(n);
        for (auto& x : seq) {
            x = rng() % sigma;
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
    WaveletMatrix bwm(seq, bits);
    WaveletMatrix4 qwm(seq, qlevels);
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
TEST(FMIndex, LocateDocsWithoutSetDocStarts) {
    // default: whole corpus is one document starting at 0 (no underflow)
    std::string text = "abcabc";
    FMIndex fm;
    fm.Build(bytes(text), text.size(), 1);
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
    fm.Build(bytes(text), text.size(), 4);
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
    fm.Build(bytes(text), text.size(), 4);
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
    std::string text = "the quick brown fox the lazy dog the the quick fox";
    std::vector<std::string> docs = {"the quick brown fox", " the lazy dog",
                                     " the the quick fox"};
    std::string concat;
    std::vector<uint64_t> starts;
    for (auto& d : docs) {
        starts.push_back(concat.size());
        concat += d;
    }
    FMIndex fm;
    fm.Build(bytes(concat), concat.size(), 4);
    fm.SetDocStarts(starts);
    std::string blob = fm.Serialize();

    // LoadView requires 8-byte-aligned bytes; a uint64 buffer guarantees it.
    std::vector<uint64_t> aligned((blob.size() + 7) / 8, 0);
    std::memcpy(aligned.data(), blob.data(), blob.size());
    FMIndex view = FMIndex::LoadView(
        reinterpret_cast<const uint8_t*>(aligned.data()), blob.size());

    for (std::string pat : {"the", "quick", "fox", "zzz", "the the"}) {
        CHECK_EQ(view.Count(bytes(pat), pat.size()),
                 fm.Count(bytes(pat), pat.size()));
        CHECK_EQ(view.Locate(bytes(pat), pat.size()),
                 fm.Locate(bytes(pat), pat.size()));
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
    fm.Build(bytes(text), text.size(), 4);
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
            CHECK_EQ(mapped->index().Locate(bytes(pat), pat.size()),
                     fm.Locate(bytes(pat), pat.size()));
        }
    }
    std::remove(path.c_str());
}

TEST(FMIndex, PrefixDocsAnchored) {
    std::vector<std::string> docs = {"error: disk full", "warning: low mem",
                                     "error: timeout", "erratic value"};
    std::string concat;
    std::vector<uint64_t> starts;
    for (auto& d : docs) {
        starts.push_back(concat.size());
        concat += d;
    }
    FMIndex fm;
    fm.Build(bytes(concat), concat.size(), 4);
    fm.SetDocStarts(starts);

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
    std::vector<std::string> docs = {"path/to/a.log", "path/to/b.txt",
                                     "c.log", "logistics"};
    std::string concat;
    std::vector<uint64_t> starts;
    for (auto& d : docs) {
        starts.push_back(concat.size());
        concat += d;
    }
    FMIndex fm;
    fm.Build(bytes(concat), concat.size(), 4);
    fm.SetDocStarts(starts);

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

TEST(FMIndex, CaseInsensitiveMatch) {
    std::string text = "The Quick BROWN fox; the quick brown FOX.";
    FMIndex fm;
    fm.Build(bytes(text), text.size(), 4, /*case_insensitive=*/true);
    CHECK(fm.case_insensitive());

    // all case variants of "quick" find both occurrences.
    for (std::string pat : {"quick", "QUICK", "QuIcK", "Quick"}) {
        CHECK_EQ(fm.Count(bytes(pat), pat.size()), size_t(2));
    }
    // "fox" (2 occurrences: "fox" and "FOX")
    CHECK_EQ(fm.Count(bytes("FOX"), 3), size_t(2));
    CHECK_EQ(fm.Count(bytes("fox"), 3), size_t(2));

    // Locate offsets still point into the ORIGINAL text.
    auto pos = fm.Locate(bytes("the"), 3);
    std::vector<uint64_t> expect = {0, 21};  // "The" at 0, "the" at 21
    CHECK_EQ(pos, expect);

    // non-letters unaffected; digits/punctuation exact.
    CHECK_EQ(fm.Count(bytes(";"), 1), size_t(1));

    // case-sensitive index does NOT fold.
    FMIndex cs;
    cs.Build(bytes(text), text.size(), 4);
    CHECK(!cs.case_insensitive());
    CHECK_EQ(cs.Count(bytes("QUICK"), 5), size_t(0));
    CHECK_EQ(cs.Count(bytes("quick"), 5), size_t(1));
}

TEST(FMIndex, CaseInsensitiveSurvivesRoundTrip) {
    std::string text = "Hello WORLD hello world";
    FMIndex fm;
    fm.Build(bytes(text), text.size(), 4, /*case_insensitive=*/true);
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
        fm.Build(bytes(text), text.size(), rate);
        // random windows, including edges and over-long lengths
        for (int q = 0; q < 30; ++q) {
            uint64_t pos = rng() % n;
            size_t len = rng() % (n + 5);
            std::string got = fm.Extract(pos, len);
            size_t want_len = std::min<size_t>(len, n - pos);
            CHECK_EQ(got, text.substr(pos, want_len));
        }
        // whole-text extraction
        CHECK_EQ(fm.Extract(0, n), text);
        // out-of-range
        CHECK(fm.Extract(n, 5).empty());
    }
}

TEST(FMIndex, ExtractContextAroundMatchAfterLoad) {
    std::string text = "the quick brown fox jumps over the lazy dog";
    FMIndex fm;
    fm.Build(bytes(text), text.size(), 4);
    std::string blob = fm.Serialize();
    FMIndex fm2 = FMIndex::Deserialize(blob);  // derived structs rebuilt on load

    for (std::string pat : {"quick", "fox", "the", "dog"}) {
        auto pos = fm2.Locate(bytes(pat), pat.size());
        CHECK(!pos.empty());
        for (uint64_t p : pos) {
            // the extracted window at the match equals the pattern
            CHECK_EQ(fm2.Extract(p, pat.size()), pat);
            // and a little context on each side matches the original text
            uint64_t start = p >= 3 ? p - 3 : 0;
            size_t clen = std::min<size_t>(pat.size() + 6, text.size() - start);
            CHECK_EQ(fm2.Extract(start, clen), text.substr(start, clen));
        }
    }
}

TEST(FMIndex, ConcurrentQueriesMatchSingleThreaded) {
    // Build one immutable index, hammer it from many threads, and check every
    // thread's answers equal the single-threaded ground truth (verifies the
    // const/lock-free query claim: no data races, no shared mutable state).
    std::mt19937 rng(7);
    std::string text;
    std::vector<uint64_t> starts;
    for (int d = 0; d < 400; ++d) {
        starts.push_back(text.size());
        size_t dl = 5 + rng() % 40;
        for (size_t i = 0; i < dl; ++i) text += char('a' + rng() % 8);
    }
    FMIndex fm;
    fm.Build(bytes(text), text.size(), 8);
    fm.SetDocStarts(starts);

    std::vector<std::string> pats;
    for (int i = 0; i < 64; ++i) {
        size_t pl = 1 + rng() % 5;
        std::string p;
        for (size_t j = 0; j < pl; ++j) p += char('a' + rng() % 8);
        pats.push_back(p);
    }
    // ground truth (single-threaded)
    std::vector<size_t> truth_count(pats.size());
    std::vector<std::vector<uint64_t>> truth_loc(pats.size());
    std::vector<std::string> truth_ex(pats.size());
    for (size_t i = 0; i < pats.size(); ++i) {
        truth_count[i] = fm.Count(bytes(pats[i]), pats[i].size());
        truth_loc[i] = fm.Locate(bytes(pats[i]), pats[i].size());
        truth_ex[i] = fm.Extract(i * 3 % text.size(), 10);
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
                if (fm.Locate(bytes(pats[i]), pats[i].size()) != truth_loc[i])
                    mismatches++;
                if (fm.Extract(i * 3 % text.size(), 10) != truth_ex[i])
                    mismatches++;
            }
        });
    }
    for (auto& th : ts) th.join();
    CHECK_EQ(mismatches.load(), 0);
}

int
main() {
    setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered: survive a crash
    return simple_test::run_all();
}
