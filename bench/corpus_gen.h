// Shared deterministic corpus + query generation, so every benchmark harness
// (ours, sdsl, lance) runs on identical inputs and their result dumps can be
// diffed for correctness.
#pragma once
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace fmbench {

// kind 0 = DNA {A,C,G,T}; kind 1 = word-like English.
inline std::string
make_corpus(size_t n, int kind, uint32_t seed) {
    std::mt19937 rng(seed);
    std::string s;
    s.reserve(n);
    if (kind == 0) {
        static const char* dna = "ACGT";
        while (s.size() < n) {
            s.push_back(dna[rng() % 4]);
        }
    } else {
        static const char* alpha = "abcdefghijklmnopqrstuvwxyz";
        while (s.size() < n) {
            size_t wlen = 2 + rng() % 8;
            for (size_t i = 0; i < wlen && s.size() < n; ++i) {
                s.push_back(alpha[rng() % 26]);
            }
            if (s.size() < n) {
                s.push_back(' ');
            }
        }
    }
    s.resize(n);
    return s;
}

// Random existing substrings (guaranteed matches) of length L.
inline std::vector<std::string>
make_queries(const std::string& corpus, size_t count, size_t L, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<std::string> qs;
    qs.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        size_t pos = rng() % (corpus.size() - L);
        qs.push_back(corpus.substr(pos, L));
    }
    return qs;
}

// Standard sweep used by all harnesses.
inline std::vector<size_t>
default_sizes() {
    return {256 * 1024, 1024 * 1024, 4 * 1024 * 1024};
}
constexpr size_t kQueryCount = 20000;
constexpr size_t kQueryLen = 16;
constexpr uint32_t kCorpusSeed = 42;
constexpr uint32_t kQuerySeed = 7;

}  // namespace fmbench
