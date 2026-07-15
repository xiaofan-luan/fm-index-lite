// Benchmark for the separator-rank fast paths on the anchored prefix/suffix
// document queries (LIKE 'p%' / LIKE '%p').
//
// It compares, on identical query sets, two ways of answering each query:
//
//   OLD  "locate-all-then-filter": locate EVERY occurrence of the pattern in the
//        corpus (LocateDocs), then keep the ones whose offset sits on a document
//        boundary. This is what the anchored queries used to do.
//   NEW  separator-rank: CountPrefixDocs / CountSuffixDocs are O(|pattern|) with
//        no locate at all; LocatePrefixDocs / LocateSuffixDocs locate only the
//        boundary rows (prefix: skip interior rows via a wavelet access scan;
//        suffix: seed the backward search on the separator interval so only the
//        matching document-end rows are ever produced).
//
// The OLD numbers are reconstructed here from the public LocateDocs API + known
// document lengths, so this file needs no access to private internals and the
// two paths are measured apples-to-apples on the same index.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "index/fmindex/FMIndex.h"

using clk = std::chrono::steady_clock;
using milvus::index::fmindex::FMIndex;

static const uint8_t*
b(const std::string& s) {
    return reinterpret_cast<const uint8_t*>(s.data());
}

// A corpus tuned to expose the divergence between the two paths: a small
// alphabet so the query tokens ALSO appear all over the interior of documents
// (making "locate everything" expensive), while a controlled fraction of the
// documents actually begin / end with the token (the anchored answer).
struct Corpus {
    std::vector<std::string> docs;
    std::vector<std::string_view> views;
    std::vector<uint32_t> len;  // doc content length, for the OLD suffix filter
};

static Corpus
make_corpus(size_t ndocs, uint32_t seed) {
    std::mt19937 rng(seed);
    static const char* alpha = "abcd";  // tiny alphabet -> tokens recur interior
    const std::string pre = "log";      // shared 3-byte prefix token
    const std::string suf = "log";      // shared 3-byte suffix token
    Corpus c;
    c.docs.reserve(ndocs);
    for (size_t i = 0; i < ndocs; ++i) {
        size_t body = 24 + rng() % 40;
        std::string s;
        s.reserve(body + 8);
        bool want_prefix = (rng() % 4) == 0;  // ~1/4 docs begin with "log"
        bool want_suffix = (rng() % 4) == 0;  // ~1/4 docs end with "log"
        if (want_prefix) {
            s += pre;
        }
        for (size_t k = 0; k < body; ++k) {
            s.push_back(alpha[rng() % 4]);
        }
        if (want_suffix) {
            s += suf;
        }
        c.docs.push_back(std::move(s));
    }
    c.views.reserve(ndocs);
    c.len.reserve(ndocs);
    for (auto& d : c.docs) {
        c.views.emplace_back(d);
        c.len.push_back(static_cast<uint32_t>(d.size()));
    }
    return c;
}

// OLD path: distinct docs that BEGIN with p, via locate-all-then-filter.
static size_t
old_prefix_docs(const FMIndex& fm, const std::string& p) {
    auto hits = fm.LocateDocs(b(p), p.size());
    std::unordered_set<uint64_t> docs;
    for (auto& [doc, off] : hits) {
        if (off == 0) {
            docs.insert(doc);
        }
    }
    return docs.size();
}

// OLD path: distinct docs that END with p, via locate-all-then-filter.
static size_t
old_suffix_docs(const FMIndex& fm, const std::string& p,
                const std::vector<uint32_t>& len) {
    auto hits = fm.LocateDocs(b(p), p.size());
    std::unordered_set<uint64_t> docs;
    for (auto& [doc, off] : hits) {
        if (off + p.size() == len[doc]) {
            docs.insert(doc);
        }
    }
    return docs.size();
}

// Run f repeatedly until a wall-time budget elapses; return queries/sec. Auto-
// scaling the iteration count keeps both the O(1) count path (millions/sec) and
// the slow locate-everything path (hundreds/sec) accurately timed in the same
// harness.
template <typename F>
static double
timed(double budget_ms, F&& f) {
    volatile size_t sink = 0;
    auto t0 = clk::now();
    size_t reps = 0;
    double ms = 0.0;
    do {
        for (int i = 0; i < 16; ++i) {
            sink += f();
        }
        reps += 16;
        ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    } while (ms < budget_ms);
    (void)sink;
    return reps / (ms / 1000.0);
}

int
main(int argc, char** argv) {
    size_t ndocs = (argc > 1) ? std::stoull(argv[1]) : 1000000;  // 1M docs
    uint32_t rate = (argc > 2) ? std::stoul(argv[2]) : 8;        // default 8
    double budget = (argc > 3) ? std::stod(argv[3]) : 300.0;     // ms per op

    fprintf(stderr, "generating %zu docs...\n", ndocs);
    Corpus c = make_corpus(ndocs, 42);
    size_t bytes = 0;
    for (auto& d : c.docs) {
        bytes += d.size();
    }
    fprintf(stderr, "corpus %.1f MB, building (sample_rate=%u)...\n",
            bytes / 1e6, rate);
    FMIndex fm;
    fm.Build(c.views, rate);

    // Two query regimes:
    //   "log*"  — a distinctive marker (letters outside the body alphabet): it
    //             occurs ONLY at the boundaries it was injected on, so almost
    //             nothing is located-and-discarded. Isolates the O(1)-count win.
    //   "aaa"/"aab"/"abc" — body-alphabet tokens that recur heavily in document
    //             interiors; only a small fraction of docs actually begin/end
    //             with them. This is the realistic LIKE case where the OLD path
    //             locates a huge interior occurrence set to answer a boundary
    //             question, and the seeded fast path locates only the answer.
    std::vector<std::string> queries = {"log", "logac", "aaa", "aab", "abc"};

    printf("# corpus=%zu docs (%.1f MB)  sample_rate=%u  budget=%.0fms/op\n",
           ndocs, bytes / 1e6, rate, budget);
    printf("# %-8s %-8s %10s %14s %14s %7s\n", "query", "op", "answer",
           "OLD_q/s", "NEW_q/s", "speedup");

    for (auto& q : queries) {
        size_t pc_new = fm.CountPrefixDocs(b(q), q.size());
        size_t sc_new = fm.CountSuffixDocs(b(q), q.size());
        size_t pl_new = fm.LocatePrefixDocs(b(q), q.size()).size();
        size_t sl_new = fm.LocateSuffixDocs(b(q), q.size()).size();

        // Correctness cross-check against the OLD reconstruction before timing.
        size_t pc_old = old_prefix_docs(fm, q);
        size_t sc_old = old_suffix_docs(fm, q, c.len);
        if (pc_new != pc_old || sc_new != sc_old || pl_new != pc_old ||
            sl_new != sc_old) {
            printf("!! MISMATCH q=%s prefix new=%zu old=%zu  suffix new=%zu old=%zu\n",
                   q.c_str(), pc_new, pc_old, sc_new, sc_old);
        }

        double pc_o = timed(budget, [&] { return old_prefix_docs(fm, q); });
        double pc_n = timed(budget, [&] { return fm.CountPrefixDocs(b(q), q.size()); });
        double pl_o = timed(budget, [&] { return old_prefix_docs(fm, q); });
        double pl_n =
            timed(budget, [&] { return fm.LocatePrefixDocs(b(q), q.size()).size(); });
        double sc_o = timed(budget, [&] { return old_suffix_docs(fm, q, c.len); });
        double sc_n = timed(budget, [&] { return fm.CountSuffixDocs(b(q), q.size()); });
        double sl_o = timed(budget, [&] { return old_suffix_docs(fm, q, c.len); });
        double sl_n =
            timed(budget, [&] { return fm.LocateSuffixDocs(b(q), q.size()).size(); });

        printf("  %-8s %-8s %10zu %14.0f %14.0f %6.1fx\n", q.c_str(),
               "pfxCount", pc_new, pc_o, pc_n, pc_n / pc_o);
        printf("  %-8s %-8s %10zu %14.0f %14.0f %6.1fx\n", q.c_str(),
               "pfxDocs", pl_new, pl_o, pl_n, pl_n / pl_o);
        printf("  %-8s %-8s %10zu %14.0f %14.0f %6.1fx\n", q.c_str(),
               "sfxCount", sc_new, sc_o, sc_n, sc_n / sc_o);
        printf("  %-8s %-8s %10zu %14.0f %14.0f %6.1fx\n", q.c_str(),
               "sfxDocs", sl_new, sl_o, sl_n, sl_n / sl_o);
    }
    return 0;
}
