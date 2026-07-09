// 1 GiB build + query benchmark for the document-scoped FMIndex.
// Reports build time, peak RSS, on-disk index size, and per-primitive latency.
#include <sys/resource.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "index/fmindex/FMIndex.h"

using namespace milvus::index::fmindex;
using clk = std::chrono::steady_clock;

static double
ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}
static const uint8_t*
B(const std::string& s) {
    return reinterpret_cast<const uint8_t*>(s.data());
}
// Peak resident set size in MB (macOS: ru_maxrss is bytes; Linux: KiB).
static double
peak_rss_mb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return ru.ru_maxrss / (1024.0 * 1024.0);
#else
    return ru.ru_maxrss / 1024.0;
#endif
}

int
main(int argc, char** argv) {
    const size_t target = (argc > 1 ? std::stoull(argv[1]) : 1) * (size_t(1) << 30);
    const size_t doc_len = 1024;  // ~1 KiB documents (rows)

    // ---- generate word-like documents totalling ~target bytes ----
    fprintf(stderr, "generating ~%.2f GiB corpus...\n",
            target / double(1 << 30));
    std::mt19937 rng(42);
    static const char* alpha = "abcdefghijklmnopqrstuvwxyz";
    std::vector<std::string> docs;
    size_t total = 0;
    docs.reserve(target / doc_len + 1);
    while (total < target) {
        std::string d;
        d.reserve(doc_len);
        while (d.size() < doc_len) {
            size_t wlen = 2 + rng() % 8;
            for (size_t i = 0; i < wlen && d.size() < doc_len; ++i)
                d.push_back(alpha[rng() % 26]);
            if (d.size() < doc_len) d.push_back(' ');
        }
        total += d.size();
        docs.push_back(std::move(d));
    }
    std::vector<std::string_view> views(docs.begin(), docs.end());
    fprintf(stderr, "  %zu docs, %.3f GiB total\n", docs.size(),
            total / double(1 << 30));

    // ---- build ----
    double rss_before = peak_rss_mb();
    FMIndex fm;
    auto t0 = clk::now();
    fm.Build(views, /*sa_sample_rate=*/32);
    double build_ms = ms_since(t0);
    double rss_after = peak_rss_mb();
    printf("BUILD   %.1f s   peak RSS %.2f GB (%.1fx corpus)\n",
           build_ms / 1000.0, rss_after / 1024.0,
           rss_after * 1024.0 * 1024.0 / total);

    // ---- on-disk index size (streamed, no full in-RAM blob) ----
    std::string path = "/tmp/fmidx_gig.fmix";
    auto ts = clk::now();
    bool ok = fm.SerializeToFile(path);
    double ser_ms = ms_since(ts);
    size_t idx_bytes = 0;
    if (ok) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (f) { std::fseek(f, 0, SEEK_END); idx_bytes = std::ftell(f); std::fclose(f); }
    }
    printf("INDEX   %.2f GB on disk (%.2fx corpus)   serialize %.1f s\n",
           idx_bytes / double(1 << 30), double(idx_bytes) / total,
           ser_ms / 1000.0);
    std::remove(path.c_str());

    // ---- queries: 20k length-16 substrings taken from within single docs ----
    const size_t NQ = 20000, QL = 16;
    std::vector<std::string> qs;
    qs.reserve(NQ);
    std::mt19937 qr(7);
    for (size_t i = 0; i < NQ; ++i) {
        const std::string& d = docs[qr() % docs.size()];
        size_t off = qr() % (d.size() - QL);
        qs.push_back(d.substr(off, QL));
    }

    // Count (per-query)
    t0 = clk::now();
    uint64_t occ = 0;
    for (auto& q : qs) occ += fm.Count(B(q), q.size());
    double count_ms = ms_since(t0);
    printf("COUNT   %.2f us/query   %.0f qps   (%llu total occ)\n",
           count_ms * 1000.0 / NQ, NQ / (count_ms / 1000.0),
           (unsigned long long)occ);

    // CountBatch (bulk)
    std::vector<std::pair<const uint8_t*, size_t>> batch;
    batch.reserve(NQ);
    for (auto& q : qs) batch.emplace_back(B(q), q.size());
    t0 = clk::now();
    auto bres = fm.CountBatch(batch);
    double batch_ms = ms_since(t0);
    uint64_t bsum = 0;
    for (size_t v : bres) bsum += v;
    printf("BATCH   %.2f us/query   %.0f qps   %s\n",
           batch_ms * 1000.0 / NQ, NQ / (batch_ms / 1000.0),
           bsum == occ ? "(matches Count)" : "(MISMATCH!)");

    // LocateDocs / MatchingDocs on a subset (heavier: LF walks per hit)
    const size_t NL = 4000;
    t0 = clk::now();
    uint64_t lhits = 0;
    for (size_t i = 0; i < NL; ++i)
        lhits += fm.LocateDocs(B(qs[i]), qs[i].size()).size();
    double loc_ms = ms_since(t0);
    printf("LOCDOCS %.2f us/query   %.0f qps   (%llu hits)\n",
           loc_ms * 1000.0 / NL, NL / (loc_ms / 1000.0),
           (unsigned long long)lhits);

    t0 = clk::now();
    uint64_t mdocs = 0;
    for (size_t i = 0; i < NL; ++i)
        mdocs += fm.MatchingDocs(B(qs[i]), qs[i].size()).size();
    double md_ms = ms_since(t0);
    printf("MATCHDOC %.2f us/query  %.0f qps   (%llu docs)\n",
           md_ms * 1000.0 / NL, NL / (md_ms / 1000.0),
           (unsigned long long)mdocs);

    // FuzzyMatchingDocs k=1 on a small subset (backtracking, much heavier)
    const size_t NF = 300;
    t0 = clk::now();
    uint64_t fdocs = 0;
    for (size_t i = 0; i < NF; ++i)
        fdocs += fm.FuzzyMatchingDocs(B(qs[i]), qs[i].size(), 1).size();
    double f_ms = ms_since(t0);
    printf("FUZZY-1 %.2f ms/query   %.0f qps   (%llu docs)\n",
           f_ms / NF, NF / (f_ms / 1000.0), (unsigned long long)fdocs);

    // Extract a document window
    t0 = clk::now();
    const size_t NE = 4000;
    size_t echk = 0;
    for (size_t i = 0; i < NE; ++i)
        echk += fm.Extract(i % docs.size(), 0, 64).size();
    double e_ms = ms_since(t0);
    printf("EXTRACT %.2f us/call    %.0f qps   (%zu bytes)\n",
           e_ms * 1000.0 / NE, NE / (e_ms / 1000.0), echk);

    printf("\n(rss_before=%.0f MB)\n", rss_before);
    return 0;
}
