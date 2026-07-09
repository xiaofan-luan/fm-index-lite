// Licensed to the LF AI & Data foundation under Apache-2.0.
// Benchmark for OUR FM-index: build time, query throughput, index size.
// Dumps per-query counts to results_ours.txt for cross-impl result diffing.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "bench/corpus_gen.h"
#include "index/fmindex/FMIndex.h"

using namespace milvus::index::fmindex;
using clk = std::chrono::steady_clock;

static double
ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}
static const uint8_t*
bytes(const std::string& s) {
    return reinterpret_cast<const uint8_t*>(s.data());
}

int
main(int argc, char** argv) {
    std::vector<size_t> sizes = fmbench::default_sizes();
    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; ++i) {
            sizes.push_back(std::stoull(argv[i]));
        }
    }
    const std::vector<uint32_t> sample_rates = {4, 32};

    printf("%-6s %-9s %-5s %-11s %-10s %-7s %-12s %-12s %-12s\n", "kind",
           "corpus", "rate", "build_ms", "idx_MB", "ratio", "count_q/s",
           "batch_q/s", "locate_q/s");
    printf("RESULT\tkind\tcorpus_bytes\tsample_rate\tbuild_ms\tindex_bytes\t"
           "ratio\tcount_qps\tlocate_qps\ttotal_occ\n");

    std::ofstream results("results_ours.txt");

    for (int kind : {0, 1}) {
        const char* kname = kind == 0 ? "dna" : "text";
        for (size_t n : sizes) {
            std::string corpus =
                fmbench::make_corpus(n, kind, fmbench::kCorpusSeed);
            auto queries = fmbench::make_queries(
                corpus, fmbench::kQueryCount, fmbench::kQueryLen,
                fmbench::kQuerySeed);
            for (uint32_t rate : sample_rates) {
                FMIndex fm;
                auto t0 = clk::now();
                fm.Build(bytes(corpus), corpus.size(), rate);
                double build_ms = ms_since(t0);

                std::string blob = fm.Serialize();
                double ratio = double(blob.size()) / double(corpus.size());

                uint64_t total_occ = 0;
                t0 = clk::now();
                for (auto& q : queries) {
                    total_occ += fm.Count(bytes(q), q.size());
                }
                double count_qps = queries.size() / (ms_since(t0) / 1000.0);

                // batched count throughput (memory-level parallelism)
                std::vector<std::pair<const uint8_t*, size_t>> batch;
                batch.reserve(queries.size());
                for (auto& q : queries) {
                    batch.emplace_back(bytes(q), q.size());
                }
                t0 = clk::now();
                auto bres = fm.CountBatch(batch);
                double batch_qps = queries.size() / (ms_since(t0) / 1000.0);
                uint64_t bsum = 0;
                for (size_t v : bres) {
                    bsum += v;
                }
                if (bsum != total_occ) {
                    printf("  BATCH MISMATCH kind=%s n=%zu\n", kname, n);
                }

                size_t LN = queries.size() / 4;
                t0 = clk::now();
                for (size_t i = 0; i < LN; ++i) {
                    volatile auto sz =
                        fm.Locate(bytes(queries[i]), queries[i].size()).size();
                    (void)sz;
                }
                double locate_qps = LN / (ms_since(t0) / 1000.0);

                printf(
                    "%-6s %-9zu %-5u %-11.1f %-10.3f %-7.3f %-12.0f %-12.0f "
                    "%-12.0f\n",
                    kname, n, rate, build_ms, blob.size() / (1024.0 * 1024.0),
                    ratio, count_qps, batch_qps, locate_qps);
                printf("RESULT\t%s\t%zu\t%u\t%.1f\t%zu\t%.4f\t%.0f\t%.0f\t%llu\n",
                       kname, n, rate, build_ms, blob.size(), ratio, count_qps,
                       locate_qps, (unsigned long long)total_occ);
            }
            // result dump for diffing (counts are sample-rate-independent)
            FMIndex fm;
            fm.Build(bytes(corpus), corpus.size(), 32);
            for (auto& q : queries) {
                results << kname << '\t' << n << '\t'
                        << fm.Count(bytes(q), q.size()) << '\n';
            }
        }
    }
    results.close();
    printf("\nquery-count dump -> results_ours.txt\n");
    return 0;
}
