// Licensed to the LF AI & Data foundation under Apache-2.0.
// Benchmark the rank-block granularity (block_bytes) space/latency tradeoff:
// as block_bytes grows 8 -> 128, the resident wavelet rank directory shrinks
// ~1/block_bytes (one 8-byte rel_ sample per block) while rank (Count /
// MatchingDocs backward search) slows, since each rank does up to
// block_bytes/8 - 1 extra SWAR popcounts. Results are identical across
// block_bytes (see the BlockGranularityResultsInvariant unit test); this only
// measures cost. Usage: ./bench_blocksize [corpus_bytes] [kind 0=dna 1=text].
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
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
    size_t corpus_bytes = 8 * 1024 * 1024;
    int kind = 1;  // 0 = DNA, 1 = word-like text
    const uint32_t rate = 8;
    if (argc > 1) {
        corpus_bytes = std::stoull(argv[1]);
    }
    if (argc > 2) {
        kind = std::atoi(argv[2]);
    }

    std::string corpus =
        fmbench::make_corpus(corpus_bytes, kind, fmbench::kCorpusSeed);
    auto queries = fmbench::make_queries(
        corpus, fmbench::kQueryCount, fmbench::kQueryLen, fmbench::kQuerySeed);
    std::vector<std::string_view> docs{std::string_view(corpus)};
    const double corpus_mb = corpus_bytes / 1048576.0;

    printf("corpus=%.1f MiB  kind=%s  sample_rate=%u  queries=%zu  qlen=%zu\n",
           corpus_mb,
           kind == 0 ? "dna" : "text",
           rate,
           queries.size(),
           fmbench::kQueryLen);
    printf("%-9s %-9s %-11s %-13s %-13s %-13s %-10s\n",
           "block_B",
           "dir_MB",
           "dir/corpus",
           "count_q/s",
           "match_q/s",
           "locate_q/s",
           "build_ms");

    for (uint32_t bb : {8u, 16u, 32u, 64u, 128u}) {
        FMIndex fm;
        auto tb = clk::now();
        fm.Build(docs, rate, /*ci=*/false, /*fw=*/false, /*block_bytes=*/bb);
        double build_ms = ms_since(tb);

        double dir_mb = fm.rank_directory_bytes() / 1048576.0;

        volatile size_t sink = 0;
        auto t1 = clk::now();
        for (const auto& q : queries) {
            sink += fm.Count(bytes(q), q.size());
        }
        double count_qps = queries.size() / (ms_since(t1) / 1000.0);

        auto t2 = clk::now();
        for (const auto& q : queries) {
            sink += fm.MatchingDocs(bytes(q), q.size()).size();
        }
        double match_qps = queries.size() / (ms_since(t2) / 1000.0);

        // Locate (LF-walk enumeration) — the cost PatternMatch pays to turn an
        // SA interval into row ids. Rank-bound like Count but with the sampled
        // bitvector's rank on the hot path too.
        auto t3 = clk::now();
        for (const auto& q : queries) {
            sink += fm.LocateDocs(bytes(q), q.size()).size();
        }
        double locate_qps = queries.size() / (ms_since(t3) / 1000.0);
        (void)sink;

        printf("%-9u %-9.3f %-11.3f %-13.0f %-13.0f %-13.0f %-10.1f\n",
               bb,
               dir_mb,
               dir_mb / corpus_mb,
               count_qps,
               match_qps,
               locate_qps,
               build_ms);
    }
    return 0;
}
