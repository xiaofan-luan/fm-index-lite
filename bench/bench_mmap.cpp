// Compare query throughput and load latency: in-RAM vs Deserialize(copy) vs
// mmap zero-copy (cold first touch and warm).
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "bench/corpus_gen.h"
#include "index/fmindex/FMIndex.h"
#include "index/fmindex/MmapFmIndex.h"

using namespace milvus::index::fmindex;
using clk = std::chrono::steady_clock;
static double
ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static const uint8_t*
B(const std::string& s) {
    return reinterpret_cast<const uint8_t*>(s.data());
}

static double
count_qps(const FMIndex& fm, const std::vector<std::string>& qs) {
    auto t0 = clk::now();
    uint64_t acc = 0;
    for (auto& q : qs) {
        acc += fm.Count(B(q), q.size());
    }
    auto t1 = clk::now();
    if (acc == 12345678987654321ULL) {
        printf("x");  // defeat DCE
    }
    return qs.size() / (ms(t0, t1) / 1000.0);
}
static double
batch_qps(const FMIndex& fm,
          const std::vector<std::pair<const uint8_t*, size_t>>& b) {
    auto t0 = clk::now();
    auto r = fm.CountBatch(b);
    auto t1 = clk::now();
    if (r.size() == 999999999) {
        printf("x");
    }
    return b.size() / (ms(t0, t1) / 1000.0);
}

int
main() {
    size_t n = 4 * 1024 * 1024;
    std::string corpus = fmbench::make_corpus(n, 1, fmbench::kCorpusSeed);
    auto qs = fmbench::make_queries(corpus, 20000, 16, fmbench::kQuerySeed);
    std::vector<std::pair<const uint8_t*, size_t>> batch;
    for (auto& q : qs) {
        batch.emplace_back(B(q), q.size());
    }

    FMIndex ram;
    ram.Build(B(corpus), corpus.size(), 32);
    std::string blob = ram.Serialize();
    printf("index blob = %.2f MB\n", blob.size() / (1024.0 * 1024.0));

    // load latencies
    auto t0 = clk::now();
    FMIndex de = FMIndex::Deserialize(blob);
    auto t1 = clk::now();
    printf("Deserialize(copy) load = %.1f ms\n", ms(t0, t1));

    SaveToFile(ram, "/tmp/fmidx_bench.fmix");
    t0 = clk::now();
    auto mapped = MappedFMIndex::Open("/tmp/fmidx_bench.fmix");
    t1 = clk::now();
    printf("mmap Open+LoadView  = %.1f ms\n", ms(t0, t1));

    printf("\n%-22s %-13s %-13s\n", "variant", "count_q/s", "batch_q/s");
    printf("%-22s %-13.0f %-13.0f\n", "in-RAM (built)",
           count_qps(ram, qs), batch_qps(ram, batch));
    printf("%-22s %-13.0f %-13.0f\n", "Deserialize (copy)",
           count_qps(de, qs), batch_qps(de, batch));
    // cold: first query pass faults pages in
    double cold = count_qps(mapped->index(), qs);
    printf("%-22s %-13.0f (cold, page faults)\n", "mmap first pass", cold);
    printf("%-22s %-13.0f %-13.0f\n", "mmap warm",
           count_qps(mapped->index(), qs), batch_qps(mapped->index(), batch));
    std::remove("/tmp/fmidx_bench.fmix");
    return 0;
}
