// Build-memory benchmark over one long document. Keeping the row count at one
// makes the reported amplification about the byte corpus itself rather than
// std::string objects or document-boundary metadata.
#include <sys/resource.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include "index/fmindex/FMIndex.h"

using milvus::index::fmindex::FMIndex;

static uint64_t
peak_rss_bytes() {
    struct rusage ru {};
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return static_cast<uint64_t>(ru.ru_maxrss);
#else
    return static_cast<uint64_t>(ru.ru_maxrss) * 1024;
#endif
}

int
main(int argc, char** argv) {
    const size_t mib = argc > 1 ? std::stoull(argv[1]) : 64;
    const uint32_t sample_rate =
        argc > 2 ? static_cast<uint32_t>(std::stoul(argv[2])) : 8;
    const uint32_t block_bytes =
        argc > 3 ? static_cast<uint32_t>(std::stoul(argv[3])) : 64;
    const size_t corpus_bytes = mib * (size_t{1} << 20);

    // A deterministic printable-byte corpus keeps the alphabet large enough to
    // exercise the normal multi-level wavelet build without allocating a second
    // source buffer.
    std::string document(corpus_bytes, '\0');
    uint32_t state = 0x9e3779b9u;
    for (size_t i = 0; i < document.size(); ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        document[i] = static_cast<char>(32 + state % 95);
    }
    std::vector<std::string_view> docs{document};
    const uint64_t rss_with_input = peak_rss_bytes();

    FMIndex index;
    const auto start = std::chrono::steady_clock::now();
    index.Build(docs, sample_rate, false, false, block_bytes);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    const uint64_t peak = peak_rss_bytes();

    std::printf(
        "corpus_mib=%zu sample_rate=%u block_bytes=%u build_seconds=%.3f "
        "rss_with_input_mib=%.1f peak_rss_mib=%.1f peak_x=%.3f "
        "build_extra_x=%.3f\n",
        mib,
        sample_rate,
        block_bytes,
        seconds,
        rss_with_input / double(size_t{1} << 20),
        peak / double(size_t{1} << 20),
        peak / double(corpus_bytes),
        (peak - rss_with_input) / double(corpus_bytes));
    return 0;
}
