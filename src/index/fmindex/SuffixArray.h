// Licensed to the LF AI & Data foundation under Apache-2.0.
// Portions translated from Lance (lance_index::scalar::fmindex), Apache-2.0.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

#include "libsais.h"  // Apache-2.0, vendored under third_party/libsais

namespace milvus::index::fmindex {

// Suffix array via libsais (SA-IS, linear time). Input is a symbol vector
// (bytes remapped to 1..256 plus a sentinel 0, alphabet <= 257). Produces the
// standard suffix array (end-of-string smallest), matching the brute-force
// oracle. For n < 2^31; a 64-bit path (libsais64) is a later change.
inline std::vector<uint32_t>
build_suffix_array(const std::vector<uint32_t>& s) {
    const size_t n = s.size();
    std::vector<uint32_t> sa(n);
    if (n == 0) {
        return sa;
    }
    if (n == 1) {
        sa[0] = 0;
        return sa;
    }
    // libsais_int wants a mutable int32 text and an int32 SA buffer.
    std::vector<int32_t> t(s.begin(), s.end());
    int32_t k = 0;
    for (uint32_t v : s) {
        k = std::max(k, static_cast<int32_t>(v));
    }
    ++k;  // alphabet size = max symbol + 1
    const int32_t fs = 6 * 1024;  // recommended free space for performance
    std::vector<int32_t> saint(n + fs);
    int32_t rc =
        libsais_int(t.data(), saint.data(), static_cast<int32_t>(n), k, fs);
    (void)rc;  // rc == 0 on success; inputs here are always valid
    for (size_t i = 0; i < n; ++i) {
        sa[i] = static_cast<uint32_t>(saint[i]);
    }
    return sa;
}

}  // namespace milvus::index::fmindex
