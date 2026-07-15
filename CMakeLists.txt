cmake_minimum_required(VERSION 3.16)
project(fm_index_lite CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization flags. -O3 over -O2 is marginal on its own — the query hot path is
# memory-latency bound (a serial chain of dependent rank cache misses), which
# instruction-level optimization can't hide. The flag that DOES matter is on x86:
# the SWAR quad-rank's __builtin_popcountll degrades to a *software* popcount
# unless the hardware POPCNT instruction is enabled. Measured (gcc 14, 4 MB): the
# batched-count path is ~1.5x slower and single count ~1.2x slower without it.
# arm64 has NEON popcount unconditionally, so the arch flag below is a no-op there.
add_compile_options(-O3 -Wall -Wextra)

include(CheckCXXCompilerFlag)
# x86-64-v2 (SSE4.2 + POPCNT) is a portable floor — every x86-64 CPU since ~2010.
# Bump to x86-64-v3 if the deployment guarantees AVX2, or -march=native for a
# non-portable local build. The check makes this skip cleanly on arm64.
check_cxx_compiler_flag("-march=x86-64-v2" FMINDEX_HAS_X86_64_V2)
if(FMINDEX_HAS_X86_64_V2)
  add_compile_options(-march=x86-64-v2)
endif()

find_package(Threads REQUIRED)

# Opt-in: parallelize the (single-index) suffix-array build via libsais OpenMP.
# OFF by default, so the default build pulls in NO OpenMP dependency and stays
# single-threaded. The natural parallelism for the shard model — building many
# per-segment indexes concurrently — needs neither this switch nor OpenMP.
option(FMINDEX_OPENMP "Parallel suffix-array build via libsais OpenMP" OFF)

# Include root mirrors Milvus (internal/core/src) so headers are included as
# "index/fmindex/...", making the eventual port a straight copy.
include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/third_party/libsais)

add_library(fmindex STATIC
    src/index/fmindex/FMIndex.cpp
    third_party/libsais/libsais.c
    third_party/libsais/libsais64.c)  # int64 SA for corpora >= 2 GiB

if(FMINDEX_OPENMP)
  find_package(OpenMP REQUIRED)
  # LIBSAIS_OPENMP both compiles the parallel paths inside libsais and exposes
  # the *_omp declarations in libsais.h; it must be visible to consumers that
  # include SuffixArray.h too, hence PUBLIC.
  target_compile_definitions(fmindex PUBLIC LIBSAIS_OPENMP)
  target_link_libraries(fmindex PUBLIC OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
  message(STATUS "fm-index-lite: OpenMP suffix-array build ENABLED")
endif()

add_executable(test_fmindex test/test_fmindex.cpp)
target_include_directories(test_fmindex PRIVATE ${CMAKE_SOURCE_DIR}/test)
target_link_libraries(test_fmindex PRIVATE fmindex Threads::Threads)

add_executable(bench_fmindex bench/bench_fmindex.cpp)
target_link_libraries(bench_fmindex PRIVATE fmindex)

add_executable(bench_mmap bench/bench_mmap.cpp)
target_link_libraries(bench_mmap PRIVATE fmindex)

add_executable(bench_anchored bench/bench_anchored.cpp)
target_link_libraries(bench_anchored PRIVATE fmindex)

add_executable(demo bench/demo.cpp)
target_link_libraries(demo PRIVATE fmindex)

add_executable(bench_gig bench/bench_gig.cpp)
target_link_libraries(bench_gig PRIVATE fmindex)

enable_testing()
add_test(NAME fmindex_tests COMMAND test_fmindex)
