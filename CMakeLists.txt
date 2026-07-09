cmake_minimum_required(VERSION 3.16)
project(fm_index_lite CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O2 -Wall -Wextra)

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

add_executable(demo bench/demo.cpp)
target_link_libraries(demo PRIVATE fmindex)

add_executable(bench_gig bench/bench_gig.cpp)
target_link_libraries(bench_gig PRIVATE fmindex)

enable_testing()
add_test(NAME fmindex_tests COMMAND test_fmindex)
