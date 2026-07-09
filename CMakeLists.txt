cmake_minimum_required(VERSION 3.16)
project(fm_index_lite CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O2 -Wall -Wextra)

find_package(Threads REQUIRED)

# Include root mirrors Milvus (internal/core/src) so headers are included as
# "index/fmindex/...", making the eventual port a straight copy.
include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/third_party/libsais)

add_library(fmindex STATIC
    src/index/fmindex/FMIndex.cpp
    third_party/libsais/libsais.c)

add_executable(test_fmindex test/test_fmindex.cpp)
target_include_directories(test_fmindex PRIVATE ${CMAKE_SOURCE_DIR}/test)
target_link_libraries(test_fmindex PRIVATE fmindex Threads::Threads)

add_executable(bench_fmindex bench/bench_fmindex.cpp)
target_link_libraries(bench_fmindex PRIVATE fmindex)

add_executable(bench_mmap bench/bench_mmap.cpp)
target_link_libraries(bench_mmap PRIVATE fmindex)

add_executable(demo bench/demo.cpp)
target_link_libraries(demo PRIVATE fmindex)

enable_testing()
add_test(NAME fmindex_tests COMMAND test_fmindex)
