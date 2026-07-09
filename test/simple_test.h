// Minimal dependency-free test harness (so the project builds with just a
// compiler + CMake, no GoogleTest). When ported into Milvus, the tests are
// rewritten in GoogleTest; the FMIndex sources move over unchanged.
#pragma once
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace simple_test {

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>&
registry() {
    static std::vector<TestCase> r;
    return r;
}
inline int&
failures() {
    static int f = 0;
    return f;
}
inline int&
checks() {
    static int c = 0;
    return c;
}

struct Registrar {
    Registrar(const std::string& n, std::function<void()> fn) {
        registry().push_back({n, std::move(fn)});
    }
};

inline int
run_all() {
    int failed_tests = 0;
    for (auto& tc : registry()) {
        int before = failures();
        tc.fn();
        if (failures() > before) {
            ++failed_tests;
            printf("[ FAIL ] %s\n", tc.name.c_str());
        } else {
            printf("[  OK  ] %s\n", tc.name.c_str());
        }
    }
    printf("\n%zu tests, %d checks, %d check-failures, %d failed tests\n",
           registry().size(), checks(), failures(), failed_tests);
    return failed_tests == 0 ? 0 : 1;
}

}  // namespace simple_test

#define TEST(suite, name)                                                    \
    static void suite##_##name##_body();                                     \
    static simple_test::Registrar suite##_##name##_reg(#suite "." #name,     \
                                                       suite##_##name##_body); \
    static void suite##_##name##_body()

#define CHECK(cond)                                                       \
    do {                                                                  \
        ++simple_test::checks();                                          \
        if (!(cond)) {                                                    \
            ++simple_test::failures();                                    \
            printf("  CHECK failed: %s (%s:%d)\n", #cond, __FILE__,       \
                   __LINE__);                                             \
        }                                                                 \
    } while (0)

#define CHECK_EQ(a, b)                                                     \
    do {                                                                   \
        ++simple_test::checks();                                          \
        if (!((a) == (b))) {                                              \
            ++simple_test::failures();                                    \
            printf("  CHECK_EQ failed: %s == %s (%s:%d)\n", #a, #b,       \
                   __FILE__, __LINE__);                                   \
        }                                                                 \
    } while (0)
