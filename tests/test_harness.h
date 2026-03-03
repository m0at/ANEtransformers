// test_harness.h -- Minimal test framework for ANE inference validation
// Reports pass/fail with numerical error metrics (max abs, mean abs, cosine sim)
#pragma once
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <mach/mach.h>
#import <mach/mach_time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------- colour codes ----------
#define C_GREEN  "\033[32m"
#define C_RED    "\033[31m"
#define C_YELLOW "\033[33m"
#define C_RESET  "\033[0m"

// ---------- error metrics ----------
typedef struct {
    float max_abs;       // max |a-b|
    float mean_abs;      // mean |a-b|
    float rms;           // sqrt(mean((a-b)^2))
    float cosine_sim;    // dot(a,b) / (|a||b|)
    float max_rel;       // max |a-b|/max(|a|,|b|,1e-8)
} ErrorMetrics;

static ErrorMetrics compute_error(const float *actual, const float *expected, int n) {
    ErrorMetrics m = {0};
    double sum_abs = 0, sum_sq = 0;
    double dot_ab = 0, dot_aa = 0, dot_bb = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(actual[i] - expected[i]);
        float denom = fmaxf(fmaxf(fabsf(actual[i]), fabsf(expected[i])), 1e-8f);
        float rel = d / denom;
        if (d > m.max_abs) m.max_abs = d;
        if (rel > m.max_rel) m.max_rel = rel;
        sum_abs += d;
        sum_sq += (double)d * d;
        dot_ab += (double)actual[i] * expected[i];
        dot_aa += (double)actual[i] * actual[i];
        dot_bb += (double)expected[i] * expected[i];
    }
    m.mean_abs = (float)(sum_abs / n);
    m.rms = (float)sqrt(sum_sq / n);
    double denom = sqrt(dot_aa) * sqrt(dot_bb);
    m.cosine_sim = (denom > 0) ? (float)(dot_ab / denom) : 1.0f;
    return m;
}

// ---------- test result ----------
typedef enum { TEST_PASS, TEST_FAIL, TEST_SKIP } TestStatus;

typedef struct {
    const char *name;
    const char *level;     // "L1:unit" "L2:layer" etc.
    TestStatus status;
    ErrorMetrics err;
    double elapsed_ms;
    char message[256];
} TestResult;

// ---------- global test state ----------
static int g_tests_run = 0;
static int g_tests_pass = 0;
static int g_tests_fail = 0;
static int g_tests_skip = 0;
static TestResult *g_results = NULL;
static int g_results_cap = 0;

static mach_timebase_info_data_t g_test_tb;
static bool g_test_tb_init = false;
static double test_ms(uint64_t t) {
    if (!g_test_tb_init) { mach_timebase_info(&g_test_tb); g_test_tb_init = true; }
    return (double)t * g_test_tb.numer / g_test_tb.denom / 1e6;
}

static void test_record(TestResult r) {
    if (g_tests_run >= g_results_cap) {
        g_results_cap = g_results_cap ? g_results_cap * 2 : 64;
        g_results = realloc(g_results, g_results_cap * sizeof(TestResult));
    }
    g_results[g_tests_run++] = r;
    if (r.status == TEST_PASS) g_tests_pass++;
    else if (r.status == TEST_FAIL) g_tests_fail++;
    else g_tests_skip++;

    const char *tag;
    if (r.status == TEST_PASS) tag = C_GREEN "PASS" C_RESET;
    else if (r.status == TEST_FAIL) tag = C_RED "FAIL" C_RESET;
    else tag = C_YELLOW "SKIP" C_RESET;

    printf("  [%s] %-40s", tag, r.name);
    if (r.status == TEST_PASS || r.status == TEST_FAIL) {
        printf("  max_abs=%.6f  cos=%.8f  rms=%.6f",
               r.err.max_abs, r.err.cosine_sim, r.err.rms);
    }
    if (r.message[0]) printf("  (%s)", r.message);
    if (r.elapsed_ms > 0) printf("  [%.1f ms]", r.elapsed_ms);
    printf("\n");
}

// ---------- assertion helpers ----------

// Check that error is within tolerance. Returns TestResult.
static TestResult test_check(const char *name, const char *level,
                              const float *actual, const float *expected, int n,
                              float tol_max_abs, float tol_cosine, double elapsed_ms) {
    TestResult r;
    r.name = name;
    r.level = level;
    r.elapsed_ms = elapsed_ms;
    r.message[0] = 0;
    r.err = compute_error(actual, expected, n);
    if (r.err.max_abs <= tol_max_abs && r.err.cosine_sim >= tol_cosine) {
        r.status = TEST_PASS;
    } else {
        r.status = TEST_FAIL;
        snprintf(r.message, sizeof(r.message),
                 "tol: max_abs<=%.4f cos>=%.6f", tol_max_abs, tol_cosine);
    }
    return r;
}

static TestResult test_skip(const char *name, const char *level, const char *reason) {
    TestResult r;
    r.name = name;
    r.level = level;
    r.status = TEST_SKIP;
    r.elapsed_ms = 0;
    memset(&r.err, 0, sizeof(r.err));
    snprintf(r.message, sizeof(r.message), "%s", reason);
    return r;
}

// ---------- summary ----------
static void test_summary(void) {
    printf("\n========================================\n");
    printf("  %d tests: " C_GREEN "%d passed" C_RESET, g_tests_run, g_tests_pass);
    if (g_tests_fail) printf(", " C_RED "%d failed" C_RESET, g_tests_fail);
    if (g_tests_skip) printf(", " C_YELLOW "%d skipped" C_RESET, g_tests_skip);
    printf("\n========================================\n");
}

// ---------- random fill ----------
static void fill_random(float *buf, int n, float scale) {
    for (int i = 0; i < n; i++)
        buf[i] = scale * ((float)arc4random() / (float)UINT32_MAX - 0.5f);
}

static void fill_ones(float *buf, int n) {
    for (int i = 0; i < n; i++) buf[i] = 1.0f;
}

// ---------- memory usage ----------
static size_t get_resident_bytes(void) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                                  (task_info_t)&info, &count);
    return (kr == KERN_SUCCESS) ? info.resident_size : 0;
}
