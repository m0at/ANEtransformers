// test_lm_head_neon.m
// LM head matmul benchmark: fp32 cblas vs NEON fp16 variants
// Build: xcrun clang -O3 -fobjc-arc -o test_lm_head_neon test_lm_head_neon.m -framework Foundation -framework Accelerate

#import <Foundation/Foundation.h>
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <dispatch/dispatch.h>

#define VOCAB      50257
#define D_MODEL    768
#define WARMUP     50
#define ITERS      500
// GCD chunk size: number of rows per task
#define CHUNK_SIZE 512

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Convert fp32 array to fp16 using NEON
static void cvt_f32_to_f16(const float *src, __fp16 *dst, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t a = vld1q_f32(src + i);
        float32x4_t b = vld1q_f32(src + i + 4);
        float16x4_t ha = vcvt_f16_f32(a);
        float16x4_t hb = vcvt_f16_f32(b);
        vst1_f16(dst + i, ha);
        vst1_f16(dst + i + 4, hb);
    }
    for (; i < n; i++) dst[i] = (__fp16)src[i];
}

// Horizontal sum of float32x4 -> scalar
static inline float hsum_f32x4(float32x4_t v) {
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(s, s), 0);
}

// --- Approach 1: cblas_sgemv fp32 ---
static void bench_fp32_cblas(const float *W, const float *x, float *out) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                VOCAB, D_MODEL, 1.0f, W, D_MODEL,
                x, 1, 0.0f, out, 1);
}

// --- Approach 2: NEON fp16 dot product, fp16 accumulation ---
// Process 4 rows at once to better utilize NEON pipeline
static void bench_fp16_neon(const __fp16 *W, const __fp16 *x, float *out) {
    int r = 0;
    for (; r + 4 <= VOCAB; r += 4) {
        const __fp16 *r0 = W + (size_t)(r+0) * D_MODEL;
        const __fp16 *r1 = W + (size_t)(r+1) * D_MODEL;
        const __fp16 *r2 = W + (size_t)(r+2) * D_MODEL;
        const __fp16 *r3 = W + (size_t)(r+3) * D_MODEL;
        float16x8_t a0 = vdupq_n_f16(0), a1 = vdupq_n_f16(0);
        float16x8_t a2 = vdupq_n_f16(0), a3 = vdupq_n_f16(0);
        int i = 0;
        for (; i + 8 <= D_MODEL; i += 8) {
            float16x8_t xv = vld1q_f16(x + i);
            a0 = vfmaq_f16(a0, vld1q_f16(r0 + i), xv);
            a1 = vfmaq_f16(a1, vld1q_f16(r1 + i), xv);
            a2 = vfmaq_f16(a2, vld1q_f16(r2 + i), xv);
            a3 = vfmaq_f16(a3, vld1q_f16(r3 + i), xv);
        }
        // hsum each fp16x8 -> f32
        out[r+0] = hsum_f32x4(vcvt_f32_f16(vadd_f16(vget_low_f16(a0), vget_high_f16(a0))));
        out[r+1] = hsum_f32x4(vcvt_f32_f16(vadd_f16(vget_low_f16(a1), vget_high_f16(a1))));
        out[r+2] = hsum_f32x4(vcvt_f32_f16(vadd_f16(vget_low_f16(a2), vget_high_f16(a2))));
        out[r+3] = hsum_f32x4(vcvt_f32_f16(vadd_f16(vget_low_f16(a3), vget_high_f16(a3))));
        // tail (D_MODEL=768 is divisible by 8, so no tail needed, but keep for safety)
        for (; i < D_MODEL; i++) {
            out[r+0] += (float)r0[i] * (float)x[i];
            out[r+1] += (float)r1[i] * (float)x[i];
            out[r+2] += (float)r2[i] * (float)x[i];
            out[r+3] += (float)r3[i] * (float)x[i];
        }
    }
    // remaining rows
    for (; r < VOCAB; r++) {
        const __fp16 *row = W + (size_t)r * D_MODEL;
        float16x8_t acc = vdupq_n_f16(0);
        int i = 0;
        for (; i + 8 <= D_MODEL; i += 8)
            acc = vfmaq_f16(acc, vld1q_f16(row + i), vld1q_f16(x + i));
        out[r] = hsum_f32x4(vcvt_f32_f16(vadd_f16(vget_low_f16(acc), vget_high_f16(acc))));
        for (; i < D_MODEL; i++) out[r] += (float)row[i] * (float)x[i];
    }
}

// --- Approach 3: NEON fp16 weights, fp32 accumulation, 4-row unrolled ---
static void bench_fp16_fp32acc_neon(const __fp16 *W, const __fp16 *x, float *out) {
    int r = 0;
    for (; r + 4 <= VOCAB; r += 4) {
        const __fp16 *r0 = W + (size_t)(r+0) * D_MODEL;
        const __fp16 *r1 = W + (size_t)(r+1) * D_MODEL;
        const __fp16 *r2 = W + (size_t)(r+2) * D_MODEL;
        const __fp16 *r3 = W + (size_t)(r+3) * D_MODEL;
        float32x4_t a0l = vdupq_n_f32(0), a0h = vdupq_n_f32(0);
        float32x4_t a1l = vdupq_n_f32(0), a1h = vdupq_n_f32(0);
        float32x4_t a2l = vdupq_n_f32(0), a2h = vdupq_n_f32(0);
        float32x4_t a3l = vdupq_n_f32(0), a3h = vdupq_n_f32(0);
        int i = 0;
        for (; i + 8 <= D_MODEL; i += 8) {
            float16x8_t xv = vld1q_f16(x + i);
            float16x8_t p0 = vmulq_f16(vld1q_f16(r0 + i), xv);
            float16x8_t p1 = vmulq_f16(vld1q_f16(r1 + i), xv);
            float16x8_t p2 = vmulq_f16(vld1q_f16(r2 + i), xv);
            float16x8_t p3 = vmulq_f16(vld1q_f16(r3 + i), xv);
            a0l = vaddq_f32(a0l, vcvt_f32_f16(vget_low_f16(p0)));
            a0h = vaddq_f32(a0h, vcvt_f32_f16(vget_high_f16(p0)));
            a1l = vaddq_f32(a1l, vcvt_f32_f16(vget_low_f16(p1)));
            a1h = vaddq_f32(a1h, vcvt_f32_f16(vget_high_f16(p1)));
            a2l = vaddq_f32(a2l, vcvt_f32_f16(vget_low_f16(p2)));
            a2h = vaddq_f32(a2h, vcvt_f32_f16(vget_high_f16(p2)));
            a3l = vaddq_f32(a3l, vcvt_f32_f16(vget_low_f16(p3)));
            a3h = vaddq_f32(a3h, vcvt_f32_f16(vget_high_f16(p3)));
        }
        out[r+0] = hsum_f32x4(vaddq_f32(a0l, a0h));
        out[r+1] = hsum_f32x4(vaddq_f32(a1l, a1h));
        out[r+2] = hsum_f32x4(vaddq_f32(a2l, a2h));
        out[r+3] = hsum_f32x4(vaddq_f32(a3l, a3h));
        for (; i < D_MODEL; i++) {
            out[r+0] += (float)r0[i] * (float)x[i];
            out[r+1] += (float)r1[i] * (float)x[i];
            out[r+2] += (float)r2[i] * (float)x[i];
            out[r+3] += (float)r3[i] * (float)x[i];
        }
    }
    for (; r < VOCAB; r++) {
        const __fp16 *row = W + (size_t)r * D_MODEL;
        float32x4_t al = vdupq_n_f32(0), ah = vdupq_n_f32(0);
        int i = 0;
        for (; i + 8 <= D_MODEL; i += 8) {
            float16x8_t p = vmulq_f16(vld1q_f16(row + i), vld1q_f16(x + i));
            al = vaddq_f32(al, vcvt_f32_f16(vget_low_f16(p)));
            ah = vaddq_f32(ah, vcvt_f32_f16(vget_high_f16(p)));
        }
        out[r] = hsum_f32x4(vaddq_f32(al, ah));
        for (; i < D_MODEL; i++) out[r] += (float)row[i] * (float)x[i];
    }
}

// --- Approach 4a: fp16 fp16acc + coarse GCD parallel ---
static void bench_fp16_neon_parallel(const __fp16 *W, const __fp16 *x, float *out) {
    int nchunks = (VOCAB + CHUNK_SIZE - 1) / CHUNK_SIZE;
    dispatch_apply(nchunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t chunk) {
        int rstart = (int)chunk * CHUNK_SIZE;
        int rend   = rstart + CHUNK_SIZE;
        if (rend > VOCAB) rend = VOCAB;
        int r = rstart;
        for (; r + 4 <= rend; r += 4) {
            const __fp16 *r0 = W + (size_t)(r+0) * D_MODEL;
            const __fp16 *r1 = W + (size_t)(r+1) * D_MODEL;
            const __fp16 *r2 = W + (size_t)(r+2) * D_MODEL;
            const __fp16 *r3 = W + (size_t)(r+3) * D_MODEL;
            float16x8_t a0 = vdupq_n_f16(0), a1 = vdupq_n_f16(0);
            float16x8_t a2 = vdupq_n_f16(0), a3 = vdupq_n_f16(0);
            int i = 0;
            for (; i + 8 <= D_MODEL; i += 8) {
                float16x8_t xv = vld1q_f16(x + i);
                a0 = vfmaq_f16(a0, vld1q_f16(r0 + i), xv);
                a1 = vfmaq_f16(a1, vld1q_f16(r1 + i), xv);
                a2 = vfmaq_f16(a2, vld1q_f16(r2 + i), xv);
                a3 = vfmaq_f16(a3, vld1q_f16(r3 + i), xv);
            }
            out[r+0] = hsum_f32x4(vcvt_f32_f16(vadd_f16(vget_low_f16(a0), vget_high_f16(a0))));
            out[r+1] = hsum_f32x4(vcvt_f32_f16(vadd_f16(vget_low_f16(a1), vget_high_f16(a1))));
            out[r+2] = hsum_f32x4(vcvt_f32_f16(vadd_f16(vget_low_f16(a2), vget_high_f16(a2))));
            out[r+3] = hsum_f32x4(vcvt_f32_f16(vadd_f16(vget_low_f16(a3), vget_high_f16(a3))));
        }
        for (; r < rend; r++) {
            const __fp16 *row = W + (size_t)r * D_MODEL;
            float16x8_t acc = vdupq_n_f16(0);
            for (int i = 0; i + 8 <= D_MODEL; i += 8)
                acc = vfmaq_f16(acc, vld1q_f16(row + i), vld1q_f16(x + i));
            out[r] = hsum_f32x4(vcvt_f32_f16(vadd_f16(vget_low_f16(acc), vget_high_f16(acc))));
        }
    });
}

// --- Approach 4b: fp16 fp32acc + coarse GCD parallel ---
static void bench_fp16_fp32acc_parallel(const __fp16 *W, const __fp16 *x, float *out) {
    int nchunks = (VOCAB + CHUNK_SIZE - 1) / CHUNK_SIZE;
    dispatch_apply(nchunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t chunk) {
        int rstart = (int)chunk * CHUNK_SIZE;
        int rend   = rstart + CHUNK_SIZE;
        if (rend > VOCAB) rend = VOCAB;
        int r = rstart;
        for (; r + 4 <= rend; r += 4) {
            const __fp16 *r0 = W + (size_t)(r+0) * D_MODEL;
            const __fp16 *r1 = W + (size_t)(r+1) * D_MODEL;
            const __fp16 *r2 = W + (size_t)(r+2) * D_MODEL;
            const __fp16 *r3 = W + (size_t)(r+3) * D_MODEL;
            float32x4_t a0l = vdupq_n_f32(0), a0h = vdupq_n_f32(0);
            float32x4_t a1l = vdupq_n_f32(0), a1h = vdupq_n_f32(0);
            float32x4_t a2l = vdupq_n_f32(0), a2h = vdupq_n_f32(0);
            float32x4_t a3l = vdupq_n_f32(0), a3h = vdupq_n_f32(0);
            int i = 0;
            for (; i + 8 <= D_MODEL; i += 8) {
                float16x8_t xv = vld1q_f16(x + i);
                float16x8_t p0 = vmulq_f16(vld1q_f16(r0 + i), xv);
                float16x8_t p1 = vmulq_f16(vld1q_f16(r1 + i), xv);
                float16x8_t p2 = vmulq_f16(vld1q_f16(r2 + i), xv);
                float16x8_t p3 = vmulq_f16(vld1q_f16(r3 + i), xv);
                a0l = vaddq_f32(a0l, vcvt_f32_f16(vget_low_f16(p0)));
                a0h = vaddq_f32(a0h, vcvt_f32_f16(vget_high_f16(p0)));
                a1l = vaddq_f32(a1l, vcvt_f32_f16(vget_low_f16(p1)));
                a1h = vaddq_f32(a1h, vcvt_f32_f16(vget_high_f16(p1)));
                a2l = vaddq_f32(a2l, vcvt_f32_f16(vget_low_f16(p2)));
                a2h = vaddq_f32(a2h, vcvt_f32_f16(vget_high_f16(p2)));
                a3l = vaddq_f32(a3l, vcvt_f32_f16(vget_low_f16(p3)));
                a3h = vaddq_f32(a3h, vcvt_f32_f16(vget_high_f16(p3)));
            }
            out[r+0] = hsum_f32x4(vaddq_f32(a0l, a0h));
            out[r+1] = hsum_f32x4(vaddq_f32(a1l, a1h));
            out[r+2] = hsum_f32x4(vaddq_f32(a2l, a2h));
            out[r+3] = hsum_f32x4(vaddq_f32(a3l, a3h));
        }
        for (; r < rend; r++) {
            const __fp16 *row = W + (size_t)r * D_MODEL;
            float32x4_t al = vdupq_n_f32(0), ah = vdupq_n_f32(0);
            for (int i = 0; i + 8 <= D_MODEL; i += 8) {
                float16x8_t p = vmulq_f16(vld1q_f16(row + i), vld1q_f16(x + i));
                al = vaddq_f32(al, vcvt_f32_f16(vget_low_f16(p)));
                ah = vaddq_f32(ah, vcvt_f32_f16(vget_high_f16(p)));
            }
            out[r] = hsum_f32x4(vaddq_f32(al, ah));
        }
    });
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static void run(const char *label, void (^fn)(void), const float *ref, const float *out, int check) {
    for (int i = 0; i < WARMUP; i++) fn();
    double t0 = now_sec();
    for (int i = 0; i < ITERS; i++) fn();
    double elapsed = (now_sec() - t0) / ITERS * 1000.0;
    if (check) {
        float d = max_abs_diff(ref, out, VOCAB);
        printf("  %-42s  avg: %6.3f ms   max_err: %.4f %s\n",
               label, elapsed, d, d < 0.5f ? "(OK)" : "(WARN)");
    } else {
        printf("  %-42s  avg: %6.3f ms\n", label, elapsed);
    }
}

int main(void) {
    printf("LM Head Benchmark  VOCAB=%d  D_MODEL=%d  WARMUP=%d  ITERS=%d\n",
           VOCAB, D_MODEL, WARMUP, ITERS);
    printf("GCD chunk size: %d rows/task (%d tasks)\n\n",
           CHUNK_SIZE, (VOCAB + CHUNK_SIZE - 1) / CHUNK_SIZE);

    size_t wsize = (size_t)VOCAB * D_MODEL;
    float *W32 = (float *)malloc(wsize * sizeof(float));
    float *x32 = (float *)malloc(D_MODEL * sizeof(float));
    float *ref  = (float *)malloc(VOCAB * sizeof(float));
    float *out  = (float *)malloc(VOCAB * sizeof(float));

    srand(42);
    for (size_t i = 0; i < wsize; i++) W32[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.02f;
    for (int i = 0; i < D_MODEL; i++) x32[i]  = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;

    __fp16 *W16 = (__fp16 *)malloc(wsize * sizeof(__fp16));
    __fp16 *x16 = (__fp16 *)malloc(D_MODEL * sizeof(__fp16));
    cvt_f32_to_f16(W32, W16, (int)wsize);
    cvt_f32_to_f16(x32, x16, D_MODEL);

    printf("fp32 weight matrix: %.1f MB\n", wsize * 4.0 / 1e6);
    printf("fp16 weight matrix: %.1f MB\n\n", wsize * 2.0 / 1e6);

    // Baseline - stores into ref
    run("1. fp32 cblas_sgemv (baseline)",
        ^{ bench_fp32_cblas(W32, x32, ref); }, ref, ref, 0);

    run("2. NEON fp16 accum, 4-row unroll",
        ^{ bench_fp16_neon(W16, x16, out); }, ref, out, 1);

    run("3. NEON fp16 weights, fp32 accum, 4-row unroll",
        ^{ bench_fp16_fp32acc_neon(W16, x16, out); }, ref, out, 1);

    run("4a. NEON fp16 accum + GCD parallel",
        ^{ bench_fp16_neon_parallel(W16, x16, out); }, ref, out, 1);

    run("4b. NEON fp16 fp32acc + GCD parallel",
        ^{ bench_fp16_fp32acc_parallel(W16, x16, out); }, ref, out, 1);

    free(W32); free(x32); free(ref); free(out);
    free(W16); free(x16);
    return 0;
}
