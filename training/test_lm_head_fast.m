// test_lm_head_fast.m
// Benchmarks multiple approaches to LM head: [50257, 768] x [768, 1]
// Build: xcrun clang -O2 -fobjc-arc -o test_lm_head_fast test_lm_head_fast.m -framework Foundation -framework Accelerate

#import <Foundation/Foundation.h>
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#include <dispatch/dispatch.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define VOCAB    50257
#define HIDDEN   768
#define WARMUP   50
#define ITERS    500

static mach_timebase_info_data_t tb;

static inline double ticks_to_ms(uint64_t ticks) {
    return (double)ticks * tb.numer / tb.denom / 1e6;
}

static void rand_fp32(float *buf, int n) {
    for (int i = 0; i < n; i++)
        buf[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

// Convert fp32 row-major weight matrix to fp16
static void fp32_to_fp16(const float *src, __fp16 *dst, int n) {
    vImage_Buffer s = { (void *)src, 1, (size_t)n, n * sizeof(float) };
    vImage_Buffer d = { dst, 1, (size_t)n, n * sizeof(__fp16) };
    vImageConvert_PlanarFtoPlanar16F(&s, &d, 0);
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

int main(void) {
    mach_timebase_info(&tb);

    printf("Allocating buffers: weight [%d x %d], input [%d]...\n", VOCAB, HIDDEN, HIDDEN);

#define ALIGN_UP(n, a) (((n) + (a) - 1) & ~((a) - 1))
    // Allocate aligned buffers (size must be multiple of alignment for aligned_alloc)
    float *W    = (float *)aligned_alloc(64, ALIGN_UP((size_t)VOCAB * HIDDEN * sizeof(float), 64));
    float *x    = (float *)aligned_alloc(64, ALIGN_UP(HIDDEN * sizeof(float), 64));
    float *out0 = (float *)aligned_alloc(64, ALIGN_UP(VOCAB * sizeof(float), 64)); // baseline
    float *out  = (float *)aligned_alloc(64, ALIGN_UP(VOCAB * sizeof(float), 64)); // candidate
    __fp16 *W16 = (__fp16 *)aligned_alloc(64, ALIGN_UP((size_t)VOCAB * HIDDEN * sizeof(__fp16), 64));

    if (!W || !x || !out0 || !out || !W16) {
        fprintf(stderr, "Allocation failed (W=%p x=%p out0=%p out=%p W16=%p)\n",
                (void*)W, (void*)x, (void*)out0, (void*)out, (void*)W16);
        return 1;
    }

    srand(42);
    rand_fp32(W, VOCAB * HIDDEN);
    rand_fp32(x, HIDDEN);

    printf("Converting weights to fp16...\n");
    // Convert entire weight matrix to fp16 row by row
    for (int r = 0; r < VOCAB; r++) {
        fp32_to_fp16(W + (size_t)r * HIDDEN, W16 + (size_t)r * HIDDEN, HIDDEN);
    }
    printf("Ready.\n\n");

    uint64_t t0, t1, total;

    // -----------------------------------------------------------------------
    // 1. BASELINE: cblas_sgemv fp32
    // -----------------------------------------------------------------------
    // y = W * x: W is [VOCAB x HIDDEN], x is [HIDDEN], y is [VOCAB]
    // sgemv: y = alpha * A * x + beta * y
    //   CblasRowMajor, CblasNoTrans, M=VOCAB, N=HIDDEN, A=W, lda=HIDDEN
    for (int i = 0; i < WARMUP; i++)
        cblas_sgemv(CblasRowMajor, CblasNoTrans, VOCAB, HIDDEN,
                    1.0f, W, HIDDEN, x, 1, 0.0f, out0, 1);

    total = 0;
    for (int i = 0; i < ITERS; i++) {
        t0 = mach_absolute_time();
        cblas_sgemv(CblasRowMajor, CblasNoTrans, VOCAB, HIDDEN,
                    1.0f, W, HIDDEN, x, 1, 0.0f, out0, 1);
        t1 = mach_absolute_time();
        total += t1 - t0;
    }
    printf("[1] Baseline sgemv fp32:         %.3f ms\n", ticks_to_ms(total) / ITERS);

    // -----------------------------------------------------------------------
    // 2. fp16 weights + vDSP conversion per row
    // -----------------------------------------------------------------------
    // For each row, convert HIDDEN fp16 -> fp32 scratch, then dot product
    float *scratch = (float *)aligned_alloc(64, ALIGN_UP(HIDDEN * sizeof(float), 64));

    for (int i = 0; i < WARMUP; i++) {
        for (int r = 0; r < VOCAB; r++) {
            vImage_Buffer src16 = { (void *)(W16 + (size_t)r * HIDDEN), 1, HIDDEN, HIDDEN * sizeof(__fp16) };
            vImage_Buffer dst32 = { scratch, 1, HIDDEN, HIDDEN * sizeof(float) };
            vImageConvert_Planar16FtoPlanarF(&src16, &dst32, 0);
            vDSP_dotpr(scratch, 1, x, 1, &out[r], HIDDEN);
        }
    }

    total = 0;
    for (int i = 0; i < ITERS; i++) {
        t0 = mach_absolute_time();
        for (int r = 0; r < VOCAB; r++) {
            vImage_Buffer src16 = { (void *)(W16 + (size_t)r * HIDDEN), 1, HIDDEN, HIDDEN * sizeof(__fp16) };
            vImage_Buffer dst32 = { scratch, 1, HIDDEN, HIDDEN * sizeof(float) };
            vImageConvert_Planar16FtoPlanarF(&src16, &dst32, 0);
            vDSP_dotpr(scratch, 1, x, 1, &out[r], HIDDEN);
        }
        t1 = mach_absolute_time();
        total += t1 - t0;
    }
    printf("[2] fp16 weights + vDSP conv:    %.3f ms  (max_err=%.4f)\n",
           ticks_to_ms(total) / ITERS, max_abs_diff(out0, out, VOCAB));

    // -----------------------------------------------------------------------
    // 3. GCD parallel sgemv splits
    // -----------------------------------------------------------------------
    // Split VOCAB rows across threads; each calls cblas_sdot
    int ncores = 8; // P-cores on M5
    int chunk = (VOCAB + ncores - 1) / ncores;

    for (int i = 0; i < WARMUP; i++) {
        dispatch_apply(ncores, DISPATCH_APPLY_AUTO, ^(size_t tid) {
            int start = (int)tid * chunk;
            int end   = start + chunk;
            if (end > VOCAB) end = VOCAB;
            int rows  = end - start;
            if (rows <= 0) return;
            cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, HIDDEN,
                        1.0f, W + (size_t)start * HIDDEN, HIDDEN,
                        x, 1, 0.0f, out + start, 1);
        });
    }

    total = 0;
    for (int i = 0; i < ITERS; i++) {
        t0 = mach_absolute_time();
        dispatch_apply(ncores, DISPATCH_APPLY_AUTO, ^(size_t tid) {
            int start = (int)tid * chunk;
            int end   = start + chunk;
            if (end > VOCAB) end = VOCAB;
            int rows  = end - start;
            if (rows <= 0) return;
            cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, HIDDEN,
                        1.0f, W + (size_t)start * HIDDEN, HIDDEN,
                        x, 1, 0.0f, out + start, 1);
        });
        t1 = mach_absolute_time();
        total += t1 - t0;
    }
    printf("[3] GCD parallel sgemv (%d chunks): %.3f ms  (max_err=%.4f)\n",
           ncores, ticks_to_ms(total) / ITERS, max_abs_diff(out0, out, VOCAB));

    // -----------------------------------------------------------------------
    // 4. GCD parallel + fp16 storage: each thread converts its chunk
    // -----------------------------------------------------------------------
    // Each thread gets its own scratch buffer to avoid false sharing
    int nthreads = ncores;
    float **tbufs = (float **)malloc(nthreads * sizeof(float *));
    for (int t = 0; t < nthreads; t++)
        tbufs[t] = (float *)aligned_alloc(64, ALIGN_UP(HIDDEN * sizeof(float), 64));

    for (int i = 0; i < WARMUP; i++) {
        dispatch_apply(nthreads, DISPATCH_APPLY_AUTO, ^(size_t tid) {
            int start = (int)tid * chunk;
            int end   = start + chunk;
            if (end > VOCAB) end = VOCAB;
            float *tb2 = tbufs[tid];
            for (int r = start; r < end; r++) {
                vImage_Buffer s16 = { (void *)(W16 + (size_t)r * HIDDEN), 1, HIDDEN, HIDDEN * sizeof(__fp16) };
                vImage_Buffer d32 = { tb2, 1, HIDDEN, HIDDEN * sizeof(float) };
                vImageConvert_Planar16FtoPlanarF(&s16, &d32, 0);
                vDSP_dotpr(tb2, 1, x, 1, &out[r], HIDDEN);
            }
        });
    }

    total = 0;
    for (int i = 0; i < ITERS; i++) {
        t0 = mach_absolute_time();
        dispatch_apply(nthreads, DISPATCH_APPLY_AUTO, ^(size_t tid) {
            int start = (int)tid * chunk;
            int end   = start + chunk;
            if (end > VOCAB) end = VOCAB;
            float *tb2 = tbufs[tid];
            for (int r = start; r < end; r++) {
                vImage_Buffer s16 = { (void *)(W16 + (size_t)r * HIDDEN), 1, HIDDEN, HIDDEN * sizeof(__fp16) };
                vImage_Buffer d32 = { tb2, 1, HIDDEN, HIDDEN * sizeof(float) };
                vImageConvert_Planar16FtoPlanarF(&s16, &d32, 0);
                vDSP_dotpr(tb2, 1, x, 1, &out[r], HIDDEN);
            }
        });
        t1 = mach_absolute_time();
        total += t1 - t0;
    }
    printf("[4] GCD parallel + fp16:         %.3f ms  (max_err=%.4f)\n",
           ticks_to_ms(total) / ITERS, max_abs_diff(out0, out, VOCAB));

    // -----------------------------------------------------------------------
    // 5. cblas_sgemm trick: treat as [VOCAB,HIDDEN] x [HIDDEN,1]
    // -----------------------------------------------------------------------
    // sgemm: C = alpha*A*B + beta*C
    //   M=VOCAB, N=1, K=HIDDEN, A=W [M,K], B=x [K,N], C=out [M,N]
    for (int i = 0; i < WARMUP; i++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    VOCAB, 1, HIDDEN,
                    1.0f, W, HIDDEN, x, 1, 0.0f, out, 1);

    total = 0;
    for (int i = 0; i < ITERS; i++) {
        t0 = mach_absolute_time();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    VOCAB, 1, HIDDEN,
                    1.0f, W, HIDDEN, x, 1, 0.0f, out, 1);
        t1 = mach_absolute_time();
        total += t1 - t0;
    }
    printf("[5] cblas_sgemm trick:           %.3f ms  (max_err=%.4f)\n",
           ticks_to_ms(total) / ITERS, max_abs_diff(out0, out, VOCAB));

    // -----------------------------------------------------------------------
    // 6. vDSP_dotpr per row (serial)
    // -----------------------------------------------------------------------
    for (int i = 0; i < WARMUP; i++) {
        for (int r = 0; r < VOCAB; r++)
            vDSP_dotpr(W + (size_t)r * HIDDEN, 1, x, 1, &out[r], HIDDEN);
    }

    total = 0;
    for (int i = 0; i < ITERS; i++) {
        t0 = mach_absolute_time();
        for (int r = 0; r < VOCAB; r++)
            vDSP_dotpr(W + (size_t)r * HIDDEN, 1, x, 1, &out[r], HIDDEN);
        t1 = mach_absolute_time();
        total += t1 - t0;
    }
    printf("[6] vDSP_dotpr per row:          %.3f ms  (max_err=%.4f)\n",
           ticks_to_ms(total) / ITERS, max_abs_diff(out0, out, VOCAB));

    // cleanup
    free(W); free(x); free(out0); free(out); free(W16); free(scratch);
    for (int t = 0; t < nthreads; t++) free(tbufs[t]);
    free(tbufs);

    return 0;
}
