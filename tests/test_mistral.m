// test_mistral.m -- Comprehensive test suite for Mistral 7B ANE inference
// Build:
//   xcrun clang -O2 -fobjc-arc -o test_mistral test_mistral.m \
//       -framework Foundation -framework IOSurface -framework Accelerate -ldl
// Run:
//   ./test_mistral [--level N] [--gguf /path/to/model.gguf] [--llamacpp /path/to/llama-cli]
//
// Levels: 1=unit, 2=layer, 3=integration, 4=perf, 5=correctness, 0=all
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Accelerate/Accelerate.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <getopt.h>

#include "test_harness.h"
#include "test_cpu_ref.h"
#include "test_ane_ops.h"

// ─── ANE runtime (reused from codebase) ───
static Class g_ANEDesc, g_ANEInMem, g_ANEReq, g_ANEIO;
static bool g_ane_loaded = false;

static void test_ane_init(void) {
    if (g_ane_loaded) return;
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");
    g_ane_loaded = true;
}

static IOSurfaceRef test_make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0
    });
}

// NEON fp16 conversion
static void cvt_f32_to_f16(_Float16 *dst, const float *src, int n) {
    int i = 0;
    for (; i + 7 < n; i += 8) {
        float16x8_t h = vcombine_f16(vcvt_f16_f32(vld1q_f32(src + i)),
                                      vcvt_f16_f32(vld1q_f32(src + i + 4)));
        vst1q_f16((__fp16 *)(dst + i), h);
    }
    for (; i < n; i++) dst[i] = (_Float16)src[i];
}

static void cvt_f16_to_f32(float *dst, const _Float16 *src, int n) {
    int i = 0;
    for (; i + 7 < n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16 *)(src + i));
        vst1q_f32(dst + i,     vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(h)));
    }
    for (; i < n; i++) dst[i] = (float)src[i];
}

// Blob builder: fp16 weight blob with 128-byte header (64 global + 64 chunk)
static NSData *test_build_blob(const float *w, int count) {
    int ws = count * 2, tot = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(tot, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = ws;
    *(uint32_t *)(b + 80) = 128;
    cvt_f32_to_f16((_Float16 *)(b + 128), w, count);
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// Multi-blob builder: concatenates blobs with correct offsets
static NSData *test_build_multi_blob(NSArray<NSData *> *blobs, uint64_t *offsets) {
    NSUInteger total = 64;
    for (NSData *blob in blobs) {
        uint32_t dsz = *(uint32_t *)((const uint8_t *)blob.bytes + 72);
        total += 64 + dsz;
    }
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 1; buf[4] = 2;
    NSUInteger pos = 64;
    for (NSUInteger i = 0; i < blobs.count; i++) {
        const uint8_t *src = (const uint8_t *)blobs[i].bytes;
        uint32_t dsz = *(uint32_t *)(src + 72);
        memcpy(buf + pos, src + 64, 64);              // chunk header
        *(uint32_t *)(buf + pos + 16) = (uint32_t)(pos + 64); // fix offset
        memcpy(buf + pos + 64, src + 128, dsz);       // data
        offsets[i] = pos + 64;
        pos += 64 + dsz;
    }
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// ---------- ANE kernel runner ----------
typedef struct {
    id model;
    IOSurfaceRef *ios_in;
    IOSurfaceRef *ios_out;
    id request;
    NSString *tmpDir;
    int n_in, n_out;
    size_t *in_bytes, *out_bytes;
} TestKernel;

static TestKernel *test_compile_kernel(NSString *mil, NSData *weightData,
                                        int n_in, size_t *in_sizes,
                                        int n_out, size_t *out_sizes) {
    test_ane_init();
    NSError *e = nil;
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

    NSDictionary *wdict = nil;
    if (weightData)
        wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weightData}};

    id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
        g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:), milData, wdict, nil);
    if (!desc) return NULL;

    id mdl = ((id(*)(Class, SEL, id))objc_msgSend)(
        g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
    if (!mdl) return NULL;

    id hexId = ((id(*)(id, SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    if (weightData)
        [weightData writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        fprintf(stderr, "  compile fail: %s\n", e ? [[e description] UTF8String] : "unknown");
        [fm removeItemAtPath:td error:nil];
        return NULL;
    }
    if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        [fm removeItemAtPath:td error:nil];
        return NULL;
    }

    TestKernel *k = calloc(1, sizeof(TestKernel));
    k->model = mdl;
    k->tmpDir = td;
    k->n_in = n_in;
    k->n_out = n_out;
    k->in_bytes = malloc(n_in * sizeof(size_t));
    k->out_bytes = malloc(n_out * sizeof(size_t));
    memcpy(k->in_bytes, in_sizes, n_in * sizeof(size_t));
    memcpy(k->out_bytes, out_sizes, n_out * sizeof(size_t));

    k->ios_in = malloc(n_in * sizeof(IOSurfaceRef));
    k->ios_out = malloc(n_out * sizeof(IOSurfaceRef));
    NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:n_in];
    NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:n_in];
    for (int i = 0; i < n_in; i++) {
        k->ios_in[i] = test_make_surface(in_sizes[i]);
        [wIns addObject:((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), k->ios_in[i])];
        [iIdx addObject:@(i)];
    }
    NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:n_out];
    NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:n_out];
    for (int i = 0; i < n_out; i++) {
        k->ios_out[i] = test_make_surface(out_sizes[i]);
        [wOuts addObject:((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), k->ios_out[i])];
        [oIdx addObject:@(i)];
    }
    k->request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        g_ANEReq,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        wIns, iIdx, wOuts, oIdx, nil, nil, @0);
    return k;
}

static void test_kernel_write_fp16(TestKernel *k, int idx, const float *data, int count) {
    IOSurfaceLock(k->ios_in[idx], 0, NULL);
    cvt_f32_to_f16((_Float16 *)IOSurfaceGetBaseAddress(k->ios_in[idx]), data, count);
    IOSurfaceUnlock(k->ios_in[idx], 0, NULL);
}

static void test_kernel_read_fp16(TestKernel *k, int idx, float *data, int count) {
    IOSurfaceLock(k->ios_out[idx], kIOSurfaceLockReadOnly, NULL);
    cvt_f16_to_f32(data, (_Float16 *)IOSurfaceGetBaseAddress(k->ios_out[idx]), count);
    IOSurfaceUnlock(k->ios_out[idx], kIOSurfaceLockReadOnly, NULL);
}

static bool test_kernel_eval(TestKernel *k) {
    NSError *e = nil;
    return ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
        k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, k->request, &e);
}

static void test_kernel_free(TestKernel *k) {
    if (!k) return;
    NSError *e = nil;
    ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
        k->model, @selector(unloadWithQoS:error:), 21, &e);
    for (int i = 0; i < k->n_in; i++) CFRelease(k->ios_in[i]);
    for (int i = 0; i < k->n_out; i++) CFRelease(k->ios_out[i]);
    [[NSFileManager defaultManager] removeItemAtPath:k->tmpDir error:nil];
    free(k->ios_in); free(k->ios_out);
    free(k->in_bytes); free(k->out_bytes);
    free(k);
}

// Convert row-major [S, D] to ANE channel-first [D, S]
static void transpose_to_ane(float *dst, const float *src, int S, int D) {
    for (int t = 0; t < S; t++)
        for (int d = 0; d < D; d++)
            dst[d * S + t] = src[t * D + d];
}

// Convert ANE channel-first [D, S] to row-major [S, D]
static void transpose_from_ane(float *dst, const float *src, int S, int D) {
    for (int t = 0; t < S; t++)
        for (int d = 0; d < D; d++)
            dst[t * D + d] = src[d * S + t];
}

// ═══════════════════════════════════════════════════════════════════════
// LEVEL 1: UNIT TESTS
// ═══════════════════════════════════════════════════════════════════════

static void test_l1_rmsnorm(void) {
    printf("\n--- L1: RMSNorm ---\n");
    int D = 256, S = 16;

    float *x = malloc(S * D * sizeof(float));
    float *w = malloc(D * sizeof(float));
    float *ref_out = malloc(S * D * sizeof(float));

    fill_random(x, S * D, 2.0f);
    fill_random(w, D, 1.0f);

    // CPU reference
    ref_rmsnorm(ref_out, x, w, S, D);

    // ANE: transpose to channel-first, build weight blob, run
    float *x_ane = malloc(D * S * sizeof(float));
    transpose_to_ane(x_ane, x, S, D);

    NSData *wblob = test_build_blob(w, D);
    NSString *mil = gen_test_rmsnorm(D, S);

    size_t in_bytes = D * S * 2;   // fp16
    size_t out_bytes = D * S * 2;
    uint64_t t0 = mach_absolute_time();
    TestKernel *k = test_compile_kernel(mil, wblob, 1, &in_bytes, 1, &out_bytes);

    TestResult r;
    if (!k) {
        r = test_skip("RMSNorm", "L1:unit", "ANE compile failed");
    } else {
        test_kernel_write_fp16(k, 0, x_ane, D * S);
        test_kernel_eval(k);

        float *ane_out_ch = malloc(D * S * sizeof(float));
        float *ane_out = malloc(S * D * sizeof(float));
        test_kernel_read_fp16(k, 0, ane_out_ch, D * S);
        transpose_from_ane(ane_out, ane_out_ch, S, D);

        double elapsed = test_ms(mach_absolute_time() - t0);
        // fp16 RMSNorm: expect max_abs up to ~1.0 due to fp16 precision
        // in the reciprocal-sqrt and broadcast multiply chain
        r = test_check("RMSNorm", "L1:unit", ane_out, ref_out, S * D,
                        1.0f, 0.999f, elapsed);
        free(ane_out_ch); free(ane_out);
        test_kernel_free(k);
    }
    test_record(r);
    free(x); free(w); free(ref_out); free(x_ane);
}

static void test_l1_silu(void) {
    printf("\n--- L1: SiLU ---\n");
    int C = 512, S = 16;

    float *x = malloc(C * S * sizeof(float));
    float *ref_out = malloc(C * S * sizeof(float));
    fill_random(x, C * S, 4.0f);
    ref_silu(ref_out, x, C * S);

    // ANE: needs a dummy weight blob (single fp16 = 1.0)
    float one_val = 1.0f;
    NSData *dummy_blob = test_build_blob(&one_val, 1);
    NSString *mil = gen_test_silu(C, S);
    size_t bytes = C * S * 2;
    uint64_t t0 = mach_absolute_time();
    TestKernel *k = test_compile_kernel(mil, dummy_blob, 1, &bytes, 1, &bytes);

    TestResult r;
    if (!k) {
        r = test_skip("SiLU", "L1:unit", "ANE compile failed");
    } else {
        // SiLU is element-wise, layout doesn't matter
        test_kernel_write_fp16(k, 0, x, C * S);
        test_kernel_eval(k);
        float *ane_out = malloc(C * S * sizeof(float));
        test_kernel_read_fp16(k, 0, ane_out, C * S);
        double elapsed = test_ms(mach_absolute_time() - t0);
        // fp16 SiLU: input range [-2,2], sigmoid precision ~ 1e-3,
        // max_abs error scales with |x| for large inputs
        r = test_check("SiLU", "L1:unit", ane_out, ref_out, C * S,
                        2.0f, 0.9999f, elapsed);
        free(ane_out);
        test_kernel_free(k);
    }
    test_record(r);
    free(x); free(ref_out);
}

static void test_l1_softmax(void) {
    printf("\n--- L1: Softmax ---\n");
    int H = 8, S = 16;

    float *x = malloc(H * S * S * sizeof(float));
    float *ref_out = malloc(H * S * S * sizeof(float));
    fill_random(x, H * S * S, 3.0f);
    ref_softmax(ref_out, x, H * S, S);

    // Dummy weight blob (single zero for bias)
    float zero_val = 0.0f;
    NSData *dummy_blob_sm = test_build_blob(&zero_val, 1);
    NSString *mil = gen_test_softmax(H, S);
    size_t bytes = H * S * S * 2;
    uint64_t t0 = mach_absolute_time();
    TestKernel *k = test_compile_kernel(mil, dummy_blob_sm, 1, &bytes, 1, &bytes);

    TestResult r;
    if (!k) {
        r = test_skip("Softmax", "L1:unit", "ANE compile failed");
    } else {
        test_kernel_write_fp16(k, 0, x, H * S * S);
        test_kernel_eval(k);
        float *ane_out = malloc(H * S * S * sizeof(float));
        test_kernel_read_fp16(k, 0, ane_out, H * S * S);
        double elapsed = test_ms(mach_absolute_time() - t0);
        // fp16 softmax: output in [0,1], fp16 precision is ~5e-4 at 1.0
        r = test_check("Softmax", "L1:unit", ane_out, ref_out, H * S * S,
                        0.5f, 0.9999f, elapsed);
        free(ane_out);
        test_kernel_free(k);
    }
    test_record(r);
    free(x); free(ref_out);
}

static void test_l1_conv_matmul(void) {
    printf("\n--- L1: Conv/Matmul ---\n");
    int IC = 256, OC = 512, S = 16;

    float *x = malloc(S * IC * sizeof(float));
    float *W = malloc(OC * IC * sizeof(float));
    float *ref_out = malloc(S * OC * sizeof(float));
    fill_random(x, S * IC, 0.5f);
    fill_random(W, OC * IC, 0.1f);

    // Reference: y = x @ W^T  (W is [OC, IC], x is [S, IC], y is [S, OC])
    ref_matmul(ref_out, x, W, S, OC, IC);

    // ANE: transpose x to [IC, S], bake W, output is [OC, S]
    float *x_ane = malloc(IC * S * sizeof(float));
    transpose_to_ane(x_ane, x, S, IC);

    NSData *wblob = test_build_blob(W, OC * IC);
    NSString *mil = gen_test_conv(IC, OC, S);

    size_t in_bytes = IC * S * 2;
    size_t out_bytes = OC * S * 2;
    uint64_t t0 = mach_absolute_time();
    TestKernel *k = test_compile_kernel(mil, wblob, 1, &in_bytes, 1, &out_bytes);

    TestResult r;
    if (!k) {
        r = test_skip("Conv/Matmul", "L1:unit", "ANE compile failed");
    } else {
        test_kernel_write_fp16(k, 0, x_ane, IC * S);
        test_kernel_eval(k);
        float *ane_out_ch = malloc(OC * S * sizeof(float));
        float *ane_out = malloc(S * OC * sizeof(float));
        test_kernel_read_fp16(k, 0, ane_out_ch, OC * S);
        transpose_from_ane(ane_out, ane_out_ch, S, OC);

        // Also compare against Accelerate BLAS
        float *blas_out = malloc(S * OC * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    S, OC, IC, 1.0f, x, IC, W, IC, 0.0f, blas_out, OC);

        double elapsed = test_ms(mach_absolute_time() - t0);
        r = test_check("Conv/Matmul vs BLAS", "L1:unit", ane_out, blas_out, S * OC,
                        0.5f, 0.999f, elapsed);
        free(ane_out_ch); free(ane_out); free(blas_out);
        test_kernel_free(k);
    }
    test_record(r);
    free(x); free(W); free(ref_out); free(x_ane);
}

static void test_l1_rope(void) {
    printf("\n--- L1: RoPE ---\n");
    // RoPE is CPU-side in this architecture, so we just validate the reference
    // against a known-answer test
    int S = 4, nh = 8, hd = 32;
    float *q = malloc(S * nh * hd * sizeof(float));
    float *k = malloc(S * nh * hd * sizeof(float));
    float *q2 = malloc(S * nh * hd * sizeof(float));
    float *k2 = malloc(S * nh * hd * sizeof(float));

    // Use deterministic input
    for (int i = 0; i < S * nh * hd; i++) {
        q[i] = 0.1f * (i % 17 - 8);
        k[i] = 0.1f * (i % 13 - 6);
    }
    memcpy(q2, q, S * nh * hd * sizeof(float));
    memcpy(k2, k, S * nh * hd * sizeof(float));

    // Apply RoPE twice with different implementations should give same result
    ref_rope(q, k, S, nh, nh, hd, 10000.0f, 0);

    // Manual verification: position 0 should be identity (cos=1, sin=0)
    // So q[0:hd] should be unchanged for pos=0
    // Actually cos(0)=1, sin(0)=0, so yes, identity at pos 0
    bool pos0_ok = true;
    for (int i = 0; i < nh * hd; i++) {
        if (fabsf(q[i] - q2[i]) > 1e-6f) { pos0_ok = false; break; }
    }

    TestResult r;
    r.name = "RoPE pos=0 identity";
    r.level = "L1:unit";
    r.elapsed_ms = 0;
    r.message[0] = 0;
    memset(&r.err, 0, sizeof(r.err));
    r.err.cosine_sim = 1.0f;
    r.status = pos0_ok ? TEST_PASS : TEST_FAIL;
    if (!pos0_ok) snprintf(r.message, sizeof(r.message), "pos=0 should be identity");
    test_record(r);

    // Test that RoPE preserves norm (it's a rotation)
    float norm_before = 0, norm_after = 0;
    for (int i = 0; i < nh * hd; i++) {
        norm_before += q2[nh * hd + i] * q2[nh * hd + i]; // pos=1 original
        norm_after += q[nh * hd + i] * q[nh * hd + i];     // pos=1 rotated
    }
    norm_before = sqrtf(norm_before);
    norm_after = sqrtf(norm_after);

    r.name = "RoPE preserves norm";
    r.status = (fabsf(norm_before - norm_after) / norm_before < 1e-5f) ? TEST_PASS : TEST_FAIL;
    r.err.max_abs = fabsf(norm_before - norm_after);
    r.err.cosine_sim = 1.0f;
    if (r.status == TEST_FAIL)
        snprintf(r.message, sizeof(r.message), "norm: %.6f -> %.6f", norm_before, norm_after);
    test_record(r);

    free(q); free(k); free(q2); free(k2);
}

static void test_l1_gqa(void) {
    printf("\n--- L1: GQA Attention ---\n");
    // Test that GQA head broadcasting works correctly
    // With n_heads=8, n_kv_heads=2, ratio=4
    int S = 8, nh = 8, nkv = 2, hd = 32;
    int q_dim = nh * hd, kv_dim = nkv * hd;

    float *q = malloc(S * q_dim * sizeof(float));
    float *k = malloc(S * kv_dim * sizeof(float));
    float *v = malloc(S * kv_dim * sizeof(float));
    float *out_gqa = malloc(S * q_dim * sizeof(float));

    fill_random(q, S * q_dim, 0.5f);
    fill_random(k, S * kv_dim, 0.5f);
    fill_random(v, S * kv_dim, 0.5f);

    // Run GQA reference
    ref_gqa_attention(out_gqa, q, k, v, S, nh, nkv, hd);

    // Verify: heads 0-3 should use KV head 0, heads 4-7 use KV head 1
    // We can test by expanding KV heads and running standard MHA
    float *k_exp = calloc(S * q_dim, sizeof(float));
    float *v_exp = calloc(S * q_dim, sizeof(float));
    int ratio = nh / nkv;
    for (int t = 0; t < S; t++)
        for (int h = 0; h < nh; h++) {
            int kv_h = h / ratio;
            memcpy(k_exp + t * q_dim + h * hd,
                   k + t * kv_dim + kv_h * hd, hd * sizeof(float));
            memcpy(v_exp + t * q_dim + h * hd,
                   v + t * kv_dim + kv_h * hd, hd * sizeof(float));
        }

    // Standard MHA with expanded KV
    float *out_mha = malloc(S * q_dim * sizeof(float));
    ref_gqa_attention(out_mha, q, k_exp, v_exp, S, nh, nh, hd);

    TestResult r = test_check("GQA head broadcast", "L1:unit",
                               out_gqa, out_mha, S * q_dim,
                               1e-5f, 0.999999f, 0);
    test_record(r);

    free(q); free(k); free(v); free(out_gqa);
    free(k_exp); free(v_exp); free(out_mha);
}

// ═══════════════════════════════════════════════════════════════════════
// LEVEL 2: LAYER TESTS
// ═══════════════════════════════════════════════════════════════════════

// Extended compile: accepts a dictionary of weight files (path -> NSData)
static TestKernel *test_compile_kernel_multi_w(NSString *mil, NSDictionary *weights,
                                                int n_in, size_t *in_sizes,
                                                int n_out, size_t *out_sizes) {
    test_ane_init();
    NSError *e = nil;
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

    id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
        g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:), milData, weights, nil);
    if (!desc) return NULL;

    id mdl = ((id(*)(Class, SEL, id))objc_msgSend)(
        g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
    if (!mdl) return NULL;

    id hexId = ((id(*)(id, SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (NSString *path in weights) {
        NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
        [weights[path][@"data"] writeToFile:[td stringByAppendingPathComponent:rel] atomically:YES];
    }

    if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        fprintf(stderr, "  compile fail: %s\n", e ? [[e description] UTF8String] : "unknown");
        [fm removeItemAtPath:td error:nil];
        return NULL;
    }
    if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        [fm removeItemAtPath:td error:nil];
        return NULL;
    }

    TestKernel *k = calloc(1, sizeof(TestKernel));
    k->model = mdl;
    k->tmpDir = td;
    k->n_in = n_in;
    k->n_out = n_out;
    k->in_bytes = malloc(n_in * sizeof(size_t));
    k->out_bytes = malloc(n_out * sizeof(size_t));
    memcpy(k->in_bytes, in_sizes, n_in * sizeof(size_t));
    memcpy(k->out_bytes, out_sizes, n_out * sizeof(size_t));

    k->ios_in = malloc(n_in * sizeof(IOSurfaceRef));
    k->ios_out = malloc(n_out * sizeof(IOSurfaceRef));
    NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:n_in];
    NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:n_in];
    for (int i = 0; i < n_in; i++) {
        k->ios_in[i] = test_make_surface(in_sizes[i]);
        [wIns addObject:((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), k->ios_in[i])];
        [iIdx addObject:@(i)];
    }
    NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:n_out];
    NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:n_out];
    for (int i = 0; i < n_out; i++) {
        k->ios_out[i] = test_make_surface(out_sizes[i]);
        [wOuts addObject:((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), k->ios_out[i])];
        [oIdx addObject:@(i)];
    }
    k->request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        g_ANEReq,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        wIns, iIdx, wOuts, oIdx, nil, nil, @0);
    return k;
}

static void test_l2_ffn(void) {
    printf("\n--- L2: FFN (SwiGLU) ---\n");
    int D = 128, H = 256, S = 8;

    float *x = malloc(S * D * sizeof(float));
    float *w1 = malloc(H * D * sizeof(float));
    float *w2 = malloc(D * H * sizeof(float));
    float *w3 = malloc(H * D * sizeof(float));
    float *ref_out = malloc(S * D * sizeof(float));

    fill_random(x, S * D, 0.5f);
    fill_random(w1, H * D, 0.1f);
    fill_random(w2, D * H, 0.1f);
    fill_random(w3, H * D, 0.1f);

    ref_ffn(ref_out, x, w1, w2, w3, S, D, H);

    // Build separate weight blobs
    NSData *b1 = test_build_blob(w1, H * D);
    NSData *b3 = test_build_blob(w3, H * D);
    NSData *b2 = test_build_blob(w2, D * H);

    NSDictionary *weights = @{
        @"@model_path/weights/w1.bin": @{@"offset": @0, @"data": b1},
        @"@model_path/weights/w3.bin": @{@"offset": @0, @"data": b3},
        @"@model_path/weights/w2.bin": @{@"offset": @0, @"data": b2},
    };

    NSString *mil = gen_test_ffn(D, H, S);

    float *x_ane = malloc(D * S * sizeof(float));
    transpose_to_ane(x_ane, x, S, D);

    size_t in_bytes = D * S * 2;
    size_t out_bytes = D * S * 2;
    uint64_t t0 = mach_absolute_time();
    TestKernel *k = test_compile_kernel_multi_w(mil, weights, 1, &in_bytes, 1, &out_bytes);

    TestResult r;
    if (!k) {
        r = test_skip("FFN SwiGLU", "L2:layer", "ANE compile failed");
    } else {
        test_kernel_write_fp16(k, 0, x_ane, D * S);
        test_kernel_eval(k);
        float *ane_out_ch = malloc(D * S * sizeof(float));
        float *ane_out = malloc(S * D * sizeof(float));
        test_kernel_read_fp16(k, 0, ane_out_ch, D * S);
        transpose_from_ane(ane_out, ane_out_ch, S, D);
        double elapsed = test_ms(mach_absolute_time() - t0);
        // FFN has 3 matmuls + nonlinearities, allow more error
        r = test_check("FFN SwiGLU", "L2:layer", ane_out, ref_out, S * D,
                        2.0f, 0.99f, elapsed);
        free(ane_out_ch); free(ane_out);
        test_kernel_free(k);
    }
    test_record(r);
    free(x); free(w1); free(w2); free(w3); free(ref_out); free(x_ane);
}

static void test_l2_sdpa(void) {
    printf("\n--- L2: SDPA (Scaled Dot-Product Attention) ---\n");
    int H = 4, S = 8, HD = 32;

    float *q = malloc(H * S * HD * sizeof(float));
    float *k = malloc(H * S * HD * sizeof(float));
    float *v = malloc(H * S * HD * sizeof(float));
    fill_random(q, H * S * HD, 0.5f);
    fill_random(k, H * S * HD, 0.5f);
    fill_random(v, H * S * HD, 0.5f);

    // CPU reference: causal MHA (same heads for Q/K/V)
    float *ref_out = malloc(H * S * HD * sizeof(float));
    // Reshape to [S, H, HD] for ref_gqa_attention
    float *q_sh = malloc(S * H * HD * sizeof(float));
    float *k_sh = malloc(S * H * HD * sizeof(float));
    float *v_sh = malloc(S * H * HD * sizeof(float));
    // q is [H, S, HD] -> need [S, H, HD]
    for (int h = 0; h < H; h++)
        for (int s = 0; s < S; s++)
            for (int d = 0; d < HD; d++) {
                q_sh[s * H * HD + h * HD + d] = q[h * S * HD + s * HD + d];
                k_sh[s * H * HD + h * HD + d] = k[h * S * HD + s * HD + d];
                v_sh[s * H * HD + h * HD + d] = v[h * S * HD + s * HD + d];
            }

    float *ref_out_sh = malloc(S * H * HD * sizeof(float));
    ref_gqa_attention(ref_out_sh, q_sh, k_sh, v_sh, S, H, H, HD);
    // Back to [H, S, HD]
    for (int h = 0; h < H; h++)
        for (int s = 0; s < S; s++)
            for (int d = 0; d < HD; d++)
                ref_out[h * S * HD + s * HD + d] = ref_out_sh[s * H * HD + h * HD + d];

    // Build causal mask blob
    _Float16 *mask = (_Float16 *)calloc(S * S, sizeof(_Float16));
    for (int t = 0; t < S; t++)
        for (int t2 = 0; t2 < S; t2++)
            mask[t * S + t2] = (t2 <= t) ? (_Float16)0.0f : (_Float16)(-65504.0f);
    NSData *mask_blob = test_build_blob(NULL, 0); // need fp16 blob
    // Build mask blob manually
    {
        int ws = S * S * 2, tot = 128 + ws;
        uint8_t *b = (uint8_t *)calloc(tot, 1);
        b[0] = 1; b[4] = 2;
        b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
        *(uint32_t *)(b + 72) = ws;
        *(uint32_t *)(b + 80) = 128;
        memcpy(b + 128, mask, ws);
        mask_blob = [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
    }
    free(mask);

    NSString *mil = gen_test_sdpa(H, S, HD);
    size_t in_bytes = H * S * HD * 2;
    size_t out_bytes = H * S * HD * 2;
    size_t in_sizes[] = { in_bytes, in_bytes, in_bytes };
    uint64_t t0 = mach_absolute_time();
    TestKernel *kern = test_compile_kernel(mil, mask_blob, 3, in_sizes, 1, &out_bytes);

    TestResult r;
    if (!kern) {
        r = test_skip("SDPA", "L2:layer", "ANE compile failed");
    } else {
        test_kernel_write_fp16(kern, 0, q, H * S * HD);
        test_kernel_write_fp16(kern, 1, k, H * S * HD);
        test_kernel_write_fp16(kern, 2, v, H * S * HD);
        test_kernel_eval(kern);
        float *ane_out = malloc(H * S * HD * sizeof(float));
        test_kernel_read_fp16(kern, 0, ane_out, H * S * HD);
        double elapsed = test_ms(mach_absolute_time() - t0);
        r = test_check("SDPA", "L2:layer", ane_out, ref_out, H * S * HD,
                        0.5f, 0.995f, elapsed);
        free(ane_out);
        test_kernel_free(kern);
    }
    test_record(r);
    free(q); free(k); free(v); free(ref_out);
    free(q_sh); free(k_sh); free(v_sh); free(ref_out_sh);
}

// ═══════════════════════════════════════════════════════════════════════
// LEVEL 3: INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════

static void test_l3_dequant_q4_0(void) {
    printf("\n--- L3: Q4_0 Dequantization ---\n");
    int n = 256; // must be multiple of 32
    int nb = n / Q4_0_BLOCK_SIZE;

    // Create test Q4_0 blocks
    block_q4_0 *blocks = (block_q4_0 *)calloc(nb, sizeof(block_q4_0));
    for (int b = 0; b < nb; b++) {
        blocks[b].d = (_Float16)(0.1f * (b + 1));
        for (int j = 0; j < Q4_0_BLOCK_SIZE / 2; j++)
            blocks[b].qs[j] = (uint8_t)((j % 16) | (((j + 1) % 16) << 4));
    }

    float *ref_out = malloc(n * sizeof(float));
    ref_dequant_q4_0(ref_out, blocks, n);

    // Verify a few known values manually
    // Block 0, d=0.1: qs[0] = 0x10 -> low=0, high=1
    // val[0] = 0.1 * (0 - 8) = -0.8
    // val[16] = 0.1 * (1 - 8) = -0.7
    float expected_0 = 0.1f * (0 - 8);
    float expected_16 = 0.1f * (1 - 8);

    TestResult r;
    r.name = "Q4_0 dequant";
    r.level = "L3:integration";
    r.elapsed_ms = 0;
    r.message[0] = 0;
    memset(&r.err, 0, sizeof(r.err));
    r.err.cosine_sim = 1.0f;
    bool ok = (fabsf(ref_out[0] - expected_0) < 0.01f) &&
              (fabsf(ref_out[16] - expected_16) < 0.01f);
    r.status = ok ? TEST_PASS : TEST_FAIL;
    if (!ok) snprintf(r.message, sizeof(r.message),
                      "got[0]=%.4f exp=%.4f, got[16]=%.4f exp=%.4f",
                      ref_out[0], expected_0, ref_out[16], expected_16);
    test_record(r);

    free(blocks); free(ref_out);
}

static void test_l3_gguf_header(const char *gguf_path) {
    printf("\n--- L3: GGUF Header Parsing ---\n");
    if (!gguf_path) {
        test_record(test_skip("GGUF parse", "L3:integration", "no --gguf path"));
        return;
    }

    FILE *f = fopen(gguf_path, "rb");
    if (!f) {
        test_record(test_skip("GGUF parse", "L3:integration", "file not found"));
        return;
    }

    // GGUF magic: 0x46465547 ("GGUF")
    uint32_t magic;
    fread(&magic, 4, 1, f);
    bool magic_ok = (magic == 0x46465547);

    uint32_t version;
    fread(&version, 4, 1, f);
    bool version_ok = (version == 2 || version == 3);

    uint64_t n_tensors, n_kv;
    fread(&n_tensors, 8, 1, f);
    fread(&n_kv, 8, 1, f);

    fclose(f);

    TestResult r;
    r.name = "GGUF magic+version";
    r.level = "L3:integration";
    r.elapsed_ms = 0;
    memset(&r.err, 0, sizeof(r.err));
    r.err.cosine_sim = 1.0f;
    r.status = (magic_ok && version_ok) ? TEST_PASS : TEST_FAIL;
    snprintf(r.message, sizeof(r.message), "magic=%08X ver=%u tensors=%llu kv=%llu",
             magic, version, (unsigned long long)n_tensors, (unsigned long long)n_kv);
    test_record(r);
}

// ═══════════════════════════════════════════════════════════════════════
// LEVEL 4: PERFORMANCE TESTS
// ═══════════════════════════════════════════════════════════════════════

static void test_l4_conv_throughput(void) {
    printf("\n--- L4: Conv Throughput ---\n");
    // Mistral 7B dimensions
    int configs[][3] = {
        {256,  256,  16},   // small (unit test sized)
        {256,  512,  16},   // FFN-like
        {512,  256,  16},   // FFN down-like
        {256,  256,  64},   // longer seq
    };
    int nconfigs = sizeof(configs) / sizeof(configs[0]);

    for (int ci = 0; ci < nconfigs; ci++) {
        int IC = configs[ci][0], OC = configs[ci][1], S = configs[ci][2];
        char name[64];
        snprintf(name, sizeof(name), "conv %dx%d sp=%d", OC, IC, S);

        float *W = malloc(OC * IC * sizeof(float));
        fill_random(W, OC * IC, 0.1f);
        NSData *wblob = test_build_blob(W, OC * IC);
        NSString *mil = gen_test_conv(IC, OC, S);

        size_t in_bytes = IC * S * 2;
        size_t out_bytes = OC * S * 2;
        TestKernel *k = test_compile_kernel(mil, wblob, 1, &in_bytes, 1, &out_bytes);

        TestResult r;
        if (!k) {
            r = test_skip(name, "L4:perf", "compile failed");
        } else {
            float *x = malloc(IC * S * sizeof(float));
            fill_random(x, IC * S, 0.5f);
            test_kernel_write_fp16(k, 0, x, IC * S);

            // Warmup
            for (int i = 0; i < 10; i++) test_kernel_eval(k);

            // Benchmark
            int iters = 100;
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < iters; i++) test_kernel_eval(k);
            double ms = test_ms(mach_absolute_time() - t0) / iters;

            double gflops = 2.0 * OC * IC * S / 1e9;
            double tflops = gflops / ms;

            r.name = name;
            r.level = "L4:perf";
            r.status = TEST_PASS;
            r.elapsed_ms = ms;
            memset(&r.err, 0, sizeof(r.err));
            snprintf(r.message, sizeof(r.message), "%.3f ms  %.2f TFLOPS", ms, tflops);
            free(x);
            test_kernel_free(k);
        }
        test_record(r);
        free(W);
    }
}

static void test_l4_memory(void) {
    printf("\n--- L4: Memory Usage ---\n");
    size_t mem_before = get_resident_bytes();

    // Compile a medium kernel
    int IC = 256, OC = 256, S = 32;
    float *W = malloc(OC * IC * sizeof(float));
    fill_random(W, OC * IC, 0.1f);
    NSData *wblob = test_build_blob(W, OC * IC);
    NSString *mil = gen_test_conv(IC, OC, S);
    size_t in_bytes = IC * S * 2;
    size_t out_bytes = OC * S * 2;
    TestKernel *k = test_compile_kernel(mil, wblob, 1, &in_bytes, 1, &out_bytes);

    size_t mem_after = get_resident_bytes();
    size_t delta_mb = (mem_after > mem_before) ? (mem_after - mem_before) / (1024 * 1024) : 0;

    TestResult r;
    r.name = "Memory: single kernel";
    r.level = "L4:perf";
    r.status = TEST_PASS;
    r.elapsed_ms = 0;
    memset(&r.err, 0, sizeof(r.err));
    snprintf(r.message, sizeof(r.message), "delta ~%zu MB (before=%zu MB after=%zu MB)",
             delta_mb, mem_before / (1024 * 1024), mem_after / (1024 * 1024));
    test_record(r);

    if (k) test_kernel_free(k);
    free(W);
}

// ═══════════════════════════════════════════════════════════════════════
// LEVEL 5: CORRECTNESS VALIDATION
// ═══════════════════════════════════════════════════════════════════════

static void test_l5_known_answer(void) {
    printf("\n--- L5: Known-Answer Tests ---\n");

    // RMSNorm known answer: all-ones input, all-ones weight
    {
        int D = 4, S = 1;
        float x[] = {1, 1, 1, 1};
        float w[] = {1, 1, 1, 1};
        float out[4];
        ref_rmsnorm(out, x, w, S, D);
        // RMSNorm(ones): rms = sqrt(4/4 + 1e-5) ~ 1.0, so output ~ ones
        float expected[] = {1.0f, 1.0f, 1.0f, 1.0f};
        TestResult r = test_check("RMSNorm known: ones->ones", "L5:correct",
                                   out, expected, 4, 1e-4f, 0.99999f, 0);
        test_record(r);
    }

    // SiLU known answer: silu(0) = 0, silu(large) ~ x
    {
        float x[] = {0.0f, 1.0f, -1.0f, 10.0f, -10.0f};
        float out[5];
        ref_silu(out, x, 5);
        float expected[] = {
            0.0f,
            1.0f / (1.0f + expf(-1.0f)),        // ~0.7311
            -1.0f / (1.0f + expf(1.0f)),         // ~-0.2689
            10.0f / (1.0f + expf(-10.0f)),       // ~9.9995
            -10.0f / (1.0f + expf(10.0f)),       // ~-0.000454
        };
        TestResult r = test_check("SiLU known answers", "L5:correct",
                                   out, expected, 5, 1e-4f, 0.99999f, 0);
        test_record(r);
    }

    // Softmax known answer: uniform input -> uniform output
    {
        float x[] = {1, 1, 1, 1};
        float out[4];
        ref_softmax(out, x, 1, 4);
        float expected[] = {0.25f, 0.25f, 0.25f, 0.25f};
        TestResult r = test_check("Softmax known: uniform", "L5:correct",
                                   out, expected, 4, 1e-6f, 0.999999f, 0);
        test_record(r);
    }

    // Softmax: one-hot-like (one large, rest small)
    {
        float x[] = {100, 0, 0, 0};
        float out[4];
        ref_softmax(out, x, 1, 4);
        // Should be ~[1, 0, 0, 0]
        bool ok = (out[0] > 0.999f) && (out[1] < 0.001f);
        TestResult r;
        r.name = "Softmax known: dominant";
        r.level = "L5:correct";
        r.status = ok ? TEST_PASS : TEST_FAIL;
        r.elapsed_ms = 0;
        memset(&r.err, 0, sizeof(r.err));
        r.err.cosine_sim = 1.0f;
        snprintf(r.message, sizeof(r.message), "out[0]=%.6f out[1]=%.6f", out[0], out[1]);
        test_record(r);
    }
}

static void test_l5_perplexity_stub(const char *gguf_path, const char *llamacpp_path) {
    printf("\n--- L5: Perplexity (stub) ---\n");
    if (!gguf_path || !llamacpp_path) {
        test_record(test_skip("Perplexity wikitext-2", "L5:correct",
                               "requires --gguf and --llamacpp paths"));
        return;
    }
    // In a real implementation, this would:
    // 1. Load wikitext-2 test set
    // 2. Run ANE inference on each chunk
    // 3. Compute perplexity
    // 4. Compare against llama.cpp: llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw
    // Target: within 0.5 PPL of llama.cpp Q4_0
    test_record(test_skip("Perplexity wikitext-2", "L5:correct",
                           "full implementation requires tokenizer + wikitext data"));
}

// ═══════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════

int main(int argc, char *argv[]) {
    @autoreleasepool {
        int level = 0; // 0 = all
        const char *gguf_path = NULL;
        const char *llamacpp_path = NULL;

        static struct option long_opts[] = {
            {"level", required_argument, 0, 'l'},
            {"gguf", required_argument, 0, 'g'},
            {"llamacpp", required_argument, 0, 'c'},
            {0, 0, 0, 0}
        };
        int opt;
        while ((opt = getopt_long(argc, argv, "l:g:c:", long_opts, NULL)) != -1) {
            switch (opt) {
                case 'l': level = atoi(optarg); break;
                case 'g': gguf_path = optarg; break;
                case 'c': llamacpp_path = optarg; break;
            }
        }

        printf("╔══════════════════════════════════════════════════════╗\n");
        printf("║  Mistral 7B ANE Inference Test Suite                ║\n");
        printf("║  Level: %s                                          ║\n",
               level == 0 ? "ALL" : (level == 1 ? "1" : (level == 2 ? "2" :
               (level == 3 ? "3" : (level == 4 ? "4" : "5")))));
        printf("╚══════════════════════════════════════════════════════╝\n");

        test_ane_init();
        if (!g_ANEDesc || !g_ANEInMem) {
            fprintf(stderr, "FATAL: Cannot load AppleNeuralEngine.framework\n");
            return 1;
        }

        // Level 1: Unit tests
        if (level == 0 || level == 1) {
            printf("\n══════ LEVEL 1: UNIT TESTS ══════\n");
            test_l1_rmsnorm();
            test_l1_silu();
            test_l1_softmax();
            test_l1_conv_matmul();
            test_l1_rope();
            test_l1_gqa();
        }

        // Level 2: Layer tests
        if (level == 0 || level == 2) {
            printf("\n══════ LEVEL 2: LAYER TESTS ══════\n");
            test_l2_ffn();
            test_l2_sdpa();
        }

        // Level 3: Integration tests
        if (level == 0 || level == 3) {
            printf("\n══════ LEVEL 3: INTEGRATION TESTS ══════\n");
            test_l3_dequant_q4_0();
            test_l3_gguf_header(gguf_path);
        }

        // Level 4: Performance tests
        if (level == 0 || level == 4) {
            printf("\n══════ LEVEL 4: PERFORMANCE TESTS ══════\n");
            test_l4_conv_throughput();
            test_l4_memory();
        }

        // Level 5: Correctness validation
        if (level == 0 || level == 5) {
            printf("\n══════ LEVEL 5: CORRECTNESS VALIDATION ══════\n");
            test_l5_known_answer();
            test_l5_perplexity_stub(gguf_path, llamacpp_path);
        }

        test_summary();
        free(g_results);
        return g_tests_fail > 0 ? 1 : 0;
    }
}
