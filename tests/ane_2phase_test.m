// ane_2phase_test.m — Validate ANE 2-phase loading strategy
// Phase 1: Compile programs + save hexIds
// Phase 2: Create dummy model objects, forge hexIds, load from daemon cache
//
// Build: cd mistral && xcrun clang -O2 -ffast-math -fobjc-arc -Wall -Wno-unused-function
//   -Wno-unused-variable -mcpu=apple-m4 -I. -o ane_2phase_test ../tests/ane_2phase_test.m
//   -framework Foundation -framework IOSurface -framework Accelerate -ldl

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <mach/mach_time.h>
#include <stdlib.h>
#include <string.h>
#include "../training/ane_runtime.h"
#include "ane_mil_gen_mistral.h"

static double ms_since(uint64_t start) {
    static mach_timebase_info_data_t tb = {0};
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)(mach_absolute_time() - start) * tb.numer / tb.denom / 1e6;
}

// Fill buffer with random fp16 values in [-0.1, 0.1]
static void fill_random_fp16(_Float16 *buf, size_t n) {
    for (size_t i = 0; i < n; i++)
        buf[i] = (_Float16)((drand48() - 0.5) * 0.2);
}

// Fill buffer with random fp32 values in [-0.1, 0.1]
static void fill_random_fp32(float *buf, size_t n) {
    for (size_t i = 0; i < n; i++)
        buf[i] = (float)((drand48() - 0.5) * 0.2);
}

// ─── Test 1: compile + get_hexid ────────────────────────────────────────────

static int test_compile_get_hexid(NSString **out_hexId, NSData **out_milText, NSData **out_blob) {
    printf("Test 1: compile+get_hexid ... ");
    fflush(stdout);

    int in_ch = 256, out_ch = 256, S = 16;

    // Generate MIL
    NSString *milStr = mil_gen_conv_baked(in_ch, out_ch, S);
    NSData *milText = [milStr dataUsingEncoding:NSUTF8StringEncoding];

    // Generate random weights and build blob
    size_t nw = (size_t)out_ch * in_ch;
    _Float16 *w = malloc(nw * sizeof(_Float16));
    fill_random_fp16(w, nw);
    NSData *blob = mil_build_single_weight_blob(w, out_ch, in_ch);
    free(w);

    // Compile and get hexId
    NSString *hexId = ane_compile_and_get_hexid(milText, blob);
    if (!hexId) {
        printf("FAIL (hexId is nil)\n");
        return 1;
    }
    if ([hexId length] < 16) {
        printf("FAIL (hexId too short: %lu chars)\n", (unsigned long)[hexId length]);
        return 1;
    }

    printf("PASS (hexId=%s...)\n", [[hexId substringToIndex:16] UTF8String]);

    *out_hexId = hexId;
    *out_milText = milText;
    *out_blob = blob;
    return 0;
}

// ─── Test 2: forged load + compare outputs ──────────────────────────────────

static int test_forged_load(NSString *hexId, NSData *milText, NSData *blob) {
    printf("Test 2: forged load ... ");
    fflush(stdout);

    int in_ch = 256, out_ch = 256, S = 16;
    size_t inBytes = (size_t)1 * in_ch * 1 * S * sizeof(float);
    size_t outBytes = (size_t)1 * out_ch * 1 * S * sizeof(float);

    // Compile normally for reference
    size_t inSz[] = {inBytes};
    size_t outSz[] = {outBytes};
    ANEKernel *ref = ane_compile_cached(milText, blob, 1, inSz, 1, outSz);
    if (!ref) {
        printf("FAIL (normal compile failed)\n");
        return 1;
    }

    // Prepare input
    float *input = malloc(inBytes);
    fill_random_fp32(input, in_ch * S);

    // Run reference
    ane_write_input(ref, 0, input, inBytes);
    if (!ane_eval(ref)) {
        printf("FAIL (reference eval failed)\n");
        ane_free(ref);
        free(input);
        return 1;
    }
    float *ref_out = malloc(outBytes);
    ane_read_output(ref, 0, ref_out, outBytes);
    ane_free_ex(ref, true); // keep tmpDir for cache

    // Forged load
    ANEKernel *forged = ane_load_forged(milText, hexId, 1, inSz, 1, outSz);
    if (!forged) {
        printf("FAIL (forged load returned NULL)\n");
        free(input);
        free(ref_out);
        return 1;
    }

    // Run forged with same input
    ane_write_input(forged, 0, input, inBytes);
    if (!ane_eval(forged)) {
        printf("FAIL (forged eval failed)\n");
        ane_free(forged);
        free(input);
        free(ref_out);
        return 1;
    }
    float *forged_out = malloc(outBytes);
    ane_read_output(forged, 0, forged_out, outBytes);

    // Compare
    float max_diff = 0;
    size_t n = outBytes / sizeof(float);
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(ref_out[i] - forged_out[i]);
        if (d > max_diff) max_diff = d;
    }

    ane_free(forged);
    free(input);
    free(ref_out);
    free(forged_out);

    if (max_diff > 0.001f) {
        printf("FAIL (max_diff=%.4f)\n", max_diff);
        return 1;
    }
    printf("PASS (max_diff=%.4f)\n", max_diff);
    return 0;
}

// ─── Test 3: manifest save/load ─────────────────────────────────────────────

static int test_manifest_roundtrip(NSString *hexId) {
    printf("Test 3: manifest save/load ... ");
    fflush(stdout);

    NSString *path = @"/tmp/ane_test_manifest.plist";
    NSDictionary *manifest = @{
        @"layer0_Q":   hexId,
        @"layer0_KV":  @"DEADBEEF01234567DEADBEEF01234567",
        @"layer0_Wo":  @"CAFEBABE01234567CAFEBABE01234567",
        @"layer0_FFN": @"0123456789ABCDEF0123456789ABCDEF",
    };

    NSError *err = nil;
    NSData *plistData = [NSPropertyListSerialization dataWithPropertyList:manifest
        format:NSPropertyListXMLFormat_v1_0 options:0 error:&err];
    if (!plistData) {
        printf("FAIL (serialize: %s)\n", [[err description] UTF8String]);
        return 1;
    }
    [plistData writeToFile:path atomically:YES];

    // Load back
    NSData *loaded = [NSData dataWithContentsOfFile:path];
    NSDictionary *restored = [NSPropertyListSerialization propertyListWithData:loaded
        options:NSPropertyListImmutable format:nil error:&err];
    if (!restored) {
        printf("FAIL (deserialize: %s)\n", [[err description] UTF8String]);
        return 1;
    }

    // Verify all keys round-trip
    for (NSString *key in manifest) {
        if (![manifest[key] isEqualToString:restored[key]]) {
            printf("FAIL (key %s mismatch)\n", [key UTF8String]);
            return 1;
        }
    }

    // Cleanup
    [[NSFileManager defaultManager] removeItemAtPath:path error:nil];

    printf("PASS\n");
    return 0;
}

// ─── Test 4: scale test (8 programs, 2 layers x 4 programs) ────────────────

typedef struct {
    NSString *name;
    int in_ch, out_ch, S;
    bool is_ffn;
    int ffn_hidden;
} ProgramSpec;

static int test_scale(void) {
    printf("Test 4: scale test (8 programs) ...\n");
    fflush(stdout);

    int S = 16;
    // 2 layers x 4 programs: Q(256x256), KV(64x256 fused -> single conv for simplicity),
    // Wo(256x256), FFN(hidden=128, dim=256)
    ProgramSpec specs[8] = {
        {@"L0_Q",   256, 256, S, false, 0},
        {@"L0_KV",  256,  64, S, false, 0},
        {@"L0_Wo",  256, 256, S, false, 0},
        {@"L0_FFN", 256, 256, S, true, 128},
        {@"L1_Q",   256, 256, S, false, 0},
        {@"L1_KV",  256,  64, S, false, 0},
        {@"L1_Wo",  256, 256, S, false, 0},
        {@"L1_FFN", 256, 256, S, true, 128},
    };

    NSMutableArray<NSString*> *hexIds = [NSMutableArray arrayWithCapacity:8];
    NSMutableArray<NSData*> *milTexts = [NSMutableArray arrayWithCapacity:8];

    // Phase 1: compile all + get hexIds
    uint64_t t0 = mach_absolute_time();

    for (int i = 0; i < 8; i++) {
        ProgramSpec *sp = &specs[i];
        NSString *milStr;
        NSData *blob;

        if (sp->is_ffn) {
            milStr = mil_gen_ffn_fused(sp->in_ch, sp->ffn_hidden, sp->S);
            // Build FFN blob: W1[hidden,dim] + W3[hidden,dim] + W2[dim,hidden]
            size_t nup = (size_t)sp->ffn_hidden * sp->in_ch;
            size_t ndn = (size_t)sp->in_ch * sp->ffn_hidden;
            _Float16 *w1 = malloc(nup * 2); fill_random_fp16(w1, nup);
            _Float16 *w3 = malloc(nup * 2); fill_random_fp16(w3, nup);
            _Float16 *w2 = malloc(ndn * 2); fill_random_fp16(w2, ndn);
            blob = mil_build_ffn_fused_blob(w1, w3, sp->ffn_hidden, w2, sp->in_ch);
            free(w1); free(w3); free(w2);
        } else {
            milStr = mil_gen_conv_baked(sp->in_ch, sp->out_ch, sp->S);
            size_t nw = (size_t)sp->out_ch * sp->in_ch;
            _Float16 *w = malloc(nw * 2); fill_random_fp16(w, nw);
            blob = mil_build_single_weight_blob(w, sp->out_ch, sp->in_ch);
            free(w);
        }

        NSData *milText = [milStr dataUsingEncoding:NSUTF8StringEncoding];
        NSString *hexId = ane_compile_and_get_hexid(milText, blob);
        if (!hexId) {
            printf("  FAIL (compile failed for %s)\n", [sp->name UTF8String]);
            return 1;
        }
        [hexIds addObject:hexId];
        [milTexts addObject:milText];
    }

    double phase1_ms = ms_since(t0);
    printf("  Phase 1 (compile): %.0fms\n", phase1_ms);

    // Phase 2: forged load all 8
    uint64_t t1 = mach_absolute_time();

    for (int i = 0; i < 8; i++) {
        ProgramSpec *sp = &specs[i];
        size_t inBytes, outBytes;

        if (sp->is_ffn) {
            inBytes = (size_t)sp->in_ch * sp->S * sizeof(float);
            outBytes = (size_t)sp->in_ch * sp->S * sizeof(float);
        } else {
            inBytes = (size_t)sp->in_ch * sp->S * sizeof(float);
            outBytes = (size_t)sp->out_ch * sp->S * sizeof(float);
        }

        size_t inSz[] = {inBytes};
        size_t outSz[] = {outBytes};
        ANEKernel *k = ane_load_forged(milTexts[i], hexIds[i], 1, inSz, 1, outSz);
        if (!k) {
            printf("  FAIL (forged load failed for %s)\n", [sp->name UTF8String]);
            return 1;
        }
        ane_free_ex(k, true);
    }

    double phase2_ms = ms_since(t1);
    printf("  Phase 2 (forged load): %.0fms\n", phase2_ms);
    printf("  PASS\n");
    return 0;
}

// ─── Test 5: cache miss handling ────────────────────────────────────────────

static int test_cache_miss(void) {
    printf("Test 5: cache miss ... ");
    fflush(stdout);

    int in_ch = 256, out_ch = 256, S = 16;
    NSString *milStr = mil_gen_conv_baked(in_ch, out_ch, S);
    NSData *milText = [milStr dataUsingEncoding:NSUTF8StringEncoding];

    // Fake hexId — should not exist in daemon cache
    NSString *fakeHexId = @"AABBCCDD11223344AABBCCDD11223344AABBCCDD11223344AABBCCDD11223344";

    size_t inBytes = (size_t)in_ch * S * sizeof(float);
    size_t outBytes = (size_t)out_ch * S * sizeof(float);
    size_t inSz[] = {inBytes};
    size_t outSz[] = {outBytes};

    ANEKernel *k = ane_load_forged(milText, fakeHexId, 1, inSz, 1, outSz);
    if (k != NULL) {
        printf("FAIL (expected NULL, got kernel)\n");
        ane_free(k);
        return 1;
    }

    printf("PASS (NULL returned)\n");
    return 0;
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        srand48(42);
        printf("=== ANE 2-Phase Test ===\n");

        int fails = 0;
        NSString *hexId = nil;
        NSData *milText = nil;
        NSData *blob = nil;

        fails += test_compile_get_hexid(&hexId, &milText, &blob);
        if (!fails) fails += test_forged_load(hexId, milText, blob);
        fails += test_manifest_roundtrip(hexId ?: @"placeholder");
        fails += test_scale();
        fails += test_cache_miss();

        printf("\n%s (%d/5 passed)\n", fails ? "SOME TESTS FAILED" : "ALL TESTS PASSED", 5 - fails);
        return fails;
    }
}
