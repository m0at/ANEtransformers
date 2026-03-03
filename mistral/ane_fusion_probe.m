// ane_fusion_probe.m — Test fused K+V (2 outputs, same dim) and Wo+RMSNorm+FFN mega fusion
// Goal: reduce from 5 programs/layer to 3, getting 32×3=96 programs (under 128 limit)
//
// Tests:
//   1. Fused K+V: 2 parallel conv1x1 [4096→1024], same input, 2 outputs
//   2. Single conv with 2 identical outputs (control — verify multi-output works at all)
//   3. Wo+RMSNorm+FFN: conv + reduce_mean + rsqrt + mul + (3 convs + sigmoid + mul) + conv
//
// Build: clang -O2 -framework Foundation -framework IOSurface -framework CoreML ane_fusion_probe.m -o ane_fusion_probe

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include "../training/ane_runtime.h"
#include "ane_mil_gen_mistral.h"
#include <math.h>
#include <mach/mach_time.h>

static double probe_time_ms(void) {
    static mach_timebase_info_data_t tbi = {0};
    if (tbi.denom == 0) mach_timebase_info(&tbi);
    return (double)mach_absolute_time() * tbi.numer / tbi.denom / 1e6;
}

// ─── MIL generators ─────────────────────────────────────────────────────────

// Fused K+V: 2 parallel convs, same shape [kv_dim, dim], shared input, 2 outputs
static NSString *mil_gen_kv_fused(int dim, int kv_dim, int S) {
    NSUInteger wkv_size = (NSUInteger)kv_dim * dim * 2;  // fp16 bytes per weight
    NSUInteger cs_kv = 64 + wkv_size;                    // chunk size (hdr + data)

    // Blob layout: [0:64) global hdr, [64: 64+cs_kv) Wk chunk, [64+cs_kv: ...) Wv chunk
    NSUInteger off_wk = 64;
    NSUInteger off_wv = 64 + cs_kv;

    return [NSString stringWithFormat:
        MIL_HEADER
        @"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        MIL_CONV_CONSTS
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        // Wk
        "        tensor<fp16, [%d, %d, 1, 1]> Wk = const()[name = string(\"Wk\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        // Wv
        "        tensor<fp16, [%d, %d, 1, 1]> Wv = const()[name = string(\"Wv\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        // Convolutions
        "        tensor<fp16, [1, %d, 1, %d]> k16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wk, x = x16)[name = string(\"conv_k\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> v16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wv, x = x16)[name = string(\"conv_v\")];\n"
        // Cast outputs
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> k = cast(dtype = to_fp32, x = k16)[name = string(\"cast_k\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> v = cast(dtype = to_fp32, x = v16)[name = string(\"cast_v\")];\n"
        "    } -> (k, v);\n}\n",
        dim, S, dim, S,
        // Wk
        kv_dim, dim, kv_dim, dim, (unsigned long)off_wk,
        // Wv
        kv_dim, dim, kv_dim, dim, (unsigned long)off_wv,
        // conv outputs
        kv_dim, S, kv_dim, S,
        // cast outputs
        kv_dim, S, kv_dim, S];
}

// Build K+V blob: Wk[kv_dim, dim] + Wv[kv_dim, dim]
static NSData *mil_build_kv_fused_blob(const _Float16 *wk, const _Float16 *wv,
                                        int kv_dim, int dim) {
    NSUInteger wkv_size = (NSUInteger)kv_dim * dim * 2;
    NSUInteger cs_kv = 64 + wkv_size;
    NSUInteger total = 64 + cs_kv + cs_kv;

    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;

    // Wk chunk at offset 64
    write_chunk_header(buf + 64, (uint32_t)wkv_size, (uint32_t)(64 + 64));
    memcpy(buf + 64 + 64, wk, wkv_size);

    // Wv chunk at offset 64 + cs_kv
    NSUInteger wv_chunk = 64 + cs_kv;
    write_chunk_header(buf + wv_chunk, (uint32_t)wkv_size, (uint32_t)(wv_chunk + 64));
    memcpy(buf + wv_chunk + 64, wv, wkv_size);

    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// ─── Wo + RMSNorm + FFN mega fusion ──────────────────────────────────────────
// Input:  [1, dim, 1, S] fp32 (attention output)
// Also needs: rms_ffn weights [dim] baked as bias
// Output: [1, dim, 1, S] fp32 (FFN output, before residual)
//
// MIL graph:
//   x16 = cast(x, fp16)
//   wo_out = conv(x16, Wo)             [1, dim, 1, S]   — Wo projection
//   // RMSNorm: norm = x * rsqrt(mean(x^2) + eps)
//   sq = mul(wo_out, wo_out)            [1, dim, 1, S]
//   // reduce_mean over channel axis (axis=1)
//   ms = reduce_mean(sq, axes=[1])      [1, 1, 1, S]
//   eps_c = const(1e-5)
//   ms_eps = add(ms, eps_c)             [1, 1, 1, S]
//   inv = rsqrt(ms_eps)                 [1, 1, 1, S]
//   normed = mul(wo_out, inv)           [1, dim, 1, S]  — broadcast
//   // rms_weight * normed (element-wise per channel)
//   // Bake rms_weight as [dim, 1, 1, 1] and use mul with broadcast
//   rms_w = const(...)                  [1, dim, 1, 1]
//   scaled = mul(normed, rms_w)         [1, dim, 1, S]
//   // FFN
//   gate = conv(scaled, W1)             [1, hidden, 1, S]
//   up = conv(scaled, W3)               [1, hidden, 1, S]
//   sig = sigmoid(gate)
//   silu = mul(gate, sig)
//   h = mul(silu, up)
//   out16 = conv(h, W2)                 [1, dim, 1, S]
//   out = cast(out16, fp32)
//
// NOTE: The residual add (X + wo_out) must happen OUTSIDE since we need the
// un-normed wo_out for the residual. So we actually need 2 outputs:
//   1. wo_out (for residual before RMSNorm)
//   2. ffn_out (final FFN output)
//
// Wait — that defeats the purpose. Let's think about this differently.
// The caller has: attn_output (in fp32). It needs:
//   1. wo_proj = Wo @ attn_output
//   2. X += wo_proj  (residual)
//   3. normed = rmsnorm(X)
//   4. ffn_out = FFN(normed)
//   5. X += ffn_out  (residual)
//
// We can't fuse step 2 because we need X (the running hidden state) which is NOT
// part of this program's input. The program only gets attn_output.
//
// Alternative: input BOTH attn_output AND X (residual stream).
// But ANE can't take runtime inputs for matmul... wait, these are element-wise ops.
// The matmul (Wo conv) only uses attn_output. The residual add and RMSNorm use the
// result + X. So we need 2 inputs:
//   Input 0: attn_output [1, dim, 1, S]
//   Input 1: X (residual) [1, dim, 1, S]
//
// But does ANE support 2 runtime inputs? The API supports it (nInputs=2).
// The issue was that runtime inputs can't be used as MATMUL operands.
// But Input 1 (X) is only used for add (element-wise), not matmul.
// The matmul (Wo conv) uses Input 0 with BAKED weights. Should work!
//
// MIL graph:
//   attn16 = cast(attn_out, fp16)
//   x16 = cast(x_residual, fp16)
//   wo_out = conv(attn16, Wo)           [1, dim, 1, S]
//   residual = add(wo_out, x16)         [1, dim, 1, S]  — residual connection
//   sq = mul(residual, residual)
//   ms = reduce_mean(sq, axes=[1], keep_dims=true)
//   eps_c = const(1e-5)
//   ms_eps = add(ms, eps_c)
//   inv = rsqrt(ms_eps)
//   normed = mul(residual, inv)         — broadcast inv [1,1,1,S] × residual [1,dim,1,S]
//   rms_w = const(...)                  [1, dim, 1, 1] baked
//   scaled = mul(normed, rms_w)
//   gate = conv(scaled, W1)
//   up = conv(scaled, W3)
//   sig = sigmoid(gate)
//   silu = mul(gate, sig)
//   h = mul(silu, up)
//   ffn_out16 = conv(h, W2)
//   ffn_out = cast(ffn_out16, fp32)
//   residual_out = cast(residual, fp32)  — also output the post-Wo residual
//
// Outputs: residual_out (new X = old_X + wo_proj), ffn_out (to add to X later)
// OR: single output = residual + ffn_out (both residuals done inside ANE!)
//
// YES! We can do BOTH residuals inside:
//   final = add(residual, ffn_out16)    [1, dim, 1, S]
//   out = cast(final, fp32)
// Single input (attn_output), single output (new X with both residuals applied).
//
// Wait, but then we need X (old residual) as input too. 2 inputs, 1 output.

static NSString *mil_gen_wo_rmsnorm_ffn(int dim, int hidden, int S, float rms_eps) {
    // Weight blob layout:
    // [0:64)                                    global header
    // [64: 64+cs_wo)                            Wo chunk: 64B hdr + dim*dim*2 data
    // [64+cs_wo: 64+cs_wo+cs_rms)               rms_w chunk: 64B hdr + dim*2 data
    // [64+cs_wo+cs_rms: +cs_up)                 W1 chunk: 64B hdr + hidden*dim*2 data
    // [+cs_up: +2*cs_up)                        W3 chunk
    // [+2*cs_up: +2*cs_up+cs_dn)                W2 chunk: 64B hdr + dim*hidden*2 data
    NSUInteger wo_size  = (NSUInteger)dim * dim * 2;
    NSUInteger rms_size = (NSUInteger)dim * 2;
    NSUInteger w_up_size = (NSUInteger)hidden * dim * 2;
    NSUInteger w_dn_size = (NSUInteger)dim * hidden * 2;

    NSUInteger cs_wo  = 64 + wo_size;
    NSUInteger cs_rms = 64 + rms_size;
    NSUInteger cs_up  = 64 + w_up_size;
    NSUInteger cs_dn  = 64 + w_dn_size;

    NSUInteger off_wo  = 64;
    NSUInteger off_rms = 64 + cs_wo;
    NSUInteger off_w1  = 64 + cs_wo + cs_rms;
    NSUInteger off_w3  = 64 + cs_wo + cs_rms + cs_up;
    NSUInteger off_w2  = 64 + cs_wo + cs_rms + 2*cs_up;

    return [NSString stringWithFormat:
        MIL_HEADER
        @"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> attn_out, tensor<fp32, [1, %d, 1, %d]> x_res) {\n"
        MIL_CONV_CONSTS
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> attn16 = cast(dtype = to_fp16, x = attn_out)[name = string(\"cast_attn\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x_res)[name = string(\"cast_xres\")];\n"
        // Wo conv
        "        tensor<fp16, [%d, %d, 1, 1]> Wo = const()[name = string(\"Wo\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> wo_out = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wo, x = attn16)[name = string(\"conv_wo\")];\n"
        // Residual 1: X + wo_out
        "        tensor<fp16, [1, %d, 1, %d]> res1 = add(x = wo_out, y = x16)[name = string(\"residual1\")];\n"
        // RMSNorm: norm = x * rsqrt(mean(x^2) + eps)
        "        tensor<fp16, [1, %d, 1, %d]> sq = mul(x = res1, y = res1)[name = string(\"square\")];\n"
        "        tensor<fp16, [1, 1, 1, %d]> ms = reduce_mean(x = sq, axes = [1], keep_dims = true)[name = string(\"mean_sq\")];\n"
        "        tensor<fp16, [1, 1, 1, 1]> eps_val = const()[name = string(\"eps_val\"), val = tensor<fp16, [1, 1, 1, 1]>([%e])];\n"
        "        tensor<fp16, [1, 1, 1, %d]> ms_eps = add(x = ms, y = eps_val)[name = string(\"add_eps\")];\n"
        "        tensor<fp16, [1, 1, 1, %d]> inv = rsqrt(x = ms_eps)[name = string(\"rsqrt\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> normed = mul(x = res1, y = inv)[name = string(\"normed\")];\n"
        // RMS weight (baked, [1, dim, 1, 1] for channel-wise broadcast)
        "        tensor<fp16, [1, %d, 1, 1]> rms_w = const()[name = string(\"rms_w\"), "
        "val = tensor<fp16, [1, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> scaled = mul(x = normed, y = rms_w)[name = string(\"rms_scaled\")];\n"
        // W1 (gate)
        "        tensor<fp16, [%d, %d, 1, 1]> W1 = const()[name = string(\"W1\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        // W3 (up)
        "        tensor<fp16, [%d, %d, 1, 1]> W3 = const()[name = string(\"W3\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        // W2 (down)
        "        tensor<fp16, [%d, %d, 1, 1]> W2 = const()[name = string(\"W2\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        // Gate + Up convs
        "        tensor<fp16, [1, %d, 1, %d]> gate = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W1, x = scaled)[name = string(\"conv_gate\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> up = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W3, x = scaled)[name = string(\"conv_up\")];\n"
        // SiLU
        "        tensor<fp16, [1, %d, 1, %d]> sig = sigmoid(x = gate)[name = string(\"sigmoid\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> silu = mul(x = gate, y = sig)[name = string(\"silu\")];\n"
        // Gated
        "        tensor<fp16, [1, %d, 1, %d]> h = mul(x = silu, y = up)[name = string(\"gated\")];\n"
        // Down projection
        "        tensor<fp16, [1, %d, 1, %d]> ffn_out = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W2, x = h)[name = string(\"conv_down\")];\n"
        // Residual 2: res1 + ffn_out
        "        tensor<fp16, [1, %d, 1, %d]> final16 = add(x = res1, y = ffn_out)[name = string(\"residual2\")];\n"
        // Cast output
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> out = cast(dtype = to_fp32, x = final16)[name = string(\"cast_out\")];\n"
        "    } -> (out);\n}\n",
        // func signature: 2 inputs
        dim, S, dim, S,
        // cast inputs
        dim, S, dim, S,
        // Wo
        dim, dim, dim, dim, (unsigned long)off_wo,
        // Wo conv output
        dim, S,
        // residual1
        dim, S,
        // square
        dim, S,
        // reduce_mean output
        S,
        // eps
        rms_eps,
        // add_eps, rsqrt
        S, S,
        // normed
        dim, S,
        // rms_w
        dim, dim, (unsigned long)off_rms,
        // scaled
        dim, S,
        // W1
        hidden, dim, hidden, dim, (unsigned long)off_w1,
        // W3
        hidden, dim, hidden, dim, (unsigned long)off_w3,
        // W2
        dim, hidden, dim, hidden, (unsigned long)off_w2,
        // gate, up
        hidden, S, hidden, S,
        // sigmoid, silu
        hidden, S, hidden, S,
        // gated
        hidden, S,
        // down conv
        dim, S,
        // residual2
        dim, S,
        // cast out
        dim, S];
}

// Build Wo+RMSNorm+FFN blob
static NSData *mil_build_wo_rmsnorm_ffn_blob(const _Float16 *wo, int dim,
                                               const _Float16 *rms_w,
                                               const _Float16 *w1, const _Float16 *w3,
                                               int hidden, const _Float16 *w2) {
    NSUInteger wo_size   = (NSUInteger)dim * dim * 2;
    NSUInteger rms_size  = (NSUInteger)dim * 2;
    NSUInteger w_up_size = (NSUInteger)hidden * dim * 2;
    NSUInteger w_dn_size = (NSUInteger)dim * hidden * 2;

    NSUInteger cs_wo  = 64 + wo_size;
    NSUInteger cs_rms = 64 + rms_size;
    NSUInteger cs_up  = 64 + w_up_size;
    NSUInteger cs_dn  = 64 + w_dn_size;
    NSUInteger total  = 64 + cs_wo + cs_rms + cs_up + cs_up + cs_dn;

    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;

    // Wo chunk at 64
    write_chunk_header(buf + 64, (uint32_t)wo_size, (uint32_t)(64 + 64));
    memcpy(buf + 64 + 64, wo, wo_size);

    // rms_w chunk — note: shape is [1, dim, 1, 1] but data is just dim fp16 values
    NSUInteger rms_off = 64 + cs_wo;
    write_chunk_header(buf + rms_off, (uint32_t)rms_size, (uint32_t)(rms_off + 64));
    memcpy(buf + rms_off + 64, rms_w, rms_size);

    // W1 chunk
    NSUInteger w1_off = 64 + cs_wo + cs_rms;
    write_chunk_header(buf + w1_off, (uint32_t)w_up_size, (uint32_t)(w1_off + 64));
    memcpy(buf + w1_off + 64, w1, w_up_size);

    // W3 chunk
    NSUInteger w3_off = 64 + cs_wo + cs_rms + cs_up;
    write_chunk_header(buf + w3_off, (uint32_t)w_up_size, (uint32_t)(w3_off + 64));
    memcpy(buf + w3_off + 64, w3, w_up_size);

    // W2 chunk
    NSUInteger w2_off = 64 + cs_wo + cs_rms + 2*cs_up;
    write_chunk_header(buf + w2_off, (uint32_t)w_dn_size, (uint32_t)(w2_off + 64));
    memcpy(buf + w2_off + 64, w2, w_dn_size);

    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// ─── Layout helpers ─────────────────────────────────────────────────────────
// ANE uses channel-first layout: [1, C, 1, S] stored as data[c * S + s]
// Our CPU data is token-first: data[t * dim + d]
// Transpose between them.

static void transpose_to_ane(const float *token_first, float *channel_first, int C, int S) {
    // token_first[t * C + c] → channel_first[c * S + t]
    for (int t = 0; t < S; t++)
        for (int c = 0; c < C; c++)
            channel_first[c * S + t] = token_first[t * C + c];
}

static void transpose_from_ane(const float *channel_first, float *token_first, int C, int S) {
    // channel_first[c * S + t] → token_first[t * C + c]
    for (int t = 0; t < S; t++)
        for (int c = 0; c < C; c++)
            token_first[t * C + c] = channel_first[c * S + t];
}

// ─── Reference implementations ─────────────────────────────────────────────

static void ref_matvec_fp16(const _Float16 *W, int out_dim, int in_dim,
                             const float *x, float *y, int S) {
    // x is token-first [S, in_dim], y is token-first [S, out_dim]
    for (int t = 0; t < S; t++) {
        for (int o = 0; o < out_dim; o++) {
            float sum = 0;
            for (int i = 0; i < in_dim; i++)
                sum += (float)W[o * in_dim + i] * x[t * in_dim + i];
            y[t * out_dim + o] = sum;
        }
    }
}

static void ref_rmsnorm(const float *in, float *out, const float *w,
                          int dim, int S, float eps) {
    for (int t = 0; t < S; t++) {
        float ss = 0;
        for (int i = 0; i < dim; i++) ss += in[t*dim+i] * in[t*dim+i];
        float inv = 1.0f / sqrtf(ss / dim + eps);
        for (int i = 0; i < dim; i++)
            out[t*dim+i] = in[t*dim+i] * inv * w[i];
    }
}

static float ref_silu(float x) {
    return x / (1.0f + expf(-x));
}

// ─── Test 1: Fused K+V ──────────────────────────────────────────────────────

static bool test_fused_kv(int dim, int kv_dim, int S) {
    printf("\n═══ TEST 1: Fused K+V [%d→%d] × 2 outputs, S=%d ═══\n", dim, kv_dim, S);

    // Generate random weights
    _Float16 *wk = (_Float16*)malloc((size_t)kv_dim * dim * sizeof(_Float16));
    _Float16 *wv = (_Float16*)malloc((size_t)kv_dim * dim * sizeof(_Float16));
    for (int i = 0; i < kv_dim * dim; i++) {
        wk[i] = (_Float16)(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
        wv[i] = (_Float16)(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
    }

    // Generate MIL
    NSString *mil = mil_gen_kv_fused(dim, kv_dim, S);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

    // Build blob
    NSData *blob = mil_build_kv_fused_blob(wk, wv, kv_dim, dim);

    printf("  MIL: %lu bytes, Blob: %lu bytes\n",
           (unsigned long)milData.length, (unsigned long)blob.length);

    // Compile
    size_t in_size = (size_t)dim * S * sizeof(float);
    size_t out_size = (size_t)kv_dim * S * sizeof(float);
    size_t inSizes[] = {in_size};
    size_t outSizes[] = {out_size, out_size};

    double t0 = probe_time_ms();
    ANEKernel *k = ane_compile(milData, blob, 1, inSizes, 2, outSizes);
    double t_compile = probe_time_ms() - t0;

    if (!k) {
        printf("  COMPILE FAILED\n");
        free(wk); free(wv);
        return false;
    }
    printf("  Compiled in %.1f ms\n", t_compile);

    // Prepare input (token-first)
    float *input_tf = (float*)calloc((size_t)dim * S, sizeof(float));
    float *input_ane = (float*)calloc((size_t)dim * S, sizeof(float));
    for (int i = 0; i < dim * S; i++)
        input_tf[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    transpose_to_ane(input_tf, input_ane, dim, S);

    // Run ANE
    ane_write_input(k, 0, input_ane, in_size);
    t0 = probe_time_ms();
    bool ok = ane_eval(k);
    double t_eval = probe_time_ms() - t0;

    if (!ok) {
        printf("  EVAL FAILED\n");
        free(wk); free(wv); free(input_tf); free(input_ane);
        ane_free(k);
        return false;
    }
    printf("  Eval: %.2f ms\n", t_eval);

    // Read outputs (ANE channel-first) and transpose to token-first
    float *k_ane = (float*)calloc((size_t)kv_dim * S, sizeof(float));
    float *v_ane = (float*)calloc((size_t)kv_dim * S, sizeof(float));
    float *k_out = (float*)calloc((size_t)kv_dim * S, sizeof(float));
    float *v_out = (float*)calloc((size_t)kv_dim * S, sizeof(float));
    ane_read_output(k, 0, k_ane, out_size);
    ane_read_output(k, 1, v_ane, out_size);
    transpose_from_ane(k_ane, k_out, kv_dim, S);
    transpose_from_ane(v_ane, v_out, kv_dim, S);

    // Reference (token-first)
    float *k_ref = (float*)calloc((size_t)kv_dim * S, sizeof(float));
    float *v_ref = (float*)calloc((size_t)kv_dim * S, sizeof(float));
    ref_matvec_fp16(wk, kv_dim, dim, input_tf, k_ref, S);
    ref_matvec_fp16(wv, kv_dim, dim, input_tf, v_ref, S);

    // Compare
    float max_err_k = 0, max_err_v = 0;
    int n_zero_k = 0, n_zero_v = 0;
    for (int i = 0; i < kv_dim * S; i++) {
        float ek = fabsf(k_out[i] - k_ref[i]);
        float ev = fabsf(v_out[i] - v_ref[i]);
        if (ek > max_err_k) max_err_k = ek;
        if (ev > max_err_v) max_err_v = ev;
        if (k_out[i] == 0.0f) n_zero_k++;
        if (v_out[i] == 0.0f) n_zero_v++;
    }

    int total_elems = kv_dim * S;
    printf("  K output: max_err=%.6f, zeros=%d/%d (%.1f%%)\n",
           max_err_k, n_zero_k, total_elems, 100.0f * n_zero_k / total_elems);
    printf("  V output: max_err=%.6f, zeros=%d/%d (%.1f%%)\n",
           max_err_v, n_zero_v, total_elems, 100.0f * n_zero_v / total_elems);

    // Print sample values (token-first: first token's first 4 dims)
    printf("  K[0:4] ANE: [%.4f, %.4f, %.4f, %.4f]\n",
           k_out[0], k_out[1], k_out[2], k_out[3]);
    printf("  K[0:4] ref: [%.4f, %.4f, %.4f, %.4f]\n",
           k_ref[0], k_ref[1], k_ref[2], k_ref[3]);
    printf("  V[0:4] ANE: [%.4f, %.4f, %.4f, %.4f]\n",
           v_out[0], v_out[1], v_out[2], v_out[3]);
    printf("  V[0:4] ref: [%.4f, %.4f, %.4f, %.4f]\n",
           v_ref[0], v_ref[1], v_ref[2], v_ref[3]);

    bool pass = (max_err_k < 0.05f && max_err_v < 0.05f &&
                 n_zero_k < total_elems/2 && n_zero_v < total_elems/2);
    printf("  %s\n", pass ? "PASS" : "FAIL (outputs are zeros or too inaccurate)");

    free(wk); free(wv); free(input_tf); free(input_ane);
    free(k_ane); free(v_ane); free(k_out); free(v_out);
    free(k_ref); free(v_ref);
    ane_free(k);
    return pass;
}

// ─── Test 2: Fused QKV with SAME dimensions (control test) ──────────────────
// If K+V fails, test 3 identical outputs to see if multi-output is fundamentally broken

static bool test_fused_same_dim_3out(int dim, int out_dim, int S) {
    printf("\n═══ TEST 2: 3 identical convs [%d→%d], 3 outputs, S=%d ═══\n", dim, out_dim, S);

    // Use the existing QKV generator but with ALL same dimensions
    NSString *mil = mil_gen_qkv_baked(dim, out_dim, out_dim, S);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

    // Random weights (3 matrices, all [out_dim, dim])
    _Float16 *wq = (_Float16*)malloc((size_t)out_dim * dim * sizeof(_Float16));
    _Float16 *wk = (_Float16*)malloc((size_t)out_dim * dim * sizeof(_Float16));
    _Float16 *wv = (_Float16*)malloc((size_t)out_dim * dim * sizeof(_Float16));
    for (int i = 0; i < out_dim * dim; i++) {
        wq[i] = (_Float16)(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
        wk[i] = (_Float16)(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
        wv[i] = (_Float16)(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
    }

    NSData *blob = mil_build_qkv_baked_blob(wq, out_dim, wk, wv, out_dim, dim);

    size_t in_size = (size_t)dim * S * sizeof(float);
    size_t out_size = (size_t)out_dim * S * sizeof(float);
    size_t inSizes[] = {in_size};
    size_t outSizes[] = {out_size, out_size, out_size};

    double t0 = probe_time_ms();
    ANEKernel *k = ane_compile(milData, blob, 1, inSizes, 3, outSizes);
    double t_compile = probe_time_ms() - t0;

    if (!k) {
        printf("  COMPILE FAILED\n");
        free(wq); free(wk); free(wv);
        return false;
    }
    printf("  Compiled in %.1f ms\n", t_compile);

    float *input_tf = (float*)calloc((size_t)dim * S, sizeof(float));
    float *input_ane = (float*)calloc((size_t)dim * S, sizeof(float));
    for (int i = 0; i < dim * S; i++)
        input_tf[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    transpose_to_ane(input_tf, input_ane, dim, S);

    ane_write_input(k, 0, input_ane, in_size);
    t0 = probe_time_ms();
    bool ok = ane_eval(k);
    double t_eval = probe_time_ms() - t0;

    if (!ok) {
        printf("  EVAL FAILED\n");
        free(wq); free(wk); free(wv); free(input_tf); free(input_ane);
        ane_free(k);
        return false;
    }
    printf("  Eval: %.2f ms\n", t_eval);

    float *q_ane = (float*)calloc((size_t)out_dim * S, sizeof(float));
    float *k_ane2 = (float*)calloc((size_t)out_dim * S, sizeof(float));
    float *v_ane = (float*)calloc((size_t)out_dim * S, sizeof(float));
    float *q_out = (float*)calloc((size_t)out_dim * S, sizeof(float));
    float *k_out = (float*)calloc((size_t)out_dim * S, sizeof(float));
    float *v_out = (float*)calloc((size_t)out_dim * S, sizeof(float));
    ane_read_output(k, 0, q_ane, out_size);
    ane_read_output(k, 1, k_ane2, out_size);
    ane_read_output(k, 2, v_ane, out_size);
    transpose_from_ane(q_ane, q_out, out_dim, S);
    transpose_from_ane(k_ane2, k_out, out_dim, S);
    transpose_from_ane(v_ane, v_out, out_dim, S);

    float *q_ref = (float*)calloc((size_t)out_dim * S, sizeof(float));
    float *k_ref = (float*)calloc((size_t)out_dim * S, sizeof(float));
    float *v_ref = (float*)calloc((size_t)out_dim * S, sizeof(float));
    ref_matvec_fp16(wq, out_dim, dim, input_tf, q_ref, S);
    ref_matvec_fp16(wk, out_dim, dim, input_tf, k_ref, S);
    ref_matvec_fp16(wv, out_dim, dim, input_tf, v_ref, S);

    int total_elems = out_dim * S;
    float max_err_q = 0, max_err_k2 = 0, max_err_v = 0;
    int nz_q = 0, nz_k = 0, nz_v = 0;
    for (int i = 0; i < total_elems; i++) {
        float eq = fabsf(q_out[i] - q_ref[i]);
        float ek = fabsf(k_out[i] - k_ref[i]);
        float ev = fabsf(v_out[i] - v_ref[i]);
        if (eq > max_err_q) max_err_q = eq;
        if (ek > max_err_k2) max_err_k2 = ek;
        if (ev > max_err_v) max_err_v = ev;
        if (q_out[i] == 0.0f) nz_q++;
        if (k_out[i] == 0.0f) nz_k++;
        if (v_out[i] == 0.0f) nz_v++;
    }

    printf("  Q: max_err=%.6f, zeros=%d/%d\n", max_err_q, nz_q, total_elems);
    printf("  K: max_err=%.6f, zeros=%d/%d\n", max_err_k2, nz_k, total_elems);
    printf("  V: max_err=%.6f, zeros=%d/%d\n", max_err_v, nz_v, total_elems);
    printf("  Q[0:4]: [%.4f, %.4f, %.4f, %.4f]\n", q_out[0], q_out[1], q_out[2], q_out[3]);
    printf("  Q ref:  [%.4f, %.4f, %.4f, %.4f]\n", q_ref[0], q_ref[1], q_ref[2], q_ref[3]);

    bool pass = (max_err_q < 0.05f && max_err_k2 < 0.05f && max_err_v < 0.05f &&
                 nz_q < total_elems/2 && nz_k < total_elems/2 && nz_v < total_elems/2);
    printf("  %s\n", pass ? "PASS" : "FAIL");

    free(wq); free(wk); free(wv); free(input_tf); free(input_ane);
    free(q_ane); free(k_ane2); free(v_ane);
    free(q_out); free(k_out); free(v_out);
    free(q_ref); free(k_ref); free(v_ref);
    ane_free(k);
    return pass;
}

// ─── Test 3: Wo + RMSNorm + FFN mega fusion ─────────────────────────────────

static bool test_wo_rmsnorm_ffn(int dim, int hidden, int S) {
    printf("\n═══ TEST 3: Wo+RMSNorm+FFN mega fusion [dim=%d, hidden=%d, S=%d] ═══\n",
           dim, hidden, S);

    float rms_eps = 1e-5f;

    // Small test dimensions to keep reference computation fast
    _Float16 *wo_w  = (_Float16*)malloc((size_t)dim * dim * sizeof(_Float16));
    _Float16 *rms_w = (_Float16*)malloc((size_t)dim * sizeof(_Float16));
    _Float16 *w1    = (_Float16*)malloc((size_t)hidden * dim * sizeof(_Float16));
    _Float16 *w3    = (_Float16*)malloc((size_t)hidden * dim * sizeof(_Float16));
    _Float16 *w2    = (_Float16*)malloc((size_t)dim * hidden * sizeof(_Float16));

    srand(42);
    for (int i = 0; i < dim * dim; i++)
        wo_w[i] = (_Float16)(((float)rand() / RAND_MAX - 0.5f) * 0.02f);
    for (int i = 0; i < dim; i++)
        rms_w[i] = (_Float16)(0.8f + ((float)rand() / RAND_MAX) * 0.4f);  // ~1.0
    for (int i = 0; i < hidden * dim; i++) {
        w1[i] = (_Float16)(((float)rand() / RAND_MAX - 0.5f) * 0.02f);
        w3[i] = (_Float16)(((float)rand() / RAND_MAX - 0.5f) * 0.02f);
    }
    for (int i = 0; i < dim * hidden; i++)
        w2[i] = (_Float16)(((float)rand() / RAND_MAX - 0.5f) * 0.02f);

    // Generate MIL
    NSString *mil = mil_gen_wo_rmsnorm_ffn(dim, hidden, S, rms_eps);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

    // Build blob
    NSData *blob = mil_build_wo_rmsnorm_ffn_blob(wo_w, dim, rms_w, w1, w3, hidden, w2);

    printf("  MIL: %lu bytes, Blob: %lu bytes (%.1f MB)\n",
           (unsigned long)milData.length, (unsigned long)blob.length,
           (double)blob.length / (1024*1024));

    // Compile
    size_t in_size = (size_t)dim * S * sizeof(float);
    size_t inSizes[] = {in_size, in_size};  // attn_out, x_residual
    size_t outSizes[] = {in_size};           // single output

    double t0 = probe_time_ms();
    ANEKernel *k = ane_compile(milData, blob, 2, inSizes, 1, outSizes);
    double t_compile = probe_time_ms() - t0;

    if (!k) {
        printf("  COMPILE FAILED\n");
        // Print first part of MIL for debugging
        NSString *milStr = [[NSString alloc] initWithData:milData encoding:NSUTF8StringEncoding];
        if (milStr.length > 500) milStr = [milStr substringToIndex:500];
        printf("  MIL preview:\n%s\n...\n", [milStr UTF8String]);
        free(wo_w); free(rms_w); free(w1); free(w3); free(w2);
        return false;
    }
    printf("  Compiled in %.1f ms\n", t_compile);

    // Prepare inputs (token-first)
    float *attn_tf = (float*)calloc((size_t)dim * S, sizeof(float));
    float *xres_tf = (float*)calloc((size_t)dim * S, sizeof(float));
    float *attn_ane = (float*)calloc((size_t)dim * S, sizeof(float));
    float *xres_ane = (float*)calloc((size_t)dim * S, sizeof(float));
    srand(123);
    for (int i = 0; i < dim * S; i++) {
        attn_tf[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        xres_tf[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    transpose_to_ane(attn_tf, attn_ane, dim, S);
    transpose_to_ane(xres_tf, xres_ane, dim, S);

    // Run ANE
    ane_write_input(k, 0, attn_ane, in_size);
    ane_write_input(k, 1, xres_ane, in_size);
    t0 = probe_time_ms();
    bool ok = ane_eval(k);
    double t_eval = probe_time_ms() - t0;

    if (!ok) {
        printf("  EVAL FAILED\n");
        free(wo_w); free(rms_w); free(w1); free(w3); free(w2);
        free(attn_tf); free(xres_tf); free(attn_ane); free(xres_ane);
        ane_free(k);
        return false;
    }
    printf("  Eval: %.2f ms\n", t_eval);

    // Read output and transpose
    float *out_ane = (float*)calloc((size_t)dim * S, sizeof(float));
    float *ane_out = (float*)calloc((size_t)dim * S, sizeof(float));
    ane_read_output(k, 0, out_ane, in_size);
    transpose_from_ane(out_ane, ane_out, dim, S);

    // Reference computation (token-first)
    float *wo_out = (float*)calloc((size_t)dim * S, sizeof(float));
    float *res1   = (float*)calloc((size_t)dim * S, sizeof(float));
    float *normed = (float*)calloc((size_t)dim * S, sizeof(float));
    float *gate   = (float*)calloc((size_t)hidden * S, sizeof(float));
    float *up_ref = (float*)calloc((size_t)hidden * S, sizeof(float));
    float *ffn_out = (float*)calloc((size_t)dim * S, sizeof(float));
    float *ref_out = (float*)calloc((size_t)dim * S, sizeof(float));

    float *rms_w32 = (float*)calloc(dim, sizeof(float));
    for (int i = 0; i < dim; i++) rms_w32[i] = (float)rms_w[i];

    ref_matvec_fp16(wo_w, dim, dim, attn_tf, wo_out, S);
    for (int i = 0; i < dim * S; i++) res1[i] = wo_out[i] + xres_tf[i];
    ref_rmsnorm(res1, normed, rms_w32, dim, S, rms_eps);
    ref_matvec_fp16(w1, hidden, dim, normed, gate, S);
    ref_matvec_fp16(w3, hidden, dim, normed, up_ref, S);
    for (int i = 0; i < hidden * S; i++)
        gate[i] = ref_silu(gate[i]) * up_ref[i];
    ref_matvec_fp16(w2, dim, hidden, gate, ffn_out, S);
    for (int i = 0; i < dim * S; i++) ref_out[i] = res1[i] + ffn_out[i];

    float max_err = 0, max_rel = 0;
    int n_zero = 0;
    for (int i = 0; i < dim * S; i++) {
        float err = fabsf(ane_out[i] - ref_out[i]);
        if (err > max_err) max_err = err;
        float mag = fabsf(ref_out[i]);
        if (mag > 1e-6f) {
            float rel = err / mag;
            if (rel > max_rel) max_rel = rel;
        }
        if (ane_out[i] == 0.0f) n_zero++;
    }

    int total_elems = dim * S;
    printf("  max_abs_err=%.6f, max_rel_err=%.4f, zeros=%d/%d\n",
           max_err, max_rel, n_zero, total_elems);
    printf("  ANE[0:4]: [%.4f, %.4f, %.4f, %.4f]\n",
           ane_out[0], ane_out[1], ane_out[2], ane_out[3]);
    printf("  ref[0:4]: [%.4f, %.4f, %.4f, %.4f]\n",
           ref_out[0], ref_out[1], ref_out[2], ref_out[3]);

    bool pass = (max_err < 1.0f && max_rel < 0.5f && n_zero < total_elems/2);
    printf("  %s\n", pass ? "PASS" : "FAIL");

    free(wo_w); free(rms_w); free(rms_w32);
    free(w1); free(w3); free(w2);
    free(attn_tf); free(xres_tf); free(attn_ane); free(xres_ane);
    free(out_ane); free(ane_out);
    free(wo_out); free(res1); free(normed);
    free(gate); free(up_ref); free(ffn_out); free(ref_out);
    ane_free(k);
    return pass;
}

// ─── Test 4: Separate K and V (control — verify single-output convs work) ───

static bool test_separate_kv(int dim, int kv_dim, int S) {
    printf("\n═══ TEST 4 (control): Separate K, V convs [%d→%d], S=%d ═══\n", dim, kv_dim, S);

    srand(999);
    _Float16 *wk = (_Float16*)malloc((size_t)kv_dim * dim * sizeof(_Float16));
    _Float16 *wv = (_Float16*)malloc((size_t)kv_dim * dim * sizeof(_Float16));
    for (int i = 0; i < kv_dim * dim; i++) {
        wk[i] = (_Float16)(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
        wv[i] = (_Float16)(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
    }

    NSString *mil_k = mil_gen_conv_baked(dim, kv_dim, S);
    NSString *mil_v = mil_gen_conv_baked(dim, kv_dim, S);
    NSData *milData_k = [mil_k dataUsingEncoding:NSUTF8StringEncoding];
    NSData *milData_v = [mil_v dataUsingEncoding:NSUTF8StringEncoding];

    NSData *blob_k = mil_build_single_weight_blob(wk, kv_dim, dim);
    NSData *blob_v = mil_build_single_weight_blob(wv, kv_dim, dim);

    size_t in_size = (size_t)dim * S * sizeof(float);
    size_t out_size = (size_t)kv_dim * S * sizeof(float);

    ANEKernel *kk = ane_compile(milData_k, blob_k, 1, &in_size, 1, &out_size);
    ANEKernel *kv = ane_compile(milData_v, blob_v, 1, &in_size, 1, &out_size);

    if (!kk || !kv) {
        printf("  COMPILE FAILED\n");
        free(wk); free(wv);
        if (kk) ane_free(kk); if (kv) ane_free(kv);
        return false;
    }

    float *input_tf = (float*)calloc((size_t)dim * S, sizeof(float));
    float *input_ane = (float*)calloc((size_t)dim * S, sizeof(float));
    for (int i = 0; i < dim * S; i++)
        input_tf[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    transpose_to_ane(input_tf, input_ane, dim, S);

    float *k_ane_buf = (float*)calloc((size_t)kv_dim * S, sizeof(float));
    float *v_ane_buf = (float*)calloc((size_t)kv_dim * S, sizeof(float));
    float *k_out = (float*)calloc((size_t)kv_dim * S, sizeof(float));
    float *v_out = (float*)calloc((size_t)kv_dim * S, sizeof(float));

    ane_write_input(kk, 0, input_ane, in_size);
    ane_eval(kk);
    ane_read_output(kk, 0, k_ane_buf, out_size);
    transpose_from_ane(k_ane_buf, k_out, kv_dim, S);

    ane_write_input(kv, 0, input_ane, in_size);
    ane_eval(kv);
    ane_read_output(kv, 0, v_ane_buf, out_size);
    transpose_from_ane(v_ane_buf, v_out, kv_dim, S);

    float *k_ref = (float*)calloc((size_t)kv_dim * S, sizeof(float));
    float *v_ref = (float*)calloc((size_t)kv_dim * S, sizeof(float));
    ref_matvec_fp16(wk, kv_dim, dim, input_tf, k_ref, S);
    ref_matvec_fp16(wv, kv_dim, dim, input_tf, v_ref, S);

    float max_err_k = 0, max_err_v = 0;
    int total_elems = kv_dim * S;
    for (int i = 0; i < total_elems; i++) {
        float ek = fabsf(k_out[i] - k_ref[i]);
        float ev = fabsf(v_out[i] - v_ref[i]);
        if (ek > max_err_k) max_err_k = ek;
        if (ev > max_err_v) max_err_v = ev;
    }

    printf("  K: max_err=%.6f, V: max_err=%.6f\n", max_err_k, max_err_v);
    printf("  K[0:4]: [%.4f, %.4f, %.4f, %.4f] ref: [%.4f, %.4f, %.4f, %.4f]\n",
           k_out[0], k_out[1], k_out[2], k_out[3],
           k_ref[0], k_ref[1], k_ref[2], k_ref[3]);

    bool pass = (max_err_k < 0.05f && max_err_v < 0.05f);
    printf("  %s\n", pass ? "PASS" : "FAIL");

    free(wk); free(wv); free(input_tf); free(input_ane);
    free(k_ane_buf); free(v_ane_buf);
    free(k_out); free(v_out); free(k_ref); free(v_ref);
    ane_free(kk); ane_free(kv);
    return pass;
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        printf("═══════════════════════════════════════════════════════\n");
        printf("  ANE Fusion Probe — Reduce programs/layer from 5→3\n");
        printf("═══════════════════════════════════════════════════════\n");

        // Use smaller test dimensions for faster reference computation
        // Full Mistral dims: dim=4096, kv_dim=1024, hidden=14336
        // Test dims: proportionally scaled down
        int dim = 256;
        int kv_dim = 64;
        int hidden = 512;
        int S = 32;  // minimum ANE sequence length for fp32 I/O is 16

        bool use_full = false;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--full") == 0) use_full = true;
        }

        if (use_full) {
            dim = 4096;
            kv_dim = 1024;
            hidden = 14336;
            S = 64;
            printf("\n  Using FULL Mistral dimensions (dim=%d, kv=%d, hidden=%d, S=%d)\n",
                   dim, kv_dim, hidden, S);
        } else {
            printf("\n  Using SMALL test dimensions (dim=%d, kv=%d, hidden=%d, S=%d)\n",
                   dim, kv_dim, hidden, S);
            printf("  (Use --full for Mistral-scale test)\n");
        }

        // Run control test first
        srand(42);
        bool t4 = test_separate_kv(dim, kv_dim, S);

        // Test fused K+V (same-dim multi-output)
        srand(42);
        bool t1 = test_fused_kv(dim, kv_dim, S);

        // Test 3 same-dim outputs
        srand(42);
        bool t2 = test_fused_same_dim_3out(dim, kv_dim, S);

        // Test mega fusion (at small scale)
        // srand(42) inside test
        bool t3 = test_wo_rmsnorm_ffn(dim, hidden, S);

        printf("\n═══ SUMMARY ═══\n");
        printf("  Test 4 (control, separate K+V):   %s\n", t4 ? "PASS" : "FAIL");
        printf("  Test 1 (fused K+V, 2 outputs):    %s\n", t1 ? "PASS" : "FAIL");
        printf("  Test 2 (3 same-dim outputs):       %s\n", t2 ? "PASS" : "FAIL");
        printf("  Test 3 (Wo+RMSNorm+FFN mega):     %s\n", t3 ? "PASS" : "FAIL");

        if (t1 && t3) {
            printf("\n  --> 3 programs/layer FEASIBLE: 32×3 = 96 programs (under 128)\n");
            printf("      P1: Q conv [dim→dim]\n");
            printf("      P2: Fused K+V [dim→kv_dim] × 2\n");
            printf("      P3: Wo+RMSNorm+FFN mega [2 inputs → 1 output]\n");
        } else {
            if (!t1) printf("\n  --> Fused K+V FAILED. Multi-output broken on ANE.\n");
            if (!t3) printf("\n  --> Mega fusion FAILED. reduce_mean/rsqrt may not work on ANE.\n");
        }

        printf("\n");
        return 0;
    }
}
