// ane_mil_gen_mistral.h — Fused MIL generators for Mistral 7B ANE baked-weight prefill
// Builds on ane_mil_gen.h conv patterns. All weights baked via BLOBFILE.
#pragma once

#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <string.h>

// ─── MIL boilerplate ────────────────────────────────────────────────────────
#define MIL_HEADER \
    @"program(1.3)\n" \
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"

#define MIL_CONV_CONSTS \
    "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n" \
    "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n" \
    "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n" \
    "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n" \
    "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"

// ─── Weight blob helpers ────────────────────────────────────────────────────

// Write a single DEADBEEF chunk header at buf.
// data_size: bytes of fp16 weight data
// abs_data_offset: absolute offset from blob start to fp16 data
static inline void write_chunk_header(uint8_t *buf, uint32_t data_size, uint32_t abs_data_offset) {
    buf[0] = 0xEF; buf[1] = 0xBE; buf[2] = 0xAD; buf[3] = 0xDE; // magic
    buf[4] = 0x01;
    *(uint32_t*)(buf + 8) = data_size;
    *(uint32_t*)(buf + 16) = abs_data_offset;
}

// Convert fp16 weight data from Q4_0 source using provided dequant function,
// or from fp32 source directly.
// Returns pointer to fp16 data within the blob buffer.
typedef void (*dequant_to_fp16_fn)(const void *src, _Float16 *dst, int rows, int cols);

// ─── Single conv (Wo) ───────────────────────────────────────────────────────
// Identical to mil_gen_conv from ane_mil_gen.h, included for self-containment.

static NSString *mil_gen_conv_baked(int in_ch, int out_ch, int S) {
    return [NSString stringWithFormat:
        MIL_HEADER
        @"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        MIL_CONV_CONSTS
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n}\n",
        in_ch, S, in_ch, S,
        out_ch, in_ch, out_ch, in_ch,
        out_ch, S, out_ch, S];
}

// ─── Fused K+V (2 parallel convs, same shape) ────────────────────────────
// Input:  [1, dim, 1, S] fp32
// Output: K[1, kv_dim, 1, S], V[1, kv_dim, 1, S] fp32
// ANE supports 2 outputs correctly (3 outputs broken — only last output is valid).

static NSString *mil_gen_kv_fused(int dim, int kv_dim, int S) {
    NSUInteger wkv_size = (NSUInteger)kv_dim * dim * 2;
    NSUInteger cs_kv = 64 + wkv_size;

    NSUInteger off_wk = 64;
    NSUInteger off_wv = 64 + cs_kv;

    return [NSString stringWithFormat:
        MIL_HEADER
        @"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        MIL_CONV_CONSTS
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wk = const()[name = string(\"Wk\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wv = const()[name = string(\"Wv\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> k16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wk, x = x16)[name = string(\"conv_k\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> v16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wv, x = x16)[name = string(\"conv_v\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> k = cast(dtype = to_fp32, x = k16)[name = string(\"cast_k\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> v = cast(dtype = to_fp32, x = v16)[name = string(\"cast_v\")];\n"
        "    } -> (k, v);\n}\n",
        dim, S, dim, S,
        kv_dim, dim, kv_dim, dim, (unsigned long)off_wk,
        kv_dim, dim, kv_dim, dim, (unsigned long)off_wv,
        kv_dim, S, kv_dim, S,
        kv_dim, S, kv_dim, S];
}

// ─── Fused QKV (3 parallel convs) — BROKEN: only last output valid ────────
// Input:  [1, dim, 1, S] fp32
// Output: Q[1, q_dim, 1, S], K[1, kv_dim, 1, S], V[1, kv_dim, 1, S] fp32
// Wq: [q_dim, dim], Wk: [kv_dim, dim], Wv: [kv_dim, dim]

static NSString *mil_gen_qkv_baked(int dim, int q_dim, int kv_dim, int S) {
    // Chunk layout in blob:
    // [0:64)    global header
    // [64: 64+cs_q)   Wq chunk:  64B hdr + q_dim*dim*2 data
    // [64+cs_q: 64+cs_q+cs_kv)  Wk chunk: 64B hdr + kv_dim*dim*2 data
    // [64+cs_q+cs_kv: ...)  Wv chunk
    NSUInteger wq_size = (NSUInteger)q_dim * dim * 2;
    NSUInteger wkv_size = (NSUInteger)kv_dim * dim * 2;
    NSUInteger cs_q = 64 + wq_size;
    NSUInteger cs_kv = 64 + wkv_size;

    // BLOBFILE offset points to chunk HEADER (DEADBEEF), not data
    NSUInteger off_wq = 64;                // chunk header for Wq
    NSUInteger off_wk = 64 + cs_q;         // chunk header for Wk
    NSUInteger off_wv = 64 + cs_q + cs_kv; // chunk header for Wv

    return [NSString stringWithFormat:
        MIL_HEADER
        @"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        MIL_CONV_CONSTS
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        // Wq
        "        tensor<fp16, [%d, %d, 1, 1]> Wq = const()[name = string(\"Wq\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        // Wk
        "        tensor<fp16, [%d, %d, 1, 1]> Wk = const()[name = string(\"Wk\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        // Wv
        "        tensor<fp16, [%d, %d, 1, 1]> Wv = const()[name = string(\"Wv\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        // Convolutions
        "        tensor<fp16, [1, %d, 1, %d]> q16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wq, x = x16)[name = string(\"conv_q\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> k16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wk, x = x16)[name = string(\"conv_k\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> v16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wv, x = x16)[name = string(\"conv_v\")];\n"
        // Cast outputs
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> q = cast(dtype = to_fp32, x = q16)[name = string(\"cast_q\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> k = cast(dtype = to_fp32, x = k16)[name = string(\"cast_k\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> v = cast(dtype = to_fp32, x = v16)[name = string(\"cast_v\")];\n"
        "    } -> (q, k, v);\n}\n",
        dim, S, dim, S,
        // Wq
        q_dim, dim, q_dim, dim, (unsigned long)off_wq,
        // Wk
        kv_dim, dim, kv_dim, dim, (unsigned long)off_wk,
        // Wv
        kv_dim, dim, kv_dim, dim, (unsigned long)off_wv,
        // conv outputs
        q_dim, S, kv_dim, S, kv_dim, S,
        // cast outputs
        q_dim, S, kv_dim, S, kv_dim, S];
}

// ─── Fused FFN: W1(gate) + W3(up) → SiLU → W2(down) ────────────────────────
// Input:  [1, dim, 1, S] fp32
// Output: [1, dim, 1, S] fp32
//
// MIL graph:
//   x16 = cast(x, fp16)
//   gate = conv(x16, W1)    [1, hidden, 1, S]
//   up   = conv(x16, W3)    [1, hidden, 1, S]
//   sig  = sigmoid(gate)
//   silu = mul(gate, sig)   SiLU = gate * sigmoid(gate)
//   h    = mul(silu, up)    gated output
//   out16 = conv(h, W2)     [1, dim, 1, S]
//   out  = cast(out16, fp32)
//
// All intermediates stay in ANE SRAM — no CPU round-trips.

static NSString *mil_gen_ffn_fused(int dim, int hidden, int S) {
    // Chunk layout:
    // [0:64)                         global header
    // [64: 64+cs_up)                 W1 chunk: 64B hdr + hidden*dim*2 data
    // [64+cs_up: 64+2*cs_up)         W3 chunk: same size as W1
    // [64+2*cs_up: 64+2*cs_up+cs_dn) W2 chunk: 64B hdr + dim*hidden*2 data
    // Note: W1 and W3 are [hidden, dim], W2 is [dim, hidden] — same element count
    NSUInteger w_up_size = (NSUInteger)hidden * dim * 2;   // fp16 bytes
    NSUInteger w_dn_size = (NSUInteger)dim * hidden * 2;   // same count, different shape
    NSUInteger cs_up = 64 + w_up_size;
    NSUInteger cs_dn = 64 + w_dn_size;

    // BLOBFILE offset points to chunk HEADER (DEADBEEF), not data
    NSUInteger off_w1 = 64;                    // chunk header for W1
    NSUInteger off_w3 = 64 + cs_up;            // chunk header for W3
    NSUInteger off_w2 = 64 + 2*cs_up;          // chunk header for W2

    return [NSString stringWithFormat:
        MIL_HEADER
        @"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        MIL_CONV_CONSTS
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        // W1 (gate)
        "        tensor<fp16, [%d, %d, 1, 1]> W1 = const()[name = string(\"W1\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        // W3 (up)
        "        tensor<fp16, [%d, %d, 1, 1]> W3 = const()[name = string(\"W3\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        // W2 (down)
        "        tensor<fp16, [%d, %d, 1, 1]> W2 = const()[name = string(\"W2\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        // Gate conv
        "        tensor<fp16, [1, %d, 1, %d]> gate = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W1, x = x16)[name = string(\"conv_gate\")];\n"
        // Up conv
        "        tensor<fp16, [1, %d, 1, %d]> up = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W3, x = x16)[name = string(\"conv_up\")];\n"
        // SiLU = gate * sigmoid(gate)
        "        tensor<fp16, [1, %d, 1, %d]> sig = sigmoid(x = gate)[name = string(\"sigmoid\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> silu = mul(x = gate, y = sig)[name = string(\"silu\")];\n"
        // Gated output
        "        tensor<fp16, [1, %d, 1, %d]> h = mul(x = silu, y = up)[name = string(\"gated\")];\n"
        // Down projection
        "        tensor<fp16, [1, %d, 1, %d]> out16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W2, x = h)[name = string(\"conv_down\")];\n"
        // Cast output
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> out = cast(dtype = to_fp32, x = out16)[name = string(\"cast_out\")];\n"
        "    } -> (out);\n}\n",
        // func signature
        dim, S, dim, S,
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
        // cast out
        dim, S];
}

// ─── Weight blob builders ───────────────────────────────────────────────────

// Build single-weight blob (for Wo conv)
static NSData *mil_build_single_weight_blob(const _Float16 *weights_fp16, int out_ch, int in_ch) {
    NSUInteger wsize = (NSUInteger)out_ch * in_ch * 2;
    NSUInteger total = 64 + 64 + wsize;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    write_chunk_header(buf + 64, (uint32_t)wsize, 128);
    memcpy(buf + 128, weights_fp16, wsize);
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// Build fused K+V blob: Wk[kv_dim, dim] + Wv[kv_dim, dim]
static NSData *mil_build_kv_fused_blob(const _Float16 *wk, const _Float16 *wv,
                                        int kv_dim, int dim) {
    NSUInteger wkv_size = (NSUInteger)kv_dim * dim * 2;
    NSUInteger cs_kv = 64 + wkv_size;
    NSUInteger total = 64 + cs_kv + cs_kv;

    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;

    write_chunk_header(buf + 64, (uint32_t)wkv_size, (uint32_t)(64 + 64));
    memcpy(buf + 64 + 64, wk, wkv_size);

    NSUInteger wv_chunk = 64 + cs_kv;
    write_chunk_header(buf + wv_chunk, (uint32_t)wkv_size, (uint32_t)(wv_chunk + 64));
    memcpy(buf + wv_chunk + 64, wv, wkv_size);

    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// Build fused QKV blob: Wq[q_dim, dim] + Wk[kv_dim, dim] + Wv[kv_dim, dim]
static NSData *mil_build_qkv_baked_blob(const _Float16 *wq, int q_dim,
                                         const _Float16 *wk, const _Float16 *wv, int kv_dim,
                                         int dim) {
    NSUInteger wq_size = (NSUInteger)q_dim * dim * 2;
    NSUInteger wkv_size = (NSUInteger)kv_dim * dim * 2;
    NSUInteger cs_q = 64 + wq_size;
    NSUInteger cs_kv = 64 + wkv_size;
    NSUInteger total = 64 + cs_q + cs_kv + cs_kv;

    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;

    // Wq chunk at offset 64
    write_chunk_header(buf + 64, (uint32_t)wq_size, (uint32_t)(64 + 64));
    memcpy(buf + 64 + 64, wq, wq_size);

    // Wk chunk at offset 64 + cs_q
    NSUInteger wk_chunk = 64 + cs_q;
    write_chunk_header(buf + wk_chunk, (uint32_t)wkv_size, (uint32_t)(wk_chunk + 64));
    memcpy(buf + wk_chunk + 64, wk, wkv_size);

    // Wv chunk at offset 64 + cs_q + cs_kv
    NSUInteger wv_chunk = 64 + cs_q + cs_kv;
    write_chunk_header(buf + wv_chunk, (uint32_t)wkv_size, (uint32_t)(wv_chunk + 64));
    memcpy(buf + wv_chunk + 64, wv, wkv_size);

    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// Build fused FFN blob: W1[hidden, dim] + W3[hidden, dim] + W2[dim, hidden]
static NSData *mil_build_ffn_fused_blob(const _Float16 *w1, const _Float16 *w3,
                                         int hidden, const _Float16 *w2, int dim) {
    NSUInteger w_up_size = (NSUInteger)hidden * dim * 2;
    NSUInteger w_dn_size = (NSUInteger)dim * hidden * 2;  // same byte count
    NSUInteger cs_up = 64 + w_up_size;
    NSUInteger cs_dn = 64 + w_dn_size;
    NSUInteger total = 64 + cs_up + cs_up + cs_dn;

    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;

    // W1 chunk
    write_chunk_header(buf + 64, (uint32_t)w_up_size, (uint32_t)(64 + 64));
    memcpy(buf + 64 + 64, w1, w_up_size);

    // W3 chunk
    NSUInteger w3_chunk = 64 + cs_up;
    write_chunk_header(buf + w3_chunk, (uint32_t)w_up_size, (uint32_t)(w3_chunk + 64));
    memcpy(buf + w3_chunk + 64, w3, w_up_size);

    // W2 chunk
    NSUInteger w2_chunk = 64 + 2 * cs_up;
    write_chunk_header(buf + w2_chunk, (uint32_t)w_dn_size, (uint32_t)(w2_chunk + 64));
    memcpy(buf + w2_chunk + 64, w2, w_dn_size);

    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}
