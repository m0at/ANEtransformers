// metal_matvec.h — Metal GPU backend for Mistral 7B on Apple Silicon
// All decode operations on GPU: GEMV, RMSNorm, RoPE, KV cache, GQA attention.
// Single command buffer for all 32 layers = 1 GPU submission per decode step.
#pragma once

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <mach-o/dyld.h>

// ─── Shader dispatch constants (must match q4_0_metal.metal) ─────────────────
#define GEMV_FAST_ROWS   4
#define GEMV_FAST_TPG    128
#define GEMV_MAX_ROWS    8
#define GEMV_MAX_TPG     256
#define GEMV_FAST_LIMIT  4096
#define METAL_MAX_LAYERS 64
#define GQA_THREADS      256

// ─── Metal context ───────────────────────────────────────────────────────────
typedef struct {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;

    // Compute pipelines
    id<MTLComputePipelineState> gemv_fast;
    id<MTLComputePipelineState> gemv_max;
    id<MTLComputePipelineState> gemm;
    id<MTLComputePipelineState> rmsnorm_p;
    id<MTLComputePipelineState> silu_mul_p;
    id<MTLComputePipelineState> vadd_p;
    id<MTLComputePipelineState> rope_p;
    id<MTLComputePipelineState> kv_write_p;
    id<MTLComputePipelineState> gqa_att_p;

    // Weights: single buffer wrapping entire mmap'd GGUF (zero-copy)
    id<MTLBuffer> weights_buf;
    const void   *weights_base;

    // Running hidden state
    id<MTLBuffer> x_buf;       // [dim]

    // Scratch buffers
    id<MTLBuffer> xb_buf;      // [max(dim, hidden)]
    id<MTLBuffer> xb2_buf;     // [dim] — attention output
    id<MTLBuffer> q_buf;       // [dim]
    id<MTLBuffer> k_buf;       // [kv_dim]
    id<MTLBuffer> v_buf;       // [kv_dim]
    id<MTLBuffer> hb_buf;      // [hidden]
    id<MTLBuffer> hb2_buf;     // [hidden]

    // RoPE theta inverses
    id<MTLBuffer> theta_inv_buf;  // [head_dim/2]

    // KV cache (Metal shared buffers — CPU and GPU both access)
    id<MTLBuffer> k_cache_buf;    // [n_layers * max_seq * kv_dim] fp16
    id<MTLBuffer> v_cache_buf;    // [n_layers * max_seq * kv_dim] fp16
    int kv_dim;
    int max_seq;

    // Attention scratch
    id<MTLBuffer> att_scratch_buf;  // [n_heads * max_seq] fp32

    // Per-layer RMSNorm weights
    int n_layers;
    id<MTLBuffer> rms_att_bufs[METAL_MAX_LAYERS];
    id<MTLBuffer> rms_ffn_bufs[METAL_MAX_LAYERS];

    // Batch decode buffers (speculative verify, S tokens at once)
    int batch_S;  // current max batch size (0 = not allocated)
    id<MTLBuffer> x_batch_buf;     // [S * dim]
    id<MTLBuffer> xb_batch_buf;    // [S * max(dim, hidden)]
    id<MTLBuffer> xb2_batch_buf;   // [S * dim]
    id<MTLBuffer> q_batch_buf;     // [S * dim]
    id<MTLBuffer> k_batch_buf;     // [S * kv_dim]
    id<MTLBuffer> v_batch_buf;     // [S * kv_dim]
    id<MTLBuffer> hb_batch_buf;    // [S * hidden]
    id<MTLBuffer> hb2_batch_buf;   // [S * hidden]
    id<MTLBuffer> att_batch_scratch_buf;  // [S * n_heads * max_seq]

    // Batch pipelines
    id<MTLComputePipelineState> rmsnorm_batch_p;
    id<MTLComputePipelineState> rope_batch_p;
    id<MTLComputePipelineState> kv_write_batch_p;
    id<MTLComputePipelineState> gqa_att_causal_p;
} MetalContext;

// ─── Find shader source or metallib next to executable ───────────────────────
static NSString *metal_find_file(NSString *filename) {
    char exe[4096];
    uint32_t sz = sizeof(exe);
    if (_NSGetExecutablePath(exe, &sz) != 0) return nil;
    NSString *dir = [[NSString stringWithUTF8String:exe] stringByDeletingLastPathComponent];
    NSString *path = [dir stringByAppendingPathComponent:filename];
    if ([[NSFileManager defaultManager] fileExistsAtPath:path]) return path;
    return nil;
}

// ─── Load Metal library ─────────────────────────────────────────────────────
static id<MTLLibrary> metal_load_library(id<MTLDevice> device) {
    NSError *err = nil;

    NSString *libpath = metal_find_file(@"q4_0_metal.metallib");
    if (libpath) {
        id<MTLLibrary> lib = [device newLibraryWithURL:[NSURL fileURLWithPath:libpath] error:&err];
        if (lib) { fprintf(stderr, "Metal: loaded pre-compiled metallib\n"); return lib; }
    }

    NSString *srcpath = metal_find_file(@"q4_0_metal.metal");
    if (!srcpath) {
        fprintf(stderr, "Metal: q4_0_metal.metal not found\n");
        return nil;
    }

    NSString *source = [NSString stringWithContentsOfFile:srcpath
                                                 encoding:NSUTF8StringEncoding error:&err];
    if (!source) { fprintf(stderr, "Metal: failed to read shader source\n"); return nil; }

    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    opts.mathMode = MTLMathModeFast;

    id<MTLLibrary> lib = [device newLibraryWithSource:source options:opts error:&err];
    if (!lib) {
        fprintf(stderr, "Metal: shader compilation failed: %s\n",
                [[err localizedDescription] UTF8String]);
        return nil;
    }
    fprintf(stderr, "Metal: compiled shaders from source\n");
    return lib;
}

// ─── Initialize Metal context ────────────────────────────────────────────────
static MetalContext *metal_context_init(void *mmap_base, size_t mmap_len,
                                        int dim, int kv_dim, int hidden, int vocab,
                                        int n_heads, int n_layers, int max_seq) {
    MetalContext *ctx = (MetalContext *)calloc(1, sizeof(MetalContext));
    ctx->kv_dim = kv_dim;
    ctx->max_seq = max_seq;
    ctx->n_layers = n_layers;

    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) { fprintf(stderr, "Metal: no GPU\n"); free(ctx); return NULL; }
    ctx->queue = [ctx->device newCommandQueue];

    NSError *err = nil;
    id<MTLLibrary> lib = metal_load_library(ctx->device);
    if (!lib) { free(ctx); return NULL; }

    #define MAKE_PIPE(var, name) do { \
        id<MTLFunction> fn = [lib newFunctionWithName:@name]; \
        if (!fn) { fprintf(stderr, "Metal: shader '%s' not found\n", name); free(ctx); return NULL; } \
        var = [ctx->device newComputePipelineStateWithFunction:fn error:&err]; \
        if (!var) { fprintf(stderr, "Metal: pipeline '%s': %s\n", \
            name, [[err localizedDescription] UTF8String]); free(ctx); return NULL; } \
    } while(0)

    MAKE_PIPE(ctx->gemv_fast,  "q4_0_gemv_fast");
    MAKE_PIPE(ctx->gemv_max,   "q4_0_gemv_max");
    MAKE_PIPE(ctx->gemm,       "q4_0_gemm");
    MAKE_PIPE(ctx->rmsnorm_p,  "rmsnorm");
    MAKE_PIPE(ctx->silu_mul_p, "silu_mul");
    MAKE_PIPE(ctx->vadd_p,     "vadd");
    MAKE_PIPE(ctx->rope_p,     "rope");
    MAKE_PIPE(ctx->kv_write_p, "kv_cache_write");
    MAKE_PIPE(ctx->gqa_att_p,  "gqa_attention");

    // Batch pipelines (optional — created if shaders present, NULL otherwise)
    #define MAKE_PIPE_OPT(var, name) do { \
        id<MTLFunction> fn = [lib newFunctionWithName:@name]; \
        if (fn) { var = [ctx->device newComputePipelineStateWithFunction:fn error:&err]; } \
        else { var = nil; } \
    } while(0)
    MAKE_PIPE_OPT(ctx->rmsnorm_batch_p,  "rmsnorm_batch");
    MAKE_PIPE_OPT(ctx->rope_batch_p,     "rope_batch");
    MAKE_PIPE_OPT(ctx->kv_write_batch_p, "kv_cache_write_batch");
    MAKE_PIPE_OPT(ctx->gqa_att_causal_p, "gqa_attention_causal");
    #undef MAKE_PIPE_OPT

    #undef MAKE_PIPE

    // Wrap mmap as single Metal buffer (zero-copy)
    size_t page = getpagesize();
    size_t aligned_len = (mmap_len + page - 1) & ~(page - 1);
    ctx->weights_buf = [ctx->device newBufferWithBytesNoCopy:mmap_base
                                                      length:aligned_len
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
    if (!ctx->weights_buf) {
        fprintf(stderr, "Metal: failed to wrap mmap\n"); free(ctx); return NULL;
    }
    ctx->weights_base = mmap_base;

    // Scratch buffers
    #define ALLOC_BUF(var, bytes) do { \
        var = [ctx->device newBufferWithLength:(size_t)(bytes) \
                                      options:MTLResourceStorageModeShared]; \
        if (!var) { fprintf(stderr, "Metal: alloc failed\n"); free(ctx); return NULL; } \
    } while(0)

    ALLOC_BUF(ctx->x_buf,      dim * 4);
    ALLOC_BUF(ctx->xb_buf,     hidden * 4);
    ALLOC_BUF(ctx->xb2_buf,    dim * 4);
    ALLOC_BUF(ctx->q_buf,      dim * 4);
    ALLOC_BUF(ctx->k_buf,      kv_dim * 4);
    ALLOC_BUF(ctx->v_buf,      kv_dim * 4);
    ALLOC_BUF(ctx->hb_buf,     hidden * 4);
    ALLOC_BUF(ctx->hb2_buf,    hidden * 4);

    // KV cache: [n_layers * max_seq * kv_dim] fp16
    size_t kv_sz = (size_t)n_layers * max_seq * kv_dim * 2;  // sizeof(half) = 2
    ALLOC_BUF(ctx->k_cache_buf, kv_sz);
    ALLOC_BUF(ctx->v_cache_buf, kv_sz);
    memset([ctx->k_cache_buf contents], 0, kv_sz);
    memset([ctx->v_cache_buf contents], 0, kv_sz);

    // Attention scratch: [n_heads * max_seq] fp32
    ALLOC_BUF(ctx->att_scratch_buf, (size_t)n_heads * max_seq * 4);
    #undef ALLOC_BUF

    fprintf(stderr, "Metal: %s — all pipelines ready (1 CB/step)\n",
            [[ctx->device name] UTF8String]);
    fprintf(stderr, "Metal: KV cache %.1f MB (GPU shared)\n", kv_sz * 2.0 / 1e6);
    return ctx;
}

// ─── Set theta_inv buffer (call once after model load) ──────────────────────
static void metal_set_theta_inv(MetalContext *ctx, const float *theta_inv, int half_dim) {
    size_t sz = (size_t)half_dim * sizeof(float);
    ctx->theta_inv_buf = [ctx->device newBufferWithBytesNoCopy:(void *)theta_inv
                                                        length:sz
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];
}

// ─── Set per-layer RMSNorm weight buffers ───────────────────────────────────
static void metal_set_rmsnorm_bufs(MetalContext *ctx, int layer,
                                    const float *rms_att, const float *rms_ffn, int dim) {
    if (layer >= METAL_MAX_LAYERS) return;
    size_t sz = (size_t)dim * sizeof(float);
    ctx->rms_att_bufs[layer] = [ctx->device newBufferWithBytesNoCopy:(void *)rms_att
                                                              length:sz
                                                             options:MTLResourceStorageModeShared
                                                         deallocator:nil];
    ctx->rms_ffn_bufs[layer] = [ctx->device newBufferWithBytesNoCopy:(void *)rms_ffn
                                                              length:sz
                                                             options:MTLResourceStorageModeShared
                                                         deallocator:nil];
}

// ─── Allocate batch buffers for GEMM prefill / speculative verify ────────────
// max_att_len: max attention length per token (base_seq_len + S for the largest chunk)
static int metal_alloc_batch_buffers(MetalContext *ctx, int S, int dim, int kv_dim,
                                      int hidden, int n_heads, int max_att_len) {
    if (ctx->batch_S >= S) return 0;  // already large enough

    // Release old batch buffers if any
    ctx->x_batch_buf = nil;
    ctx->xb_batch_buf = nil;
    ctx->xb2_batch_buf = nil;
    ctx->q_batch_buf = nil;
    ctx->k_batch_buf = nil;
    ctx->v_batch_buf = nil;
    ctx->hb_batch_buf = nil;
    ctx->hb2_batch_buf = nil;
    ctx->att_batch_scratch_buf = nil;

    #define ALLOC_BATCH(var, bytes) do { \
        var = [ctx->device newBufferWithLength:(size_t)(bytes) \
                                      options:MTLResourceStorageModeShared]; \
        if (!var) { fprintf(stderr, "Metal: batch alloc failed\n"); return -1; } \
    } while(0)

    int max_dim = (dim > hidden) ? dim : hidden;
    ALLOC_BATCH(ctx->x_batch_buf,            (size_t)S * dim * 4);
    ALLOC_BATCH(ctx->xb_batch_buf,           (size_t)S * max_dim * 4);
    ALLOC_BATCH(ctx->xb2_batch_buf,          (size_t)S * dim * 4);
    ALLOC_BATCH(ctx->q_batch_buf,            (size_t)S * dim * 4);
    ALLOC_BATCH(ctx->k_batch_buf,            (size_t)S * kv_dim * 4);
    ALLOC_BATCH(ctx->v_batch_buf,            (size_t)S * kv_dim * 4);
    ALLOC_BATCH(ctx->hb_batch_buf,           (size_t)S * hidden * 4);
    ALLOC_BATCH(ctx->hb2_batch_buf,          (size_t)S * hidden * 4);
    ALLOC_BATCH(ctx->att_batch_scratch_buf,  (size_t)S * n_heads * max_att_len * 4);
    #undef ALLOC_BATCH

    ctx->batch_S = S;
    size_t total = (size_t)S * (dim + max_dim + dim + dim + kv_dim + kv_dim +
                                 hidden + hidden) * 4 +
                   (size_t)S * n_heads * max_att_len * 4;
    fprintf(stderr, "Metal: batch buffers allocated for S=%d att=%d (%.1f MB)\n",
            S, max_att_len, (double)total / 1e6);
    return 0;
}

// ─── Wire KV cache: point KVCache struct at Metal buffer contents ───────────
// Call after metal_context_init. Replaces the malloc'd KV cache with GPU-accessible memory.
static void metal_wire_kv_cache(MetalContext *ctx, void *kv_ptr) {
    // KVCache has k_cache and v_cache pointers — replace them
    // with pointers into the Metal shared buffers.
    // We cast via char* to access the struct fields portably.
    // KVCache layout: n_layers, n_kv_heads, head_dim, max_seq, kv_dim, k_cache, v_cache, ...
    typedef struct { int a,b,c,d,e; _Float16 *k; _Float16 *v; int f,g; } KVLayout;
    KVLayout *kv = (KVLayout *)kv_ptr;
    // Free old malloc'd cache
    free(kv->k);
    free(kv->v);
    // Point at Metal buffer contents
    kv->k = (_Float16 *)[ctx->k_cache_buf contents];
    kv->v = (_Float16 *)[ctx->v_cache_buf contents];
}

// ─── Compute byte offset within mmap buffer ─────────────────────────────────
static inline size_t metal_woff(MetalContext *ctx, const void *weight_ptr) {
    return (const uint8_t *)weight_ptr - (const uint8_t *)ctx->weights_base;
}

// ─── Encode helpers ─────────────────────────────────────────────────────────

static void metal_encode_gemv(MetalContext *ctx, id<MTLComputeCommandEncoder> enc,
                               const void *W_ptr, id<MTLBuffer> x_buf,
                               id<MTLBuffer> y_buf,
                               uint32_t out_dim, uint32_t in_dim) {
    size_t w_off = metal_woff(ctx, W_ptr);
    if (in_dim <= GEMV_FAST_LIMIT) {
        [enc setComputePipelineState:ctx->gemv_fast];
        [enc setBuffer:ctx->weights_buf offset:w_off atIndex:0];
        [enc setBuffer:x_buf offset:0 atIndex:1];
        [enc setBuffer:y_buf offset:0 atIndex:2];
        [enc setBytes:&out_dim length:4 atIndex:3];
        [enc setBytes:&in_dim  length:4 atIndex:4];
        uint32_t n_tg = (out_dim + GEMV_FAST_ROWS - 1) / GEMV_FAST_ROWS;
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(GEMV_FAST_TPG, 1, 1)];
    } else {
        [enc setComputePipelineState:ctx->gemv_max];
        [enc setBuffer:ctx->weights_buf offset:w_off atIndex:0];
        [enc setBuffer:x_buf offset:0 atIndex:1];
        [enc setBuffer:y_buf offset:0 atIndex:2];
        [enc setBytes:&out_dim length:4 atIndex:3];
        [enc setBytes:&in_dim  length:4 atIndex:4];
        uint32_t n_tg = (out_dim + GEMV_MAX_ROWS - 1) / GEMV_MAX_ROWS;
        [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(GEMV_MAX_TPG, 1, 1)];
    }
}

static void metal_encode_silu_mul(MetalContext *ctx, id<MTLComputeCommandEncoder> enc,
                                   id<MTLBuffer> gate_buf, id<MTLBuffer> up_buf,
                                   id<MTLBuffer> out_buf, uint32_t n) {
    [enc setComputePipelineState:ctx->silu_mul_p];
    [enc setBuffer:gate_buf offset:0 atIndex:0];
    [enc setBuffer:up_buf   offset:0 atIndex:1];
    [enc setBuffer:out_buf  offset:0 atIndex:2];
    [enc setBytes:&n length:4 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void metal_encode_vadd(MetalContext *ctx, id<MTLComputeCommandEncoder> enc,
                               id<MTLBuffer> a_buf, id<MTLBuffer> b_buf,
                               id<MTLBuffer> out_buf, uint32_t n) {
    [enc setComputePipelineState:ctx->vadd_p];
    [enc setBuffer:a_buf  offset:0 atIndex:0];
    [enc setBuffer:b_buf  offset:0 atIndex:1];
    [enc setBuffer:out_buf offset:0 atIndex:2];
    [enc setBytes:&n length:4 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void metal_encode_rmsnorm(MetalContext *ctx, id<MTLComputeCommandEncoder> enc,
                                  id<MTLBuffer> x_buf, id<MTLBuffer> out_buf,
                                  id<MTLBuffer> w_buf, uint32_t dim, float eps) {
    [enc setComputePipelineState:ctx->rmsnorm_p];
    [enc setBuffer:x_buf   offset:0 atIndex:0];
    [enc setBuffer:out_buf offset:0 atIndex:1];
    [enc setBuffer:w_buf   offset:0 atIndex:2];
    [enc setBytes:&dim length:4 atIndex:3];
    [enc setBytes:&eps length:4 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void metal_encode_rope(MetalContext *ctx, id<MTLComputeCommandEncoder> enc,
                               id<MTLBuffer> q_buf, id<MTLBuffer> k_buf,
                               uint32_t pos, uint32_t n_heads, uint32_t n_kv_heads,
                               uint32_t head_dim) {
    [enc setComputePipelineState:ctx->rope_p];
    [enc setBuffer:q_buf offset:0 atIndex:0];
    [enc setBuffer:k_buf offset:0 atIndex:1];
    [enc setBytes:&pos length:4 atIndex:2];
    [enc setBuffer:ctx->theta_inv_buf offset:0 atIndex:3];
    [enc setBytes:&n_heads length:4 atIndex:4];
    [enc setBytes:&n_kv_heads length:4 atIndex:5];
    [enc setBytes:&head_dim length:4 atIndex:6];
    uint32_t n_threads = n_heads * (head_dim / 2);
    [enc dispatchThreadgroups:MTLSizeMake((n_threads + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void metal_encode_kv_write(MetalContext *ctx, id<MTLComputeCommandEncoder> enc,
                                   id<MTLBuffer> k_fp32, id<MTLBuffer> v_fp32,
                                   int layer, uint32_t cache_pos, uint32_t kv_dim) {
    size_t layer_off = (size_t)layer * ctx->max_seq * kv_dim * 2;  // bytes
    [enc setComputePipelineState:ctx->kv_write_p];
    [enc setBuffer:k_fp32 offset:0 atIndex:0];
    [enc setBuffer:v_fp32 offset:0 atIndex:1];
    [enc setBuffer:ctx->k_cache_buf offset:layer_off atIndex:2];
    [enc setBuffer:ctx->v_cache_buf offset:layer_off atIndex:3];
    [enc setBytes:&cache_pos length:4 atIndex:4];
    [enc setBytes:&kv_dim length:4 atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((kv_dim + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void metal_encode_gqa_attention(MetalContext *ctx, id<MTLComputeCommandEncoder> enc,
                                        id<MTLBuffer> q_buf, id<MTLBuffer> out_buf,
                                        int layer, uint32_t n_heads, uint32_t n_kv_heads,
                                        uint32_t head_dim, uint32_t kv_dim,
                                        uint32_t seq_len, uint32_t max_seq, uint32_t ring_off) {
    size_t layer_off = (size_t)layer * max_seq * kv_dim * 2;
    [enc setComputePipelineState:ctx->gqa_att_p];
    [enc setBuffer:q_buf offset:0 atIndex:0];
    [enc setBuffer:ctx->k_cache_buf offset:layer_off atIndex:1];
    [enc setBuffer:ctx->v_cache_buf offset:layer_off atIndex:2];
    [enc setBuffer:out_buf offset:0 atIndex:3];
    [enc setBytes:&n_heads length:4 atIndex:4];
    [enc setBytes:&n_kv_heads length:4 atIndex:5];
    [enc setBytes:&head_dim length:4 atIndex:6];
    [enc setBytes:&kv_dim length:4 atIndex:7];
    [enc setBytes:&seq_len length:4 atIndex:8];
    [enc setBytes:&max_seq length:4 atIndex:9];
    [enc setBytes:&ring_off length:4 atIndex:10];
    [enc setBuffer:ctx->att_scratch_buf offset:0 atIndex:11];
    [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(GQA_THREADS, 1, 1)];
}

// ─── Batch encode helpers (speculative verify) ──────────────────────────────

static void metal_encode_gemm(MetalContext *ctx, id<MTLComputeCommandEncoder> enc,
                                const void *W_ptr, id<MTLBuffer> x_buf,
                                id<MTLBuffer> y_buf,
                                uint32_t out_dim, uint32_t in_dim, uint32_t S) {
    size_t w_off = metal_woff(ctx, W_ptr);
    [enc setComputePipelineState:ctx->gemm];
    [enc setBuffer:ctx->weights_buf offset:w_off atIndex:0];
    [enc setBuffer:x_buf offset:0 atIndex:1];
    [enc setBuffer:y_buf offset:0 atIndex:2];
    [enc setBytes:&out_dim length:4 atIndex:3];
    [enc setBytes:&in_dim  length:4 atIndex:4];
    [enc setBytes:&S       length:4 atIndex:5];
    uint32_t gx = (out_dim + 31) / 32;
    uint32_t gy = (S + 15) / 16;
    [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static void metal_encode_rmsnorm_batch(MetalContext *ctx, id<MTLComputeCommandEncoder> enc,
                                        id<MTLBuffer> x_buf, id<MTLBuffer> out_buf,
                                        id<MTLBuffer> w_buf, uint32_t dim, float eps,
                                        uint32_t S) {
    [enc setComputePipelineState:ctx->rmsnorm_batch_p];
    [enc setBuffer:x_buf   offset:0 atIndex:0];
    [enc setBuffer:out_buf offset:0 atIndex:1];
    [enc setBuffer:w_buf   offset:0 atIndex:2];
    [enc setBytes:&dim length:4 atIndex:3];
    [enc setBytes:&eps length:4 atIndex:4];
    [enc setBytes:&S   length:4 atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(S, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void metal_encode_rope_batch(MetalContext *ctx, id<MTLComputeCommandEncoder> enc,
                                     id<MTLBuffer> q_buf, id<MTLBuffer> k_buf,
                                     uint32_t base_pos, uint32_t n_heads, uint32_t n_kv,
                                     uint32_t head_dim, uint32_t S) {
    uint32_t max_hg = (n_heads > n_kv) ? n_heads : n_kv;
    uint32_t n_threads = S * max_hg * (head_dim / 2);
    [enc setComputePipelineState:ctx->rope_batch_p];
    [enc setBuffer:q_buf offset:0 atIndex:0];
    [enc setBuffer:k_buf offset:0 atIndex:1];
    [enc setBytes:&base_pos length:4 atIndex:2];
    [enc setBuffer:ctx->theta_inv_buf offset:0 atIndex:3];
    [enc setBytes:&n_heads  length:4 atIndex:4];
    [enc setBytes:&n_kv     length:4 atIndex:5];
    [enc setBytes:&head_dim length:4 atIndex:6];
    [enc setBytes:&S        length:4 atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake((n_threads + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void metal_encode_kv_write_batch(MetalContext *ctx, id<MTLComputeCommandEncoder> enc,
                                          id<MTLBuffer> k_buf, id<MTLBuffer> v_buf,
                                          int layer, uint32_t base_cache_pos,
                                          uint32_t kv_dim, uint32_t max_seq, uint32_t S) {
    size_t layer_off = (size_t)layer * max_seq * kv_dim * 2;
    uint32_t n_threads = S * kv_dim;
    [enc setComputePipelineState:ctx->kv_write_batch_p];
    [enc setBuffer:k_buf offset:0 atIndex:0];
    [enc setBuffer:v_buf offset:0 atIndex:1];
    [enc setBuffer:ctx->k_cache_buf offset:layer_off atIndex:2];
    [enc setBuffer:ctx->v_cache_buf offset:layer_off atIndex:3];
    [enc setBytes:&base_cache_pos length:4 atIndex:4];
    [enc setBytes:&kv_dim  length:4 atIndex:5];
    [enc setBytes:&max_seq length:4 atIndex:6];
    [enc setBytes:&S       length:4 atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake((n_threads + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void metal_encode_gqa_attention_causal(MetalContext *ctx, id<MTLComputeCommandEncoder> enc,
                                                id<MTLBuffer> q_buf, id<MTLBuffer> out_buf,
                                                int layer, uint32_t n_heads, uint32_t n_kv,
                                                uint32_t head_dim, uint32_t kv_dim,
                                                uint32_t base_seq_len, uint32_t max_seq,
                                                uint32_t ring_off, uint32_t S,
                                                uint32_t att_stride) {
    size_t layer_off = (size_t)layer * max_seq * kv_dim * 2;
    [enc setComputePipelineState:ctx->gqa_att_causal_p];
    [enc setBuffer:q_buf offset:0 atIndex:0];
    [enc setBuffer:ctx->k_cache_buf offset:layer_off atIndex:1];
    [enc setBuffer:ctx->v_cache_buf offset:layer_off atIndex:2];
    [enc setBuffer:out_buf offset:0 atIndex:3];
    [enc setBytes:&n_heads      length:4 atIndex:4];
    [enc setBytes:&n_kv         length:4 atIndex:5];
    [enc setBytes:&head_dim     length:4 atIndex:6];
    [enc setBytes:&kv_dim       length:4 atIndex:7];
    [enc setBytes:&base_seq_len length:4 atIndex:8];
    [enc setBytes:&max_seq      length:4 atIndex:9];
    [enc setBytes:&ring_off     length:4 atIndex:10];
    [enc setBuffer:ctx->att_batch_scratch_buf offset:0 atIndex:11];
    [enc setBytes:&S            length:4 atIndex:12];
    [enc setBytes:&att_stride   length:4 atIndex:13];
    [enc dispatchThreadgroups:MTLSizeMake(n_heads * S, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(GQA_THREADS, 1, 1)];
}

// ─── Cleanup ─────────────────────────────────────────────────────────────────
static void metal_context_free(MetalContext *ctx) {
    if (!ctx) return;
    ctx->device = nil; ctx->queue = nil;
    ctx->gemv_fast = nil; ctx->gemv_max = nil; ctx->gemm = nil;
    ctx->rmsnorm_p = nil; ctx->silu_mul_p = nil; ctx->vadd_p = nil;
    ctx->rope_p = nil; ctx->kv_write_p = nil; ctx->gqa_att_p = nil;
    ctx->weights_buf = nil; ctx->theta_inv_buf = nil;
    ctx->x_buf = nil; ctx->xb_buf = nil; ctx->xb2_buf = nil;
    ctx->q_buf = nil; ctx->k_buf = nil; ctx->v_buf = nil;
    ctx->hb_buf = nil; ctx->hb2_buf = nil;
    ctx->k_cache_buf = nil; ctx->v_cache_buf = nil;
    ctx->att_scratch_buf = nil;
    // Batch buffers
    ctx->x_batch_buf = nil; ctx->xb_batch_buf = nil; ctx->xb2_batch_buf = nil;
    ctx->q_batch_buf = nil; ctx->k_batch_buf = nil; ctx->v_batch_buf = nil;
    ctx->hb_batch_buf = nil; ctx->hb2_batch_buf = nil;
    ctx->att_batch_scratch_buf = nil;
    // Batch pipelines
    ctx->rmsnorm_batch_p = nil; ctx->rope_batch_p = nil;
    ctx->kv_write_batch_p = nil; ctx->gqa_att_causal_p = nil;
    for (int i = 0; i < ctx->n_layers; i++) {
        ctx->rms_att_bufs[i] = nil;
        ctx->rms_ffn_bufs[i] = nil;
    }
    free(ctx);
}
