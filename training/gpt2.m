// gpt2.m — GPT-2 (124M) inference on Apple Neural Engine
// Fused attention + fused FFN per layer, CPU LayerNorm + residual + embedding
// Two modes: recompute (full sequence each step) and KV-cache decode
#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>
#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <Accelerate/Accelerate.h>

#include "gpt2_tokenizer.h"

#define DIM 768
#define HEADS 12
#define HD (DIM/HEADS)
#define HIDDEN 3072
#define N_LAYERS 12
#define VOCAB 50257
#define MAX_SEQ 1024

#define MIL_HDR \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"
#define CONV_CONST \
    "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n" \
    "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n" \
    "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"

// --- ANE boilerplate ---
static Class g_D, g_I, g_AR, g_AIO;
static mach_timebase_info_data_t g_tb;
static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
}
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}
typedef struct { id model; NSString *td; } Kern;
static Kern compile_mil(NSString *mil, NSDictionary *wd) {
    Kern k = {nil, nil};
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), md, wd ?: @{}, nil);
    if (!desc) { printf("  desc=NULL\n"); return k; }
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (NSString *path in wd) {
        NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
        NSString *full = [td stringByAppendingPathComponent:rel];
        [[NSFileManager defaultManager] createDirectoryAtPath:[full stringByDeletingLastPathComponent]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [wd[path][@"data"] writeToFile:full atomically:YES];
    }
    NSError *e = nil;
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        printf("  compile FAIL: %s\n", e?[[[e localizedDescription] substringToIndex:MIN(300,(int)[[e localizedDescription] length])] UTF8String]:"");
        [[NSFileManager defaultManager] removeItemAtPath:td error:nil]; return k;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        printf("  load FAIL\n"); [[NSFileManager defaultManager] removeItemAtPath:td error:nil]; return k;
    }
    k.model = mdl; k.td = td;
    return k;
}
static BOOL ane_eval(Kern *k, IOSurfaceRef *ins, int nin, IOSurfaceRef *outs, int nout) {
    NSMutableArray *inArr = [NSMutableArray array], *inIdx = [NSMutableArray array];
    NSMutableArray *outArr = [NSMutableArray array], *outIdx = [NSMutableArray array];
    for (int i = 0; i < nin; i++) {
        [inArr addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ins[i])];
        [inIdx addObject:@(i)];
    }
    for (int i = 0; i < nout; i++) {
        [outArr addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), outs[i])];
        [outIdx addObject:@(i)];
    }
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        inArr, inIdx, outArr, outIdx, nil, nil, @0);
    NSError *e = nil;
    return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
}
static void cleanup_kern(Kern *k) {
    if (!k->model) return;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(k->model, @selector(unloadWithQoS:error:), 21, &e);
    [[NSFileManager defaultManager] removeItemAtPath:k->td error:nil];
    k->model = nil;
}

// --- Weight loading ---
typedef struct {
    _Float16 *wte;    // [VOCAB, DIM]
    _Float16 *wpe;    // [MAX_SEQ, DIM]
    _Float16 *ln_f_w; // [DIM]
    _Float16 *ln_f_b; // [DIM]
    struct {
        NSData *wq, *wk, *wv, *wo;  // conv weight blobs
        NSData *bq, *bk, *bv, *bo;  // bias blobs
        _Float16 *ln1_w, *ln1_b;    // layernorm params (CPU)
        _Float16 *ln2_w, *ln2_b;
        NSData *w1, *w2;             // FFN weight blobs
        NSData *b1, *b2;             // FFN bias blobs
    } layer[N_LAYERS];
} Weights;

static _Float16 *load_fp16_raw(const char *path, int count) {
    NSData *d = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
    if (!d || (int)[d length] < 128 + count*2) { printf("FAIL loading %s\n", path); return NULL; }
    _Float16 *buf = (_Float16*)malloc(count * 2);
    memcpy(buf, (uint8_t*)[d bytes] + 128, count * 2);
    return buf;
}
static NSData *load_blob(const char *path) {
    NSData *d = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
    if (!d) printf("FAIL loading %s\n", path);
    return d;
}

static BOOL load_weights(Weights *w, const char *dir) {
    char p[512];
    snprintf(p,512,"%s/wte.bin",dir); w->wte = load_fp16_raw(p, VOCAB*DIM);
    snprintf(p,512,"%s/wpe.bin",dir); w->wpe = load_fp16_raw(p, MAX_SEQ*DIM);
    snprintf(p,512,"%s/ln_f_w.bin",dir); w->ln_f_w = load_fp16_raw(p, DIM);
    snprintf(p,512,"%s/ln_f_b.bin",dir); w->ln_f_b = load_fp16_raw(p, DIM);
    if (!w->wte || !w->wpe || !w->ln_f_w || !w->ln_f_b) return NO;

    for (int i = 0; i < N_LAYERS; i++) {
        char ld[512]; snprintf(ld,512,"%s/layer_%02d",dir,i);
        snprintf(p,512,"%s/ln1_w.bin",ld); w->layer[i].ln1_w = load_fp16_raw(p, DIM);
        snprintf(p,512,"%s/ln1_b.bin",ld); w->layer[i].ln1_b = load_fp16_raw(p, DIM);
        snprintf(p,512,"%s/ln2_w.bin",ld); w->layer[i].ln2_w = load_fp16_raw(p, DIM);
        snprintf(p,512,"%s/ln2_b.bin",ld); w->layer[i].ln2_b = load_fp16_raw(p, DIM);
        snprintf(p,512,"%s/wq.bin",ld); w->layer[i].wq = load_blob(p);
        snprintf(p,512,"%s/wk.bin",ld); w->layer[i].wk = load_blob(p);
        snprintf(p,512,"%s/wv.bin",ld); w->layer[i].wv = load_blob(p);
        snprintf(p,512,"%s/wo.bin",ld); w->layer[i].wo = load_blob(p);
        snprintf(p,512,"%s/bq.bin",ld); w->layer[i].bq = load_blob(p);
        snprintf(p,512,"%s/bk.bin",ld); w->layer[i].bk = load_blob(p);
        snprintf(p,512,"%s/bv.bin",ld); w->layer[i].bv = load_blob(p);
        snprintf(p,512,"%s/bo.bin",ld); w->layer[i].bo = load_blob(p);
        snprintf(p,512,"%s/w1.bin",ld); w->layer[i].w1 = load_blob(p);
        snprintf(p,512,"%s/w2.bin",ld); w->layer[i].w2 = load_blob(p);
        snprintf(p,512,"%s/b1.bin",ld); w->layer[i].b1 = load_blob(p);
        snprintf(p,512,"%s/b2.bin",ld); w->layer[i].b2 = load_blob(p);
        if (!w->layer[i].wq || !w->layer[i].ln1_w) return NO;
    }
    return YES;
}

// --- Decode weights (fp16 for NEON) ---
typedef struct {
    __fp16 *wqkv;           // [3*DIM, DIM] fused QKV
    __fp16 *wo;             // [DIM, DIM]
    float *bq, *bk, *bv, *bo;  // [DIM] each — biases stay fp32 (tiny)
    __fp16 *w1;             // [HIDDEN, DIM]
    __fp16 *w2;             // [DIM, HIDDEN]
    float *b1, *b2;         // [HIDDEN], [DIM]
} LayerF16;

static __fp16 *blob_to_f16(NSData *blob, int count) {
    __fp16 *out = (__fp16*)malloc(count * sizeof(__fp16));
    memcpy(out, (uint8_t*)[blob bytes] + 128, count * sizeof(__fp16));
    return out;
}
static float *blob_to_f32(NSData *blob, int count) {
    float *out = (float*)malloc(count * sizeof(float));
    const _Float16 *src = (const _Float16 *)((uint8_t*)[blob bytes] + 128);
    for (int i = 0; i < count; i++) out[i] = (float)src[i];
    return out;
}

// --- KV cache ---
typedef struct {
    float *k;  // [DIM x MAX_SEQ], row-major: k[d * MAX_SEQ + t]
    float *v;  // [DIM x MAX_SEQ]
} KVCache;

// --- CPU ops ---
static void layer_norm(float *out, const float *x, const _Float16 *w, const _Float16 *b, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        const float *xt = x + t*dim;
        float *ot = out + t*dim;
        float mean = 0;
        for (int i = 0; i < dim; i++) mean += xt[i];
        mean /= dim;
        float var = 0;
        for (int i = 0; i < dim; i++) { float d = xt[i] - mean; var += d*d; }
        float inv = 1.0f / sqrtf(var / dim + 1e-5f);
        for (int i = 0; i < dim; i++)
            ot[i] = (xt[i] - mean) * inv * (float)w[i] + (float)b[i];
    }
}

static void embed_tokens(float *out, const int *tokens, const _Float16 *wte, const _Float16 *wpe, int seq) {
    for (int t = 0; t < seq; t++)
        for (int d = 0; d < DIM; d++)
            out[t*DIM+d] = (float)wte[tokens[t]*DIM+d] + (float)wpe[t*DIM+d];
}

// LM head: NEON fp16 + GCD parallel matmul, then sample
// wte_f16 is [VOCAB x DIM] in fp16, hidden is [DIM] in fp32
static inline float hsum_f32x4(float32x4_t v) {
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(s, s), 0);
}
#define LM_CHUNK 512
static int lm_head_sample(const float *hidden, const __fp16 *wte_f16, float temperature) {
    __fp16 *xh = (__fp16*)malloc(DIM * sizeof(__fp16));
    for (int i = 0; i < DIM; i += 8) {
        float32x4_t a = vld1q_f32(hidden + i);
        float32x4_t b = vld1q_f32(hidden + i + 4);
        vst1_f16(xh + i, vcvt_f16_f32(a));
        vst1_f16(xh + i + 4, vcvt_f16_f32(b));
    }
    float *logits = (float*)malloc(VOCAB * sizeof(float));
    int nchunks = (VOCAB + LM_CHUNK - 1) / LM_CHUNK;
    dispatch_apply(nchunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t chunk) {
        int rstart = (int)chunk * LM_CHUNK;
        int rend = rstart + LM_CHUNK;
        if (rend > VOCAB) rend = VOCAB;
        int r = rstart;
        for (; r + 4 <= rend; r += 4) {
            const __fp16 *r0 = wte_f16 + (size_t)(r+0) * DIM;
            const __fp16 *r1 = wte_f16 + (size_t)(r+1) * DIM;
            const __fp16 *r2 = wte_f16 + (size_t)(r+2) * DIM;
            const __fp16 *r3 = wte_f16 + (size_t)(r+3) * DIM;
            float16x8_t a0 = vdupq_n_f16(0), a1 = vdupq_n_f16(0);
            float16x8_t a2 = vdupq_n_f16(0), a3 = vdupq_n_f16(0);
            for (int i = 0; i < DIM; i += 8) {
                float16x8_t xv = vld1q_f16(xh + i);
                a0 = vfmaq_f16(a0, vld1q_f16(r0 + i), xv);
                a1 = vfmaq_f16(a1, vld1q_f16(r1 + i), xv);
                a2 = vfmaq_f16(a2, vld1q_f16(r2 + i), xv);
                a3 = vfmaq_f16(a3, vld1q_f16(r3 + i), xv);
            }
            logits[r+0] = hsum_f32x4(vcvt_f32_f16(vadd_f16(vget_low_f16(a0), vget_high_f16(a0))));
            logits[r+1] = hsum_f32x4(vcvt_f32_f16(vadd_f16(vget_low_f16(a1), vget_high_f16(a1))));
            logits[r+2] = hsum_f32x4(vcvt_f32_f16(vadd_f16(vget_low_f16(a2), vget_high_f16(a2))));
            logits[r+3] = hsum_f32x4(vcvt_f32_f16(vadd_f16(vget_low_f16(a3), vget_high_f16(a3))));
        }
        for (; r < rend; r++) {
            const __fp16 *row = wte_f16 + (size_t)r * DIM;
            float16x8_t acc = vdupq_n_f16(0);
            for (int i = 0; i < DIM; i += 8)
                acc = vfmaq_f16(acc, vld1q_f16(row + i), vld1q_f16(xh + i));
            logits[r] = hsum_f32x4(vcvt_f32_f16(vadd_f16(vget_low_f16(acc), vget_high_f16(acc))));
        }
    });
    int best = 0;
    if (temperature <= 0) {
        float bestv = -1e30f;
        for (int v = 0; v < VOCAB; v++)
            if (logits[v] > bestv) { bestv = logits[v]; best = v; }
    } else {
        float maxl = -1e30f;
        for (int v = 0; v < VOCAB; v++) if (logits[v] > maxl) maxl = logits[v];
        float sum = 0;
        for (int v = 0; v < VOCAB; v++) { logits[v] = expf((logits[v]-maxl)/temperature); sum += logits[v]; }
        float r = (float)drand48() * sum;
        float acc = 0;
        for (int v = 0; v < VOCAB; v++) { acc += logits[v]; if (acc >= r) { best = v; break; } }
    }
    free(xh);
    free(logits);
    return best;
}

// NEON fp16 matvec core: rows [rstart, rend) of W[M,K] @ x[K] → out[rstart..rend)
static void neon_f16_matvec_range(float *out, const __fp16 *W, const __fp16 *x, int rstart, int rend, int K) {
    int r = rstart;
    for (; r + 4 <= rend; r += 4) {
        const __fp16 *r0 = W + (size_t)(r+0) * K;
        const __fp16 *r1 = W + (size_t)(r+1) * K;
        const __fp16 *r2 = W + (size_t)(r+2) * K;
        const __fp16 *r3 = W + (size_t)(r+3) * K;
        float16x8_t a0 = vdupq_n_f16(0), a1 = vdupq_n_f16(0);
        float16x8_t a2 = vdupq_n_f16(0), a3 = vdupq_n_f16(0);
        for (int i = 0; i < K; i += 8) {
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
        const __fp16 *row = W + (size_t)r * K;
        float16x8_t acc = vdupq_n_f16(0);
        for (int i = 0; i < K; i += 8)
            acc = vfmaq_f16(acc, vld1q_f16(row + i), vld1q_f16(x + i));
        out[r] = hsum_f32x4(vcvt_f32_f16(vadd_f16(vget_low_f16(acc), vget_high_f16(acc))));
    }
}

// Serial matvec
static void neon_f16_matvec(float *out, const __fp16 *W, const __fp16 *x, int M, int K) {
    neon_f16_matvec_range(out, W, x, 0, M, K);
}


// --- CPU decode: single-token attention with KV cache ---
static void cpu_attn_decode(float *out, const float *x, int pos,
                            LayerF16 *lw, KVCache *kv) {
    // Convert x to fp16 once
    __fp16 xh[DIM];
    for (int i = 0; i < DIM; i += 8) {
        vst1_f16(xh + i, vcvt_f16_f32(vld1q_f32(x + i)));
        vst1_f16(xh + i + 4, vcvt_f16_f32(vld1q_f32(x + i + 4)));
    }

    // Fused QKV: [3*DIM, DIM] @ x → qkv[3*DIM]
    float qkv[3 * DIM];
    neon_f16_matvec(qkv, lw->wqkv, xh, 3 * DIM, DIM);
    float *q = qkv, *k_new = qkv + DIM, *v_new = qkv + 2*DIM;
    for (int d = 0; d < DIM; d++) { q[d] += lw->bq[d]; k_new[d] += lw->bk[d]; v_new[d] += lw->bv[d]; }

    // Write to cache
    for (int d = 0; d < DIM; d++) {
        kv->k[d * MAX_SEQ + pos] = k_new[d];
        kv->v[d * MAX_SEQ + pos] = v_new[d];
    }

    int T = pos + 1;
    float scale = 1.0f / sqrtf((float)HD);

    // Multi-head attention
    float attn_out[DIM];
    for (int h = 0; h < HEADS; h++) {
        int off = h * HD;
        float scores[MAX_SEQ];
        float maxs = -1e30f;
        for (int t = 0; t < T; t++) {
            float s = 0;
            for (int d = 0; d < HD; d++)
                s += q[off+d] * kv->k[(off+d) * MAX_SEQ + t];
            scores[t] = s * scale;
            if (scores[t] > maxs) maxs = scores[t];
        }
        float sum = 0;
        for (int t = 0; t < T; t++) { scores[t] = expf(scores[t] - maxs); sum += scores[t]; }
        float inv = 1.0f / sum;
        for (int t = 0; t < T; t++) scores[t] *= inv;
        for (int d = 0; d < HD; d++) {
            float s = 0;
            for (int t = 0; t < T; t++)
                s += scores[t] * kv->v[(off+d) * MAX_SEQ + t];
            attn_out[off+d] = s;
        }
    }
    // Wo @ attn_out + bo — convert attn_out to fp16 for NEON matvec
    __fp16 attn_h[DIM];
    for (int i = 0; i < DIM; i += 8) {
        vst1_f16(attn_h + i, vcvt_f16_f32(vld1q_f32(attn_out + i)));
        vst1_f16(attn_h + i + 4, vcvt_f16_f32(vld1q_f32(attn_out + i + 4)));
    }
    neon_f16_matvec(out, lw->wo, attn_h, DIM, DIM);
    for (int d = 0; d < DIM; d++) out[d] += lw->bo[d];
}

// CPU decode FFN: single token, NEON fp16 weights
static void cpu_ffn_decode(float *out, const float *x, LayerF16 *lw) {
    // Convert x to fp16
    __fp16 xh[DIM];
    for (int i = 0; i < DIM; i += 8) {
        vst1_f16(xh + i, vcvt_f16_f32(vld1q_f32(x + i)));
        vst1_f16(xh + i + 4, vcvt_f16_f32(vld1q_f32(x + i + 4)));
    }
    // h = W1 @ x + b1
    float h[HIDDEN];
    neon_f16_matvec(h, lw->w1, xh, HIDDEN, DIM);
    for (int d = 0; d < HIDDEN; d++) h[d] += lw->b1[d];
    // GeLU
    for (int d = 0; d < HIDDEN; d++) {
        float v = h[d];
        h[d] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
    // Convert h to fp16 for W2 matvec
    __fp16 hh[HIDDEN];
    for (int i = 0; i < HIDDEN; i += 8) {
        vst1_f16(hh + i, vcvt_f16_f32(vld1q_f32(h + i)));
        vst1_f16(hh + i + 4, vcvt_f16_f32(vld1q_f32(h + i + 4)));
    }
    // out = W2 @ h + b2
    neon_f16_matvec(out, lw->w2, hh, DIM, HIDDEN);
    for (int d = 0; d < DIM; d++) out[d] += lw->b2[d];
}

// --- MIL generators ---

// Fused attention: QKV convs + reshape + matmul + scale + mask + softmax + matmul + reshape + Wo
// GPT-2: includes bias on all convs, causal mask
static NSString *gen_attn_mil(int layer, int seq) {
    float scale_val = 1.0f / sqrtf((float)HD);
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, seq];
    [m appendString:@CONV_CONST];
    // Weight + bias declarations
    NSString *ld = [NSString stringWithFormat:@"layer_%02d", layer];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@/wq.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM,ld];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> Bq = const()[name=string(\"Bq\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@/bq.bin\"), offset=uint64(64)))];\n", DIM,DIM,ld];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@/wk.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM,ld];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> Bk = const()[name=string(\"Bk\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@/bk.bin\"), offset=uint64(64)))];\n", DIM,DIM,ld];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@/wv.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM,ld];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> Bv = const()[name=string(\"Bv\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@/bv.bin\"), offset=uint64(64)))];\n", DIM,DIM,ld];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@/wo.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM,ld];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> Bo = const()[name=string(\"Bo\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@/bo.bin\"), offset=uint64(64)))];\n", DIM,DIM,ld];
    // QKV projections + bias (conv then add)
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qc = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=x)[name=string(\"cq\")];\n", DIM,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = add(x=qc,y=Bq)[name=string(\"aq\")];\n", DIM,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kc = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=x)[name=string(\"ck\")];\n", DIM,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = add(x=kc,y=Bk)[name=string(\"ak\")];\n", DIM,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vc = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=x)[name=string(\"cv\")];\n", DIM,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = add(x=vc,y=Bv)[name=string(\"av\")];\n", DIM,seq];
    // Reshape + transpose to multi-head
    [m appendFormat:@"        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS,HD,seq];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q4 = reshape(shape=qsh,x=qf)[name=string(\"rq\")];\n", HEADS,HD,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];\n", HEADS,seq,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k4 = reshape(shape=qsh,x=kf)[name=string(\"rk\")];\n", HEADS,HD,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=k4)[name=string(\"tk\")];\n", HEADS,seq,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v4 = reshape(shape=qsh,x=vf)[name=string(\"rv\")];\n", HEADS,HD,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=v4)[name=string(\"tv\")];\n", HEADS,seq,HD];
    // Q @ K^T
    [m appendString:@"        bool tx = const()[name=string(\"tx\"), val=bool(false)];\n"];
    [m appendString:@"        bool ty = const()[name=string(\"ty\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=tx,transpose_y=ty,x=q,y=k)[name=string(\"mm1\")];\n", HEADS,seq,seq];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", scale_val];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n", HEADS,seq,seq];
    // Causal mask
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];\n", seq,seq,seq,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];\n", HEADS,seq,seq];
    [m appendString:@"        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n", HEADS,seq,seq];
    // scores @ V
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> a4 = matmul(transpose_x=tx,transpose_y=tx,x=aw,y=v)[name=string(\"mm2\")];\n", HEADS,seq,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> at = transpose(perm=pm,x=a4)[name=string(\"ta\")];\n", HEADS,HD,seq];
    [m appendFormat:@"        tensor<int32, [4]> os = const()[name=string(\"os\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> af = reshape(shape=os,x=at)[name=string(\"ra\")];\n", DIM,seq];
    // Wo projection + bias
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> oc = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wo,x=af)[name=string(\"co\")];\n", DIM,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = add(x=oc,y=Bo)[name=string(\"ao\")];\n", DIM,seq];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// Fused FFN: W1+GeLU+W2 (GPT-2 uses GeLU, not gated SiLU)
// GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
static NSString *gen_ffn_mil(int layer, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, seq];
    [m appendString:@CONV_CONST];
    NSString *ld = [NSString stringWithFormat:@"layer_%02d", layer];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W1 = const()[name=string(\"W1\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@/w1.bin\"), offset=uint64(64)))];\n", HIDDEN,DIM,HIDDEN,DIM,ld];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> B1 = const()[name=string(\"B1\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@/b1.bin\"), offset=uint64(64)))];\n", HIDDEN,HIDDEN,ld];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W2 = const()[name=string(\"W2\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@/w2.bin\"), offset=uint64(64)))];\n", DIM,HIDDEN,DIM,HIDDEN,ld];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> B2 = const()[name=string(\"B2\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/%@/b2.bin\"), offset=uint64(64)))];\n", DIM,DIM,ld];
    // W1 @ x + b1
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> hc = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=x)[name=string(\"c1\")];\n", HIDDEN,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h = add(x=hc,y=B1)[name=string(\"ab1\")];\n", HIDDEN,seq];
    // GeLU: 0.5 * h * (1 + tanh(sqrt(2/pi) * (h + 0.044715 * h^3)))
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h2 = mul(x=h,y=h)[name=string(\"h2\")];\n", HIDDEN,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = mul(x=h2,y=h)[name=string(\"h3\")];\n", HIDDEN,seq];
    [m appendFormat:@"        fp16 c1 = const()[name=string(\"c1v\"), val=fp16(0.044715)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> t1 = mul(x=h3,y=c1)[name=string(\"t1\")];\n", HIDDEN,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> t2 = add(x=h,y=t1)[name=string(\"t2\")];\n", HIDDEN,seq];
    [m appendFormat:@"        fp16 c2 = const()[name=string(\"c2v\"), val=fp16(0.7978845608)];\n"]; // sqrt(2/pi)
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> t3 = mul(x=t2,y=c2)[name=string(\"t3\")];\n", HIDDEN,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> t4 = tanh(x=t3)[name=string(\"t4\")];\n", HIDDEN,seq];
    [m appendFormat:@"        fp16 c3 = const()[name=string(\"c3v\"), val=fp16(1.0)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> t5 = add(x=t4,y=c3)[name=string(\"t5\")];\n", HIDDEN,seq];
    [m appendFormat:@"        fp16 c4 = const()[name=string(\"c4v\"), val=fp16(0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> t6 = mul(x=h,y=c4)[name=string(\"t6\")];\n", HIDDEN,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> gelu = mul(x=t6,y=t5)[name=string(\"gelu\")];\n", HIDDEN,seq];
    // W2 @ gelu + b2
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> oc2 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=gelu)[name=string(\"c2\")];\n", DIM,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = add(x=oc2,y=B2)[name=string(\"ab2\")];\n", DIM,seq];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// --- Main ---
int main(int argc, char **argv) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        srand48(42);
        mach_timebase_info(&g_tb);
        ane_init();

        const char *weights_dir = "gpt2_weights";
        const char *prompt = "The meaning of life is";
        int max_tokens = 100;
        float temperature = 0.8f;

        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--prompt") && i+1 < argc) prompt = argv[++i];
            else if (!strcmp(argv[i], "--tokens") && i+1 < argc) max_tokens = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--temp") && i+1 < argc) temperature = atof(argv[++i]);
            else if (!strcmp(argv[i], "--weights") && i+1 < argc) weights_dir = argv[++i];
            else if (!strcmp(argv[i], "--greedy")) temperature = 0;
        }

        printf("GPT-2 (124M) on Apple Neural Engine\n");
        printf("Prompt: \"%s\"\n", prompt);
        printf("Max tokens: %d, Temperature: %.1f\n\n", max_tokens, temperature);

        // Load tokenizer
        printf("Loading tokenizer...\n");
        GPT2Tok *tok = gpt2_tok_load(weights_dir);
        if (!tok) { printf("Failed to load tokenizer from %s\n", weights_dir); return 1; }

        // Tokenize prompt
        int n_prompt = 0;
        int *prompt_tokens = gpt2_tok_encode(tok, prompt, &n_prompt);
        printf("Prompt tokens (%d): ", n_prompt);
        for (int i = 0; i < n_prompt; i++) printf("%d ", prompt_tokens[i]);
        printf("\n");

        // Load weights
        printf("Loading weights...\n");
        Weights w = {};
        uint64_t t0 = mach_absolute_time();
        if (!load_weights(&w, weights_dir)) { printf("Failed to load weights\n"); return 1; }
        printf("Weights loaded in %.0f ms\n", tb_ms(mach_absolute_time()-t0));

        // wte already fp16 in blob — copy to aligned buffer for NEON LM head
        __fp16 *wte_f16 = (__fp16*)malloc(VOCAB * DIM * sizeof(__fp16));
        memcpy(wte_f16, w.wte, VOCAB * DIM * sizeof(__fp16));

        // Pre-load layer weights as fp16 for NEON decode, fuse QKV
        LayerF16 lf[N_LAYERS];
        KVCache kv[N_LAYERS];
        for (int i = 0; i < N_LAYERS; i++) {
            // Fuse Wq,Wk,Wv into [3*DIM, DIM] contiguous
            __fp16 *wq = blob_to_f16(w.layer[i].wq, DIM*DIM);
            __fp16 *wk = blob_to_f16(w.layer[i].wk, DIM*DIM);
            __fp16 *wv = blob_to_f16(w.layer[i].wv, DIM*DIM);
            lf[i].wqkv = (__fp16*)malloc(3 * DIM * DIM * sizeof(__fp16));
            memcpy(lf[i].wqkv, wq, DIM*DIM*sizeof(__fp16));
            memcpy(lf[i].wqkv + DIM*DIM, wk, DIM*DIM*sizeof(__fp16));
            memcpy(lf[i].wqkv + 2*DIM*DIM, wv, DIM*DIM*sizeof(__fp16));
            free(wq); free(wk); free(wv);
            lf[i].wo = blob_to_f16(w.layer[i].wo, DIM*DIM);
            lf[i].bq = blob_to_f32(w.layer[i].bq, DIM);
            lf[i].bk = blob_to_f32(w.layer[i].bk, DIM);
            lf[i].bv = blob_to_f32(w.layer[i].bv, DIM);
            lf[i].bo = blob_to_f32(w.layer[i].bo, DIM);
            lf[i].w1 = blob_to_f16(w.layer[i].w1, HIDDEN*DIM);
            lf[i].w2 = blob_to_f16(w.layer[i].w2, DIM*HIDDEN);
            lf[i].b1 = blob_to_f32(w.layer[i].b1, HIDDEN);
            lf[i].b2 = blob_to_f32(w.layer[i].b2, DIM);
            kv[i].k = (float*)calloc(DIM * MAX_SEQ, sizeof(float));
            kv[i].v = (float*)calloc(DIM * MAX_SEQ, sizeof(float));
        }

        // Build causal mask for current sequence length
        // We'll start with prompt length and grow as needed
        // For recompute mode: recompile kernels when seq length changes (use buckets)
        #define N_BUCKETS 6
        static const int seq_buckets[N_BUCKETS] = {32, 64, 128, 256, 512, 1024};

        // Build mask blob for each bucket
        NSData *mask_blobs[N_BUCKETS];
        for (int bi = 0; bi < N_BUCKETS; bi++) {
            int bsz = seq_buckets[bi];
            _Float16 *mask = (_Float16*)calloc(bsz*bsz, sizeof(_Float16));
            for (int t = 0; t < bsz; t++)
                for (int t2 = 0; t2 < bsz; t2++)
                    mask[t*bsz+t2] = (t2 <= t) ? (_Float16)0.0f : (_Float16)(-65504.0f);
            int wsize = bsz*bsz*2, total = 128+wsize;
            uint8_t *buf = (uint8_t*)calloc(total,1);
            buf[0]=1; buf[4]=2; buf[64]=0xEF; buf[65]=0xBE; buf[66]=0xAD; buf[67]=0xDE; buf[68]=1;
            *(uint32_t*)(buf+72)=wsize; *(uint32_t*)(buf+80)=128;
            memcpy(buf+128, mask, wsize);
            mask_blobs[bi] = [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
            free(mask);
        }

        // Compiled kernel cache
        Kern kern_attn[N_LAYERS][N_BUCKETS];
        Kern kern_ffn[N_LAYERS][N_BUCKETS];
        memset(kern_attn, 0, sizeof(kern_attn));
        memset(kern_ffn, 0, sizeof(kern_ffn));

        #define FIND_BUCKET(s) ({ int _b=N_BUCKETS-1; for(int _i=0;_i<N_BUCKETS;_i++) if(seq_buckets[_i]>=(s)){_b=_i;break;} _b; })

        // Compile kernels for a bucket
        int cur_bucket = FIND_BUCKET(n_prompt);
        for (int compile_bi = cur_bucket; compile_bi <= cur_bucket; compile_bi++) {
            int bsz = seq_buckets[compile_bi];
            printf("Compiling kernels for seq=%d...\n", bsz);
            uint64_t ct0 = mach_absolute_time();
            for (int l = 0; l < N_LAYERS; l++) {
                NSDictionary *attn_wd = @{
                    [NSString stringWithFormat:@"@model_path/weights/layer_%02d/wq.bin",l]: @{@"offset":@0, @"data":w.layer[l].wq},
                    [NSString stringWithFormat:@"@model_path/weights/layer_%02d/wk.bin",l]: @{@"offset":@0, @"data":w.layer[l].wk},
                    [NSString stringWithFormat:@"@model_path/weights/layer_%02d/wv.bin",l]: @{@"offset":@0, @"data":w.layer[l].wv},
                    [NSString stringWithFormat:@"@model_path/weights/layer_%02d/wo.bin",l]: @{@"offset":@0, @"data":w.layer[l].wo},
                    [NSString stringWithFormat:@"@model_path/weights/layer_%02d/bq.bin",l]: @{@"offset":@0, @"data":w.layer[l].bq},
                    [NSString stringWithFormat:@"@model_path/weights/layer_%02d/bk.bin",l]: @{@"offset":@0, @"data":w.layer[l].bk},
                    [NSString stringWithFormat:@"@model_path/weights/layer_%02d/bv.bin",l]: @{@"offset":@0, @"data":w.layer[l].bv},
                    [NSString stringWithFormat:@"@model_path/weights/layer_%02d/bo.bin",l]: @{@"offset":@0, @"data":w.layer[l].bo},
                    @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":mask_blobs[compile_bi]},
                };
                kern_attn[l][compile_bi] = compile_mil(gen_attn_mil(l, bsz), attn_wd);
                if (!kern_attn[l][compile_bi].model) { printf("  FAIL: attn layer %d seq %d\n", l, bsz); return 1; }

                NSDictionary *ffn_wd = @{
                    [NSString stringWithFormat:@"@model_path/weights/layer_%02d/w1.bin",l]: @{@"offset":@0, @"data":w.layer[l].w1},
                    [NSString stringWithFormat:@"@model_path/weights/layer_%02d/w2.bin",l]: @{@"offset":@0, @"data":w.layer[l].w2},
                    [NSString stringWithFormat:@"@model_path/weights/layer_%02d/b1.bin",l]: @{@"offset":@0, @"data":w.layer[l].b1},
                    [NSString stringWithFormat:@"@model_path/weights/layer_%02d/b2.bin",l]: @{@"offset":@0, @"data":w.layer[l].b2},
                };
                kern_ffn[l][compile_bi] = compile_mil(gen_ffn_mil(l, bsz), ffn_wd);
                if (!kern_ffn[l][compile_bi].model) { printf("  FAIL: ffn layer %d seq %d\n", l, bsz); return 1; }
            }
            printf("  Compiled %d kernels in %.0f ms\n", N_LAYERS*2, tb_ms(mach_absolute_time()-ct0));
        }

        // Allocate working buffers
        int total_seq = n_prompt + max_tokens;
        if (total_seq > MAX_SEQ) total_seq = MAX_SEQ;
        int *tokens = (int*)malloc(total_seq * sizeof(int));
        memcpy(tokens, prompt_tokens, n_prompt * sizeof(int));
        free(prompt_tokens);

        float *hidden = (float*)malloc(MAX_SEQ * DIM * sizeof(float));
        float *ln_out = (float*)malloc(MAX_SEQ * DIM * sizeof(float));

        // Print prompt
        {
            char *prompt_str = gpt2_tok_decode(tok, tokens, n_prompt);
            printf("\n--- Generation ---\n%s", prompt_str);
            free(prompt_str);
        }

        // === PREFILL: ANE processes full prompt ===
        int seq_len = n_prompt;
        {
            uint64_t step_t0 = mach_absolute_time();
            int bi = FIND_BUCKET(seq_len);
            int bsz = seq_buckets[bi];

            embed_tokens(hidden, tokens, w.wte, w.wpe, seq_len);

            size_t io_sz = DIM * bsz * 2;
            IOSurfaceRef surf_in = make_surface(io_sz);
            IOSurfaceRef surf_out = make_surface(io_sz);
            IOSurfaceRef ins[] = {surf_in}, outs[] = {surf_out};

            for (int l = 0; l < N_LAYERS; l++) {
                layer_norm(ln_out, hidden, w.layer[l].ln1_w, w.layer[l].ln1_b, DIM, seq_len);

                // Fill KV cache for this layer from ln_out (all prompt tokens)
                // Wk is in wqkv[DIM*DIM .. 2*DIM*DIM), Wv is wqkv[2*DIM*DIM ..)
                {
                    float *wk32 = (float*)malloc(DIM*DIM*sizeof(float));
                    float *wv32 = (float*)malloc(DIM*DIM*sizeof(float));
                    const __fp16 *wk16 = lf[l].wqkv + DIM*DIM;
                    const __fp16 *wv16 = lf[l].wqkv + 2*DIM*DIM;
                    for (int j = 0; j < DIM*DIM; j++) { wk32[j] = (float)wk16[j]; wv32[j] = (float)wv16[j]; }
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        DIM, seq_len, DIM, 1.0f,
                        wk32, DIM, ln_out, DIM, 0.0f, kv[l].k, MAX_SEQ);
                    for (int d = 0; d < DIM; d++)
                        for (int t = 0; t < seq_len; t++)
                            kv[l].k[d * MAX_SEQ + t] += lf[l].bk[d];
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        DIM, seq_len, DIM, 1.0f,
                        wv32, DIM, ln_out, DIM, 0.0f, kv[l].v, MAX_SEQ);
                    for (int d = 0; d < DIM; d++)
                        for (int t = 0; t < seq_len; t++)
                            kv[l].v[d * MAX_SEQ + t] += lf[l].bv[d];
                    free(wk32); free(wv32);
                }

                // ANE attention (full sequence)
                IOSurfaceLock(surf_in, 0, NULL);
                _Float16 *pin = (_Float16*)IOSurfaceGetBaseAddress(surf_in);
                memset(pin, 0, io_sz);
                for (int t = 0; t < seq_len; t++)
                    for (int c = 0; c < DIM; c++)
                        pin[c*bsz+t] = (_Float16)ln_out[t*DIM+c];
                IOSurfaceUnlock(surf_in, 0, NULL);
                ane_eval(&kern_attn[l][bi], ins, 1, outs, 1);
                IOSurfaceLock(surf_out, kIOSurfaceLockReadOnly, NULL);
                _Float16 *pout = (_Float16*)IOSurfaceGetBaseAddress(surf_out);
                for (int t = 0; t < seq_len; t++)
                    for (int c = 0; c < DIM; c++)
                        hidden[t*DIM+c] += (float)pout[c*bsz+t];
                IOSurfaceUnlock(surf_out, kIOSurfaceLockReadOnly, NULL);

                layer_norm(ln_out, hidden, w.layer[l].ln2_w, w.layer[l].ln2_b, DIM, seq_len);

                // ANE FFN (full sequence)
                IOSurfaceLock(surf_in, 0, NULL);
                pin = (_Float16*)IOSurfaceGetBaseAddress(surf_in);
                memset(pin, 0, io_sz);
                for (int t = 0; t < seq_len; t++)
                    for (int c = 0; c < DIM; c++)
                        pin[c*bsz+t] = (_Float16)ln_out[t*DIM+c];
                IOSurfaceUnlock(surf_in, 0, NULL);
                ane_eval(&kern_ffn[l][bi], ins, 1, outs, 1);
                IOSurfaceLock(surf_out, kIOSurfaceLockReadOnly, NULL);
                pout = (_Float16*)IOSurfaceGetBaseAddress(surf_out);
                for (int t = 0; t < seq_len; t++)
                    for (int c = 0; c < DIM; c++)
                        hidden[t*DIM+c] += (float)pout[c*bsz+t];
                IOSurfaceUnlock(surf_out, kIOSurfaceLockReadOnly, NULL);
            }

            CFRelease(surf_in);
            CFRelease(surf_out);

            layer_norm(ln_out, hidden, w.ln_f_w, w.ln_f_b, DIM, seq_len);
            int next_token = lm_head_sample(ln_out + (seq_len-1)*DIM, wte_f16, temperature);

            char *tok_str = gpt2_tok_decode(tok, &next_token, 1);
            printf("%s", tok_str);
            free(tok_str);

            if (next_token == 50256) { printf("\n"); goto done; }
            tokens[seq_len] = next_token;
            seq_len++;

            fprintf(stderr, "\r[prefill: %d tok, %.1f ms, ANE]",
                n_prompt, tb_ms(mach_absolute_time() - step_t0));
        }

        // === DECODE: CPU with KV cache ===
        float x_dec[DIM], ln_buf[DIM], attn_buf[DIM], ffn_buf[DIM];
        for (int step = 1; step < max_tokens && seq_len < MAX_SEQ; step++) {
            uint64_t step_t0 = mach_absolute_time();
            int pos = seq_len - 1;  // position of latest token

            // Embed single token
            for (int d = 0; d < DIM; d++)
                x_dec[d] = (float)w.wte[tokens[pos]*DIM+d] + (float)w.wpe[pos*DIM+d];

            for (int l = 0; l < N_LAYERS; l++) {
                // LN1 (single token)
                layer_norm(ln_buf, x_dec, w.layer[l].ln1_w, w.layer[l].ln1_b, DIM, 1);
                // Attention with KV cache
                cpu_attn_decode(attn_buf, ln_buf, pos, &lf[l], &kv[l]);
                for (int d = 0; d < DIM; d++) x_dec[d] += attn_buf[d];
                // LN2
                layer_norm(ln_buf, x_dec, w.layer[l].ln2_w, w.layer[l].ln2_b, DIM, 1);
                // FFN
                cpu_ffn_decode(ffn_buf, ln_buf, &lf[l]);
                for (int d = 0; d < DIM; d++) x_dec[d] += ffn_buf[d];
            }

            // Final LN + LM head
            layer_norm(ln_buf, x_dec, w.ln_f_w, w.ln_f_b, DIM, 1);
            int next_token = lm_head_sample(ln_buf, wte_f16, temperature);

            double step_ms = tb_ms(mach_absolute_time() - step_t0);

            char *tok_str = gpt2_tok_decode(tok, &next_token, 1);
            printf("%s", tok_str);
            free(tok_str);

            if (next_token == 50256) { printf("\n"); break; }
            tokens[seq_len] = next_token;
            seq_len++;

            if ((step+1) % 10 == 0)
                fprintf(stderr, "\r[step %d/%d, seq=%d, %.1f ms/tok, decode/CPU]",
                    step+1, max_tokens, seq_len, step_ms);
        }
        done:

        printf("\n\n--- Done ---\n");
        printf("Generated %d tokens (total seq=%d)\n", seq_len - n_prompt, seq_len);

        // Cleanup
        for (int bi = 0; bi < N_BUCKETS; bi++)
            for (int l = 0; l < N_LAYERS; l++) {
                cleanup_kern(&kern_attn[l][bi]);
                cleanup_kern(&kern_ffn[l][bi]);
            }
        gpt2_tok_free(tok);
        free(tokens); free(hidden); free(ln_out);
    }
    return 0;
}
