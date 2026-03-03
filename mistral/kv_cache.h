// kv_cache.h — KV cache for Mistral 7B inference
// Sequence-major layout: k_cache[layer][seq][n_kv_heads][head_dim]
// Address: layer * max_seq * kv_dim + t * kv_dim + kvh * head_dim + d
// Writes are contiguous 2KB memcpies; reads stride by kv_dim (1KB) across seq.
#pragma once
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>

typedef struct {
    int n_layers;
    int n_kv_heads;    // 8
    int head_dim;      // 128
    int max_seq;       // configurable (4096-65536)
    int kv_dim;        // n_kv_heads * head_dim = 1024
    _Float16 *k_cache; // [n_layers][max_seq][n_kv_heads][head_dim]
    _Float16 *v_cache; // [n_layers][max_seq][n_kv_heads][head_dim]
    int pos;           // current write position (for ring buffer)
    int len;           // valid entries (0..max_seq)
} KVCache;

static KVCache kv_alloc(int n_layers, int n_kv_heads, int head_dim, int max_seq) {
    KVCache c;
    c.n_layers = n_layers;
    c.n_kv_heads = n_kv_heads;
    c.head_dim = head_dim;
    c.max_seq = max_seq;
    c.kv_dim = n_kv_heads * head_dim;
    size_t sz = (size_t)n_layers * max_seq * c.kv_dim * sizeof(_Float16);
    c.k_cache = (_Float16 *)calloc(sz, 1);
    c.v_cache = (_Float16 *)calloc(sz, 1);
    c.pos = 0;
    c.len = 0;
    return c;
}

// Pointer to K[layer, seq_pos, :, :] — kv_dim contiguous fp16 values
static inline _Float16 *kv_k_at(const KVCache *cache, int layer, int seq_pos) {
    int t = seq_pos % cache->max_seq;
    return cache->k_cache + (size_t)layer * cache->max_seq * cache->kv_dim
                          + (size_t)t * cache->kv_dim;
}

// Pointer to V[layer, seq_pos, :, :] — kv_dim contiguous fp16 values
static inline _Float16 *kv_v_at(const KVCache *cache, int layer, int seq_pos) {
    int t = seq_pos % cache->max_seq;
    return cache->v_cache + (size_t)layer * cache->max_seq * cache->kv_dim
                          + (size_t)t * cache->kv_dim;
}

// Write k_vec and v_vec ([kv_dim] fp16 each) at position pos for given layer.
// Contiguous 2KB memcpy per call.
static void kv_write(KVCache *cache, int layer, int pos, const _Float16 *k_vec, const _Float16 *v_vec) {
    size_t sz = (size_t)cache->kv_dim * sizeof(_Float16);
    memcpy(kv_k_at(cache, layer, pos), k_vec, sz);
    memcpy(kv_v_at(cache, layer, pos), v_vec, sz);
}

// Pointer to K layer base: [max_seq][n_kv_heads][head_dim]
// Caller indexes as base[t * kv_dim + kvh * head_dim + d]
static inline _Float16 *kv_k(const KVCache *cache, int layer) {
    return cache->k_cache + (size_t)layer * cache->max_seq * cache->kv_dim;
}

// Pointer to V layer base: [max_seq][n_kv_heads][head_dim]
static inline _Float16 *kv_v(const KVCache *cache, int layer) {
    return cache->v_cache + (size_t)layer * cache->max_seq * cache->kv_dim;
}

static void kv_free(KVCache *cache) {
    free(cache->k_cache);
    free(cache->v_cache);
    cache->k_cache = NULL;
    cache->v_cache = NULL;
}
