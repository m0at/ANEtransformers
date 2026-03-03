// weight_similarity.c — Analyze weight similarity across Mistral 7B layers
// Tests whether ANE compiled programs can be shared via weight deduplication
//
// Build: clang -O2 -o weight_similarity weight_similarity.c -lm
// Run:   ./weight_similarity ~/models/mistral-7b-instruct-v0.2.Q4_0.gguf

#include "gguf_loader.h"
#include "dequant.h"
#include <math.h>
#include <Accelerate/Accelerate.h>

#define N_LAYERS 32
#define N_WEIGHT_TYPES 7

static const char *weight_names[N_WEIGHT_TYPES] = {
    "Wq", "Wk", "Wv", "Wo", "W1(gate)", "W3(up)", "W2(down)"
};

static const char *weight_fmts[N_WEIGHT_TYPES] = {
    "blk.%d.attn_q.weight",
    "blk.%d.attn_k.weight",
    "blk.%d.attn_v.weight",
    "blk.%d.attn_output.weight",
    "blk.%d.ffn_gate.weight",
    "blk.%d.ffn_up.weight",
    "blk.%d.ffn_down.weight",
};

// Dequant a Q4_0 tensor to fp32
static float *dequant_tensor_f32(GGUFFile *f, const char *name, uint64_t *out_nel) {
    GGUFTensor *t = gguf_find(f, name);
    if (!t) { fprintf(stderr, "Tensor not found: %s\n", name); return NULL; }

    uint64_t nel = gguf_nelements(t);
    *out_nel = nel;

    const void *raw = gguf_data(f, t);
    float *out = (float *)malloc(nel * sizeof(float));

    if (t->type == GGML_TYPE_Q4_0) {
        uint64_t n_blocks = nel / QK4_0;
        const block_q4_0 *blocks = (const block_q4_0 *)raw;
        _Float16 tmp[QK4_0];
        for (uint64_t b = 0; b < n_blocks; b++) {
            dequant_q4_0_block_neon(&blocks[b], tmp);
            for (int j = 0; j < QK4_0; j++)
                out[b * QK4_0 + j] = (float)tmp[j];
        }
    } else {
        fprintf(stderr, "Unsupported type %d for %s\n", t->type, name);
        free(out);
        return NULL;
    }
    return out;
}

// Cosine similarity between two float vectors
static double cosine_sim(const float *a, const float *b, uint64_t n) {
    double dot = 0, na = 0, nb = 0;
    // Use vDSP for speed
    float fdot, fna, fnb;
    vDSP_dotpr(a, 1, b, 1, &fdot, (vDSP_Length)n);
    vDSP_dotpr(a, 1, a, 1, &fna, (vDSP_Length)n);
    vDSP_dotpr(b, 1, b, 1, &fnb, (vDSP_Length)n);
    dot = fdot; na = fna; nb = fnb;
    if (na < 1e-30 || nb < 1e-30) return 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

// L2 distance
static double l2_dist(const float *a, const float *b, uint64_t n) {
    double sum = 0;
    for (uint64_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return sqrt(sum);
}

// Frobenius norm
static double frob_norm(const float *a, uint64_t n) {
    float fn;
    vDSP_dotpr(a, 1, a, 1, &fn, (vDSP_Length)n);
    return sqrt((double)fn);
}

// Count unique Q4_0 blobs after rounding scales to given precision
static int count_unique_blobs_rounded(GGUFFile *f, const char *fmt, int n_layers, int precision_bits) {
    // Get raw Q4_0 data for each layer, round scales, hash
    uint64_t hashes[N_LAYERS];
    size_t blob_sizes[N_LAYERS];

    for (int l = 0; l < n_layers; l++) {
        char name[128];
        snprintf(name, sizeof(name), fmt, l);
        GGUFTensor *t = gguf_find(f, name);
        if (!t) { hashes[l] = l; blob_sizes[l] = 0; continue; }

        size_t bytes = gguf_tensor_bytes(t);
        blob_sizes[l] = bytes;
        uint64_t nel = gguf_nelements(t);
        uint64_t n_blocks = nel / QK4_0;

        // Copy and round the scale values
        const block_q4_0 *src = (const block_q4_0 *)gguf_data(f, t);

        // FNV-1a hash of rounded data
        uint64_t h = 14695981039346656037ULL;
        float round_factor = (float)(1 << precision_bits);

        for (uint64_t b = 0; b < n_blocks; b++) {
            // Round scale to precision_bits
            float scale = (float)src[b].d;
            float rounded = roundf(scale * round_factor) / round_factor;
            uint32_t rbits;
            memcpy(&rbits, &rounded, 4);

            h ^= rbits; h *= 1099511628211ULL;
            // Hash the quantized nibbles unchanged
            for (int j = 0; j < QK4_0/2; j++) {
                h ^= src[b].qs[j]; h *= 1099511628211ULL;
            }
        }
        hashes[l] = h;
    }

    // Count unique
    int unique = 0;
    for (int i = 0; i < n_layers; i++) {
        int dup = 0;
        for (int j = 0; j < i; j++) {
            if (hashes[i] == hashes[j]) { dup = 1; break; }
        }
        if (!dup) unique++;
    }
    return unique;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    GGUFFile *f = gguf_open(argv[1]);
    if (!f) return 1;
    gguf_print_info(f);

    int n_layers = f->cfg.n_layers;
    if (n_layers > N_LAYERS) n_layers = N_LAYERS;

    printf("\n=== WEIGHT SIMILARITY ANALYSIS ===\n");
    printf("Layers: %d, Weight types: %d\n\n", n_layers, N_WEIGHT_TYPES);

    for (int wt = 0; wt < N_WEIGHT_TYPES; wt++) {
        printf("--- %s ---\n", weight_names[wt]);

        // Load all layers' weights for this type
        float *weights[N_LAYERS] = {0};
        uint64_t nel = 0;

        for (int l = 0; l < n_layers; l++) {
            char name[128];
            snprintf(name, sizeof(name), weight_fmts[wt], l);
            weights[l] = dequant_tensor_f32(f, name, &nel);
            if (!weights[l]) { fprintf(stderr, "Failed to load %s\n", name); goto next_wt; }
        }

        printf("Shape: %llu x %llu (%llu elements, %.1f MB fp32)\n",
               (unsigned long long)(wt < 4 ? (wt == 1 || wt == 2 ? 1024 : 4096) :
                (wt == 6 ? 4096 : 14336)),
               (unsigned long long)(wt == 6 ? 14336 : 4096),
               (unsigned long long)nel, nel * 4.0 / 1048576.0);

        // 1. Pairwise cosine similarity
        double min_cos = 1.0, max_cos = -1.0, avg_cos = 0;
        int best_i = 0, best_j = 1, worst_i = 0, worst_j = 1;
        int n_pairs = 0;

        for (int i = 0; i < n_layers; i++) {
            for (int j = i + 1; j < n_layers; j++) {
                double cs = cosine_sim(weights[i], weights[j], nel);
                avg_cos += cs;
                n_pairs++;
                if (cs > max_cos) { max_cos = cs; best_i = i; best_j = j; }
                if (cs < min_cos) { min_cos = cs; worst_i = i; worst_j = j; }
            }
        }
        avg_cos /= n_pairs;

        printf("Cosine similarity: min=%.6f (L%d-L%d) max=%.6f (L%d-L%d) avg=%.6f\n",
               min_cos, worst_i, worst_j, max_cos, best_i, best_j, avg_cos);

        // 2. Pairwise L2 distance (normalized by Frobenius norm)
        double min_l2 = 1e30, max_l2 = 0, avg_l2 = 0;
        double norms[N_LAYERS];
        for (int i = 0; i < n_layers; i++)
            norms[i] = frob_norm(weights[i], nel);

        for (int i = 0; i < n_layers; i++) {
            for (int j = i + 1; j < n_layers; j++) {
                double d = l2_dist(weights[i], weights[j], nel);
                double rel = d / ((norms[i] + norms[j]) / 2.0);
                avg_l2 += rel;
                if (rel < min_l2) min_l2 = rel;
                if (rel > max_l2) max_l2 = rel;
            }
        }
        avg_l2 /= n_pairs;

        printf("Relative L2 dist: min=%.6f max=%.6f avg=%.6f\n", min_l2, max_l2, avg_l2);

        // 3. Top-5 most similar pairs
        printf("Top-5 most similar pairs:\n");
        // Simple selection sort for top 5
        double top_cos[5] = {-2,-2,-2,-2,-2};
        int top_i[5] = {0}, top_j[5] = {0};

        for (int i = 0; i < n_layers; i++) {
            for (int j = i + 1; j < n_layers; j++) {
                double cs = cosine_sim(weights[i], weights[j], nel);
                // Insert into top 5
                for (int k = 0; k < 5; k++) {
                    if (cs > top_cos[k]) {
                        // Shift down
                        for (int m = 4; m > k; m--) {
                            top_cos[m] = top_cos[m-1];
                            top_i[m] = top_i[m-1];
                            top_j[m] = top_j[m-1];
                        }
                        top_cos[k] = cs;
                        top_i[k] = i;
                        top_j[k] = j;
                        break;
                    }
                }
            }
        }
        for (int k = 0; k < 5; k++)
            printf("  L%02d - L%02d: cos=%.6f\n", top_i[k], top_j[k], top_cos[k]);

        // 4. Quantized blob uniqueness (rounding test)
        printf("Unique Q4_0 blobs after rounding scales:\n");
        for (int bits = 16; bits >= 4; bits -= 2) {
            int unique = count_unique_blobs_rounded(f, weight_fmts[wt], n_layers, bits);
            printf("  %2d-bit precision: %d/%d unique\n", bits, unique, n_layers);
        }

        // 5. K-means quality impact estimate
        // Compute: if we replace each layer's weights with the centroid of its cluster,
        // what's the relative error?
        // Simple test: for the most similar pair, compute substitution error
        {
            double cs = top_cos[0];
            int bi = top_i[0], bj = top_j[0];
            // Error of using layer bi's weights for layer bj
            double sub_err = l2_dist(weights[bi], weights[bj], nel) / norms[bj];
            printf("Best-pair substitution error (L%d->L%d): %.4f%% (cos=%.6f)\n",
                   bi, bj, sub_err * 100.0, cs);
        }

        printf("\n");

        next_wt:
        for (int l = 0; l < n_layers; l++) free(weights[l]);
    }

    // === K-MEANS CLUSTERING ANALYSIS ===
    printf("\n=== K-MEANS CLUSTERING (Wq only, K=4,8,16) ===\n");
    {
        // Load all Wq weights
        float *wq[N_LAYERS] = {0};
        uint64_t nel = 0;
        for (int l = 0; l < n_layers; l++) {
            char name[128];
            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
            wq[l] = dequant_tensor_f32(f, name, &nel);
        }

        for (int K = 4; K <= 16; K *= 2) {
            printf("\n--- K=%d clusters ---\n", K);

            // Simple K-means: init with evenly spaced layers
            int assignments[N_LAYERS];
            float *centroids[16]; // max K=16

            for (int k = 0; k < K; k++) {
                centroids[k] = (float *)malloc(nel * sizeof(float));
                int init_layer = k * n_layers / K;
                memcpy(centroids[k], wq[init_layer], nel * sizeof(float));
            }

            // 20 iterations of K-means
            for (int iter = 0; iter < 20; iter++) {
                // Assign each layer to nearest centroid
                for (int l = 0; l < n_layers; l++) {
                    double best_dist = 1e30;
                    int best_k = 0;
                    for (int k = 0; k < K; k++) {
                        double d = l2_dist(wq[l], centroids[k], nel);
                        if (d < best_dist) { best_dist = d; best_k = k; }
                    }
                    assignments[l] = best_k;
                }

                // Update centroids
                for (int k = 0; k < K; k++) {
                    memset(centroids[k], 0, nel * sizeof(float));
                    int count = 0;
                    for (int l = 0; l < n_layers; l++) {
                        if (assignments[l] == k) {
                            for (uint64_t i = 0; i < nel; i++)
                                centroids[k][i] += wq[l][i];
                            count++;
                        }
                    }
                    if (count > 0) {
                        float inv = 1.0f / count;
                        for (uint64_t i = 0; i < nel; i++)
                            centroids[k][i] *= inv;
                    }
                }
            }

            // Print clusters and per-layer error
            printf("Cluster assignments:\n");
            for (int k = 0; k < K; k++) {
                printf("  C%d: ", k);
                int count = 0;
                for (int l = 0; l < n_layers; l++) {
                    if (assignments[l] == k) { printf("L%d ", l); count++; }
                }
                printf("(%d layers)\n", count);
            }

            // Per-layer substitution error
            double max_err = 0, avg_err = 0;
            for (int l = 0; l < n_layers; l++) {
                int k = assignments[l];
                double norm = frob_norm(wq[l], nel);
                double err = l2_dist(wq[l], centroids[k], nel) / norm;
                avg_err += err;
                if (err > max_err) max_err = err;
            }
            avg_err /= n_layers;
            printf("Substitution error: avg=%.4f%% max=%.4f%%\n", avg_err*100, max_err*100);
            printf("Programs needed: %d (vs %d without sharing)\n", K * 5, n_layers * 5);

            for (int k = 0; k < K; k++) free(centroids[k]);
        }

        for (int l = 0; l < n_layers; l++) free(wq[l]);
    }

    // === RAW Q4_0 BLOB COMPARISON ===
    printf("\n=== RAW Q4_0 BLOB COMPARISON (byte-level) ===\n");
    for (int wt = 0; wt < N_WEIGHT_TYPES; wt++) {
        printf("%s: ", weight_names[wt]);

        // Compare raw Q4_0 bytes between layers
        size_t sizes[N_LAYERS];
        const void *ptrs[N_LAYERS];
        int valid = 1;

        for (int l = 0; l < n_layers; l++) {
            char name[128];
            snprintf(name, sizeof(name), weight_fmts[wt], l);
            GGUFTensor *t = gguf_find(f, name);
            if (!t) { valid = 0; break; }
            ptrs[l] = gguf_data(f, t);
            sizes[l] = gguf_tensor_bytes(t);
        }
        if (!valid) { printf("SKIP\n"); continue; }

        // Count identical pairs
        int identical = 0, total = 0;
        double min_byte_diff_pct = 100.0;
        int min_i = 0, min_j = 1;

        for (int i = 0; i < n_layers; i++) {
            for (int j = i + 1; j < n_layers; j++) {
                total++;
                if (memcmp(ptrs[i], ptrs[j], sizes[i]) == 0) {
                    identical++;
                } else {
                    // Count differing bytes
                    const uint8_t *a = (const uint8_t *)ptrs[i];
                    const uint8_t *b = (const uint8_t *)ptrs[j];
                    uint64_t diff = 0;
                    for (size_t k = 0; k < sizes[i]; k++)
                        diff += (a[k] != b[k]);
                    double pct = 100.0 * diff / sizes[i];
                    if (pct < min_byte_diff_pct) {
                        min_byte_diff_pct = pct;
                        min_i = i; min_j = j;
                    }
                }
            }
        }
        printf("%d/%d identical pairs, min byte diff=%.2f%% (L%d-L%d)\n",
               identical, total, min_byte_diff_pct, min_i, min_j);
    }

    gguf_close(f);
    return 0;
}
