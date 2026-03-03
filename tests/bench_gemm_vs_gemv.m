// bench_gemm_vs_gemv.m — Profile Metal GEMM vs CPU SDOT at various S values
// Measures raw kernel time for the 4 key matrix shapes in Mistral 7B decode.
// Usage: ./bench_gemm_vs_gemv --model path/to/mistral-7b-q4_0.gguf

#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <Accelerate/Accelerate.h>

#include "gguf_loader.h"
#include "dequant.h"
#include "kv_cache.h"
#include "tokenizer.h"
#include "mistral_model.h"
#include "metal_matvec.h"

static double time_ms(void) {
    static mach_timebase_info_data_t tbi = {0};
    if (tbi.denom == 0) mach_timebase_info(&tbi);
    return (double)mach_absolute_time() * tbi.numer / tbi.denom / 1e6;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *model_path = NULL;
        static struct option long_opts[] = {
            {"model", required_argument, 0, 'm'},
            {0, 0, 0, 0}
        };
        int opt;
        while ((opt = getopt_long(argc, argv, "m:", long_opts, NULL)) != -1) {
            if (opt == 'm') model_path = optarg;
        }
        if (!model_path) {
            fprintf(stderr, "Usage: %s --model <gguf>\n", argv[0]);
            return 1;
        }

        // Load model
        MistralModel *model = mistral_load(model_path);
        if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
        MistralConfig *cfg = &model->cfg;
        int dim = cfg->dim;           // 4096
        int hidden = cfg->hidden_dim; // 14336
        int kv_dim = cfg->n_kv_heads * cfg->head_dim;  // 1024

        fprintf(stderr, "Model: dim=%d hidden=%d kv_dim=%d\n", dim, hidden, kv_dim);

        // Init Metal
        MetalContext *metal_ctx = metal_context_init(
            model->gguf->mmap_base, model->gguf->mmap_len,
            dim, kv_dim, hidden, cfg->vocab_size,
            cfg->n_heads, cfg->n_layers, 4096);
        if (!metal_ctx) { fprintf(stderr, "Metal init failed\n"); return 1; }

        // Weight pointers for layer 0
        LayerWeights *lw = &model->layers[0];

        // Test shapes: (out_dim, in_dim, name)
        struct { int out; int in; const char *name; const void *W; } shapes[] = {
            {dim,    dim,    "QKV (4096x4096)",   lw->wq},
            {dim,    dim,    "Wo  (4096x4096)",   lw->wo},
            {hidden, dim,    "W1  (14336x4096)",  lw->w1},
            {dim,    hidden, "W2  (4096x14336)",  lw->w2},
        };
        int n_shapes = 4;

        // S values to test
        int S_vals[] = {1, 2, 3, 4, 5, 8, 16};
        int n_S = 7;
        int max_S = 16;

        // Allocate batch buffers for max S
        metal_alloc_batch_buffers(metal_ctx, max_S, dim, kv_dim, hidden, cfg->n_heads);

        // CPU input/output buffers
        int max_dim = (dim > hidden) ? dim : hidden;
        float *x_cpu = (float *)calloc((size_t)max_S * max_dim, sizeof(float));
        float *y_cpu = (float *)calloc((size_t)max_S * max_dim, sizeof(float));
        // Q8 quantized input for SDOT path
        int max_q8_blocks = max_dim / QK8_0;
        block_q8_0 *x_q8 = (block_q8_0 *)calloc((size_t)max_S * max_q8_blocks, sizeof(block_q8_0));

        // Fill with random data
        for (int i = 0; i < max_S * max_dim; i++)
            x_cpu[i] = ((float)arc4random() / (float)UINT32_MAX - 0.5f) * 0.1f;

        int warmup = 3;
        int reps = 10;

        fprintf(stderr, "\n%s\n", "═══════════════════════════════════════════════════════════════════════════════");
        fprintf(stderr, "  GEMM vs GEMV Benchmark — Metal GPU vs CPU SDOT (M5, %d warmup, %d reps)\n", warmup, reps);
        fprintf(stderr, "%s\n\n", "═══════════════════════════════════════════════════════════════════════════════");

        // Header
        printf("%-20s  %3s  %10s  %10s  %8s  %10s\n",
               "Shape", "S", "GPU (ms)", "CPU (ms)", "Ratio", "GPU GB/s");
        printf("%-20s  %3s  %10s  %10s  %8s  %10s\n",
               "--------------------", "---", "----------", "----------", "--------", "----------");

        for (int si = 0; si < n_shapes; si++) {
            int out_dim = shapes[si].out;
            int in_dim  = shapes[si].in;
            const void *W = shapes[si].W;
            size_t weight_bytes = (size_t)out_dim * (in_dim / 32) * 18;  // Q4_0: 18 bytes per block

            for (int sv = 0; sv < n_S; sv++) {
                int S = S_vals[sv];

                // ── GPU GEMM/GEMV ──
                // Fill GPU input buffer
                id<MTLBuffer> in_buf = (S == 1) ? metal_ctx->xb_buf : metal_ctx->xb_batch_buf;
                id<MTLBuffer> out_buf = (S == 1) ? metal_ctx->q_buf : metal_ctx->q_batch_buf;

                memcpy([in_buf contents], x_cpu, (size_t)S * in_dim * sizeof(float));

                // Warmup
                for (int w = 0; w < warmup; w++) {
                    @autoreleasepool {
                        id<MTLCommandBuffer> cb = [metal_ctx->queue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        if (S == 1)
                            metal_encode_gemv(metal_ctx, enc, W, in_buf, out_buf, out_dim, in_dim);
                        else
                            metal_encode_gemm(metal_ctx, enc, W, in_buf, out_buf, out_dim, in_dim, S);
                        [enc endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                    }
                }

                // Timed runs
                double gpu_total = 0;
                for (int r = 0; r < reps; r++) {
                    @autoreleasepool {
                        id<MTLCommandBuffer> cb = [metal_ctx->queue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        double t0 = time_ms();
                        if (S == 1)
                            metal_encode_gemv(metal_ctx, enc, W, in_buf, out_buf, out_dim, in_dim);
                        else
                            metal_encode_gemm(metal_ctx, enc, W, in_buf, out_buf, out_dim, in_dim, S);
                        [enc endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                        gpu_total += time_ms() - t0;
                    }
                }
                double gpu_ms = gpu_total / reps;

                // ── CPU SDOT ──
                // Quantize inputs to Q8 for each token
                for (int s = 0; s < S; s++)
                    quantize_f32_to_q8_0(x_cpu + (size_t)s * in_dim,
                                          x_q8 + (size_t)s * (in_dim / QK8_0),
                                          in_dim);

                for (int w = 0; w < warmup; w++) {
                    for (int s = 0; s < S; s++)
                        q4_matvec_sdot_parallel(W, GGML_TYPE_Q4_0,
                                                x_q8 + (size_t)s * (in_dim / QK8_0),
                                                y_cpu + (size_t)s * out_dim,
                                                out_dim, in_dim);
                }

                double cpu_total = 0;
                for (int r = 0; r < reps; r++) {
                    double t0 = time_ms();
                    for (int s = 0; s < S; s++)
                        q4_matvec_sdot_parallel(W, GGML_TYPE_Q4_0,
                                                x_q8 + (size_t)s * (in_dim / QK8_0),
                                                y_cpu + (size_t)s * out_dim,
                                                out_dim, in_dim);
                    cpu_total += time_ms() - t0;
                }
                double cpu_ms = cpu_total / reps;

                // Compute bandwidth (GPU reads weights + input, writes output)
                double gpu_bytes = (double)weight_bytes + (double)S * in_dim * 4 + (double)S * out_dim * 4;
                double gpu_bw = gpu_bytes / (gpu_ms / 1000.0) / 1e9;

                printf("%-20s  %3d  %10.2f  %10.2f  %8.2fx  %8.1f\n",
                       shapes[si].name, S, gpu_ms, cpu_ms, cpu_ms / gpu_ms, gpu_bw);
            }
            printf("\n");
        }

        // Cleanup
        free(x_cpu);
        free(y_cpu);
        metal_context_free(metal_ctx);
        mistral_free(model);
    }
    return 0;
}
