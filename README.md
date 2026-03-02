# Running Transformers on Apple's Neural Engine

You might be asking, "why the FUCK would you pick GPT2?"

Have you read the art bro? Have you? Nah. I doubt it.

GPT2 had more soul in its theoretical pinky finger than all of us combined.

But I digress..

Running transformer inference and training directly on Apple's Neural Engine via reverse-engineered private APIs. No CoreML, no Metal, no GPU — pure ANE compute through `_ANEInMemoryModel` and MIL programs compiled at runtime.

Forked from [maderix/ANEtransformers](https://github.com/maderix/ANEtransformers) which demonstrated ANE training (Stories110M, 12-layer forward+backward on ANE). This fork extends the project with GPT-2 inference on ANE, systematic M5 hardware investigation, and fused kernel optimization.

## What's Here

### GPT-2 Inference on ANE (`training/gpt2.m`)

Complete GPT-2 (124M) inference engine. Two-phase architecture:

1. **ANE prefill** — Full sequence processed on ANE using fused attention and FFN kernels. One fused attention kernel (QKV + multi-head SDPA + causal mask + softmax + output projection) and one fused FFN kernel (W1 + GELU + W2) per layer, compiled per sequence-length bucket (32, 64, 128, 256, 512, 1024). Embedding and LayerNorm on CPU with Accelerate.

2. **CPU decode with KV cache** — Single-token generation runs entirely on CPU using NEON fp16 matmul (4-row unrolled), bypassing ANE dispatch overhead. LM head via GCD-parallel NEON fp16 over 50,257 vocab rows.

Includes a from-scratch BPE tokenizer (`gpt2_tokenizer.h`) and weight converter (`gpt2_convert.py`) that pulls weights from HuggingFace with no PyTorch dependency.

```bash
cd training

# Download and convert weights
pip install safetensors huggingface_hub
python3 gpt2_convert.py

# Build and run
make gpt2
./gpt2 --prompt "The meaning of life is" --tokens 100 --temp 0.8
```

### ANE Training (upstream)

The original [maderix](https://github.com/maderix/ANEtransformers) work: training a 109M-parameter Llama2-architecture transformer (Stories110M) directly on ANE. 12-layer forward+backward pass, 6 ANE kernel types per layer (72 kernels/step), 107 ms/step on M4. Adam optimizer, gradient accumulation, checkpoint/resume via `exec()` restart. See `training/train_large.m` and the [training README](training/README.md).

### M5 Hardware Investigation

Systematic probing of ANE behavior on Apple M5 (H16 family, same as M4). Key findings documented in [`training/m5result.md`](training/m5result.md):

| Question | Result |
|----------|--------|
| Can weights be swapped without recompile? | **No.** Weights baked at compile time. File overwrite + reload ignored. |
| Does `weightsBuffer` IOSurface override compiled weights? | **No.** Same output regardless. |
| Does QoS affect ANE frequency? | **No.** All QoS 0-63 work. Fixed frequency, no latency difference. |
| Can `_ANEChainingRequest` chain kernels without CPU round-trips? | **Validates but rejected by driver** (Error Code=15). Likely requires entitlements only CoreML holds. |
| Can `_ANEPerformanceStats` expose hardware counters? | Class exists with `hwExecutionTime` but requires factory construction via model `perfStatsMask`. |
| Real-time eval path? | `beginRealTimeTask` returns NO (needs entitlement). `evaluateRealTimeWithModel` works but no perf gain. |

### Fused Kernel Benchmarks (`training/bench_fused.m`)

Quantifies the value of operation fusion on ANE:

- **Dispatch overhead**: ~60-80 us per sequential ANE dispatch
- **Fused vs separate**: 1.5-3x speedup from fusing multiple convolutions into single MIL programs
- **Conclusion**: Fused MIL is the only viable path to high ANE utilization. Chaining API is inaccessible without Apple-internal entitlements. Intermediates stay in ANE SRAM when fused.

Full investigation: [`training/CHAINING_INVESTIGATION.md`](training/CHAINING_INVESTIGATION.md)

## How It Works

1. **MIL generation** — Objective-C constructs MIL program text at runtime: convolutions (linear layers), matmul (attention), softmax, element-wise ops
2. **In-memory compilation** — `_ANEInMemoryModelDescriptor` compiles MIL text + fp16 weight blobs directly to ANE programs, no disk `.mlmodelc` needed
3. **IOSurface I/O** — Tensors via IOSurface shared memory in `[1, C, 1, S]` channel-first fp16 format. Spatial dimension padded to minimum stride of 32. Minimum surface size 49152 bytes.
4. **Weight baking** — Weights compiled as BLOBFILE constants into MIL programs. No runtime weight update possible — must recompile to change weights.
5. **NEON fp16** — ARM NEON intrinsics for fast fp32-fp16 conversion at IOSurface boundaries and for CPU-side matmul during decode

## File Structure

```
├── api_exploration.m               # Initial ANE private API discovery
├── inmem_basic.m                   # In-memory MIL compilation proof-of-concept
├── inmem_bench.m                   # ANE dispatch latency benchmarks
├── inmem_peak.m                    # Peak TFLOPS measurement (2048x2048 matmul)
├── sram_bench.m                    # ANE SRAM bandwidth probing
├── sram_probe.m                    # SRAM size/layout exploration
└── training/
    ├── gpt2.m                      # GPT-2 124M inference: ANE prefill + CPU KV-cache decode
    ├── gpt2_convert.py             # HuggingFace → ANE weight converter (no PyTorch)
    ├── gpt2_tokenizer.h            # Self-contained BPE tokenizer (header-only C)
    ├── bench_fused.m               # Fused vs separate kernel benchmarks
    ├── CHAINING_INVESTIGATION.md   # Full chaining API reverse-engineering writeup
    ├── m5result.md                 # M5 ANE probe results
    ├── train_large.m               # 12-layer Stories110M ANE training (upstream)
    ├── stories_config.h            # Training model config and structs
    ├── stories_io.h                # Training IOSurface I/O and kernel compile/eval
    ├── stories_mil.h               # Training MIL generators (6 kernel types)
    ├── stories_cpu_ops.h           # vDSP RMSNorm, cross-entropy, Adam, embeddings
    ├── dashboard.py                # Training TUI: loss curves, power, text generation
    ├── ane_runtime.h               # ANE private API wrapper
    ├── ane_mil_gen.h               # MIL generation helpers
    ├── test_chaining.m             # Chaining API experiments
    ├── test_weight_reload.m        # Weight swap without recompile test
    ├── test_ane_advanced.m         # weightsBuffer, procedureIndex, shared events probe
    ├── test_qos_sweep.m            # QoS 0-63 latency sweep
    ├── test_perf_stats.m           # _ANEPerformanceStats introspection
    ├── test_decode_attn.m          # Multi-input decode attention kernel validation
    ├── test_multi_input.m          # Multi-input IOSurface size constraints
    ├── test_ffn_seq1.m             # FFN at seq=1 (decode mode) validation
    ├── test_lm_head_ane.m          # LM head on ANE feasibility
    ├── test_lm_head_fast.m         # LM head CPU benchmark (6 approaches)
    ├── test_lm_head_neon.m         # LM head NEON fp16 benchmark
    ├── docs/                       # Roadmap: GPT-2 XL, streaming, sampling, interactive
    └── Makefile
```

## Building

Requires macOS 15+ on Apple Silicon (tested on M4, M5). No external dependencies — system frameworks + private ANE APIs resolved at runtime via `objc_msgSend`.

```bash
cd training

# GPT-2 inference
make gpt2
./gpt2 --prompt "Once upon a time" --tokens 200

# Stories110M training (upstream)
make train_large
./train_large

# Benchmarks
make bench_fused
./bench_fused
```

## Known ANE Constraints

- **Weights are immutable after compile** — no hot-swap, no `weightsBuffer` override, no file-swap reload
- **SDPA causal masking ignored by hardware** — must decompose into Q@K^T + mask add + softmax + scores@V
- **~119 compile limit per process** — ANE compiler leaks resources; training uses `exec()` restart
- **IOSurface minimum 49152 bytes** — even for tiny tensors (seq=1 decode)
- **Spatial stride padded to 32** — `[1, C, 1, W]` surfaces have stride `max(W, 32)`
- **Chaining API inaccessible** — `_ANEChainingRequest` validates but driver rejects (Error Code=15)

## Disclaimer

This project is independent research into Apple Neural Engine architecture. It uses undocumented APIs discovered through runtime introspection for research and educational purposes under fair use and interoperability provisions (see *Sega v. Accolade*, 1992; DMCA §1201(f)). No Apple proprietary code or binaries are included in this repository. This project is not affiliated with or endorsed by Apple Inc. Use at your own risk.

## License

MIT — see [LICENSE](LICENSE)

