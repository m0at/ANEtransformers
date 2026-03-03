# ANE Fused K+V: 2-Output Conv Programs

## Summary

The Apple Neural Engine supports multi-output MIL programs with **exactly 2 outputs**.
This lets us fuse K and V projections into a single compiled program, reducing
per-layer program count from 5 to 4 (160 → 128 total for Mistral 7B's 32 layers).

3-output programs (fused Q+K+V) are **broken** — only the last output is correct.

## The Problem

ANE requires baked weights, so every weight matrix needs its own compiled program.
Mistral 7B has 7 weight matrices per layer × 32 layers = 224 unique matrices. With
fused FFN (W1+W3+W2+sigmoid+mul in one program), we had 5 programs per layer:

```
Q:  conv [dim→dim]        — 1 program
K:  conv [dim→kv_dim]     — 1 program
V:  conv [dim→kv_dim]     — 1 program
Wo: conv [dim→dim]        — 1 program
FFN: 3 convs + sigmoid    — 1 program
                            ─────────
                            5 × 32 = 160 programs (over 128 limit!)
```

## The Fix: Fused K+V

K and V share the same dimensions `[dim→kv_dim]` and the same input tensor.
A single MIL program with 2 baked conv ops and 2 outputs handles both:

```
Q:      conv [dim→dim]        — 1 program
K+V:    2 convs [dim→kv_dim]  — 1 program, 2 outputs
Wo:     conv [dim→dim]        — 1 program
FFN:    3 convs + sigmoid     — 1 program
                                ─────────
                                4 × 32 = 128 programs (at limit!)
```

## MIL Program Structure

```
func main<ios18>(tensor<fp32, [1, dim, 1, S]> x) {
    x16 = cast(x, fp16);

    Wk = const(BLOBFILE("weight.bin", offset=64));       // [kv_dim, dim, 1, 1]
    Wv = const(BLOBFILE("weight.bin", offset=...));      // [kv_dim, dim, 1, 1]

    k16 = conv(weight=Wk, x=x16);   // [1, kv_dim, 1, S]
    v16 = conv(weight=Wv, x=x16);   // [1, kv_dim, 1, S]

    k = cast(k16, fp32);
    v = cast(v16, fp32);
} -> (k, v);    // 2 outputs
```

Key requirement: both outputs must have the **same shape** `[1, kv_dim, 1, S]`.

## Weight Blob Layout (DEADBEEF format)

```
Offset    Content
──────────────────────────────────────
0x000     Global header (64 bytes)
            [0]=0x01, [4]=0x02, rest zeros

0x040     Wk chunk header (64 bytes)
            [0:4] = 0xDEADBEEF (magic)
            [4]   = 0x01
            [8:12] = data_size (kv_dim × dim × 2)
            [16:20] = absolute data offset (0x080)

0x080     Wk fp16 data (kv_dim × dim × 2 bytes)

0x080 + wk_size    Wv chunk header (64 bytes)
                     Same structure, offset points to Wv data

0x080 + wk_size + 64    Wv fp16 data
```

Total blob size: `64 + 2 × (64 + kv_dim × dim × 2)` bytes.

For Mistral 7B (kv_dim=1024, dim=4096): `64 + 2 × (64 + 8,388,608)` = **16.8 MB**.
For Qwen2.5-3B (kv_dim=256, dim=2048): `64 + 2 × (64 + 1,048,576)` = **2.1 MB**.

## I/O Setup

The program has 1 input and 2 outputs. IOSurface allocation:

```c
size_t inSz  = dim * S * sizeof(float);      // input: x
size_t outSzK = kv_dim * S * sizeof(float);   // output 0: K
size_t outSzV = kv_dim * S * sizeof(float);   // output 1: V
size_t outSizes[2] = {outSzK, outSzV};

ANEKernel *k = ane_compile(milData, blob, 1, &inSz, 2, outSizes);
```

Reading outputs:

```c
ane_write_input(k, 0, x_data, inSz);
ane_eval(k);
ane_read_output(k, 0, K_out, outSzK);   // K projection
ane_read_output(k, 1, V_out, outSzV);   // V projection
```

## Why 3 Outputs Fail

Tested empirically: with 3 outputs (fused Q+K+V), only the **last output** (index 2)
contains correct data. Outputs 0 and 1 are garbled/zeros. This is NOT related to
mixed dimensions — even 3 identical `[256→64]` convs fail for outputs 0 and 1.

This appears to be a hardware or driver limitation of the ANE's multi-output
routing. The workaround is to never use more than 2 outputs per program.

## Data Layout Warning

ANE tensors use **channel-first** layout: `[1, C, 1, S]` stored as `data[c * S + s]`.

Standard transformer code uses token-first: `data[t * dim + d]`.

**Every input must be transposed** from `[S, dim]` (token-major) to `[dim, S]`
(channel-major) before writing to IOSurface. Every output must be transposed back.

```c
// Token-first → Channel-first (before ANE dispatch)
for (int c = 0; c < dim; c++)
    for (int s = 0; s < S; s++)
        ane_buf[c * S + s] = token_buf[s * dim + c];

// Channel-first → Token-first (after ANE dispatch)
for (int c = 0; c < kv_dim; c++)
    for (int s = 0; s < S; s++)
        token_buf[s * kv_dim + c] = ane_buf[c * S + s];
```

At dim=4096, S=64: 256K floats = 1MB per transpose. Negligible vs conv compute.

## Code References

- MIL generator: `mistral/ane_mil_gen_mistral.h` → `mil_gen_kv_fused()`
- Blob builder: `mistral/ane_mil_gen_mistral.h` → `mil_build_kv_fused_blob()`
- Prefill dispatch: `mistral/mistral_ane_prefill.h` → `ANE_LK_KV` enum
- Runtime: `training/ane_runtime.h` → `ane_compile()`, `ane_eval()`

## Applicability to Other Models

For Qwen2.5-3B (kv_heads=2, kv_dim=256): K+V blob is only 2.1MB — trivially small.
With K+V fused, programs per layer drop to 4, giving 36 × 4 = 144 (still over 128).
Better strategy for Qwen: run K+V on CPU (kv_dim=256 is tiny) → 3 programs/layer = 108.
