# ANE 2-Phase Loading: Cold Compile + Warm Forge

## Summary

ANE programs require baked weights, meaning every layer needs its own compiled programs.
For Mistral 7B (32 layers x 4 programs = 128 total), cold compilation takes ~80s. The
2-phase strategy eliminates this on subsequent runs:

- **Phase 1 (cold start):** Stream-compile all 128 programs, save hexIds to manifest (~80s)
- **Phase 2 (warm start):** Forged-load from daemon cache using saved hexIds (~220ms)
- **Speedup:** ~400x faster warm-start initialization

## The Problem

ANE's private API (`_ANEInMemoryModel`) requires weights baked into compiled programs.
No way to share compiled programs across layers — each weight matrix gets its own.

Mistral 7B breakdown:
```
Per layer (×32):
  Q:    conv [4096→4096]     — 1 program, 32MB blob
  K+V:  2 convs [4096→1024]  — 1 program, 16.8MB blob (fused, 2 outputs)
  Wo:   conv [4096→4096]     — 1 program, 32MB blob
  FFN:  3 convs + sigmoid    — 1 program, 352MB blob (W1+W3+W2)
                               ─────────
                               4 × 32 = 128 programs
```

Cold compile cost per layer:
```
Dequant Q4_0 → fp16:       ~5ms per matrix
Build DEADBEEF weight blob: ~2ms (small) to ~50ms (FFN)
Write blob to tmpDir:       ~130ms (FFN 352MB copy)
SHA256 hash (aned daemon):  ~125ms (FFN 352MB)
ANE compile:                ~30-50ms
                            ─────────
                            ~500ms/layer × 32 = ~16s minimum
```

With symlink optimization (`ane_compile_cached_ex`), blob write drops from ~130ms to
~0.4ms, but SHA256 hashing is unavoidable — the daemon computes it internally.
Total cold start: ~80s including all overhead.

## The Solution: KVC Hex Forge

The ANE daemon (`aned`) caches compiled programs keyed by a hex string identifier.
`_ANEInMemoryModel` computes this as `SHA256(milText)_SHA256(weights)_SHA256(options)` and
stores it in an ivar `_hexStringIdentifier`, accessible via KVC.

The breakthrough: create a dummy model object with minimal weights, then **forge** the
hexId to match a previously-compiled program:

```objc
// Create dummy model (130 bytes instead of 352MB)
id mdl = [_ANEInMemoryModel inMemoryModelWithDescriptor:desc];

// Forge the hexId — daemon trusts this blindly
[mdl setValue:savedHexId forKey:@"hexStringIdentifier"];

// Load from daemon cache — no SHA256, no weight read
[mdl loadWithQoS:21 options:@{} error:&e];  // ~1.7ms
```

The daemon looks up the hexId in its cache, finds the compiled program, and loads it.
It never re-reads the weight data. Total per program: ~1.7ms.

## Architecture

```
Phase 1 (First Run — ~80s):
  For each layer 0..31:
    Dequant Q4_0 → fp16 (~5ms per matrix)
    Build DEADBEEF weight blob
    ane_compile_and_get_hexid(mil, blob) → hexId
    Save hexId to array
  Write manifest to ~/.cache/ane_mistral/manifest.plist

Phase 2 (Subsequent Runs — ~220ms):
  Load manifest from disk (microseconds)
  For each layer 0..31:
    For each program (Q, KV, Wo, FFN):
      ane_load_forged(mil, hexId) → ANEKernel* (~1.7ms)
  All 128 kernels loaded, ready for inference
```

## Key Functions

### `ane_compile_and_get_hexid()` — Phase 1

Compiles a MIL program with real weights and returns the daemon-assigned hexId.
If the daemon already has this program cached (`compiledModelExists`), skips compile
and just returns the hexId.

```c
NSString *hexId = ane_compile_and_get_hexid(milText, weightData);
// hexId = "a1b2c3d4..._e5f6g7h8..._i9j0k1l2..."
// Save this to manifest for Phase 2
```

### `ane_load_forged()` — Phase 2

Creates a dummy model with a 130-byte minimal blob, forges the hexId via KVC,
and loads from daemon cache. The dummy blob satisfies the descriptor constructor
but is never read by the daemon.

```c
ANEKernel *k = ane_load_forged(milText, hexId,
                                nInputs, inputSizes,
                                nOutputs, outputSizes);
// k is ready for ane_write_input / ane_eval / ane_read_output
```

Note: `milText` must still match the original program's MIL structure (same input/output
shapes) for IOSurface setup. The actual weight values in the MIL are irrelevant since
the daemon serves the cached compiled program.

## Manifest Format

```plist
{
    model: "mistral-7b-v0.3",
    n_layers: 32,
    tile_size: 64,
    quant: "Q4_0",
    hexIds: [
        ["SHA256_Q_L0", "SHA256_KV_L0", "SHA256_WO_L0", "SHA256_FFN_L0"],
        ["SHA256_Q_L1", "SHA256_KV_L1", "SHA256_WO_L1", "SHA256_FFN_L1"],
        ...
    ]
}
```

Stored at `~/.cache/ane_mistral/manifest.plist`. Read/write via `NSPropertyListSerialization`.

Validation on load: check `n_layers`, `tile_size`, and `quant` match current config.
If mismatched, discard manifest and fall through to Phase 1.

## Memory Layout

128 loaded `ANEKernel` objects at runtime:

```
ANEKernel structs:     128 × 96 bytes = ~12 KB
IOSurfaces:            128 × (inputs + outputs) × ~1MB each
Compiled programs:     In daemon memory (not user process)
```

IOSurface optimization: since layers execute sequentially, input/output surfaces can
be shared across programs within the same layer. Only need surfaces for the largest
program (FFN: 1 input + 1 output at `hidden_dim × S × 4` bytes each).

With S=64, dim=4096, hidden=14336:
```
Input:   4096 × 64 × 4 = 1 MB
Output: 14336 × 64 × 4 = 3.5 MB (FFN, largest)
Total:  ~5 MB shared surfaces (not 256 MB)
```

## Risks and Mitigations

### 1. Daemon cache lost on reboot

The `aned` daemon cache is in-memory only. After reboot, Phase 2 loads will fail
(daemon returns error on unknown hexId). Mitigation: detect load failure, fall through
to Phase 1 recompile, update manifest.

```c
ANEKernel *k = ane_load_forged(mil, hexId, ...);
if (!k) {
    // Cache miss — recompile this program
    hexId = ane_compile_and_get_hexid(mil, realWeightData);
    k = ane_compile_cached(mil, realWeightData, ...);
    // Update manifest with new hexId
}
```

### 2. Stale manifest (model weights changed)

If the GGUF file changes, saved hexIds won't match. Mitigation: store a hash of the
GGUF file path + modification time in the manifest. Invalidate on mismatch.

### 3. Forged unload crash

Calling `unloadWithQoS:` on a forged model may crash because the descriptor is
built from a dummy blob. Mitigation: wrap in `@try/@catch`:

```objc
@try {
    [mdl unloadWithQoS:21 error:&e];
} @catch (NSException *ex) {
    // Silently ignore — daemon will clean up on process exit
}
```

### 4. Tile size mismatch

If the user requests a different sequence length than the manifest's `tile_size`,
all hexIds are invalid (MIL shapes differ). Mitigation: include `tile_size` in
manifest, recompile if changed.

## Code References

- `ane_compile_and_get_hexid()`: `training/ane_runtime.h`
- `ane_load_forged()`: `training/ane_runtime.h`
- `ane_compile_cached_ex()`: `training/ane_runtime.h` (symlink optimization)
- Manifest I/O: `mistral/mistral_ane_prefill.h`
- Compile-all: `mistral/mistral_ane_prefill.h` → `ane_compile_all_programs()`
- Forward loop: `mistral/mistral_ane_prefill.h` → `ane_baked_prefill_forward()`
- MIL generation: `mistral/ane_mil_gen_mistral.h`
