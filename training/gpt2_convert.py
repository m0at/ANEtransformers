#!/usr/bin/env python3
"""Download GPT-2 small (124M) and convert weights to ANE blob format.
Uses safetensors + huggingface_hub directly, no torch dependency."""

import struct, os, json, base64
import numpy as np
from safetensors import safe_open
from huggingface_hub import hf_hub_download

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpt2_weights")
MODEL_ID = "openai-community/gpt2"


def save_blob(path, arr):
    data = arr.astype(np.float16).flatten().tobytes()
    wsize = len(data)
    buf = bytearray(128 + wsize)
    buf[0] = 1; buf[4] = 2
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE
    buf[68] = 1
    struct.pack_into('<I', buf, 72, wsize)
    struct.pack_into('<I', buf, 80, 128)
    buf[128:] = data
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(buf)


def bytes_to_unicode():
    """GPT-2's byte-to-unicode mapping."""
    bs = list(range(ord('!'), ord('~')+1)) + list(range(ord('¡'), ord('¬')+1)) + list(range(ord('®'), ord('ÿ')+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Downloading GPT-2 weights...")
    sf_path = hf_hub_download(MODEL_ID, "model.safetensors")
    print(f"  safetensors: {sf_path}")

    f = safe_open(sf_path, framework="numpy")
    keys = list(f.keys())
    print(f"  {len(keys)} tensors")

    count = 0
    def save(name, arr):
        nonlocal count
        path = os.path.join(OUT_DIR, name)
        save_blob(path, arr)
        count += 1
        print(f"  {name}: {arr.shape} -> {arr.size*2} bytes")

    # Global weights
    print("\nGlobal weights:")
    save("wte.bin", f.get_tensor("wte.weight"))           # [50257, 768]
    save("wpe.bin", f.get_tensor("wpe.weight"))            # [1024, 768]
    save("ln_f_w.bin", f.get_tensor("ln_f.weight"))        # [768]
    save("ln_f_b.bin", f.get_tensor("ln_f.bias"))          # [768]

    # Per-layer
    for i in range(12):
        p = f"h.{i}"
        d = f"layer_{i:02d}"
        print(f"\nLayer {i}:")

        save(f"{d}/ln1_w.bin", f.get_tensor(f"{p}.ln_1.weight"))
        save(f"{d}/ln1_b.bin", f.get_tensor(f"{p}.ln_1.bias"))

        # c_attn: [768, 2304] in GPT-2 Conv1D format -> split + transpose
        c_attn_w = f.get_tensor(f"{p}.attn.c_attn.weight")  # [768, 2304]
        save(f"{d}/wq.bin", c_attn_w[:, :768].T.copy())
        save(f"{d}/wk.bin", c_attn_w[:, 768:1536].T.copy())
        save(f"{d}/wv.bin", c_attn_w[:, 1536:].T.copy())

        c_attn_b = f.get_tensor(f"{p}.attn.c_attn.bias")    # [2304]
        save(f"{d}/bq.bin", c_attn_b[:768])
        save(f"{d}/bk.bin", c_attn_b[768:1536])
        save(f"{d}/bv.bin", c_attn_b[1536:])

        save(f"{d}/wo.bin", f.get_tensor(f"{p}.attn.c_proj.weight").T.copy())
        save(f"{d}/bo.bin", f.get_tensor(f"{p}.attn.c_proj.bias"))

        save(f"{d}/ln2_w.bin", f.get_tensor(f"{p}.ln_2.weight"))
        save(f"{d}/ln2_b.bin", f.get_tensor(f"{p}.ln_2.bias"))

        save(f"{d}/w1.bin", f.get_tensor(f"{p}.mlp.c_fc.weight").T.copy())
        save(f"{d}/b1.bin", f.get_tensor(f"{p}.mlp.c_fc.bias"))

        save(f"{d}/w2.bin", f.get_tensor(f"{p}.mlp.c_proj.weight").T.copy())
        save(f"{d}/b2.bin", f.get_tensor(f"{p}.mlp.c_proj.bias"))

    # Tokenizer
    print("\nTokenizer:")
    vocab_path = hf_hub_download(MODEL_ID, "vocab.json")
    merges_path = hf_hub_download(MODEL_ID, "merges.txt")

    with open(vocab_path) as vf:
        encoder = json.load(vf)  # token_string -> id

    byte_map = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_map.items()}

    # encoder.txt: id -> base64(raw_bytes)
    id_to_token = {v: k for k, v in encoder.items()}
    enc_out = os.path.join(OUT_DIR, "encoder.txt")
    with open(enc_out, 'w') as ef:
        for tid in range(len(encoder)):
            tok_str = id_to_token.get(tid, "")
            raw = bytes([byte_decoder[c] for c in tok_str])
            b64 = base64.b64encode(raw).decode('ascii')
            ef.write(f"{tid}\t{b64}\n")
    print(f"  encoder.txt: {len(encoder)} tokens")

    # merges.txt
    with open(merges_path) as mf:
        lines = mf.read().strip().split('\n')
    # Skip the first line if it's a header (#version)
    if lines[0].startswith('#'):
        lines = lines[1:]
    merge_out = os.path.join(OUT_DIR, "merges.txt")
    with open(merge_out, 'w') as mof:
        for line in lines:
            mof.write(line + '\n')
    print(f"  merges.txt: {len(lines)} merges")

    print(f"\nDone. {count} weight files saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
