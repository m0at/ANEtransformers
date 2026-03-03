#!/usr/bin/env python3
"""
Needle-in-a-Haystack probe designed to isolate KV-cache eviction failures.

Key idea:
- Build a long "DOCUMENT" with multiple unique needles inserted at known token depths.
- Ask questions that require retrieving needles from different depths.
- Compare:
    (A) baseline (no eviction)
    (B) simulated eviction policies that keep only a subset of tokens
        *while preserving original RoPE positions via position_ids*.

Backends:
- vLLM: realistic serving-style prefill/decode (no KV eviction simulation here)
- transformers: manual prefill+greedy decode + eviction simulation via token selection + position_ids

Usage examples at bottom.
"""

import argparse
import inspect
import random
import re
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------
# Data model
# ---------------------------

@dataclass
class Needle:
    needle_id: str
    value: str
    depth_frac: float          # desired insertion depth within document portion
    start_token_idx: int = -1  # populated after build
    is_decoy: bool = False     # decoy needles have similar IDs but different values


NEEDLE_RE = re.compile(r"\bV\d{12}\b")  # value format: V + 12 digits
MULTI_VALUE_RE = re.compile(r"V\d{12}")  # for multi-needle extraction


# ---------------------------
# Prompt construction
# ---------------------------

def _rand_value(rng: random.Random) -> str:
    # 12 digits keeps it unambiguous but short
    return "V" + "".join(str(rng.randint(0, 9)) for _ in range(12))


def _make_needles(num_needles: int, seed: int) -> List[Needle]:
    rng = random.Random(seed)

    # Depths: force coverage of early/mid/late/extreme tail.
    # If you want a smooth retention curve, set --depth_mode linear.
    depths = []
    if num_needles == 1:
        depths = [0.5]
    else:
        # Start with some "stress points"
        stress = [0.02, 0.08, 0.20, 0.35, 0.50, 0.65, 0.80, 0.92, 0.98]
        if num_needles <= len(stress):
            depths = stress[:num_needles]
        else:
            # Fill extras linearly in the middle
            depths = stress[:]
            extra = num_needles - len(stress)
            for i in range(extra):
                depths.append((i + 1) / (extra + 1))
            depths = sorted(depths[:num_needles])

    needles: List[Needle] = []
    for i, d in enumerate(depths):
        nid = f"N{i:02d}_{rng.randint(100000, 999999)}"
        needles.append(Needle(needle_id=nid, value=_rand_value(rng), depth_frac=float(d)))
    return needles


def _make_decoys(needles: List[Needle], seed: int) -> List[Needle]:
    """
    For each real needle, create a decoy with a similar ID (appends 'X') and a
    different value. Decoys are placed at depth >= 0.90 so they sit in the
    "recency zone" — the part of the context that survives most eviction policies.

    If the model returns a decoy value instead of the real one, that's a clean
    signal of recency-biased substitution.
    """
    rng = random.Random(seed + 7777)
    decoys: List[Needle] = []
    for nd in needles:
        decoy_id = nd.needle_id + "X"
        decoy_val = _rand_value(rng)
        # Place decoys in the last 10% of the document
        decoy_depth = 0.90 + rng.random() * 0.09
        decoys.append(Needle(
            needle_id=decoy_id,
            value=decoy_val,
            depth_frac=decoy_depth,
            is_decoy=True,
        ))
    return decoys


def _repeat_to_length(pattern: List[int], n_tokens: int) -> List[int]:
    if n_tokens <= 0:
        return []
    if len(pattern) == 0:
        raise ValueError("Filler pattern tokenized to 0 tokens; choose a different filler string.")
    reps = n_tokens // len(pattern)
    rem = n_tokens % len(pattern)
    return pattern * reps + pattern[:rem]


def build_document_tokens(
    tokenizer,
    doc_tokens: int,
    needles: List[Needle],
    filler_text: str = " the"
) -> Tuple[List[int], List[Needle]]:
    """
    Build the DOCUMENT portion of exactly doc_tokens tokens, inserting needle lines
    near specified depth fractions.

    Returns:
        doc_ids: list of token ids length == doc_tokens
        needles: needles with start_token_idx filled (relative to DOCUMENT start)
    """
    filler_ids = tokenizer.encode(filler_text, add_special_tokens=False)
    if len(filler_ids) == 0:
        raise ValueError(f"Filler text {filler_text!r} produced 0 tokens.")

    # Needle token ids
    needle_tok: List[Tuple[Needle, List[int]]] = []
    for nd in needles:
        # Keep needles visually distinctive & parseable.
        # Put them on their own lines to reduce model confusion.
        needle_text = f"\n[NEEDLE id={nd.needle_id} value={nd.value}]\n"
        ids = tokenizer.encode(needle_text, add_special_tokens=False)
        needle_tok.append((nd, ids))

    # Desired start positions in DOCUMENT token space
    desired_starts = []
    for nd, ids in needle_tok:
        pos = int(max(0.0, min(0.999999, nd.depth_frac)) * doc_tokens)
        desired_starts.append(pos)

    # Build sequentially, assuming needles are short relative to spacing.
    doc_ids: List[int] = []
    cursor = 0
    for (nd, ids), pos in sorted(zip(needle_tok, desired_starts), key=lambda x: x[1]):
        # Ensure we never exceed doc_tokens; if too tight, clamp.
        pos = max(cursor, min(pos, doc_tokens))
        filler_needed = pos - cursor
        doc_ids.extend(_repeat_to_length(filler_ids, filler_needed))
        cursor += filler_needed

        if cursor + len(ids) > doc_tokens:
            # Not enough room: truncate the needle insertion (rare unless doc_tokens is tiny).
            ids = ids[: max(0, doc_tokens - cursor)]
        nd.start_token_idx = cursor
        doc_ids.extend(ids)
        cursor += len(ids)

        if cursor >= doc_tokens:
            break

    # Fill remainder
    if cursor < doc_tokens:
        doc_ids.extend(_repeat_to_length(filler_ids, doc_tokens - cursor))

    doc_ids = doc_ids[:doc_tokens]
    assert len(doc_ids) == doc_tokens
    return doc_ids, needles


def build_full_prompt_ids(
    tokenizer,
    doc_ids: List[int],
    query_needle: Needle,
) -> Tuple[List[int], str]:
    """
    Build the full prompt (prefix + DOCUMENT + suffix question).
    Returns token ids and the expected exact value.
    """
    prefix = (
        "You will be given a long document.\n"
        "The document contains lines of the form:\n"
        "[NEEDLE id=<ID> value=<VALUE>]\n"
        "Your job is to answer questions about them.\n\n"
        "DOCUMENT START\n"
    )
    suffix = (
        "\nDOCUMENT END\n\n"
        f"Question: What is the value for id {query_needle.needle_id}?\n"
        "Reply with ONLY the value (e.g., V123...).\n"
        "Answer:"
    )

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)

    ids: List[int] = []
    if tokenizer.bos_token_id is not None:
        ids.append(tokenizer.bos_token_id)
    ids.extend(prefix_ids)
    ids.extend(doc_ids)
    ids.extend(suffix_ids)

    return ids, query_needle.value


def build_multi_needle_prompt_ids(
    tokenizer,
    doc_ids: List[int],
    query_needles: List[Needle],
) -> Tuple[List[int], List[str]]:
    """
    Build a prompt that asks for multiple needle values at once.
    Returns token ids and list of expected values (in query order).
    """
    id_list = ", ".join(nd.needle_id for nd in query_needles)
    prefix = (
        "You will be given a long document.\n"
        "The document contains lines of the form:\n"
        "[NEEDLE id=<ID> value=<VALUE>]\n"
        "Your job is to answer questions about them.\n\n"
        "DOCUMENT START\n"
    )
    suffix = (
        "\nDOCUMENT END\n\n"
        f"Question: What are the values for ids {id_list}?\n"
        "Reply with ONLY the values separated by commas, in the same order as asked.\n"
        "Answer:"
    )

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)

    ids: List[int] = []
    if tokenizer.bos_token_id is not None:
        ids.append(tokenizer.bos_token_id)
    ids.extend(prefix_ids)
    ids.extend(doc_ids)
    ids.extend(suffix_ids)

    expected = [nd.value for nd in query_needles]
    return ids, expected


def extract_multi_values(text: str) -> List[str]:
    return MULTI_VALUE_RE.findall(text)


# ---------------------------
# Eviction policy (token selection)
# ---------------------------

def keep_indices_none(seq_len: int) -> torch.LongTensor:
    return torch.arange(seq_len, dtype=torch.long)


def keep_indices_keep_last(seq_len: int, keep_last: int, keep_first: int) -> torch.LongTensor:
    keep_last = max(0, min(seq_len, keep_last))
    keep_first = max(0, min(seq_len, keep_first))
    idx = set(range(keep_first)) | set(range(seq_len - keep_last, seq_len))
    return torch.tensor(sorted(idx), dtype=torch.long)


def keep_indices_uniform_budget(seq_len: int, budget: int, keep_first: int, keep_last: int) -> torch.LongTensor:
    """
    Keep:
      - first keep_first tokens
      - last keep_last tokens
      - plus uniformly sampled tokens from the middle to hit 'budget' total kept tokens

    This approximates "compression/eviction with a fixed KV budget".
    """
    budget = max(1, min(seq_len, budget))
    keep_first = max(0, min(seq_len, keep_first))
    keep_last = max(0, min(seq_len, keep_last))

    base = set(range(keep_first)) | set(range(seq_len - keep_last, seq_len))
    if len(base) >= budget:
        # If base already exceeds budget, trim from the middle of base (keep endpoints).
        base_sorted = sorted(base)
        return torch.tensor(base_sorted[:budget], dtype=torch.long)

    remaining = budget - len(base)
    mid_start = keep_first
    mid_end = seq_len - keep_last
    mid_len = max(0, mid_end - mid_start)
    if mid_len == 0 or remaining == 0:
        return torch.tensor(sorted(base), dtype=torch.long)

    # Uniform sample positions in the middle
    step = max(1, mid_len // remaining)
    sampled = set(range(mid_start, mid_end, step))
    idx = sorted(base | sampled)

    # If overshoot, downsample deterministically
    if len(idx) > budget:
        # Keep endpoints, thin the middle
        head = [i for i in idx if i < keep_first]
        tail = [i for i in idx if i >= seq_len - keep_last]
        middle = [i for i in idx if keep_first <= i < seq_len - keep_last]
        need = budget - len(head) - len(tail)
        if need <= 0:
            idx = (head + tail)[:budget]
        else:
            stride = max(1, len(middle) // need)
            middle2 = middle[::stride][:need]
            idx = sorted(head + middle2 + tail)

    return torch.tensor(idx, dtype=torch.long)


def compute_keep_indices(args, seq_len: int) -> torch.LongTensor:
    if args.evict_policy == "none":
        return keep_indices_none(seq_len)
    if args.evict_policy == "keep_last":
        return keep_indices_keep_last(seq_len, keep_last=args.keep_last, keep_first=args.keep_first)
    if args.evict_policy == "uniform_budget":
        return keep_indices_uniform_budget(seq_len, budget=args.budget, keep_first=args.keep_first, keep_last=args.keep_last)
    raise ValueError(f"Unknown evict_policy: {args.evict_policy}")


# ---------------------------
# Transformers backend: manual prefill + greedy decode with position_ids
# ---------------------------

def _forward_supports(model, name: str) -> bool:
    try:
        sig = inspect.signature(model.forward)
        return name in sig.parameters
    except Exception:
        return False


@torch.inference_mode()
def transformers_greedy_generate(
    model,
    tokenizer,
    input_ids_full: torch.LongTensor,        # shape [1, S_kept]
    position_ids_full: Optional[torch.LongTensor],  # shape [1, S_kept] with original positions
    max_new_tokens: int,
    chunk_size: int = 4096,
) -> Tuple[str, Dict[str, float]]:
    """
    Manual chunked prefill to reduce activation peak + greedy decode.
    Returns decoded output (generated portion only) and timing stats.
    """
    device = next(model.parameters()).device
    input_ids_full = input_ids_full.to(device)
    if position_ids_full is not None:
        position_ids_full = position_ids_full.to(device)

    use_position_ids = position_ids_full is not None and _forward_supports(model, "position_ids")
    use_cache_position = position_ids_full is not None and (not use_position_ids) and _forward_supports(model, "cache_position")

    # Prefill (chunked)
    t0 = time.perf_counter()
    past = None
    last_logits = None

    S = input_ids_full.shape[1]
    for start in range(0, S, chunk_size):
        end = min(S, start + chunk_size)
        chunk_ids = input_ids_full[:, start:end]

        kwargs = dict(use_cache=True, past_key_values=past)
        if position_ids_full is not None:
            chunk_pos = position_ids_full[:, start:end]
            if use_position_ids:
                kwargs["position_ids"] = chunk_pos
            elif use_cache_position:
                # cache_position is often expected as 1D
                kwargs["cache_position"] = chunk_pos.squeeze(0)

        out = model(input_ids=chunk_ids, **kwargs)
        past = out.past_key_values
        last_logits = out.logits[:, -1, :]

    prefill_s = time.perf_counter() - t0

    # Decode (greedy)
    t1 = time.perf_counter()
    generated: List[int] = []
    assert last_logits is not None

    # Next absolute position: last kept original position + 1
    if position_ids_full is not None:
        next_pos = int(position_ids_full[0, -1].item()) + 1
    else:
        # fallback: contiguous
        next_pos = S

    next_token = torch.argmax(last_logits, dim=-1)  # [1]

    for step in range(max_new_tokens):
        kwargs = dict(use_cache=True, past_key_values=past)
        if position_ids_full is not None:
            pos = torch.tensor([[next_pos]], dtype=torch.long, device=device)
            if use_position_ids:
                kwargs["position_ids"] = pos
            elif use_cache_position:
                kwargs["cache_position"] = pos.squeeze(0)

        out = model(input_ids=next_token.unsqueeze(0), **kwargs)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)
        generated.append(int(next_token.item()))
        next_pos += 1

        # crude early stop: newline or EOS
        if tokenizer.eos_token_id is not None and generated[-1] == tokenizer.eos_token_id:
            break
        if step >= 2:
            text_so_far = tokenizer.decode(generated, skip_special_tokens=True)
            if "\n" in text_so_far:
                break

    decode_s = time.perf_counter() - t1
    gen_text = tokenizer.decode(generated, skip_special_tokens=True)

    stats = {
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "gen_tokens": float(len(generated)),
        "tok_per_s_decode": (len(generated) / decode_s) if decode_s > 0 else 0.0,
    }
    return gen_text, stats


# ---------------------------
# vLLM backend
# ---------------------------

def vllm_generate(model_name: str, prompt_text: str, max_new_tokens: int, temperature: float, max_model_len: Optional[int]):
    from vllm import LLM, SamplingParams

    sp = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
    llm = LLM(model=model_name, dtype="bfloat16", max_model_len=max_model_len)
    t0 = time.perf_counter()
    out = llm.generate([prompt_text], sp)
    dt = time.perf_counter() - t0
    gen_text = out[0].outputs[0].text
    return gen_text, {"wall_s": dt}


# ---------------------------
# Scoring
# ---------------------------

def extract_value(text: str) -> Optional[str]:
    m = NEEDLE_RE.search(text)
    return m.group(0) if m else None


def run_one_query_transformers(args, tokenizer, model, full_ids: List[int], keep_idx: torch.LongTensor, expected: str):
    # Build kept ids + original positions to preserve RoPE phases for kept tokens
    full = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0)  # [1, S]
    kept = full[:, keep_idx]
    pos = keep_idx.unsqueeze(0)  # [1, S_kept]

    # Reset max memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    gen_text, stats = transformers_greedy_generate(
        model=model,
        tokenizer=tokenizer,
        input_ids_full=kept,
        position_ids_full=pos if args.preserve_positions else None,
        max_new_tokens=args.max_new_tokens,
        chunk_size=args.chunk_size,
    )

    got = extract_value(gen_text)
    ok = (got == expected)

    if torch.cuda.is_available():
        stats["peak_vram_gb"] = float(torch.cuda.max_memory_allocated() / 1e9)
    stats["extracted"] = got or ""
    stats["ok"] = ok
    return ok, gen_text, stats


def run_one_query_vllm(args, tokenizer, model_name: str, full_ids: List[int], expected: str):
    # decode full prompt to text for vLLM
    prompt_text = tokenizer.decode(full_ids, skip_special_tokens=False)

    gen_text, stats = vllm_generate(
        model_name=model_name,
        prompt_text=prompt_text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_model_len=args.max_model_len,
    )
    got = extract_value(gen_text)
    ok = (got == expected)
    stats["extracted"] = got or ""
    stats["ok"] = ok
    return ok, gen_text, stats


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    ap.add_argument("--doc_tokens", type=int, default=131072, help="DOCUMENT token length (excluding prefix/suffix)")
    ap.add_argument("--needles", type=int, default=8, help="Number of needles inserted across depth")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--decoys", action="store_true", help="Insert decoy needles near end with similar IDs to detect recency substitution")
    ap.add_argument("--multi_needle", type=int, default=0, metavar="K",
                    help="Also run K-needle multi-retrieval queries (0=disabled). Tests partial recall under eviction.")

    # Eviction
    ap.add_argument("--evict_policy", choices=["none", "keep_last", "uniform_budget"], default="none")
    ap.add_argument("--keep_first", type=int, default=16, help="Always keep first N tokens")
    ap.add_argument("--keep_last", type=int, default=8192, help="Always keep last N tokens")
    ap.add_argument("--budget", type=int, default=32768, help="Total tokens kept for uniform_budget policy")
    ap.add_argument("--preserve_positions", action="store_true", default=True, help="Preserve original RoPE positions via position_ids (recommended)")
    ap.add_argument("--no_preserve_positions", dest="preserve_positions", action="store_false")

    # Transformers perf
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--attn_impl", choices=["auto", "flash_attention_2", "sdpa", "eager"], default="auto")
    ap.add_argument("--chunk_size", type=int, default=4096)

    # vLLM
    ap.add_argument("--max_model_len", type=int, default=None, help="vLLM max_model_len (optional)")

    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Build needles + document once
    needles = _make_needles(args.needles, args.seed)
    decoys: List[Needle] = []
    if args.decoys:
        decoys = _make_decoys(needles, args.seed)
    all_needles = needles + decoys
    doc_ids, all_needles = build_document_tokens(tokenizer, args.doc_tokens, all_needles)
    # Split back: real needles are first, decoys after
    needles = [nd for nd in all_needles if not nd.is_decoy]
    decoys = [nd for nd in all_needles if nd.is_decoy]

    # Load model if transformers backend
    model = None
    if args.backend == "transformers":
        torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            device_map=None,              # keep simple; use a single GPU
            attn_implementation=None if args.attn_impl == "auto" else args.attn_impl,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()

    # Sweep: query each needle; report accuracy vs depth
    results = []
    print("\n--- Probe configuration ---")
    print(f"backend={args.backend} model={args.model}")
    print(f"doc_tokens={args.doc_tokens} needles={args.needles} seed={args.seed}")
    print(f"evict_policy={args.evict_policy} keep_first={args.keep_first} keep_last={args.keep_last} budget={args.budget}")
    print(f"preserve_positions={args.preserve_positions}")
    print(f"decoys={args.decoys} ({len(decoys)} decoy needles) multi_needle={args.multi_needle}")
    print("------------------------------------------------\n")

    for nd in needles:
        full_ids, expected = build_full_prompt_ids(tokenizer, doc_ids, nd)
        seq_len = len(full_ids)
        keep_idx = compute_keep_indices(args, seq_len)

        # Always ensure we kept the final token of the prompt
        if int(keep_idx[-1].item()) != seq_len - 1:
            # force-keep last token (question/Answer:)
            keep_idx = torch.unique(torch.cat([keep_idx, torch.tensor([seq_len - 1], dtype=torch.long)])).sort().values

        depth = nd.depth_frac
        print(f"[query] id={nd.needle_id} depth={depth:.3f} expected={expected}")

        if args.backend == "transformers":
            ok, gen_text, stats = run_one_query_transformers(args, tokenizer, model, full_ids, keep_idx, expected)
        else:
            ok, gen_text, stats = run_one_query_vllm(args, tokenizer, args.model, full_ids, expected)

        # Check for decoy substitution: did the model return a decoy's value instead?
        decoy_hit = ""
        if args.decoys and not ok:
            got_val = stats.get("extracted", "")
            for dc in decoys:
                if got_val == dc.value:
                    decoy_hit = f" DECOY_SUBSTITUTION(returned {dc.needle_id} value)"
                    break

        results.append((depth, nd.needle_id, ok, stats, decoy_hit))
        print(f"  ok={ok} extracted={stats.get('extracted','')!r}{decoy_hit}")
        if args.backend == "transformers":
            print(f"  prefill_s={stats['prefill_s']:.3f} decode_s={stats['decode_s']:.3f} tok/s(decode)={stats['tok_per_s_decode']:.1f} peak_vram_gb={stats.get('peak_vram_gb',0):.2f}")
        else:
            print(f"  wall_s={stats['wall_s']:.3f}")
        print(f"  raw_gen={gen_text.strip()!r}\n")

    # --- Multi-needle queries ---
    multi_results: List[Tuple[str, int, int, List[str], List[str]]] = []
    if args.multi_needle >= 2 and len(needles) >= 2:
        k = min(args.multi_needle, len(needles))
        # Build query groups: sliding window of size k across needles sorted by depth
        sorted_needles = sorted(needles, key=lambda n: n.depth_frac)
        groups: List[List[Needle]] = []
        for start_i in range(0, len(sorted_needles) - k + 1, max(1, k // 2)):
            groups.append(sorted_needles[start_i:start_i + k])
        # Also add a "spread" group: first + last + middle
        if len(sorted_needles) >= k:
            spread = [sorted_needles[0], sorted_needles[-1]]
            mid_indices = [int(i * (len(sorted_needles) - 1) / (k - 1)) for i in range(1, k - 1)]
            spread[1:1] = [sorted_needles[j] for j in mid_indices]
            groups.append(spread[:k])

        print("\n--- Multi-needle queries (K={}) ---\n".format(k))
        for gi, group in enumerate(groups):
            full_ids, expected_vals = build_multi_needle_prompt_ids(tokenizer, doc_ids, group)
            seq_len = len(full_ids)
            keep_idx = compute_keep_indices(args, seq_len)
            if int(keep_idx[-1].item()) != seq_len - 1:
                keep_idx = torch.unique(torch.cat([keep_idx, torch.tensor([seq_len - 1], dtype=torch.long)])).sort().values

            group_label = "+".join(f"{nd.needle_id}@{nd.depth_frac:.2f}" for nd in group)
            print(f"[multi-query {gi}] {group_label}")
            print(f"  expected: {expected_vals}")

            if args.backend == "transformers":
                full = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0)
                kept = full[:, keep_idx]
                pos = keep_idx.unsqueeze(0)
                gen_text, stats = transformers_greedy_generate(
                    model=model, tokenizer=tokenizer,
                    input_ids_full=kept,
                    position_ids_full=pos if args.preserve_positions else None,
                    max_new_tokens=args.max_new_tokens * k,
                    chunk_size=args.chunk_size,
                )
            else:
                prompt_text = tokenizer.decode(full_ids, skip_special_tokens=False)
                gen_text, stats = vllm_generate(
                    model_name=args.model, prompt_text=prompt_text,
                    max_new_tokens=args.max_new_tokens * k,
                    temperature=args.temperature, max_model_len=args.max_model_len,
                )

            got_vals = extract_multi_values(gen_text)
            hits = sum(1 for ev in expected_vals if ev in got_vals)
            print(f"  got: {got_vals}")
            print(f"  recall: {hits}/{len(expected_vals)}")
            print(f"  raw_gen={gen_text.strip()!r}\n")
            multi_results.append((group_label, hits, len(expected_vals), got_vals, expected_vals))

    # Summary
    results_sorted = sorted(results, key=lambda x: x[0])
    acc = sum(1 for _, _, ok, _, _ in results_sorted if ok) / max(1, len(results_sorted))
    print("\n=== Retention Curve Summary ===")
    print(f"accuracy={acc*100:.1f}% ({sum(1 for _,_,ok,_,_ in results_sorted if ok)}/{len(results_sorted)})")
    for depth, nid, ok, stats, decoy_hit in results_sorted:
        mark = "pass" if ok else "FAIL"
        print(f"{mark} depth={depth:.3f} id={nid} extracted={stats.get('extracted','')!r}{decoy_hit}")

    if args.decoys:
        n_decoy_subs = sum(1 for _, _, _, _, dh in results_sorted if dh)
        print(f"\nDecoy substitutions: {n_decoy_subs}/{len(results_sorted)} (recency bias signal)")

    if multi_results:
        total_hits = sum(h for _, h, _, _, _ in multi_results)
        total_possible = sum(t for _, _, t, _, _ in multi_results)
        print(f"\nMulti-needle recall: {total_hits}/{total_possible} ({100*total_hits/max(1,total_possible):.1f}%)")
        for label, hits, total, got, exp in multi_results:
            mark = "pass" if hits == total else "PARTIAL" if hits > 0 else "FAIL"
            print(f"  {mark} {hits}/{total} {label}")

    print("\nTip: If accuracy collapses primarily for early depths under keep_last/uniform_budget, that's eviction.")
    print("     If accuracy is bad even with evict_policy=none, it's not eviction (quant/RoPE/attention spikes/etc).")
    if args.decoys:
        print("     DECOY_SUBSTITUTION = model returned a nearby decoy's value -> recency bias confirmed.")


if __name__ == "__main__":
    main()
