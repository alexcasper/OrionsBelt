#!/usr/bin/env python3
"""Per-layer latency profiling for Qwen3.5 hybrid GDN model.

Registers forward pre/post hooks on each decoder layer to measure wall-clock
time per layer, broken down by layer type (linear_attention / full_attention).

Produces a CSV with columns:
  phase, ctx_len, layer_idx, layer_type, p50_us, p95_us, mean_us, n_samples

Usage:
  ORIONS_FORCE_FP32=1 /tmp/model_venv/bin/python3 bench/profile_layers.py \
    --model Qwen3.5-0.8B --contexts 32,64,128 --repeats 3 --decode-tokens 3
"""
import argparse
import csv
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("ORIONS_FORCE_FP32", "1")

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = REPO / "weights"


def load_model(model_label: str):
    """Load model + tokenizer, return (model, tokenizer, cfg, layer_types)."""
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = WEIGHTS / f"Qwen--{model_label}"
    with open(model_path / "config.json") as f:
        raw_cfg = json.load(f)
    cfg = raw_cfg.get("text_config", raw_cfg)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    layer_types = cfg.get("layer_types", [])
    return model, tokenizer, cfg, layer_types


def tokenize_to_length(tokenizer, max_tokens):
    """Produce exactly max_tokens tokens using diverse padding text."""
    pad_text = "The quick brown fox jumps over the lazy dog. "
    ids = tokenizer.encode(pad_text, add_special_tokens=True)
    pad_ids = tokenizer.encode(pad_text, add_special_tokens=False)
    while len(ids) < max_tokens:
        ids.extend(pad_ids)
    return ids[:max_tokens]


def run_profiling(model, tokenizer, layer_types, contexts, repeats, decode_tokens):
    """Run prefill + decode profiling with per-layer hooks.

    Returns {(layer_idx, phase, ctx): [elapsed_us, ...]},
    full_attn_layer_set, linear_layer_set.
    """
    import torch

    all_times = defaultdict(list)
    start_times = {}
    current = {"phase": "prefill", "ctx": 0}

    layers = model.model.layers
    handles = []

    for idx, layer in enumerate(layers):
        def make_pre(i):
            def pre(module, inp):
                start_times[i] = time.perf_counter()
            return pre

        def make_post(i):
            def post(module, inp, out):
                if i in start_times:
                    elapsed = (time.perf_counter() - start_times[i]) * 1e6
                    all_times[(i, current["phase"], current["ctx"])].append(elapsed)
            return post

        handles.append(layer.register_forward_pre_hook(make_pre(idx)))
        handles.append(layer.register_forward_hook(make_post(idx)))

    full_attn = {i for i, t in enumerate(layer_types) if t == "full_attention"}
    linear = {i for i, t in enumerate(layer_types) if t == "linear_attention"}

    for ctx in contexts:
        for rep in range(repeats):
            input_ids = tokenize_to_length(tokenizer, ctx)
            input_tensor = torch.tensor([input_ids], dtype=torch.long)

            current["phase"] = "prefill"
            current["ctx"] = ctx
            with torch.no_grad():
                _ = model(input_tensor)

            current["phase"] = "decode"
            with torch.no_grad():
                past = None
                cur_ids = input_tensor
                for _ in range(decode_tokens):
                    out = model(cur_ids, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    cur_ids = out.logits[:, -1:, :].argmax(dim=-1)

        print(f"  ctx={ctx} done ({repeats} repeats, {decode_tokens} decode tokens)", flush=True)

    for h in handles:
        h.remove()

    return all_times, full_attn, linear


def write_csv(all_times, full_attn, linear, output_path):
    """Write per-layer timing CSV and print summary."""
    rows = []
    for (idx, phase, ctx), samples in sorted(all_times.items()):
        if not samples:
            continue
        p50 = statistics.median(samples)
        p95 = max(samples) if len(samples) < 20 else sorted(samples)[int(len(samples) * 0.95)]
        mean = statistics.mean(samples)
        ltype = "full_attention" if idx in full_attn else "linear_attention"
        rows.append({
            "phase": phase, "ctx_len": ctx, "layer_idx": idx,
            "layer_type": ltype,
            "p50_us": f"{p50:.1f}", "p95_us": f"{p95:.1f}",
            "mean_us": f"{mean:.1f}", "n_samples": len(samples),
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "phase", "ctx_len", "layer_idx", "layer_type",
            "p50_us", "p95_us", "mean_us", "n_samples",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {output_path}")

    # Summary by layer type and phase
    print("\n=== Summary (p50 µs, aggregated by layer type) ===")
    print(f"{'Phase':10s} {'Ctx':>5s} {'Layer Type':18s} "
          f"{'p50 (total)':>12s} {'p50 (avg/layer)':>15s} {'Layers':>7s}")
    print("-" * 75)
    for phase in ["prefill", "decode"]:
        for ctx in sorted({k[2] for k in all_times}):
            for name, lset in [("full_attention", full_attn), ("linear_attention", linear)]:
                totals = [statistics.median(all_times[(i, phase, ctx)])
                          for i in sorted(lset)
                          if (i, phase, ctx) in all_times and all_times[(i, phase, ctx)]]
                if totals:
                    total = sum(totals)
                    print(f"{phase:10s} {ctx:5d} {name:18s} "
                          f"{total:12.1f} {total / len(totals):15.1f} {len(totals):7d}")


def main():
    parser = argparse.ArgumentParser(description="Per-layer latency profiling")
    parser.add_argument("--model", default="Qwen3.5-0.8B")
    parser.add_argument("--contexts", default="32,64,128")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    contexts = [int(c) for c in args.contexts.split(",")]

    print(f"Loading {args.model}...", flush=True)
    model, tokenizer, cfg, layer_types = load_model(args.model)
    print(f"  {cfg['num_hidden_layers']} layers, "
          f"{sum(1 for t in layer_types if t == 'full_attention')} full-attn, "
          f"{sum(1 for t in layer_types if t == 'linear_attention')} linear-attn", flush=True)

    print(f"\nProfiling: contexts={contexts}, repeats={args.repeats}, "
          f"decode_tokens={args.decode_tokens}", flush=True)

    all_times, full_attn, linear = run_profiling(
        model, tokenizer, layer_types, contexts, args.repeats, args.decode_tokens)

    if args.output is None:
        args.output = str(REPO / "results" / "raw" / "rk3588-t4_layer_profile.csv")
    write_csv(all_times, full_attn, linear, args.output)


if __name__ == "__main__":
    main()
