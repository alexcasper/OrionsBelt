#!/usr/bin/env python3
"""Per-layer latency profiling for Qwen3.5 hybrid GDN model.

Registers forward hooks on each decoder layer to measure wall-clock time
per layer, broken down by layer type (linear_attention / full_attention).

Produces a CSV with columns:
  phase, ctx_len, layer_idx, layer_type, p50_us, p95_us, n_samples

Usage:
  ORIONS_FORCE_FP32=1 /tmp/model_venv/bin/python3 bench/profile_layers.py \
    --model Qwen3.5-0.8B --contexts 32,64,128 --repeats 3 --decode-tokens 3
"""
import argparse
import csv
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

# Force fp32 before importing torch
os.environ.setdefault("ORIONS_FORCE_FP32", "1")

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = REPO / "weights"


def load_model(model_label: str):
    """Load model + tokenizer, return (model, tokenizer, cfg)."""
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = WEIGHTS / f"Qwen--{model_label}"
    cfg_path = model_path / "config.json"
    with open(cfg_path) as f:
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

    # Determine layer types
    layer_types = cfg.get("layer_types", [])

    return model, tokenizer, cfg, layer_types


class LayerTimer:
    """Collect per-layer wall-clock timings via forward hooks."""

    def __init__(self, model, layer_types):
        self.times = defaultdict(list)  # (layer_idx, phase) -> [elapsed_us, ...]
        self.layer_types = layer_types
        self.phase = "prefill"
        self.handles = []
        self._install(model)

    def _install(self, model):
        """Hook every decoder layer."""
        layers = model.model.layers
        for idx, layer in enumerate(layers):
            h = layer.register_forward_hook(self._make_hook(idx))
            self.handles.append(h)

    def _make_hook(self, idx):
        def hook(module, inp, out):
            t = time.perf_counter()
            # Store start time — we'll measure on the NEXT hook call
            # Actually, we need a pre-hook for start and post-hook for end.
            # Simpler: measure total forward time of each layer.
            # We'll use pre+post hooks.
            pass
        return hook

    def _install_v2(self, model):
        """Use pre + post hooks for accurate per-layer timing."""
        self.handles.clear()
        layers = model.model.layers
        self._start_times = {}

        for idx, layer in enumerate(layers):
            def make_pre(i):
                def pre(module, inp):
                    self._start_times[i] = time.perf_counter()
                return pre

            def make_post(i):
                def post(module, inp, out):
                    if i in self._start_times:
                        elapsed = (time.perf_counter() - self._start_times[i]) * 1e6
                        self.times[(i, self.phase)].append(elapsed)
                return post

            self.handles.append(layer.register_forward_pre_hook(make_pre(idx)))
            self.handles.append(layer.register_forward_hook(make_post(idx)))

    def set_phase(self, phase):
        self.phase = phase

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def tokenize_to_length(tokenizer, max_tokens):
    """Produce exactly max_tokens tokens using diverse padding text."""
    pad_text = "The quick brown fox jumps over the lazy dog. "
    ids = tokenizer.encode("The quick brown fox jumps over the lazy dog. ", add_special_tokens=True)
    pad_ids = tokenizer.encode(pad_text, add_special_tokens=False)
    while len(ids) < max_tokens:
        ids.extend(pad_ids)
    return ids[:max_tokens]


def run_profiling(model, tokenizer, cfg, layer_types, contexts, repeats, decode_tokens):
    import torch

    timer = LayerTimer(model, layer_types)
    timer._install_v2(model)

    full_attn_layers = {i for i, t in enumerate(layer_types) if t == "full_attention"}
    linear_layers = {i for i, t in enumerate(layer_types) if t == "linear_attention"}

    for ctx in contexts:
        for rep in range(repeats):
            input_ids = tokenize_to_length(tokenizer, ctx)
            input_tensor = torch.tensor([input_ids], dtype=torch.long)

            # Prefill phase
            timer.set_phase("prefill")
            with torch.no_grad():
                _ = model(input_tensor)

            # Decode phase: generate one token at a time
            timer.set_phase("decode")
            with torch.no_grad():
                past = None
                cur_ids = input_tensor
                for _ in range(decode_tokens):
                    out = model(cur_ids, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    next_token = out.logits[:, -1:, :].argmax(dim=-1)
                    cur_ids = next_token

            if rep == 0:
                print(f"  ctx={ctx} rep {rep+1}/{repeats} done", flush=True)

    timer.remove()
    return timer.times, full_attn_layers, linear_layers


def summarize_and_write(times, full_attn_layers, linear_layers, output_path, contexts, repeats):
    rows = []
    # Per-layer rows
    for (idx, phase), samples in sorted(times.items()):
        if not samples:
            continue
        p50 = statistics.median(samples)
        if len(samples) >= 20:
            sorted_s = sorted(samples)
            p95 = sorted_s[int(len(sorted_s) * 0.95)]
        else:
            p95 = max(samples)
        ltype = "full_attention" if idx in full_attn_layers else "linear_attention"
        # Average across all repeats at this ctx — we stored per-iteration so group by ctx
        # Actually we need to know which ctx this came from. Let me restructure.
        rows.append({
            "layer_idx": idx,
            "layer_type": ltype,
            "phase": phase,
            "p50_us": f"{p50:.1f}",
            "p95_us": f"{p95:.1f}",
            "n_samples": len(samples),
        })

    # We need to track ctx per timing. Let me restructure the data format.
    pass


def run_profiling_v2(model, tokenizer, cfg, layer_types, contexts, repeats, decode_tokens):
    """Profiling that tracks ctx_len per timing."""
    import torch

    # Collect: {(layer_idx, phase, ctx): [elapsed_us, ...]}
    all_times = defaultdict(list)
    start_times = {}

    layers = model.model.layers
    handles = []
    current = {"phase": "prefill", "ctx": 0}

    for idx, layer in enumerate(layers):
        def make_pre(i):
            def pre(module, inp):
                start_times[i] = time.perf_counter()
            return pre

        def make_post(i):
            def post(module, inp, out):
                if i in start_times:
                    elapsed = (time.perf_counter() - start_times[i]) * 1e6
                    key = (i, current["phase"], current["ctx"])
                    all_times[key].append(elapsed)
            return post

        handles.append(layer.register_forward_pre_hook(make_pre(idx)))
        handles.append(layer.register_forward_hook(make_post(idx)))

    full_attn_layers = {i for i, t in enumerate(layer_types) if t == "full_attention"}
    linear_layers = {i for i, t in enumerate(layer_types) if t == "linear_attention"}

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
                    next_token = out.logits[:, -1:, :].argmax(dim=-1)
                    cur_ids = next_token

        print(f"  ctx={ctx} done ({repeats} repeats, {decode_tokens} decode tokens)", flush=True)

    for h in handles:
        h.remove()

    return all_times, full_attn_layers, linear_layers


def write_csv(all_times, full_attn_layers, linear_layers, output_path):
    """Write per-layer timing CSV."""
    rows = []
    for (idx, phase, ctx), samples in sorted(all_times.items()):
        if not samples:
            continue
        p50 = statistics.median(samples)
        if len(samples) >= 20:
            sorted_s = sorted(samples)
            p95 = sorted_s[int(len(sorted_s) * 0.95)]
        else:
            p95 = max(samples)
        mean = statistics.mean(samples)
        ltype = "full_attention" if idx in full_attn_layers else "linear_attention"
        rows.append({
            "phase": phase,
            "ctx_len": ctx,
            "layer_idx": idx,
            "layer_type": ltype,
            "p50_us": f"{p50:.1f}",
            "p95_us": f"{p95:.1f}",
            "mean_us": f"{mean:.1f}",
            "n_samples": len(samples),
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "phase", "ctx_len", "layer_idx", "layer_type",
            "p50_us", "p95_us", "mean_us", "n_samples",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {output_path}")

    # Print summary by layer type and phase
    print("\n=== Summary (p50 µs, aggregated by layer type) ===")
    print(f"{'Phase':10s} {'Ctx':>5s} {'Layer Type':18s} {'p50 (total)':>12s} {'p50 (avg/layer)':>15s} {'Layers':>7s}")
    print("-" * 75)

    for phase in ["prefill", "decode"]:
        for ctx in sorted(set(k[2] for k in all_times)):
            for ltype_name, ltype_set in [("full_attention", full_attn_layers), ("linear_attention", linear_layers)]:
                layer_totals = []
                for idx in sorted(ltype_set):
                    key = (idx, phase, ctx)
                    if key in all_times and all_times[key]:
                        layer_totals.append(statistics.median(all_times[key]))
                if layer_totals:
                    total = sum(layer_totals)
                    avg = total / len(layer_totals)
                    print(f"{phase:10s} {ctx:5d} {ltype_name:18s} {total:12.1f} {avg:15.1f} {len(layer_totals):7d}")


def main():
    parser = argparse.ArgumentParser(description="Per-layer latency profiling")
    parser.add_argument("--model", default="Qwen3.5-0.8B")
    parser.add_argument("--contexts", default="32,64,128", help="Comma-separated context lengths")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=3)
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    contexts = [int(c) for c in args.contexts.split(",")]

    print(f"Loading {args.model}...", flush=True)
    model, tokenizer, cfg, layer_types = load_model(args.model)
    print(f"  {cfg['num_hidden_layers']} layers, "
          f"{sum(1 for t in layer_types if t == 'full_attention')} full-attn, "
          f"{sum(1 for t in layer_types if t == 'linear_attention')} linear-attn", flush=True)

    print(f"\nProfiling: contexts={contexts}, repeats={args.repeats}, "
          f"decode_tokens={args.decode_tokens}", flush=True)

    all_times, full_attn, linear_attn = run_profiling_v2(
        model, tokenizer, cfg, layer_types, contexts, args.repeats, args.decode_tokens
    )

    if args.output is None:
        args.output = str(REPO / "results" / "raw" / "rk3588-t4_layer_profile.csv")

    write_csv(all_times, full_attn, linear_attn, args.output)


if __name__ == "__main__":
    main()
