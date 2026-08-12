#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / AAIF
# SPDX-License-Identifier: Apache-2.0
"""RULER multi-key retrieval evaluation for GDN-1 vs GDN-2 (bead ob-zak).

Tests whether the model can retrieve key-value pairs from a multi-key context,
using log-likelihood scoring (no autoregressive generation needed).

For each prompt with N key-value pairs:
  1. One key is queried: "What is the value of X?"
  2. Score each candidate answer (correct + distractors) by computing
     log P(answer | prompt) via teacher-forced forward pass.
  3. Accuracy = correct answer has highest log-likelihood among candidates.

Usage:
    # GDN-1 baseline only
    python3 bench/gdn2_ruler.py --model models/Qwen3.5-0.8B

    # GDN-2 (swap layer 0 + 30-step adaptation, then evaluate)
    python3 bench/gdn2_ruler.py --model models/Qwen3.5-0.8B --gdn2 --steps 30

    # Both models, write results
    python3 bench/gdn2_ruler.py --model models/Qwen3.5-0.8B --gdn2 --csv
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

# Import from gdn2_swap.py
sys.path.insert(0, os.path.dirname(__file__))
from gdn2_swap import swap_gdn1_to_gdn2  # noqa: E402


def generate_prompts(num_prompts, context_length, num_keys, seed_base=100):
    """Generate multi-key retrieval prompts."""
    # Add repo root to path for corpus import
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import bench.corpus as corpus

    prompts = []
    for i in range(num_prompts):
        item = corpus.generate_niah_multikey(
            context_length=context_length,
            num_keys=num_keys,
            seed=seed_base + i,
        )
        # Extract all key-value pairs from the prompt
        pairs = {}
        for m in re.finditer(r'value of "([^"]+)" is "([^"]+)"', item.prompt):
            pairs[m.group(1)] = m.group(2)

        # Identify correct answer and distractors
        query_key = item.metadata["query_key"]
        correct = item.metadata["query_value"]
        distractors = [v for k, v in pairs.items() if k != query_key]

        prompts.append(
            {
                "prompt": item.prompt,
                "query_key": query_key,
                "correct_answer": correct,
                "distractor_answers": distractors,
                "est_tokens": item.metadata["est_tokens"],
                "num_keys": item.metadata["num_keys"],
                "needle_depth": item.needle_depth,
                "seed": seed_base + i,
            }
        )
    return prompts


def score_answer_logprob(model, tokenizer, prompt, answer):
    """Compute log P(answer | prompt) via teacher forcing.

    Tokenizes prompt + ' ' + answer, forward pass, extracts per-token
    log-probabilities at answer positions.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    answer_ids = tokenizer(" " + answer, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]
    answer_len = answer_ids.shape[1]

    # Concatenate
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)

    with torch.no_grad():
        logits = model(full_ids).logits  # [1, seq_len, vocab_size]

    # Logits at position i predict token i+1
    # Answer tokens are at positions [prompt_len, prompt_len+1, ..., prompt_len+answer_len-1]
    # They are predicted by logits at positions [prompt_len-1, ..., prompt_len+answer_len-2]
    answer_logits = logits[
        0, prompt_len - 1 : prompt_len + answer_len - 1, :
    ]  # [answer_len, vocab]
    log_probs = F.log_softmax(answer_logits.float(), dim=-1)  # [answer_len, vocab]

    # Gather log-probs for actual answer tokens
    answer_token_ids = answer_ids[0]  # [answer_len]
    token_log_probs = log_probs.gather(1, answer_token_ids.unsqueeze(1)).squeeze(1)  # [answer_len]

    total_logprob = token_log_probs.sum().item()
    avg_logprob = total_logprob / answer_len
    return total_logprob, avg_logprob


def evaluate_retrieval(model, tokenizer, prompts, max_time_secs=None):
    """Run retrieval evaluation on a list of prompts.

    Returns accuracy and per-prompt details.
    """
    results = []
    correct_count = 0
    t0 = time.time()

    for i, p in enumerate(prompts):
        elapsed = time.time() - t0
        if max_time_secs and elapsed > max_time_secs:
            print(f"  Time budget exhausted after {i}/{len(prompts)} prompts", flush=True)
            break

        candidates = [p["correct_answer"]] + p["distractor_answers"]
        scores = []
        for cand in candidates:
            total_lp, avg_lp = score_answer_logprob(model, tokenizer, p["prompt"], cand)
            scores.append(
                {
                    "answer": cand,
                    "total_logprob": total_lp,
                    "avg_logprob": avg_lp,
                    "is_correct": cand == p["correct_answer"],
                }
            )

        # Sort by total log-prob (higher = better)
        scores.sort(key=lambda x: x["total_logprob"], reverse=True)
        hit = scores[0]["is_correct"]
        correct_count += int(hit)

        # Log-prob gap: correct answer's rank and margin
        correct_score = next(s["total_logprob"] for s in scores if s["is_correct"])
        best_score = scores[0]["total_logprob"]
        margin = correct_score - best_score  # 0 if correct is best, negative otherwise

        prompt_elapsed = time.time() - t0
        print(
            f"  Prompt {i + 1}/{len(prompts)}: {'✓' if hit else '✗'} "
            f"query={p['query_key'][:20]:20s} "
            f"correct_lp={correct_score:.2f} margin={margin:+.2f} "
            f"({prompt_elapsed:.0f}s elapsed)",
            flush=True,
        )

        results.append(
            {
                "seed": p["seed"],
                "query_key": p["query_key"],
                "correct_answer": p["correct_answer"],
                "hit": hit,
                "correct_logprob": correct_score,
                "margin": margin,
                "num_candidates": len(candidates),
                "scores": scores,
            }
        )

    accuracy = correct_count / len(results) if results else 0.0
    total_time = time.time() - t0
    return {
        "accuracy": accuracy,
        "num_prompts": len(results),
        "num_correct": correct_count,
        "total_time_s": total_time,
        "details": results,
    }


def capture_manifest():
    """Capture provenance metadata matching the canonical manifest schema.

    Uses the nested ``git: {sha, dirty}`` structure expected by
    ``bench/manifest.py`` and ``tests/test_manifest_sha_provenance.py``,
    so every RULER manifest passes the same provenance checks as all
    other committed manifests.
    """
    import platform

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        _status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        # Exclude results/ and .beads/ — output data, not source changes.
        import re

        _OUTPUT_RE = re.compile(r"^[ ?][M?] (results/|\.beads/)")
        dirty = any(not _OUTPUT_RE.match(line) for line in _status.splitlines() if line)
    except Exception:
        sha, dirty = "unknown", False
    return {
        "manifest_version": 1,
        "git": {
            "sha": sha,
            "dirty": dirty,
        },
        "host": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
        },
        "software": {
            "python_version": platform.python_version(),
            "torch": torch.__version__,
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main():
    parser = argparse.ArgumentParser(description="RULER retrieval eval (ob-zak)")
    parser.add_argument("--model", default="models/Qwen3.5-0.8B")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--num-keys", type=int, default=5)
    parser.add_argument(
        "--gdn2", action="store_true", help="Swap layer 0 to GDN-2 + adapt before eval"
    )
    parser.add_argument("--layers", default="0", help="Layers to swap (GDN-2 mode)")
    parser.add_argument("--steps", type=int, default=30, help="Adaptation steps (GDN-2 mode)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--csv", action="store_true")
    parser.add_argument(
        "--max-time",
        type=int,
        default=2400,
        help="Max evaluation time in seconds (default 2400; increase for slow devices)",
    )
    args = parser.parse_args()

    layer_indices = [int(x) for x in args.layers.split(",")]
    tag = "gdn2" if args.gdn2 else "gdn1"

    print(f"=== RULER Multi-Key Retrieval ({tag}) ===", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(
        f"Prompts: {args.num_prompts}, ctx: {args.context_length}, keys: {args.num_keys}",
        flush=True,
    )
    if args.gdn2:
        print(f"GDN-2 swap: layers {layer_indices}, {args.steps} steps, lr={args.lr}", flush=True)

    # ── Generate prompts ──
    print("\n--- Generating prompts ---", flush=True)
    prompts = generate_prompts(args.num_prompts, args.context_length, args.num_keys)

    print(
        f"Generated {len(prompts)} prompts "
        f"(~{prompts[0]['est_tokens']} tokens each, {prompts[0]['num_keys']} keys)",
        flush=True,
    )

    # ── Load model ──
    print("\n--- Loading model ---", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()

    # ── GDN-2 swap + adaptation ──
    if args.gdn2:
        print("\n--- GDN-2 swap + isolated adaptation ---", flush=True)

        # Capture GDN-1 reference I/O for isolated training
        from gdn2_swap import ADAPTATION_TEXT

        adapt_ids = tokenizer(ADAPTATION_TEXT, return_tensors="pt", truncation=True, max_length=64)[
            "input_ids"
        ]
        captured = {}

        def make_pre_hook(idx):
            def pre_fn(mod, fn_args, fn_kwargs):
                hs = fn_args[0] if fn_args else fn_kwargs.get("hidden_states")
                captured.setdefault(idx, {})["input"] = hs.detach().clone()

            return pre_fn

        def make_post_hook(idx):
            def post_fn(mod, fn_args, fn_kwargs, out):
                captured.setdefault(idx, {})["output"] = out.detach().clone()

            return post_fn

        handles = []
        for idx in layer_indices:
            handles.append(
                model.model.layers[idx].linear_attn.register_forward_pre_hook(
                    make_pre_hook(idx), with_kwargs=True
                )
            )
            handles.append(
                model.model.layers[idx].linear_attn.register_forward_hook(
                    make_post_hook(idx), with_kwargs=True
                )
            )

        with torch.no_grad():
            model(input_ids=adapt_ids, use_cache=False)
        for h in handles:
            h.remove()

        # Swap
        swap_gdn1_to_gdn2(model, layer_indices)

        # Freeze all, unfreeze new gate params
        for p in model.parameters():
            p.requires_grad = False
        trainable = []
        for idx in layer_indices:
            mod = model.model.layers[idx].linear_attn
            for pname in ("in_proj_erase_gate", "in_proj_write_gate"):
                for p in getattr(mod, pname).parameters():
                    p.requires_grad = True
                    trainable.append(p)

        optimizer = torch.optim.AdamW(trainable, lr=args.lr)
        model.train()

        t0 = time.time()
        for step in range(args.steps):
            optimizer.zero_grad()
            for idx in layer_indices:
                mod = model.model.layers[idx].linear_attn
                out = mod(captured[idx]["input"])
                loss = F.mse_loss(out.float(), captured[idx]["output"].float())
                loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            if step == 0 or (step + 1) % 10 == 0:
                print(
                    f"  Adapt step {step + 1}/{args.steps}: "
                    f"mse={loss.item():.6f} ({time.time() - t0:.1f}s)",
                    flush=True,
                )

        model.eval()
        print(f"Adaptation complete ({time.time() - t0:.1f}s)", flush=True)

    # ── Run retrieval evaluation ──
    print(f"\n--- Retrieval evaluation ({tag}) ---", flush=True)
    results = evaluate_retrieval(model, tokenizer, prompts, max_time_secs=args.max_time)

    print(f"\n=== {tag.upper()} Results ===", flush=True)
    print(
        f"Accuracy: {results['accuracy']:.1%} ({results['num_correct']}/{results['num_prompts']})",
        flush=True,
    )
    print(f"Total time: {results['total_time_s']:.0f}s", flush=True)

    # ── Output results JSON ──
    manifest = capture_manifest()
    output = {
        "experiment": "ruler_multikey_retrieval",
        "bead": "ob-zak",
        "model": args.model,
        "variant": tag,
        "context_length": args.context_length,
        "num_keys": args.num_keys,
        "num_prompts": args.num_prompts,
        "accuracy": results["accuracy"],
        "num_correct": results["num_correct"],
        "total_time_s": results["total_time_s"],
        "gdn2_config": {
            "layers": layer_indices,
            "steps": args.steps,
            "lr": args.lr,
        }
        if args.gdn2
        else None,
        "details": results["details"],
        "manifest": manifest,
    }

    print("\n=== RESULTS JSON ===", flush=True)
    print(json.dumps(output, indent=2), flush=True)

    # ── Write CSV ──
    if args.csv:
        csv_dir = "results/raw"
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"ruler_{tag}_t3.csv")
        with open(csv_path, "w") as f:
            f.write("prompt_idx,seed,query_key,correct_answer,hit,correct_logprob,margin\n")
            for i, d in enumerate(results["details"]):
                f.write(
                    f"{i + 1},{d['seed']},{d['query_key']},"
                    f"{d['correct_answer']},{int(d['hit'])},"
                    f"{d['correct_logprob']:.4f},{d['margin']:.4f}\n"
                )
        print(f"\nCSV: {csv_path}", flush=True)

        manifest_dir = "results/manifests"
        os.makedirs(manifest_dir, exist_ok=True)
        manifest_path = os.path.join(manifest_dir, f"ruler_{tag}_t3.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
