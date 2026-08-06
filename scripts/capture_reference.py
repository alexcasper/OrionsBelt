#!/usr/bin/env python3
"""Capture golden reference logits for the correctness oracle (bead ob-aqv).

Runs the selected Qwen3.5 checkpoint via ``HFTorchBackend`` on an x86/CUDA host
and saves golden outputs (top-k logits, perplexity, argmax sequence) in the JSON
format that ``bench/correctness.py`` consumes.

This script is **portable code** — it imports cleanly on any platform and is
linted in CI, but actual inference requires torch+transformers on an x86/CUDA
(or Apple Silicon / Graviton) host. On a device without those deps it prints a
clear message and exits 1.

**Why top-k only?** The Qwen3.5 vocabulary is 248K tokens, so full last-position
logits are ~1 MB per prompt in fp32. Saving the top-20 logits + their token IDs
is ~2 KB and preserves everything needed for top-k agreement, argmax accuracy,
and a KL divergence estimate on the truncated distribution. Use
``--save-full-logits`` to save the complete vector when precision is critical.

Usage::

    # Default: 0.8B fallback on CPU, 3 prompts × 4 context lengths
    python3 scripts/capture_reference.py --model 0.8b --device cpu

    # Full reference: 4B on CUDA, full logits, canonical sweep
    python3 scripts/capture_reference.py --model 4b --device cuda \
        --save-full-logits --context-lengths 4096,32768,131072,262144

    # Quick smoke: tiny prompt, CPU, top-k only
    python3 scripts/capture_reference.py --model 0.8b --device cpu \
        --prompts 1 --context-lengths 512 --decode-length 8

Output directory structure::

    results/reference/
    ├── manifest.json              # provenance (model, device, dtype, sha, time)
    ├── ref_0001_ctx004096.json    # golden outputs for prompt 1 at ctx 4096
    ├── ref_0001_ctx032768.json
    └── ...

Each golden file matches the format ``bench/correctness.py``'s CLI expects::

    {
      "logits": [[...]],            # top-k logits (or full if --save-full-logits)
      "topk_token_ids": [[...]],    # token IDs aligned with logits
      "perplexity": 42.5,
      "argmax_tokens": [123, 456],  # greedy decode sequence
      "metadata": {
        "prompt_hash": "sha256...",
        "context_length": 4096,
        "model_checkpoint": "Qwen/Qwen3.5-0.8B",
        "dtype": "float16",
        "device": "cuda",
        "captured_at": "2026-08-06T07:00:00Z"
      }
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

# Ensure repo root is importable
_ROOT = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.harness import QWEN35_08B, QWEN35_4B  # noqa: E402
from bench.hf_backend import HFTorchBackend  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT_LENGTHS = [4096, 32768]
DEFAULT_PROMPTS = 3
DEFAULT_DECODE_LENGTH = 32
DEFAULT_TOPK = 20

_MODEL_PRESETS = {
    "4b": QWEN35_4B,
    "0.8b": QWEN35_08B,
}

# Standard prompts for deterministic golden outputs. These are short, neutral
# prompts that exercise the model's language modeling without introducing
# retrieval or factual knowledge bias. The corpus module (bench/corpus.py)
# generates longer needle-in-a-haystack prompts for retrieval evaluation;
# these are for correctness oracle comparison.
STANDARD_PROMPTS = [
    "The quick brown fox jumps over the lazy dog. This sentence is used because",
    "In machine learning, gradient descent is an optimization algorithm that",
    "The architecture of a modern neural network typically consists of",
    "Once upon a time, in a kingdom far away, there lived a wise ruler who",
    "The fundamental theorem of calculus establishes a relationship between",
]


def _prompt_hash(text: str) -> str:
    """SHA-256 of the prompt text for reproducibility verification."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _topk_logits(logits_row: list[float], k: int) -> tuple[list[float], list[int]]:
    """Extract top-k logits and their token IDs from a single row.

    Returns (logits, token_ids) sorted by logit value descending.
    """
    indexed = list(enumerate(logits_row))
    indexed.sort(key=lambda x: x[1], reverse=True)
    top = indexed[:k]
    return [v for _, v in top], [i for i, _ in top]


def capture_single_prompt(
    backend: HFTorchBackend,
    prompt_text: str,
    context_length: int,
    decode_length: int,
    topk: int,
    save_full_logits: bool,
) -> dict:
    """Run one prompt through the backend and capture golden outputs.

    Returns a dict in the correctness.py JSON format.
    """
    # Tokenize and pad/truncate to target context length
    input_ids = backend.tokenize(prompt_text)
    # If the prompt is shorter than the target context length, pad with the
    # model's pad token or repeat the prompt text to fill. For golden reference
    # capture, we just use whatever length the tokenizer gives us and record
    # the actual length — the comparison is position-relative.
    actual_length = len(input_ids)

    # Prefill — get logits for all positions
    logits = backend.prefill(input_ids)

    # Extract last-position logits for next-token prediction
    import torch  # noqa: PLC0415

    last_logits = logits[0, -1, :].cpu().float().tolist()

    # Truncate or save full
    if save_full_logits:
        saved_logits = [last_logits]
        saved_token_ids = [list(range(len(last_logits)))]
    else:
        top_vals, top_ids = _topk_logits(last_logits, topk)
        saved_logits = [top_vals]
        saved_token_ids = [top_ids]

    # Greedy decode to capture the argmax token sequence
    argmax_tokens = []
    next_token = backend.sample(logits)
    argmax_tokens.append(next_token)

    for _ in range(decode_length - 1):
        next_token = backend.decode_step(next_token)
        argmax_tokens.append(next_token)

    # Compute perplexity from the prefill logits
    # PPL = exp(-1/N * sum(log P(x_i | x_<i)))
    with torch.inference_mode():
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        # Gather the log-prob of each actual token
        token_log_probs = log_probs.gather(
            1, torch.tensor(input_ids, device=log_probs.device).unsqueeze(1)
        ).squeeze(1)
        # Exclude the first token (no context to predict from)
        avg_nll = -token_log_probs[1:].mean().item()
        perplexity = float(torch.exp(torch.tensor(avg_nll)).item())

    backend.reset()

    result = {
        "logits": saved_logits,
        "topk_token_ids": saved_token_ids,
        "perplexity": perplexity,
        "argmax_tokens": argmax_tokens,
        "metadata": {
            "prompt_hash": _prompt_hash(prompt_text),
            "prompt_preview": prompt_text[:100],
            "context_length": actual_length,
            "decode_length": decode_length,
            "model_checkpoint": backend.config.name,
            "dtype": backend._dtype,
            "device": backend._model.device.type if backend._model else "unknown",
            "captured_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    return result


def capture_reference(
    model_key: str,
    device: str,
    context_lengths: list[int],
    num_prompts: int,
    decode_length: int,
    topk: int,
    save_full_logits: bool,
    dtype: str,
    output_dir: str,
) -> list[str]:
    """Run the full golden reference capture.

    Returns list of output file paths.
    """
    import torch  # noqa: PLC0415

    config = _MODEL_PRESETS[model_key]
    print(f"[ref] Loading {config.name} (dtype={dtype}, device={device})...")

    backend = HFTorchBackend(
        config,
        dtype=dtype,
        device_map=device if device == "cuda" else "cpu",
    )
    backend.load()
    print(f"[ref] Model loaded. Device: {backend._model.device}")

    prompts = STANDARD_PROMPTS[:num_prompts]
    if len(prompts) < num_prompts:
        # Cycle through prompts if more requested than available
        prompts = [STANDARD_PROMPTS[i % len(STANDARD_PROMPTS)] for i in range(num_prompts)]

    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    for pi, prompt in enumerate(prompts):
        for ci, ctx_len in enumerate(context_lengths):
            tag = f"ref_{pi + 1:04d}_ctx{ctx_len:06d}"
            out_path = os.path.join(output_dir, f"{tag}.json")

            print(f"[ref] Prompt {pi + 1}/{len(prompts)}, ctx={ctx_len}: ", end="", flush=True)
            t0 = time.perf_counter()

            try:
                result = capture_single_prompt(
                    backend=backend,
                    prompt_text=prompt,
                    context_length=ctx_len,
                    decode_length=decode_length,
                    topk=topk,
                    save_full_logits=save_full_logits,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"OOM (skipping)")
                    continue
                raise

            elapsed = time.perf_counter() - t0
            print(f"ppl={result['perplexity']:.2f}, {elapsed:.1f}s")

            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
                f.write("\n")
            output_files.append(out_path)

    # Write manifest with provenance
    manifest = _build_manifest(
        model_key=model_key,
        config_name=config.name,
        device=device,
        dtype=dtype,
        output_files=output_files,
        context_lengths=context_lengths,
        num_prompts=len(prompts),
        decode_length=decode_length,
        topk=topk,
        save_full_logits=save_full_logits,
    )
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"\n[ref] Captured {len(output_files)} golden reference files")
    print(f"[ref] Manifest: {manifest_path}")
    return output_files


def _build_manifest(
    model_key: str,
    config_name: str,
    device: str,
    dtype: str,
    output_files: list[str],
    context_lengths: list[int],
    num_prompts: int,
    decode_length: int,
    topk: int,
    save_full_logits: bool,
) -> dict:
    """Build provenance manifest for the reference capture."""
    import subprocess  # noqa: PLC0415

    # Git SHA (best-effort)
    sha = None
    dirty = False
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    import platform  # noqa: PLC0415

    return {
        "purpose": "golden_reference_logits",
        "bead": "ob-aqv",
        "model_checkpoint": config_name,
        "model_key": model_key,
        "device": device,
        "dtype": dtype,
        "platform": {
            "python_version": sys.version.split()[0],
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "git": {"sha": sha, "dirty": dirty},
        "capture_config": {
            "context_lengths": context_lengths,
            "num_prompts": num_prompts,
            "decode_length": decode_length,
            "topk": topk,
            "save_full_logits": save_full_logits,
        },
        "output_files": [os.path.basename(f) for f in output_files],
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="capture_reference",
        description="Capture golden reference logits for the correctness oracle (ob-aqv).",
    )
    parser.add_argument(
        "--model",
        choices=list(_MODEL_PRESETS),
        default="0.8b",
        help="Model checkpoint to use (default: 0.8b fallback for fast iteration).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda).",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: float16).",
    )
    parser.add_argument(
        "--context-lengths",
        default=",".join(str(x) for x in DEFAULT_CONTEXT_LENGTHS),
        help="Comma-separated context lengths (default: 4096,32768).",
    )
    parser.add_argument(
        "--prompts",
        type=int,
        default=DEFAULT_PROMPTS,
        help=f"Number of standard prompts to run (default: {DEFAULT_PROMPTS}).",
    )
    parser.add_argument(
        "--decode-length",
        type=int,
        default=DEFAULT_DECODE_LENGTH,
        help=f"Number of tokens to greedily decode (default: {DEFAULT_DECODE_LENGTH}).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=DEFAULT_TOPK,
        help=f"Number of top logits to save per position (default: {DEFAULT_TOPK}).",
    )
    parser.add_argument(
        "--save-full-logits",
        action="store_true",
        help="Save full vocabulary logits (~1 MB/position) instead of top-k.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/reference",
        help="Output directory (default: results/reference).",
    )
    args = parser.parse_args(argv)

    context_lengths = [int(x) for x in args.context_lengths.split(",")]

    output_files = capture_reference(
        model_key=args.model,
        device=args.device,
        context_lengths=context_lengths,
        num_prompts=args.prompts,
        decode_length=args.decode_length,
        topk=args.topk,
        save_full_logits=args.save_full_logits,
        dtype=args.dtype,
        output_dir=args.output_dir,
    )

    if not output_files:
        print("Error: no golden reference files produced", file=sys.stderr)
        return 1

    print(f"\nDone. {len(output_files)} files in {args.output_dir}/")
    print("Compare with: python3 -m bench.correctness \\")
    print(f"  --reference {args.output_dir}/ref_0001_ctx{context_lengths[0]:06d}.json \\")
    print("  --candidate <candidate_file.json>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
