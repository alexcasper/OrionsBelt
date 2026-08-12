#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Generate golden reference outputs from a HuggingFace Qwen3.5 checkpoint.

This is the **correctness oracle source** (bead ``ob-aqv``).  It runs the
unmodified HuggingFace model to produce trusted logits, perplexity, and
generated tokens that every on-device optimisation is later validated against
(bead ``ob-3uh`` — ``bench/correctness.py`` consumes the output).

Design principles
-----------------
- **Numerically trustworthy.** Runs in float32 on CPU to avoid FP16/CUDA
  non-determinism.  The golden reference should be the most precise version;
  candidates (quantised, GPU, NPU) are compared *against* it with tolerances.
- **Provenance-complete.** Records git SHA, model checkpoint, torch/transformers
  versions, device, governor, and thermals so any consumer can verify
  reproducibility.
- **Compact output.** Full per-position logits would be GiB-scale, so we save
  the last-position full-vocab distribution plus top-k slices for a window of
  positions, and a scalar perplexity.
- **Real tokens only.** Context lengths are achieved by truncating a long
  canonical text — never padding — so every position carries meaningful signal
  and perplexity is trustworthy.
- **Degrades gracefully.** If torch/transformers are missing, exits with a
  clear message rather than crashing on import.

Usage::

    python3 scripts/generate_reference.py \\
        --model-path models/Qwen3.5-0.8B \\
        --output results/reference/qwen35-0.8b_reference.json

    # Short smoke run (single prompt, short context only):
    python3 scripts/generate_reference.py --model-path models/Qwen3.5-0.8B --smoke
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is importable
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _TORCH_AVAILABLE = True
    _IMPORT_ERROR = ""
except ImportError as e:  # pragma: no cover
    _TORCH_AVAILABLE = False
    _IMPORT_ERROR = str(e)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Canonical test texts
# ---------------------------------------------------------------------------
# Each entry is a seed prompt.  The tokenizer encodes the seed, and we prepend
# it to a long filler text so we can truncate to exact context lengths without
# padding.  This way every position in every run carries real signal.

# Long filler used to reach target context lengths.  Mix of factual prose,
# code, and structured text so the logit distributions are varied and
# exercise both GDN linear-attention layers and full-attention layers.
_LONG_FILLER = """\
The history of computing is a story of abstraction layers. At the bottom,
silicon wafers are etched with billions of transistors that switch on and off
billions of times per second. Above the transistors sit logic gates, then
adders and multipliers, then pipelined execution units, then instruction
schedulers, then operating systems, then compilers, then applications. Each
layer hides the complexity of the one below it.

In the early days of computing, programmers wrote machine code by hand. Every
instruction was a binary number that told the processor exactly what to do.
Assembly language was the first abstraction — short mnemonic codes like MOV,
ADD, and JMP replaced raw binary. Then came higher-level languages like
Fortran, COBOL, and C, which let programmers express algorithms in terms that
humans could read.

Modern processors are extraordinarily complex. A single Cortex-A76 core can
execute multiple instructions per clock cycle using superscalar out-of-order
execution. It has branch predictors, cache hierarchies, and vector units. The
Armv8.2-A architecture adds dot-product instructions that multiply and
accumulate pairs of 8-bit integers, which is exactly the operation needed for
quantized neural network inference.

Memory bandwidth is often the real bottleneck, not compute. The RK3588 has
four Cortex-A76 cores at 2.3 GHz and four Cortex-A55 cores at 1.8 GHz, sharing
a dual-channel LPDDR4X memory controller with a theoretical peak of about 51.2
GiB/s. In practice, sustained bandwidth is much lower, and for
bandwidth-bound kernels like the gated delta-rule scan, achieved throughput is
a fraction of that peak.

The gated delta network replaces traditional attention with a linear recurrent
update. At each time step, a gate controls how much of the previous state is
retained and how much is updated. The delta rule writes new information into
the state while simultaneously erasing old information along the same
direction. This is more expressive than simple additive RNNs and avoids the
quadratic cost of self-attention.

In practice, the chunkwise formulation processes blocks of tokens at once,
using matrix multiply for the intra-chunk attention and the recurrent update
for cross-chunk state propagation. The chunk size trades off parallelism
against sequential dependency: larger chunks enable more matrix-multiply work
that maps well to wide SIMD units, while smaller chunks keep the recurrent
state fresher but reduce arithmetic intensity.

The memory advantage is dramatic at long context. A standard transformer with
key-value cache stores N x D tensors per layer, growing linearly with sequence
length. At 262K tokens, a single layer's KV cache for the Qwen 3.5 4B model
consumes hundreds of megabytes. The gated delta state is a fixed-size tensor
that never grows, so the memory footprint is constant regardless of context
length.

This constant-memory property is what makes GDN attractive for edge devices.
An RK3588 with 32 GB of RAM can hold the model weights and a large batch of
recurrent states simultaneously, whereas a full-attention model of the same
size would exhaust memory at long context. The trade-off is that the recurrent
scan is sequential by nature, limiting parallelism during decode.

Optimizations for the gated delta scan include loop interleaving across
multiple sequences (batch parallelism), dot-product instructions for the
element-wise multiply-accumulate, and careful register allocation to keep the
recurrent state in registers rather than spilling to the stack. On Armv8.2,
the SDOT instruction computes a dot product of two 8-bit vectors, which can
accelerate quantized inference if the state is kept in 8-bit precision.

The question of numerical precision is critical for recurrent models. Unlike
feed-forward layers where rounding errors are local, a recurrent state
accumulates errors over every time step. At sequence length 262K, even a
small per-step bias can compound into a significant output drift. This is why
the golden reference is computed in float32, and candidate implementations
are validated with explicit tolerances that scale with context length.

Code generation is another important workload. Consider a function that
implements the gated delta update in Python:

def gated_delta_update(state, delta, gate, beta):
    state = state * gate + delta * beta
    return state

This is the core operation. When vectorized across the hidden dimension, it
becomes an element-wise multiply-add pattern that maps to SIMD instructions.
The challenge is not the arithmetic but the memory access pattern: the state
tensor must stay resident in registers or L1 cache across the entire sequence.

Thermal management is a real concern on edge devices. Under sustained load,
the RK3588 will throttle its clock frequency to stay within thermal limits.
This means that benchmark numbers collected without recording the thermal
state are not reproducible. The governor setting, ambient temperature, and
cooling solution all affect measured throughput.
"""

PROMPTS: list[dict] = [
    {
        "id": "factual",
        "text": (
            "The gating mechanism in Gated Delta Networks allows selective "
            "forgetting of past information. Unlike standard attention, the "
            "recurrent state does not grow with sequence length, making it "
            "suitable for long-context inference on memory-constrained edge "
            "devices. The key insight is that"
        ),
    },
    {
        "id": "code",
        "text": (
            "def gated_delta_scan(state, delta, alpha, beta):\n"
            '    """Chunkwise gated delta-rule update."""\n'
            "    state = state * alpha + delta * beta.unsqueeze(-1)\n"
            "    return state\n\n"
            "# The above implements the core recurrence. To make it"
        ),
    },
    {
        "id": "sequential",
        "text": (
            "Count from one to ten in order: one two three four five six "
            "seven eight nine ten. Now continue: eleven twelve thirteen "
            "fourteen fifteen sixteen seventeen eighteen nineteen twenty. "
            "The pattern continues with"
        ),
    },
    {
        "id": "reasoning",
        "text": (
            "Question: If a linear attention model processes a sequence of "
            "length N with a fixed-size recurrent state of dimension D, how "
            "does the memory footprint scale compared to full attention which "
            "stores N key-value pairs?\n\nAnswer: The linear attention model "
            "uses O(D) memory for the recurrent state regardless of N, while "
            "full attention uses O(N*D) for the KV cache. Therefore the "
            "advantage of linear attention is"
        ),
    },
]

# Context lengths to test.  We achieve each by encoding the seed prompt
# followed by filler text, then truncating to exactly the target length.
# This way every position carries real tokens — no padding.
CONTEXT_LENGTHS = [128, 512, 2048]

SMOKE_CONTEXT_LENGTHS = [128]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Return the current git commit SHA (short)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    """Return True if the working tree has uncommitted source changes.

    Excludes results/ and .beads/ — output data, not source code.
    Matches the filtering in scripts/capture_manifest.sh and bench/manifest.py.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
        )
        _OUTPUT_RE = re.compile(r"^[ ?][M?] (results/|\.beads/)")
        filtered = [
            line for line in result.stdout.splitlines()
            if not _OUTPUT_RE.match(line)
        ]
        return len(filtered) > 0
    except Exception:
        return True


def _governor() -> str:
    """Read the CPU governor state."""
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            return f.read().strip()
    except (OSError, FileNotFoundError):
        return "unknown"


def _thermals() -> list[int] | str:
    """Read thermal zone temperatures."""
    zones = sorted(Path("/sys/class/thermal").glob("thermal_zone*"))
    temps = []
    for z in zones:
        with contextlib.suppress(OSError, ValueError):
            temps.append(int(z.joinpath("temp").read_text().strip()))
    return temps if temps else "unknown"


def _hostname() -> str:
    try:
        return subprocess.check_output(["hostname"], text=True).strip()
    except Exception:
        return platform.node()


def collect_provenance(model_path: str) -> dict:
    """Collect full provenance metadata for reproducibility."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "hostname": _hostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "numpy_version": np.__version__,
        "model_path": str(model_path),
        "model_repo": "Qwen/Qwen3.5-0.8B",
        "device": "cpu",
        "dtype": "float32",
        "cpu_governor": _governor(),
        "thermals_pre": _thermals(),
    }


# ---------------------------------------------------------------------------
# Reference inference
# ---------------------------------------------------------------------------


def load_model(model_path: str):
    """Load the model and tokenizer in float32 on CPU."""
    print(f"Loading model from {model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {model.config.model_type} ({param_count / 1e6:.1f}M params)", flush=True)
    return model, tokenizer


def _build_sequence(seed_ids: list[int], filler_ids: list[int], target_length: int) -> list[int]:
    """Build a sequence of exactly target_length tokens from seed + filler.

    The seed prompt comes first, followed by filler tokens, truncated to
    target_length.  This way every position carries real content.
    """
    combined = seed_ids + filler_ids
    if len(combined) < target_length:
        # Not enough filler — repeat it
        reps = (target_length // len(filler_ids)) + 1
        combined = seed_ids + (filler_ids * reps)
    return combined[:target_length]


def run_reference_inference(
    model,
    tokenizer,
    prompts: list[dict],
    context_lengths: list[int],
    top_k: int = 20,
    decode_steps: int = 8,
) -> list[dict]:
    """Run reference inference and collect golden outputs.

    For each prompt x context_length:
      - Builds a real-token sequence truncated to the target length
      - Runs the full forward pass (no cache — clean-room logits)
      - Saves the last-position full-vocab logits
      - Saves top-k logits and indices for the last 4 positions
      - Computes sequence perplexity over all positions
      - Runs N greedy decode steps (with KV cache) and records generated IDs
    """
    filler_ids = tokenizer.encode(_LONG_FILLER)
    results = []

    for prompt in prompts:
        seed_ids = tokenizer.encode(prompt["text"])
        for ctx_len in context_lengths:
            entry_id = f"{prompt['id']}_{ctx_len}"
            input_ids = _build_sequence(seed_ids, filler_ids, ctx_len)
            print(f"  [{entry_id}] ctx={ctx_len} ...", flush=True)

            input_tensor = torch.tensor([input_ids], dtype=torch.long)

            # --- Forward pass (no cache for clean logits) ---
            t0 = time.time()
            with torch.inference_mode():
                outputs = model(input_tensor, use_cache=False, output_hidden_states=False)
            forward_ms = (time.time() - t0) * 1000

            logits = outputs.logits  # [1, seq_len, vocab_size]

            # --- Perplexity over all positions ---
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_tensor[:, 1:].contiguous()
            loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
            total_loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
            ).item()
            n_tokens = shift_labels.numel()
            avg_loss = total_loss / n_tokens
            perplexity = float(np.exp(avg_loss))

            # --- Last-position full-vocab logits ---
            last_logits = logits[0, -1, :].float().cpu().numpy()

            # --- Top-k for last 4 positions ---
            window = min(4, ctx_len)
            topk_data = []
            for pos in range(-window, 0):
                pos_logits = logits[0, pos, :].float().cpu().numpy()
                topk_idx = np.argsort(pos_logits)[-top_k:][::-1]
                topk_val = pos_logits[topk_idx]
                topk_data.append(
                    {
                        "position_from_end": pos,
                        "indices": topk_idx.tolist(),
                        "values": topk_val.tolist(),
                    }
                )

            # --- Decode steps (greedy, with KV cache) ---
            generated = []
            with torch.inference_mode():
                past = None
                cur_input = input_tensor
                for _step in range(decode_steps):
                    out = model(
                        cur_input if past is None else cur_input[:, -1:],
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = out.past_key_values
                    next_token = int(torch.argmax(out.logits[:, -1, :]).item())
                    generated.append(next_token)
                    cur_input = torch.tensor([[next_token]], dtype=torch.long)

            generated_text = tokenizer.decode(generated, skip_special_tokens=True)
            argmax_token = int(np.argmax(last_logits))
            argmax_text = tokenizer.decode([argmax_token])

            result = {
                "entry_id": entry_id,
                "prompt_id": prompt["id"],
                "context_length": ctx_len,
                "prompt_text": prompt["text"],
                "perplexity": perplexity,
                "avg_nll": avg_loss,
                "forward_ms": forward_ms,
                "argmax_token": argmax_token,
                "argmax_token_text": argmax_text,
                "last_position_logits": last_logits.tolist(),
                "topk_window": topk_data,
                "generated_token_ids": generated,
                "generated_text": generated_text,
            }
            results.append(result)
            print(
                f"    ppl={perplexity:.3f}  forward={forward_ms:.0f}ms  "
                f"argmax={argmax_token}({argmax_text!r})  "
                f"gen='{generated_text[:50]}...'",
                flush=True,
            )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="generate_reference",
        description="Generate golden reference outputs from a HuggingFace model.",
    )
    parser.add_argument(
        "--model-path",
        default="models/Qwen3.5-0.8B",
        help="Path to the model checkpoint directory.",
    )
    parser.add_argument(
        "--output",
        default="results/reference/qwen35-0.8b_reference.json",
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k logits to save per position window.",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=8,
        help="Greedy decode steps after the prompt.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: single prompt, short context only.",
    )

    args = parser.parse_args(argv)

    if not _TORCH_AVAILABLE:
        print(
            f"ERROR: torch/transformers not available: {_IMPORT_ERROR}\n"
            "Install with: pip install torch transformers",
            file=sys.stderr,
        )
        return 1

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: model path not found: {model_path}", file=sys.stderr)
        print("Run: python3 scripts/fetch_weights.py --model 0.8B", file=sys.stderr)
        return 1

    # --- Provenance ---
    provenance = collect_provenance(str(model_path))
    print(f"Provenance: git={provenance['git_sha']} host={provenance['hostname']}")
    print(
        f"  torch={provenance['torch_version']} transformers={provenance['transformers_version']}"
    )

    # --- Load model ---
    model, tokenizer = load_model(str(model_path))

    # --- Select prompts and context lengths ---
    prompts = PROMPTS[:1] if args.smoke else PROMPTS
    context_lengths = SMOKE_CONTEXT_LENGTHS if args.smoke else CONTEXT_LENGTHS

    # --- Run reference inference ---
    print(
        f"\nRunning reference inference ({len(prompts)} prompts x {len(context_lengths)} lengths)..."
    )
    results = run_reference_inference(
        model,
        tokenizer,
        prompts,
        context_lengths,
        top_k=args.top_k,
        decode_steps=args.decode_steps,
    )

    # --- Post-inference thermals ---
    provenance["thermals_post"] = _thermals()

    # --- Write output ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_version": 1,
        "provenance": provenance,
        "config": {
            "model_path": str(model_path),
            "dtype": "float32",
            "device": "cpu",
            "top_k": args.top_k,
            "decode_steps": args.decode_steps,
            "context_lengths": context_lengths,
        },
        "entries": results,
        "summary": {
            "num_entries": len(results),
            "mean_perplexity": float(np.mean([r["perplexity"] for r in results])),
            "context_lengths_tested": sorted(set(r["context_length"] for r in results)),
        },
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nWrote {output_path} ({size_mb:.1f} MiB)")
    print(f"  {len(results)} entries, mean ppl={output['summary']['mean_perplexity']:.3f}")
    print(f"  Context lengths: {output['summary']['context_lengths_tested']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
