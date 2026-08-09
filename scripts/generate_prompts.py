#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Generate reproducible long-context prompt corpus (ob-del).

Produces two prompt types at each canonical context length (4K, 32K, 128K, 262K):

1. **Needle-in-haystack** — a single fact ("needle") hidden at a controlled depth
   in filler text, followed by a question. Tests whether the model can retrieve
   a specific fact from a long context.

2. **RULER multi-key** — N key-value pairs scattered throughout filler text,
   followed by retrieval questions about specific keys. Tests multi-key retrieval,
   which is the GDN-2 hypothesis's claimed strength (ADR 0001).

Deterministic: fixed seeds, committed alongside the script. Re-running produces
bit-identical output. The generated files go to ``bench/prompts/``.

Usage::

    python3 scripts/generate_prompts.py             # generate all
    python3 scripts/generate_prompts.py --small      # 4K + 32K only (fast)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CANONICAL_LENGTHS = (4096, 32768, 131072, 262144)
CHARS_PER_TOKEN = 4
MASTER_SEED = 20260802
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "bench" / "prompts"

# ---------------------------------------------------------------------------
# Filler text (varied topics so needle-in-haystack is non-trivial)
# ---------------------------------------------------------------------------

_FILLER_TOPICS = [
    (
        "The history of computing traces from mechanical calculators through vacuum tube "
        "computers to the silicon era. Each generation brought orders of magnitude more "
        "transistors, faster clocks, and lower energy per operation. The shift from single-core "
        "to multi-core architectures in the mid-2000s marked a turning point in how software "
        "must be designed, forcing developers to think about concurrency and parallelism."
    ),
    (
        "Linear attention mechanisms reduce the computational complexity of transformers from "
        "quadratic to linear in sequence length. This is achieved by decomposing the attention "
        "matrix into kernel features and exploiting associativity of matrix multiplication. "
        "Gated DeltaNet extends this with a delta rule that allows the model to correct "
        "previously written information, improving quality without sacrificing efficiency."
    ),
    (
        "Memory bandwidth is often the bottleneck for inference on edge devices. Unlike "
        "datacenter GPUs with HBM running at terabytes per second, mobile and embedded "
        "processors share LPDDR memory between CPU, GPU, and accelerators. This makes "
        "arithmetic intensity rather than raw compute the determining factor for throughput."
    ),
    (
        "The Raspberry Pi 5 uses a quad-core Cortex-A76 processor with Armv8.2-A architecture "
        "and dot-product instructions. While it has the newest cores in our device fleet, "
        "its single-channel LPDDR4X gives it the lowest memory bandwidth at approximately "
        "seventeen gigabytes per second, creating an interesting contrast with older devices."
    ),
    (
        "Quantization reduces the precision of neural network weights from floating point to "
        "lower-bit integer representations. INT4 quantization cuts memory traffic by four times "
        "compared to FP16, which is significant when weight streaming dominates decode latency. "
        "However, recurrent state must remain in higher precision because quantization errors "
        "compound across the sequence rather than staying local to each step."
    ),
    (
        "The CIX P1 SoC in the Radxa Orion O6 combines a twelve-core CPU, an Immortalis G720 "
        "GPU, and a neural processing unit rated at twenty-eight point eight TOPS. The "
        "heterogeneous architecture raises the question of which layers of a hybrid model "
        "should run on which engine, and what the dispatch overhead costs at each boundary."
    ),
    (
        "Recurrent neural networks maintain a fixed-size state that is updated at each time "
        "step. This gives them O(1) memory complexity per token, in contrast to transformers "
        "whose key-value cache grows linearly with sequence length. The trade-off is that "
        "recurrent models cannot attend to arbitrary past positions, which can limit their "
        "ability to retrieve specific information from long contexts."
    ),
    (
        "Thermal management is critical for sustained performance on passively cooled devices. "
        "When a chip reaches its thermal limit, it reduces clock frequencies to stay within "
        "safe operating temperatures. This thermal throttling can cause throughput to drop by "
        "thirty percent or more, making it essential to monitor temperatures during benchmarking "
        "and to use performance governor for consistent measurements."
    ),
]


def _filler_text(target_chars: int, rng: random.Random) -> str:
    """Generate filler text of approximately *target_chars* characters."""
    if target_chars <= 0:
        return ""
    parts: list[str] = []
    total = 0
    while total < target_chars:
        para = rng.choice(_FILLER_TOPICS)
        parts.append(para)
        total += len(para) + 1
    return "\n\n".join(parts)[:target_chars]


# ---------------------------------------------------------------------------
# Needle-in-haystack
# ---------------------------------------------------------------------------

_NEEDLES = [
    (
        "The secret passcode for the archive is 7294.",
        "What is the secret passcode for the archive?",
        "7294",
    ),
    (
        "The annual revenue figure mentioned is 3.7 million dollars.",
        "What is the annual revenue figure mentioned?",
        "3.7 million",
    ),
    (
        "The designated meeting room is Cambridge-3B on the fourth floor.",
        "Which meeting room was designated?",
        "Cambridge-3B",
    ),
    (
        "The encryption key identifier is KEY-ALPHA-9981.",
        "What is the encryption key identifier?",
        "KEY-ALPHA-9981",
    ),
    (
        "The calibration constant was set to 0.0042 during the experiment.",
        "What value was the calibration constant set to?",
        "0.0042",
    ),
]


def generate_needle(target_tokens: int, rng: random.Random) -> tuple[str, dict]:
    """Generate a needle-in-haystack prompt at approximately *target_tokens* tokens.

    Returns ``(prompt_text, metadata)``.
    """
    needle_text, question, answer = rng.choice(_NEEDLES)

    target_chars = target_tokens * CHARS_PER_TOKEN
    filler_target = target_chars - len(needle_text) - len(question) - 4

    filler = _filler_text(filler_target, rng)
    mid = len(filler) // 2
    boundary = filler.rfind("\n\n", 0, mid)
    if boundary == -1:
        boundary = mid

    haystack = filler[:boundary] + "\n\n" + needle_text + "\n\n" + filler[boundary:]
    prompt = haystack + "\n\n" + question
    actual_tokens = len(prompt) // CHARS_PER_TOKEN

    metadata = {
        "type": "needle_in_haystack",
        "target_tokens": target_tokens,
        "actual_tokens_approx": actual_tokens,
        "needle": needle_text,
        "question": question,
        "expected_answer": answer,
        "needle_depth_fraction": boundary / len(filler) if len(filler) > 0 else 0.5,
        "seed": MASTER_SEED,
    }
    return prompt, metadata


# ---------------------------------------------------------------------------
# RULER multi-key
# ---------------------------------------------------------------------------


def _random_key(rng: random.Random) -> str:
    letters = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(3))
    digits = "".join(rng.choice("0123456789") for _ in range(4))
    return f"{letters}-{digits}"


def _random_value(rng: random.Random) -> str:
    letters = "".join(rng.choice("QRSTUVWXYZ") for _ in range(3))
    digits = "".join(rng.choice("0123456789") for _ in range(4))
    return f"{letters}-{digits}"


def generate_ruler(target_tokens: int, rng: random.Random) -> tuple[str, dict]:
    """Generate a RULER-style multi-key retrieval prompt.

    Returns ``(prompt_text, metadata)``.
    """
    num_keys = min(20, max(3, target_tokens // 512))
    keys = [_random_key(rng) for _ in range(num_keys)]
    values = [_random_value(rng) for _ in range(num_keys)]

    target_chars = target_tokens * CHARS_PER_TOKEN
    queried = keys[:5]
    question = f"Retrieve the values for the following keys: {', '.join(queried)}."
    filler_target = target_chars - len(question) - num_keys * 60 - 4

    num_segments = num_keys + 1
    segment_chars = max(100, filler_target // num_segments)

    parts: list[str] = []
    for i in range(num_keys):
        parts.append(_filler_text(segment_chars, rng))
        parts.append(f"Record {keys[i]} corresponds to value {values[i]}.")
    remaining = filler_target - sum(len(p) + 2 for p in parts)
    if remaining > 0:
        parts.append(_filler_text(remaining, rng))

    prompt = "\n\n".join(parts) + "\n\n" + question
    actual_tokens = len(prompt) // CHARS_PER_TOKEN

    metadata = {
        "type": "ruler_multi_key",
        "target_tokens": target_tokens,
        "actual_tokens_approx": actual_tokens,
        "num_keys": num_keys,
        "queried_keys": queried,
        "expected_answers": {keys[i]: values[i] for i in range(min(5, num_keys))},
        "seed": MASTER_SEED,
    }
    return prompt, metadata


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------


def generate_all(context_lengths: tuple[int, ...] = CANONICAL_LENGTHS) -> list[Path]:
    """Generate all prompt types at all context lengths.

    Returns the list of files written.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for ctx in context_lengths:
        # Needle-in-haystack
        rng_n = random.Random(MASTER_SEED + ctx)
        prompt, meta = generate_needle(ctx, rng_n)
        txt_path = OUTPUT_DIR / f"needle_{ctx}.txt"
        json_path = OUTPUT_DIR / f"needle_{ctx}.json"
        txt_path.write_text(prompt + "\n", encoding="utf-8")
        json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        written.extend([txt_path, json_path])

        # RULER multi-key
        rng_r = random.Random(MASTER_SEED + ctx + 100000)
        prompt, meta = generate_ruler(ctx, rng_r)
        txt_path = OUTPUT_DIR / f"ruler_{ctx}.txt"
        json_path = OUTPUT_DIR / f"ruler_{ctx}.json"
        txt_path.write_text(prompt + "\n", encoding="utf-8")
        json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        written.extend([txt_path, json_path])

    # Write a manifest summarizing the corpus
    manifest = {
        "generator": "scripts/generate_prompts.py",
        "master_seed": MASTER_SEED,
        "chars_per_token": CHARS_PER_TOKEN,
        "context_lengths": list(context_lengths),
        "types": ["needle_in_haystack", "ruler_multi_key"],
        "total_files": len(written),
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    written.append(manifest_path)

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate long-context prompt corpus (ob-del)")
    parser.add_argument("--small", action="store_true", help="Generate only 4K + 32K (fast)")
    args = parser.parse_args(argv)

    lengths = (4096, 32768) if args.small else CANONICAL_LENGTHS
    written = generate_all(lengths)
    print(f"Generated {len(written)} files in {OUTPUT_DIR}")
    for p in written:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
