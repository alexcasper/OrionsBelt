#!/usr/bin/env python3
"""Long-context prompt corpus generator for the GDN benchmark.

Bead ``ob-del``. Generates reproducible needle-in-haystack and RULER-style multi-key
retrieval prompts across the canonical sweep range (4K–262K target tokens). Every
prompt is deterministic given its seed — no external data sources, no randomness
outside a seeded PRNG.

Two task families:

1. **Needle-in-haystack (NIAH)** — insert a single fact at a specified depth in a
   filler text, then ask a retrieval question. The simplest and most sensitive test
   of whether the model's recurrent state / KV cache retains information from early
   positions.

2. **Multi-key (RULER-style)** — scatter N key-value pairs across the context, then
   ask for one specific value. Tests whether the model can distinguish among similar
   keys when multiple needles compete for attention.

Token-length estimation: we use a word-count heuristic (≈ 1.33 tokens/word for
English) because the generator must run without a tokenizer dependency. The actual
token count is verified at harness runtime by the backend's ``tokenize()`` method.
The heuristic targets are deliberately ~5% under the sweep point to leave room for
the question and needle text.

CLI::

    python3 -m bench.prompts --all                 # generate all corpora
    python3 -m bench.prompts --niah --depths 0,50,100
    python3 -m bench.prompts --multikey --keys 10

Output: JSON files in ``bench/prompts_data/``, one per (task, sweep_point, depth).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _REPO_ROOT / "bench" / "prompts_data"

CANONICAL_SWEEP_POINTS = [4096, 32768, 131072, 262144]

# Token-to-word heuristic: Qwen tokenizers average ~1.33 tokens/word for English.
# We target 5% under to leave room for the question suffix.
TOKENS_PER_WORD = 1.33
TARGET_FILLER_RATIO = 0.93  # filler is 93% of context; rest is needle + question

# Default NIAH depths (percentile positions in the context)
DEFAULT_DEPTHS = [0, 10, 25, 50, 75, 90, 100]

# Default multi-key counts
DEFAULT_MULTIKEY_COUNTS = [1, 3, 10, 50]

# Deterministic base seed (so all corpora share a reproducible lineage)
BASE_SEED = 42

# Word bank for generating varied filler text (deliberately generic, no semantic content)
_WORD_BANK = [
    "system",
    "process",
    "data",
    "value",
    "function",
    "element",
    "channel",
    "structure",
    "operation",
    "model",
    "parameter",
    "config",
    "state",
    "memory",
    "signal",
    "output",
    "input",
    "layer",
    "network",
    "sequence",
    "vector",
    "matrix",
    "tensor",
    "gradient",
    "weight",
    "bias",
    "kernel",
    "filter",
    "buffer",
    "cache",
    "queue",
    "stack",
    "register",
    "address",
    "pointer",
    "index",
    "offset",
    "boundary",
    "threshold",
    "frequency",
    "latency",
    "throughput",
    "bandwidth",
    "capacity",
    "density",
    "velocity",
    "acceleration",
    "position",
    "distance",
    "angle",
    "coordinate",
    "dimension",
    "resolution",
    "precision",
    "accuracy",
    "stability",
    "efficiency",
    "reliability",
    "scalability",
    "flexibility",
    "compatibility",
    "interoperability",
    "synchronization",
    "optimization",
    "parallel",
    "serial",
    "asynchronous",
    "distributed",
    "centralized",
    "hierarchical",
    "sequential",
    "recursive",
    "iterative",
    "adaptive",
    "predictive",
    "reactive",
    "proactive",
    "passive",
    "active",
    "static",
    "dynamic",
    "continuous",
    "discrete",
    "analog",
    "digital",
    "binary",
    "hexadecimal",
    "octal",
    "decimal",
    "fractional",
    "integral",
    "differential",
    "exponential",
    "logarithmic",
    "linear",
    "nonlinear",
    "periodic",
    "aperiodic",
    "stochastic",
    "deterministic",
]

_SENTENCE_TEMPLATES = [
    "The {a} {b} of the {c} {d} is {e}.",
    "Each {a} in the {c} {d} must maintain its {b}.",
    "When the {d} changes, the {a} {b} adjusts its {c}.",
    "The {c} {a} processes {b} data through a {d} pipeline.",
    "A {d}'s {b} depends on its {a} and the current {c}.",
    "During {e}, the {a} {d} records a new {b} for each {c}.",
    "The {c} parameter controls how the {a} {b} scales with {d}.",
    "Every {d} cycle, the {a} checks whether its {b} matches the {c}.",
    "The {b} of a {c} {d} is proportional to its {a} and inversely to its {e}.",
    "If the {a} exceeds the {c} {e}, the {b} triggers a {d} reset.",
]


# ---------------------------------------------------------------------------
# Filler text generation
# ---------------------------------------------------------------------------

def _generate_filler(rng: random.Random, num_words: int) -> str:
    """Generate deterministic English-like filler text of approximately num_words words."""
    words = []
    while len(words) < num_words:
        template = rng.choice(_SENTENCE_TEMPLATES)
        fillers = {slot: rng.choice(_WORD_BANK) for slot in ("a", "b", "c", "d", "e")}
        sentence = template.format(**fillers)
        sentence_words = sentence.split()
        words.extend(sentence_words)
    # Trim to exact word count
    return " ".join(words[:num_words])


def _target_filler_words(target_tokens: int) -> int:
    """Estimate the number of filler words needed to reach the target token count."""
    return max(1, int(target_tokens * TARGET_FILLER_RATIO / TOKENS_PER_WORD))


def _estimate_tokens(text: str) -> int:
    """Rough token estimate for the generated text."""
    return int(len(text.split()) * TOKENS_PER_WORD)


# ---------------------------------------------------------------------------
# 1. Needle-in-haystack (NIAH)
# ---------------------------------------------------------------------------

class NIAHResult(NamedTuple):
    """A single needle-in-haystack prompt."""
    prompt: str
    needle: str
    expected_answer: str
    needle_depth_pct: int
    est_tokens: int
    seed: int


def generate_niah(
    target_tokens: int,
    needle_depth_pct: int = 50,
    seed: int = BASE_SEED,
) -> NIAHResult:
    """Generate a needle-in-haystack prompt.

    The needle is a memorable fact inserted at the specified depth (%) in the
    filler text. The question at the end asks the model to retrieve it.

    Args:
        target_tokens: Target context length in tokens (approximate).
        needle_depth_pct: Position of the needle in the context (0=start, 100=end).
        seed: PRNG seed for deterministic filler generation.
    """
    rng = random.Random(seed)

    # Generate a unique needle
    needle_id = rng.randint(10000, 99999)
    needle_color = rng.choice(["crimson", "azure", "emerald", "golden", "silver", "violet"])
    needle = f"The passcode for the gate is {needle_color}-{needle_id}."
    expected = f"{needle_color}-{needle_id}"

    # Build the filler
    filler_words = _target_filler_words(target_tokens)
    filler = _generate_filler(rng, filler_words)

    # Split filler at the needle depth
    filler_word_list = filler.split()
    insert_pos = int(len(filler_word_list) * needle_depth_pct / 100)
    insert_pos = max(0, min(insert_pos, len(filler_word_list)))

    before = " ".join(filler_word_list[:insert_pos])
    after = " ".join(filler_word_list[insert_pos:])

    # Assemble the prompt
    parts = [p for p in [before, needle, after] if p]
    context = "\n\n".join(parts)

    question = (
        "\n\nQuestion: Based on the text above, what is the passcode for the gate? "
        "Answer with only the passcode."
    )
    prompt = context + question
    est_tokens = _estimate_tokens(prompt)

    return NIAHResult(
        prompt=prompt,
        needle=needle,
        expected_answer=expected,
        needle_depth_pct=needle_depth_pct,
        est_tokens=est_tokens,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# 2. Multi-key (RULER-style)
# ---------------------------------------------------------------------------

class MultiKeyResult(NamedTuple):
    """A multi-key retrieval prompt."""
    prompt: str
    query_key: str
    expected_answer: str
    num_keys: int
    est_tokens: int
    seed: int


def generate_multikey(
    target_tokens: int,
    num_keys: int = 10,
    seed: int = BASE_SEED,
) -> MultiKeyResult:
    """Generate a RULER-style multi-key retrieval prompt.

    Scatters ``num_keys`` key-value pairs across the context, then asks for
    one specific value. Tests retrieval precision when multiple facts compete.

    Args:
        target_tokens: Target context length in tokens (approximate).
        num_keys: Number of key-value pairs to embed.
        seed: PRNG seed for deterministic generation.
    """
    rng = random.Random(seed)

    # Generate key-value pairs with memorable values
    pairs = []
    used_ids = set()
    for _ in range(num_keys):
        pair_id = rng.randint(1000, 9999)
        while pair_id in used_ids:
            pair_id = rng.randint(1000, 9999)
        used_ids.add(pair_id)
        pairs.append((f"item_{pair_id}", f"value_{pair_id}"))

    # Select the query key (not the first or last, to avoid position bias)
    query_idx = rng.randint(1, max(1, num_keys - 2)) if num_keys > 2 else 0
    query_key, query_value = pairs[query_idx]

    # Build filler interspersed with key-value pairs
    filler_words = _target_filler_words(target_tokens)
    filler = _generate_filler(rng, filler_words)
    filler_sentences = filler.split(". ")

    # Distribute key-value pairs across the filler
    output_parts: list[str] = []
    kv_interval = max(1, len(filler_sentences) // (num_keys + 1))
    kv_idx = 0
    for i, sent in enumerate(filler_sentences):
        output_parts.append(sent.rstrip() + ".")
        # Insert a KV pair at regular intervals
        insert_at = (kv_idx + 1) * kv_interval
        if kv_idx < num_keys and i >= insert_at:
            key, value = pairs[kv_idx]
            output_parts.append(f"[{key}] = {value}")
            kv_idx += 1

    # Insert any remaining KV pairs
    while kv_idx < num_keys:
        key, value = pairs[kv_idx]
        insert_pos = rng.randint(0, len(output_parts))
        output_parts.insert(insert_pos, f"[{key}] = {value}")
        kv_idx += 1

    context = " ".join(output_parts)
    question = (
        f"\n\nQuestion: What is the value assigned to {query_key}? "
        f"Answer with only the value."
    )
    prompt = context + question
    est_tokens = _estimate_tokens(prompt)

    return MultiKeyResult(
        prompt=prompt,
        query_key=query_key,
        expected_answer=query_value,
        num_keys=num_keys,
        est_tokens=est_tokens,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Corpus generation (CLI)
# ---------------------------------------------------------------------------

def generate_corpus(
    output_dir: Path | None = None,
    sweep_points: list[int] | None = None,
    depths: list[int] | None = None,
    multikey_counts: list[int] | None = None,
) -> list[Path]:
    """Generate all corpus files to disk.

    Returns list of output file paths.
    """
    if output_dir is None:
        output_dir = _OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if sweep_points is None:
        sweep_points = CANONICAL_SWEEP_POINTS
    if depths is None:
        depths = DEFAULT_DEPTHS
    if multikey_counts is None:
        multikey_counts = DEFAULT_MULTIKEY_COUNTS

    written: list[Path] = []

    for target_tokens in sweep_points:
        label = _sweep_label(target_tokens)

        # NIAH prompts
        for depth in depths:
            seed = BASE_SEED + hash((target_tokens, depth, "niah")) % 100000
            result = generate_niah(target_tokens, depth, seed)
            fname = f"niah_{label}_d{depth}.json"
            data = {
                "task": "niah",
                "target_tokens": target_tokens,
                "needle_depth_pct": depth,
                "est_tokens": result.est_tokens,
                "seed": seed,
                "needle": result.needle,
                "expected_answer": result.expected_answer,
                "prompt": result.prompt,
            }
            out = output_dir / fname
            out.write_text(json.dumps(data, indent=2) + "\n")
            written.append(out)

        # Multi-key prompts
        for nkeys in multikey_counts:
            seed = BASE_SEED + hash((target_tokens, nkeys, "multikey")) % 100000
            result = generate_multikey(target_tokens, nkeys, seed)
            fname = f"multikey_{label}_k{nkeys}.json"
            data = {
                "task": "multikey",
                "target_tokens": target_tokens,
                "num_keys": nkeys,
                "est_tokens": result.est_tokens,
                "seed": seed,
                "query_key": result.query_key,
                "expected_answer": result.expected_answer,
                "prompt": result.prompt,
            }
            out = output_dir / fname
            out.write_text(json.dumps(data, indent=2) + "\n")
            written.append(out)

    return written


# Canonical labels for the sweep points (short names used in filenames)
_SWEEP_LABELS = {
    4096: "4K",
    32768: "32K",
    131072: "128K",
    262144: "262K",
}


def _sweep_label(tokens: int) -> str:
    """4K, 32K, 128K, 262K style labels."""
    if tokens in _SWEEP_LABELS:
        return _SWEEP_LABELS[tokens]
    if tokens >= 1024:
        k = tokens / 1024
        if k == int(k):
            return f"{int(k)}K"
        return f"{k:.0f}K"
    return str(tokens)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate long-context prompt corpus for GDN benchmark.",
    )
    parser.add_argument("--all", action="store_true", help="Generate all corpora")
    parser.add_argument("--niah", action="store_true", help="Generate NIAH prompts only")
    parser.add_argument("--multikey", action="store_true", help="Generate multi-key prompts only")
    parser.add_argument("--depths", type=str, default=None,
                        help="Comma-separated NIAH depths (default: 0,10,25,50,75,90,100)")
    parser.add_argument("--keys", type=str, default=None,
                        help="Comma-separated multi-key counts (default: 1,3,10,50)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: bench/prompts_data)")
    args = parser.parse_args(argv)

    if not any([args.all, args.niah, args.multikey]):
        parser.print_help()
        return 1

    depths = [int(d) for d in args.depths.split(",")] if args.depths else None
    multikey_counts = [int(k) for k in args.keys.split(",")] if args.keys else None
    output_dir = Path(args.output) if args.output else None

    do_niah = args.all or args.niah
    do_multikey = args.all or args.multikey

    if not do_niah:
        depths = []  # skip NIAH
    if not do_multikey:
        multikey_counts = []  # skip multi-key

    written = generate_corpus(
        output_dir=output_dir,
        depths=depths or (DEFAULT_DEPTHS if do_niah else []),
        multikey_counts=multikey_counts or (DEFAULT_MULTIKEY_COUNTS if do_multikey else []),
    )

    print(f"Generated {len(written)} corpus files:")
    for p in written:
        try:
            rel = p.relative_to(_REPO_ROOT)
        except ValueError:
            rel = p
        print(f"  {rel}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
