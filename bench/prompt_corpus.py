"""Long-context prompt corpus: needle-in-haystack and RULER-style multi-key retrieval.

Bead ``ob-del``. Generates deterministic, reproducible prompts at every canonical
sweep length (4K / 32K / 128K / 262K). Serves double duty:

1. **Benchmark input** for ``bench/harness.py`` — the sweep needs real prompts at
   each context length, not placeholder text.
2. **Retieval-quality evaluation** for the GDN-2 long-context hypothesis (ob-zak):
   needle-in-haystack tests whether the model can find a specific fact buried in
   filler, and RULER multi-key tests whether it can retrieve multiple needles.

All generation is deterministic (fixed seeds, no randomness beyond a seeded PRNG),
so two runs on different machines produce byte-identical prompts. This is required
for reproducibility (PLAN.md §9) and for the retrieval evaluation, where the answer
must be checkable against a known ground truth.

Stdlib-only, like the rest of the bench/ package.

Usage::

    python3 bench/prompt_corpus.py --format csv --output results/raw/prompts.csv
    python3 bench/prompt_corpus.py --format jsonl --output prompts.jsonl
    python3 bench/prompt_corpus.py --needle 4096 --depth 0.5  # print one needle prompt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field

# Canonical sweep points (matching bench/schema.py CANONICAL_CONTEXT_LENGTHS).
CANONICAL_LENGTHS = (4096, 32768, 131072, 262144)

# Master seed for all generation — committed, never changed. Changing it
# invalidates all previously-collected retrieval results.
MASTER_SEED = 20260802


# ===========================================================================
# Deterministic PRNG (seeded, reproducible, stdlib-only)
# ===========================================================================


class SeededPRNG:
    """Deterministic PRNG using hashlib for cross-platform reproducibility.

    Python's ``random`` module with a fixed seed IS reproducible across runs on
    the same Python version, but its algorithm has changed between versions.
    Using SHA-256 as the mixing function guarantees byte-identical output on
    any Python ≥ 3.10, which is the project floor (pyproject.toml).
    """

    def __init__(self, seed: int):
        self._counter = 0
        self._seed = seed

    def _next_bytes(self, n: int) -> bytes:
        self._counter += 1
        h = hashlib.sha256(f"{self._seed}:{self._counter}".encode())
        return h.digest()[:n]

    def next_float(self) -> float:
        """Uniform float in [0, 1)."""
        raw = int.from_bytes(self._next_bytes(8), "big")
        return raw / (1 << 64)

    def next_int(self, lo: int, hi: int) -> int:
        """Uniform int in [lo, hi)."""
        return lo + int(self.next_float() * (hi - lo))

    def choice(self, items: list) -> object:
        return items[self.next_int(0, len(items))]

    def shuffle(self, items: list) -> list:
        """Fisher-Yates shuffle, deterministic."""
        result = list(items)
        for i in range(len(result) - 1, 0, -1):
            j = self.next_int(0, i + 1)
            result[i], result[j] = result[j], result[i]
        return result


# ===========================================================================
# Filler text (deterministic, diverse, ~tokenizable)
# ===========================================================================

# A pool of sentences that are grammatical but informationally bland — they fill
# context without introducing facts that could be confused with the needle.
# Deliberately NOT from a real corpus: real text may contain numbers, names, or
# facts that a model could confuse with the planted needle. These are synthetic.
_FILLER_SENTENCES = [
    "The system processes data in sequential order without modification.",
    "Each iteration follows the same computational path as before.",
    "The output remains consistent across multiple independent runs.",
    "Parameters are initialized to their default values at startup.",
    "The buffer accumulates values until the threshold is reached.",
    "Configuration options are loaded from the specified path.",
    "The scheduler distributes tasks according to the queue priority.",
    "Memory allocation follows the standard heap-based strategy.",
    "The pipeline executes stages in a fixed dependency order.",
    "Error handling routes exceptions to the designated callback.",
    "The cache invalidates entries based on the time-to-live value.",
    "Network connections are established with configurable timeouts.",
    "The logger writes entries in a structured text format.",
    "Resource cleanup occurs after the main processing loop exits.",
    "The tokenizer splits input text into discrete vocabulary units.",
    "Gradient updates are applied with the configured learning rate.",
    "The attention mechanism computes scores across all positions.",
    "Batch normalization adjusts activations using running statistics.",
    "The embedding layer maps token indices to dense vector representations.",
    "Regularization penalties discourage overly large parameter values.",
    "The optimizer adjusts weights to minimize the loss function.",
    "Activation functions introduce nonlinearity into the transformation.",
    "The residual connection adds the input to the transformed output.",
    "Layer normalization rescales features to zero mean and unit variance.",
    "Dropout randomly zeroes activations during the training phase.",
    "Positional encodings inject order information into the sequence.",
    "The decoder generates tokens autoregressively from the context.",
    "Beam search explores multiple hypotheses during decoding.",
    "Temperature scaling controls the sharpness of the output distribution.",
    "Top-k filtering restricts sampling to the most probable tokens.",
]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 4 characters (whitespace-separated heuristic).

    Not a real tokenizer — the harness calls ``backend.tokenize()`` for the actual
    count. This estimate is only for sizing filler to hit a target length.
    """
    return max(1, len(text) // 4)


def _build_filler(target_tokens: int, prng: SeededPRNG) -> str:
    """Build deterministic filler text of approximately ``target_tokens`` tokens."""
    parts: list[str] = []
    current = 0
    idx = 0
    while current < target_tokens:
        sentence = _FILLER_SENTENCES[idx % len(_FILLER_SENTENCES)]
        parts.append(sentence)
        current += _estimate_tokens(sentence)
        idx += 1
    return " ".join(parts)


# ===========================================================================
# Needle-in-haystack
# ===========================================================================

# The needles: a set of (question, answer, needle_text) triples. The needle_text
# is planted into the filler at a specified depth; the question asks for the answer.
# Using diverse, unambiguous needles so the model's retrieval is genuinely tested.
NEEDLES = [
    {
        "question": "What is the magic number?",
        "answer": "47291",
        "needle": "The magic number for this session is 47291.",
    },
    {
        "question": "What color is the signal lamp?",
        "answer": "amber",
        "needle": "The signal lamp is set to amber during maintenance.",
    },
    {
        "question": "Who sent the authorization code?",
        "answer": "Dr. Vasquez",
        "needle": "Dr. Vasquez sent the authorization code at noon.",
    },
    {
        "question": "What is the delivery route number?",
        "answer": "C-17",
        "needle": "The delivery route number is C-17 for all shipments.",
    },
    {
        "question": "What temperature is the reactor?",
        "answer": "847 degrees",
        "needle": "The reactor is currently at 847 degrees Celsius.",
    },
]


@dataclass
class NeedlePrompt:
    """A single needle-in-haystack prompt at a specific context length and depth."""

    context_length: int
    needle_index: int
    depth: float          # 0.0 = beginning, 1.0 = end, 0.5 = middle
    question: str
    expected_answer: str
    needle_text: str
    full_prompt: str
    estimated_tokens: int


def generate_needle_prompt(
    context_length: int,
    needle_idx: int = 0,
    depth: float = 0.5,
    seed: int = MASTER_SEED,
) -> NeedlePrompt:
    """Generate one needle-in-haystack prompt.

    Args:
        context_length: Target token count for the full prompt.
        needle_idx: Which needle to plant (0-4, cycling).
        depth: Where to plant the needle in the filler (0.0=start, 1.0=end).
        seed: PRNG seed for deterministic filler generation.
    """
    prng = SeededPRNG(seed + context_length + needle_idx)
    needle = NEEDLES[needle_idx % len(NEEDLES)]

    # Build the instruction prefix and question suffix.
    prefix = "You are a helpful assistant. Read the following text carefully and answer the question at the end.\n\n"
    suffix = f"\n\nQuestion: {needle['question']}\nAnswer:"

    # Budget for filler: total - prefix - suffix - needle
    prefix_tokens = _estimate_tokens(prefix)
    suffix_tokens = _estimate_tokens(suffix)
    needle_tokens = _estimate_tokens(needle["needle"])
    filler_target = max(100, context_length - prefix_tokens - suffix_tokens - needle_tokens)

    filler = _build_filler(filler_target, prng)

    # Insert the needle at the specified depth in the filler.
    split_point = int(len(filler) * depth)
    before = filler[:split_point].rstrip()
    after = filler[split_point:].lstrip()
    body = f"{before} {needle['needle']} {after}"

    full_prompt = prefix + body + suffix
    estimated = _estimate_tokens(full_prompt)

    return NeedlePrompt(
        context_length=context_length,
        needle_index=needle_idx,
        depth=depth,
        question=needle["question"],
        expected_answer=needle["answer"],
        needle_text=needle["needle"],
        full_prompt=full_prompt,
        estimated_tokens=estimated,
    )


# ===========================================================================
# RULER-style multi-key retrieval
# ===========================================================================

@dataclass
class MultiKeyPrompt:
    """A RULER-style multi-key retrieval prompt.

    Plants ``num_needles`` key-value pairs at random positions in the filler,
    then asks the model to retrieve all of them. Tests whether the model can
    maintain multiple facts across a long context, not just one.
    """

    context_length: int
    num_needles: int
    needles: list[dict]       # list of {key, value, question}
    full_prompt: str
    expected_answers: dict[str, str]
    estimated_tokens: int


# Synthetic key-value pairs for multi-key retrieval. Keys are UUID-like to avoid
# collision with filler text; values are distinctive short strings.
def _generate_kv_pairs(n: int, prng: SeededPRNG) -> list[dict]:
    """Generate n unique key-value pairs deterministically."""
    pairs = []
    used_keys = set()
    for _ in range(n):
        # Deterministic key: "KEY-XXXX" where XXXX cycles through a large space
        key_num = prng.next_int(10000, 99999)
        key = f"PASS-{key_num}"
        while key in used_keys:
            key_num = prng.next_int(10000, 99999)
            key = f"PASS-{key_num}"
        used_keys.add(key)

        # Deterministic value: a color + number
        colors = ["crimson", "indigo", "emerald", "goldenrod", "silver", "violet", "coral"]
        color = prng.choice(colors)
        value_num = prng.next_int(100, 999)
        value = f"{color}-{value_num}"

        pairs.append({
            "key": key,
            "value": value,
            "needle": f"The passcode for {key} is {value}.",
            "question": f"What is the passcode for {key}?",
        })
    return pairs


def generate_multikey_prompt(
    context_length: int,
    num_needles: int = 3,
    seed: int = MASTER_SEED,
) -> MultiKeyPrompt:
    """Generate a RULER-style multi-key retrieval prompt.

    Args:
        context_length: Target token count.
        num_needles: Number of key-value pairs to plant (default 3).
        seed: PRNG seed.
    """
    prng = SeededPRNG(seed + context_length + num_needles * 1000)

    prefix = "You are a helpful assistant. Read the following text and remember all the passcodes mentioned.\n\n"
    questions_intro = "\n\nBased on the text above, answer the following questions:\n"

    kv_pairs = _generate_kv_pairs(num_needles, prng)

    # Build questions suffix
    questions = []
    expected = {}
    for kv in kv_pairs:
        questions.append(f"Q: {kv['question']}")
        expected[kv["key"]] = kv["value"]
    suffix = questions_intro + "\n".join(questions)

    # Budget for filler
    prefix_tokens = _estimate_tokens(prefix)
    suffix_tokens = _estimate_tokens(suffix)
    needle_tokens = sum(_estimate_tokens(kv["needle"]) for kv in kv_pairs)
    filler_target = max(100, context_length - prefix_tokens - suffix_tokens - needle_tokens)

    filler = _build_filler(filler_target, prng)

    # Split filler into num_needles+1 segments, insert needles at the boundaries
    segments = []
    seg_len = len(filler) // (num_needles + 1)
    pos = 0
    for i in range(num_needles):
        end = min(pos + seg_len, len(filler))
        segments.append(filler[pos:end].strip())
        segments.append(kv_pairs[i]["needle"])
        pos = end
    segments.append(filler[pos:].strip())

    body = " ".join(s for s in segments if s)
    full_prompt = prefix + body + suffix
    estimated = _estimate_tokens(full_prompt)

    return MultiKeyPrompt(
        context_length=context_length,
        num_needles=num_needles,
        needles=kv_pairs,
        full_prompt=full_prompt,
        expected_answers=expected,
        estimated_tokens=estimated,
    )


# ===========================================================================
# Full corpus generation
# ===========================================================================

@dataclass
class CorpusEntry:
    """One entry in the prompt corpus, ready for CSV/JSONL serialization."""

    prompt_id: str
    prompt_type: str       # "needle" or "multikey"
    context_length: int
    estimated_tokens: int
    question: str
    expected_answer: str
    full_prompt: str
    metadata: dict = field(default_factory=dict)


def generate_corpus(
    context_lengths: list[int] | None = None,
    needle_indices: list[int] | None = None,
    depths: list[float] | None = None,
    multikey_counts: list[int] | None = None,
    seed: int = MASTER_SEED,
) -> list[CorpusEntry]:
    """Generate the full prompt corpus across all sweep points.

    For each context length, generates:
      - needle prompts at multiple depths (0.0, 0.25, 0.5, 0.75, 1.0)
      - multi-key prompts with 1, 3, 5 keys

    Returns a flat list of CorpusEntry objects.
    """
    if context_lengths is None:
        context_lengths = list(CANONICAL_LENGTHS)
    if needle_indices is None:
        needle_indices = [0]  # one needle per depth for now
    if depths is None:
        depths = [0.0, 0.25, 0.5, 0.75, 1.0]
    if multikey_counts is None:
        multikey_counts = [1, 3, 5]

    entries: list[CorpusEntry] = []

    for ctx in context_lengths:
        # Needle prompts
        for ni in needle_indices:
            for depth in depths:
                np = generate_needle_prompt(ctx, ni, depth, seed)
                entries.append(CorpusEntry(
                    prompt_id=f"needle_{ctx}_d{int(depth*100):02d}_n{ni}",
                    prompt_type="needle",
                    context_length=ctx,
                    estimated_tokens=np.estimated_tokens,
                    question=np.question,
                    expected_answer=np.expected_answer,
                    full_prompt=np.full_prompt,
                    metadata={"depth": depth, "needle_index": ni},
                ))

        # Multi-key prompts
        for nk in multikey_counts:
            mk = generate_multikey_prompt(ctx, nk, seed)
            entries.append(CorpusEntry(
                prompt_id=f"multikey_{ctx}_k{nk}",
                prompt_type="multikey",
                context_length=ctx,
                estimated_tokens=mk.estimated_tokens,
                question="; ".join(q["question"] for q in mk.needles),
                expected_answer=json.dumps(mk.expected_answers),
                full_prompt=mk.full_prompt,
                metadata={"num_keys": nk},
            ))

    return entries


# ===========================================================================
# Serialization
# ===========================================================================


def write_csv(entries: list[CorpusEntry], path: str) -> None:
    """Write corpus entries as CSV (prompt text in a column, metadata as JSON)."""
    fields = [
        "prompt_id", "prompt_type", "context_length", "estimated_tokens",
        "question", "expected_answer", "metadata_json", "full_prompt",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in entries:
            writer.writerow({
                "prompt_id": e.prompt_id,
                "prompt_type": e.prompt_type,
                "context_length": e.context_length,
                "estimated_tokens": e.estimated_tokens,
                "question": e.question,
                "expected_answer": e.expected_answer,
                "metadata_json": json.dumps(e.metadata),
                "full_prompt": e.full_prompt,
            })


def write_jsonl(entries: list[CorpusEntry], path: str) -> None:
    """Write corpus entries as JSONL (one JSON object per line)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            obj = {
                "prompt_id": e.prompt_id,
                "prompt_type": e.prompt_type,
                "context_length": e.context_length,
                "estimated_tokens": e.estimated_tokens,
                "question": e.question,
                "expected_answer": e.expected_answer,
                "full_prompt": e.full_prompt,
                "metadata": e.metadata,
            }
            f.write(json.dumps(obj) + "\n")


# ===========================================================================
# CLI
# ===========================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prompt_corpus",
        description="Generate deterministic needle-in-haystack and RULER-style multi-key "
                    "retrieval prompts for benchmarking and GDN-2 evaluation.",
    )
    parser.add_argument(
        "--context-lengths",
        type=str,
        default=",".join(str(c) for c in CANONICAL_LENGTHS),
        help="Comma-separated target token counts.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "jsonl"],
        default="csv",
        help="Output format.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path. If omitted, prints summary to stderr.",
    )
    parser.add_argument(
        "--needle",
        type=int,
        default=None,
        help="Print one needle prompt at this context length (for eyeballing).",
    )
    parser.add_argument(
        "--depth",
        type=float,
        default=0.5,
        help="Needle depth (0=start, 1=end, 0.5=middle). Use with --needle.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=MASTER_SEED,
        help=f"PRNG seed (default: {MASTER_SEED}).",
    )
    args = parser.parse_args(argv)

    # Single-prompt eyeball mode
    if args.needle is not None:
        np = generate_needle_prompt(args.needle, 0, args.depth, args.seed)
        print(f"=== Needle prompt (ctx={args.needle}, depth={args.depth}) ===")
        print(f"Estimated tokens: {np.estimated_tokens}")
        print(f"Question: {np.question}")
        print(f"Expected answer: {np.expected_answer}")
        print(f"Needle text: {np.needle_text}")
        print("\n--- Full prompt (first 500 chars) ---")
        print(np.full_prompt[:500] + "...")
        return 0

    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    entries = generate_corpus(context_lengths=context_lengths, seed=args.seed)

    if args.output:
        if args.format == "csv":
            write_csv(entries, args.output)
        else:
            write_jsonl(entries, args.output)
        print(f"Wrote {len(entries)} prompts to {args.output}", file=sys.stderr)
    else:
        # Summary to stdout
        print(f"Prompt corpus: {len(entries)} entries")
        print(f"  Seed: {args.seed}")
        print(f"  Context lengths: {context_lengths}")
        by_type: dict[str, int] = {}
        for e in entries:
            by_type[e.prompt_type] = by_type.get(e.prompt_type, 0) + 1
        for t, c in sorted(by_type.items()):
            print(f"  {t}: {c} prompts")
        # Show a sample
        if entries:
            sample = entries[0]
            print(f"\n  Sample ({sample.prompt_id}):")
            print(f"    type={sample.prompt_type}  ctx={sample.context_length}")
            print(f"    est_tokens={sample.estimated_tokens}")
            print(f"    question={sample.question[:80]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
