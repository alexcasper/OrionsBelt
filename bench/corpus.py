"""Long-context prompt corpus — needle-in-a-haystack and RULER multi-key tasks.

Bead ``ob-del``.  Generates reproducible evaluation prompts across the canonical
sweep range (4K – 262K tokens) with deterministic seeds.  Serves double duty:

1. **Throughput benchmarking** — feeds the harness with realistic prompt texts
   at each context-length sweep point, so prefill and decode metrics are measured
   against genuine long-context inputs rather than synthetic byte arrays.
2. **Retrieval-quality evaluation** — the needle/multi-key tasks test whether the
   GDN recurrent-state architecture preserves long-context retrieval quality,
   which is the GDN-2 research hypothesis (E8, bead ``ob-zak``).

Two task families are generated:

- **NIAH single** (needle-in-a-haystack): one factual statement hidden at a
  controlled depth in a filler context.  The model must extract it.
- **Multi-key** (RULER-style): N key-value pairs embedded in the context.  The
  model must return the value for a queried key, testing associative recall.

Usage::

    # Generate all prompts at all sweep points
    python -m bench.corpus --output-dir prompts/

    # Generate specific task / length
    python -m bench.corpus --task niah_single --context-lengths 4096,32768

    # Write a single prompt to stdout (for piping to a model)
    python -m bench.corpus --task niah_single --context-lengths 4096 --stdout

The generated prompts are deterministic: the same seed always produces the same
prompt text, so results are reproducible across runs and devices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical context-length sweep points (PLAN.md §5, METRICS.md §7).
CANONICAL_LENGTHS = [4096, 32768, 131072, 262144]

# Approximate chars-per-token for English text (GPT-style BPE average).
# This is an estimate — when a tokenizer is available, pass ``char_token_ratio``
# to use the exact ratio for the target model.
DEFAULT_CHARS_PER_TOKEN = 4.0

# Master seed — changing this invalidates all generated prompts.
MASTER_SEED = 20260802

# Needle positions to test (fraction of context length, 0.0 = start, 1.0 = end).
NIAH_DEPTHS = [0.0, 0.25, 0.50, 0.75, 1.0]

# Number of multi-key pairs at each context length (scales with context).
MULTIKEY_COUNTS = {
    4096: 10,
    32768: 50,
    131072: 200,
    262144: 400,
}

# ---------------------------------------------------------------------------
# Filler text — public domain passages for the haystack
# ---------------------------------------------------------------------------

# A pool of public-domain passages used as filler text.  These are short enough
# to embed directly (no external file dependency) and varied enough that the
# model cannot memorise their structure to locate the needle.
FILLER_PASSAGES = [
    # From "The Art of War" by Sun Tzu (public domain)
    "The art of war is of vital importance to the State. It is a matter of life "
    "and death, a road either to safety or to ruin. Hence it is a subject of "
    "inquiry which can on no account be neglected. The art of war, then, is "
    "governed by five constant factors, to be taken into account in one's "
    "deliberations, when seeking to determine the conditions obtaining in the "
    "field. These are: the moral law, heaven, earth, the commander, and method "
    "and discipline.",
    # From "Pride and Prejudice" by Jane Austen (public domain)
    "It is a truth universally acknowledged, that a single man in possession of "
    "a good fortune, must be in want of a wife. However little known the "
    "feelings or views of such a man may be on his first entering a "
    "neighbourhood, this truth is so well fixed in the minds of the "
    "surrounding families, that he is considered as the rightful property of "
    "someone or other of their daughters.",
    # From "The Adventures of Sherlock Holmes" by Arthur Conan Doyle (PD)
    "Being a private detective, Sherlock Holmes had extraordinary powers of "
    "observation and deduction. He could tell a man's occupation from the "
    "wear on his boots, or trace a person's movements from the mud on their "
    "clothing. His methods were scientific, his reasoning always logical, and "
    "his conclusions were rarely wrong. He believed that when you have "
    "eliminated the impossible, whatever remains, however improbable, must be "
    "the truth.",
    # From "Moby Dick" by Herman Melville (public domain)
    "Call me Ishmael. Some years ago, never mind how long precisely, having "
    "little or no money in my purse, and nothing particular to interest me on "
    "shore, I thought I would sail about a little and see the watery part of "
    "the world. It is a way I have of driving off the spleen, and regulating "
    "the circulation. Whenever I find myself growing grim about the mouth, "
    "whenever it is a damp, drizzly November in my soul, I account it high "
    "time to get to sea as soon as I can.",
    # From "A Tale of Two Cities" by Charles Dickens (public domain)
    "It was the best of times, it was the worst of times, it was the age of "
    "wisdom, it was the age of foolishness, it was the epoch of belief, it was "
    "the epoch of incredulity, it was the season of Light, it was the season "
    "of Darkness, it was the spring of hope, it was the winter of despair, we "
    "had everything before us, we had nothing before us, we were all going "
    "direct to Heaven, we were all going direct the other way.",
    # From "The Time Machine" by H.G. Wells (public domain)
    "The Time Traveller, for so it will be convenient to speak of him, was "
    "expounding a recondite matter to us. His grey eyes shone and twinkled, and "
    "his usually pale face was flushed and animated. The fire burned brightly, "
    "and the soft radiance of the incandescent lights in the lilies of silver "
    "caught the bubbles that flashed and passed in our glasses.",
    # From "The Decline and Fall of the Roman Empire" by Edward Gibbon (PD)
    "In the second century of the Christian era, the Empire of Rome "
    "comprehended the fairest part of the earth, and the most civilised "
    "portion of mankind. The frontiers of that extensive monarchy were "
    "guarded by ancient renown and disciplined valour. The gentle but "
    "powerful influence of laws and manners had gradually cemented the union "
    "of the provinces.",
    # From "Walden" by Henry David Thoreau (public domain)
    "I went to the woods because I wished to live deliberately, to front only "
    "the essential facts of life, and see if I could not learn what it had to "
    "teach, and not, when I came to die, discover that I had not lived. I did "
    "not wish to live what was not life, living is so dear, nor did I wish to "
    "practise resignation, unless it was quite necessary.",
    # From "The Federalist Papers" No. 10 by James Madison (public domain)
    "A faction is a number of citizens, whether amounting to a majority or a "
    "minority of the whole, who are united and actuated by some common impulse "
    "of passion, or of interest, adversed to the rights of other citizens, or "
    "to the permanent and aggregate interests of the community. The latent "
    "causes of faction are thus sown in the nature of man.",
    # From "On the Origin of Species" by Charles Darwin (public domain)
    "It is interesting to contemplate an entangled bank, clothed with many "
    "plants of many kinds, with birds singing on the bushes, with various "
    "insects flitting about, and with worms crawling through the damp earth, "
    "and to reflect that these elaborately constructed forms, so different "
    "from each other, and dependent on each other in so complex a manner, "
    "have all been produced by laws acting around us.",
]

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class PromptItem:
    """One generated evaluation prompt with its ground-truth answer."""

    task_type: str  # "niah_single" | "niah_multikey"
    context_length: int  # target token count
    prompt: str  # full prompt text
    expected_answer: str  # ground-truth answer
    needle_depth: float  # position fraction (0.0 = start, 1.0 = end)
    seed: int  # deterministic generation seed
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "context_length": self.context_length,
            "prompt": self.prompt,
            "expected_answer": self.expected_answer,
            "needle_depth": self.needle_depth,
            "seed": self.seed,
            "metadata": self.metadata,
            "prompt_chars": len(self.prompt),
            "est_tokens": int(len(self.prompt) / DEFAULT_CHARS_PER_TOKEN),
        }


# ---------------------------------------------------------------------------
# Haystack generation
# ---------------------------------------------------------------------------


def generate_haystack(
    target_chars: int,
    rng: random.Random,
) -> str:
    """Build a filler text of approximately ``target_chars`` characters.

    Cycles through public-domain passages, selecting random starting points
    within each passage to avoid trivially periodic patterns.  The result is
    coherent prose that cannot be distinguished from the needle by structure
    alone.
    """
    chunks: list[str] = []
    current_len = 0
    passage_idx = rng.randint(0, len(FILLER_PASSAGES) - 1)

    while current_len < target_chars:
        passage = FILLER_PASSAGES[passage_idx]
        # Occasionally start mid-passage for variety
        if rng.random() < 0.3 and len(passage) > 50:
            start = rng.randint(0, min(40, len(passage) // 4))
            chunk = passage[start:]
        else:
            chunk = passage
        chunks.append(chunk)
        current_len += len(chunk) + 2  # +2 for the separator
        passage_idx = (passage_idx + 1) % len(FILLER_PASSAGES)

    text = "  ".join(chunks)
    # Trim to target
    if len(text) > target_chars:
        text = text[:target_chars].rsplit(" ", 1)[0] + "."
    return text


# ---------------------------------------------------------------------------
# Needle-in-a-Haystack (single needle)
# ---------------------------------------------------------------------------

# Needle templates — the "magic number" pattern from the NIAH literature,
# varied so the model cannot memorise a single answer.
NIAH_TEMPLATES = [
    (
        "The magic number stored in the registry is {number}.",
        "What is the magic number stored in the registry?",
        "{number}",
    ),
    (
        "The password for the north gate has been set to {number}.",
        "What is the password for the north gate?",
        "{number}",
    ),
    (
        "According to the log entry, the calibration code is {number}.",
        "What is the calibration code mentioned in the log?",
        "{number}",
    ),
    (
        "The treasure chest identifier is {number}.",
        "What is the treasure chest identifier?",
        "{number}",
    ),
    (
        "Dr. Watson noted that the patient ID is {number}.",
        "What is the patient ID that Dr. Watson noted?",
        "{number}",
    ),
]


def generate_niah_single(
    context_length: int,
    depth: float,
    seed: int,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> PromptItem:
    """Generate one needle-in-a-haystack prompt.

    Args:
        context_length: target token count for the full prompt.
        depth: needle position as fraction of context (0.0 = start, 1.0 = end).
        seed: deterministic generation seed.
        chars_per_token: chars-to-tokens ratio for length estimation.

    Returns:
        A PromptItem with the needle embedded in the haystack.
    """
    rng = random.Random(seed)

    # Reserve space for the question (~40 tokens)
    question_tokens = 40
    needle_tokens = 20
    haystack_tokens = context_length - question_tokens - needle_tokens
    target_chars = int(haystack_tokens * chars_per_token)

    # Generate the haystack
    haystack = generate_haystack(target_chars, rng)

    # Select needle template and number
    template_idx = rng.randint(0, len(NIAH_TEMPLATES) - 1)
    needle_stmt, question, answer_fmt = NIAH_TEMPLATES[template_idx]
    number = rng.randint(10000, 99999)
    needle = needle_stmt.format(number=number)
    expected_answer = answer_fmt.format(number=number)

    # Insert the needle at the target depth
    insert_pos = int(len(haystack) * depth)
    # Find a sentence boundary near the target position
    search_start = max(0, insert_pos - 100)
    boundary = haystack.find(". ", search_start)
    if boundary == -1 or boundary > insert_pos + 200:
        boundary = insert_pos  # fallback: exact position
    else:
        boundary += 2  # insert after the period+space

    full_context = haystack[:boundary] + needle + " " + haystack[boundary:]
    full_prompt = full_context + "\n\n" + question

    return PromptItem(
        task_type="niah_single",
        context_length=context_length,
        prompt=full_prompt,
        expected_answer=expected_answer,
        needle_depth=depth,
        seed=seed,
        metadata={
            "needle_template_idx": template_idx,
            "needle_value": str(number),
            "haystack_chars": len(haystack),
            "prompt_chars": len(full_prompt),
            "est_tokens": int(len(full_prompt) / chars_per_token),
            "chars_per_token": chars_per_token,
        },
    )


# ---------------------------------------------------------------------------
# RULER Multi-Key retrieval
# ---------------------------------------------------------------------------

# Key-value pairs for the multi-key task.  Keys are unique identifiers; values
# are random strings that the model must recall exactly.

MULTIKEY_KEY_PREFIXES = [
    "config_alpha",
    "config_beta",
    "config_gamma",
    "config_delta",
    "config_epsilon",
    "config_zeta",
    "config_eta",
    "config_theta",
    "config_iota",
    "config_kappa",
    "config_lambda",
    "config_mu",
    "config_nu",
    "config_xi",
    "config_omicron",
    "config_pi",
    "config_rho",
    "config_sigma",
    "config_tau",
    "config_upsilon",
]

MULTIKEY_VALUE_ADJECTIVES = [
    "crimson",
    "azure",
    "emerald",
    "golden",
    "silver",
    "violet",
    "amber",
    "coral",
    "ivory",
    "jade",
    "lavender",
    "magenta",
    "navy",
    "obsidian",
    "pearl",
    "quartz",
    "ruby",
    "sapphire",
    "teal",
    "umber",
]

MULTIKEY_VALUE_NOUNS = [
    "falcon",
    "compass",
    "beacon",
    "cylinder",
    "horizon",
    "marble",
    "obelisk",
    "pendulum",
    "quiver",
    "rampart",
    "scepter",
    "tundra",
    "vellum",
    "whirlpool",
    "zephyr",
    "anchor",
    "bridge",
    "cavern",
    "delta",
    "engine",
]


def _generate_value(rng: random.Random) -> str:
    """Generate a unique value string for a multi-key pair."""
    adj = rng.choice(MULTIKEY_VALUE_ADJECTIVES)
    noun = rng.choice(MULTIKEY_VALUE_NOUNS)
    num = rng.randint(100, 999)
    return f"{adj}-{noun}-{num}"


def generate_niah_multikey(
    context_length: int,
    num_keys: int | None = None,
    seed: int = 0,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> PromptItem:
    """Generate one RULER-style multi-key retrieval prompt.

    Embeds ``num_keys`` key-value pairs in the haystack, then asks the model
    to return the value for one randomly selected key.

    Args:
        context_length: target token count for the full prompt.
        num_keys: number of key-value pairs (auto-scaled if None).
        seed: deterministic generation seed.
        chars_per_token: chars-to-tokens ratio.

    Returns:
        A PromptItem with embedded key-value pairs and a query.
    """
    rng = random.Random(seed)

    if num_keys is None:
        num_keys = MULTIKEY_COUNTS.get(context_length, 50)

    # Generate unique keys and values
    pairs: list[tuple[str, str]] = []
    used_keys: set = set()
    used_values: set = set()

    for i in range(num_keys):
        # Generate a unique key
        prefix = MULTIKEY_KEY_PREFIXES[i % len(MULTIKEY_KEY_PREFIXES)]
        key_idx = i
        key = f"{prefix}_{key_idx:04d}"
        while key in used_keys:
            key_idx += 1
            key = f"{prefix}_{key_idx:04d}"
        used_keys.add(key)

        # Generate a unique value
        value = _generate_value(rng)
        while value in used_values:
            value = _generate_value(rng)
        used_values.add(value)

        pairs.append((key, value))

    # Reserve space for the question and pairs
    question_tokens = 30
    pair_tokens = num_keys * 8  # ~8 tokens per key-value statement
    haystack_tokens = max(100, context_length - question_tokens - pair_tokens)
    target_chars = int(haystack_tokens * chars_per_token)

    # Generate the filler haystack
    haystack = generate_haystack(target_chars, rng)

    # Build key-value statements and distribute them through the haystack
    kv_statements = [f'Remember: the value of "{k}" is "{v}".' for k, v in pairs]
    rng.shuffle(kv_statements)

    # Interleave KV statements into the haystack at roughly equal intervals
    segment_size = len(haystack) // (num_keys + 1)
    parts: list[str] = []
    cursor = 0

    for stmt in kv_statements:
        insert_pos = cursor + segment_size
        if insert_pos > len(haystack):
            insert_pos = len(haystack)
        parts.append(haystack[cursor:insert_pos])
        parts.append(" " + stmt + " ")
        cursor = insert_pos

    parts.append(haystack[cursor:])

    # Select a random key to query (not the first or last, to avoid edge effects)
    query_idx = rng.randint(num_keys // 4, 3 * num_keys // 4)
    query_key, query_value = pairs[query_idx]

    question = f'What is the value of "{query_key}"?'
    full_prompt = "".join(parts) + "\n\n" + question

    # Depth is the position of the queried pair
    # After shuffling, the queried pair's position is query_idx (in shuffled order)
    depth = query_idx / max(num_keys, 1)

    return PromptItem(
        task_type="niah_multikey",
        context_length=context_length,
        prompt=full_prompt,
        expected_answer=query_value,
        needle_depth=depth,
        seed=seed,
        metadata={
            "num_keys": num_keys,
            "query_key": query_key,
            "query_value": query_value,
            "all_keys": [k for k, _ in pairs],
            "prompt_chars": len(full_prompt),
            "est_tokens": int(len(full_prompt) / chars_per_token),
            "chars_per_token": chars_per_token,
        },
    )


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


@dataclass
class CorpusConfig:
    """Configuration for batch corpus generation."""

    context_lengths: list[int] = field(default_factory=lambda: list(CANONICAL_LENGTHS))
    tasks: list[str] = field(default_factory=lambda: ["niah_single", "niah_multikey"])
    niah_depths: list[float] = field(default_factory=lambda: list(NIAH_DEPTHS))
    master_seed: int = MASTER_SEED


def generate_corpus(
    config: CorpusConfig,
) -> list[PromptItem]:
    """Generate a full evaluation corpus from a config.

    For NIAH single: one prompt per (context_length, depth) combination.
    For multi-key: one prompt per context_length (depth is randomised internally).
    """
    items: list[PromptItem] = []

    for ctx_len in config.context_lengths:
        for task in config.tasks:
            if task == "niah_single":
                for depth in config.niah_depths:
                    # Deterministic per-depth seed
                    seed = _depth_seed(config.master_seed, ctx_len, depth)
                    items.append(generate_niah_single(ctx_len, depth, seed))
            elif task == "niah_multikey":
                seed = _task_seed(config.master_seed, ctx_len, "multikey")
                items.append(generate_niah_multikey(ctx_len, seed=seed))

    return items


def _depth_seed(master: int, ctx_len: int, depth: float) -> int:
    """Derive a deterministic seed for a (context_length, depth) pair."""
    key = f"{master}:{ctx_len}:{depth:.2f}:niah"
    h = hashlib.md5(key.encode()).hexdigest()
    return int(h[:8], 16)


def _task_seed(master: int, ctx_len: int, task_tag: str) -> int:
    """Derive a deterministic seed for a (context_length, task) pair."""
    key = f"{master}:{ctx_len}:{task_tag}"
    h = hashlib.md5(key.encode()).hexdigest()
    return int(h[:8], 16)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def save_corpus(items: list[PromptItem], output_dir: str) -> list[str]:
    """Save each PromptItem as a JSON file under ``output_dir``.

    Returns the list of written file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []

    for item in items:
        # Filename: <task>_<ctx>_<depth_or_keys>.json
        if item.task_type == "niah_single":
            fname = (
                f"{item.task_type}_{item.context_length}_d{int(item.needle_depth * 100):03d}.json"
            )
        else:
            nkeys = item.metadata.get("num_keys", 0)
            fname = f"{item.task_type}_{item.context_length}_k{nkeys:04d}.json"

        path = os.path.join(output_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(item.to_dict(), f, indent=2, ensure_ascii=False)
        paths.append(path)

    return paths


def save_manifest(items: list[PromptItem], output_dir: str) -> str:
    """Save a manifest summarising all generated prompts."""
    manifest_path = os.path.join(output_dir, "manifest.json")
    summary: dict = {
        "master_seed": MASTER_SEED,
        "total_prompts": len(items),
        "by_task": {},
        "by_context": {},
        "context_lengths": sorted({item.context_length for item in items}),
        "task_types": sorted({item.task_type for item in items}),
    }
    for item in items:
        summary["by_task"][item.task_type] = summary["by_task"].get(item.task_type, 0) + 1
        ctx_key = str(item.context_length)
        summary["by_context"][ctx_key] = summary["by_context"].get(ctx_key, 0) + 1

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return manifest_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate long-context evaluation prompts (NIAH + multi-key)."
    )
    parser.add_argument(
        "--output-dir",
        default="prompts/",
        help="Output directory for prompt JSON files (default: prompts/)",
    )
    parser.add_argument(
        "--context-lengths",
        default=",".join(str(c) for c in CANONICAL_LENGTHS),
        help="Comma-separated context lengths (default: 4096,32768,131072,262144)",
    )
    parser.add_argument(
        "--task",
        choices=["niah_single", "niah_multikey", "all"],
        default="all",
        help="Which task type to generate (default: all)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write the first prompt to stdout instead of saving files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=MASTER_SEED,
        help=f"Master seed (default: {MASTER_SEED})",
    )
    args = parser.parse_args(argv)

    ctx_lengths = [int(x.strip()) for x in args.context_lengths.split(",")]
    tasks = ["niah_single", "niah_multikey"] if args.task == "all" else [args.task]

    config = CorpusConfig(
        context_lengths=ctx_lengths,
        tasks=tasks,
        master_seed=args.seed,
    )

    items = generate_corpus(config)

    if args.stdout:
        if items:
            print(items[0].prompt)
        return 0

    save_corpus(items, args.output_dir)
    manifest = save_manifest(items, args.output_dir)

    print(f"Generated {len(items)} prompts in {args.output_dir}/")
    print(f"  Tasks: {', '.join(tasks)}")
    print(f"  Context lengths: {', '.join(str(c) for c in ctx_lengths)}")
    print(f"  Master seed: {args.seed}")
    print(f"  Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
