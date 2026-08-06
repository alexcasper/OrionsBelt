"""Benchmark runner CLI: context sweep with warmup, repeats, and percentiles.

Bead ``ob-ljh``. Protocol: ``docs/METRICS.md``. Schema: ``docs/RESULTS_SCHEMA.md``
/ ``bench/schema.py``.

Sweeps context lengths (4K / 32K / 128K / 262K by default) with configurable
warmup and repeat counts, emitting per-repeat rows in the frozen tidy/long
schema. Each context point is independently useful -- a truncated sweep still
yields publishable data (METRICS.md section 7; PLAN.md section 5/R4).

The harness is **backend-agnostic**: any class implementing the ``Backend``
protocol can be plugged in. A ``SyntheticBackend`` is provided for testing, CI
smoke runs, and development before the real model engine (``ob-aqv``) lands.

Targets Python 3.10+, stdlib-only (same constraint as ``bench/schema.py`` and
``bench/manifest.py``). The only non-stdlib import is from sibling bench modules.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure repo root is importable when run as a script (python3 bench/harness.py).
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.manifest import capture, manifest_ref  # noqa: E402
from bench.manifest import write as write_manifest  # noqa: E402
from bench.memory import ModelConfig  # noqa: E402
from bench.metrics import Summary, summarize  # noqa: E402
from bench.schema import (  # noqa: E402
    Device,
    Engine,
    LayerClass,
    ResultRow,
    validate_rows,
    write_csv,
)

# ---------------------------------------------------------------------------
# Model configuration (for analytic memory attribution, METRICS.md section 5)
# ---------------------------------------------------------------------------
#
# ``ModelConfig`` and the config-driven ``from_hf_config`` constructor live in
# ``bench/memory.py`` and are re-exported here for back-compat (backends, tests,
# and scripts import them from ``bench.harness``). The predicted
# ``peak_memory_bytes`` columns are computed by ``bench/memory.py`` via the
# analytical weights/KV/state formulas ported from ``origin/bench/t4`` — that
# module is the single source of truth for those columns (ob-7m6).


# Verified presets for the two checkpoints this project targets (ADR 0003,
# GDN_LAYER_AUDIT.md, QUANTIZATION_POLICY.md §"248K vocabulary"). Dimensions
# read from the real ``config.json`` ``text_config``; layer counts (24/8 and
# 18/6) are *derived* by the ``full_attention_interval=4`` → 3:1 derivation in
# ``ModelConfig.layer_types``, never hardcoded.
QWEN35_4B = ModelConfig(
    name="Qwen/Qwen3.5-4B",
    hidden_size=2560,
    num_hidden_layers=32,
    num_attention_heads=16,
    num_key_value_heads=4,  # GQA, verified from config.json (ob-37v)
    full_attn_head_dim=256,  # verified from config.json head_dim (ob-37v)
    linear_num_value_heads=32,
    linear_num_key_heads=16,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    intermediate_size=9216,
    vocab_size=248320,  # 248K vocabulary (QUANTIZATION_POLICY.md)
    tie_word_embeddings=True,
    full_attention_interval=4,
)

QWEN35_08B = ModelConfig(
    name="Qwen/Qwen3.5-0.8B",
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=8,
    num_key_value_heads=2,  # GQA, verified from config.json (ob-37v)
    full_attn_head_dim=256,  # verified from config.json head_dim (ob-37v)
    linear_num_value_heads=16,
    linear_num_key_heads=16,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    intermediate_size=3584,  # verified from HF config.json 2026-08-06
    vocab_size=248320,
    tie_word_embeddings=True,
    full_attention_interval=4,
)


def load_config_from_hub(repo_id: str, timeout: int = 10) -> ModelConfig:
    """Fetch a checkpoint's config.json from HuggingFace and build a ModelConfig.

    Stdlib-only (json + urllib). Requires network access — use
    ``load_config_from_dict`` with a local file in offline/CI environments.
    """
    import json
    import urllib.request

    url = f"https://huggingface.co/{repo_id}/raw/main/config.json"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return load_config_from_dict(data, name=repo_id)


_MODEL_PRESETS: dict[str, ModelConfig] = {
    "4b": QWEN35_4B,
    "0.8b": QWEN35_08B,
}


def load_config_from_dict(data: dict, name: str = "") -> ModelConfig:
    """Build a :class:`ModelConfig` from a raw Qwen3.5 ``config.json`` dict.

    Thin wrapper over ``ModelConfig.from_hf_config`` (ported from
    ``origin/bench/t4``). Resolves the layer structure — explicit
    ``layer_types`` array translated to the equivalent interval, else
    ``full_attention_interval`` read from the config — and maps
    ``mamba_ssm_dtype`` to ``state_dtype_bytes``. Verified against both
    Qwen3.5-4B and Qwen3.5-0.8B configs from HuggingFace (ob-37v, ob-xh3.2).
    """
    return ModelConfig.from_hf_config(data, name=name)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class Backend(ABC):
    """Inference backend interface.

    The harness handles **all** timing (monotonic clock per METRICS.md section 1).
    The backend provides the compute operations the harness times around. This
    separation means a ``Backend`` implementation never needs to know about the
    results schema, percentiles, or CSV emission -- it just does inference.
    """

    config: ModelConfig

    @abstractmethod
    def load(self) -> None:
        """Load model weights, prepare allocator arenas.

        Called once before the sweep. Per METRICS.md section 1, model load time
        is excluded from all reported metrics.
        """

    @abstractmethod
    def tokenize(self, text: str) -> list[int]:
        """Tokenize a prompt string to a list of token ids.

        The actual ``len(input_ids)`` -- not the sweep's ``context_length`` -- is
        what gets recorded as the prompt token count (METRICS.md section 2).
        """

    @abstractmethod
    def prefill(self, input_ids: list[int]) -> Any:
        """Run the full prefill forward pass over ``input_ids``.

        Returns logits (or any object ``sample()`` accepts). The harness times
        the interval [start, return] as the prefill forward duration
        (METRICS.md section 2).
        """

    @abstractmethod
    def sample(self, logits: Any) -> int:
        """Sample (argmax) the next token from the last position's logits.

        Token 1 belongs to prefill, not decode (METRICS.md section 1).
        """

    @abstractmethod
    def decode_step(self, token_id: int) -> int:
        """Single-token autoregressive decode step.

        Takes the previous token id, returns the next. EOS does not stop early
        during benchmarking (METRICS.md section 4).
        """

    @abstractmethod
    def memory_bytes(self) -> dict[str, int]:
        """Current memory attribution: ``{weights, kv_cache, recurrent_state}``.

        Computed analytically from model config + current state, never from
        process RSS (METRICS.md section 5.0). The harness calls this at the
        phase-end sampling instants (section 5.1).
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset recurrent state and KV cache.

        Called before each repeat so every measurement starts from a clean state.
        """


# ---------------------------------------------------------------------------
# Synthetic backend (for testing, CI smoke runs, development)
# ---------------------------------------------------------------------------


class SyntheticBackend(Backend):
    """Deterministic synthetic backend for testing and CI smoke runs.

    Produces no real inference -- it simulates the *timing structure* so the
    harness mechanics (warmup, repeats, timing boundaries, CSV emission,
    memory attribution) can be exercised without a model. Timing is near-zero
    by default; optional per-token nanosecond delays can simulate bandwidth-
    bound behavior for demos::

        python3 bench/harness.py --context-lengths 4096 --prefill-ns-per-token 500
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        prefill_ns_per_token: int = 0,
        decode_ns_per_step: int = 0,
    ):
        self.config = config
        self.prefill_ns_per_token = prefill_ns_per_token
        self.decode_ns_per_step = decode_ns_per_step
        self._seq_len = 0

    def load(self) -> None:
        pass

    def tokenize(self, text: str) -> list[int]:
        # Rough BPE approximation: ~4 chars per token.
        n = max(1, len(text) // 4)
        vocab = self.config.vocab_size
        return [(i * 7919 + 13) % vocab for i in range(n)]

    def prefill(self, input_ids: list[int]) -> Any:
        self._seq_len = len(input_ids)
        if self.prefill_ns_per_token:
            _busy_sleep(self.prefill_ns_per_token * len(input_ids))
        return None

    def sample(self, logits: Any) -> int:
        return 42

    def decode_step(self, token_id: int) -> int:
        self._seq_len += 1
        if self.decode_ns_per_step:
            _busy_sleep(self.decode_ns_per_step)
        return (token_id + 1) % self.config.vocab_size

    def memory_bytes(self) -> dict[str, int]:
        from bench.memory import kv_cache_bytes, recurrent_state_bytes, weights_bytes

        return {
            "weights": weights_bytes(self.config),
            "kv_cache": kv_cache_bytes(self.config, self._seq_len),
            "recurrent_state": recurrent_state_bytes(self.config),
        }

    def reset(self) -> None:
        self._seq_len = 0


def _busy_sleep(ns: int) -> None:
    """Busy-wait for approximately *ns* nanoseconds (avoids ``time.sleep`` jitter)."""
    target = time.perf_counter_ns() + ns
    while time.perf_counter_ns() < target:
        pass


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    """Configuration for one harness sweep run.

    Every field maps directly to a column or protocol parameter documented in
    ``docs/RESULTS_SCHEMA.md`` and ``docs/METRICS.md``.
    """

    context_lengths: list[int]
    warmup_count: int = 3
    repeat_count: int = 10
    decode_length: int = 257  # 1 prefill token + 256 decode tokens (METRICS.md section 4)
    # Identity fields for the CSV (RESULTS_SCHEMA.md section 3)
    device: str = "generic_aarch64"
    engine_gdn: str = "cpu"
    engine_full_attention: str = "cpu"
    model_checkpoint: str = "Qwen/Qwen3.5-4B@synthetic"
    quantization: str = "fp16"
    # Output
    manifest_dir: str = "results/manifests"
    notes: str = ""
    # Provenance escape hatch: run outside a git repo. Off by default so an
    # un-attributable run cannot be produced by accident (see _git_short_sha).
    allow_missing_sha: bool = False

    def __post_init__(self) -> None:
        """Fail fast on values the frozen schema would reject.

        The CLI already constrains these via argparse ``choices``, but callers
        that build a SweepConfig directly — ``scripts/run_ablation.py``,
        ``bench/hf_backend.py``, tests — bypassed that entirely. Without this the
        first sign of a typo'd device is ``validate_rows`` raising *after* the
        whole sweep has run, which at 262K context is an expensive way to find
        out. Checked here rather than in ``run_sweep`` so the object cannot exist
        in an invalid state.
        """
        valid_devices = {d.value for d in Device}
        valid_engines = {e.value for e in Engine}
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {sorted(valid_devices)}, got {self.device!r}")
        if self.engine_gdn not in valid_engines:
            raise ValueError(
                f"engine_gdn must be one of {sorted(valid_engines)}, got {self.engine_gdn!r}"
            )
        if self.engine_full_attention not in valid_engines:
            raise ValueError(
                f"engine_full_attention must be one of {sorted(valid_engines)}, "
                f"got {self.engine_full_attention!r}"
            )
        if self.repeat_count < 5:
            raise ValueError(
                "repeat_count must be >= 5 (METRICS.md section 7: 'never report N < 5'), "
                f"got {self.repeat_count}"
            )

    # Prompt source (ob-mrd.2): "synthetic" generates filler text,
    # "needle"/"ruler" load from the committed corpus (ob-del).
    prompt_type: str = "synthetic"
    prompt_dir: str = "bench/prompts"


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

_BASE_PROMPT = (
    "The future of efficient AI inference on Arm silicon depends on understanding "
    "the memory access patterns of novel architectures. Gated DeltaNet layers "
    "decode in O(1) memory per token, unlike full attention which grows a linear "
    "KV cache. This benchmark measures that architectural difference precisely. "
)


def load_corpus_prompt(
    prompt_type: str, context_length: int, prompt_dir: str = "bench/prompts"
) -> str:
    """Load a committed prompt from the corpus (ob-del).

    ``prompt_type`` is "needle" or "ruler". Falls back to ``generate_prompt``
    if the corpus file is not found, so a truncated corpus doesn't break the sweep.
    """
    from pathlib import Path

    path = Path(prompt_dir) / f"{prompt_type}_{context_length}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").rstrip("\n")
    # Graceful fallback — the sweep should never crash on a missing prompt file
    return generate_prompt(context_length)


def generate_prompt(target_tokens: int, chars_per_token: int = 4) -> str:
    """Generate synthetic prompt text of approximately *target_tokens* tokens.

    The harness records ``len(input_ids)`` -- the *actual* token count after
    tokenization -- not this target (METRICS.md section 2). Until the
    long-context prompt corpus (``ob-del``) is available, this generates
    repeatable synthetic text of the right approximate length.
    """
    target_chars = target_tokens * chars_per_token
    repeats = target_chars // len(_BASE_PROMPT) + 1
    text = _BASE_PROMPT * repeats
    return text[:target_chars]


# ---------------------------------------------------------------------------
# Timing data + single-repeat execution
# ---------------------------------------------------------------------------


@dataclass
class RepeatTimings:
    """Raw timing data from one measured repeat.

    Every field corresponds to a specific timing boundary in
    ``docs/METRICS.md`` sections 1-5. The harness converts these to
    ``ResultRow`` objects; the timing data itself is never written to disk.
    """

    prompt_token_count: int
    # Durations (seconds, from a monotonic clock per METRICS.md section 1)
    prefill_duration: float  # t_prefill_logits - t_prefill_start (section 2)
    ttft_duration: float  # t_first_token - t_submit (section 3)
    decode_duration: float  # t_N - t_decode_start (section 4)
    decode_token_count: int  # N - 1 (section 4: token 1 excluded)
    # Memory at phase-end sampling instants (METRICS.md section 5.1)
    mem_prefill: dict[str, int]
    mem_decode: dict[str, int]


def run_one_repeat(backend: Backend, prompt_text: str, decode_length: int) -> RepeatTimings:
    """Execute one full prefill + decode cycle, collecting timing per METRICS.md.

    The caller is responsible for warmup vs. measured distinction: this function
    always runs the full protocol and returns timing data. Warmup repeats should
    simply discard the return value (METRICS.md section 7).
    """
    backend.reset()

    # t_submit: prompt handed to harness, model loaded and warm (METRICS.md section 1)
    t_submit = time.perf_counter()

    input_ids = backend.tokenize(prompt_text)
    prompt_token_count = len(input_ids)

    # t_prefill_start: prefill forward pass invoked, AFTER tokenization (section 2)
    t_prefill_start = time.perf_counter()
    logits = backend.prefill(input_ids)
    # t_prefill_logits: forward pass complete, logits materialized, BEFORE sampling
    t_prefill_logits = time.perf_counter()

    # Sample peak memory at prefill-end (METRICS.md section 5.1)
    mem_prefill = backend.memory_bytes()

    # Token 1: sampled from prefill's last-position logits (METRICS.md section 1)
    token1 = backend.sample(logits)
    # t_first_token: token 1 id produced, AFTER sampling
    t_first_token = time.perf_counter()

    # Decode: tokens 2..N (METRICS.md section 4). EOS does not stop early.
    t_decode_start = t_first_token
    current_token = token1
    for _ in range(decode_length - 1):
        current_token = backend.decode_step(current_token)
    t_n = time.perf_counter()

    # Sample peak memory at decode-end (METRICS.md section 5.1)
    mem_decode = backend.memory_bytes()

    return RepeatTimings(
        prompt_token_count=prompt_token_count,
        prefill_duration=t_prefill_logits - t_prefill_start,
        ttft_duration=t_first_token - t_submit,
        decode_duration=t_n - t_decode_start,
        decode_token_count=decode_length - 1,
        mem_prefill=mem_prefill,
        mem_decode=mem_decode,
    )


# ---------------------------------------------------------------------------
# Convert timing data to schema-conformant rows
# ---------------------------------------------------------------------------

_METRIC_LABELS = {
    "prefill_tokens_per_sec": "tokens_per_sec",
    "decode_tokens_per_sec": "tokens_per_sec",
    "ttft_seconds": "seconds",
    "peak_memory_bytes": "bytes",
}

_MEM_COMPONENTS = ("weights", "kv_cache", "recurrent_state")


def _rows_from_timing(
    timing: RepeatTimings,
    *,
    run_id: str,
    git_sha: str,
    manifest_ref_str: str,
    config: SweepConfig,
    context_length: int,
    repeat_idx: int,
    notes: str | None = None,
) -> list[ResultRow]:
    """Convert one repeat's timing data to a list of schema-conformant rows.

    Produces 9 rows per repeat: 3 throughput/latency metrics + 6 memory rows
    (3 components x 2 phases). Each row is one measurement in the tidy/long
    format (RESULTS_SCHEMA.md section 1).
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[ResultRow] = []

    def _row(
        metric_name: str,
        phase: str,
        value: float,
        metric_component: str | None = None,
    ) -> ResultRow:
        return ResultRow(
            run_id=run_id,
            timestamp=ts,
            git_sha=git_sha,
            manifest_ref=manifest_ref_str,
            device=config.device,
            engine_gdn=config.engine_gdn,
            engine_full_attention=config.engine_full_attention,
            model_checkpoint=config.model_checkpoint,
            quantization=config.quantization,
            context_length=context_length,
            phase=phase,
            metric_name=metric_name,
            metric_component=metric_component,
            value=value,
            unit=_METRIC_LABELS[metric_name],
            repeat_index=repeat_idx,
            repeat_count=config.repeat_count,
            layer_class=LayerClass.ALL.value,
            notes=config.notes if notes is None else notes,
        )

    # Prefill throughput (METRICS.md section 2)
    prefill_tps = (
        timing.prompt_token_count / timing.prefill_duration if timing.prefill_duration > 0 else 0.0
    )
    rows.append(_row("prefill_tokens_per_sec", "prefill", prefill_tps))

    # TTFT (METRICS.md section 3)
    rows.append(_row("ttft_seconds", "prefill", timing.ttft_duration))

    # Decode throughput (METRICS.md section 4)
    decode_tps = (
        timing.decode_token_count / timing.decode_duration if timing.decode_duration > 0 else 0.0
    )
    rows.append(_row("decode_tokens_per_sec", "decode", decode_tps))

    # Memory: prefill phase, 3 components (METRICS.md section 5)
    for comp in _MEM_COMPONENTS:
        rows.append(
            _row(
                "peak_memory_bytes",
                "prefill",
                float(timing.mem_prefill[comp]),
                comp,
            )
        )

    # Memory: decode phase, 3 components
    for comp in _MEM_COMPONENTS:
        rows.append(
            _row(
                "peak_memory_bytes",
                "decode",
                float(timing.mem_decode[comp]),
                comp,
            )
        )

    return rows


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------


def _git_short_sha() -> str | None:
    """Short git SHA for the run_id convention, or None outside a git repo.

    Returns ``None`` rather than a placeholder on purpose. The frozen schema
    validates ``git_sha`` as 7–40 lowercase hex, so any stand-in value that
    parses (``"0000000"``) would let a CSV with no real provenance validate
    clean and look publishable — defeating the rule in PLAN.md §9 that a number
    without a manifest is not a result. Callers must decide explicitly.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def run_sweep(backend: Backend, config: SweepConfig) -> list[ResultRow]:
    """Run the full context sweep, returning validated schema-conformant rows.

    Each context length is independent: if one fails (OOM, timeout), its error
    is reported on stderr and the sweep continues to the next point, so a
    truncated sweep still yields publishable data (PLAN.md section 5/R4).
    """
    backend.load()

    git_sha = _git_short_sha()
    notes = config.notes
    if git_sha is None:
        if not config.allow_missing_sha:
            raise RuntimeError(
                "cannot resolve a git SHA: 'git rev-parse --short HEAD' failed, so this "
                "run would be un-attributable. The frozen schema cannot represent "
                "'unknown', so rather than stamp a placeholder that validates clean, the "
                "sweep refuses. Run from inside the repo, or pass --allow-missing-sha to "
                "record the run as explicitly un-attributable."
            )
        # Schema requires 7-40 hex, so a placeholder is unavoidable here — but mark it
        # in notes so the CSV itself carries the caveat rather than looking publishable.
        git_sha = "0000000"
        marker = "UNATTRIBUTABLE: no git SHA available; git_sha is a placeholder"
        notes = f"{notes}; {marker}" if notes else marker

    now = datetime.now(timezone.utc)
    run_id = f"{config.device}_{now.strftime('%Y%m%dT%H%M%SZ')}_{git_sha}"
    mref = manifest_ref(run_id, results_dir=config.manifest_dir)

    all_rows: list[ResultRow] = []

    for ctx_len in config.context_lengths:
        try:
            if config.prompt_type in ("needle", "ruler"):
                prompt_text = load_corpus_prompt(config.prompt_type, ctx_len, config.prompt_dir)
            else:
                prompt_text = generate_prompt(ctx_len)

            # Warmup repeats: full protocol, discarded, never written (METRICS.md section 7)
            for _ in range(config.warmup_count):
                run_one_repeat(backend, prompt_text, config.decode_length)

            # Measured repeats
            for repeat_idx in range(config.repeat_count):
                timing = run_one_repeat(backend, prompt_text, config.decode_length)
                rows = _rows_from_timing(
                    timing,
                    run_id=run_id,
                    git_sha=git_sha,
                    manifest_ref_str=mref,
                    config=config,
                    context_length=ctx_len,
                    repeat_idx=repeat_idx,
                    notes=notes,
                )
                all_rows.extend(rows)
        except Exception as exc:
            print(
                f"  WARNING: context_length={ctx_len} failed ({exc}); continuing to next point",
                file=sys.stderr,
            )

    # Validate every row before returning (RESULTS_SCHEMA.md contract)
    if all_rows:
        validate_rows(all_rows)

    return all_rows


# ---------------------------------------------------------------------------
# Summary computation + human-readable report
# ---------------------------------------------------------------------------


@dataclass
class MetricSummary:
    """One (context_length, phase, metric, component) group's statistics."""

    context_length: int
    phase: str
    metric_name: str
    metric_component: str
    summary: Summary


def compute_summaries(rows: list[ResultRow]) -> list[MetricSummary]:
    """Group rows by (context_length, phase, metric_name, metric_component)
    and compute p50/p95 summaries (METRICS.md section 7)."""
    groups: dict[tuple[int, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row.context_length, row.phase, row.metric_name, row.metric_component or "")
        groups[key].append(row.value)

    results: list[MetricSummary] = []
    for (ctx, phase, metric, comp), values in sorted(groups.items()):
        results.append(
            MetricSummary(
                context_length=ctx,
                phase=phase,
                metric_name=metric,
                metric_component=comp,
                summary=summarize(values),
            )
        )
    return results


def _fmt_bytes(n: float) -> str:
    """Format a byte count as KiB/MiB/GiB for human readability."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


def _fmt_value(metric_name: str, value: float) -> str:
    if metric_name.endswith("_per_sec"):
        return f"{value:>12.2f}"
    if metric_name == "ttft_seconds":
        return f"{value * 1000:>9.2f}ms"
    if metric_name == "peak_memory_bytes":
        return f"{_fmt_bytes(value):>12}"
    return f"{value:>12.4g}"


def print_summary(summaries: list[MetricSummary], config: SweepConfig) -> None:
    """Print a human-readable p50/p95 report to stdout.

    This is the eyeball view: look at it before trusting the CSV. The CSV
    contains per-repeat rows; this report aggregates them into percentiles.
    """
    print()
    print("=" * 80)
    print(f"  Benchmark sweep: {config.model_checkpoint}")
    print(
        f"  device={config.device}  engine_gdn={config.engine_gdn}  "
        f"engine_fa={config.engine_full_attention}"
    )
    print(
        f"  quantization={config.quantization}  warmup={config.warmup_count}  "
        f"repeats={config.repeat_count}  decode_len={config.decode_length}"
    )
    print("=" * 80)

    current_ctx = None
    for ms in summaries:
        if ms.context_length != current_ctx:
            current_ctx = ms.context_length
            print(f"\n  context_length = {ms.context_length:,}")

        label = ms.metric_name
        if ms.metric_component:
            label += f" [{ms.metric_component}]"
        s = ms.summary
        spread_pct = s.normalized_spread * 100 if s.normalized_spread != math.inf else math.inf

        print(
            f"    {ms.phase:>7}  {label:<45}  "
            f"p50 {_fmt_value(ms.metric_name, s.p50)}  "
            f"p95 {_fmt_value(ms.metric_name, s.p95)}  "
            f"spread {spread_pct:.1f}%"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Derived from the frozen schema rather than restated. These were hardcoded
# literal lists duplicating schema.Device/schema.Engine; they happened to agree,
# but nothing kept them in step, so a schema change would have silently left the
# CLI accepting a value the CSV validator then rejects at the end of a sweep.
_ENGINE_CHOICES = [e.value for e in Engine]
_DEVICE_CHOICES = [d.value for d in Device]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench.harness",
        description=(
            "GDN benchmark harness -- context sweep with warmup, repeats, and percentiles (ob-ljh)."
        ),
    )
    parser.add_argument(
        "--backend",
        default="synthetic",
        choices=["synthetic", "hf"],
        help="Inference backend (default: synthetic). 'hf' runs a real Qwen3.5 via "
        "HuggingFace transformers and requires torch (see bench/hf_backend.py).",
    )
    parser.add_argument(
        "--model",
        default="4b",
        choices=list(_MODEL_PRESETS),
        help="Model checkpoint preset (default: 4b)",
    )
    parser.add_argument(
        "--context-lengths",
        default="4096,32768,131072,262144",
        help="Comma-separated context lengths to sweep (default: 4K,32K,128K,262K)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup repeats, discarded (METRICS.md section 7, default: 3)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Measured repeats per context length (default: 10)",
    )
    parser.add_argument(
        "--decode-length",
        type=int,
        default=257,
        help="Total generation length: 1 prefill + N-1 decode tokens (default: 257)",
    )
    parser.add_argument("--device", default="generic_aarch64", choices=_DEVICE_CHOICES)
    parser.add_argument("--engine-gdn", default="cpu", choices=_ENGINE_CHOICES)
    parser.add_argument("--engine-full-attention", default="cpu", choices=_ENGINE_CHOICES)
    parser.add_argument("--model-checkpoint", default="", help="Override model checkpoint string")
    parser.add_argument("--quantization", default="fp16")
    parser.add_argument(
        "--output",
        "-o",
        default="",
        help="CSV output path (default: results/raw/<run_id>.csv)",
    )
    parser.add_argument("--manifest-dir", default="results/manifests")
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Don't write CSV or manifest, just print the summary",
    )
    parser.add_argument("--notes", default="", help="Notes for every row")
    parser.add_argument(
        "--allow-missing-sha",
        action="store_true",
        help="Permit a run outside a git repo. The CSV's git_sha becomes a placeholder "
        "and every row is stamped UNATTRIBUTABLE in notes. Never use for published results.",
    )
    parser.add_argument(
        "--prompt-type",
        default="synthetic",
        choices=["synthetic", "needle", "ruler"],
        help="Prompt source: synthetic (default), needle (haystack), or ruler (multi-key)",
    )
    parser.add_argument(
        "--prompt-dir", default="bench/prompts", help="Directory for committed prompt corpus"
    )
    # Synthetic-backend timing simulation (for demos only; zero by default)
    parser.add_argument(
        "--prefill-ns-per-token",
        type=int,
        default=0,
        help="Synthetic backend: nanoseconds of busy-wait per prefill token (default: 0)",
    )
    parser.add_argument(
        "--decode-ns-per-step",
        type=int,
        default=0,
        help="Synthetic backend: nanoseconds of busy-wait per decode step (default: 0)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate minimum repeat count (METRICS.md section 7: never N < 5)
    if args.repeats < 5:
        parser.error(
            f"--repeats must be >= 5 (METRICS.md section 7: "
            f"'never report N < 5'), got {args.repeats}"
        )

    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    model_cfg = _MODEL_PRESETS[args.model]

    # Build backend
    if args.backend == "synthetic":
        backend: Backend = SyntheticBackend(
            model_cfg,
            prefill_ns_per_token=args.prefill_ns_per_token,
            decode_ns_per_step=args.decode_ns_per_step,
        )
    elif args.backend == "hf":
        # Imported lazily: hf_backend pulls in torch/transformers on instantiation, and
        # the synthetic path must stay usable on hosts that have neither.
        from bench.hf_backend import HFTorchBackend

        try:
            backend = HFTorchBackend(model_cfg, quantization=args.quantization)
        except ImportError as exc:
            parser.error(f"--backend hf needs torch + transformers installed: {exc}")
    else:
        parser.error(f"unknown backend: {args.backend}")
        return 1  # unreachable, keeps type checker happy

    model_checkpoint = args.model_checkpoint or f"{model_cfg.name}@{args.backend}"
    config = SweepConfig(
        context_lengths=context_lengths,
        warmup_count=args.warmup,
        repeat_count=args.repeats,
        decode_length=args.decode_length,
        device=args.device,
        engine_gdn=args.engine_gdn,
        engine_full_attention=args.engine_full_attention,
        model_checkpoint=model_checkpoint,
        quantization=args.quantization,
        manifest_dir=args.manifest_dir,
        notes=args.notes,
        prompt_type=args.prompt_type,
        prompt_dir=args.prompt_dir,
        allow_missing_sha=args.allow_missing_sha,
    )

    # Run the sweep
    rows = run_sweep(backend, config)
    summaries = compute_summaries(rows)
    print_summary(summaries, config)

    if not args.no_csv:
        run_id = rows[0].run_id if rows else "empty"
        output_csv = args.output or f"results/raw/{run_id}.csv"
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, output_csv)

        manifest = capture(
            run_id=run_id,
            backend=args.backend,
            model_checkpoint=model_checkpoint,
            quantization=args.quantization,
            decode_length=args.decode_length,
            warmup_count=args.warmup,
            repeat_count=args.repeats,
            context_lengths=context_lengths,
        )
        manifest_path = f"{config.manifest_dir}/{run_id}.json"
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        write_manifest(manifest, manifest_path)

        print(f"  CSV:      {output_csv}  ({len(rows)} rows)")
        print(f"  Manifest: {manifest_path}")

    print(f"  All {len(rows)} rows validated against frozen schema.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
