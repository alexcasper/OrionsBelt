#!/usr/bin/env python3
"""Benchmark runner CLI with context sweep, warmup, repeats, and percentiles.

Bead ``ob-ljh`` (``t-harness-core``).  This is the orchestrator that drives the
context-length sweep (4K / 32K / 128K / 262K), manages warmup repeats, times each
measurement with a monotonic clock, and emits per-repeat rows in the frozen
results schema (``bench/schema.py``).

**Design: backend-agnostic.** The harness never touches a model directly. Every
model runtime (transformers, llama.cpp, NPU, ...) implements the
:class:`BenchmarkBackend` protocol below, and the harness calls those primitives
while it controls all timer start/stop instants. This keeps the timing logic in
exactly one place and makes the harness fully portable: it is finished and tested
before any hardware arrives (PLAN.md section 1).

Timing boundaries follow ``docs/METRICS.md`` precisely:

    t_submit ────────────────────────────────────────────────►  (prompt handed in)
        │ tokenize
        t_tok_done
        │ ──── t_prefill_start ──── (prefill forward) ──── t_prefill_logits
        │                                                       │ sample token 1
        │                                                  t_first_token
        │ ◄── decode loop: token 2 … token N ──────────────► t_N

- ``prefill_tokens_per_sec`` = prompt_tokens / (t_prefill_logits − t_prefill_start)
- ``ttft_seconds``            = t_first_token − t_submit
- ``decode_tokens_per_sec``   = (N−1) / (t_N − t_first_token)
- Token 1 belongs to prefill, **not** decode (METRICS.md section 1).
- Model loading, weight deserialization, and JIT compile are excluded from every
  timer — they are process-lifetime costs, not per-request costs.

Stdlib-only (``time``, ``argparse``, ``statistics``, ``json``, ``os``) so the
harness runs on the board without any third-party dependency.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# -- Schema integration -------------------------------------------------------
# bench/schema.py and bench/manifest.py are sibling modules. Import them lazily
# so a bare ``python3 bench/harness.py --help`` works even if the PYTHONPATH
# has not been set up for ``from bench.schema import …``.

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

import manifest as _manifest  # noqa: E402
import schema as _schema  # noqa: E402  (bench/schema.py)

# ---------------------------------------------------------------------------
# Defaults — mirrors docs/METRICS.md section 7
# ---------------------------------------------------------------------------

CANONICAL_CONTEXTS: tuple[int, ...] = (4096, 32768, 131072, 262144)
DEFAULT_WARMUP = 3
DEFAULT_DECODE_TOKENS = 257  # 1 prefill token + 256 decode-phase tokens (METRICS.md §4)
DEFAULT_DEV_REPEATS = 10
DEFAULT_HEADLINE_REPEATS = 30

# At 128K/262K, 30 repeats per point is not practical (METRICS.md §7 table).
EXPENSIVE_CONTEXT_THRESHOLD = 131072
EXPENSIVE_CONTEXT_REPEATS = 10


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


@dataclass
class MemoryBreakdown:
    """Three-component memory attribution (METRICS.md section 5).

    Every value is a byte count. ``weights`` is expected flat across context
    length; ``kv_cache`` grows linearly; ``recurrent_state`` stays O(1).
    """

    weights: int
    kv_cache: int
    recurrent_state: int


class BenchmarkBackend(ABC):
    """Interface a model runtime implements so the harness can time it.

    The harness calls these primitives and manages all timer boundaries itself
    — backends must **not** do their own timing, because the exact start/stop
    instants are load-bearing (METRICS.md sections 1–4).
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights, build any runtime artefacts.

        Called once before the sweep starts. Never timed — excluded from every
        metric per METRICS.md section 1.
        """

    @abstractmethod
    def tokenize(self, text: str, max_tokens: int) -> list[int]:
        """Tokenise ``text``, truncating to at most ``max_tokens`` ids.

        Returns the actual token ids fed to prefill — ``len(result)`` is the
        numerator for ``prefill_tokens_per_sec`` (METRICS.md section 2).
        """

    @abstractmethod
    def prefill(self, input_ids: list[int]) -> None:
        """Run the prefill forward pass over the full prompt.

        Timer starts immediately before and stops immediately after this call.
        Tokenisation has already completed by this point.
        """

    @abstractmethod
    def sample_first_token(self) -> int:
        """Sample token 1 from the last position's logits.

        ``t_first_token`` is recorded immediately after this returns.
        Token 1 is excluded from the decode-token count (METRICS.md section 1).
        """

    @abstractmethod
    def decode_step(self) -> int:
        """Run one autoregressive decode step; return the produced token id.

        Called ``N−1`` times to measure decode throughput (METRICS.md section 4).
        EOS is ignored for benchmarking (METRICS.md §4 generation-length convention).
        """

    @abstractmethod
    def memory_breakdown(self, seq_len: int) -> MemoryBreakdown:
        """Return the three-component memory attribution at ``seq_len`` tokens.

        Called at the phase-end sampling instants (METRICS.md section 5.1).
        Must derive from model introspection / known tensor shapes, never from
        process RSS (METRICS.md section 5.0).
        """


# ---------------------------------------------------------------------------
# Mock backend — for CI and smoke tests
# ---------------------------------------------------------------------------


class MockBackend(BenchmarkBackend):
    """A synthetic backend that exercises the harness without a real model.

    Produces deterministic, schema-valid numbers. Each prefill/decode call does
    a small amount of CPU work (a few array multiplications) so the monotonic
    clock registers non-zero elapsed time — enough to validate timing logic and
    percentile computation without being slow.

    Memory figures are synthetic but structurally realistic: weights are flat,
    kv_cache grows linearly, recurrent_state stays constant.
    """

    def __init__(
        self,
        *,
        weights_bytes: int = 8_000_000_000,
        kv_cache_bytes_per_token: int = 1_024,
        recurrent_state_bytes: int = 50_331_648,  # 48 MiB — METRICS.md §9
        prefill_work: int = 200,
        decode_work: int = 300,
    ) -> None:
        self._weights = weights_bytes
        self._kv_per_token = kv_cache_bytes_per_token
        self._recurrent = recurrent_state_bytes
        self._prefill_work = prefill_work
        self._decode_work = decode_work
        self._seq_len = 0

    def load(self) -> None:
        pass  # instant

    def tokenize(self, text: str, max_tokens: int) -> list[int]:
        # Deterministic pseudo-tokens; capped at max_tokens.
        n = min(max_tokens, len(text.encode()) or max_tokens)
        n = max(n, 1)
        return list(range(n))

    def prefill(self, input_ids: list[int]) -> None:
        self._seq_len = len(input_ids)
        _burn(self._prefill_work * len(input_ids))

    def sample_first_token(self) -> int:
        _burn(50)
        return 0

    def decode_step(self) -> int:
        self._seq_len += 1
        _burn(self._decode_work)
        return 0

    def memory_breakdown(self, seq_len: int) -> MemoryBreakdown:
        return MemoryBreakdown(
            weights=self._weights,
            kv_cache=self._kv_per_token * seq_len,
            recurrent_state=self._recurrent,
        )


def _burn(iterations: int) -> None:
    """Do a small, variable amount of CPU work so the clock registers time.

    Uses a simple accumulator loop rather than ``time.sleep`` so the work is
    deterministic and does not interact with the scheduler.
    """
    acc = 0.0
    for i in range(iterations):
        acc += (i * 0.1) ** 0.5
    # Prevent the optimiser from removing the loop.
    if acc != acc:  # NaN check — never true, but compiler can't prove it
        raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Percentile helpers
# ---------------------------------------------------------------------------


def percentile(data: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (METRICS.md section 7).

    With N=10–30 repeats this is honest about what the sample size can resolve.
    Never extrapolates beyond observed data.
    """
    if not data:
        raise ValueError("percentile of empty sequence")
    s = sorted(data)
    n = len(s)
    if n == 1:
        return s[0]
    # Nearest-rank: rank = ceil(pct/100 * n), clamped to [1, n], 1-indexed.
    import math

    rank = max(1, min(n, math.ceil(pct / 100.0 * n)))
    return s[rank - 1]


@dataclass
class MetricSummary:
    """p50 / p95 / spread for a single metric across repeats."""

    p50: float
    p95: float
    spread: float  # p95 - p50
    n: int


def summarise(values: Sequence[float]) -> MetricSummary:
    p50 = percentile(values, 50)
    p95 = percentile(values, 95)
    return MetricSummary(p50=p50, p95=p95, spread=p95 - p50, n=len(values))


# ---------------------------------------------------------------------------
# Single-repeat measurement
# ---------------------------------------------------------------------------


@dataclass
class RepeatResult:
    """Raw measurements from one timed repeat at one context length."""

    prompt_token_count: int
    prefill_elapsed: float  # t_prefill_logits - t_prefill_start
    ttft_elapsed: float  # t_first_token - t_submit
    decode_elapsed: float  # t_N - t_first_token
    decode_token_count: int  # N - 1
    # Memory at the two phase-end sampling instants
    mem_prefill_weights: int
    mem_prefill_kv: int
    mem_prefill_recurrent: int
    mem_decode_weights: int
    mem_decode_kv: int
    mem_decode_recurrent: int


def run_one_repeat(
    backend: BenchmarkBackend,
    *,
    context_length: int,
    decode_tokens: int,
    prompt_text: str = "",
) -> RepeatResult:
    """Execute one full prefill + decode sequence and return raw timings.

    All timers use ``time.perf_counter()`` (monotonic, METRICS.md section 1).
    Model loading is assumed to have already happened.
    """
    n_decode = decode_tokens - 1  # token 1 belongs to prefill

    # -- t_submit: the instant the prompt is handed in (METRICS.md §1, §3) --
    t_submit = time.perf_counter()
    input_ids = backend.tokenize(prompt_text or "x" * context_length, context_length)

    # -- Prefill forward pass --
    t_prefill_start = time.perf_counter()
    backend.prefill(input_ids)
    t_prefill_logits = time.perf_counter()

    # -- Sample token 1 (belongs to prefill, not decode) --
    backend.sample_first_token()
    t_first_token = time.perf_counter()

    # -- Memory at prefill-end sampling instant --
    prefill_mem = backend.memory_breakdown(len(input_ids))

    # -- Decode loop: tokens 2..N --
    for _ in range(n_decode):
        backend.decode_step()
    t_n = time.perf_counter()

    # -- Memory at decode-end sampling instant --
    decode_mem = backend.memory_breakdown(len(input_ids) + n_decode)

    return RepeatResult(
        prompt_token_count=len(input_ids),
        prefill_elapsed=t_prefill_logits - t_prefill_start,
        ttft_elapsed=t_first_token - t_submit,
        decode_elapsed=t_n - t_first_token,
        decode_token_count=n_decode,
        mem_prefill_weights=prefill_mem.weights,
        mem_prefill_kv=prefill_mem.kv_cache,
        mem_prefill_recurrent=prefill_mem.recurrent_state,
        mem_decode_weights=decode_mem.weights,
        mem_decode_kv=decode_mem.kv_cache,
        mem_decode_recurrent=decode_mem.recurrent_state,
    )


# ---------------------------------------------------------------------------
# Context-point runner (warmup + measured repeats → schema-conformant rows)
# ---------------------------------------------------------------------------


def _short_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_id(device: str) -> str:
    """``<device>_<yyyymmddTHHMMSSZ>_<short_sha>`` per RESULTS_SCHEMA.md section 2."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{device}_{ts}_{_short_git_sha()}"


def repeats_for_context(context_length: int, requested: int | None) -> int:
    """Pick repeat count, honouring the METRICS.md section 7 tier table.

    At 128K/262K, 30 repeats per point is impractical; the default drops to 10.
    An explicit ``--repeats`` override always wins.
    """
    if requested is not None:
        return max(5, requested)  # never below 5 (METRICS.md §7)
    if context_length >= EXPENSIVE_CONTEXT_THRESHOLD:
        return EXPENSIVE_CONTEXT_REPEATS
    return DEFAULT_HEADLINE_REPEATS


def run_context_point(
    backend: BenchmarkBackend,
    *,
    context_length: int,
    run_id: str,
    manifest_ref: str,
    git_sha: str,
    device: str,
    engine_gdn: str,
    engine_full_attention: str,
    model_checkpoint: str,
    quantization: str,
    warmup: int,
    repeat_count: int,
    decode_tokens: int,
    prompt_text: str = "",
) -> list[_schema.ResultRow]:
    """Run warmup + measured repeats at one context length, return schema rows."""
    # -- Warmup repeats (discarded, never written) --
    for _ in range(warmup):
        run_one_repeat(backend, context_length=context_length, decode_tokens=decode_tokens, prompt_text=prompt_text)

    # -- Measured repeats --
    results: list[RepeatResult] = []
    for _ in range(repeat_count):
        results.append(
            run_one_repeat(
                backend,
                context_length=context_length,
                decode_tokens=decode_tokens,
                prompt_text=prompt_text,
            )
        )

    # -- Convert to schema rows (one row per measurement per repeat) --
    rows: list[_schema.ResultRow] = []
    ts = _iso_now()
    for i, r in enumerate(results):
        common = dict(
            run_id=run_id,
            timestamp=ts,
            git_sha=git_sha,
            manifest_ref=manifest_ref,
            device=device,
            engine_gdn=engine_gdn,
            engine_full_attention=engine_full_attention,
            model_checkpoint=model_checkpoint,
            quantization=quantization,
            context_length=context_length,
            repeat_index=i,
            repeat_count=repeat_count,
        )

        # Prefill throughput
        tps = r.prompt_token_count / r.prefill_elapsed if r.prefill_elapsed > 0 else 0.0
        rows.append(
            _schema.ResultRow(
                **common,
                phase=_schema.Phase.PREFILL.value,
                metric_name=_schema.MetricName.PREFILL_TOKENS_PER_SEC.value,
                metric_component=None,
                value=tps,
                unit=_schema.Unit.TOKENS_PER_SEC.value,
            )
        )

        # TTFT
        rows.append(
            _schema.ResultRow(
                **common,
                phase=_schema.Phase.PREFILL.value,
                metric_name=_schema.MetricName.TTFT_SECONDS.value,
                metric_component=None,
                value=r.ttft_elapsed,
                unit=_schema.Unit.SECONDS.value,
            )
        )

        # Decode throughput
        dtps = r.decode_token_count / r.decode_elapsed if r.decode_elapsed > 0 else 0.0
        rows.append(
            _schema.ResultRow(
                **common,
                phase=_schema.Phase.DECODE.value,
                metric_name=_schema.MetricName.DECODE_TOKENS_PER_SEC.value,
                metric_component=None,
                value=dtps,
                unit=_schema.Unit.TOKENS_PER_SEC.value,
            )
        )

        # Memory: 3 components × 2 phases = 6 rows
        for phase, mem_prefix in (
            (_schema.Phase.PREFILL.value, "mem_prefill"),
            (_schema.Phase.DECODE.value, "mem_decode"),
        ):
            for comp, attr in (
                (_schema.MemoryComponent.WEIGHTS.value, "weights"),
                (_schema.MemoryComponent.KV_CACHE.value, "kv"),
                (_schema.MemoryComponent.RECURRENT_STATE.value, "recurrent"),
            ):
                bytes_val = getattr(r, f"{mem_prefix}_{attr}")
                rows.append(
                    _schema.ResultRow(
                        **common,
                        phase=phase,
                        metric_name=_schema.MetricName.PEAK_MEMORY_BYTES.value,
                        metric_component=comp,
                        value=float(bytes_val),
                        unit=_schema.Unit.BYTES.value,
                    )
                )

    return rows


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    """All parameters for one benchmark sweep invocation."""

    device: str = "generic_aarch64"
    engine_gdn: str = "cpu"
    engine_full_attention: str = "cpu"
    model_checkpoint: str = "Qwen/Qwen3.5-4B"
    quantization: str = "fp16"
    contexts: tuple[int, ...] = CANONICAL_CONTEXTS
    warmup: int = DEFAULT_WARMUP
    repeats: int | None = None  # None = auto per METRICS.md §7 tiers
    decode_tokens: int = DEFAULT_DECODE_TOKENS
    output_dir: str = "results/raw"
    manifests_dir: str = "results/manifests"
    backend_name: str = "mock"
    write_manifest: bool = True
    print_summary: bool = True


@dataclass
class SweepResult:
    """Outcome of a full sweep — per-context CSV paths and aggregate summary."""

    run_id: str
    csv_paths: list[str] = field(default_factory=list)
    manifest_path: str = ""
    summaries: dict[int, dict[str, MetricSummary]] = field(default_factory=dict)


def run_sweep(
    backend: BenchmarkBackend,
    config: SweepConfig,
) -> SweepResult:
    """Run a full context-length sweep and write schema-conformant CSVs.

    Each context point is written **immediately** after it completes, so a
    truncated sweep (e.g. 262K OOMs or takes too long) still yields publishable
    data for the earlier points — "each context point must be independently
    useful" (bead ob-ljh).
    """
    backend.load()

    run_id = _run_id(config.device)
    git_sha = _short_git_sha()
    manifest_path = _manifest.manifest_ref(run_id, results_dir=config.manifests_dir)

    # Capture provenance (PLAN.md section 9: every run emits a manifest)
    if config.write_manifest:
        try:
            mdata = _manifest.capture(
                run_id=run_id,
                harness="bench/harness.py",
                backend=config.backend_name,
                device=config.device,
                engine_gdn=config.engine_gdn,
                engine_full_attention=config.engine_full_attention,
                model_checkpoint=config.model_checkpoint,
                quantization=config.quantization,
            )
            full_manifest_path = os.path.join(config.manifests_dir, f"{run_id}.json")
            _manifest.write(mdata, full_manifest_path)
        except Exception:
            pass  # manifest probe failures are non-fatal

    result = SweepResult(run_id=run_id, manifest_path=manifest_path)
    os.makedirs(config.output_dir, exist_ok=True)

    for ctx in config.contexts:
        n_reps = repeats_for_context(ctx, config.repeats)
        rows = run_context_point(
            backend,
            context_length=ctx,
            run_id=run_id,
            manifest_ref=manifest_path,
            git_sha=git_sha,
            device=config.device,
            engine_gdn=config.engine_gdn,
            engine_full_attention=config.engine_full_attention,
            model_checkpoint=config.model_checkpoint,
            quantization=config.quantization,
            warmup=config.warmup,
            repeat_count=n_reps,
            decode_tokens=config.decode_tokens,
        )

        # Validate before writing (schema.py write_csv validates by default)
        csv_path = os.path.join(config.output_dir, f"{run_id}_ctx{ctx}.csv")
        _schema.write_csv(rows, csv_path)
        result.csv_paths.append(csv_path)

        # Compute summary always (it's data, not display)
        result.summaries[ctx] = _summarise_rows(rows, n_reps)

    if config.print_summary:
        _print_summary(result)

    return result


def _summarise_rows(
    rows: list[_schema.ResultRow], repeat_count: int
) -> dict[str, MetricSummary]:
    """Group rows by (phase, metric_name, metric_component) and compute percentiles."""
    groups: dict[str, list[float]] = {}
    for row in rows:
        key = f"{row.phase}/{row.metric_name}"
        if row.metric_component:
            key += f"/{row.metric_component}"
        groups.setdefault(key, []).append(row.value)

    return {key: summarise(vals) for key, vals in groups.items()}


def _fmt_bytes(n: float) -> str:
    """Human-readable byte count for the summary table."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


def _print_summary(result: SweepResult) -> None:
    """Print a p50/p95 summary table to stdout."""
    print(f"\n{'='*72}")
    print(f"  Sweep complete — run_id: {result.run_id}")
    print(f"  Manifest: {result.manifest_path}")
    print(f"{'='*72}")

    for ctx in sorted(result.summaries):
        summaries = result.summaries[ctx]
        print(f"\n  context_length = {ctx:>7,}")
        print(f"  {'metric':<45} {'p50':>12} {'p95':>12} {'spread':>10} {'N':>4}")
        print(f"  {'-'*45} {'-'*12} {'-'*12} {'-'*10} {'-'*4}")

        for key in sorted(summaries):
            s = summaries[key]
            is_bytes = "peak_memory_bytes" in key
            if is_bytes:
                p50 = _fmt_bytes(s.p50)
                p95 = _fmt_bytes(s.p95)
                spread = _fmt_bytes(s.spread)
            else:
                p50 = f"{s.p50:.4g}"
                p95 = f"{s.p95:.4g}"
                spread = f"{s.spread:.4g}"
            print(f"  {key:<45} {p50:>12} {p95:>12} {spread:>10} {s.n:>4}")

    print(f"\n  CSV files written: {len(result.csv_paths)}")
    for p in result.csv_paths:
        print(f"    {p}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_contexts(s: str) -> tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid context list: {s!r}") from None


def build_backend(name: str, **kwargs: Any) -> BenchmarkBackend:
    """Instantiate a backend by name. Currently only 'mock' is built in."""
    if name == "mock":
        return MockBackend(**kwargs)
    if name == "hf":
        from hf_backend import HFTorchBackend
        return HFTorchBackend(**kwargs)
    raise ValueError(
        f"unknown backend {name!r}. Available: 'mock', 'hf'."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench/harness.py",
        description="Benchmark runner CLI — context sweep, warmup, repeats, percentiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Timing boundaries follow docs/METRICS.md. Output is schema-conformant CSV
(bench/schema.py). Every run emits a manifest (bench/manifest.py).

Examples:
  # Smoke test with the mock backend (fast, no model needed):
  python3 bench/harness.py --backend mock --contexts 4096 --repeats 5

  # Full headline sweep:
  python3 bench/harness.py --backend mock --device generic_aarch64 \\
      --contexts 4096,32768,131072,262144 --repeats 30
""",
    )
    parser.add_argument(
        "--backend",
        default="mock",
        help="Backend to benchmark (default: mock). Only 'mock' is built in.",
    )
    parser.add_argument("--device", default="generic_aarch64", help="Device label for the run")
    parser.add_argument(
        "--engine-gdn",
        default="cpu",
        dest="engine_gdn",
        help="Engine label for GDN layers (default: cpu)",
    )
    parser.add_argument(
        "--engine-full-attention",
        default="cpu",
        dest="engine_full_attention",
        help="Engine label for full-attention layers (default: cpu)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-4B",
        dest="model_checkpoint",
        help="Model checkpoint label (default: Qwen/Qwen3.5-4B)",
    )
    parser.add_argument("--quantization", default="fp16", help="Quantization label (default: fp16)")
    parser.add_argument(
        "--contexts",
        type=_parse_contexts,
        default=CANONICAL_CONTEXTS,
        help="Comma-separated context lengths (default: 4096,32768,131072,262144)",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help=f"Warmup repeats (default: {DEFAULT_WARMUP})")
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help=(
            "Measured repeats per context point. "
            f"Default: {DEFAULT_HEADLINE_REPEATS} for < {EXPENSIVE_CONTEXT_THRESHOLD:,}, "
            f"{EXPENSIVE_CONTEXT_REPEATS} for longer contexts (METRICS.md section 7)."
        ),
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=DEFAULT_DECODE_TOKENS,
        dest="decode_tokens",
        help=f"Total tokens to generate including token 1 (default: {DEFAULT_DECODE_TOKENS})",
    )
    parser.add_argument("--output-dir", default="results/raw", dest="output_dir", help="Directory for CSV output")
    parser.add_argument(
        "--manifests-dir",
        default="results/manifests",
        dest="manifests_dir",
        help="Directory for run manifests",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip manifest generation (testing only)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the stdout summary table",
    )

    args = parser.parse_args(argv)

    config = SweepConfig(
        device=args.device,
        engine_gdn=args.engine_gdn,
        engine_full_attention=args.engine_full_attention,
        model_checkpoint=args.model_checkpoint,
        quantization=args.quantization,
        contexts=tuple(args.contexts),
        warmup=args.warmup,
        repeats=args.repeats,
        decode_tokens=args.decode_tokens,
        output_dir=args.output_dir,
        manifests_dir=args.manifests_dir,
        backend_name=args.backend,
        write_manifest=not args.no_manifest,
        print_summary=not args.quiet,
    )

    backend = build_backend(args.backend)
    run_sweep(backend, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
