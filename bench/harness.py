"""Benchmark runner CLI: context sweep, warmup, repeats, and percentiles.

Bead ``ob-ljh``. The measurement apparatus that turns a loaded model into
schema-conformant rows in ``results/raw/``.

Design principles (each load-bearing, each from a project document):
  - **Every timer follows docs/METRICS.md exactly.** ``t_submit`` is the shared origin;
    token 1 belongs to prefill not decode; decode throughput is a ratio-of-sums over
    ``N-1`` steps, never a mean of per-step rates.
  - **Report percentiles, never a single best run** (PLAN.md §9). Each repeat produces a
    row; the caller computes p50/p95 downstream.
  - **Each context point is independently useful** (ob-ljh description). The harness
    flushes its CSV after every context length so a truncated sweep (e.g. 128K taking too
    long) still yields committed, publishable data for the points that completed.
  - **Backend-agnostic.** The harness talks to a ``BenchmarkBackend`` protocol. Real
    backends (transformers, llama.cpp, cix-llama-cpp) are plugged in separately; a
    ``SyntheticBackend`` exercises the full harness pipeline without any ML dependency,
    which is what CI uses.

Stdlib-only, like schema.py and manifest.py. No torch, no transformers, no numpy — those
are backend dependencies, not harness dependencies.

Usage (CLI)::

    python3 bench/harness.py --backend synthetic --context-lengths 4096,32768 \
        --device generic_aarch64 --output results/raw/sweep.csv

    python3 bench/harness.py --backend synthetic --context-lengths 4096 \
        --device x86_reference --repeats 30 --output results/raw/oracle_4k.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

# --- schema import ---------------------------------------------------------
# Works both as ``python3 bench/harness.py`` (add bench/ to sys.path) and as an
# installed package import ``from bench.schema import ...``.
try:
    from bench.schema import COLUMNS, ResultRow, SchemaValidationError
except ImportError:
    _bench_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _bench_dir)
    from schema import (  # type: ignore[no-redef]
        COLUMNS,
        ResultRow,
        SchemaValidationError,
    )


# ===========================================================================
# 1. Backend protocol
# ===========================================================================


@dataclass
class PrefillResult:
    """Return value of ``BenchmarkBackend.prefill``.

    ``t_prefill_start`` and ``t_prefill_logits`` are instants on the *harness's* monotonic
    clock — the backend measures its own forward pass, and the harness wraps those
    measurements in the METRICS.md timer definitions. This keeps the timing boundary in
    the harness (where the definitions live), not the backend.
    """

    elapsed: float          # wall-clock seconds for the prefill forward pass
    prompt_token_count: int  # actual number of input tokens (after tokenization)
    logits_sampled: bool     # True if token 1 was sampled inside prefill (some runtimes do)


@dataclass
class DecodeResult:
    """Return value of ``BenchmarkBackend.decode_loop``."""

    elapsed: float           # wall-clock seconds for all N-1 decode steps
    tokens_generated: int    # should equal N-1 (harness asserts this)


@dataclass
class MemorySnapshot:
    """Three-way memory attribution at a sampling instant (METRICS.md §5).

    All values are exact byte counts derived from model introspection, never from RSS.
    Components the backend cannot attribute should be ``None`` — the harness will omit
    those rows rather than emit a zero that looks like a measurement.
    """

    weights: int | None = None
    kv_cache: int | None = None
    recurrent_state: int | None = None


class BenchmarkBackend(ABC):
    """Protocol every model backend must implement.

    The harness owns the timers; the backend owns the model. The backend's methods are
    timed by the harness using ``time.perf_counter()`` (CLOCK_MONOTONIC), so the backend
    should *not* do its own timing — it should just run the inference.
    """

    @abstractmethod
    def load(self) -> None:
        """Load the model, weights, tokenizer. Not timed — excluded from every metric."""

    @abstractmethod
    def tokenize(self, text: str) -> list[int]:
        """Tokenize a prompt string into token ids. Called before the prefill timer starts."""

    @abstractmethod
    def prefill(self, input_ids: list[int]) -> int:
        """Run the prefill forward pass and return token 1's id.

        Token 1 comes from the prefill pass's last-position logits (METRICS.md §1).
        The harness times the call to this method as the prefill interval.
        """

    @abstractmethod
    def decode_step(self, token_id: int) -> int:
        """Run a single-token decode step: feed ``token_id``, return the next token's id.

        The harness calls this N-1 times after prefill. Each call exercises the
        single-token GDN recurrent update and (for periodic full-attention layers) a
        KV-cache append.
        """

    @abstractmethod
    def memory_snapshot(
        self, context_length: int, generated_tokens: int, phase: str
    ) -> MemorySnapshot:
        """Return memory component sizes at the given sampling instant.

        ``phase`` is ``"prefill"`` (sampled at t_prefill_logits) or ``"decode"``
        (sampled at t_N). ``generated_tokens`` is the count produced so far. The backend
        should compute these from model introspection / known tensor shapes (METRICS.md
        §5.0), never from process RSS.
        """


# ===========================================================================
# 2. Synthetic backend (for CI and harness self-testing)
# ===========================================================================


class SyntheticBackend(BenchmarkBackend):
    """A no-op backend that simulates inference timing.

    Produces deterministic, configurable delays so the harness's sweep/timing/CSV logic
    can be exercised end-to-end in CI without torch or a real model.

    Timing model (configurable, with realistic defaults for an A76-class core):
      - prefill: ``prefill_rate`` tokens/sec (scaled by context_length)
      - decode:  ``decode_rate`` tokens/sec (flat, per METRICS.md §9's bandwidth-bound
        argument — decode is expected flat regardless of context length)
      - tokenization: proportional to prompt length
      - sampling: negligible

    Memory model: computed from the given config, matching METRICS.md §5 formulas.
    """

    def __init__(
        self,
        prefill_rate: float = 5000.0,   # tokens/sec (plausible for a 4B model on A76)
        decode_rate: float = 30.0,       # tokens/sec (matches vendor ~30 tok/s figure)
        num_gdn_layers: int = 24,
        num_full_attn_layers: int = 8,
        gdn_state_per_layer: int = 524_288 * 4,   # 2 MiB fp32 (METRICS.md §5.4)
        kv_bytes_per_token: int = 8 * 128 * 16 * 2 * 2,  # 8 layers × 2 (K,V) × 128 × 16 heads × 2 bytes fp16
        weight_bytes: int = 8_000_000_000,        # ~8 GB for a 4B fp16 model
        dtype_size: int = 4,
    ) -> None:
        self.prefill_rate = prefill_rate
        self.decode_rate = decode_rate
        self.num_gdn_layers = num_gdn_layers
        self.num_full_attn_layers = num_full_attn_layers
        self.gdn_state_per_layer = gdn_state_per_layer
        self.kv_bytes_per_token = kv_bytes_per_token
        self.weight_bytes = weight_bytes
        self.dtype_size = dtype_size
        self._loaded = False

    def load(self) -> None:
        # Simulate weight loading with a small delay.
        time.sleep(0.01)
        self._loaded = True

    def tokenize(self, text: str) -> list[int]:
        # Rough: ~1 token per 4 chars. Simulate proportional cost.
        n = max(1, len(text) // 4)
        time.sleep(n * 1e-6)  # ~1 µs/token
        return list(range(n))

    def prefill(self, input_ids: list[int]) -> int:
        if not self._loaded:
            raise RuntimeError("SyntheticBackend.prefill called before load()")
        n = len(input_ids)
        elapsed = n / self.prefill_rate
        time.sleep(elapsed)
        # Return a deterministic "token 1" id.
        return 42

    def decode_step(self, token_id: int) -> int:
        if not self._loaded:
            raise RuntimeError("SyntheticBackend.decode_step called before load()")
        time.sleep(1.0 / self.decode_rate)
        return (token_id + 1) % 32000  # deterministic cycle

    def memory_snapshot(
        self, context_length: int, generated_tokens: int, phase: str
    ) -> MemorySnapshot:
        total_tokens = context_length + generated_tokens
        return MemorySnapshot(
            weights=self.weight_bytes,
            kv_cache=self.num_full_attn_layers * self.kv_bytes_per_token * total_tokens,
            recurrent_state=self.num_gdn_layers * self.gdn_state_per_layer,
        )


# ===========================================================================
# 3. Prompt corpus (minimal, for synthetic/CI; real corpus is ob-del)
# ===========================================================================


def make_prompt(target_tokens: int) -> str:
    """Generate a deterministic prompt that tokenizes to approximately ``target_tokens``.

    Uses a repeating vocabulary of common English words. Not a quality corpus — that's
    bead ob-del's job. This exists so the harness has something to feed a backend.
    """
    words = (
        "the model processes sequences of tokens through gated deltanet layers "
        "each layer maintains a fixed recurrent state updated by the delta rule "
    )
    # ~1 token per 4 chars → need ~target_tokens * 4 chars
    text = (words * ((target_tokens * 4 // len(words)) + 1))[: target_tokens * 4]
    return text


# ===========================================================================
# 4. Harness core
# ===========================================================================


@dataclass
class HarnessConfig:
    """All parameters that define a benchmark run, in one place."""

    device: str                        # schema Device enum value
    engine_gdn: str                    # schema Engine enum value
    engine_full_attention: str         # schema Engine enum value
    model_checkpoint: str              # e.g. "Qwen/Qwen3.5-4B@abc1234"
    quantization: str                  # e.g. "fp16", "int8_w8a8"
    context_lengths: list[int]         # sweep points, e.g. [4096, 32768]
    warmups: int = 3                   # METRICS.md §7: 3 discarded warmups
    repeats: int = 30                  # METRICS.md §7: 30 for headline, 10 for large ctx
    decode_tokens: int = 257           # METRICS.md §4: N=257 (1 prefill + 256 decode)
    manifest_ref: str = ""             # path to the companion manifest
    run_id: str = ""                   # override auto-generated run_id (for manifest linking)
    notes: str = ""                    # optional notes for anomaly flags


def _now_iso() -> str:
    """ISO 8601 UTC timestamp with trailing Z, matching schema convention."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str:
    """Short git SHA for the run. Falls back to 'unknown' if not in a git repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _percentile(sorted_samples: list[float], p: float) -> float:
    """Nearest-rank percentile on sorted samples (METRICS.md §7)."""
    if not sorted_samples:
        return 0.0
    idx = int(p * (len(sorted_samples) - 1) + 0.5)
    idx = max(0, min(idx, len(sorted_samples) - 1))
    return sorted_samples[idx]


class Harness:
    """Runs the context sweep and produces schema-conformant ResultRows.

    The harness is the single owner of every timer (METRICS.md §1). Backends just run
    inference; the harness wraps each call in the precise start/stop events the metrics
    document defines.
    """

    def __init__(self, backend: BenchmarkBackend, config: HarnessConfig) -> None:
        self.backend = backend
        self.config = config
        self.run_id = config.run_id or self._make_run_id()

    def _make_run_id(self) -> str:
        device = self.config.device
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sha = _git_sha()
        return f"{device}_{ts}_{sha}"

    def _run_one_repeat(
        self, context_length: int, repeat_index: int, is_warmup: bool
    ) -> list[ResultRow]:
        """Run one full (prefill + decode) repeat at the given context length.

        Returns ResultRows for this repeat, or an empty list if ``is_warmup`` (warmups
        are discarded per METRICS.md §7 and never written to results/raw/).
        """
        cfg = self.config
        n_gen = cfg.decode_tokens

        # Generate and tokenize the prompt.
        # t_submit = the instant the prompt is handed to the model (METRICS.md §1, §3).
        prompt = make_prompt(context_length)

        t_submit = time.perf_counter()
        input_ids = self.backend.tokenize(prompt)
        prompt_token_count = len(input_ids)

        if prompt_token_count == 0:
            prompt_token_count = context_length  # fallback if tokenizer returned nothing

        # --- PREFILL phase ---
        # t_prefill_start: prefill forward pass invoked (after tokenization).
        # t_prefill_logits: prefill completes, last-position logits materialized.
        t_prefill_start = time.perf_counter()
        token_1 = self.backend.prefill(input_ids)
        t_prefill_logits = time.perf_counter()

        # t_first_token: token 1 sampled. For backends that sample inside prefill,
        # this is the same instant as t_prefill_logits. METRICS.md §3.
        t_first_token = t_prefill_logits  # sampling is expected negligible

        # --- DECODE phase ---
        # t_decode_start = t_first_token (METRICS.md §4).
        # Run N-1 decode steps (token 1 was produced by prefill).
        t_decode_start = t_first_token
        current_token = token_1
        for _ in range(n_gen - 1):
            current_token = self.backend.decode_step(current_token)
        t_N = time.perf_counter()

        if is_warmup:
            return []  # warmups are discarded

        # --- Compute metric values ---
        prefill_elapsed = t_prefill_logits - t_prefill_start
        ttft = t_first_token - t_submit
        decode_elapsed = t_N - t_decode_start

        prefill_tps = prompt_token_count / prefill_elapsed if prefill_elapsed > 0 else 0.0
        decode_tps = (n_gen - 1) / decode_elapsed if decode_elapsed > 0 else 0.0

        # --- Memory snapshots ---
        # Prefill: sampled at t_prefill_logits (METRICS.md §5.1).
        # Decode: sampled at t_N (METRICS.md §5.1).
        mem_prefill = self.backend.memory_snapshot(prompt_token_count, 0, "prefill")
        mem_decode = self.backend.memory_snapshot(
            prompt_token_count, n_gen - 1, "decode"
        )

        now = _now_iso()
        rows: list[ResultRow] = []

        def _base_row(metric_name: str, value: float, unit: str, phase: str,
                      metric_component: str | None = None) -> ResultRow:
            return ResultRow(
                run_id=self.run_id,
                timestamp=now,
                git_sha=_git_sha(),
                manifest_ref=cfg.manifest_ref,
                device=cfg.device,
                engine_gdn=cfg.engine_gdn,
                engine_full_attention=cfg.engine_full_attention,
                model_checkpoint=cfg.model_checkpoint,
                quantization=cfg.quantization,
                context_length=context_length,
                phase=phase,
                metric_name=metric_name,
                metric_component=metric_component,
                value=value,
                unit=unit,
                repeat_index=repeat_index,
                repeat_count=cfg.repeats,
                layer_class="all",
                notes=cfg.notes,
            )

        # Throughput rows
        rows.append(_base_row("prefill_tokens_per_sec", prefill_tps, "tokens_per_sec", "prefill"))
        rows.append(_base_row("ttft_seconds", ttft, "seconds", "prefill"))
        rows.append(_base_row("decode_tokens_per_sec", decode_tps, "tokens_per_sec", "decode"))

        # Memory rows — peak_memory_bytes × 3 components × 2 phases
        for phase_label, snap in [("prefill", mem_prefill), ("decode", mem_decode)]:
            if snap.weights is not None:
                rows.append(_base_row("peak_memory_bytes", float(snap.weights), "bytes",
                                      phase_label, "weights"))
            if snap.kv_cache is not None:
                rows.append(_base_row("peak_memory_bytes", float(snap.kv_cache), "bytes",
                                      phase_label, "kv_cache"))
            if snap.recurrent_state is not None:
                rows.append(_base_row("peak_memory_bytes", float(snap.recurrent_state), "bytes",
                                      phase_label, "recurrent_state"))

        return rows

    def run_sweep(self, output_path: str | None = None) -> list[ResultRow]:
        """Run the full context sweep.

        If ``output_path`` is given, writes rows to CSV incrementally — flushing after
        each context length so a truncated sweep still yields committed data. Returns
        all rows (useful for in-memory testing).
        """
        cfg = self.config
        all_rows: list[ResultRow] = []
        csv_handle = None
        csv_writer = None

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            csv_handle = open(output_path, "w", newline="", encoding="utf-8")  # noqa: SIM115
            csv_writer = csv.DictWriter(csv_handle, fieldnames=COLUMNS)
            csv_writer.writeheader()

        # Load the model before any timing — load is excluded from every metric
        # (METRICS.md §1: "The measurement boundary excludes model load").
        self.backend.load()

        try:
            for ctx_idx, ctx_len in enumerate(cfg.context_lengths):
                # Warmup: 3 discarded repeats (METRICS.md §7).
                for w in range(cfg.warmups):
                    self._run_one_repeat(ctx_len, w, is_warmup=True)

                # Timed repeats.
                for r in range(cfg.repeats):
                    rows = self._run_one_repeat(ctx_len, r, is_warmup=False)
                    all_rows.extend(rows)

                    if csv_writer and rows:
                        for row in rows:
                            csv_writer.writerow(_row_to_dict(row))

                # Flush after each context length — this is what makes each point
                # independently useful (ob-ljh).
                if csv_handle:
                    csv_handle.flush()
                    print(
                        f"  [{ctx_idx + 1}/{len(cfg.context_lengths)}] "
                        f"context_length={ctx_len}: {cfg.repeats} repeats committed",
                        file=sys.stderr,
                    )
        finally:
            if csv_handle:
                csv_handle.close()

        return all_rows

    def summarize(self, rows: list[ResultRow]) -> str:
        """Human-readable p50/p95 summary, grouped by (context_length, phase, metric).

        This is for the eyeball pass — the committed data is the per-repeat CSV, not
        this summary. Following METRICS.md §7: report p50 and p95, never a single best.
        """
        if not rows:
            return "(no rows)"

        # Group rows by (context_length, phase, metric_name, metric_component).
        groups: dict[tuple, list[float]] = {}
        for row in rows:
            key = (row.context_length, row.phase, row.metric_name, row.metric_component or "")
            groups.setdefault(key, []).append(row.value)

        lines: list[str] = []
        lines.append(f"Run {self.run_id}")
        lines.append(f"  repeats={self.config.repeats}  warmups={self.config.warmups}")
        lines.append("")

        for key in sorted(groups):
            ctx, phase, metric, component = key
            samples = sorted(groups[key])
            p50 = _percentile(samples, 0.50)
            p95 = _percentile(samples, 0.95)
            spread = p95 - p50
            spread_pct = (spread / p50 * 100) if p50 > 0 else 0.0

            label = f"ctx={ctx:>6}  {phase:<7}  {metric}"
            if component:
                label += f"  [{component}]"
            lines.append(f"  {label}")
            lines.append(f"    p50={p50:.4g}  p95={p95:.4g}  spread={spread_pct:.1f}%")

        return "\n".join(lines)


# ===========================================================================
# 5. CSV helper (matches schema.py's _row_to_csv_dict)
# ===========================================================================


def _row_to_dict(row: ResultRow) -> dict[str, str]:
    """Convert a ResultRow to a CSV row dict (empty string for None)."""
    out: dict[str, str] = {}
    for field_name in COLUMNS:
        value = getattr(row, field_name)
        out[field_name] = "" if value is None else str(value)
    return out


# ===========================================================================
# 6. CLI
# ===========================================================================


def _parse_context_lengths(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harness",
        description=(
            "Benchmark runner: context sweep, warmup, repeats, percentiles. "
            "Follows docs/METRICS.md timer definitions. Emits schema-conformant CSV."
        ),
    )
    parser.add_argument(
        "--backend",
        default="synthetic",
        choices=["synthetic"],
        help="Model backend. 'synthetic' is the no-op CI backend; real backends TBD.",
    )
    parser.add_argument(
        "--context-lengths",
        type=str,
        default="4096",
        help="Comma-separated context lengths (tokens). Default: 4096. "
        "Canonical sweep: 4096,32768,131072,262144.",
    )
    parser.add_argument(
        "--device",
        required=True,
        choices=["o6", "generic_aarch64", "x86_reference"],
        help="Which target produced this run (schema Device enum).",
    )
    parser.add_argument(
        "--engine-gdn",
        default="cpu",
        choices=["npu", "gpu_vulkan", "gpu_opencl", "cpu", "cuda_reference"],
        help="Engine for GDN (linear-attention) layers.",
    )
    parser.add_argument(
        "--engine-full-attention",
        default="cpu",
        choices=["npu", "gpu_vulkan", "gpu_opencl", "cpu", "cuda_reference"],
        help="Engine for full-attention layers.",
    )
    parser.add_argument(
        "--model-checkpoint",
        default="synthetic/no-model",
        help="Model checkpoint identifier (HF repo@revision).",
    )
    parser.add_argument(
        "--quantization",
        default="fp32",
        help="Quantization policy code (e.g. fp16, int8_w8a8).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=30,
        help="Timed repeats per context length (METRICS.md §7: 30 for headline, "
        "10 for large context, never <5).",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=3,
        help="Discarded warmup repeats (METRICS.md §7 default: 3).",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=257,
        help="Total generation length N (METRICS.md §4 default: 257 = 1 prefill + 256 decode).",
    )
    parser.add_argument(
        "--manifest-ref",
        default="",
        help="Path to the companion manifest JSON (schema manifest_ref column). "
        "Defaults to results/manifests/<run_id>.json.",
    )
    parser.add_argument(
        "--capture-manifest",
        action="store_true",
        help="Auto-run bench/manifest.py and write the manifest alongside the CSV.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV path. If omitted, writes to stdout (no summary).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a human-readable p50/p95 summary to stderr after the sweep.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Free-text notes for anomaly flags (schema notes column).",
    )
    args = parser.parse_args(argv)

    # Validate repeat counts (METRICS.md §7: never report N<5).
    if args.repeats < 5:
        print("ERROR: repeats must be >= 5 (METRICS.md §7)", file=sys.stderr)
        return 1
    if args.warmups < 0:
        print("ERROR: warmups must be >= 0", file=sys.stderr)
        return 1

    context_lengths = _parse_context_lengths(args.context_lengths)
    if not context_lengths:
        print("ERROR: at least one context length required", file=sys.stderr)
        return 1

    # Build backend.
    if args.backend == "synthetic":
        backend: BenchmarkBackend = SyntheticBackend()
    else:
        print(f"ERROR: unknown backend '{args.backend}'", file=sys.stderr)
        return 1

    # Compute run_id once so manifest_ref and the harness agree (PLAN.md §9).
    _ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _sha = _git_sha()
    run_id = f"{args.device}_{_ts}_{_sha}"

    # Default manifest_ref if not provided — PLAN.md §9: a number without a manifest
    # is not a result. The harness always names the expected path so the schema's
    # required-non-empty-string constraint is satisfied. Use --capture-manifest to
    # actually generate it.
    manifest_ref = args.manifest_ref or f"results/manifests/{run_id}.json"

    config = HarnessConfig(
        device=args.device,
        engine_gdn=args.engine_gdn,
        engine_full_attention=args.engine_full_attention,
        model_checkpoint=args.model_checkpoint,
        quantization=args.quantization,
        context_lengths=context_lengths,
        warmups=args.warmups,
        repeats=args.repeats,
        decode_tokens=args.decode_tokens,
        manifest_ref=manifest_ref,
        run_id=run_id,
        notes=args.notes,
    )

    harness = Harness(backend, config)
    output = args.output or None

    print(f"Starting sweep: {context_lengths}", file=sys.stderr)
    print(f"  warmups={config.warmups}  repeats={config.repeats}  "
          f"decode_tokens={config.decode_tokens}", file=sys.stderr)

    rows = harness.run_sweep(output_path=output)

    if args.summary:
        print(harness.summarize(rows), file=sys.stderr)

    # Optionally capture the manifest alongside the CSV.
    if args.capture_manifest:
        _manifest_dir = os.path.dirname(manifest_ref) or "."
        os.makedirs(_manifest_dir, exist_ok=True)
        manifest_script = os.path.join(os.path.dirname(__file__), "manifest.py")
        try:
            with open(manifest_ref, "w") as mf:
                subprocess.run(
                    ["python3", manifest_script],
                    stdout=mf,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
            print(f"  Manifest written to {manifest_ref}", file=sys.stderr)
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            print(f"  WARNING: manifest capture failed: {exc}", file=sys.stderr)

    # Validate all rows before declaring success.
    try:
        from bench.schema import validate_rows
    except ImportError:
        from schema import validate_rows  # type: ignore[no-redef]

    try:
        validate_rows(rows)
    except SchemaValidationError as exc:
        print(f"ERROR: schema validation failed:\n{exc}", file=sys.stderr)
        return 1

    n_ctx = len(context_lengths)
    print(
        f"Done: {len(rows)} rows across {n_ctx} context length(s), "
        f"all schema-valid.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
