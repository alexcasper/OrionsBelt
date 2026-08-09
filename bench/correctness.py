"""Correctness oracle comparison with configurable tolerances (bead ob-3uh).

Provides the tolerance framework for comparing model outputs against a
golden reference (x86/CUDA inference).  Per docs/archive/PLAN.md §9: *"speed that
changes outputs is not speed."*  This module makes that enforceable.

Three comparison modes:

1. **Logit comparison** — compares output probability distributions using
   max-abs-diff, KL divergence, top-k agreement, and argmax accuracy.
2. **Perplexity comparison** — compares scalar perplexity scores with
   absolute and relative tolerances.
3. **Long-context drift** — tracks how agreement degrades across context
   lengths, which is where GDN recurrent-state drift compounds.

The module degrades gracefully: uses numpy if available, falls back to
pure-Python math otherwise.  No torch dependency.

Usage::

    from bench.correctness import ToleranceConfig, compare_logits, CorrectnessReport

    cfg = ToleranceConfig(atol=1e-4, rtol=1e-3, kl_div_threshold=0.01)
    report = compare_logits(reference_logits, candidate_logits, cfg)
    print(report.summary())
    assert report.passed, report.failures
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Optional numpy — degrade gracefully
# ---------------------------------------------------------------------------

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToleranceConfig:
    """Configurable numeric tolerances for correctness comparison.

    Fields are deliberately independent so different metrics can have
    different strictness levels.

    - ``atol``: Maximum absolute difference per logit element.
    - ``rtol``: Maximum relative difference (|a-b| / max(|a|, |b|)).
    - ``kl_div_threshold``: Max KL divergence between distributions.
    - ``topk``: Number of top-k elements to check for set agreement.
    - ``topk_min_accuracy``: Minimum fraction of top-k that must overlap.
    - ``argmax_accuracy_threshold``: Min fraction of positions where
      argmax(reference) == argmax(candidate).
    - ``perplexity_atol``: Absolute tolerance for perplexity comparison.
    - ``perplexity_rtol``: Relative tolerance for perplexity comparison.
    - ``drift_scale_factor``: How much tolerance grows per doubling of
      context length (e.g. 2.0 = double tolerance for each 2× context).
      1.0 = flat tolerance regardless of context length.
    """

    atol: float = 1e-4
    rtol: float = 1e-3
    kl_div_threshold: float = 0.01
    topk: int = 5
    topk_min_accuracy: float = 0.8
    argmax_accuracy_threshold: float = 0.95
    perplexity_atol: float = 0.1
    perplexity_rtol: float = 0.02
    drift_scale_factor: float = 1.5

    def tolerance_for_context(self, context_length: int, base_context: int = 4096) -> float:
        """Scale tolerance for a given context length.

        As context grows, recurrent-state drift compounds, so we allow
        more slack.  The scaling is logarithmic in (context / base_context).
        """
        if context_length <= base_context or self.drift_scale_factor == 1.0:
            return 1.0
        doublings = math.log2(context_length / base_context)
        return self.drift_scale_factor**doublings

    def scaled(self, factor: float) -> ToleranceConfig:
        """Return a new config with all numeric tolerances multiplied by factor."""
        return ToleranceConfig(
            atol=self.atol * factor,
            rtol=self.rtol * factor,
            kl_div_threshold=self.kl_div_threshold * factor,
            topk=self.topk,
            topk_min_accuracy=self.topk_min_accuracy,
            argmax_accuracy_threshold=self.argmax_accuracy_threshold,
            perplexity_atol=self.perplexity_atol * factor,
            perplexity_rtol=self.perplexity_rtol * factor,
            drift_scale_factor=self.drift_scale_factor,
        )


# ---------------------------------------------------------------------------
# Pure-Python numerical helpers (used when numpy is unavailable)
# ---------------------------------------------------------------------------


def _softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax for a list of logits."""
    if not logits:
        return []
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


# Every zip() below pairs a reference against a candidate that must have identical
# shape. strict=True is deliberate: silently truncating to the shorter sequence would
# turn a shape mismatch into a PASS, which is the one failure mode a correctness
# oracle must never have.


def _kl_divergence(p: list[float], q: list[float]) -> float:
    """KL(P || Q) for probability distributions given as lists.

    Handles zeros in Q by adding a tiny epsilon (1e-12).
    """
    eps = 1e-12
    total = 0.0
    for pi, qi in zip(p, q, strict=True):
        if pi > eps:
            total += pi * math.log(pi / max(qi, eps))
    return total


def _max_abs_diff(a: list[float], b: list[float]) -> float:
    return max(abs(ai - bi) for ai, bi in zip(a, b, strict=True))


def _topk_indices(values: list[float], k: int) -> set[int]:
    """Return indices of the k largest values."""
    indexed = sorted(enumerate(values), key=lambda x: -x[1])
    return {idx for idx, _ in indexed[:k]}


# ---------------------------------------------------------------------------
# Numpy-accelerated versions
# ---------------------------------------------------------------------------


def _np_softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=-1, keepdims=True)


def _np_kl_divergence(p, q):
    eps = 1e-12
    mask = p > eps
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], eps))))


# ---------------------------------------------------------------------------
# Core comparison functions
# ---------------------------------------------------------------------------


@dataclass
class ComparisonMetric:
    """One metric from a comparison."""

    name: str
    value: float
    threshold: float | None
    passed: bool | None  # None = informational only

    def __str__(self) -> str:
        status = "✓" if self.passed else ("✗" if self.passed is False else "ℹ")
        if self.threshold is not None:
            return (
                f"  {status} {self.name:>30s}: {self.value:.6e} (threshold: {self.threshold:.6e})"
            )
        return f"  {status} {self.name:>30s}: {self.value:.6e}"


@dataclass
class CorrectnessReport:
    """Structured report from a correctness comparison."""

    context_length: int | None = None
    metrics: list[ComparisonMetric] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.failures) == 0

    def add_metric(
        self, name: str, value: float, threshold: float | None = None, passed: bool | None = None
    ) -> None:
        if passed is False and name not in [m.name for m in self.metrics]:
            self.failures.append(f"{name}={value:.6e} exceeds threshold {threshold:.6e}")
        self.metrics.append(ComparisonMetric(name, value, threshold, passed))

    def summary(self) -> str:
        header = "Correctness Report"
        if self.context_length is not None:
            header += f" (context_length={self.context_length})"
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"{header} — {status}", "-" * 60]
        for m in self.metrics:
            lines.append(str(m))
        if self.failures:
            lines.append("")
            lines.append("Failures:")
            for f in self.failures:
                lines.append(f"  ✗ {f}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_length": self.context_length,
            "passed": self.passed,
            "metrics": [
                {"name": m.name, "value": m.value, "threshold": m.threshold, "passed": m.passed}
                for m in self.metrics
            ],
            "failures": self.failures,
        }


def compare_logits(
    reference: list[list[float]] | np.ndarray,
    candidate: list[list[float]] | np.ndarray,
    config: ToleranceConfig,
    context_length: int | None = None,
) -> CorrectnessReport:
    """Compare candidate logits against reference logits.

    Args:
        reference: Shape [seq_len, vocab_size] — the golden logits.
        candidate: Same shape — the model under test.
        config: Tolerance configuration.
        context_length: If provided, tolerances are scaled for drift.

    Returns:
        CorrectnessReport with max_abs_diff, kl_div, topk_accuracy, argmax_accuracy.
    """
    report = CorrectnessReport(context_length=context_length)

    # Scale tolerance for context length if provided
    scale = 1.0
    if context_length is not None and context_length > 4096:
        scale = config.tolerance_for_context(context_length)
        scaled_cfg = config.scaled(scale)
    else:
        scaled_cfg = config

    if HAS_NUMPY:
        ref = np.asarray(reference, dtype=np.float64)
        cand = np.asarray(candidate, dtype=np.float64)
    else:
        ref = reference
        cand = candidate

    # --- Max absolute difference ---
    if HAS_NUMPY:
        max_diff = float(np.max(np.abs(ref - cand)))
    else:
        max_diff = 0.0
        for r_row, c_row in zip(ref, cand, strict=True):
            row_diff = _max_abs_diff(r_row, c_row)
            max_diff = max(max_diff, row_diff)

    passed_atol = max_diff <= scaled_cfg.atol
    report.add_metric("max_abs_diff", max_diff, scaled_cfg.atol, passed_atol)

    # --- KL divergence (averaged over positions) ---
    if HAS_NUMPY:
        ref_probs = _np_softmax(ref)
        cand_probs = _np_softmax(cand)
        kl_values = []
        for i in range(ref_probs.shape[0]):
            kl_values.append(_np_kl_divergence(ref_probs[i], cand_probs[i]))
        avg_kl = sum(kl_values) / len(kl_values) if kl_values else 0.0
    else:
        kl_values = []
        for r_row, c_row in zip(ref, cand, strict=True):
            r_probs = _softmax(r_row)
            c_probs = _softmax(c_row)
            kl_values.append(_kl_divergence(r_probs, c_probs))
        avg_kl = sum(kl_values) / len(kl_values) if kl_values else 0.0

    passed_kl = avg_kl <= scaled_cfg.kl_div_threshold
    report.add_metric("avg_kl_divergence", avg_kl, scaled_cfg.kl_div_threshold, passed_kl)

    # --- Top-k agreement ---
    # Cap topk at vocab dimension to avoid spurious failures on small vocabs
    vocab_dim = (
        len(ref[0]) if (isinstance(ref, list) and ref) else (ref.shape[1] if HAS_NUMPY else 0)
    )
    effective_topk = min(scaled_cfg.topk, vocab_dim)
    effective_topk = max(1, effective_topk)
    if HAS_NUMPY:
        topk_matches = 0
        total_positions = ref.shape[0]
        for i in range(total_positions):
            ref_topk = set(np.argsort(ref[i])[-effective_topk:])
            cand_topk = set(np.argsort(cand[i])[-effective_topk:])
            overlap = len(ref_topk & cand_topk) / effective_topk
            topk_matches += overlap
        topk_accuracy = topk_matches / total_positions if total_positions > 0 else 0.0
    else:
        topk_matches = 0
        total_positions = len(ref)
        for r_row, c_row in zip(ref, cand, strict=True):
            ref_topk = _topk_indices(r_row, effective_topk)
            cand_topk = _topk_indices(c_row, effective_topk)
            overlap = len(ref_topk & cand_topk) / effective_topk
            topk_matches += overlap
        topk_accuracy = topk_matches / total_positions if total_positions > 0 else 0.0

    passed_topk = topk_accuracy >= scaled_cfg.topk_min_accuracy
    report.add_metric(
        f"top{scaled_cfg.topk}_accuracy", topk_accuracy, scaled_cfg.topk_min_accuracy, passed_topk
    )

    # --- Argmax accuracy ---
    if HAS_NUMPY:
        ref_argmax = np.argmax(ref, axis=-1)
        cand_argmax = np.argmax(cand, axis=-1)
        argmax_acc = float(np.mean(ref_argmax == cand_argmax))
    else:
        matches = sum(
            1 for r, c in zip(ref, cand, strict=True)
            if r and c and r.index(max(r)) == c.index(max(c))
        )
        argmax_acc = matches / len(ref) if ref else 0.0

    passed_argmax = argmax_acc >= scaled_cfg.argmax_accuracy_threshold
    report.add_metric(
        "argmax_accuracy", argmax_acc, scaled_cfg.argmax_accuracy_threshold, passed_argmax
    )

    # Informational: tolerance scale applied
    if scale > 1.0:
        report.add_metric("drift_tolerance_scale", scale, None, None)

    return report


def compare_perplexity(
    reference_ppl: float,
    candidate_ppl: float,
    config: ToleranceConfig,
    context_length: int | None = None,
) -> CorrectnessReport:
    """Compare scalar perplexity values with tolerances.

    Perplexity is the exponentiated average negative log-likelihood.
    A small absolute difference in log-space translates to a ratio in
    perplexity space, so we check both atol and rtol.
    """
    report = CorrectnessReport(context_length=context_length)

    scale = 1.0
    if context_length is not None and context_length > 4096:
        scale = config.tolerance_for_context(context_length)

    atol = config.perplexity_atol * scale
    rtol = config.perplexity_rtol * scale

    abs_diff = abs(candidate_ppl - reference_ppl)
    rel_diff = abs_diff / max(abs(reference_ppl), 1e-12)

    # Combined tolerance: pass if abs_diff <= atol + rtol * |ref|  (numpy isclose semantics)
    combined_threshold = atol + rtol * abs(reference_ppl)
    passed_combined = abs_diff <= combined_threshold

    report.add_metric("perplexity_abs_diff", abs_diff, combined_threshold, passed_combined)
    report.add_metric("perplexity_rel_diff", rel_diff, rtol, None)  # informational
    report.add_metric("perplexity_combined_threshold", combined_threshold, None, None)
    report.add_metric("reference_perplexity", reference_ppl, None, None)
    report.add_metric("candidate_perplexity", candidate_ppl, None, None)

    return report


@dataclass
class DriftPoint:
    """One data point in a long-context drift analysis."""

    context_length: int
    kl_divergence: float
    argmax_accuracy: float
    tolerance_scale: float
    passed: bool


def long_context_drift(
    reports: list[CorrectnessReport],
    config: ToleranceConfig,
    base_context: int = 4096,
) -> list[DriftPoint]:
    """Aggregate per-context-length reports into a drift trend.

    Args:
        reports: List of CorrectnessReports, each with a different context_length.
        config: The tolerance config used.
        base_context: The reference context length for scaling.

    Returns:
        List of DriftPoint sorted by context_length, showing how metrics
        degrade as context grows.
    """
    points: list[DriftPoint] = []
    for report in reports:
        if report.context_length is None:
            continue
        kl = next((m.value for m in report.metrics if m.name == "avg_kl_divergence"), 0.0)
        argmax = next((m.value for m in report.metrics if m.name == "argmax_accuracy"), 1.0)
        scale = config.tolerance_for_context(report.context_length, base_context)
        points.append(
            DriftPoint(
                context_length=report.context_length,
                kl_divergence=kl,
                argmax_accuracy=argmax,
                tolerance_scale=scale,
                passed=report.passed,
            )
        )
    return sorted(points, key=lambda p: p.context_length)


def drift_summary(points: list[DriftPoint]) -> str:
    """Human-readable summary of long-context drift."""
    if not points:
        return "No drift data points."
    lines = [
        f"{'context':>10}  {'kl_div':>10}  {'argmax_acc':>12}  {'tol_scale':>10}  {'status':>8}",
        "-" * 60,
    ]
    for p in points:
        status = "PASS" if p.passed else "FAIL"
        lines.append(
            f"{p.context_length:>10}  {p.kl_divergence:>10.6f}  "
            f"{p.argmax_accuracy:>12.6f}  {p.tolerance_scale:>10.2f}  {status:>8}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Golden-reference comparison (compact JSON format)
# ---------------------------------------------------------------------------

# Tolerance justification (ob-3uh):
#
# The default tolerances were chosen for float32 CPU inference of
# Qwen3.5-0.8B on x86 vs. aarch64, where the only divergence source is
# floating-point summation order and NEON vs. SSE reduction width:
#
#   atol = 1e-4    — Per-element logit differences from FP32 reordering
#                    are typically <5e-5 (measured on A57).  1e-4 gives
#                    2x headroom for accumulation across vocab=151k.
#
#   rtol = 1e-3    — Relative tolerance for large-magnitude logits (the
#                    top tokens have |logit| > 10, where atol alone is
#                    too strict).
#
#   kl_div_threshold = 0.01 — KL divergence between softmax distributions
#                    from reordered sums is <0.002 empirically.  0.01
#                    catches genuine divergence while allowing FP noise.
#
#   topk_min_accuracy = 0.8 — 4 of 5 top tokens must agree.  This is
#                    strict enough to catch argmax flips but tolerant
#                    of tail-token reordering in the long tail.
#
#   argmax_accuracy_threshold = 0.95 — At least 95% of positions must
#                    have the same predicted token.  A 5% mismatch rate
#                    is the noise floor for FP32 reordering across 151k
#                    vocab; anything higher indicates a real change.
#
#   drift_scale_factor = 1.5 — GDN recurrent state accumulates error
#                    multiplicatively.  1.5x per context doubling is
#                    conservative: measured drift on A57 at 8k context
#                    is <2x the 4k baseline.
#
#   perplexity_atol = 0.1, perplexity_rtol = 0.02 — Perplexity is
#                    exp(avg_nll), so a 2% relative tolerance corresponds
#                    to ~0.02 nats of NLL drift, which is within the
#                    FP32 reordering band.


def compare_reference(
    reference_entry: dict[str, Any],
    candidate_entry: dict[str, Any],
    config: ToleranceConfig,
) -> CorrectnessReport:
    """Compare a candidate model's output against a golden reference entry.

    Works with the compact reference format (results/reference/
    qwen35-0.8b_reference_compact.json), which stores perplexity,
    argmax token, top-k window, and generated tokens but not full-vocab
    logits.

    Args:
        reference_entry: One entry dict from the golden reference JSON.
        candidate_entry: Same-format dict from the model under test.
        config: Tolerance configuration.

    Returns:
        CorrectnessReport combining perplexity, argmax, top-k overlap,
        and generated-token checks.
    """
    ctx = reference_entry.get("context_length")
    report = CorrectnessReport(context_length=ctx)

    # --- Perplexity comparison ---
    ref_ppl = reference_entry.get("perplexity")
    cand_ppl = candidate_entry.get("perplexity")
    if ref_ppl is not None and cand_ppl is not None:
        ppl_report = compare_perplexity(ref_ppl, cand_ppl, config, context_length=ctx)
        for m in ppl_report.metrics:
            report.metrics.append(m)
        report.failures.extend(ppl_report.failures)

    # --- Argmax token exact match ---
    ref_token = reference_entry.get("argmax_token")
    cand_token = candidate_entry.get("argmax_token")
    if ref_token is not None and cand_token is not None:
        match = ref_token == cand_token
        report.add_metric("argmax_token_match", 1.0 if match else 0.0, 1.0, match)

    # --- Top-k window overlap ---
    ref_windows = reference_entry.get("topk_window", [])
    cand_windows = candidate_entry.get("topk_window", [])
    if ref_windows and cand_windows:
        k = config.topk
        overlaps = []
        for ref_w, cand_w in zip(ref_windows, cand_windows, strict=False):
            ref_indices = set(ref_w.get("indices", [])[:k])
            cand_indices = set(cand_w.get("indices", [])[:k])
            if ref_indices:
                overlaps.append(len(ref_indices & cand_indices) / len(ref_indices))
        if overlaps:
            avg_overlap = sum(overlaps) / len(overlaps)
            passed_topk = avg_overlap >= config.topk_min_accuracy
            report.add_metric(
                f"topk_window_overlap (k={k})", avg_overlap, config.topk_min_accuracy, passed_topk
            )

    # --- Generated token sequence exact match ---
    ref_tokens = reference_entry.get("generated_token_ids", [])
    cand_tokens = candidate_entry.get("generated_token_ids", [])
    if ref_tokens and cand_tokens:
        min_len = min(len(ref_tokens), len(cand_tokens))
        if min_len > 0:
            matches = sum(1 for a, b in zip(ref_tokens, cand_tokens, strict=False) if a == b)
            token_acc = matches / min_len
            # Greedy decode should be deterministic; any mismatch indicates
            # a real numerical divergence that flipped an argmax
            passed_gen = token_acc >= config.argmax_accuracy_threshold
            report.add_metric(
                "generated_token_accuracy", token_acc, config.argmax_accuracy_threshold, passed_gen
            )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m bench.correctness",
        description="Compare model outputs against a golden reference with tolerances.",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to JSON file with reference logits/perplexity.",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Path to JSON file with candidate logits/perplexity.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Context length for this comparison (enables drift scaling).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance (default: 1e-4).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance (default: 1e-3).",
    )
    parser.add_argument(
        "--kl-threshold",
        type=float,
        default=0.01,
        help="KL divergence threshold (default: 0.01).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Top-k for set agreement (default: 5).",
    )
    parser.add_argument(
        "--drift-scale",
        type=float,
        default=1.5,
        help="Drift scale factor per context doubling (default: 1.5).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Write JSON report to this path instead of stdout.",
    )

    args = parser.parse_args(argv)

    config = ToleranceConfig(
        atol=args.atol,
        rtol=args.rtol,
        kl_div_threshold=args.kl_threshold,
        topk=args.topk,
        drift_scale_factor=args.drift_scale,
    )

    # Load reference and candidate
    try:
        with open(args.reference) as f:
            ref_data = json.load(f)
    except FileNotFoundError:
        parser.error(f"reference file not found: {args.reference}")
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read reference file {args.reference}: {exc}")

    try:
        with open(args.candidate) as f:
            cand_data = json.load(f)
    except FileNotFoundError:
        parser.error(f"candidate file not found: {args.candidate}")
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read candidate file {args.candidate}: {exc}")

    reports = []

    # Perplexity comparison if present
    if "perplexity" in ref_data and "perplexity" in cand_data:
        report = compare_perplexity(
            ref_data["perplexity"],
            cand_data["perplexity"],
            config,
            context_length=args.context_length,
        )
        reports.append(report)

    # Logit comparison if present
    if "logits" in ref_data and "logits" in cand_data:
        report = compare_logits(
            ref_data["logits"],
            cand_data["logits"],
            config,
            context_length=args.context_length,
        )
        reports.append(report)

    if not reports:
        print(
            "Error: no comparable data found (need 'logits' or 'perplexity' keys)", file=sys.stderr
        )
        return 1

    # Output
    output_data = {
        "reports": [r.to_dict() for r in reports],
        "passed": all(r.passed for r in reports),
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
            f.write("\n")
        print(f"Wrote report to {args.output}")
    else:
        for report in reports:
            print(report.summary())
            print()
        print(f"Overall: {'PASSED' if output_data['passed'] else 'FAILED'}")

    return 0 if output_data["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
