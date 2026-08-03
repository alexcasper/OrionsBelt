"""Metric computation helpers for the benchmark harness.

Implements the statistical protocol from docs/METRICS.md section 7 (nearest-rank
percentiles, p50/p95 mandatory, spread metrics) as pure functions. Stdlib-only
-- same constraint as ``bench/schema.py``: this has to import cleanly on the board
and in the NOE Compiler environment, where we do not control the dependency set.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile.

    For a sorted sequence of *N* values, the *p*-th percentile (0–100) is the
    value at rank ``ceil(p / 100 * N)`` (1-indexed), clamped to ``[1, N]``.
    This is the method mandated by docs/METRICS.md section 7: at the project's
    repeat counts (10–30) it resolves p50 and p95 from actual data rather than
    interpolating between samples.

    Raises ``ValueError`` for an empty sequence or *p* outside [0, 100].
    """
    if not values:
        raise ValueError("cannot compute percentile of an empty sequence")
    if not 0 <= p <= 100:
        raise ValueError(f"p must be in [0, 100], got {p}")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    rank = max(1, min(n, math.ceil(p / 100.0 * n)))
    return sorted_vals[rank - 1]


@dataclass(frozen=True)
class Summary:
    """Statistical summary of a set of repeated measurements.

    Per METRICS.md section 7: p50 and p95 are mandatory; ``spread`` and
    ``normalized_spread`` make cross-context noise comparable despite very
    different absolute magnitudes (PLAN.md R7: thermal variance on
    passively-cooled edge hardware).
    """

    n: int
    p50: float
    p95: float
    spread: float  # p95 - p50
    normalized_spread: float  # (p95 - p50) / p50, or inf if p50 == 0


def summarize(values: Sequence[float]) -> Summary:
    """Compute the mandatory statistical summary for a set of measurements.

    Raises ``ValueError`` for an empty sequence.
    """
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    p50 = percentile(values, 50)
    p95 = percentile(values, 95)
    spread = p95 - p50
    normalized_spread = spread / p50 if p50 != 0 else math.inf
    return Summary(
        n=len(values),
        p50=p50,
        p95=p95,
        spread=spread,
        normalized_spread=normalized_spread,
    )


__all__ = ["percentile", "Summary", "summarize"]
