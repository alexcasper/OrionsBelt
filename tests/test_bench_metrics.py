"""Comprehensive edge-case tests for bench/metrics.py (bead ob-1lm).

The existing tests in test_harness.py cover the happy path; this file adds
the edge cases that protect the statistical protocol (docs/METRICS.md section 7)
from silent numerical regressions.

Coverage:
  * p0 / p100 boundary values (nearest-rank minimum and maximum)
  * Negative numbers and mixed-sign sequences
  * Duplicate values (no uniqueness assumption)
  * Two-value sequences (minimum non-trivial N)
  * Large N (100+ values, rank-clamping at the edges)
  * Float p parameter values
  * All-identical values
  * Summary immutability (frozen dataclass)
  * Summary spread / normalized_spread formulas
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.metrics import percentile, summarize  # noqa: E402

# ---------------------------------------------------------------------------
# percentile — boundary values
# ---------------------------------------------------------------------------


class TestPercentileBoundaries:
    def test_p0_returns_minimum(self):
        assert percentile([5, 3, 1, 4, 2], 0) == 1

    def test_p100_returns_maximum(self):
        assert percentile([5, 3, 1, 4, 2], 100) == 5

    def test_p0_p100_bracket_all_values(self):
        vals = [42, 7, 99, 13, 56]
        assert percentile(vals, 0) == min(vals)
        assert percentile(vals, 100) == max(vals)

    def test_p0_equals_p100_when_identical(self):
        assert percentile([7, 7, 7], 0) == 7
        assert percentile([7, 7, 7], 100) == 7


# ---------------------------------------------------------------------------
# percentile — negative numbers
# ---------------------------------------------------------------------------


class TestPercentileNegatives:
    def test_all_negative(self):
        vals = [-50, -10, -30, -20, -40]
        assert percentile(vals, 50) == -30
        assert percentile(vals, 0) == -50
        assert percentile(vals, 100) == -10

    def test_mixed_signs(self):
        vals = [-5, 0, 5, -10, 10]
        assert percentile(vals, 0) == -10
        assert percentile(vals, 50) == 0
        assert percentile(vals, 100) == 10

    def test_single_negative(self):
        assert percentile([-42], 50) == -42
        assert percentile([-42], 0) == -42
        assert percentile([-42], 100) == -42


# ---------------------------------------------------------------------------
# percentile — duplicates and small N
# ---------------------------------------------------------------------------


class TestPercentileDuplicates:
    def test_all_identical(self):
        vals = [5] * 10
        assert percentile(vals, 0) == 5
        assert percentile(vals, 50) == 5
        assert percentile(vals, 95) == 5
        assert percentile(vals, 100) == 5

    def test_duplicates_with_outlier(self):
        vals = [1, 1, 1, 1, 1, 1, 1, 1, 1, 100]
        assert percentile(vals, 50) == 1
        assert percentile(vals, 95) == 100

    def test_two_distinct_values(self):
        vals = [1, 2]
        assert percentile(vals, 0) == 1
        assert percentile(vals, 100) == 2

    def test_repeated_median(self):
        """Repeated value in the middle should not skew nearest-rank."""
        vals = [1, 2, 2, 2, 2, 2, 2, 2, 2, 10]
        assert percentile(vals, 50) == 2


class TestPercentileSmallN:
    def test_two_values_p50(self):
        # ceil(50/100 * 2) = rank 1, value = sorted[0]
        assert percentile([10, 20], 50) == 10

    def test_two_values_p95(self):
        # ceil(95/100 * 2) = ceil(1.9) = rank 2, value = sorted[1]
        assert percentile([10, 20], 95) == 20


# ---------------------------------------------------------------------------
# percentile — large N
# ---------------------------------------------------------------------------


class TestPercentileLargeN:
    def test_100_values(self):
        vals = list(range(1, 101))
        # p50: ceil(50/100 * 100) = rank 50, value = 50
        assert percentile(vals, 50) == 50
        # p95: ceil(95/100 * 100) = rank 95, value = 95
        assert percentile(vals, 95) == 95

    def test_1000_values(self):
        vals = list(range(1000))
        # rank = ceil(50/100 * 1000) = 500, sorted_vals[499] = 499 (0-indexed)
        assert percentile(vals, 50) == 499
        # rank = ceil(95/100 * 1000) = 950, sorted_vals[949] = 949
        assert percentile(vals, 95) == 949

    def test_large_n_unsorted_input(self):
        """The function must sort internally; shuffled input gives same result."""
        import random

        random.seed(42)
        vals = list(range(1, 101))
        shuffled = vals[:]
        random.shuffle(shuffled)
        assert percentile(shuffled, 50) == 50
        assert percentile(shuffled, 95) == 95

    def test_large_n_p0_p100(self):
        vals = list(range(100, 200))
        assert percentile(vals, 0) == 100
        assert percentile(vals, 100) == 199


# ---------------------------------------------------------------------------
# percentile — float p values
# ---------------------------------------------------------------------------


class TestPercentileFloatP:
    def test_float_p50(self):
        assert percentile([10, 20, 30, 40, 50], 50.0) == 30

    def test_float_p_between_integers(self):
        """p=49.9 and p=50.1 should give the same rank as p=50 at this N."""
        vals = [10, 20, 30, 40, 50]
        # ceil(49.9/100 * 5) = ceil(2.495) = rank 3 -> 30
        assert percentile(vals, 49.9) == 30
        # ceil(50.1/100 * 5) = ceil(2.505) = rank 3 -> 30
        assert percentile(vals, 50.1) == 30

    def test_p_zero_float(self):
        assert percentile([5, 3, 1], 0.0) == 1


# ---------------------------------------------------------------------------
# summarize — comprehensive edge cases
# ---------------------------------------------------------------------------


class TestSummarizeEdgeCases:
    def test_all_identical_values(self):
        s = summarize([42, 42, 42, 42, 42])
        assert s.n == 5
        assert s.p50 == 42
        assert s.p95 == 42
        assert s.spread == 0
        assert s.normalized_spread == 0.0

    def test_single_value(self):
        s = summarize([99])
        assert s.n == 1
        assert s.p50 == 99
        assert s.p95 == 99
        assert s.spread == 0

    def test_two_values(self):
        s = summarize([10, 20])
        assert s.n == 2
        assert s.p50 == 10  # ceil(50/100 * 2) = rank 1
        assert s.p95 == 20  # ceil(95/100 * 2) = rank 2
        assert s.spread == 10

    def test_negative_values(self):
        s = summarize([-30, -20, -10, -5, -1])
        assert s.n == 5
        assert s.p50 == -10
        assert s.p95 == -1
        assert s.spread == 9  # -1 - (-10)
        assert s.normalized_spread == pytest.approx(9 / 10)

    def test_mixed_signs(self):
        s = summarize([-10, -5, 0, 5, 10])
        assert s.p50 == 0
        assert s.p95 == 10
        assert s.spread == 10

    def test_large_values(self):
        s = summarize([1e9, 2e9, 3e9])
        assert s.p50 == 2e9
        assert s.spread == 1e9

    def test_float_values(self):
        s = summarize([1.5, 2.5, 3.5])
        assert s.p50 == 2.5
        assert s.normalized_spread == pytest.approx((3.5 - 2.5) / 2.5)


class TestSummarizeFormula:
    """Verify the spread/normalized_spread formulas from METRICS.md section 7."""

    def test_spread_is_p95_minus_p50(self):
        s = summarize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert s.spread == s.p95 - s.p50

    def test_normalized_spread_is_spread_over_p50(self):
        s = summarize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert s.normalized_spread == pytest.approx(s.spread / s.p50)

    def test_zero_p50_gives_inf_normalized_spread(self):
        s = summarize([0, 0, 0, 0, 0])
        assert s.normalized_spread == math.inf

    def test_near_zero_p50(self):
        """Very small p50 should give a large but finite normalized_spread."""
        s = summarize([0.001, 0.001, 0.001, 1.0])
        assert s.p50 == 0.001
        assert math.isfinite(s.normalized_spread)
        assert s.normalized_spread > 100


class TestSummaryDataclass:
    def test_frozen(self):
        """Summary must be immutable (frozen=True)."""
        s = summarize([1, 2, 3, 4, 5])
        with pytest.raises((AttributeError, TypeError)):
            s.p50 = 999

    def test_fields(self):
        s = summarize([10, 20, 30])
        assert hasattr(s, "n")
        assert hasattr(s, "p50")
        assert hasattr(s, "p95")
        assert hasattr(s, "spread")
        assert hasattr(s, "normalized_spread")

    def test_n_matches_input_length(self):
        s = summarize(list(range(1, 31)))
        assert s.n == 30
