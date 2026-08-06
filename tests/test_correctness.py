"""Tests for bench/correctness.py — tolerance comparison framework (ob-3uh scaffolding).

Validates logit comparison, perplexity comparison, long-context drift
analysis, and tolerance scaling.  No torch or GPU required — pure stdlib
+ optional numpy.

Run without pytest::

    PYTHONPATH=bench:. python3 tests/test_correctness.py
"""

import json as _json
import math
import os
import random
import sys
from pathlib import Path as _Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))

import correctness  # noqa: E402
from correctness import (  # noqa: E402
    ComparisonMetric,
    CorrectnessReport,
    DriftPoint,
    ToleranceConfig,
    _kl_divergence,
    _max_abs_diff,
    _np_kl_divergence,
    _np_softmax,
    _softmax,
    _topk_indices,
    compare_logits,
    compare_perplexity,
    drift_summary,
    long_context_drift,
)
from correctness import main as correctness_main  # noqa: E402

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

_failures = []
_passes = 0


def check(condition, msg):
    global _passes
    if condition:
        _passes += 1
    else:
        _failures.append(msg)


# ---------------------------------------------------------------------------
# Pure-Python numerical helpers
# ---------------------------------------------------------------------------


class TestSoftmax:
    def test_sums_to_one(self):
        result = _softmax([1.0, 2.0, 3.0])
        check(abs(sum(result) - 1.0) < 1e-12, "softmax doesn't sum to 1")

    def test_all_equal(self):
        result = _softmax([5.0, 5.0, 5.0])
        check(all(abs(r - 1 / 3) < 1e-12 for r in result), "uniform softmax incorrect")

    def test_empty(self):
        check(_softmax([]) == [], "empty softmax should return empty list")

    def test_large_values(self):
        """Numerical stability with large logits."""
        result = _softmax([1000.0, 1001.0, 1002.0])
        check(all(0 <= r <= 1 for r in result), "softmax overflow on large values")
        check(abs(sum(result) - 1.0) < 1e-12, "large-value softmax doesn't sum to 1")


class TestKLDivergence:
    def test_identical_distributions(self):
        p = _softmax([1.0, 2.0, 3.0])
        check(abs(_kl_divergence(p, p)) < 1e-10, "KL(p||p) should be ~0")

    def test_different_distributions(self):
        p = [0.9, 0.1]
        q = [0.5, 0.5]
        kl = _kl_divergence(p, q)
        check(kl > 0, "KL of different distributions should be positive")

    def test_zero_handling(self):
        """Zero probability in P shouldn't cause issues."""
        p = [0.0, 1.0]
        q = [0.5, 0.5]
        kl = _kl_divergence(p, q)
        check(kl > 0, "KL with zero in P should still compute")

    def test_zero_in_q(self):
        """Zero in Q should be handled with epsilon."""
        p = [0.5, 0.5]
        q = [1.0, 0.0]
        kl = _kl_divergence(p, q)
        check(math.isfinite(kl), "KL with zero in Q should be finite (epsilon)")


class TestMaxAbsDiff:
    def test_identical(self):
        check(_max_abs_diff([1.0, 2.0], [1.0, 2.0]) == 0.0, "identical max_abs_diff should be 0")

    def test_difference(self):
        check(_max_abs_diff([1.0, 2.0], [1.5, 2.0]) == 0.5, "max_abs_diff incorrect")


class TestTopKIndices:
    def test_basic(self):
        result = _topk_indices([1.0, 3.0, 2.0, 5.0, 4.0], 3)
        check(result == {1, 3, 4}, f"topk_indices wrong: {result}")

    def test_k_equals_len(self):
        result = _topk_indices([1.0, 2.0], 2)
        check(result == {0, 1}, "topk_indices with k=len should return all")


# ---------------------------------------------------------------------------
# ToleranceConfig
# ---------------------------------------------------------------------------


class TestToleranceConfig:
    def test_defaults(self):
        cfg = ToleranceConfig()
        check(cfg.atol == 1e-4, "default atol wrong")
        check(cfg.rtol == 1e-3, "default rtol wrong")
        check(cfg.kl_div_threshold == 0.01, "default kl threshold wrong")
        check(cfg.topk == 5, "default topk wrong")

    def test_tolerance_for_context_no_scaling(self):
        """At base context, scale should be 1.0."""
        cfg = ToleranceConfig(drift_scale_factor=1.5)
        check(cfg.tolerance_for_context(4096) == 1.0, "base context scale should be 1.0")

    def test_tolerance_for_context_scaling(self):
        """Larger context should get larger tolerance."""
        cfg = ToleranceConfig(drift_scale_factor=2.0)
        scale_8k = cfg.tolerance_for_context(8192)
        scale_16k = cfg.tolerance_for_context(16384)
        check(scale_8k == 2.0, f"2x context should scale by 2.0, got {scale_8k}")
        check(scale_16k == 4.0, f"4x context should scale by 4.0, got {scale_16k}")

    def test_tolerance_flat_when_factor_1(self):
        """With drift_scale_factor=1, tolerance should never scale."""
        cfg = ToleranceConfig(drift_scale_factor=1.0)
        check(cfg.tolerance_for_context(1000000) == 1.0, "flat tolerance should always be 1.0")

    def test_scaled_preserves_topk(self):
        """scaled() should not change topk or accuracy thresholds."""
        cfg = ToleranceConfig(topk=10, topk_min_accuracy=0.9)
        scaled = cfg.scaled(3.0)
        check(scaled.topk == 10, "scaled topk changed")
        check(scaled.topk_min_accuracy == 0.9, "scaled topk_min_accuracy changed")
        check(scaled.atol == cfg.atol * 3.0, "scaled atol wrong")


# ---------------------------------------------------------------------------
# Logit comparison
# ---------------------------------------------------------------------------


class TestCompareLogits:
    def test_identical_logits_pass(self):
        """Identical logits should pass all checks."""
        logits = [[1.0, 2.0, 3.0, 0.5], [0.1, 0.2, 0.3, 0.4]]
        cfg = ToleranceConfig()
        report = compare_logits(logits, logits, cfg)
        check(report.passed, "identical logits should pass")
        check(len(report.failures) == 0, "identical logits should have no failures")

    def test_small_difference_passes(self):
        """Tiny differences within tolerance should pass."""
        ref = [[1.0, 2.0, 3.0, 0.5]]
        cand = [[1.0 + 1e-6, 2.0 + 1e-6, 3.0 + 1e-6, 0.5 + 1e-6]]
        cfg = ToleranceConfig(atol=1e-4)
        report = compare_logits(ref, cand, cfg)
        check(report.passed, "tiny difference should pass with atol=1e-4")

    def test_large_difference_fails(self):
        """Large differences should fail."""
        ref = [[1.0, 2.0, 3.0, 0.5]]
        cand = [[5.0, 1.0, 0.5, 3.0]]
        cfg = ToleranceConfig(atol=1e-4, kl_div_threshold=0.01)
        report = compare_logits(ref, cand, cfg)
        check(not report.passed, "large difference should fail")

    def test_context_scaling_relaxes_tolerance(self):
        """At large context, same difference should pass where it fails at base."""
        ref = [[1.0, 2.0, 3.0, 0.5]]
        cand = [[1.0 + 0.01, 2.0 - 0.01, 3.0, 0.5]]
        # At base context with strict tolerance, this should fail
        cfg_strict = ToleranceConfig(atol=0.005, kl_div_threshold=0.001)
        report_base = compare_logits(ref, cand, cfg_strict, context_length=4096)
        # At 131072 with generous drift scaling, should pass
        cfg_generous = ToleranceConfig(atol=0.005, kl_div_threshold=0.001, drift_scale_factor=2.0)
        report_drift = compare_logits(ref, cand, cfg_generous, context_length=131072)
        check(not report_base.passed, "should fail at base context with strict tol")
        check(
            report_drift.passed,
            f"should pass at 131072 with drift scaling, failures: {report_drift.failures}",
        )

    def test_metrics_present(self):
        """Report should have all four metric types."""
        logits = [[1.0, 2.0, 3.0]]
        cfg = ToleranceConfig()
        report = compare_logits(logits, logits, cfg)
        names = [m.name for m in report.metrics]
        check("max_abs_diff" in names, "missing max_abs_diff metric")
        check("avg_kl_divergence" in names, "missing avg_kl_divergence metric")
        check("argmax_accuracy" in names, "missing argmax_accuracy metric")

    def test_report_has_context_length(self):
        logits = [[1.0, 2.0, 3.0]]
        cfg = ToleranceConfig()
        report = compare_logits(logits, logits, cfg, context_length=32768)
        check(report.context_length == 32768, "context_length not set in report")

    def test_multi_position(self):
        """Should handle multiple positions (seq_len > 1)."""
        ref = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.1, 0.2, 0.3]]
        cfg = ToleranceConfig()
        report = compare_logits(ref, ref, cfg)
        check(report.passed, "multi-position identical logits should pass")


# ---------------------------------------------------------------------------
# Perplexity comparison
# ---------------------------------------------------------------------------


class TestComparePerplexity:
    def test_identical_passes(self):
        cfg = ToleranceConfig()
        report = compare_perplexity(10.5, 10.5, cfg)
        check(report.passed, "identical perplexity should pass")

    def test_small_diff_passes(self):
        cfg = ToleranceConfig(perplexity_atol=0.2)
        report = compare_perplexity(10.0, 10.1, cfg)
        check(report.passed, "small perplexity diff should pass")

    def test_large_diff_fails(self):
        cfg = ToleranceConfig(perplexity_atol=0.01, perplexity_rtol=0.001)
        report = compare_perplexity(10.0, 15.0, cfg)
        check(not report.passed, "large perplexity diff should fail")

    def test_relative_tolerance(self):
        """Relative tolerance should scale with magnitude."""
        cfg = ToleranceConfig(perplexity_atol=0.0, perplexity_rtol=0.05)
        # 5% of 100 = 5, so 104 should pass, 106 should fail
        check(
            compare_perplexity(100.0, 104.0, cfg).passed, "4% relative diff should pass at 5% rtol"
        )
        check(
            not compare_perplexity(100.0, 106.0, cfg).passed,
            "6% relative diff should fail at 5% rtol",
        )

    def test_context_scaling(self):
        """Larger context should relax perplexity tolerance."""
        cfg = ToleranceConfig(perplexity_atol=0.1, perplexity_rtol=0.0, drift_scale_factor=2.0)
        # At base: atol=0.1, so diff=0.15 fails
        report_base = compare_perplexity(10.0, 10.15, cfg, context_length=4096)
        check(not report_base.passed, "0.15 diff should fail at base with atol=0.1")
        # At 16384 (4x): scale = 2^2 = 4, atol = 0.4, so 0.15 passes
        report_ctx = compare_perplexity(10.0, 10.15, cfg, context_length=16384)
        check(
            report_ctx.passed,
            f"0.15 diff should pass at 16384 with scaled atol, failures: {report_ctx.failures}",
        )

    def test_metrics_present(self):
        cfg = ToleranceConfig()
        report = compare_perplexity(10.0, 10.05, cfg)
        names = [m.name for m in report.metrics]
        check("perplexity_abs_diff" in names, "missing perplexity_abs_diff")
        check("perplexity_rel_diff" in names, "missing perplexity_rel_diff")
        check("reference_perplexity" in names, "missing reference_perplexity")
        check("candidate_perplexity" in names, "missing candidate_perplexity")


# ---------------------------------------------------------------------------
# CorrectnessReport
# ---------------------------------------------------------------------------


class TestCorrectnessReport:
    def test_empty_report_passes(self):
        report = CorrectnessReport()
        check(report.passed, "empty report should pass")

    def test_add_passing_metric(self):
        report = CorrectnessReport()
        report.add_metric("test", 0.5, 1.0, True)
        check(report.passed, "passing metric shouldn't create failure")

    def test_add_failing_metric(self):
        report = CorrectnessReport()
        report.add_metric("test", 2.0, 1.0, False)
        check(not report.passed, "failing metric should fail report")
        check(len(report.failures) == 1, f"should have 1 failure, got {len(report.failures)}")

    def test_summary_contains_status(self):
        report = CorrectnessReport(context_length=8192)
        report.add_metric("test", 0.5, 1.0, True)
        text = report.summary()
        check("PASSED" in text, "summary should contain PASSED")
        check("8192" in text, "summary should contain context_length")

    def test_to_dict_structure(self):
        report = CorrectnessReport(context_length=4096)
        report.add_metric("x", 1.0, 2.0, True)
        d = report.to_dict()
        check(d["context_length"] == 4096, "to_dict context_length wrong")
        check(d["passed"] is True, "to_dict passed wrong")
        check(len(d["metrics"]) == 1, "to_dict metrics count wrong")
        check(d["metrics"][0]["name"] == "x", "to_dict metric name wrong")


# ---------------------------------------------------------------------------
# Long-context drift
# ---------------------------------------------------------------------------


class TestLongContextDrift:
    def test_basic_drift(self):
        """Drift should show increasing tolerance scale with context."""
        cfg = ToleranceConfig(drift_scale_factor=2.0)
        reports = []
        for ctx in [4096, 8192, 16384, 32768]:
            # Simulate increasing KL divergence with context
            r = CorrectnessReport(context_length=ctx)
            r.add_metric("avg_kl_divergence", 0.001 * (ctx / 4096), None, None)
            r.add_metric("argmax_accuracy", 1.0 - 0.01 * math.log2(ctx / 4096), None, None)
            reports.append(r)
        points = long_context_drift(reports, cfg)
        check(len(points) == 4, f"should have 4 drift points, got {len(points)}")
        check(points[0].context_length == 4096, "first point should be 4096")
        check(points[-1].context_length == 32768, "last point should be 32768")
        check(
            points[-1].tolerance_scale > points[0].tolerance_scale,
            "tolerance should increase with context",
        )

    def test_skips_none_context(self):
        """Reports without context_length should be skipped."""
        cfg = ToleranceConfig()
        r1 = CorrectnessReport(context_length=4096)
        r1.add_metric("avg_kl_divergence", 0.001, None, None)
        r1.add_metric("argmax_accuracy", 0.99, None, None)
        r2 = CorrectnessReport(context_length=None)
        points = long_context_drift([r1, r2], cfg)
        check(len(points) == 1, f"should skip None-context report, got {len(points)} points")

    def test_drift_summary_output(self):
        cfg = ToleranceConfig()
        r = CorrectnessReport(context_length=4096)
        r.add_metric("avg_kl_divergence", 0.001, None, None)
        r.add_metric("argmax_accuracy", 0.99, None, None)
        points = long_context_drift([r], cfg)
        text = drift_summary(points)
        check("4096" in text, "drift summary should contain context length")
        check("kl_div" in text, "drift summary should contain kl_div header")


# ---------------------------------------------------------------------------
# Integration: simulated golden vs candidate
# ---------------------------------------------------------------------------


class TestIntegrationSimulated:
    """Simulate a realistic comparison scenario."""

    def test_near_identical_model_passes(self):
        """A model with tiny numerical noise should pass."""
        random.seed(42)
        vocab = 100
        seq_len = 10
        ref = [[random.gauss(0, 1) for _ in range(vocab)] for _ in range(seq_len)]
        cand = [[v + random.gauss(0, 1e-6) for v in row] for row in ref]
        cfg = ToleranceConfig(atol=1e-4, rtol=1e-3, kl_div_threshold=0.01)
        report = compare_logits(ref, cand, cfg)
        check(report.passed, f"model with 1e-6 noise should pass, failures: {report.failures}")

    def test_corrupted_model_fails(self):
        """A model with shuffled logits should fail."""
        random.seed(42)
        vocab = 100
        seq_len = 10
        ref = [[random.gauss(0, 1) for _ in range(vocab)] for _ in range(seq_len)]
        cand = [list(reversed(row)) for row in ref]  # completely wrong
        cfg = ToleranceConfig(atol=1e-4, rtol=1e-3, kl_div_threshold=0.01)
        report = compare_logits(ref, cand, cfg)
        check(not report.passed, "reversed logits should fail")

    def test_drift_scenario_at_multiple_contexts(self):
        """Simulate drift: model degrades at longer contexts but stays within tolerance."""
        cfg = ToleranceConfig(
            atol=0.01,
            rtol=0.01,
            kl_div_threshold=0.05,
            drift_scale_factor=1.5,
        )
        random.seed(42)
        vocab = 50
        reports = []
        for ctx in [4096, 8192, 16384, 32768, 65536]:
            # Use enough positions that a single argmax flip doesn't fail
            seq_len = 50
            noise_scale = 0.001 * math.log2(ctx / 4096 + 1)
            ref = [[random.gauss(0, 1) for _ in range(vocab)] for _ in range(seq_len)]
            cand = [[v + random.gauss(0, noise_scale) for v in row] for row in ref]
            report = compare_logits(ref, cand, cfg, context_length=ctx)
            reports.append(report)

        points = long_context_drift(reports, cfg)
        # All should pass since noise is small
        all_pass = all(p.passed for p in points)
        check(all_pass, "small drift should stay within tolerance at all contexts")
        # Tolerance scale should increase
        check(points[-1].tolerance_scale > 1.0, "largest context should have scale > 1")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_all():
    tests = [
        TestSoftmax(),
        TestKLDivergence(),
        TestMaxAbsDiff(),
        TestTopKIndices(),
        TestToleranceConfig(),
        TestCompareLogits(),
        TestComparePerplexity(),
        TestCorrectnessReport(),
        TestLongContextDrift(),
        TestIntegrationSimulated(),
    ]
    for suite in tests:
        for name in sorted(dir(suite)):
            if name.startswith("test_"):
                getattr(suite, name)()


# ---------------------------------------------------------------------------
# Pytest-style tests for previously untested paths
# ---------------------------------------------------------------------------


class TestNumpySoftmax:
    """numpy-accelerated softmax helper."""

    def test_sums_to_one(self):
        import numpy as np

        result = _np_softmax(np.array([1.0, 2.0, 3.0]))
        assert abs(float(np.sum(result)) - 1.0) < 1e-9

    def test_matches_pure_python(self):
        import numpy as np

        logits = [0.5, -1.2, 3.4, 0.0]
        np_result = _np_softmax(np.array(logits))
        py_result = _softmax(logits)
        for a, b in zip(np_result, py_result, strict=True):
            assert abs(float(a) - b) < 1e-9

    def test_large_values_stable(self):
        """Numerically stable — no overflow."""
        import numpy as np

        result = _np_softmax(np.array([1000.0, 1001.0, 1002.0]))
        assert all(0 <= r <= 1 for r in result)
        assert abs(sum(result) - 1.0) < 1e-9

    def test_numpy_2d(self):
        """Works on 2D arrays (batch dimension)."""
        import numpy as np

        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = _np_softmax(arr)
        assert result.shape == (2, 3)
        for row in result:
            assert abs(sum(row) - 1.0) < 1e-9


class TestNumpyKLDivergence:
    def test_identical_distributions(self):
        import numpy as np

        p = np.array([0.2, 0.3, 0.5])
        assert abs(_np_kl_divergence(p, p)) < 1e-9

    def test_matches_pure_python(self):
        import numpy as np

        p = [0.1, 0.4, 0.5]
        q = [0.3, 0.3, 0.4]
        np_result = _np_kl_divergence(np.array(p), np.array(q))
        py_result = _kl_divergence(p, q)
        assert abs(np_result - py_result) < 1e-9

    def test_zero_handling(self):
        """p[i]=0 entries are skipped."""
        import numpy as np

        p = np.array([0.0, 0.5, 0.5])
        q = np.array([0.9, 0.05, 0.05])
        result = _np_kl_divergence(p, q)
        assert result > 0


class TestComparisonMetricStr:
    """ComparisonMetric.__str__ formatting."""

    def test_with_threshold_passed(self):
        m = ComparisonMetric("KL", 0.001, threshold=0.01, passed=True)
        s = str(m)
        assert "✓" in s
        assert "KL" in s
        assert "threshold" in s

    def test_with_threshold_failed(self):
        m = ComparisonMetric("KL", 0.5, threshold=0.01, passed=False)
        s = str(m)
        assert "✗" in s
        assert "threshold" in s

    def test_without_threshold(self):
        m = ComparisonMetric("Info", 42.0, threshold=None, passed=None)
        s = str(m)
        assert "ℹ" in s
        assert "Info" in s
        assert "threshold" not in s

    def test_passed_none_shows_info(self):
        m = ComparisonMetric("Note", 1.0, threshold=None, passed=None)
        assert "ℹ" in str(m)

    def test_passed_false_shows_x(self):
        m = ComparisonMetric("Note", 1.0, threshold=2.0, passed=False)
        assert "✗" in str(m)


class TestCorrectnessReportSummary:
    """CorrectnessReport.summary() with failures."""

    def test_summary_with_failures(self):
        report = CorrectnessReport(context_length=2048)
        report.add_metric("KL", 0.5, threshold=0.01, passed=False)
        summary = report.summary()
        assert "FAILED" in summary
        assert "Failures:" in summary
        assert "KL" in summary

    def test_summary_with_context_length(self):
        report = CorrectnessReport(context_length=4096)
        report.add_metric("Atol", 1e-6, threshold=1e-4, passed=True)
        summary = report.summary()
        assert "context_length=4096" in summary
        assert "PASSED" in summary

    def test_summary_without_context_length(self):
        report = CorrectnessReport()
        report.add_metric("Info", 1.0, threshold=None, passed=None)
        summary = report.summary()
        assert "context_length" not in summary

    def test_summary_has_separator_line(self):
        report = CorrectnessReport()
        report.add_metric("KL", 0.01, threshold=0.1, passed=True)
        summary = report.summary()
        assert "-" * 60 in summary

    def test_to_dict_serializable(self):
        report = CorrectnessReport(context_length=1024)
        report.add_metric("KL", 0.01, threshold=0.1, passed=True)
        report.add_metric("Atol", 1e-5, threshold=1e-4, passed=True)
        d = report.to_dict()
        _json.dumps(d)  # must be serializable
        assert d["context_length"] == 1024
        assert d["passed"] is True
        assert len(d["metrics"]) == 2

    def test_to_dict_with_failures(self):
        report = CorrectnessReport()
        report.add_metric("KL", 0.9, threshold=0.01, passed=False)
        d = report.to_dict()
        assert d["passed"] is False
        assert d["metrics"][0]["passed"] is False


class TestDriftSummaryEdgeCases:
    """drift_summary with edge cases."""

    def test_empty_points(self):
        result = drift_summary([])
        assert "No drift" in result

    def test_single_point_pass(self):
        points = [
            DriftPoint(
                context_length=4096,
                kl_divergence=0.001,
                argmax_accuracy=1.0,
                tolerance_scale=1.0,
                passed=True,
            )
        ]
        result = drift_summary(points)
        assert "PASS" in result
        assert "4096" in result

    def test_single_point_fail(self):
        points = [
            DriftPoint(
                context_length=131072,
                kl_divergence=0.9,
                argmax_accuracy=0.3,
                tolerance_scale=8.0,
                passed=False,
            )
        ]
        result = drift_summary(points)
        assert "FAIL" in result
        assert "131072" in result

    def test_header_row_present(self):
        points = [
            DriftPoint(
                context_length=4096,
                kl_divergence=0.01,
                argmax_accuracy=0.9,
                tolerance_scale=1.0,
                passed=True,
            )
        ]
        result = drift_summary(points)
        assert "context" in result
        assert "kl_div" in result
        assert "argmax_acc" in result


class TestMainCLI:
    """main() CLI with file I/O."""

    def _write_ref_cand(self, tmp_path, ref_logits, cand_logits, ref_ppl=None, cand_ppl=None):
        ref = {}
        cand = {}
        if ref_logits is not None:
            ref["logits"] = [ref_logits]  # wrap in position list
            cand["logits"] = [cand_logits]
        if ref_ppl is not None:
            ref["perplexity"] = ref_ppl
            cand["perplexity"] = cand_ppl
        ref_path = tmp_path / "ref.json"
        cand_path = tmp_path / "cand.json"
        ref_path.write_text(_json.dumps(ref))
        cand_path.write_text(_json.dumps(cand))
        return str(ref_path), str(cand_path)

    def test_logit_comparison_pass(self, tmp_path):
        ref, cand = self._write_ref_cand(tmp_path, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        rc = correctness_main(["--reference", ref, "--candidate", cand])
        assert rc == 0

    def test_logit_comparison_fail(self, tmp_path):
        ref, cand = self._write_ref_cand(tmp_path, [1.0, 2.0, 3.0], [10.0, 0.0, 0.0])
        rc = correctness_main(["--reference", ref, "--candidate", cand])
        assert rc == 1

    def test_perplexity_comparison(self, tmp_path):
        ref, cand = self._write_ref_cand(tmp_path, None, None, ref_ppl=10.0, cand_ppl=10.01)
        rc = correctness_main(["--reference", ref, "--candidate", cand])
        assert rc == 0

    def test_perplexity_comparison_fail(self, tmp_path):
        ref, cand = self._write_ref_cand(tmp_path, None, None, ref_ppl=10.0, cand_ppl=100.0)
        rc = correctness_main(["--reference", ref, "--candidate", cand])
        assert rc == 1

    def test_both_logits_and_perplexity(self, tmp_path):
        ref, cand = self._write_ref_cnd_combined(tmp_path)
        rc = correctness_main(["--reference", ref, "--candidate", cand])
        assert rc == 0

    def test_no_comparable_data(self, tmp_path):
        """Files without logits or perplexity → error, return 1."""
        ref_path = tmp_path / "ref.json"
        cand_path = tmp_path / "cand.json"
        ref_path.write_text(_json.dumps({"other": "data"}))
        cand_path.write_text(_json.dumps({"other": "data"}))
        rc = correctness_main(["--reference", str(ref_path), "--candidate", str(cand_path)])
        assert rc == 1

    def test_output_to_file(self, tmp_path):
        ref, cand = self._write_ref_cand(tmp_path, [1.0, 2.0], [1.0, 2.0])
        out_path = str(tmp_path / "report.json")
        rc = correctness_main(["--reference", ref, "--candidate", cand, "--output", out_path])
        assert rc == 0
        assert _Path(out_path).exists()
        data = _json.loads(_Path(out_path).read_text())
        assert "reports" in data
        assert "passed" in data

    def test_custom_tolerances(self, tmp_path):
        """Custom tolerance values are respected."""
        ref, cand = self._write_ref_cand(tmp_path, [1.0, 2.0], [1.0, 2.001])
        # Very tight tolerance → should fail
        rc = correctness_main(["--reference", ref, "--candidate", cand, "--atol", "1e-10"])
        assert rc == 1

    @staticmethod
    def _write_ref_cnd_combined(tmp_path):
        ref = {"logits": [[1.0, 2.0, 3.0]], "perplexity": 5.0}
        cand = {"logits": [[1.0, 2.0, 3.0]], "perplexity": 5.0}
        ref_path = tmp_path / "ref.json"
        cand_path = tmp_path / "cand.json"
        ref_path.write_text(_json.dumps(ref))
        cand_path.write_text(_json.dumps(cand))
        return str(ref_path), str(cand_path)


class TestNoNumpyFallback:
    """Test compare_logits pure-Python paths by forcing HAS_NUMPY=False."""

    def test_max_abs_diff_pure_python(self):
        """Max abs diff computed correctly without numpy."""
        from unittest.mock import patch

        ref = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        cand = [[1.5, 2.0, 2.5], [4.0, 5.5, 6.0]]
        cfg = ToleranceConfig(atol=1.0)
        with patch.object(correctness, "HAS_NUMPY", False):
            report = compare_logits(ref, cand, cfg)
        diff_metric = [m for m in report.metrics if m.name == "max_abs_diff"][0]
        assert diff_metric.value == pytest.approx(0.5)

    def test_kl_divergence_pure_python(self):
        """KL divergence computed correctly without numpy."""
        from unittest.mock import patch

        ref = [[1.0, 2.0, 3.0]]
        cand = [[1.0, 2.0, 3.0]]
        cfg = ToleranceConfig()
        with patch.object(correctness, "HAS_NUMPY", False):
            report = compare_logits(ref, cand, cfg)
        kl_metric = [m for m in report.metrics if m.name == "avg_kl_divergence"][0]
        assert kl_metric.value == pytest.approx(0.0, abs=1e-10)

    def test_topk_accuracy_pure_python(self):
        """Top-k agreement computed correctly without numpy."""
        from unittest.mock import patch

        ref = [[3.0, 1.0, 2.0]]
        cand = [[3.0, 2.0, 1.0]]
        cfg = ToleranceConfig(topk=2, topk_min_accuracy=0.5)
        with patch.object(correctness, "HAS_NUMPY", False):
            report = compare_logits(ref, cand, cfg)
        topk_metric = [m for m in report.metrics if "top" in m.name and "accuracy" in m.name][0]
        # top-2 of [3,1,2] = {0,2}, top-2 of [3,2,1] = {0,1} -> overlap=1/2
        assert topk_metric.value == pytest.approx(0.5)

    def test_argmax_accuracy_pure_python(self):
        """Argmax accuracy computed correctly without numpy."""
        from unittest.mock import patch

        ref = [[1.0, 3.0, 2.0], [5.0, 1.0, 2.0]]
        cand = [[1.0, 3.0, 2.0], [1.0, 5.0, 2.0]]  # second row different argmax
        cfg = ToleranceConfig()
        with patch.object(correctness, "HAS_NUMPY", False):
            report = compare_logits(ref, cand, cfg)
        argmax_metric = [m for m in report.metrics if m.name == "argmax_accuracy"][0]
        assert argmax_metric.value == pytest.approx(0.5)

    def test_identical_passes_pure_python(self):
        """Identical logits pass all checks without numpy."""
        from unittest.mock import patch

        ref = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        cfg = ToleranceConfig()
        with patch.object(correctness, "HAS_NUMPY", False):
            report = compare_logits(ref, ref, cfg)
        assert report.passed

    def test_context_scaling_pure_python(self):
        """Context-length scaling works without numpy."""
        from unittest.mock import patch

        ref = [[1.0, 2.0, 3.0]]
        cand = [[1.01, 2.0, 2.99]]
        cfg = ToleranceConfig(atol=0.005, kl_div_threshold=0.001, drift_scale_factor=2.0)
        with patch.object(correctness, "HAS_NUMPY", False):
            report = compare_logits(ref, cand, cfg, context_length=131072)
        # Should have drift_tolerance_scale metric
        scale_metric = [m for m in report.metrics if m.name == "drift_tolerance_scale"]
        assert len(scale_metric) == 1
        assert scale_metric[0].value > 1.0

    def test_empty_input_pure_python(self):
        """Empty reference list handled gracefully without numpy."""
        from unittest.mock import patch

        ref: list[list[float]] = []
        cand: list[list[float]] = []
        cfg = ToleranceConfig()
        with patch.object(correctness, "HAS_NUMPY", False):
            report = compare_logits(ref, cand, cfg)
        # Should not crash — argmax_accuracy handles empty ref
        argmax_metric = [m for m in report.metrics if m.name == "argmax_accuracy"][0]
        assert argmax_metric.value == 0.0


if __name__ == "__main__" and not pytest.version_info:
    _run_all()
    if _failures:
        print(f"\n✗ {len(_failures)} failures out of {_passes + len(_failures)} checks:")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"\n✓ All {_passes} checks passed.")
