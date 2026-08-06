"""Tests for bench/correctness.py — tolerance comparison framework (ob-3uh).

Validates logit comparison, perplexity comparison, long-context drift
analysis, golden-reference comparison, and tolerance scaling.
No torch or GPU required — pure stdlib + optional numpy.
"""

import json
import math
import random
from pathlib import Path

import pytest
from bench.correctness import (
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
    compare_reference,
    drift_summary,
    long_context_drift,
)
from bench.correctness import main as correctness_main

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REF_PATH = _REPO_ROOT / "results" / "reference" / "qwen35-0.8b_reference_compact.json"


# ---------------------------------------------------------------------------
# Pure-Python numerical helpers
# ---------------------------------------------------------------------------


class TestSoftmax:
    def test_sums_to_one(self):
        result = _softmax([1.0, 2.0, 3.0])
        assert abs(sum(result) - 1.0) < 1e-12

    def test_all_equal(self):
        result = _softmax([5.0, 5.0, 5.0])
        assert all(abs(r - 1 / 3) < 1e-12 for r in result)

    def test_empty(self):
        assert _softmax([]) == []

    def test_large_values(self):
        """Numerical stability with large logits."""
        result = _softmax([1000.0, 1001.0, 1002.0])
        assert all(0 <= r <= 1 for r in result)
        assert abs(sum(result) - 1.0) < 1e-12


class TestKLDivergence:
    def test_identical_distributions(self):
        p = _softmax([1.0, 2.0, 3.0])
        assert abs(_kl_divergence(p, p)) < 1e-10

    def test_different_distributions(self):
        p = [0.9, 0.1]
        q = [0.5, 0.5]
        kl = _kl_divergence(p, q)
        assert kl > 0

    def test_zero_handling(self):
        """Zero probability in P shouldn't cause issues."""
        p = [0.0, 1.0]
        q = [0.5, 0.5]
        kl = _kl_divergence(p, q)
        assert kl > 0

    def test_zero_in_q(self):
        """Zero in Q should be handled with epsilon."""
        p = [0.5, 0.5]
        q = [1.0, 0.0]
        kl = _kl_divergence(p, q)
        assert math.isfinite(kl)


class TestMaxAbsDiff:
    def test_identical(self):
        assert _max_abs_diff([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_difference(self):
        assert _max_abs_diff([1.0, 2.0], [1.5, 2.0]) == 0.5


class TestTopKIndices:
    def test_basic(self):
        result = _topk_indices([1.0, 3.0, 2.0, 5.0, 4.0], 3)
        assert result == {1, 3, 4}

    def test_k_equals_len(self):
        result = _topk_indices([1.0, 2.0], 2)
        assert result == {0, 1}


# ---------------------------------------------------------------------------
# ToleranceConfig
# ---------------------------------------------------------------------------


class TestToleranceConfig:
    def test_defaults(self):
        cfg = ToleranceConfig()
        assert cfg.atol == 1e-4
        assert cfg.rtol == 1e-3
        assert cfg.kl_div_threshold == 0.01
        assert cfg.topk == 5

    def test_tolerance_for_context_no_scaling(self):
        """At base context, scale should be 1.0."""
        cfg = ToleranceConfig(drift_scale_factor=1.5)
        assert cfg.tolerance_for_context(4096) == 1.0

    def test_tolerance_for_context_scaling(self):
        """Larger context should get larger tolerance."""
        cfg = ToleranceConfig(drift_scale_factor=2.0)
        scale_8k = cfg.tolerance_for_context(8192)
        scale_16k = cfg.tolerance_for_context(16384)
        assert scale_8k == 2.0
        assert scale_16k == 4.0

    def test_tolerance_flat_when_factor_1(self):
        """With drift_scale_factor=1, tolerance should never scale."""
        cfg = ToleranceConfig(drift_scale_factor=1.0)
        assert cfg.tolerance_for_context(1000000) == 1.0

    def test_scaled_preserves_topk(self):
        """scaled() should not change topk or accuracy thresholds."""
        cfg = ToleranceConfig(topk=10, topk_min_accuracy=0.9)
        scaled = cfg.scaled(3.0)
        assert scaled.topk == 10
        assert scaled.topk_min_accuracy == 0.9
        assert scaled.atol == pytest.approx(cfg.atol * 3.0)


# ---------------------------------------------------------------------------
# Logit comparison
# ---------------------------------------------------------------------------


class TestCompareLogits:
    def test_identical_logits_pass(self):
        """Identical logits should pass all checks."""
        logits = [[1.0, 2.0, 3.0, 0.5], [0.1, 0.2, 0.3, 0.4]]
        cfg = ToleranceConfig()
        report = compare_logits(logits, logits, cfg)
        assert report.passed
        assert len(report.failures) == 0

    def test_small_difference_passes(self):
        """Tiny differences within tolerance should pass."""
        ref = [[1.0, 2.0, 3.0, 0.5]]
        cand = [[1.0 + 1e-6, 2.0 + 1e-6, 3.0 + 1e-6, 0.5 + 1e-6]]
        cfg = ToleranceConfig(atol=1e-4)
        report = compare_logits(ref, cand, cfg)
        assert report.passed

    def test_large_difference_fails(self):
        """Large differences should fail."""
        ref = [[1.0, 2.0, 3.0, 0.5]]
        cand = [[5.0, 1.0, 0.5, 3.0]]
        cfg = ToleranceConfig(atol=1e-4, kl_div_threshold=0.01)
        report = compare_logits(ref, cand, cfg)
        assert not report.passed

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
        assert not report_base.passed
        assert report_drift.passed

    def test_metrics_present(self):
        """Report should have all four metric types."""
        logits = [[1.0, 2.0, 3.0]]
        cfg = ToleranceConfig()
        report = compare_logits(logits, logits, cfg)
        names = [m.name for m in report.metrics]
        assert "max_abs_diff" in names
        assert "avg_kl_divergence" in names
        assert "argmax_accuracy" in names

    def test_report_has_context_length(self):
        logits = [[1.0, 2.0, 3.0]]
        cfg = ToleranceConfig()
        report = compare_logits(logits, logits, cfg, context_length=32768)
        assert report.context_length == 32768

    def test_multi_position(self):
        """Should handle multiple positions (seq_len > 1)."""
        ref = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.1, 0.2, 0.3]]
        cfg = ToleranceConfig()
        report = compare_logits(ref, ref, cfg)
        assert report.passed


# ---------------------------------------------------------------------------
# Perplexity comparison
# ---------------------------------------------------------------------------


class TestComparePerplexity:
    def test_identical_passes(self):
        cfg = ToleranceConfig()
        report = compare_perplexity(10.5, 10.5, cfg)
        assert report.passed

    def test_small_diff_passes(self):
        cfg = ToleranceConfig(perplexity_atol=0.2)
        report = compare_perplexity(10.0, 10.1, cfg)
        assert report.passed

    def test_large_diff_fails(self):
        cfg = ToleranceConfig(perplexity_atol=0.01, perplexity_rtol=0.001)
        report = compare_perplexity(10.0, 15.0, cfg)
        assert not report.passed

    def test_relative_tolerance(self):
        """Relative tolerance should scale with magnitude."""
        cfg = ToleranceConfig(perplexity_atol=0.0, perplexity_rtol=0.05)
        # 5% of 100 = 5, so 104 should pass, 106 should fail
        assert compare_perplexity(100.0, 104.0, cfg).passed
        assert not compare_perplexity(100.0, 106.0, cfg).passed

    def test_context_scaling(self):
        """Larger context should relax perplexity tolerance."""
        cfg = ToleranceConfig(perplexity_atol=0.1, perplexity_rtol=0.0, drift_scale_factor=2.0)
        # At base: atol=0.1, so diff=0.15 fails
        report_base = compare_perplexity(10.0, 10.15, cfg, context_length=4096)
        assert not report_base.passed
        # At 16384 (4x): scale = 2^2 = 4, atol = 0.4, so 0.15 passes
        report_ctx = compare_perplexity(10.0, 10.15, cfg, context_length=16384)
        assert report_ctx.passed

    def test_metrics_present(self):
        cfg = ToleranceConfig()
        report = compare_perplexity(10.0, 10.05, cfg)
        names = [m.name for m in report.metrics]
        assert "perplexity_abs_diff" in names
        assert "perplexity_rel_diff" in names
        assert "reference_perplexity" in names
        assert "candidate_perplexity" in names


# ---------------------------------------------------------------------------
# CorrectnessReport
# ---------------------------------------------------------------------------


class TestCorrectnessReport:
    def test_empty_report_passes(self):
        report = CorrectnessReport()
        assert report.passed

    def test_add_passing_metric(self):
        report = CorrectnessReport()
        report.add_metric("test", 0.5, 1.0, True)
        assert report.passed

    def test_add_failing_metric(self):
        report = CorrectnessReport()
        report.add_metric("test", 2.0, 1.0, False)
        assert not report.passed
        assert len(report.failures) == 1

    def test_summary_contains_status(self):
        report = CorrectnessReport(context_length=8192)
        report.add_metric("test", 0.5, 1.0, True)
        text = report.summary()
        assert "PASSED" in text
        assert "8192" in text

    def test_to_dict_structure(self):
        report = CorrectnessReport(context_length=4096)
        report.add_metric("x", 1.0, 2.0, True)
        d = report.to_dict()
        assert d["context_length"] == 4096
        assert d["passed"] is True
        assert len(d["metrics"]) == 1
        assert d["metrics"][0]["name"] == "x"


# ---------------------------------------------------------------------------
# Long-context drift
# ---------------------------------------------------------------------------


class TestLongContextDrift:
    def test_basic_drift(self):
        """Drift should show increasing tolerance scale with context."""
        cfg = ToleranceConfig(drift_scale_factor=2.0)
        reports = []
        for ctx in [4096, 8192, 16384, 32768]:
            r = CorrectnessReport(context_length=ctx)
            r.add_metric("avg_kl_divergence", 0.001 * (ctx / 4096), None, None)
            r.add_metric("argmax_accuracy", 1.0 - 0.01 * math.log2(ctx / 4096), None, None)
            reports.append(r)
        points = long_context_drift(reports, cfg)
        assert len(points) == 4
        assert points[0].context_length == 4096
        assert points[-1].context_length == 32768
        assert points[-1].tolerance_scale > points[0].tolerance_scale

    def test_skips_none_context(self):
        """Reports without context_length should be skipped."""
        cfg = ToleranceConfig()
        r1 = CorrectnessReport(context_length=4096)
        r1.add_metric("avg_kl_divergence", 0.001, None, None)
        r1.add_metric("argmax_accuracy", 0.99, None, None)
        r2 = CorrectnessReport(context_length=None)
        points = long_context_drift([r1, r2], cfg)
        assert len(points) == 1

    def test_drift_summary_output(self):
        cfg = ToleranceConfig()
        r = CorrectnessReport(context_length=4096)
        r.add_metric("avg_kl_divergence", 0.001, None, None)
        r.add_metric("argmax_accuracy", 0.99, None, None)
        points = long_context_drift([r], cfg)
        text = drift_summary(points)
        assert "4096" in text
        assert "kl_div" in text


# ---------------------------------------------------------------------------
# Golden-reference comparison (compare_reference)
# ---------------------------------------------------------------------------


def _make_ref_entry(
    context_length=128,
    perplexity=20.0,
    argmax_token=42,
    topk_window=None,
    generated_token_ids=None,
):
    """Helper: build a minimal reference-style entry dict."""
    if topk_window is None:
        topk_window = [
            {"position_from_end": -1, "indices": [10, 20, 30, 40, 50], "values": [5, 4, 3, 2, 1]},
        ]
    if generated_token_ids is None:
        generated_token_ids = [42, 43, 44]
    return {
        "entry_id": "test",
        "prompt_id": "factual",
        "context_length": context_length,
        "perplexity": perplexity,
        "avg_nll": math.log(perplexity),
        "argmax_token": argmax_token,
        "topk_window": topk_window,
        "generated_token_ids": generated_token_ids,
        "generated_text": "test",
    }


class TestCompareReference:
    def test_identical_entries_pass(self):
        """Comparing an entry to itself must pass."""
        entry = _make_ref_entry()
        cfg = ToleranceConfig()
        report = compare_reference(entry, entry, cfg)
        assert report.passed

    def test_perplexity_mismatch_fails(self):
        """Large perplexity difference should fail."""
        ref = _make_ref_entry(perplexity=10.0)
        cand = _make_ref_entry(perplexity=50.0)
        cfg = ToleranceConfig(perplexity_atol=0.1, perplexity_rtol=0.01)
        report = compare_reference(ref, cand, cfg)
        assert not report.passed

    def test_argmax_token_mismatch_detected(self):
        """Different argmax token should be flagged."""
        ref = _make_ref_entry(argmax_token=42)
        cand = _make_ref_entry(argmax_token=99)
        cfg = ToleranceConfig()
        report = compare_reference(ref, cand, cfg)
        names = [m.name for m in report.metrics]
        assert "argmax_token_match" in names
        token_metric = next(m for m in report.metrics if m.name == "argmax_token_match")
        assert token_metric.value == 0.0
        assert not token_metric.passed

    def test_topk_window_overlap_measured(self):
        """Top-k window overlap should be computed."""
        ref_window = {
            "position_from_end": -1,
            "indices": [1, 2, 3, 4, 5],
            "values": [5, 4, 3, 2, 1],
        }
        # 3 of 5 overlap
        cand_window = {
            "position_from_end": -1,
            "indices": [1, 2, 3, 6, 7],
            "values": [5, 4, 3, 2, 1],
        }
        ref = _make_ref_entry(topk_window=[ref_window])
        cand = _make_ref_entry(topk_window=[cand_window])
        cfg = ToleranceConfig(topk=5, topk_min_accuracy=0.8)
        report = compare_reference(ref, cand, cfg)
        overlap_metric = next((m for m in report.metrics if "topk_window_overlap" in m.name), None)
        assert overlap_metric is not None
        assert overlap_metric.value == pytest.approx(0.6)  # 3/5
        assert not overlap_metric.passed  # 0.6 < 0.8

    def test_generated_token_accuracy(self):
        """Generated token accuracy should be measured."""
        ref = _make_ref_entry(generated_token_ids=[1, 2, 3, 4, 5])
        cand = _make_ref_entry(generated_token_ids=[1, 2, 9, 4, 5])  # 4/5 match
        cfg = ToleranceConfig(argmax_accuracy_threshold=0.8)
        report = compare_reference(ref, cand, cfg)
        gen_metric = next((m for m in report.metrics if m.name == "generated_token_accuracy"), None)
        assert gen_metric is not None
        assert gen_metric.value == pytest.approx(0.8)
        assert gen_metric.passed  # exactly at threshold

    def test_context_length_preserved(self):
        """Report should carry the reference entry's context length."""
        entry = _make_ref_entry(context_length=2048)
        cfg = ToleranceConfig()
        report = compare_reference(entry, entry, cfg)
        assert report.context_length == 2048

    def test_missing_fields_degrade_gracefully(self):
        """Missing optional fields should not crash."""
        ref = {"context_length": 128, "perplexity": 10.0}
        cand = {"context_length": 128, "perplexity": 10.05}
        cfg = ToleranceConfig()
        report = compare_reference(ref, cand, cfg)
        assert report.passed  # perplexity close enough, other fields absent


class TestCompareReferenceGoldenData:
    """End-to-end tests using the actual golden reference file."""

    @pytest.fixture(scope="class")
    def ref_data(self):
        if not _REF_PATH.exists():
            pytest.skip("Compact reference not generated — run scripts/generate_reference.py")
        with open(_REF_PATH) as f:
            return json.load(f)

    def test_identity_all_entries(self, ref_data):
        """Every entry compared against itself must pass."""
        cfg = ToleranceConfig()
        for entry in ref_data["entries"]:
            report = compare_reference(entry, entry, cfg)
            assert report.passed, f"{entry['entry_id']}: identity check failed: {report.failures}"

    def test_small_perplexity_perturbation_passes(self, ref_data):
        """Small FP-level perturbation to perplexity should still pass."""
        cfg = ToleranceConfig()
        for entry in ref_data["entries"]:
            cand = dict(entry)
            # Add 0.01 absolute noise — well within tolerance
            cand["perplexity"] = entry["perplexity"] + 0.01
            report = compare_reference(entry, cand, cfg)
            assert report.passed, f"{entry['entry_id']}: small perturbation failed"

    def test_large_perplexity_change_fails(self, ref_data):
        """A 50% perplexity change should fail."""
        cfg = ToleranceConfig()
        entry = ref_data["entries"][0]
        cand = dict(entry)
        cand["perplexity"] = entry["perplexity"] * 1.5
        report = compare_reference(entry, cand, cfg)
        assert not report.passed


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
        assert report.passed

    def test_corrupted_model_fails(self):
        """A model with shuffled logits should fail."""
        random.seed(42)
        vocab = 100
        seq_len = 10
        ref = [[random.gauss(0, 1) for _ in range(vocab)] for _ in range(seq_len)]
        cand = [list(reversed(row)) for row in ref]  # completely wrong
        cfg = ToleranceConfig(atol=1e-4, rtol=1e-3, kl_div_threshold=0.01)
        report = compare_logits(ref, cand, cfg)
        assert not report.passed

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
            seq_len = 50
            noise_scale = 0.001 * math.log2(ctx / 4096 + 1)
            ref = [[random.gauss(0, 1) for _ in range(vocab)] for _ in range(seq_len)]
            cand = [[v + random.gauss(0, noise_scale) for v in row] for row in ref]
            report = compare_logits(ref, cand, cfg, context_length=ctx)
            reports.append(report)

        points = long_context_drift(reports, cfg)
        assert all(p.passed for p in points)
        assert points[-1].tolerance_scale > 1.0


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
        json.dumps(d)  # must be serializable
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
        ref_path.write_text(json.dumps(ref))
        cand_path.write_text(json.dumps(cand))
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
        ref_path.write_text(json.dumps({"other": "data"}))
        cand_path.write_text(json.dumps({"other": "data"}))
        rc = correctness_main(["--reference", str(ref_path), "--candidate", str(cand_path)])
        assert rc == 1

    def test_output_to_file(self, tmp_path):
        ref, cand = self._write_ref_cand(tmp_path, [1.0, 2.0], [1.0, 2.0])
        out_path = str(tmp_path / "report.json")
        rc = correctness_main(["--reference", ref, "--candidate", cand, "--output", out_path])
        assert rc == 0
        assert Path(out_path).exists()
        data = json.loads(Path(out_path).read_text())
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
        ref_path.write_text(json.dumps(ref))
        cand_path.write_text(json.dumps(cand))
        return str(ref_path), str(cand_path)
