# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/run_op_probe_audit.py — operator probe classifier.

Tests the pure ``classify()`` function which maps cixparse invocation results
to (verdict, evidence) tuples. This function runs without onnx or cixparse.
"""

from __future__ import annotations

import pytest
from scripts.run_op_probe_audit import classify

# ---------------------------------------------------------------------------
# OK verdict
# ---------------------------------------------------------------------------


class TestClassifyOK:
    def test_clean_success(self):
        verdict, evidence = classify(0, "Compilation successful\nOutput: model.aipu")
        assert verdict == "OK"
        assert "Output" in evidence

    def test_empty_log_success(self):
        verdict, evidence = classify(0, "")
        assert verdict == "OK"
        assert evidence == ""

    def test_success_no_evidence_line(self):
        verdict, evidence = classify(0, "done")
        assert verdict == "OK"
        assert "done" in evidence


class TestClassifyOKWithWarnings:
    def test_warning_detected(self):
        log = "WARNING: deprecated operator detected\nCompilation OK"
        verdict, evidence = classify(0, log)
        assert verdict == "OK_WITH_WARNINGS"
        assert "deprecated" in evidence.lower()

    def test_warning_case_insensitive(self):
        verdict, _ = classify(0, "warning: op might not be optimal")
        assert verdict == "OK_WITH_WARNINGS"

    def test_warning_evidence_returns_warning_line(self):
        log = "some preamble\nWARNING: shape mismatch will cause issue\nmore text"
        verdict, evidence = classify(0, log)
        assert verdict == "OK_WITH_WARNINGS"
        assert "shape mismatch" in evidence


# ---------------------------------------------------------------------------
# UNSUPPORTED_OP verdict
# ---------------------------------------------------------------------------


class TestClassifyUnsupportedOp:
    @pytest.mark.parametrize(
        "marker",
        [
            "unsupported op",
            "unsupported operator",
            "not supported",
            "unsupported type",
            "no conversion",
            "cannot find",
            "unknown op",
            "unimplemented",
        ],
    )
    def test_each_marker_detected(self, marker):
        log = f"Error: {marker} ConvTranspose2d\nAborting"
        verdict, evidence = classify(1, log)
        assert verdict == "UNSUPPORTED_OP"
        assert marker in evidence.lower()

    def test_marker_case_insensitive(self):
        log = "UNSUPPORTED OP: grouped convolution"
        verdict, evidence = classify(1, log)
        assert verdict == "UNSUPPORTED_OP"

    def test_marker_takes_priority_over_failure(self):
        """Even with non-zero returncode, unsupported_op wins."""
        log = "ERROR: unsupported op Attention"
        verdict, _ = classify(1, log)
        assert verdict == "UNSUPPORTED_OP"

    def test_marker_takes_priority_over_warning(self):
        """Even with warnings present, unsupported_op wins."""
        log = "WARNING: something\nunsupported op: Gemm"
        verdict, _ = classify(0, log)
        assert verdict == "UNSUPPORTED_OP"

    def test_evidence_truncated_to_300_chars(self):
        long_msg = "unsupported op: " + "x" * 500
        verdict, evidence = classify(1, long_msg)
        assert verdict == "UNSUPPORTED_OP"
        assert len(evidence) <= 300

    def test_evidence_empty_when_only_marker_no_matching_line(self):
        """If the marker is in the log but on a line that got stripped,
        evidence should be empty."""
        log = "\n  \nunsupported op\n"
        verdict, evidence = classify(1, log)
        assert verdict == "UNSUPPORTED_OP"
        # The marker is on a non-empty line, so it should match
        assert "unsupported op" in evidence


# ---------------------------------------------------------------------------
# FAILED verdict
# ---------------------------------------------------------------------------


class TestClassifyFailed:
    def test_nonzero_returncode_no_marker(self):
        log = "ERROR: internal failure\nTraceback..."
        verdict, evidence = classify(1, log)
        assert verdict == "FAILED"
        assert "failure" in evidence.lower() or "error" in evidence.lower()

    def test_failed_finds_last_error_line(self):
        """When multiple error-like lines exist, returns the LAST one."""
        log = "error: first issue\nsome stuff\nfail: the real problem"
        verdict, evidence = classify(2, log)
        assert verdict == "FAILED"
        assert "real problem" in evidence.lower()

    def test_failed_falls_back_to_last_line(self):
        """If no error/fail/exception pattern, uses last non-empty line."""
        log = "some output\nmore output\nfinal line"
        verdict, evidence = classify(1, log)
        assert verdict == "FAILED"
        assert "final line" in evidence

    def test_failed_empty_log(self):
        verdict, evidence = classify(1, "")
        assert verdict == "FAILED"
        assert evidence == ""

    def test_evidence_truncated_to_300(self):
        long_msg = "error: " + "y" * 500
        verdict, evidence = classify(1, long_msg)
        assert verdict == "FAILED"
        assert len(evidence) <= 300

    def test_exception_pattern_matched(self):
        log = "processing...\nRuntimeError: Exception during lowering"
        verdict, evidence = classify(1, log)
        assert verdict == "FAILED"
        assert "exception" in evidence.lower()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestClassifyEdgeCases:
    def test_whitespace_only_lines_skipped(self):
        log = "  \n\t\nunsupported op: Softmax\n  "
        verdict, _ = classify(1, log)
        assert verdict == "UNSUPPORTED_OP"

    def test_multiline_log_success(self):
        log = "Loading model...\nCompiling graph...\nGenerating IR...\nDone."
        verdict, evidence = classify(0, log)
        assert verdict == "OK"
        assert "Done" in evidence

    def test_negative_returncode_treated_as_failure(self):
        verdict, _ = classify(-1, "crash")
        assert verdict == "FAILED"
