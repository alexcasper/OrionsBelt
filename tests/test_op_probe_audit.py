"""Tests for scripts/run_op_probe_audit.py — classify() verdict logic.

The classify() function is the core decision-making for the NPU operator-coverage
audit (bead ob-t3b.1). It maps a cixparse invocation's returncode and log output
to one of four verdicts: OK, OK_WITH_WARNINGS, UNSUPPORTED_OP, FAILED. This is
the function that determines the load-bearing finding about whether the NPU can
host GDN's sequential recurrence — a bug here would misclassify operator support.

After a refactor that moved ``import onnx`` inside cfg_for(), this module can now
be imported without onnx installed, making classify() directly testable.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from run_op_probe_audit import classify  # noqa: E402

# ---------------------------------------------------------------------------
# UNSUPPORTED_OP verdicts
# ---------------------------------------------------------------------------


class TestUnsupportedOp:
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
    def test_marker_in_log_returns_unsupported(self, marker):
        verdict, evidence = classify(0, f"Error: {marker} Conv")
        assert verdict == "UNSUPPORTED_OP"

    def test_evidence_contains_marker_line(self):
        log = "Loading model...\nError: unsupported op Scan\nDone."
        verdict, evidence = classify(0, log)
        assert verdict == "UNSUPPORTED_OP"
        assert "unsupported op Scan" in evidence

    def test_case_insensitive(self):
        verdict, _ = classify(0, "UNSUPPORTED OP SCAN")
        assert verdict == "UNSUPPORTED_OP"

    def test_marker_takes_priority_over_nonzero_rc(self):
        """Even with a non-zero returncode, an unsupported-op marker wins."""
        verdict, _ = classify(1, "Error: unsupported op Loop")
        assert verdict == "UNSUPPORTED_OP"

    def test_marker_in_multiline_log(self):
        log = "[INFO] Parsing model\n[ERROR] unsupported operator CumSum\n[INFO] Done"
        verdict, evidence = classify(0, log)
        assert verdict == "UNSUPPORTED_OP"
        assert "CumSum" in evidence

    def test_first_marker_found_is_used(self):
        """When multiple markers exist, the first in the marker list is used."""
        log = "not supported\nunknown op"
        verdict, evidence = classify(0, log)
        assert verdict == "UNSUPPORTED_OP"
        assert "not supported" in evidence

    def test_evidence_truncated_to_300_chars(self):
        long_msg = "unsupported op " + "x" * 500
        _, evidence = classify(0, long_msg)
        assert len(evidence) <= 300


# ---------------------------------------------------------------------------
# FAILED verdicts
# ---------------------------------------------------------------------------


class TestFailed:
    def test_nonzero_rc_with_error_in_log(self):
        log = "Loading model...\nFatal error: memory allocation failed\n"
        verdict, evidence = classify(1, log)
        assert verdict == "FAILED"
        assert "error" in evidence.lower()

    def test_nonzero_rc_picks_last_error_line(self):
        """The last line matching error|fail|exception is the evidence."""
        log = "error at step 1\nworking...\nerror at step 5\nall good"
        verdict, evidence = classify(1, log)
        assert verdict == "FAILED"
        assert "step 5" in evidence

    def test_nonzero_rc_no_error_keyword_falls_back_to_last_line(self):
        log = "Loading model\nProcessing\nSegmentation fault"
        verdict, evidence = classify(2, log)
        assert verdict == "FAILED"
        assert "Segmentation fault" in evidence

    def test_nonzero_rc_empty_log(self):
        verdict, evidence = classify(1, "")
        assert verdict == "FAILED"
        assert evidence == ""

    def test_nonzero_rc_whitespace_only_log(self):
        verdict, evidence = classify(1, "   \n\n   ")
        assert verdict == "FAILED"
        assert evidence == ""

    def test_failed_keyword_match(self):
        log = "Parsing model\nBuild failed: missing input\n"
        verdict, _ = classify(1, log)
        assert verdict == "FAILED"

    def test_exception_keyword_match(self):
        log = "Parsing model\nRuntimeError: unhandled exception in graph\n"
        verdict, _ = classify(1, log)
        assert verdict == "FAILED"


# ---------------------------------------------------------------------------
# OK_WITH_WARNINGS verdicts
# ---------------------------------------------------------------------------


class TestOkWithWarnings:
    def test_rc_zero_with_warning(self):
        log = "Parsing model\nWarning: deprecated attribute\nBuild complete\n"
        verdict, evidence = classify(0, log)
        assert verdict == "OK_WITH_WARNINGS"
        assert "warning" in evidence.lower()

    def test_case_insensitive_warning(self):
        log = "WARNING: unsupported attribute\n"
        verdict, _ = classify(0, log)
        assert verdict == "OK_WITH_WARNINGS"

    def test_warning_in_multiline(self):
        log = "[INFO] Model parsed\n[WARNING] Shape inference issue\n[INFO] Build OK"
        verdict, evidence = classify(0, log)
        assert verdict == "OK_WITH_WARNINGS"
        assert "warn" in evidence.lower()


# ---------------------------------------------------------------------------
# OK verdicts
# ---------------------------------------------------------------------------


class TestOk:
    def test_clean_success(self):
        log = "Parsing model\nBuilding\nBuild complete\n"
        verdict, evidence = classify(0, log)
        assert verdict == "OK"
        assert evidence == "Build complete"

    def test_empty_log(self):
        verdict, evidence = classify(0, "")
        assert verdict == "OK"
        assert evidence == ""

    def test_whitespace_only_log(self):
        verdict, evidence = classify(0, "   \n\n   ")
        assert verdict == "OK"
        assert evidence == ""

    def test_info_only_log(self):
        log = "[INFO] Model loaded\n[INFO] All checks passed\n"
        verdict, _ = classify(0, log)
        assert verdict == "OK"


# ---------------------------------------------------------------------------
# Priority / edge cases
# ---------------------------------------------------------------------------


class TestPriorityAndEdgeCases:
    def test_unsupported_op_beats_warning(self):
        """UNSUPPORTED_OP has higher priority than OK_WITH_WARNINGS."""
        log = "Warning: something\nunsupported op Scan\n"
        verdict, _ = classify(0, log)
        assert verdict == "UNSUPPORTED_OP"

    def test_unsupported_op_beats_ok(self):
        log = "Build complete\nunsupported operator Loop\n"
        verdict, _ = classify(0, log)
        assert verdict == "UNSUPPORTED_OP"

    def test_failed_beats_warning_when_no_unsupported(self):
        """Non-zero rc with warning but no unsupported marker = FAILED."""
        log = "Warning: deprecated\nError: build failed\n"
        verdict, _ = classify(1, log)
        assert verdict == "FAILED"

    def test_strips_whitespace_from_evidence(self):
        log = "  unsupported op Conv  \n"
        _, evidence = classify(0, log)
        assert evidence == "unsupported op Conv"

    def test_all_verdicts_returned(self):
        """Sanity check: all four verdict values are reachable."""
        verdicts = set()
        verdicts.add(classify(0, "all good")[0])  # OK
        verdicts.add(classify(0, "warning: something")[0])  # OK_WITH_WARNINGS
        verdicts.add(classify(1, "error: crash")[0])  # FAILED
        verdicts.add(classify(0, "unsupported op Scan")[0])  # UNSUPPORTED_OP
        assert verdicts == {"OK", "OK_WITH_WARNINGS", "FAILED", "UNSUPPORTED_OP"}

    @pytest.mark.parametrize(
        "rc,log,expected",
        [
            (0, "all good", "OK"),
            (0, "warning: x", "OK_WITH_WARNINGS"),
            (1, "error: x", "FAILED"),
            (0, "unsupported op x", "UNSUPPORTED_OP"),
            (0, "", "OK"),
            (1, "", "FAILED"),
        ],
    )
    def test_matrix(self, rc, log, expected):
        verdict, _ = classify(rc, log)
        assert verdict == expected
