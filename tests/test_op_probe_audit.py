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

from run_op_probe_audit import cfg_for, classify  # noqa: E402

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


# ---------------------------------------------------------------------------
# cfg_for() — ONNX model → NOE Common-section cfg
# ---------------------------------------------------------------------------
@pytest.fixture
def probe_onnx_models(tmp_path):
    """Generate small ONNX probe models and return (path, name) pairs."""
    import onnx

    from npu_op_probe import (
        probe_causal_conv1d,
        probe_decay_cumprod,
        probe_delta_rule_update,
        probe_gate_chain,
        probe_scan_recurrence,
    )

    probes = [
        ("causal_conv1d", probe_causal_conv1d),
        ("decay_cumprod", probe_decay_cumprod),
        ("delta_rule_update", probe_delta_rule_update),
        ("gate_chain", probe_gate_chain),
        ("scan_recurrence", probe_scan_recurrence),
    ]
    models = []
    for name, func in probes:
        result = func()
        model = result[0] if isinstance(result, tuple) else result
        path = tmp_path / f"{name}.onnx"
        onnx.save(model, str(path))
        models.append((path, name))
    return models


class TestCfgFor:
    def test_returns_string(self, probe_onnx_models):
        path, name = probe_onnx_models[0]
        cfg = cfg_for(path, name)
        assert isinstance(cfg, str)

    def test_has_common_section(self, probe_onnx_models):
        path, name = probe_onnx_models[0]
        cfg = cfg_for(path, name)
        assert "[Common]" in cfg

    def test_has_mode_build(self, probe_onnx_models):
        path, name = probe_onnx_models[0]
        cfg = cfg_for(path, name)
        assert "mode=build" in cfg

    def test_contains_model_name(self, probe_onnx_models):
        path, name = probe_onnx_models[0]
        cfg = cfg_for(path, name)
        assert f"model_name={name}" in cfg

    def test_contains_model_path(self, probe_onnx_models):
        path, name = probe_onnx_models[0]
        cfg = cfg_for(path, name)
        assert f"input_model={path.resolve()}" in cfg

    def test_has_input_and_output(self, probe_onnx_models):
        path, name = probe_onnx_models[0]
        cfg = cfg_for(path, name)
        assert "input=" in cfg
        assert "output=" in cfg

    def test_has_input_shape(self, probe_onnx_models):
        path, name = probe_onnx_models[0]
        cfg = cfg_for(path, name)
        assert "input_shape=" in cfg

    def test_excludes_initializers_from_inputs(self, probe_onnx_models):
        """Initializers (weights) must not appear in input declarations."""
        path, name = probe_onnx_models[0]
        cfg = cfg_for(path, name)
        import onnx

        model = onnx.load(str(path))
        initializer_names = {i.name for i in model.graph.initializer}
        # The input= line should not reference any initializer name
        for init_name in initializer_names:
            # input= should not start with or contain the initializer
            for line in cfg.splitlines():
                if line.startswith("input="):
                    assert init_name not in line.split("=", 1)[1]

    def test_all_probes_produce_valid_cfg(self, probe_onnx_models):
        """Every probe model generates a well-formed cfg."""
        for path, name in probe_onnx_models:
            cfg = cfg_for(path, name)
            assert "[Common]" in cfg
            assert f"model_name={name}" in cfg
            assert "mode=build" in cfg
            assert "input=" in cfg
            assert "output=" in cfg
            assert "input_shape=" in cfg

    def test_cfg_lines_are_newline_separated(self, probe_onnx_models):
        path, name = probe_onnx_models[0]
        cfg = cfg_for(path, name)
        lines = cfg.strip().split("\n")
        assert len(lines) >= 5  # [Common], mode, model_name, input_model, input, ...

    def test_cfg_ends_with_empty_line(self, probe_onnx_models):
        """cfg should end with a blank line (trailing newline)."""
        path, name = probe_onnx_models[0]
        cfg = cfg_for(path, name)
        assert cfg.endswith("\n")

    def test_input_shape_bracket_format(self, probe_onnx_models):
        """input_shape values use [dim,dim] bracket notation."""
        path, name = probe_onnx_models[0]
        cfg = cfg_for(path, name)
        assert "input_shape=[" in cfg
        assert "]" in cfg

    def test_different_names_produce_different_cfgs(self, probe_onnx_models):
        """Same model with different names → different model_name in cfg."""
        path, _ = probe_onnx_models[0]
        cfg_a = cfg_for(path, "alpha")
        cfg_b = cfg_for(path, "beta")
        assert "model_name=alpha" in cfg_a
        assert "model_name=beta" in cfg_b
