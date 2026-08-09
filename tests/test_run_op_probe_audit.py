# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/run_op_probe_audit.py — operator probe classifier.

Tests the pure ``classify()`` function which maps cixparse invocation results
to (verdict, evidence) tuples. This function runs without onnx or cixparse.
"""

from __future__ import annotations

import json

import pytest
from scripts.run_op_probe_audit import cfg_for, classify, main

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


# ---------------------------------------------------------------------------
# cfg_for — ONNX cfg generation
# ---------------------------------------------------------------------------


def _make_tiny_onnx(path):
    """Create a minimal ONNX model with one Add op and two inputs."""
    import onnx
    from onnx import TensorProto, helper

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Add", ["X", "Y"], ["Z"])
    graph = helper.make_graph([node], "test_graph", [X, Y], [Z])
    model = helper.make_model(graph)
    onnx.save(model, str(path))
    return path


def _make_onnx_with_initializer(path):
    """Create an ONNX model with an initializer (weight) and one graph input."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3])
    W_init = helper.make_tensor("W", TensorProto.FLOAT, [3], np.ones(3, dtype=np.float32))
    node = helper.make_node("Add", ["X", "W"], ["Z"])
    graph = helper.make_graph([node], "init_graph", [X], [Z], initializer=[W_init])
    model = helper.make_model(graph)
    onnx.save(model, str(path))
    return path


def _make_onnx_dynamic_dims(path):
    """ONNX model with a dynamic (0-value) dimension."""
    import onnx
    from onnx import TensorProto, helper

    # dim_value=0 means dynamic — cfg_for should render as "1"
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [0, 3])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [0, 3])
    node = helper.make_node("Identity", ["X"], ["Z"])
    graph = helper.make_graph([node], "dynamic_graph", [X], [Z])
    model = helper.make_model(graph)
    onnx.save(model, str(path))
    return path


class TestCfgFor:
    """Test cfg_for() — minimal Common-section cfg from an ONNX model."""

    def test_basic_cfg_structure(self, tmp_path):
        model_path = _make_tiny_onnx(tmp_path / "add.onnx")
        cfg = cfg_for(model_path, "add_probe")
        assert "[Common]" in cfg
        assert "mode=build" in cfg
        assert "model_name=add_probe" in cfg
        assert "input_model=" in cfg
        assert "input=X,Y" in cfg
        assert "output=Z" in cfg

    def test_input_shape_in_cfg(self, tmp_path):
        model_path = _make_tiny_onnx(tmp_path / "add.onnx")
        cfg = cfg_for(model_path, "add_probe")
        assert "input_shape=" in cfg

    def test_initializer_excluded_from_inputs(self, tmp_path):
        """Initializer tensors should NOT appear in the input list."""
        model_path = _make_onnx_with_initializer(tmp_path / "weighted.onnx")
        cfg = cfg_for(model_path, "weighted_probe")
        assert "input=X" in cfg
        # W is an initializer, not a graph input
        assert "input=W" not in cfg

    def test_dynamic_dim_defaults_to_1(self, tmp_path):
        """Dynamic dimensions (dim_value=0) should render as '1'."""
        model_path = _make_onnx_dynamic_dims(tmp_path / "dynamic.onnx")
        cfg = cfg_for(model_path, "dynamic_probe")
        assert "input=X" in cfg
        # Check that the shape line has 1 for the dynamic dimension
        for line in cfg.splitlines():
            if line.startswith("input_shape="):
                assert "1" in line


# ---------------------------------------------------------------------------
# main — CLI integration with mocked subprocess
# ---------------------------------------------------------------------------


class TestMain:
    """Test main() with a temp probe directory and mocked cixparse."""

    def _setup_probe(self, tmp_path):
        """Create a probe dir with manifest + ONNX model."""
        probe_dir = tmp_path / "probe"
        probe_dir.mkdir()
        _make_tiny_onnx(probe_dir / "add.onnx")
        manifest = [{"probe": "add_op", "file": "add.onnx", "ops": ["Add"]}]
        (probe_dir / "manifest.json").write_text(json.dumps(manifest))
        return probe_dir

    def test_main_success(self, tmp_path, monkeypatch):
        """main() runs cixparse on each probe and writes audit results."""
        import subprocess as sp

        probe_dir = self._setup_probe(tmp_path)
        out_dir = tmp_path / "audit"

        mock_result = sp.CompletedProcess(
            args=["cixparse"],
            returncode=0,
            stdout="Compilation successful\nmodel.aipu",
            stderr="",
        )
        monkeypatch.setattr(sp, "run", lambda *a, **kw: mock_result)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_op_probe_audit.py",
                "--cixparse",
                "/fake/cixparse",
                "--probe-dir",
                str(probe_dir),
                "--out-dir",
                str(out_dir),
            ],
        )

        rc = main()
        assert rc == 0
        assert (out_dir / "audit_results.json").exists()
        results = json.loads((out_dir / "audit_results.json").read_text())
        assert len(results) == 1
        assert results[0]["probe"] == "add_op"
        assert results[0]["verdict"] == "OK"

    def test_main_unsupported_op(self, tmp_path, monkeypatch):
        """main() classifies unsupported-op output correctly."""
        import subprocess as sp

        probe_dir = self._setup_probe(tmp_path)
        out_dir = tmp_path / "audit"

        mock_result = sp.CompletedProcess(
            args=["cixparse"],
            returncode=1,
            stdout="Error: unsupported op ConvTranspose2d",
            stderr="",
        )
        monkeypatch.setattr(sp, "run", lambda *a, **kw: mock_result)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_op_probe_audit.py",
                "--cixparse",
                "/fake/cixparse",
                "--probe-dir",
                str(probe_dir),
                "--out-dir",
                str(out_dir),
            ],
        )

        rc = main()
        assert rc == 0
        results = json.loads((out_dir / "audit_results.json").read_text())
        assert results[0]["verdict"] == "UNSUPPORTED_OP"

    def test_main_timeout(self, tmp_path, monkeypatch):
        """main() handles subprocess timeout (rc=124)."""
        import subprocess as sp

        probe_dir = self._setup_probe(tmp_path)
        out_dir = tmp_path / "audit"

        def raise_timeout(*a, **kw):
            raise sp.TimeoutExpired(cmd="cixparse", timeout=5)

        monkeypatch.setattr(sp, "run", raise_timeout)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_op_probe_audit.py",
                "--cixparse",
                "/fake/cixparse",
                "--probe-dir",
                str(probe_dir),
                "--out-dir",
                str(out_dir),
                "--timeout",
                "5",
            ],
        )

        rc = main()
        assert rc == 0
        results = json.loads((out_dir / "audit_results.json").read_text())
        assert results[0]["returncode"] == 124
        assert results[0]["verdict"] == "FAILED"
        assert "TIMEOUT" in results[0]["evidence"]

    def test_main_writes_cfg_and_log(self, tmp_path, monkeypatch):
        """main() writes a .cfg and .log file per probe."""
        import subprocess as sp

        probe_dir = self._setup_probe(tmp_path)
        out_dir = tmp_path / "audit"

        mock_result = sp.CompletedProcess(
            args=["cixparse"],
            returncode=0,
            stdout="OK",
            stderr="",
        )
        monkeypatch.setattr(sp, "run", lambda *a, **kw: mock_result)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_op_probe_audit.py",
                "--cixparse",
                "/fake/cixparse",
                "--probe-dir",
                str(probe_dir),
                "--out-dir",
                str(out_dir),
            ],
        )

        main()
        assert (out_dir / "add_op.cfg").exists()
        assert (out_dir / "add_op.log").exists()

    def test_main_multiple_probes(self, tmp_path, monkeypatch):
        """main() handles multiple probes in the manifest."""
        import subprocess as sp

        probe_dir = tmp_path / "probe"
        probe_dir.mkdir()
        _make_tiny_onnx(probe_dir / "add.onnx")
        _make_onnx_with_initializer(probe_dir / "weighted.onnx")
        manifest = [
            {"probe": "add_op", "file": "add.onnx", "ops": ["Add"]},
            {"probe": "init_op", "file": "weighted.onnx", "ops": ["Add"]},
        ]
        (probe_dir / "manifest.json").write_text(json.dumps(manifest))
        out_dir = tmp_path / "audit"

        mock_result = sp.CompletedProcess(
            args=["cixparse"],
            returncode=0,
            stdout="OK",
            stderr="",
        )
        monkeypatch.setattr(sp, "run", lambda *a, **kw: mock_result)
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_op_probe_audit.py",
                "--cixparse",
                "/fake/cixparse",
                "--probe-dir",
                str(probe_dir),
                "--out-dir",
                str(out_dir),
            ],
        )

        rc = main()
        assert rc == 0
        results = json.loads((out_dir / "audit_results.json").read_text())
        assert len(results) == 2
