"""Tests for scripts/rknn_op_probe.py — RKNN output parsing logic.

The two pure-Python parsing functions, ``extract_op_table`` and
``extract_key_evidence``, are the decision-making core of the RKNN
operator-coverage audit (bead ob-t3b.5). They determine whether the
RK3588's NPU actually hosts each ONNX op or silently falls back to CPU,
and they extract the evidence lines that justify each verdict.

These functions only need string input/output; no RKNN toolkit required.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from rknn_op_probe import extract_key_evidence, extract_op_table

# ---------------------------------------------------------------------------
# extract_op_table
# ---------------------------------------------------------------------------


class TestExtractOpTable:
    """Parse the 'Network Layer Information Table' from RKNN verbose output."""

    def test_basic_table(self):
        """A clean table with two NPU ops."""
        output = """
D RKNN: [10:00:01.234] N Network Layer Information Table
D RKNN: [10:00:01.234] ---
D RKNN: [10:00:01.234] ID  Op_Type        Dtype      Target
D RKNN: [10:00:01.234] ---
D RKNN: [10:00:01.234] 0   Conv           float16    NPU
D RKNN: [10:00:01.234] 1   Add            float16    NPU
D RKNN: [10:00:01.234] ---
"""
        ops = extract_op_table(output)
        assert len(ops) == 2
        assert ops[0] == {"id": 0, "op_type": "Conv", "dtype": "float16", "target": "NPU"}
        assert ops[1] == {"id": 1, "op_type": "Add", "dtype": "float16", "target": "NPU"}

    def test_cpu_fallback_detected(self):
        """A table with a CPU fallback op should be captured."""
        output = """
D RKNN: [10:00:01.234] Network Layer Information Table
---
ID  Op_Type        Dtype      Target
---
0   Conv           float16    NPU
1   CumSum         float32    CPU
2   Mul            float16    NPU
---
"""
        ops = extract_op_table(output)
        assert len(ops) == 3
        assert ops[1]["target"] == "CPU"
        assert ops[1]["op_type"] == "CumSum"

    def test_empty_output(self):
        """Empty string yields empty list."""
        assert extract_op_table("") == []

    def test_no_table_found(self):
        """Output without the table marker yields empty list."""
        output = "Some other RKNN output\nwithout the table\n"
        assert extract_op_table(output) == []

    def test_ansi_color_codes_stripped(self):
        """ANSI color codes in the table should be stripped before parsing."""
        output = """
\x1b[32mD RKNN: [10:00:01.234]\x1b[0m Network Layer Information Table
---
ID  Op_Type        Dtype      Target
---
\x1b[33m0   Conv           float16    NPU\x1b[0m
---
"""
        ops = extract_op_table(output)
        assert len(ops) == 1
        assert ops[0]["op_type"] == "Conv"

    def test_skips_header_row(self):
        """The 'ID Op_Type ...' header row should not be parsed as an op."""
        output = """
Network Layer Information Table
---
ID  Op_Type        Dtype      Target
---
0   Conv           float16    NPU
---
"""
        ops = extract_op_table(output)
        assert len(ops) == 1
        assert ops[0]["id"] == 0

    def test_multiple_separator_lines_before_data(self):
        """Multiple dashed lines before data should be handled gracefully."""
        output = """
Network Layer Information Table
---
---
---
0   Conv           float16    NPU
---
"""
        ops = extract_op_table(output)
        assert len(ops) == 1

    def test_id_must_be_numeric(self):
        """Lines where the first field is not a digit are skipped."""
        output = """
Network Layer Information Table
---
ID  Op_Type        Dtype      Target
---
0   Conv           float16    NPU
foo Bar           float16    NPU
---
"""
        ops = extract_op_table(output)
        assert len(ops) == 1

    def test_stops_after_data_separator(self):
        """Parsing should stop at the first separator line AFTER ops are found."""
        output = """
Network Layer Information Table
---
ID  Op_Type        Dtype      Target
---
0   Conv           float16    NPU
1   Add            float16    NPU
---
Some trailing text that looks like data
99  Ghost          float16    NPU
"""
        ops = extract_op_table(output)
        assert len(ops) == 2
        assert all(o["id"] in (0, 1) for o in ops)

    def test_log_prefix_stripped(self):
        """'D RKNN: [HH:MM:SS.mmm]' prefix should be stripped from each line."""
        output = "D RKNN: [10:00:01.234] Network Layer Information Table\n"
        output += "D RKNN: [10:00:01.234] ---\n"
        output += "D RKNN: [10:00:01.234] ID  Op_Type  Dtype  Target\n"
        output += "D RKNN: [10:00:01.234] ---\n"
        output += "D RKNN: [10:00:01.234] 0   Conv      float16    NPU\n"
        output += "D RKNN: [10:00:01.234] ---\n"
        ops = extract_op_table(output)
        assert len(ops) == 1
        assert ops[0]["op_type"] == "Conv"

    def test_different_log_levels(self):
        """I, W, E log level prefixes should all be stripped."""
        output = """
I RKNN: [10:00:01.234] Network Layer Information Table
---
ID  Op_Type  Dtype  Target
---
0   Conv      float16    NPU
---
"""
        ops = extract_op_table(output)
        assert len(ops) == 1

    def test_extra_columns_ignored(self):
        """Lines with more than 4 columns should still parse the first 4."""
        output = """
Network Layer Information Table
---
ID  Op_Type  Dtype  Target  Extra1  Extra2
---
0   Conv      float16    NPU    100    200
---
"""
        ops = extract_op_table(output)
        assert len(ops) == 1
        assert ops[0]["op_type"] == "Conv"


# ---------------------------------------------------------------------------
# extract_key_evidence
# ---------------------------------------------------------------------------


class TestExtractKeyEvidence:
    """Extract informative lines from RKNN output for the results table."""

    def test_finds_unsupported(self):
        output = "Building model...\nunsupported op Scan\nDone."
        evidence = extract_key_evidence(output)
        assert any("unsupported op Scan" in e for e in evidence)

    def test_finds_ret_codes(self):
        output = "CONFIG_RET=0\nLOAD_RET=0\nBUILD_RET=-1"
        evidence = extract_key_evidence(output)
        assert any("CONFIG_RET=0" in e for e in evidence)
        assert any("BUILD_RET=-1" in e for e in evidence)

    def test_finds_file_size(self):
        output = "RKNN_FILE_SIZE=4567890"
        evidence = extract_key_evidence(output)
        assert any("RKNN_FILE_SIZE=4567890" in e for e in evidence)

    def test_case_insensitive_keywords(self):
        """Both 'error' and 'ERROR' should match."""
        output = "error: something\nERROR: another thing"
        evidence = extract_key_evidence(output)
        assert len(evidence) == 2

    def test_finds_graph_not_dag(self):
        output = "Graph is not DAG\nCannot build model"
        evidence = extract_key_evidence(output)
        assert any("Graph is not DAG" in e for e in evidence)
        assert any("Cannot build" in e for e in evidence)

    def test_finds_fallback(self):
        output = "Op CumSum fallback to CPU\nop resolved"
        evidence = extract_key_evidence(output)
        assert any("fallback" in e for e in evidence)

    def test_empty_output(self):
        assert extract_key_evidence("") == []

    def test_no_keywords(self):
        """Lines without any keyword should not be included."""
        output = "Building model...\nStep 1 complete\nStep 2 complete\n"
        evidence = extract_key_evidence(output)
        assert evidence == []

    def test_ansi_codes_stripped(self):
        """ANSI color codes should be stripped from evidence lines."""
        output = "\x1b[31munsupported op Loop\x1b[0m"
        evidence = extract_key_evidence(output)
        assert len(evidence) == 1
        assert "\x1b[31m" not in evidence[0]
        assert "unsupported op Loop" in evidence[0]

    def test_lines_truncated_to_200_chars(self):
        """Evidence lines should be truncated to 200 characters."""
        long_line = "unsupported " + "x" * 300
        evidence = extract_key_evidence(long_line)
        assert len(evidence) == 1
        assert len(evidence[0]) <= 200

    def test_max_30_lines(self):
        """At most 30 evidence lines should be returned."""
        lines = [f"unsupported op type_{i}" for i in range(50)]
        evidence = extract_key_evidence("\n".join(lines))
        assert len(evidence) == 30

    def test_dedup_not_performed(self):
        """The function does not deduplicate; each matching line is kept."""
        output = "error: first\nerror: second"
        evidence = extract_key_evidence(output)
        assert len(evidence) == 2

    def test_failed_keyword(self):
        output = "Build failed: internal error"
        evidence = extract_key_evidence(output)
        assert len(evidence) >= 1

    def test_cannot_keyword(self):
        output = "Cannot find operator CumSum"
        evidence = extract_key_evidence(output)
        assert len(evidence) >= 1

    def test_not_found_keyword(self):
        output = "Op type 'Scan' not found in converter"
        evidence = extract_key_evidence(output)
        assert len(evidence) >= 1


# ---------------------------------------------------------------------------
# probe_one() — mocked subprocess.run
# ---------------------------------------------------------------------------

import json  # noqa: E402
from unittest.mock import patch  # noqa: E402

import rknn_op_probe  # noqa: E402


class _FakeProc:
    """Minimal stub for subprocess.run return value."""

    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class TestProbeOne:
    @pytest.fixture(autouse=True)
    def _setup_out_dir(self, tmp_path, monkeypatch):
        """Redirect OUT_DIR to a temp path so log files don't pollute cwd."""
        monkeypatch.setattr(rknn_op_probe, "OUT_DIR", tmp_path)

    def test_compiled_success(self, tmp_path):
        """All return codes zero → COMPILED verdict."""
        stderr = "CONFIG_RET=0\nLOAD_RET=0\nBUILD_RET=0\nEXPORT_RET=0\nRKNN_FILE_SIZE=4567890\n"
        with patch("rknn_op_probe.subprocess.run") as mock_run:
            mock_run.return_value = _FakeProc(stderr=stderr, returncode=0)
            result = rknn_op_probe.probe_one("/fake/model.onnx", "scan_recurrence")

        assert result["verdict"] == "COMPILED"
        assert result["config_ret"] == 0
        assert result["load_ret"] == 0
        assert result["build_ret"] == 0
        assert result["export_ret"] == 0
        assert result["rknn_file_size"] == 4567890
        assert result["log"] == "scan_recurrence.log"

    def test_rejected_at_load(self, tmp_path):
        """load_ret != 0 → REJECTED_AT_LOAD."""
        stderr = "CONFIG_RET=0\nLOAD_RET=1\n"
        with patch("rknn_op_probe.subprocess.run") as mock_run:
            mock_run.return_value = _FakeProc(stderr=stderr, returncode=0)
            result = rknn_op_probe.probe_one("/fake/model.onnx", "scan_recurrence")

        assert result["verdict"] == "REJECTED_AT_LOAD"
        assert result["load_ret"] == 1
        assert result["build_ret"] is None

    def test_rejected_at_build(self, tmp_path):
        """load ok but build_ret != 0 → REJECTED_AT_BUILD."""
        stderr = "CONFIG_RET=0\nLOAD_RET=0\nBUILD_RET=1\n"
        with patch("rknn_op_probe.subprocess.run") as mock_run:
            mock_run.return_value = _FakeProc(stderr=stderr, returncode=0)
            result = rknn_op_probe.probe_one("/fake/model.onnx", "conv_probe")

        assert result["verdict"] == "REJECTED_AT_BUILD"
        assert result["build_ret"] == 1

    def test_compiled_no_export(self, tmp_path):
        """load + build ok but export_ret != 0 → COMPILED_NO_EXPORT."""
        stderr = "CONFIG_RET=0\nLOAD_RET=0\nBUILD_RET=0\nEXPORT_RET=1\n"
        with patch("rknn_op_probe.subprocess.run") as mock_run:
            mock_run.return_value = _FakeProc(stderr=stderr, returncode=0)
            result = rknn_op_probe.probe_one("/fake/model.onnx", "conv_probe")

        assert result["verdict"] == "COMPILED_NO_EXPORT"

    def test_missing_return_codes_are_none(self, tmp_path):
        """When cixparse doesn't emit _RET= lines, the fields are None."""
        with patch("rknn_op_probe.subprocess.run") as mock_run:
            mock_run.return_value = _FakeProc(stdout="some output\n", returncode=0)
            result = rknn_op_probe.probe_one("/fake/model.onnx", "conv_probe")

        assert result["config_ret"] is None
        assert result["load_ret"] is None
        assert result["build_ret"] is None
        assert result["export_ret"] is None
        assert result["rknn_file_size"] is None
        # No return codes → load_ret is None, which is != 0 → REJECTED_AT_LOAD
        assert result["verdict"] == "REJECTED_AT_LOAD"

    def test_log_file_written(self, tmp_path):
        """The combined stdout+stderr is saved to {probe_name}.log."""
        stdout_text = "stdout line\n"
        stderr_text = "stderr line\nLOAD_RET=0\nBUILD_RET=0\nEXPORT_RET=0\n"
        with patch("rknn_op_probe.subprocess.run") as mock_run:
            mock_run.return_value = _FakeProc(stdout=stdout_text, stderr=stderr_text)
            rknn_op_probe.probe_one("/fake/model.onnx", "my_probe")

        log_file = tmp_path / "my_probe.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "stdout line" in content
        assert "stderr line" in content

    def test_op_table_extracted_from_output(self, tmp_path):
        """Op table from RKNN output is parsed and included in result."""
        op_table = """Network Layer Information Table
---
ID  Op_Type        Dtype      Target
---
0   Conv           float16    NPU
1   CumSum         float32    CPU
2   Mul            float16    NPU
---
"""
        stderr_text = f"{op_table}LOAD_RET=0\nBUILD_RET=0\nEXPORT_RET=0\n"
        with patch("rknn_op_probe.subprocess.run") as mock_run:
            mock_run.return_value = _FakeProc(stderr=stderr_text)
            result = rknn_op_probe.probe_one("/fake/model.onnx", "conv_probe")

        assert len(result["op_table"]) == 3
        # CPU fallback detection
        assert result["cpu_fallback_ops"] == ["CumSum:1"]
        assert "Conv:0" in result["npu_ops"]
        assert "Mul:2" in result["npu_ops"]

    def test_all_npu_no_fallback(self, tmp_path):
        """All NPU ops → empty cpu_fallback_ops list."""
        op_table = """Network Layer Information Table
---
ID  Op_Type        Dtype      Target
---
0   Conv           float16    NPU
---
LOAD_RET=0
BUILD_RET=0
EXPORT_RET=0
"""
        with patch("rknn_op_probe.subprocess.run") as mock_run:
            mock_run.return_value = _FakeProc(stderr=op_table)
            result = rknn_op_probe.probe_one("/fake/model.onnx", "conv_probe")

        assert result["cpu_fallback_ops"] == []

    def test_evidence_extracted(self, tmp_path):
        """Key evidence lines from output are captured."""
        stderr_text = "LOAD_RET=0\nunsupported op Scan\nBUILD_RET=0\nEXPORT_RET=0\n"
        with patch("rknn_op_probe.subprocess.run") as mock_run:
            mock_run.return_value = _FakeProc(stderr=stderr_text)
            result = rknn_op_probe.probe_one("/fake/model.onnx", "scan_probe")

        assert any("unsupported op Scan" in e for e in result["evidence"])

    def test_cpu_fallback_excludes_io_operators(self, tmp_path):
        """InputOperator/OutputOperator on CPU should NOT count as fallback."""
        op_table = """Network Layer Information Table
---
ID  Op_Type            Dtype      Target
---
0   InputOperator      float16    CPU
1   Conv               float16    NPU
2   OutputOperator     float16    CPU
---
LOAD_RET=0
BUILD_RET=0
EXPORT_RET=0
"""
        with patch("rknn_op_probe.subprocess.run") as mock_run:
            mock_run.return_value = _FakeProc(stderr=op_table)
            result = rknn_op_probe.probe_one("/fake/model.onnx", "conv_probe")

        assert result["cpu_fallback_ops"] == []

    def test_probe_name_in_result(self, tmp_path):
        """The probe name is echoed in the result."""
        with patch("rknn_op_probe.subprocess.run") as mock_run:
            mock_run.return_value = _FakeProc(stderr="LOAD_RET=0\nBUILD_RET=0\nEXPORT_RET=0\n")
            result = rknn_op_probe.probe_one("/fake/m.onnx", "special_probe")

        assert result["probe"] == "special_probe"

    def test_subprocess_called_with_python(self, tmp_path):
        """probe_one invokes subprocess with sys.executable and -c."""
        with patch("rknn_op_probe.subprocess.run") as mock_run:
            mock_run.return_value = _FakeProc(stderr="LOAD_RET=0\nBUILD_RET=0\nEXPORT_RET=0\n")
            rknn_op_probe.probe_one("/fake/m.onnx", "p1")

        call_args = mock_run.call_args[0][0]
        assert call_args[0] == sys.executable
        assert call_args[1] == "-c"


# ---------------------------------------------------------------------------
# main() — CLI with mocked probe_one
# ---------------------------------------------------------------------------


class TestMainCLI:
    def test_reads_manifest_and_calls_probe_one(self, tmp_path, monkeypatch):
        """main() reads manifest.json and calls probe_one for each entry."""
        probe_dir = tmp_path / "probes"
        probe_dir.mkdir()
        manifest = [
            {
                "probe": "conv_probe",
                "file": "conv.onnx",
                "ops": ["Conv"],
                "description": "Conv test",
            },
            {
                "probe": "scan_probe",
                "file": "scan.onnx",
                "ops": ["Scan"],
                "description": "Scan test",
            },
        ]
        manifest_path = probe_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        # Create fake ONNX files so os.path.exists returns True
        (probe_dir / "conv.onnx").write_text("fake")
        (probe_dir / "scan.onnx").write_text("fake")

        monkeypatch.setattr(rknn_op_probe, "PROBE_DIR", probe_dir)
        monkeypatch.setattr(rknn_op_probe, "MANIFEST", manifest_path)
        monkeypatch.setattr(rknn_op_probe, "OUT_DIR", tmp_path / "audit")

        def fake_probe_one(onnx_path, probe_name):
            return {
                "probe": probe_name,
                "config_ret": 0,
                "load_ret": 0,
                "build_ret": 0,
                "export_ret": 0,
                "rknn_file_size": 1000,
                "op_table": [],
                "evidence": [],
                "log": f"{probe_name}.log",
                "verdict": "COMPILED",
            }

        with patch("rknn_op_probe.probe_one", side_effect=fake_probe_one):
            rknn_op_probe.main()

        results_file = tmp_path / "audit" / "rknn_audit_results.json"
        assert results_file.exists()
        results = json.loads(results_file.read_text())
        assert len(results) == 2
        assert all(r["verdict"] == "COMPILED" for r in results)

    def test_missing_file_skipped(self, tmp_path, monkeypatch):
        """A probe whose ONNX file is missing is skipped with a message."""
        probe_dir = tmp_path / "probes"
        probe_dir.mkdir()
        manifest = [
            {"probe": "ghost", "file": "ghost.onnx", "ops": ["Ghost"], "description": "no file"}
        ]
        (probe_dir / "manifest.json").write_text(json.dumps(manifest))

        monkeypatch.setattr(rknn_op_probe, "PROBE_DIR", probe_dir)
        monkeypatch.setattr(rknn_op_probe, "MANIFEST", probe_dir / "manifest.json")
        monkeypatch.setattr(rknn_op_probe, "OUT_DIR", tmp_path / "audit")

        with patch("rknn_op_probe.probe_one") as mock_probe:
            rknn_op_probe.main()

        mock_probe.assert_not_called()
        # No results written (empty list)
        results_file = tmp_path / "audit" / "rknn_audit_results.json"
        assert results_file.exists()
        results = json.loads(results_file.read_text())
        assert len(results) == 0

    def test_cpu_fallback_shown_in_summary(self, tmp_path, monkeypatch, capsys):
        """Probes with CPU fallbacks get 'silent fallback' note."""
        probe_dir = tmp_path / "probes"
        probe_dir.mkdir()
        manifest = [{"probe": "scan", "file": "scan.onnx", "ops": ["Scan"], "description": "test"}]
        (probe_dir / "manifest.json").write_text(json.dumps(manifest))
        (probe_dir / "scan.onnx").write_text("fake")

        monkeypatch.setattr(rknn_op_probe, "PROBE_DIR", probe_dir)
        monkeypatch.setattr(rknn_op_probe, "MANIFEST", probe_dir / "manifest.json")
        monkeypatch.setattr(rknn_op_probe, "OUT_DIR", tmp_path / "audit")

        def fake_probe_one(onnx_path, probe_name):
            return {
                "probe": probe_name,
                "config_ret": 0,
                "load_ret": 0,
                "build_ret": 0,
                "export_ret": 0,
                "rknn_file_size": 1000,
                "op_table": [],
                "evidence": [],
                "log": f"{probe_name}.log",
                "verdict": "COMPILED",
                "cpu_fallback_ops": ["CumSum:1"],
                "npu_ops": ["Conv:0"],
            }

        with patch("rknn_op_probe.probe_one", side_effect=fake_probe_one):
            rknn_op_probe.main()

        captured = capsys.readouterr()
        assert "silent fallback" in captured.out
        assert "CumSum:1" in captured.out

    def test_all_npu_note(self, tmp_path, monkeypatch, capsys):
        """Probes compiled with no CPU fallbacks get 'all-NPU' note."""
        probe_dir = tmp_path / "probes"
        probe_dir.mkdir()
        manifest = [{"probe": "conv", "file": "conv.onnx", "ops": ["Conv"], "description": "test"}]
        (probe_dir / "manifest.json").write_text(json.dumps(manifest))
        (probe_dir / "conv.onnx").write_text("fake")

        monkeypatch.setattr(rknn_op_probe, "PROBE_DIR", probe_dir)
        monkeypatch.setattr(rknn_op_probe, "MANIFEST", probe_dir / "manifest.json")
        monkeypatch.setattr(rknn_op_probe, "OUT_DIR", tmp_path / "audit")

        def fake_probe_one(onnx_path, probe_name):
            return {
                "probe": probe_name,
                "config_ret": 0,
                "load_ret": 0,
                "build_ret": 0,
                "export_ret": 0,
                "rknn_file_size": 1000,
                "op_table": [],
                "evidence": [],
                "log": f"{probe_name}.log",
                "verdict": "COMPILED",
            }

        with patch("rknn_op_probe.probe_one", side_effect=fake_probe_one):
            rknn_op_probe.main()

        captured = capsys.readouterr()
        assert "all-NPU" in captured.out

    def test_results_include_onnx_ops_and_description(self, tmp_path, monkeypatch):
        """Each result entry has onnx_ops and description from manifest."""
        probe_dir = tmp_path / "probes"
        probe_dir.mkdir()
        manifest = [
            {
                "probe": "conv",
                "file": "conv.onnx",
                "ops": ["Conv", "Add"],
                "description": "conv + add",
            },
        ]
        (probe_dir / "manifest.json").write_text(json.dumps(manifest))
        (probe_dir / "conv.onnx").write_text("fake")

        monkeypatch.setattr(rknn_op_probe, "PROBE_DIR", probe_dir)
        monkeypatch.setattr(rknn_op_probe, "MANIFEST", probe_dir / "manifest.json")
        monkeypatch.setattr(rknn_op_probe, "OUT_DIR", tmp_path / "audit")

        def fake_probe_one(onnx_path, probe_name):
            return {
                "probe": probe_name,
                "verdict": "COMPILED",
                "load_ret": 0,
                "build_ret": 0,
                "export_ret": 0,
                "config_ret": 0,
                "rknn_file_size": None,
                "op_table": [],
                "evidence": [],
                "log": f"{probe_name}.log",
            }

        with patch("rknn_op_probe.probe_one", side_effect=fake_probe_one):
            rknn_op_probe.main()

        results = json.loads((tmp_path / "audit" / "rknn_audit_results.json").read_text())
        assert results[0]["onnx_ops"] == ["Conv", "Add"]
        assert results[0]["description"] == "conv + add"
