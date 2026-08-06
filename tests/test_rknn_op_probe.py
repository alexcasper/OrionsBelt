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
