"""Tests for bench/comparison_table.py — comparison-table generation from CSVs.

Covers the full pipeline: CSV loading → group-by summarization → markdown pivot.
Uses synthetic CSV data in the frozen tidy/long schema (RESULTS_SCHEMA.md §1–3).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.comparison_table import (  # noqa: E402
    _fmt_value,
    generate_comparison,
    generate_markdown_table,
    load_and_summarize,
)

# ---------------------------------------------------------------------------
# Helpers: write synthetic CSV rows in the frozen tidy/long schema.
# ---------------------------------------------------------------------------

COLUMNS = [
    "run_id", "timestamp", "git_sha", "manifest_ref",
    "device", "engine_gdn", "engine_full_attention",
    "model_checkpoint", "quantization",
    "context_length", "phase", "metric_name", "metric_component",
    "value", "unit", "repeat_index", "repeat_count",
]


def _row(
    *,
    run_id="test_run",
    engine_gdn="cpu",
    engine_full_attention="cpu",
    quant="fp16",
    ctx=4096,
    phase="prefill",
    metric="prefill_tokens_per_sec",
    component="",
    value=100.0,
    repeat=0,
    repeat_count=5,
):
    """Build a single tidy/long CSV row."""
    return {
        "run_id": run_id,
        "timestamp": "2026-08-06T00:00:00Z",
        "git_sha": "abcdef0",
        "manifest_ref": "results/manifests/test.json",
        "device": "generic_aarch64",
        "engine_gdn": engine_gdn,
        "engine_full_attention": engine_full_attention,
        "model_checkpoint": "Qwen/Qwen3.5-4B@abcdef0",
        "quantization": quant,
        "context_length": str(ctx),
        "phase": phase,
        "metric_name": metric,
        "metric_component": component,
        "value": str(value),
        "unit": "tokens_per_sec",
        "repeat_index": str(repeat),
        "repeat_count": str(repeat_count),
    }


def _write_csv(path, rows):
    """Write rows to a CSV with the full schema columns."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ---------------------------------------------------------------------------
# load_and_summarize
# ---------------------------------------------------------------------------

class TestLoadAndSummarize:
    def test_single_group_p50(self, tmp_path):
        """Five repeats of one metric → one summary with p50 from 5 values."""
        rows = [_row(value=v, repeat=i) for i, v in enumerate([10, 20, 30, 40, 50])]
        csv_path = tmp_path / "run.csv"
        _write_csv(csv_path, rows)

        summaries = load_and_summarize([str(csv_path)])
        assert len(summaries) == 1
        s = summaries[0]
        # Nearest-rank p50 of [10,20,30,40,50] (n=5, rank=ceil(0.5*5)=3) → 30
        assert s["p50"] == 30.0
        assert s["n"] == 5

    def test_groups_by_engine_config(self, tmp_path):
        """Different engine_gdn/engine_full_attention produce separate summaries."""
        rows = [
            _row(engine_gdn="cpu", engine_full_attention="cpu", value=100, repeat=0),
            _row(engine_gdn="cpu", engine_full_attention="cpu", value=110, repeat=1),
            _row(engine_gdn="npu", engine_full_attention="cpu", value=200, repeat=0),
            _row(engine_gdn="npu", engine_full_attention="cpu", value=210, repeat=1),
        ]
        csv_path = tmp_path / "run.csv"
        _write_csv(csv_path, rows)

        summaries = load_and_summarize([str(csv_path)])
        assert len(summaries) == 2

        # p50 nearest-rank of [100, 110] (n=2, rank=ceil(0.5*2)=1) → 100
        assert cpu_summary["p50"] == 100.0
        npu_summary = [s for s in summaries if s["engine_gdn"] == "npu"][0]
        # p50 nearest-rank of [200, 210] → 200
        assert npu_summary["p50"] == 200.0

    def test_groups_by_context_length(self, tmp_path):
        """Different context lengths are separate groups."""
        rows = [
            _row(ctx=4096, value=100, repeat=0),
            _row(ctx=4096, value=110, repeat=1),
            _row(ctx=32768, value=50, repeat=0),
            _row(ctx=32768, value=55, repeat=1),
        ]
        csv_path = tmp_path / "run.csv"
        _write_csv(csv_path, rows)

        summaries = load_and_summarize([str(csv_path)])
        ctxs = {s["context_length"] for s in summaries}
        assert ctxs == {4096, 32768}

    def test_groups_by_metric_component(self, tmp_path):
        """Memory metrics with different components are separate groups."""
        rows = [
            _row(metric="peak_memory_bytes", component="weights", value=8e9, repeat=0),
            _row(metric="peak_memory_bytes", component="kv_cache", value=1e9, repeat=0),
            _row(metric="peak_memory_bytes", component="recurrent_state", value=5e8, repeat=0),
        ]
        csv_path = tmp_path / "run.csv"
        _write_csv(csv_path, rows)

        summaries = load_and_summarize([str(csv_path)])
        assert len(summaries) == 3
        comps = {s["metric_component"] for s in summaries}
        assert comps == {"weights", "kv_cache", "recurrent_state"}

    def test_multiple_csvs_combined(self, tmp_path):
        """Multiple CSV files are combined into one group set."""
        rows1 = [_row(run_id="r1", value=100, repeat=0)]
        rows2 = [_row(run_id="r2", value=200, repeat=1)]
        p1 = tmp_path / "run1.csv"
        p2 = tmp_path / "run2.csv"
        _write_csv(p1, rows1)
        _write_csv(p2, rows2)

        summaries = load_and_summarize([str(p1), str(p2)])
        assert len(summaries) == 1
        assert summaries[0]["n"] == 2
        # p50 of [100, 200] with nearest-rank (rank=ceil(0.5*2)=1) → 100
        assert summaries[0]["p50"] == 100.0

    def test_empty_csv(self, tmp_path):
        """A CSV with only headers produces no summaries."""
        csv_path = tmp_path / "empty.csv"
        _write_csv(csv_path, [])

        summaries = load_and_summarize([str(csv_path)])
        assert summaries == []

    def test_p95_computed(self, tmp_path):
        """p95 is computed alongside p50."""
        values = list(range(10, 60, 10))  # [10,20,30,40,50]
        rows = [_row(value=v, repeat=i) for i, v in enumerate(values)]
        csv_path = tmp_path / "run.csv"
        _write_csv(csv_path, rows)

        summaries = load_and_summarize([str(csv_path)])
        # p95 nearest-rank of [10,20,30,40,50] (n=5, rank=ceil(0.95*5)=5) → 50
        assert summaries[0]["p95"] == 50.0

    def test_sorted_output(self, tmp_path):
        """Summaries are sorted by group key for deterministic output."""
        rows = [
            _row(engine_gdn="npu", value=200, repeat=0),
            _row(engine_gdn="cpu", value=100, repeat=0),
        ]
        csv_path = tmp_path / "run.csv"
        _write_csv(csv_path, rows)

        summaries = load_and_summarize([str(csv_path)])
        # Sorted by group key (engine_gdn first): "cpu" < "npu"
        assert summaries[0]["engine_gdn"] == "cpu"
        assert summaries[1]["engine_gdn"] == "npu"


# ---------------------------------------------------------------------------
# _fmt_value
# ---------------------------------------------------------------------------

class TestFmtValue:
    def test_tokens_per_sec(self):
        assert _fmt_value("prefill_tokens_per_sec", 123.456) == "123.5"

    def test_decode_tokens_per_sec(self):
        assert _fmt_value("decode_tokens_per_sec", 7.5) == "7.5"

    def test_ttft_seconds(self):
        result = _fmt_value("ttft_seconds", 0.025)
        assert "ms" in result
        assert "25.0" in result

    def test_peak_memory_bytes_gib(self):
        result = _fmt_value("peak_memory_bytes", 8e9)
        assert "GiB" in result

    def test_peak_memory_bytes_mib(self):
        result = _fmt_value("peak_memory_bytes", 100 * 1024 * 1024)
        assert "MiB" in result

    def test_generic_metric(self):
        result = _fmt_value("energy_joules", 3.14159)
        assert "3.142" in result


# ---------------------------------------------------------------------------
# generate_markdown_table
# ---------------------------------------------------------------------------

class TestGenerateMarkdownTable:
    def test_basic_table_structure(self):
        """A markdown table has a header row, separator, and at least one data row."""
        summaries = [
            {
                "engine_gdn": "cpu",
                "engine_full_attention": "cpu",
                "quantization": "fp16",
                "context_length": 4096,
                "phase": "prefill",
                "metric_name": "prefill_tokens_per_sec",
                "metric_component": "",
                "p50": 100.0,
                "p95": 120.0,
                "n": 5,
            }
        ]
        table = generate_markdown_table(summaries)
        lines = table.strip().split("\n")
        assert len(lines) >= 3  # header + separator + 1 row
        assert "|" in lines[0]
        assert "---" in lines[1]
        assert "cpu" in lines[2]

    def test_multiple_columns_pivot(self):
        """Two metrics become two columns in the table."""
        summaries = [
            {
                "engine_gdn": "cpu", "engine_full_attention": "cpu",
                "quantization": "fp16", "context_length": 4096,
                "phase": "prefill", "metric_name": "prefill_tokens_per_sec",
                "metric_component": "", "p50": 100.0, "p95": 120.0, "n": 5,
            },
            {
                "engine_gdn": "cpu", "engine_full_attention": "cpu",
                "quantization": "fp16", "context_length": 4096,
                "phase": "decode", "metric_name": "decode_tokens_per_sec",
                "metric_component": "", "p50": 7.5, "p95": 9.0, "n": 5,
            },
        ]
        table = generate_markdown_table(summaries)
        # Header should contain both metric columns
        assert "prefill/prefill_tokens_per_sec" in table
        assert "decode/decode_tokens_per_sec" in table

    def test_missing_cell_shows_dash(self):
        """A config with no data for a column shows an em-dash."""
        summaries = [
            {
                "engine_gdn": "cpu", "engine_full_attention": "cpu",
                "quantization": "fp16", "context_length": 4096,
                "phase": "prefill", "metric_name": "prefill_tokens_per_sec",
                "metric_component": "", "p50": 100.0, "p95": 120.0, "n": 5,
            },
            {
                "engine_gdn": "npu", "engine_full_attention": "cpu",
                "quantization": "fp16", "context_length": 4096,
                "phase": "decode", "metric_name": "decode_tokens_per_sec",
                "metric_component": "", "p50": 7.5, "p95": 9.0, "n": 5,
            },
        ]
        table = generate_markdown_table(summaries)
        data_lines = [line for line in lines[2:] if "|" in line]
        assert any("—" in row for row in data_lines)

    def test_context_length_label(self):
        """Context lengths >= 1024 are shown as K notation."""
        summaries = [
            {
                "engine_gdn": "cpu", "engine_full_attention": "cpu",
                "quantization": "fp16", "context_length": 4096,
                "phase": "prefill", "metric_name": "prefill_tokens_per_sec",
                "metric_component": "", "p50": 100.0, "p95": 120.0, "n": 5,
            },
        ]
        table = generate_markdown_table(summaries)
        assert "4K" in table

    def test_context_length_small_shows_raw(self):
        """Small context lengths (< 1024) show the raw integer."""
        summaries = [
            {
                "engine_gdn": "cpu", "engine_full_attention": "cpu",
                "quantization": "fp16", "context_length": 64,
                "phase": "prefill", "metric_name": "prefill_tokens_per_sec",
                "metric_component": "", "p50": 100.0, "p95": 120.0, "n": 5,
            },
        ]
        table = generate_markdown_table(summaries)
        # "64" should appear, not "0K"
        assert "| 64 |" in table

    def test_component_in_column_label(self):
        """metric_component appears in brackets in the column header."""
        summaries = [
            {
                "engine_gdn": "cpu", "engine_full_attention": "cpu",
                "quantization": "fp16", "context_length": 4096,
                "phase": "prefill", "metric_name": "peak_memory_bytes",
                "metric_component": "weights", "p50": 8e9, "p95": 8.1e9, "n": 5,
            },
        ]
        table = generate_markdown_table(summaries)
        assert "[weights]" in table


# ---------------------------------------------------------------------------
# generate_comparison (end-to-end)
# ---------------------------------------------------------------------------

class TestGenerateComparison:
    def test_end_to_end_from_csv(self, tmp_path):
        """Full pipeline: CSV → summarize → markdown table."""
        rows = [
            _row(value=100 + i * 10, repeat=i) for i in range(5)
        ]
        csv_path = tmp_path / "run.csv"
        _write_csv(csv_path, rows)

        table = generate_comparison([str(csv_path)])
        assert "prefill_tokens_per_sec" in table
        assert "cpu" in table

    def test_empty_csvs_message(self, tmp_path):
        """Empty CSVs produce a 'no data' message."""
        csv_path = tmp_path / "empty.csv"
        _write_csv(csv_path, [])

        result = generate_comparison([str(csv_path)])
        assert "No data" in result

    def test_multi_config_ablation(self, tmp_path):
        """Simulates the ablation grid: two engine configs across two contexts."""
        rows = []
        for eng_gdn, eng_fa, base_val in [("cpu", "cpu", 100), ("npu", "cpu", 200)]:
            for ctx in [4096, 32768]:
                for i in range(3):
                    rows.append(_row(
                        engine_gdn=eng_gdn,
                        engine_full_attention=eng_fa,
                        ctx=ctx,
                        value=base_val + i * 10,
                        repeat=i,
                    ))
        csv_path = tmp_path / "ablation.csv"
        _write_csv(csv_path, rows)

        table = generate_comparison([str(csv_path)])
        # Should have 4 rows (2 configs × 2 contexts)
        data_lines = [line for line in table.strip().split("\n")[2:] if "|" in line]
        assert len(data_lines) == 4


# ---------------------------------------------------------------------------
# Integration with bench.metrics.percentile
# ---------------------------------------------------------------------------

class TestPercentileIntegration:
    def test_p50_matches_metrics_module(self, tmp_path):
        """load_and_summarize p50 matches bench.metrics.percentile directly."""
        from bench.metrics import percentile

        values = [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0]
        rows = [_row(value=v, repeat=i) for i, v in enumerate(values)]
        csv_path = tmp_path / "run.csv"
        _write_csv(csv_path, rows)

        summaries = load_and_summarize([str(csv_path)])
        expected_p50 = percentile(values, 50)
        expected_p95 = percentile(values, 95)
        assert summaries[0]["p50"] == expected_p50
        assert summaries[0]["p95"] == expected_p95
