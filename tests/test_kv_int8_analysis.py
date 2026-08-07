"""Tests for bench/kv_int8_analysis.py — INT8 KV cache quantization report generator.

Covers CSV reading and markdown report generation including edge cases
(missing data, partial configs).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.kv_int8_analysis import generate_report, read_csv  # noqa: E402

# ---------------------------------------------------------------------------
# CSV columns matching the ctx-sweep format used by kv_int8_analysis
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "model",
    "ctx_len",
    "gdn_layer_us",
    "full_attn_us",
    "ffn_us",
    "total_us",
    "tok_per_sec",
    "kv_cache_mb",
]


def _write_kv_csv(path: Path, rows: list[dict]) -> Path:
    """Write a ctx-sweep CSV for kv_int8 config."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _make_config_rows(ctx_lens: list[int]) -> list[dict]:
    """Generate rows for one quantization config."""
    return [
        {
            "model": "4B",
            "ctx_len": str(ctx),
            "gdn_layer_us": "100",
            "full_attn_us": str(200 * ctx),
            "ffn_us": "50",
            "total_us": str(350 + 200 * (ctx - 1)),
            "tok_per_sec": str(10.0 / ctx),
            "kv_cache_mb": str(0.5 * ctx),
        }
        for ctx in ctx_lens
    ]


# ---------------------------------------------------------------------------
# read_csv
# ---------------------------------------------------------------------------


class TestReadCSV:
    def test_missing_file_returns_empty(self, tmp_path):
        result = read_csv(str(tmp_path / "nonexistent.csv"))
        assert result == []

    def test_valid_file_returns_rows(self, tmp_path):
        path = tmp_path / "data.csv"
        _write_kv_csv(path, _make_config_rows([1, 4096]))
        result = read_csv(str(path))
        assert len(result) == 2
        assert result[0]["ctx_len"] == "1"

    def test_empty_csv_returns_empty(self, tmp_path):
        path = tmp_path / "data.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
        result = read_csv(str(path))
        assert result == []


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def _run_report(self, tmp_path, monkeypatch):
        """Helper: ensure output dir exists and run report."""
        out_dir = tmp_path / "figures"
        out_dir.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        generate_report("rk3588-t3_big", str(out_dir))
        report_path = out_dir / "kv_int8_scaling.md"
        assert report_path.exists(), "Report file was not written"
        return report_path.read_text()

    def test_no_data_message(self, tmp_path, monkeypatch):
        """When no CSVs exist, report notes missing data per model."""
        report = self._run_report(tmp_path, monkeypatch)
        assert "No data found" in report

    def test_report_has_title(self, tmp_path, monkeypatch):
        report = self._run_report(tmp_path, monkeypatch)
        assert "# INT8 KV Cache Quantization" in report

    def test_report_has_motivation(self, tmp_path, monkeypatch):
        report = self._run_report(tmp_path, monkeypatch)
        assert "## Motivation" in report
        assert "O(n)" in report

    def test_report_has_key_takeaways(self, tmp_path, monkeypatch):
        report = self._run_report(tmp_path, monkeypatch)
        assert "## Key Takeaways" in report

    def test_report_with_data(self, tmp_path, monkeypatch):
        """With all 4 configs present, report shows throughput tables."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)

        device = "rk3588-t3_big"
        model_tag = "4b"
        ctx_lens = [1, 32, 512, 4096]
        for suffix in ["fp32w_fp32kv", "fp32w_int8kv", "int8w_fp32kv", "int8w_int8kv"]:
            _write_kv_csv(
                results_raw / f"{device}_ctx_sweep_{model_tag}_{suffix}.csv",
                _make_config_rows(ctx_lens),
            )

        report = self._run_report(tmp_path, monkeypatch)
        assert "### Throughput by configuration" in report
        assert "### Full-attention decode cost" in report
        assert "### KV cache memory footprint" in report
        assert "### Impact at ctx=4096" in report

    def test_report_partial_configs(self, tmp_path, monkeypatch):
        """Only FP32 baseline present — report still generates gracefully."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)

        device = "rk3588-t3_big"
        _write_kv_csv(
            results_raw / f"{device}_ctx_sweep_4b_fp32w_fp32kv.csv",
            _make_config_rows([1, 4096]),
        )

        report = self._run_report(tmp_path, monkeypatch)
        assert "### Throughput by configuration" in report
        # Impact table requires both fp32w_fp32kv and int8w_int8kv
        assert "Impact at" not in report

    def test_gdn_flat_section(self, tmp_path, monkeypatch):
        """GDN layer cost section appears when data is present."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)

        device = "rk3588-t3_big"
        _write_kv_csv(
            results_raw / f"{device}_ctx_sweep_4b_fp32w_fp32kv.csv",
            _make_config_rows([1, 4096]),
        )

        report = self._run_report(tmp_path, monkeypatch)
        assert "GDN layer cost" in report
        assert "O(1)" in report

    def test_speedup_shown_with_full_data(self, tmp_path, monkeypatch):
        """Impact table shows speedup ratios when both configs present."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)

        device = "rk3588-t3_big"
        ctx_lens = [1, 4096]

        # FP32 baseline
        _write_kv_csv(
            results_raw / f"{device}_ctx_sweep_4b_fp32w_fp32kv.csv",
            [
                {
                    "model": "4B",
                    "ctx_len": str(ctx),
                    "gdn_layer_us": "100",
                    "full_attn_us": str(200 * ctx),
                    "ffn_us": "50",
                    "total_us": str(350 + 200 * (ctx - 1)),
                    "tok_per_sec": str(10.0 / ctx),
                    "kv_cache_mb": str(4.0 * ctx),
                }
                for ctx in ctx_lens
            ],
        )
        # INT8 both — 2x faster decode, 4x smaller KV
        _write_kv_csv(
            results_raw / f"{device}_ctx_sweep_4b_int8w_int8kv.csv",
            [
                {
                    "model": "4B",
                    "ctx_len": str(ctx),
                    "gdn_layer_us": "100",
                    "full_attn_us": str(100 * ctx),
                    "ffn_us": "50",
                    "total_us": str(250 + 100 * (ctx - 1)),
                    "tok_per_sec": str(20.0 / ctx),
                    "kv_cache_mb": str(1.0 * ctx),
                }
                for ctx in ctx_lens
            ],
        )
        # FP32 weights + INT8 KV
        _write_kv_csv(
            results_raw / f"{device}_ctx_sweep_4b_fp32w_int8kv.csv",
            [
                {
                    "model": "4B",
                    "ctx_len": str(ctx),
                    "gdn_layer_us": "100",
                    "full_attn_us": str(150 * ctx),
                    "ffn_us": "50",
                    "total_us": str(300 + 150 * (ctx - 1)),
                    "tok_per_sec": str(15.0 / ctx),
                    "kv_cache_mb": str(1.0 * ctx),
                }
                for ctx in ctx_lens
            ],
        )

        report = self._run_report(tmp_path, monkeypatch)
        assert "Improvement" in report
        assert "INT8 KV only" in report
