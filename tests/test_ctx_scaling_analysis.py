"""Tests for bench/ctx_scaling_analysis.py — context-length scaling report generator.

Covers CSV reading, formatting, and markdown report generation including
edge cases (missing data, partial data).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.ctx_scaling_analysis import (  # noqa: E402
    CONFIGS_RK3588,
    collect_data,
    fmt_tok,
    generate_cross_device,
    generate_report,
    read_csv,
)

# ---------------------------------------------------------------------------
# CSV columns matching the ctx-sweep e2e raw format
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


def _write_ctx_csv(path: Path, rows: list[dict]) -> Path:
    """Write a ctx-sweep CSV with the expected columns."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# read_csv
# ---------------------------------------------------------------------------

class TestReadCSV:

    def test_missing_file_returns_none(self, tmp_path):
        result = read_csv(str(tmp_path / "nonexistent.csv"))
        assert result is None

    def test_valid_file_returns_rows(self, tmp_path):
        path = tmp_path / "data.csv"
        _write_ctx_csv(path, [
            {"model": "4B", "ctx_len": "1", "gdn_layer_us": "100",
             "full_attn_us": "200", "ffn_us": "50", "total_us": "350",
             "tok_per_sec": "10.0", "kv_cache_mb": "0.5"},
        ])
        result = read_csv(str(path))
        assert result is not None
        assert len(result) == 1
        assert result[0]["ctx_len"] == "1"

    def test_multiple_rows(self, tmp_path):
        path = tmp_path / "data.csv"
        _write_ctx_csv(path, [
            {"model": "4B", "ctx_len": str(ctx), "gdn_layer_us": "100",
             "full_attn_us": str(200 * ctx), "ffn_us": "50",
             "total_us": str(350 + 200 * (ctx - 1)),
             "tok_per_sec": str(10.0 / ctx), "kv_cache_mb": str(0.5 * ctx)}
            for ctx in [1, 32, 512, 4096]
        ])
        result = read_csv(str(path))
        assert result is not None
        assert len(result) == 4


# ---------------------------------------------------------------------------
# fmt_tok
# ---------------------------------------------------------------------------

class TestFmtTok:

    def test_integer(self):
        assert fmt_tok("10") == "10.00"

    def test_float(self):
        assert fmt_tok("3.14159") == "3.14"

    def test_zero(self):
        assert fmt_tok("0") == "0.00"

    def test_large(self):
        assert fmt_tok("1234.567") == "1234.57"


# ---------------------------------------------------------------------------
# collect_data
# ---------------------------------------------------------------------------

class TestCollectData:

    def test_no_csvs_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = collect_data("nonexistent-device", CONFIGS_RK3588)
        assert result == {}

    def test_finds_matching_csvs(self, tmp_path, monkeypatch):
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        _write_ctx_csv(
            results_raw / "rk3588-t3_big_ctxsweep_e2e_raw.csv",
            [{"model": "4B", "ctx_len": "1", "gdn_layer_us": "100",
              "full_attn_us": "200", "ffn_us": "50", "total_us": "350",
              "tok_per_sec": "10.0", "kv_cache_mb": "0.5"}],
        )
        monkeypatch.chdir(tmp_path)
        result = collect_data("rk3588-t3", CONFIGS_RK3588)
        assert "4B FP32 hybrid" in result


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:

    def test_no_data_message(self, tmp_path):
        """When no CSVs exist, report says so explicitly."""
        report = generate_report(
            "nonexistent-device", "a76", CONFIGS_RK3588, str(tmp_path)
        )
        assert "No ctx-sweep CSVs found" in report

    def test_report_has_title(self, tmp_path):
        report = generate_report(
            "rk3588-t3", "a76", CONFIGS_RK3588, str(tmp_path)
        )
        assert "# Context-Length Scaling" in report

    def test_report_has_generated_by(self, tmp_path):
        report = generate_report(
            "rk3588-t3", "a76", CONFIGS_RK3588, str(tmp_path)
        )
        assert "ctx_scaling_analysis.py" in report
        assert "Do not hand-edit" in report

    def test_report_mentions_core(self, tmp_path):
        report = generate_report(
            "rk3588-t3", "a76", CONFIGS_RK3588, str(tmp_path)
        )
        assert "Cortex-A76" in report

    def test_report_with_data(self, tmp_path, monkeypatch):
        """When CSVs exist, report includes throughput tables."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        _write_ctx_csv(
            results_raw / "rk3588-t3_big_ctxsweep_e2e_raw.csv",
            [
                {"model": "4B", "ctx_len": str(ctx), "gdn_layer_us": "100",
                 "full_attn_us": str(200 * ctx), "ffn_us": "50",
                 "total_us": str(350 + 200 * (ctx - 1)),
                 "tok_per_sec": str(10.0 / ctx), "kv_cache_mb": str(0.5 * ctx)}
                for ctx in [1, 32, 512, 4096]
            ]
        )
        monkeypatch.chdir(tmp_path)
        report = generate_report(
            "rk3588-t3", "a76", CONFIGS_RK3588, str(tmp_path / "figures")
        )
        assert "Throughput vs context length" in report
        assert "Full-attention share" in report
        assert "KV cache memory" in report
        assert "Key findings" in report

    def test_throughput_table_contains_values(self, tmp_path, monkeypatch):
        """Throughput table should include the CSV values."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        _write_ctx_csv(
            results_raw / "rk3588-t3_big_ctxsweep_e2e_raw.csv",
            [
                {"model": "4B", "ctx_len": "1", "gdn_layer_us": "100",
                 "full_attn_us": "200", "ffn_us": "50", "total_us": "350",
                 "tok_per_sec": "10.00", "kv_cache_mb": "0.5"},
                {"model": "4B", "ctx_len": "4096", "gdn_layer_us": "101",
                 "full_attn_us": "20000", "ffn_us": "50", "total_us": "20151",
                 "tok_per_sec": "1.50", "kv_cache_mb": "256.0"},
            ]
        )
        monkeypatch.chdir(tmp_path)
        report = generate_report(
            "rk3588-t3", "a76", CONFIGS_RK3588, str(tmp_path / "figures")
        )
        assert "10.00" in report
        assert "1.50" in report

    def test_kv_cache_table_values(self, tmp_path, monkeypatch):
        """KV cache memory table should contain MB values."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        _write_ctx_csv(
            results_raw / "rk3588-t3_big_ctxsweep_e2e_raw.csv",
            [
                {"model": "4B", "ctx_len": "4096", "gdn_layer_us": "100",
                 "full_attn_us": "20000", "ffn_us": "50", "total_us": "20150",
                 "tok_per_sec": "1.50", "kv_cache_mb": "256.0"},
            ]
        )
        monkeypatch.chdir(tmp_path)
        report = generate_report(
            "rk3588-t3", "a76", CONFIGS_RK3588, str(tmp_path / "figures")
        )
        assert "256 MB" in report

    def test_headline_section(self, tmp_path, monkeypatch):
        """Bar chart section should include max ctx in heading."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        _write_ctx_csv(
            results_raw / "rk3588-t3_big_ctxsweep_e2e_raw.csv",
            [
                {"model": "4B", "ctx_len": "1", "gdn_layer_us": "100",
                 "full_attn_us": "200", "ffn_us": "50", "total_us": "350",
                 "tok_per_sec": "10.0", "kv_cache_mb": "0.5"},
                {"model": "4B", "ctx_len": "4096", "gdn_layer_us": "101",
                 "full_attn_us": "20000", "ffn_us": "50", "total_us": "20151",
                 "tok_per_sec": "1.5", "kv_cache_mb": "256.0"},
            ]
        )
        monkeypatch.chdir(tmp_path)
        report = generate_report(
            "rk3588-t3", "a76", CONFIGS_RK3588, str(tmp_path / "figures")
        )
        assert "ctx=4096" in report


# ---------------------------------------------------------------------------
# generate_cross_device
# ---------------------------------------------------------------------------

class TestGenerateCrossDevice:

    def test_no_data_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = generate_cross_device(str(tmp_path))
        assert result is None

    def test_single_device_returns_none(self, tmp_path, monkeypatch):
        """Need 2+ devices for cross-device comparison."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        _write_ctx_csv(
            results_raw / "rk3588-t3_big_ctxsweep_e2e_raw.csv",
            [{"model": "4B", "ctx_len": "1", "gdn_layer_us": "100",
              "full_attn_us": "200", "ffn_us": "50", "total_us": "350",
              "tok_per_sec": "10.0", "kv_cache_mb": "0.5"}],
        )
        monkeypatch.chdir(tmp_path)
        result = generate_cross_device(str(tmp_path))
        assert result is None

    def test_two_devices_produces_table(self, tmp_path, monkeypatch):
        """Cross-device table requires CSVs matching each device's config patterns."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        # rk3588-t3 uses {d}_big_ctxsweep pattern
        _write_ctx_csv(
            results_raw / "rk3588-t3_big_ctxsweep_e2e_raw.csv",
            [
                {"model": "4B", "ctx_len": "1", "gdn_layer_us": "100",
                 "full_attn_us": "200", "ffn_us": "50", "total_us": "350",
                 "tok_per_sec": "10.0", "kv_cache_mb": "0.5"},
                {"model": "4B", "ctx_len": "4096", "gdn_layer_us": "101",
                 "full_attn_us": "20000", "ffn_us": "50", "total_us": "20151",
                 "tok_per_sec": "1.5", "kv_cache_mb": "256.0"},
            ]
        )
        # jetson-j1 uses {d}_4b_fp32_ctxsweep pattern
        _write_ctx_csv(
            results_raw / "jetson-j1_4b_fp32_ctxsweep_e2e_raw.csv",
            [
                {"model": "4B", "ctx_len": "1", "gdn_layer_us": "200",
                 "full_attn_us": "400", "ffn_us": "100", "total_us": "700",
                 "tok_per_sec": "3.5", "kv_cache_mb": "0.5"},
                {"model": "4B", "ctx_len": "4096", "gdn_layer_us": "201",
                 "full_attn_us": "40000", "ffn_us": "100", "total_us": "40301",
                 "tok_per_sec": "0.5", "kv_cache_mb": "256.0"},
            ]
        )
        monkeypatch.chdir(tmp_path)
        result = generate_cross_device(str(tmp_path))
        assert result is not None
        assert "A76" in result
        assert "A57" in result
        assert "slowdown" in result.lower()
