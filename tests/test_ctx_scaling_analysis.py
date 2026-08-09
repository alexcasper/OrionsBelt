# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

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
    CONFIGS_RK3588_KV_SWEEP,
    _ctx_lookup,
    collect_data,
    fmt_tok,
    generate_cross_device,
    generate_cross_validation,
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
        _write_ctx_csv(
            path,
            [
                {
                    "model": "4B",
                    "ctx_len": "1",
                    "gdn_layer_us": "100",
                    "full_attn_us": "200",
                    "ffn_us": "50",
                    "total_us": "350",
                    "tok_per_sec": "10.0",
                    "kv_cache_mb": "0.5",
                },
            ],
        )
        result = read_csv(str(path))
        assert result is not None
        assert len(result) == 1
        assert result[0]["ctx_len"] == "1"

    def test_multiple_rows(self, tmp_path):
        path = tmp_path / "data.csv"
        _write_ctx_csv(
            path,
            [
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
                for ctx in [1, 32, 512, 4096]
            ],
        )
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
            [
                {
                    "model": "4B",
                    "ctx_len": "1",
                    "gdn_layer_us": "100",
                    "full_attn_us": "200",
                    "ffn_us": "50",
                    "total_us": "350",
                    "tok_per_sec": "10.0",
                    "kv_cache_mb": "0.5",
                }
            ],
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
        report = generate_report("nonexistent-device", "a76", "A76", CONFIGS_RK3588, str(tmp_path))
        assert "No ctx-sweep CSVs found" in report

    def test_report_has_title(self, tmp_path):
        report = generate_report("rk3588-t3", "a76", "A76", CONFIGS_RK3588, str(tmp_path))
        assert "# Context-Length Scaling" in report

    def test_report_has_generated_by(self, tmp_path):
        report = generate_report("rk3588-t3", "a76", "A76", CONFIGS_RK3588, str(tmp_path))
        assert "ctx_scaling_analysis.py" in report
        assert "Do not hand-edit" in report

    def test_report_mentions_core(self, tmp_path):
        report = generate_report("rk3588-t3", "a76", "A76", CONFIGS_RK3588, str(tmp_path))
        assert "Cortex-A76" in report

    def test_report_with_data(self, tmp_path, monkeypatch):
        """When CSVs exist, report includes throughput tables."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        _write_ctx_csv(
            results_raw / "rk3588-t3_big_ctxsweep_e2e_raw.csv",
            [
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
                for ctx in [1, 32, 512, 4096]
            ],
        )
        monkeypatch.chdir(tmp_path)
        report = generate_report(
            "rk3588-t3", "a76", "A76", CONFIGS_RK3588, str(tmp_path / "figures")
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
                {
                    "model": "4B",
                    "ctx_len": "1",
                    "gdn_layer_us": "100",
                    "full_attn_us": "200",
                    "ffn_us": "50",
                    "total_us": "350",
                    "tok_per_sec": "10.00",
                    "kv_cache_mb": "0.5",
                },
                {
                    "model": "4B",
                    "ctx_len": "4096",
                    "gdn_layer_us": "101",
                    "full_attn_us": "20000",
                    "ffn_us": "50",
                    "total_us": "20151",
                    "tok_per_sec": "1.50",
                    "kv_cache_mb": "256.0",
                },
            ],
        )
        monkeypatch.chdir(tmp_path)
        report = generate_report(
            "rk3588-t3", "a76", "A76", CONFIGS_RK3588, str(tmp_path / "figures")
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
                {
                    "model": "4B",
                    "ctx_len": "4096",
                    "gdn_layer_us": "100",
                    "full_attn_us": "20000",
                    "ffn_us": "50",
                    "total_us": "20150",
                    "tok_per_sec": "1.50",
                    "kv_cache_mb": "256.0",
                },
            ],
        )
        monkeypatch.chdir(tmp_path)
        report = generate_report(
            "rk3588-t3", "a76", "A76", CONFIGS_RK3588, str(tmp_path / "figures")
        )
        assert "256 MB" in report

    def test_headline_section(self, tmp_path, monkeypatch):
        """Bar chart section should include max ctx in heading."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        _write_ctx_csv(
            results_raw / "rk3588-t3_big_ctxsweep_e2e_raw.csv",
            [
                {
                    "model": "4B",
                    "ctx_len": "1",
                    "gdn_layer_us": "100",
                    "full_attn_us": "200",
                    "ffn_us": "50",
                    "total_us": "350",
                    "tok_per_sec": "10.0",
                    "kv_cache_mb": "0.5",
                },
                {
                    "model": "4B",
                    "ctx_len": "4096",
                    "gdn_layer_us": "101",
                    "full_attn_us": "20000",
                    "ffn_us": "50",
                    "total_us": "20151",
                    "tok_per_sec": "1.5",
                    "kv_cache_mb": "256.0",
                },
            ],
        )
        monkeypatch.chdir(tmp_path)
        report = generate_report(
            "rk3588-t3", "a76", "A76", CONFIGS_RK3588, str(tmp_path / "figures")
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
            [
                {
                    "model": "4B",
                    "ctx_len": "1",
                    "gdn_layer_us": "100",
                    "full_attn_us": "200",
                    "ffn_us": "50",
                    "total_us": "350",
                    "tok_per_sec": "10.0",
                    "kv_cache_mb": "0.5",
                }
            ],
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
                {
                    "model": "4B",
                    "ctx_len": "1",
                    "gdn_layer_us": "100",
                    "full_attn_us": "200",
                    "ffn_us": "50",
                    "total_us": "350",
                    "tok_per_sec": "10.0",
                    "kv_cache_mb": "0.5",
                },
                {
                    "model": "4B",
                    "ctx_len": "4096",
                    "gdn_layer_us": "101",
                    "full_attn_us": "20000",
                    "ffn_us": "50",
                    "total_us": "20151",
                    "tok_per_sec": "1.5",
                    "kv_cache_mb": "256.0",
                },
            ],
        )
        # jetson-j1 uses {d}_4b_fp32_ctxsweep pattern
        _write_ctx_csv(
            results_raw / "jetson-j1_4b_fp32_ctxsweep_e2e_raw.csv",
            [
                {
                    "model": "4B",
                    "ctx_len": "1",
                    "gdn_layer_us": "200",
                    "full_attn_us": "400",
                    "ffn_us": "100",
                    "total_us": "700",
                    "tok_per_sec": "3.5",
                    "kv_cache_mb": "0.5",
                },
                {
                    "model": "4B",
                    "ctx_len": "4096",
                    "gdn_layer_us": "201",
                    "full_attn_us": "40000",
                    "ffn_us": "100",
                    "total_us": "40301",
                    "tok_per_sec": "0.5",
                    "kv_cache_mb": "256.0",
                },
            ],
        )
        monkeypatch.chdir(tmp_path)
        result = generate_cross_device(str(tmp_path))
        assert result is not None
        assert "A76" in result
        assert "A57" in result
        assert "slowdown" in result.lower()


# ---------------------------------------------------------------------------
# _ctx_lookup
# ---------------------------------------------------------------------------


class TestCtxLookup:
    def test_found(self):
        rows = [
            {"ctx_len": "1", "tok_per_sec": "10.0"},
            {"ctx_len": "64", "tok_per_sec": "9.5"},
        ]
        assert _ctx_lookup(rows, 64, "tok_per_sec") == "9.5"

    def test_not_found(self):
        rows = [{"ctx_len": "1", "tok_per_sec": "10.0"}]
        assert _ctx_lookup(rows, 4096, "tok_per_sec") is None

    def test_different_field(self):
        rows = [{"ctx_len": "1", "total_us": "350"}]
        assert _ctx_lookup(rows, 1, "total_us") == "350"


# ---------------------------------------------------------------------------
# collect_data — KV sweep (new naming) convention
# ---------------------------------------------------------------------------


class TestCollectDataKVSweep:
    def test_finds_new_naming_csvs(self, tmp_path, monkeypatch):
        """collect_data should find _ctx_sweep_ files without _e2e_raw suffix."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        _write_ctx_csv(
            results_raw / "rk3588-t4_big_ctx_sweep_4b_int8w_fp32kv.csv",
            [
                {
                    "model": "4B",
                    "ctx_len": "1",
                    "gdn_layer_us": "100",
                    "full_attn_us": "200",
                    "ffn_us": "50",
                    "total_us": "350",
                    "tok_per_sec": "1.83",
                    "kv_cache_mb": "0.1",
                }
            ],
        )
        monkeypatch.chdir(tmp_path)
        result = collect_data("rk3588-t4", CONFIGS_RK3588_KV_SWEEP)
        assert "4B INT8w FP32kv" in result


# ---------------------------------------------------------------------------
# generate_report — KV sweep configs
# ---------------------------------------------------------------------------


class TestGenerateReportKVSweep:
    def test_report_includes_kv_cache_section(self, tmp_path, monkeypatch):
        """KV sweep report should include INT8 KV-cache benefit section."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        for kv in ("fp32kv", "int8kv"):
            _write_ctx_csv(
                results_raw / f"rk3588-t4_big_ctx_sweep_4b_int8w_{kv}.csv",
                [
                    {
                        "model": "4B",
                        "ctx_len": "1",
                        "gdn_layer_us": "119000",
                        "full_attn_us": "38000",
                        "ffn_us": "389000",
                        "total_us": "546000",
                        "tok_per_sec": "1.83",
                        "kv_cache_mb": "0.1",
                    },
                    {
                        "model": "4B",
                        "ctx_len": "4096",
                        "gdn_layer_us": "120000",
                        "full_attn_us": "189000",
                        "ffn_us": "390000",
                        "total_us": "699000",
                        "tok_per_sec": "1.43" if kv == "int8kv" else "1.20",
                        "kv_cache_mb": "64.0",
                    },
                ],
            )
        monkeypatch.chdir(tmp_path)
        report = generate_report(
            "rk3588-t4", "a76t4", "A76", CONFIGS_RK3588_KV_SWEEP, str(tmp_path / "figures")
        )
        assert "INT8 KV-cache quantization benefit" in report
        assert "4B INT8w" in report
        # Verify the speedup is shown
        assert "1.19" in report or "1.20" in report  # int8kv/fp32kv ratio

    def test_report_mentions_cortex_a76(self, tmp_path):
        """KV sweep device should show 'Cortex-A76', not 'Cortex-A76T4'."""
        report = generate_report(
            "rk3588-t4", "a76t4", "A76", CONFIGS_RK3588_KV_SWEEP, str(tmp_path)
        )
        assert "Cortex-A76" in report
        assert "A76T4" not in report


# ---------------------------------------------------------------------------
# generate_cross_validation
# ---------------------------------------------------------------------------


class TestGenerateCrossValidation:
    def test_no_data_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = generate_cross_validation(str(tmp_path))
        assert result is None

    def test_single_device_returns_none(self, tmp_path, monkeypatch):
        """Need both t3 and t4 KV sweep data for cross-validation."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        _write_ctx_csv(
            results_raw / "rk3588-t3_big_ctx_sweep_4b_int8w_fp32kv.csv",
            [
                {
                    "model": "4B",
                    "ctx_len": "1",
                    "gdn_layer_us": "100",
                    "full_attn_us": "200",
                    "ffn_us": "50",
                    "total_us": "350",
                    "tok_per_sec": "1.84",
                    "kv_cache_mb": "0.1",
                }
            ],
        )
        monkeypatch.chdir(tmp_path)
        result = generate_cross_validation(str(tmp_path))
        assert result is None

    def test_two_devices_produces_table(self, tmp_path, monkeypatch):
        """Cross-validation requires matching KV sweep CSVs for both t3 and t4."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        for device, tok in [("rk3588-t3", "1.84"), ("rk3588-t4", "1.83")]:
            _write_ctx_csv(
                results_raw / f"{device}_big_ctx_sweep_4b_int8w_fp32kv.csv",
                [
                    {
                        "model": "4B",
                        "ctx_len": "1",
                        "gdn_layer_us": "119000",
                        "full_attn_us": "38000",
                        "ffn_us": "389000",
                        "total_us": "546000",
                        "tok_per_sec": tok,
                        "kv_cache_mb": "0.1",
                    },
                    {
                        "model": "4B",
                        "ctx_len": "4096",
                        "gdn_layer_us": "120000",
                        "full_attn_us": "189000",
                        "ffn_us": "390000",
                        "total_us": "699000",
                        "tok_per_sec": str(float(tok) * 0.65),
                        "kv_cache_mb": "64.0",
                    },
                ],
            )
        monkeypatch.chdir(tmp_path)
        result = generate_cross_validation(str(tmp_path))
        assert result is not None
        assert "Cross-Validation" in result
        assert "Consistency assessment" in result
        assert "4B INT8w FP32kv" in result
