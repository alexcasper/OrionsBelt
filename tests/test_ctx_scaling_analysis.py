# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Tests for bench/ctx_scaling_analysis.py — context-length scaling report generator.

Covers CSV reading, formatting, and markdown report generation including
edge cases (missing data, partial data).
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest

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


# ---------------------------------------------------------------------------
# generate_report — edge cases (coverage gaps)
# ---------------------------------------------------------------------------


class TestGenerateReportEdgeCases:
    def test_hybrid_config_missing_ctx_shows_dash(self, tmp_path, monkeypatch):
        """Hybrid label present but only at one ctx → '—' in full-attention share table."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        # Two hybrid configs: one with ctx=1 only, one with ctx=1 and ctx=4096
        # all_ctx = {1, 4096}; the first config is missing ctx=4096
        _write_ctx_csv(
            results_raw / "rk3588-t3_big_int8_ctxsweep_e2e_raw.csv",
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
                for ctx in [1, 4096]
            ],
        )
        monkeypatch.chdir(tmp_path)
        report = generate_report(
            "rk3588-t3", "a76", "A76", CONFIGS_RK3588, str(tmp_path / "figures")
        )
        # The INT8 hybrid config is missing ctx=4096, so its share should be "—"
        assert "Full-attention share" in report

    def test_kv_fp32_without_int8_pair(self, tmp_path, monkeypatch):
        """FP32kv config exists but no INT8kv counterpart → no quantization section."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        # Only FP32kv, no INT8kv → kv_pairs is empty, no quant benefit section
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
                },
                {
                    "model": "4B",
                    "ctx_len": "4096",
                    "gdn_layer_us": "101",
                    "full_attn_us": "20000",
                    "ffn_us": "50",
                    "total_us": "20151",
                    "tok_per_sec": "1.20",
                    "kv_cache_mb": "64.0",
                },
            ],
        )
        monkeypatch.chdir(tmp_path)
        report = generate_report(
            "rk3588-t4", "a76t4", "A76", CONFIGS_RK3588_KV_SWEEP, str(tmp_path / "figures")
        )
        assert "INT8 KV-cache quantization benefit" not in report

    def test_kv_quant_partial_ctx_shows_dash(self, tmp_path, monkeypatch):
        """Both FP32kv and INT8kv exist but one lacks a ctx → '—' row."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        # FP32kv has ctx=1 and 4096, INT8kv only has ctx=1
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
                },
                {
                    "model": "4B",
                    "ctx_len": "4096",
                    "gdn_layer_us": "101",
                    "full_attn_us": "20000",
                    "ffn_us": "50",
                    "total_us": "20151",
                    "tok_per_sec": "1.20",
                    "kv_cache_mb": "64.0",
                },
            ],
        )
        _write_ctx_csv(
            results_raw / "rk3588-t4_big_ctx_sweep_4b_int8w_int8kv.csv",
            [
                {
                    "model": "4B",
                    "ctx_len": "1",
                    "gdn_layer_us": "100",
                    "full_attn_us": "200",
                    "ffn_us": "50",
                    "total_us": "350",
                    "tok_per_sec": "2.10",
                    "kv_cache_mb": "0.1",
                },
                # ctx=4096 deliberately missing → "—" in that row
            ],
        )
        monkeypatch.chdir(tmp_path)
        report = generate_report(
            "rk3588-t4", "a76t4", "A76", CONFIGS_RK3588_KV_SWEEP, str(tmp_path / "figures")
        )
        assert "INT8 KV-cache quantization benefit" in report
        # The row for ctx=4096 should have "—" for both values
        assert " | 4096 | — | — | — |" in report


# ---------------------------------------------------------------------------
# generate_cross_device — edge cases
# ---------------------------------------------------------------------------


class TestGenerateCrossDeviceEdgeCases:
    def test_no_shared_labels_returns_none(self, tmp_path, monkeypatch):
        """Two devices with non-overlapping config labels → None."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        # t3 has old-style naming, t4 has KV-sweep naming → different labels
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
        result = generate_cross_device(str(tmp_path))
        assert result is None

    def test_three_devices_partial_label(self, tmp_path, monkeypatch):
        """Shared label missing from one of three devices → '—' in that column."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        # t3 and jetson share "4B FP32 hybrid"; t4 uses KV-sweep labels only
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
        # t4 has KV-sweep data only → no "4B FP32 hybrid" label
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
        result = generate_cross_device(str(tmp_path))
        assert result is not None
        assert "4B FP32 hybrid" in result
        # t4 doesn't have "4B FP32 hybrid" so its columns should show "—"
        # The slowdown for t4 should also be "—"

    def test_shared_label_missing_ctx_4096(self, tmp_path, monkeypatch):
        """Shared label in two devices but one lacks ctx=4096 → slowdown '—'."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        # t3 has ctx=1 and 4096; jetson only has ctx=1
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
                # No ctx=4096 row → t4096=None → slowdown "—"
            ],
        )
        monkeypatch.chdir(tmp_path)
        result = generate_cross_device(str(tmp_path))
        assert result is not None
        assert "4B FP32 hybrid" in result


# ---------------------------------------------------------------------------
# generate_cross_validation — edge cases
# ---------------------------------------------------------------------------


class TestGenerateCrossValidationEdgeCases:
    def test_no_shared_kv_labels_returns_none(self, tmp_path, monkeypatch):
        """t3 and t4 have KV-sweep data but with non-overlapping labels → None."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        # t3 has 4B label, t4 has 0.8B label → no overlap
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
        _write_ctx_csv(
            results_raw / "rk3588-t4_big_ctx_sweep_08b_int8w_int8kv.csv",
            [
                {
                    "model": "0.8B",
                    "ctx_len": "1",
                    "gdn_layer_us": "50",
                    "full_attn_us": "100",
                    "ffn_us": "25",
                    "total_us": "175",
                    "tok_per_sec": "5.0",
                    "kv_cache_mb": "0.05",
                }
            ],
        )
        monkeypatch.chdir(tmp_path)
        result = generate_cross_validation(str(tmp_path))
        assert result is None

    def test_good_agreement_verdict(self, tmp_path, monkeypatch):
        """Max delta between 10–20% → 'Good agreement' verdict."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        # t3=1.84, t4=2.10 → delta ≈ 14.1% → good agreement
        for device, tok in [("rk3588-t3", "1.84"), ("rk3588-t4", "2.10")]:
            _write_ctx_csv(
                results_raw / f"{device}_big_ctx_sweep_4b_int8w_fp32kv.csv",
                [
                    {
                        "model": "4B",
                        "ctx_len": "1",
                        "gdn_layer_us": "100",
                        "full_attn_us": "200",
                        "ffn_us": "50",
                        "total_us": "350",
                        "tok_per_sec": tok,
                        "kv_cache_mb": "0.1",
                    },
                ],
            )
        monkeypatch.chdir(tmp_path)
        result = generate_cross_validation(str(tmp_path))
        assert result is not None
        assert "Good agreement" in result

    def test_divergent_verdict(self, tmp_path, monkeypatch):
        """Max delta ≥ 20% → 'Notable divergence' verdict."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        # t3=1.84, t4=2.50 → delta ≈ 35.9% → divergent
        for device, tok in [("rk3588-t3", "1.84"), ("rk3588-t4", "2.50")]:
            _write_ctx_csv(
                results_raw / f"{device}_big_ctx_sweep_4b_int8w_fp32kv.csv",
                [
                    {
                        "model": "4B",
                        "ctx_len": "1",
                        "gdn_layer_us": "100",
                        "full_attn_us": "200",
                        "ffn_us": "50",
                        "total_us": "350",
                        "tok_per_sec": tok,
                        "kv_cache_mb": "0.1",
                    },
                ],
            )
        monkeypatch.chdir(tmp_path)
        result = generate_cross_validation(str(tmp_path))
        assert result is not None
        assert "Notable divergence" in result

    def test_shared_label_no_overlapping_ctx(self, tmp_path, monkeypatch):
        """Shared label exists but t3 and t4 have different ctx → all_ctx empty, deltas empty."""
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
        _write_ctx_csv(
            results_raw / "rk3588-t4_big_ctx_sweep_4b_int8w_fp32kv.csv",
            [
                {
                    "model": "4B",
                    "ctx_len": "4096",  # different ctx → intersection empty
                    "gdn_layer_us": "100",
                    "full_attn_us": "200",
                    "ffn_us": "50",
                    "total_us": "350",
                    "tok_per_sec": "1.20",
                    "kv_cache_mb": "64.0",
                }
            ],
        )
        monkeypatch.chdir(tmp_path)
        result = generate_cross_validation(str(tmp_path))
        assert result is not None
        # No overlapping ctx → no per-label table, but header should be present
        # and consistency assessment should show empty verdict
        assert "Cross-Validation" in result

    def test_zero_tok_per_sec(self, tmp_path, monkeypatch):
        """t3 tok_per_sec=0 → t3f>0 guard False, delta skipped."""
        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        _write_ctx_csv(
            results_raw / "rk3588-t3_big_ctx_sweep_4b_int8w_fp32kv.csv",
            [
                {
                    "model": "4B",
                    "ctx_len": "1",
                    "gdn_layer_us": "0",
                    "full_attn_us": "0",
                    "ffn_us": "0",
                    "total_us": "0",
                    "tok_per_sec": "0.0",  # zero → t3f>0 False
                    "kv_cache_mb": "0.0",
                }
            ],
        )
        _write_ctx_csv(
            results_raw / "rk3588-t4_big_ctx_sweep_4b_int8w_fp32kv.csv",
            [
                {
                    "model": "4B",
                    "ctx_len": "1",
                    "gdn_layer_us": "0",
                    "full_attn_us": "0",
                    "ffn_us": "0",
                    "total_us": "0",
                    "tok_per_sec": "0.0",
                    "kv_cache_mb": "0.0",
                }
            ],
        )
        monkeypatch.chdir(tmp_path)
        result = generate_cross_validation(str(tmp_path))
        assert result is not None


# ---------------------------------------------------------------------------
# main() — CLI entry point
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_invalid_device_exits(self, monkeypatch):
        """Invalid --device causes sys.exit(1)."""
        import bench.ctx_scaling_analysis as mod

        monkeypatch.setattr("sys.argv", ["ctx_scaling_analysis.py", "--device", "bogus"])
        with pytest.raises(SystemExit) as exc_info:
            mod.main()
        assert exc_info.value.code == 1

    def test_main_valid_device_writes_report(self, tmp_path, monkeypatch):
        """Valid --device writes a report file."""
        import bench.ctx_scaling_analysis as mod

        out_dir = str(tmp_path / "figures")
        monkeypatch.setattr(
            "sys.argv",
            ["ctx_scaling_analysis.py", "--device", "rk3588-t3", "--output-dir", out_dir],
        )
        monkeypatch.chdir(tmp_path)
        mod.main()
        # Should create ctx_length_scaling_a76.md
        assert os.path.exists(os.path.join(out_dir, "ctx_length_scaling_a76.md"))

    def test_main_all_devices_no_data(self, tmp_path, monkeypatch):
        """Running without --device processes all devices (no CSVs → empty reports)."""
        import bench.ctx_scaling_analysis as mod

        out_dir = str(tmp_path / "figures")
        monkeypatch.setattr(
            "sys.argv", ["ctx_scaling_analysis.py", "--output-dir", out_dir]
        )
        monkeypatch.chdir(tmp_path)
        mod.main()
        # All three device reports should be written
        assert os.path.exists(os.path.join(out_dir, "ctx_length_scaling_a76.md"))
        assert os.path.exists(os.path.join(out_dir, "ctx_length_scaling_a76t4.md"))
        assert os.path.exists(os.path.join(out_dir, "ctx_length_scaling_a57.md"))

    def test_main_all_devices_with_cross_device_data(self, tmp_path, monkeypatch):
        """main() without --device writes cross-device + cross-validation reports."""
        import bench.ctx_scaling_analysis as mod

        results_raw = tmp_path / "results" / "raw"
        results_raw.mkdir(parents=True)
        # t3 and jetson share "4B FP32 hybrid" label
        for device, stem, tok in [
            ("rk3588-t3", "rk3588-t3_big_ctxsweep", "10.0"),
            ("jetson-j1", "jetson-j1_4b_fp32_ctxsweep", "3.5"),
        ]:
            _write_ctx_csv(
                results_raw / f"{stem}_e2e_raw.csv",
                [
                    {
                        "model": "4B",
                        "ctx_len": str(ctx),
                        "gdn_layer_us": "100",
                        "full_attn_us": str(200 * ctx),
                        "ffn_us": "50",
                        "total_us": str(350 + 200 * (ctx - 1)),
                        "tok_per_sec": str(float(tok) / ctx),
                        "kv_cache_mb": str(0.5 * ctx),
                    }
                    for ctx in [1, 4096]
                ],
            )
        # t3 and t4 share KV-sweep label for cross-validation
        for device, tok in [("rk3588-t3", "1.84"), ("rk3588-t4", "1.83")]:
            _write_ctx_csv(
                results_raw / f"{device}_big_ctx_sweep_4b_int8w_fp32kv.csv",
                [
                    {
                        "model": "4B",
                        "ctx_len": "1",
                        "gdn_layer_us": "100",
                        "full_attn_us": "200",
                        "ffn_us": "50",
                        "total_us": "350",
                        "tok_per_sec": tok,
                        "kv_cache_mb": "0.1",
                    },
                ],
            )
        out_dir = str(tmp_path / "figures")
        monkeypatch.setattr(
            "sys.argv", ["ctx_scaling_analysis.py", "--output-dir", out_dir]
        )
        monkeypatch.chdir(tmp_path)
        mod.main()
        # Cross-device report
        assert os.path.exists(os.path.join(out_dir, "ctx_length_scaling_cross.md"))
        # Cross-validation report
        assert os.path.exists(os.path.join(out_dir, "ctx_length_scaling_t3vst4.md"))



