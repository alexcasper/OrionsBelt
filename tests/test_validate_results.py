# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/validate_results.py — benchmark CSV and manifest validation.

Covers CSV-type detection, row-level sanity checks (impossible latencies, absurd
throughput, low repeats), manifest linkage, and end-to-end exit codes.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.validate_results import (  # noqa: E402
    ABSURD_THROUGHPUT,
    E2E_SWEEP_COLS,
    GPU_MICRO_COLS,
    KLEIDIAI_GDN_KERNEL_COLS,
    KLEIDIAI_MATMUL_COLS,
    LAYER_PROFILE_COLS,
    STANDARD_COLS,
    SUSTAINED_COLS,
    Issue,
    check_ablation_manifests,
    check_manifest_exists,
    check_readme_counts,
    detect_csv_type,
    expected_columns,
    find_device_spec,
    get_git_head_sha,
    load_manifest,
    main,
    validate_csv,
    validate_ctx_sweep_row,
    validate_delta_matmul_row,
    validate_e2e_sweep_row,
    validate_gpu_micro_row,
    validate_kleidiai_gdn_kernel_row,
    validate_kleidiai_matmul_row,
    validate_layer_profile_row,
    validate_manifest,
    validate_standard_row,
    validate_sustained_row,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STD_HEADER = STANDARD_COLS  # standard benchmark columns


def _std_row(**overrides):
    """A valid standard-row dict."""
    base = {
        "model": "qwen35_4b",
        "kernel": "gdn_gated_scan",
        "dispatch_path": "sve2",
        "seq": "64",
        "channels": "128",
        "repeats": "30",
        "p50_us": "1000.0",
        "p95_us": "1100.0",
        "spread_pct": "10.0",
        "gib_per_s_p50": "2.5",
        "gflop_per_s_p50": "50.0",
    }
    base.update(overrides)
    return base


def _sustained_row(**overrides):
    """A valid sustained-row dict."""
    base = {
        "sustained_model": "qwen35_4b",
        "sustained_kernel": "gdn_gated_scan",
        "dispatch_path": "sve2",
        "elapsed_s": "10.0",
        "throughput_gibs": "2.3",
        "thermal_c": "55.0",
        "vs_first_pct": "-5.0",
    }
    base.update(overrides)
    return base


def _layer_profile_row(**overrides):
    """A valid layer-profile row dict."""
    base = {
        "phase": "decode",
        "ctx_len": "64",
        "layer_idx": "0",
        "layer_type": "linear_attention",
        "p50_us": "100.0",
        "p95_us": "120.0",
        "mean_us": "105.0",
        "n_samples": "10",
    }
    base.update(overrides)
    return base


def _e2e_sweep_row(**overrides):
    """A valid e2e context-sweep row dict."""
    base = {
        "run_id": "rk3588-t4_20260806",
        "timestamp": "2026-08-06T09:47:32Z",
        "git_sha": "a37e116",
        "manifest_ref": "results/manifests/rk3588-t4.json",
        "device": "rk3588-t4",
        "engine_gdn": "cpu",
        "engine_full_attention": "cpu",
        "model_checkpoint": "Qwen3.5-0.8B",
        "quantization": "fp32",
        "context_length": "128",
        "phase": "prefill",
        "metric_name": "prefill_tokens_per_sec",
        "metric_component": "",
        "value": "21.1",
        "unit": "tokens_per_sec",
        "repeat_index": "0",
        "repeat_count": "5",
        "layer_class": "all",
        "notes": "",
    }
    base.update(overrides)
    return base


def _write_std_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STD_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# detect_csv_type
# ---------------------------------------------------------------------------


class TestDetectCsvType:
    def test_standard_detected(self):
        assert detect_csv_type(STD_HEADER) == "standard"

    def test_sustained_detected(self):
        cols = [
            "sustained_model",
            "sustained_kernel",
            "dispatch_path",
            "elapsed_s",
            "throughput_gibs",
            "thermal_c",
            "vs_first_pct",
        ]
        assert detect_csv_type(cols) == "sustained"

    def test_power_detected(self):
        cols = [
            "timestamp_ms",
            "power_in_mw",
            "power_gpu_mw",
            "power_cpu_mw",
            "temp_milliC",
        ]
        assert detect_csv_type(cols) == "power"

    def test_unrecognized_returns_none(self):
        assert detect_csv_type(["foo", "bar", "baz"]) is None

    def test_empty_header_returns_none(self):
        assert detect_csv_type([]) is None

    def test_standard_with_extra_cols_still_standard(self):
        """Extra columns should not break detection."""
        cols = STD_HEADER + ["layer_class", "extra_col"]
        assert detect_csv_type(cols) == "standard"

    def test_layer_profile_detected(self):
        """Per-layer latency profiling CSV should be detected."""
        assert detect_csv_type(LAYER_PROFILE_COLS) == "layer_profile"

    def test_layer_profile_minimal_cols(self):
        """Detection only needs the key columns, not the full set."""
        cols = ["layer_idx", "layer_type", "p50_us", "mean_us"]
        assert detect_csv_type(cols) == "layer_profile"

    def test_e2e_sweep_detected(self):
        """E2E context-sweep CSV should be detected."""
        assert detect_csv_type(E2E_SWEEP_COLS) == "e2e_sweep"

    def test_e2e_sweep_minimal_cols(self):
        """Detection only needs the key columns."""
        cols = ["run_id", "metric_name", "metric_component", "repeat_index"]
        assert detect_csv_type(cols) == "e2e_sweep"

    def test_gpu_micro_detected(self):
        """GPU microbenchmark CSV should be detected."""
        assert detect_csv_type(GPU_MICRO_COLS) == "gpu_micro"

    def test_gpu_micro_minimal_cols(self):
        """Detection only needs the key columns."""
        cols = ["bw_mibs", "p50_ms", "dim1"]
        assert detect_csv_type(cols) == "gpu_micro"

    def test_kleidiai_matmul_detected(self):
        """KleidiAI matmul CSV should be detected."""
        assert detect_csv_type(KLEIDIAI_MATMUL_COLS) == "kleidiai_matmul"

    def test_kleidiai_matmul_minimal_cols(self):
        """Detection only needs the key columns."""
        cols = ["shape", "impl", "GiB_s", "GFLOP_s"]
        assert detect_csv_type(cols) == "kleidiai_matmul"

    def test_kleidiai_gdn_kernel_detected(self):
        """KleidiAI GDN kernel CSV should be detected."""
        assert detect_csv_type(KLEIDIAI_GDN_KERNEL_COLS) == "kleidiai_gdn_kernel"

    def test_kleidiai_gdn_kernel_minimal_cols(self):
        """Detection only needs the key columns."""
        cols = ["kernel", "shape", "p50_us", "gib_per_s_p50"]
        assert detect_csv_type(cols) == "kleidiai_gdn_kernel"

    def test_ctx_sweep_detected(self):
        """Context-length sweep CSV should be detected (most common CSV type)."""
        cols = ["model", "ctx_len", "gdn_layer_us", "full_attn_us", "kv_cache_mb"]
        assert detect_csv_type(cols) == "ctx_sweep"

    def test_ctx_sweep_minimal_cols(self):
        """Detection only needs the four key columns."""
        cols = ["ctx_len", "gdn_layer_us", "full_attn_us", "kv_cache_mb"]
        assert detect_csv_type(cols) == "ctx_sweep"

    def test_e2e_decode_detected(self):
        """E2E decode CSV should be detected."""
        cols = ["tok_per_sec_mean", "gdn_proj_pct", "ffn_pct"]
        assert detect_csv_type(cols) == "e2e_decode"

    def test_thermal_stress_detected(self):
        """Thermal stress test CSV should be detected."""
        cols = ["iteration", "tok_per_sec", "thermal_zone1_C", "elapsed_s"]
        assert detect_csv_type(cols) == "thermal_stress"

    def test_prefill_gemm_detected(self):
        """Prefill GEMM CSV should be detected."""
        cols = ["prefill_M", "ttft_ms", "tok_per_sec_prefill"]
        assert detect_csv_type(cols) == "prefill_gemm"

    def test_prefill_ab_detected(self):
        """Prefill A/B comparison CSV should be detected."""
        cols = ["variant", "prefill_len", "ttft_s", "prefill_tps"]
        assert detect_csv_type(cols) == "prefill_ab"

    def test_quant_comparison_detected(self):
        """Quantization comparison CSV should be detected."""
        cols = ["variant", "tok_per_sec", "ffn_pct", "gdn_proj_pct"]
        assert detect_csv_type(cols) == "quant_comparison"

    def test_quant_accuracy_detected(self):
        """Quantization accuracy CSV should be detected."""
        cols = ["quant_variant", "matrix", "cos_sim", "rel_err_pct"]
        assert detect_csv_type(cols) == "quant_accuracy"

    def test_cross_tool_comparison_detected(self):
        """Cross-tool comparison CSV should be detected."""
        cols = ["engine", "quant", "test", "n_tokens", "avg_ts"]
        assert detect_csv_type(cols) == "cross_tool_comparison"


# ---------------------------------------------------------------------------
# expected_columns
# ---------------------------------------------------------------------------


class TestExpectedColumns:
    def test_standard_columns(self):
        assert "p50_us" in expected_columns("standard")

    def test_sustained_columns(self):
        assert "throughput_gibs" in expected_columns("sustained")

    def test_power_columns(self):
        assert "power_in_mw" in expected_columns("power")

    def test_unknown_type_returns_empty(self):
        assert expected_columns("unknown") == []

    def test_layer_profile_columns(self):
        assert "layer_idx" in expected_columns("layer_profile")
        assert "layer_type" in expected_columns("layer_profile")

    def test_e2e_sweep_columns(self):
        assert "metric_name" in expected_columns("e2e_sweep")
        assert "context_length" in expected_columns("e2e_sweep")

    def test_gpu_micro_columns(self):
        assert "bw_mibs" in expected_columns("gpu_micro")
        assert "p50_ms" in expected_columns("gpu_micro")

    def test_kleidiai_matmul_columns(self):
        assert "us_per_call" in expected_columns("kleidiai_matmul")
        assert "GiB_s" in expected_columns("kleidiai_matmul")
        assert "GFLOP_s" in expected_columns("kleidiai_matmul")

    def test_kleidiai_gdn_kernel_columns(self):
        assert "p50_us" in expected_columns("kleidiai_gdn_kernel")
        assert "gib_per_s_p50" in expected_columns("kleidiai_gdn_kernel")
        assert "kernel" in expected_columns("kleidiai_gdn_kernel")

    def test_ctx_sweep_columns(self):
        assert "ctx_len" in expected_columns("ctx_sweep")
        assert "gdn_layer_us" in expected_columns("ctx_sweep")

    def test_delta_matmul_columns(self):
        assert "M" in expected_columns("delta_matmul")
        assert "p50_us" in expected_columns("delta_matmul")

    def test_e2e_decode_columns(self):
        assert "tok_per_sec_mean" in expected_columns("e2e_decode")

    def test_thermal_stress_columns(self):
        assert "thermal_zone1_C" in expected_columns("thermal_stress")
        assert "elapsed_s" in expected_columns("thermal_stress")

    def test_prefill_gemm_columns(self):
        assert "prefill_M" in expected_columns("prefill_gemm")
        assert "ttft_ms" in expected_columns("prefill_gemm")

    def test_prefill_ab_columns(self):
        assert "variant" in expected_columns("prefill_ab")
        assert "prefill_tps" in expected_columns("prefill_ab")

    def test_quant_comparison_columns(self):
        assert "variant" in expected_columns("quant_comparison")
        assert "tok_per_sec" in expected_columns("quant_comparison")

    def test_quant_accuracy_columns(self):
        assert "cos_sim" in expected_columns("quant_accuracy")
        assert "rel_err_pct" in expected_columns("quant_accuracy")

    def test_cross_tool_comparison_columns(self):
        assert "engine" in expected_columns("cross_tool_comparison")
        assert "avg_ts" in expected_columns("cross_tool_comparison")


# ---------------------------------------------------------------------------
# find_device_spec
# ---------------------------------------------------------------------------


class TestFindDeviceSpec:
    def test_jetson(self):
        assert find_device_spec("jetson-j1.csv") == 23.8

    def test_pi5(self):
        assert find_device_spec("pi5-r5.csv") == 15.8

    def test_rk3588(self):
        assert find_device_spec("rk3588-t4_big.csv") == 31.7

    def test_orion(self):
        assert find_device_spec("orion-o6_big.csv") == 93.1

    def test_unknown_returns_none(self):
        assert find_device_spec("o6-something.csv") is None


# ---------------------------------------------------------------------------
# validate_standard_row
# ---------------------------------------------------------------------------


class TestValidateStandardRow:
    def test_valid_row_no_issues(self):
        issues = []
        validate_standard_row(_std_row(), "test.csv", issues, 2)
        assert issues == []

    def test_negative_latency(self):
        issues = []
        validate_standard_row(_std_row(p50_us="-1.0"), "test.csv", issues, 2)
        assert any("non-positive" in i.message for i in issues)

    def test_p95_less_than_p50(self):
        issues = []
        validate_standard_row(_std_row(p50_us="1000.0", p95_us="500.0"), "test.csv", issues, 2)
        assert any("p95" in i.message and "p50" in i.message for i in issues)

    def test_absurd_throughput(self):
        issues = []
        validate_standard_row(
            _std_row(gib_per_s_p50=str(ABSURD_THROUGHPUT + 1)), "test.csv", issues, 2
        )
        assert any("absurd" in i.message for i in issues)

    def test_extreme_spread_warning(self):
        issues = []
        validate_standard_row(_std_row(spread_pct="250.0"), "test.csv", issues, 2)
        assert any("extreme spread" in i.message and i.severity == "WARNING" for i in issues)

    def test_too_few_repeats(self):
        issues = []
        validate_standard_row(_std_row(repeats="3"), "test.csv", issues, 2)
        assert any("repeats" in i.message and i.severity == "ERROR" for i in issues)

    def test_malformed_value(self):
        issues = []
        validate_standard_row(_std_row(p50_us="not_a_number"), "test.csv", issues, 2)
        assert any("cannot parse" in i.message for i in issues)

    def test_missing_column(self):
        issues = []
        row = _std_row()
        del row["p50_us"]
        validate_standard_row(row, "test.csv", issues, 2)
        assert any("cannot parse" in i.message for i in issues)


# ---------------------------------------------------------------------------
# validate_sustained_row
# ---------------------------------------------------------------------------


class TestValidateSustainedRow:
    def test_valid_row_no_issues(self):
        issues = []
        validate_sustained_row(_sustained_row(), "test.csv", issues, 2)
        assert issues == []

    def test_absurd_throughput(self):
        issues = []
        validate_sustained_row(
            _sustained_row(throughput_gibs=str(ABSURD_THROUGHPUT + 1)), "test.csv", issues, 2
        )
        assert any("absurd" in i.message for i in issues)

    def test_high_thermal_warning(self):
        issues = []
        validate_sustained_row(_sustained_row(thermal_c="130.0"), "test.csv", issues, 2)
        assert any("thermal" in i.message and i.severity == "WARNING" for i in issues)

    def test_non_positive_elapsed(self):
        issues = []
        validate_sustained_row(_sustained_row(elapsed_s="0.0"), "test.csv", issues, 2)
        assert any("elapsed" in i.message for i in issues)


# ---------------------------------------------------------------------------
# validate_layer_profile_row
# ---------------------------------------------------------------------------


class TestValidateLayerProfileRow:
    def test_valid_row_no_issues(self):
        issues = []
        validate_layer_profile_row(_layer_profile_row(), "test.csv", issues, 2)
        assert issues == []

    def test_p95_less_than_p50(self):
        issues = []
        validate_layer_profile_row(
            _layer_profile_row(p50_us="200.0", p95_us="100.0"), "test.csv", issues, 2
        )
        assert any("p95" in i.message and i.severity == "WARNING" for i in issues)

    def test_non_positive_p50(self):
        issues = []
        validate_layer_profile_row(_layer_profile_row(p50_us="0.0"), "test.csv", issues, 2)
        assert any("p50_us" in i.message for i in issues)

    def test_zero_samples(self):
        issues = []
        validate_layer_profile_row(_layer_profile_row(n_samples="0"), "test.csv", issues, 2)
        assert any("n_samples" in i.message for i in issues)

    def test_malformed_value(self):
        issues = []
        validate_layer_profile_row(_layer_profile_row(p50_us="abc"), "test.csv", issues, 2)
        assert any("cannot parse" in i.message for i in issues)


# ---------------------------------------------------------------------------
# validate_e2e_sweep_row
# ---------------------------------------------------------------------------


class TestValidateE2eSweepRow:
    def test_valid_row_no_issues(self):
        issues = []
        validate_e2e_sweep_row(_e2e_sweep_row(), "test.csv", issues, 2)
        assert issues == []

    def test_repeat_index_out_of_range(self):
        issues = []
        validate_e2e_sweep_row(
            _e2e_sweep_row(repeat_index="5", repeat_count="5"), "test.csv", issues, 2
        )
        assert any("repeat_index" in i.message for i in issues)

    def test_non_positive_throughput(self):
        issues = []
        validate_e2e_sweep_row(
            _e2e_sweep_row(value="0.0", metric_name="prefill_tokens_per_sec"),
            "test.csv",
            issues,
            2,
        )
        assert any("non-positive" in i.message for i in issues)

    def test_zero_context_length(self):
        issues = []
        validate_e2e_sweep_row(_e2e_sweep_row(context_length="0"), "test.csv", issues, 2)
        assert any("context_length" in i.message for i in issues)

    def test_malformed_value(self):
        issues = []
        validate_e2e_sweep_row(_e2e_sweep_row(value="not_a_number"), "test.csv", issues, 2)
        assert any("cannot parse" in i.message for i in issues)

    def test_negative_value(self):
        issues = []
        validate_e2e_sweep_row(_e2e_sweep_row(value="-1.5"), "test.csv", issues, 2)
        assert any("negative" in i.message for i in issues)

    def test_repeat_count_zero(self):
        issues = []
        validate_e2e_sweep_row(_e2e_sweep_row(repeat_count="0"), "test.csv", issues, 2)
        assert any("repeat_count" in i.message and i.severity == "ERROR" for i in issues)

    def test_unexpected_phase(self):
        issues = []
        validate_e2e_sweep_row(_e2e_sweep_row(phase="weird"), "test.csv", issues, 2)
        assert any("phase" in i.message and i.severity == "WARNING" for i in issues)

    def test_high_throughput_warning(self):
        issues = []
        validate_e2e_sweep_row(
            _e2e_sweep_row(value="999999", metric_name="prefill_tokens_per_sec"),
            "test.csv",
            issues,
            2,
        )
        assert any("high throughput" in i.message for i in issues)


# ---------------------------------------------------------------------------
# validate_gpu_micro_row
# ---------------------------------------------------------------------------


def _gpu_row(**overrides):
    """A valid GPU microbenchmark row."""
    base = {
        "kernel": "gdn_gated_scan",
        "dim1": "64",
        "dim2": "2048",
        "dim3": "",
        "p50_ms": "0.948",
        "p95_ms": "1.0477",
        "bw_mibs": "1598.8",
    }
    base.update(overrides)
    return base


class TestValidateGpuMicroRow:
    def test_valid_row_no_issues(self):
        issues = []
        validate_gpu_micro_row(_gpu_row(), "test.csv", issues, 2)
        assert issues == []

    def test_valid_row_no_dim3_no_p95(self):
        """Kernels like gdn_cumdecay only have p50, not p95 or dim3."""
        issues = []
        validate_gpu_micro_row(
            _gpu_row(kernel="gdn_cumdecay", dim3="", p95_ms="", bw_mibs="1182.2"),
            "test.csv",
            issues,
            2,
        )
        assert issues == []

    def test_valid_row_with_dim3(self):
        """gdn_delta_rule_decode has dim3."""
        issues = []
        validate_gpu_micro_row(
            _gpu_row(kernel="gdn_delta_rule_decode", dim1="16", dim2="128", dim3="128"),
            "test.csv",
            issues,
            2,
        )
        assert issues == []

    def test_unknown_kernel(self):
        issues = []
        validate_gpu_micro_row(_gpu_row(kernel="mystery_kernel"), "test.csv", issues, 2)
        assert any("unknown kernel" in i.message for i in issues)

    def test_non_positive_bw(self):
        issues = []
        validate_gpu_micro_row(_gpu_row(bw_mibs="0.0"), "test.csv", issues, 2)
        assert any("bw_mibs" in i.message and i.severity == "ERROR" for i in issues)

    def test_p95_less_than_p50(self):
        issues = []
        validate_gpu_micro_row(_gpu_row(p50_ms="2.0", p95_ms="1.0"), "test.csv", issues, 2)
        assert any("p95" in i.message and i.severity == "WARNING" for i in issues)

    def test_non_positive_dims(self):
        issues = []
        validate_gpu_micro_row(_gpu_row(dim1="0"), "test.csv", issues, 2)
        assert any("dim1" in i.message and i.severity == "ERROR" for i in issues)

    def test_malformed_p50(self):
        issues = []
        validate_gpu_micro_row(_gpu_row(p50_ms="abc"), "test.csv", issues, 2)
        assert any("cannot parse p50_ms" in i.message for i in issues)


# ---------------------------------------------------------------------------
# KleidiAI matmul row validation
# ---------------------------------------------------------------------------


def _kleidiai_row(**overrides):
    """A valid KleidiAI matmul benchmark row."""
    base = {
        "shape": "decode_1x128x128",
        "impl": "kleidiai",
        "M": "1",
        "K": "128",
        "N": "128",
        "us_per_call": "1.855",
        "GiB_s": "33.41",
        "GFLOP_s": "17.66",
    }
    base.update(overrides)
    return base


class TestValidateKleidiaiMatmulRow:
    def test_valid_row_no_issues(self):
        issues = []
        validate_kleidiai_matmul_row(_kleidiai_row(), "test.csv", issues, 2)
        assert issues == []

    def test_valid_naive_impl(self):
        issues = []
        validate_kleidiai_matmul_row(
            _kleidiai_row(impl="naive", us_per_call="13.620", GiB_s="4.55", GFLOP_s="2.41"),
            "test.csv",
            issues,
            2,
        )
        assert issues == []

    def test_non_positive_us(self):
        issues = []
        validate_kleidiai_matmul_row(_kleidiai_row(us_per_call="0.0"), "test.csv", issues, 2)
        assert any("non-positive us_per_call" in i.message for i in issues)

    def test_absurd_gibs(self):
        issues = []
        validate_kleidiai_matmul_row(
            _kleidiai_row(GiB_s=str(ABSURD_THROUGHPUT + 1)),
            "test.csv",
            issues,
            2,
        )
        assert any("absurd GiB_s" in i.message for i in issues)

    def test_negative_gibs(self):
        issues = []
        validate_kleidiai_matmul_row(_kleidiai_row(GiB_s="-1.0"), "test.csv", issues, 2)
        assert any("negative GiB_s" in i.message for i in issues)

    def test_negative_gflops(self):
        issues = []
        validate_kleidiai_matmul_row(_kleidiai_row(GFLOP_s="-5.0"), "test.csv", issues, 2)
        assert any("negative GFLOP_s" in i.message for i in issues)

    def test_malformed_us(self):
        issues = []
        validate_kleidiai_matmul_row(_kleidiai_row(us_per_call="abc"), "test.csv", issues, 2)
        assert any("cannot parse us_per_call" in i.message for i in issues)

    def test_malformed_gibs(self):
        issues = []
        validate_kleidiai_matmul_row(_kleidiai_row(GiB_s="N/A"), "test.csv", issues, 2)
        assert any("cannot parse GiB_s" in i.message for i in issues)


def _gdn_kernel_row(**overrides):
    """Factory for a valid kleidiai_gdn_kernel CSV row."""
    base = {
        "kernel": "cumdecay",
        "shape": "64x2560",
        "seq": "64",
        "channels": "2560",
        "repeats": "30",
        "p50_us": "536.959",
        "gib_per_s_p50": "2.27",
    }
    base.update(overrides)
    return base


class TestValidateKleidiaiGdnKernelRow:
    def test_valid_row_no_issues(self):
        issues = []
        validate_kleidiai_gdn_kernel_row(_gdn_kernel_row(), "test.csv", issues, 2)
        assert issues == []

    def test_valid_gemv_kernel(self):
        issues = []
        validate_kleidiai_gdn_kernel_row(
            _gdn_kernel_row(kernel="gemv", shape="K128_N2560", gib_per_s_p50="14.65"),
            "test.csv",
            issues,
            2,
        )
        assert issues == []

    def test_inf_gibs_is_note_not_warning(self):
        """inf GiB/s from tiny workloads should be a NOTE, not a WARNING."""
        issues = []
        validate_kleidiai_gdn_kernel_row(
            _gdn_kernel_row(p50_us="0.000", gib_per_s_p50="inf"),
            "test.csv",
            issues,
            2,
        )
        assert any("too small to measure" in i.message for i in issues)
        assert all(i.severity == "NOTE" for i in issues)

    def test_negative_us(self):
        issues = []
        validate_kleidiai_gdn_kernel_row(_gdn_kernel_row(p50_us="-1.0"), "test.csv", issues, 2)
        assert any("negative p50_us" in i.message for i in issues)

    def test_absurd_gibs(self):
        issues = []
        validate_kleidiai_gdn_kernel_row(
            _gdn_kernel_row(gib_per_s_p50=str(ABSURD_THROUGHPUT + 1)),
            "test.csv",
            issues,
            2,
        )
        assert any("absurd gib_per_s_p50" in i.message for i in issues)

    def test_negative_gibs(self):
        issues = []
        validate_kleidiai_gdn_kernel_row(
            _gdn_kernel_row(gib_per_s_p50="-1.0"), "test.csv", issues, 2
        )
        assert any("negative gib_per_s_p50" in i.message for i in issues)

    def test_malformed_us(self):
        issues = []
        validate_kleidiai_gdn_kernel_row(_gdn_kernel_row(p50_us="abc"), "test.csv", issues, 2)
        assert any("cannot parse p50_us" in i.message for i in issues)

    def test_unknown_kernel(self):
        issues = []
        validate_kleidiai_gdn_kernel_row(_gdn_kernel_row(kernel="bogus"), "test.csv", issues, 2)
        assert any("unknown kernel" in i.message for i in issues)


class TestCheckManifestExists:
    def test_exact_match(self, tmp_path):
        manifest = tmp_path / "jetson-j1.json"
        manifest.write_text("{}")
        result = check_manifest_exists("jetson-j1.csv", str(tmp_path))
        assert result is not None
        assert "jetson-j1.json" in result

    def test_no_manifest_returns_none(self, tmp_path):
        result = check_manifest_exists("nonexistent.csv", str(tmp_path))
        assert result is None

    def test_underscore_to_dash(self, tmp_path):
        """Manifest named with dashes when CSV uses underscores."""
        manifest = tmp_path / "jetson-j1.json"
        manifest.write_text("{}")
        result = check_manifest_exists("jetson_j1.csv", str(tmp_path))
        assert result is not None


# ---------------------------------------------------------------------------
# Issue
# ---------------------------------------------------------------------------


class TestIssue:
    def test_str_format(self):
        issue = Issue("ERROR", "test.csv", "something broke")
        s = str(issue)
        assert "ERROR" in s
        assert "test.csv" in s
        assert "something broke" in s


# ---------------------------------------------------------------------------
# End-to-end via main()
# ---------------------------------------------------------------------------


class TestMainEndToEnd:
    def _run_main(self, csv_dir, manifest_dir):
        """Run validate_results.main() with patched sys.argv, return exit code."""
        import scripts.validate_results as vr

        orig_argv = sys.argv
        sys.argv = [
            "validate_results.py",
            "--csv-dir",
            str(csv_dir),
            "--manifest-dir",
            str(manifest_dir),
            "--quiet",
        ]
        try:
            return vr.main()
        finally:
            sys.argv = orig_argv

    def test_clean_csv_exit_zero(self, tmp_path):
        """A valid CSV with a manifest exits 0."""
        csv_dir = tmp_path / "raw"
        man_dir = tmp_path / "manifests"
        csv_dir.mkdir()
        man_dir.mkdir()
        _write_std_csv(csv_dir / "jetson-j1.csv", [_std_row()])
        # Provide a manifest so there are no warnings
        (man_dir / "jetson-j1.json").write_text('{"git": {"sha": "abc", "dirty": false}}')
        assert self._run_main(csv_dir, man_dir) == 0

    def test_csv_with_error_exit_two(self, tmp_path):
        """A CSV with an invalid row (p95 < p50) exits 2."""
        csv_dir = tmp_path / "raw"
        man_dir = tmp_path / "manifests"
        csv_dir.mkdir()
        man_dir.mkdir()
        _write_std_csv(
            csv_dir / "jetson-j1.csv",
            [_std_row(p50_us="1000", p95_us="500")],
        )
        assert self._run_main(csv_dir, man_dir) == 2

    def test_csv_with_warning_exit_one(self, tmp_path):
        """A CSV with only warnings (no errors) exits 1."""
        csv_dir = tmp_path / "raw"
        man_dir = tmp_path / "manifests"
        csv_dir.mkdir()
        man_dir.mkdir()
        # No manifest → WARNING; valid row → no error
        _write_std_csv(csv_dir / "jetson-j1.csv", [_std_row()])
        # With --quiet, the warning about missing manifest should trigger exit 1
        result = self._run_main(csv_dir, man_dir)
        assert result == 1  # missing manifest is a WARNING

    def test_empty_csv_dir_exit_zero(self, tmp_path):
        """An empty CSV directory exits 0."""
        csv_dir = tmp_path / "raw"
        man_dir = tmp_path / "manifests"
        csv_dir.mkdir()
        man_dir.mkdir()
        assert self._run_main(csv_dir, man_dir) == 0

    def test_missing_csv_dir_exit_two(self, tmp_path):
        """A non-existent CSV directory exits 2."""
        result = self._run_main(tmp_path / "nonexistent", tmp_path / "manifests")
        assert result == 2

    def test_recursive_subdirectory_discovery(self, tmp_path):
        """CSVs in subdirectories are discovered and validated."""
        csv_dir = tmp_path / "raw"
        man_dir = tmp_path / "manifests"
        kleidiai_dir = csv_dir / "kleidiai"
        kleidiai_dir.mkdir(parents=True)
        man_dir.mkdir()

        # Write a kleidiai matmul CSV in a subdirectory
        header = ",".join(KLEIDIAI_MATMUL_COLS)
        row = "decode_1x128x128,kleidiai,1,128,128,1.855,33.41,17.66"
        (kleidiai_dir / "rk3588-t3_kleidiai_matmul.csv").write_text(f"{header}\n{row}\n")

        import scripts.validate_results as vr

        orig_argv = sys.argv
        sys.argv = [
            "validate_results.py",
            "--csv-dir",
            str(csv_dir),
            "--manifest-dir",
            str(man_dir),
        ]
        try:
            rc = vr.main()
        finally:
            sys.argv = orig_argv

        # Should discover the subdirectory CSV (exit 1 = warnings, no errors)
        assert rc >= 0  # no crash


class TestValidateCsv:
    def test_file_not_found(self):
        issues = []
        result = validate_csv("/nonexistent/file.csv", "file.csv", issues)
        assert result == (None, 0, None)
        assert any("not found" in i.message for i in issues)

    def test_empty_csv(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("")
        issues = []
        result = validate_csv(str(path), "empty.csv", issues)
        assert result == (None, 0, None)
        assert any(
            "empty" in i.message.lower() or "unreadable" in i.message.lower() for i in issues
        )

    def test_unrecognized_format(self, tmp_path):
        path = tmp_path / "unknown.csv"
        path.write_text("foo,bar,baz\n1,2,3\n")
        issues = []
        result = validate_csv(str(path), "unknown.csv", issues)
        assert result == (None, 0, None)
        assert any("unrecognized" in i.message for i in issues)

    def test_standard_csv_validated(self, tmp_path):
        header = ",".join(STANDARD_COLS)
        row = "Qwen3.5-4B,gdn_cumdecay,neon,64,4096,30,100,120,20,1.5,0.3"
        path = tmp_path / "standard.csv"
        path.write_text(f"{header}\n{row}\n")
        issues = []
        csv_type, row_count, _ = validate_csv(str(path), "standard.csv", issues)
        assert csv_type == "standard"
        assert row_count == 1

    def test_sustained_csv_validated(self, tmp_path):

        header = ",".join(SUSTAINED_COLS)
        row = "Qwen3.5-4B,gdn_gated_scan,neon,10.0,2.5,55.0,-5.0"
        path = tmp_path / "sustained.csv"
        path.write_text(f"{header}\n{row}\n")
        issues = []
        csv_type, row_count, _ = validate_csv(str(path), "sustained.csv", issues)
        assert csv_type == "sustained"
        assert row_count == 1

    def test_schema_csv_extracts_manifest_ref(self, tmp_path):
        header = ",".join(E2E_SWEEP_COLS)
        row = (
            "run1,2026-01-01,a1b2c3d,manifests/run1.json,rk3588,cpu,cpu,"
            "Qwen3.5-4B,fp16,4096,prefill,prefill_tokens_per_sec,,800.0,"
            "tokens_per_sec,0,5,all,"
        )
        path = tmp_path / "schema.csv"
        path.write_text(f"{header}\n{row}\n")
        issues = []
        csv_type, row_count, schema_ref = validate_csv(str(path), "schema.csv", issues)
        assert csv_type == "e2e_sweep"
        assert schema_ref == "manifests/run1.json"

    def test_profile_csv_validated(self, tmp_path):
        header = "phase,ctx_len,layer_idx,layer_type,p50_us,p95_us,mean_us,n_samples"
        row = "prefill,64,0,linear_attention,100.0,120.0,110.0,3"
        path = tmp_path / "profile.csv"
        path.write_text(f"{header}\n{row}\n")
        issues = []
        csv_type, row_count, _ = validate_csv(str(path), "profile.csv", issues)
        assert csv_type == "layer_profile"
        assert row_count == 1

    def test_comment_prefixed_standard_csv(self, tmp_path):
        """A CSV with a '# metadata' comment line before the header."""
        comment = "# config=big_only_a76 binary=bench_gdn_a76 affinity=4-7\n"
        header = ",".join(STANDARD_COLS)
        row = "Qwen3.5-4B,gdn_cumdecay,neon,64,4096,30,100,120,20,1.5,0.3"
        path = tmp_path / "affinity.csv"
        path.write_text(f"{comment}{header}\n{row}\n")
        issues = []
        csv_type, row_count, _ = validate_csv(str(path), "affinity.csv", issues)
        assert csv_type == "standard"
        assert row_count == 1
        assert not any(i.severity == "ERROR" for i in issues)

    def test_commented_out_header_stripped(self, tmp_path):
        """A CSV where the header itself is prefixed with '#'.

        This happens in delta_matmul_study.csv: the header line is
        '# config,binary,affinity,kernel,...' and should be uncommented.
        """
        lines = [
            "# delta_matmul_affinity_study (ob-8qt.1)\n",
            "# device: RK3588\n",
            "# config,binary,affinity,kernel,M,K,N,repeats,p50_us,p95_us,gib_per_s_p50\n",
            "big_only_a76,a76,4-7,gdn_delta_rule_matmul,1,128,128,30,3.500,3.501,17.71\n",
        ]
        path = tmp_path / "study.csv"
        path.write_text("".join(lines))
        issues = []
        csv_type, row_count, _ = validate_csv(str(path), "study.csv", issues)
        assert csv_type == "delta_matmul"
        assert row_count == 1

    def test_metadata_comment_with_commas_skipped(self, tmp_path):
        """A '#'-prefixed line with commas but prose (not identifiers) is skipped."""
        lines = [
            "# config,binary,affinity,kernel,M,K,N,repeats,p50_us,p95_us,gib_per_s_p50\n",
            "# (A76 on big cores, A55 on little cores — launched simultaneously)\n",
            "big_only_a76,a76,4-7,gdn_delta_rule_matmul,1,128,128,30,3.500,3.501,17.71\n",
        ]
        path = tmp_path / "study2.csv"
        path.write_text("".join(lines))
        issues = []
        csv_type, row_count, _ = validate_csv(str(path), "study2.csv", issues)
        assert csv_type == "delta_matmul"
        assert row_count == 1
        assert not any("cannot parse" in i.message for i in issues)

    def test_multi_section_csv_dedup_header(self, tmp_path):
        """A CSV with repeated header lines between sections (affinity study)."""
        header = ",".join(STANDARD_COLS)
        row1 = "Qwen3.5-4B,gdn_cumdecay,neon,64,4096,30,100,120,20,1.5,0.3"
        row2 = "Qwen3.5-4B,gdn_gated_scan,neon,64,4096,30,500,600,20,1.0,0.5"
        comment = "# config=all_cores_a76 binary=bench_gdn_a76 affinity=all\n"
        content = f"{header}\n{row1}\n{comment}{header}\n{row2}\n"
        path = tmp_path / "multi.csv"
        path.write_text(content)
        issues = []
        csv_type, row_count, _ = validate_csv(str(path), "multi.csv", issues)
        assert csv_type == "standard"
        assert row_count == 2

    def test_valid_json(self, tmp_path):
        path = tmp_path / "manifest.json"
        path.write_text('{"git": {"sha": "abc123"}}')
        result = load_manifest(str(path))
        assert result["git"]["sha"] == "abc123"

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json}")
        assert load_manifest(str(path)) is None


class TestValidateManifestExtra:
    def test_manifest_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{broken}")
        issues = []
        validate_manifest("test.csv", "standard", 6, str(path), issues, "abc123")
        assert any("invalid JSON" in i.message for i in issues)


# ---------------------------------------------------------------------------
# get_git_head_sha — exception handler (lines 174-175)
# ---------------------------------------------------------------------------


class TestGetGitHeadSha:
    def test_returns_none_on_subprocess_error(self):
        """When git rev-parse fails, returns None."""
        import scripts.validate_results as vr

        with (
            __import__("unittest.mock").mock.patch.object(
                vr.subprocess,
                "check_output",
                side_effect=FileNotFoundError("no git"),
            ),
        ):
            assert get_git_head_sha() is None

    def test_returns_sha_on_success(self):
        """When git rev-parse succeeds, returns stripped sha."""
        import scripts.validate_results as vr

        with __import__("unittest.mock").mock.patch.object(
            vr.subprocess,
            "check_output",
            return_value=b"abc123def456\n",
        ):
            assert get_git_head_sha() == "abc123def456"


# ---------------------------------------------------------------------------
# validate_sustained_row — parse error path (lines 263-265)
# ---------------------------------------------------------------------------


class TestValidateSustainedRowParseError:
    def test_non_numeric_elapsed(self):
        issues = []
        validate_sustained_row(
            {**_sustained_row(), "elapsed_s": "not_a_number"},
            "test.csv",
            issues,
            2,
        )
        assert any("cannot parse" in i.message for i in issues)
        assert issues[-1].severity == "ERROR"

    def test_missing_key(self):
        issues = []
        row = _sustained_row()
        del row["throughput_gibs"]
        validate_sustained_row(row, "test.csv", issues, 1)
        assert any("cannot parse" in i.message for i in issues)


# ---------------------------------------------------------------------------
# validate_csv — missing columns + exception handler
# ---------------------------------------------------------------------------


class TestValidateCsvMissingColumns:
    def test_e2e_sweep_missing_non_key_columns(self, tmp_path):
        """E2E sweep CSV with detection cols but missing some full cols → ERROR."""
        # detect_csv_type identifies e2e_sweep by these 4 cols, but E2E_SWEEP_COLS has 19
        detection_cols = {"run_id", "metric_name", "metric_component", "repeat_index"}
        partial = sorted(detection_cols)
        path = tmp_path / "partial_sweep.csv"
        path.write_text(",".join(partial) + "\n" + ",".join(["x"] * len(partial)) + "\n")
        issues = []
        csv_type, row_count, _ = validate_csv(str(path), "partial_sweep.csv", issues)
        assert csv_type == "e2e_sweep"
        assert any("missing required columns" in i.message for i in issues)


class TestValidateCsvException:
    def test_read_error_returns_none(self, tmp_path):
        """If CSV reader raises, returns (None, 0, None) with ERROR."""
        from unittest.mock import patch

        path = tmp_path / "bad.csv"
        path.write_text(",".join(STANDARD_COLS) + "\n")
        issues = []
        with patch("csv.DictReader", side_effect=OSError("io error")):
            result = validate_csv(str(path), "bad.csv", issues)
        assert result == (None, 0, None)
        assert any("cannot read CSV" in i.message for i in issues)


# ---------------------------------------------------------------------------
# main() — additional edge cases
# ---------------------------------------------------------------------------


class TestMainExtras:
    def _run_main(self, csv_dir, manifest_dir, quiet=True):
        """Run main() with patched sys.argv, return exit code."""
        argv = [
            "validate_results.py",
            "--csv-dir",
            str(csv_dir),
            "--manifest-dir",
            str(manifest_dir),
        ]
        if quiet:
            argv.append("--quiet")
        orig_argv = sys.argv
        sys.argv = argv
        try:
            return main()
        finally:
            sys.argv = orig_argv

    def test_quiet_empty_dir_no_output(self, tmp_path, capsys):
        """Empty CSV dir + --quiet exits 0 with no stdout."""
        csv_dir = tmp_path / "raw"
        csv_dir.mkdir()
        man_dir = tmp_path / "manifests"
        man_dir.mkdir()
        rc = self._run_main(csv_dir, man_dir, quiet=True)
        assert rc == 0
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_non_quiet_reports_errors_and_warnings(self, tmp_path, capsys):
        """Non-quiet mode prints errors, warnings, and notes."""
        csv_dir = tmp_path / "raw"
        csv_dir.mkdir()
        man_dir = tmp_path / "manifests"
        man_dir.mkdir()
        # CSV with an error (p95 < p50) and no manifest → warning too
        _write_std_csv(csv_dir / "jetson-j1.csv", [_std_row(p50_us="1000", p95_us="500")])
        self._run_main(csv_dir, man_dir, quiet=False)
        captured = capsys.readouterr()
        assert "error(s)" in captured.out
        assert "issue(s) found" in captured.out

    def test_non_quiet_reports_warnings_only(self, tmp_path, capsys):
        """Non-quiet mode with only warnings shows warning count and exits 1."""
        csv_dir = tmp_path / "raw"
        csv_dir.mkdir()
        man_dir = tmp_path / "manifests"
        man_dir.mkdir()
        # Valid row but no manifest → WARNING only
        _write_std_csv(csv_dir / "jetson-j1.csv", [_std_row()])
        rc = self._run_main(csv_dir, man_dir, quiet=False)
        assert rc == 1
        captured = capsys.readouterr()
        assert "warning(s)" in captured.out

    def test_non_quiet_reports_notes(self, tmp_path, capsys):
        """Non-quiet mode prints informational notes."""
        csv_dir = tmp_path / "raw"
        csv_dir.mkdir()
        man_dir = tmp_path / "manifests"
        man_dir.mkdir()
        # A valid CSV with manifest → no errors/warnings, but may have notes
        _write_std_csv(csv_dir / "jetson-j1.csv", [_std_row()])
        (man_dir / "jetson-j1.json").write_text('{"git": {"sha": "abc", "dirty": false}}')
        rc = self._run_main(csv_dir, man_dir, quiet=False)
        assert rc == 0
        captured = capsys.readouterr()
        assert "CSV(s) checked" in captured.out

    def test_schema_csv_manifest_ref_relative_path(self, tmp_path):
        """Schema CSV with relative manifest_ref resolves to manifest_dir."""
        csv_dir = tmp_path / "raw"
        csv_dir.mkdir()
        man_dir = tmp_path / "manifests"
        man_dir.mkdir()
        # Write schema CSV with a manifest_ref that doesn't exist as a file
        header = ",".join(E2E_SWEEP_COLS)
        row = (
            "run1,2026-01-01,a1b2c3d,manifests/run1.json,rk3588,cpu,cpu,"
            "Qwen3.5-4B,fp16,4096,prefill,prefill_tokens_per_sec,,800.0,"
            "tokens_per_sec,0,5,all,"
        )
        (csv_dir / "schema.csv").write_text(f"{header}\n{row}\n")
        # Run and check the exit code is non-zero (no manifest found)
        rc = self._run_main(csv_dir, man_dir, quiet=True)
        # Should exit with at least 1 (warning about missing manifest)
        assert rc >= 1

    def test_schema_csv_manifest_ref_absolute_path(self, tmp_path):
        """Schema CSV with absolute manifest_ref uses it directly."""
        csv_dir = tmp_path / "raw"
        csv_dir.mkdir()
        man_dir = tmp_path / "manifests"
        man_dir.mkdir()
        # Create the manifest at an absolute path
        man_file = tmp_path / "custom_manifest.json"
        man_file.write_text('{"git": {"sha": "abc1234", "dirty": false}}')
        header = ",".join(E2E_SWEEP_COLS)
        row = (
            f"run1,2026-01-01,a1b2c3d,{man_file},rk3588,cpu,cpu,"
            "Qwen3.5-4B,fp16,4096,prefill,prefill_tokens_per_sec,,800.0,"
            "tokens_per_sec,0,5,all,"
        )
        (csv_dir / "schema.csv").write_text(f"{header}\n{row}\n")
        rc = self._run_main(csv_dir, man_dir, quiet=True)
        assert rc == 0


# ---------------------------------------------------------------------------
# check_readme_counts
# ---------------------------------------------------------------------------


class TestCheckReadmeCounts:
    """Unit tests for the README 'Results so far' cross-check."""

    @staticmethod
    def _make_repo(tmp_path, n_csv=2, n_manifest=1, n_figures=1, n_findings=1):
        """Create a minimal repo structure with matching README counts."""
        root = tmp_path / "repo"
        raw = root / "results" / "raw"
        man = root / "results" / "manifests"
        fig = root / "results" / "figures"
        docs = root / "docs"
        raw.mkdir(parents=True)
        man.mkdir(parents=True)
        fig.mkdir(parents=True)
        docs.mkdir(parents=True)

        for i in range(n_csv):
            (raw / f"bench_{i}.csv").write_text("data")
        for i in range(n_manifest):
            (man / f"device_{i}.json").write_text("{}")
        for i in range(n_figures):
            (fig / f"plot_{i}.png").write_text("png")
        # Always create figures/README.md (the index file that should be excluded)
        (fig / "README.md").write_text("# Index")
        # FINDINGS.md with n_findings ## headers
        findings_lines = ["# FINDINGS\n"]
        for i in range(n_findings):
            findings_lines.append(f"## {i + 1}. Finding number {i + 1}\nbody\n")
        (docs / "FINDINGS.md").write_text("".join(findings_lines))

        # README.md with the "Results so far" line
        readme = (
            f"> **Results so far:** {n_csv} CSVs from the device fleet, "
            f"{n_manifest} provenance manifests, {n_figures} generated "
            f"figures/tables, {n_findings} FINDINGS sections.\n"
        )
        (root / "README.md").write_text(readme)
        return root

    def test_matching_counts_no_issues(self, tmp_path):
        """Correct counts produce no issues."""
        root = self._make_repo(tmp_path, n_csv=3, n_manifest=2, n_figures=4, n_findings=5)
        issues = []
        check_readme_counts(issues, repo_root=str(root))
        assert issues == []

    def test_wrong_csv_count(self, tmp_path):
        """Stale CSV count is flagged."""
        root = self._make_repo(tmp_path, n_csv=3)
        # Tamper with README to claim wrong count
        readme = root / "README.md"
        readme.write_text(
            "> **Results so far:** 99 CSVs from the device fleet, "
            "1 provenance manifests, 1 generated figures/tables, 1 FINDINGS sections.\n"
        )
        issues = []
        check_readme_counts(issues, repo_root=str(root))
        assert len(issues) == 1
        assert "CSVs" in issues[0].message

    def test_wrong_manifest_count(self, tmp_path):
        """Stale manifest count is flagged."""
        root = self._make_repo(tmp_path, n_manifest=5)
        readme = root / "README.md"
        readme.write_text(
            "> **Results so far:** 2 CSVs from the device fleet, "
            "99 provenance manifests, 1 generated figures/tables, 1 FINDINGS sections.\n"
        )
        issues = []
        check_readme_counts(issues, repo_root=str(root))
        msgs = [i.message for i in issues]
        assert any("manifests" in m for m in msgs)

    def test_wrong_figures_count(self, tmp_path):
        """Stale figures count is flagged."""
        root = self._make_repo(tmp_path, n_figures=3)
        readme = root / "README.md"
        readme.write_text(
            "> **Results so far:** 2 CSVs from the device fleet, "
            "1 provenance manifests, 99 generated figures/tables, 1 FINDINGS sections.\n"
        )
        issues = []
        check_readme_counts(issues, repo_root=str(root))
        msgs = [i.message for i in issues]
        assert any("figures" in m for m in msgs)

    def test_figures_excludes_readme_md(self, tmp_path):
        """figures/README.md is NOT counted as a figure."""
        root = self._make_repo(tmp_path, n_figures=2)
        # README says 2 (matching the 2 actual figures, NOT counting README.md)
        issues = []
        check_readme_counts(issues, repo_root=str(root))
        assert issues == []

    def test_wrong_findings_count(self, tmp_path):
        """Stale FINDINGS count is flagged."""
        root = self._make_repo(tmp_path, n_findings=7)
        readme = root / "README.md"
        readme.write_text(
            "> **Results so far:** 2 CSVs from the device fleet, "
            "1 provenance manifests, 1 generated figures/tables, 99 FINDINGS sections.\n"
        )
        issues = []
        check_readme_counts(issues, repo_root=str(root))
        msgs = [i.message for i in issues]
        assert any("FINDINGS" in m for m in msgs)

    def test_findings_counts_all_headers(self, tmp_path):
        """FINDINGS count includes named (non-numbered) ## headers."""
        root = self._make_repo(tmp_path, n_findings=0)
        # Add named sections (not starting with a digit)
        findings_path = root / "docs" / "FINDINGS.md"
        findings_path.write_text(
            "# FINDINGS\n"
            "## 1. First finding\nbody\n"
            "## Named section without number\nbody\n"
            "## 2a. Sub-section\nbody\n"
        )
        # README should claim 3 (all ## headers), not 2 (only numbered)
        readme = root / "README.md"
        readme.write_text(
            "> **Results so far:** 2 CSVs from the device fleet, "
            "1 provenance manifests, 1 generated figures/tables, 3 FINDINGS sections.\n"
        )
        issues = []
        check_readme_counts(issues, repo_root=str(root))
        assert issues == []  # 3 == 3, correct

    def test_no_readme_no_crash(self, tmp_path):
        """Missing README.md does not crash."""
        root = tmp_path / "norepo"
        root.mkdir()
        issues = []
        check_readme_counts(issues, repo_root=str(root))
        assert issues == []


# ---------------------------------------------------------------------------
# validate_delta_matmul_row
# ---------------------------------------------------------------------------


def _delta_matmul_row(**overrides):
    """A valid delta-rule matmul row dict."""
    base = {
        "kernel": "delta_matmul",
        "M": "128",
        "K": "256",
        "N": "512",
        "repeats": "30",
        "p50_us": "100.0",
        "p95_us": "120.0",
        "gib_per_s_p50": "5.2",
    }
    base.update(overrides)
    return base


class TestValidateDeltaMatmulRow:
    def test_valid_row_no_issues(self):
        issues = []
        validate_delta_matmul_row(_delta_matmul_row(), "test.csv", issues, 2)
        assert issues == []

    def test_non_positive_dimension(self):
        issues = []
        validate_delta_matmul_row(_delta_matmul_row(M="0", K="256", N="512"), "test.csv", issues, 2)
        assert any("non-positive matmul dim" in i.message for i in issues)

    def test_non_positive_p50(self):
        issues = []
        validate_delta_matmul_row(_delta_matmul_row(p50_us="0.0"), "test.csv", issues, 2)
        assert any("non-positive p50_us" in i.message for i in issues)

    def test_p95_less_than_p50(self):
        issues = []
        validate_delta_matmul_row(
            _delta_matmul_row(p50_us="200.0", p95_us="100.0"), "test.csv", issues, 2
        )
        assert any("p95" in i.message and i.severity == "WARNING" for i in issues)

    def test_malformed_value(self):
        issues = []
        validate_delta_matmul_row(_delta_matmul_row(p50_us="abc"), "test.csv", issues, 2)
        assert any("cannot parse" in i.message for i in issues)


# ---------------------------------------------------------------------------
# validate_ctx_sweep_row
# ---------------------------------------------------------------------------


def _ctx_sweep_row(**overrides):
    """A valid context-length sweep row dict (gdn_e2e_decode.c --ctx-sweep)."""
    base = {
        "model": "qwen35_4b",
        "ctx_len": "4096",
        "gdn_layer_us": "500.0",
        "full_attn_us": "800.0",
        "ffn_us": "300.0",
        "total_us": "1600.0",
        "tok_per_sec": "625.0",
        "kv_cache_mb": "512.0",
    }
    base.update(overrides)
    return base


class TestValidateCtxSweepRow:
    def test_valid_row_no_issues(self):
        issues = []
        validate_ctx_sweep_row(_ctx_sweep_row(), "test.csv", issues, 2)
        assert issues == []

    def test_non_positive_ctx_len(self):
        issues = []
        validate_ctx_sweep_row(_ctx_sweep_row(ctx_len="0"), "test.csv", issues, 2)
        assert any("non-positive ctx_len" in i.message for i in issues)

    def test_non_positive_gdn_layer_us(self):
        issues = []
        validate_ctx_sweep_row(_ctx_sweep_row(gdn_layer_us="0.0"), "test.csv", issues, 2)
        assert any("non-positive gdn_layer_us" in i.message for i in issues)

    def test_non_positive_ffn_us(self):
        issues = []
        validate_ctx_sweep_row(_ctx_sweep_row(ffn_us="0.0"), "test.csv", issues, 2)
        assert any("non-positive ffn_us" in i.message for i in issues)

    def test_non_positive_total_us(self):
        issues = []
        validate_ctx_sweep_row(_ctx_sweep_row(total_us="0.0"), "test.csv", issues, 2)
        assert any("non-positive total_us" in i.message for i in issues)

    def test_zero_full_attn_us_is_ok(self):
        """full_attn_us=0 is valid for --pure-gdn sweeps (no full-attention layers)."""
        issues = []
        validate_ctx_sweep_row(_ctx_sweep_row(full_attn_us="0.0"), "test.csv", issues, 2)
        assert issues == []

    def test_negative_full_attn_us(self):
        issues = []
        validate_ctx_sweep_row(_ctx_sweep_row(full_attn_us="-1.0"), "test.csv", issues, 2)
        assert any("negative full_attn_us" in i.message for i in issues)

    def test_non_positive_tok_per_sec(self):
        issues = []
        validate_ctx_sweep_row(_ctx_sweep_row(tok_per_sec="0.0"), "test.csv", issues, 2)
        assert any("non-positive tok_per_sec" in i.message for i in issues)

    def test_negative_kv_cache_mb(self):
        issues = []
        validate_ctx_sweep_row(_ctx_sweep_row(kv_cache_mb="-1.0"), "test.csv", issues, 2)
        assert any("negative kv_cache_mb" in i.message for i in issues)

    def test_malformed_value(self):
        issues = []
        validate_ctx_sweep_row(_ctx_sweep_row(gdn_layer_us="not_a_number"), "test.csv", issues, 2)
        assert any("cannot parse" in i.message for i in issues)


# ---------------------------------------------------------------------------
# check_ablation_manifests
# ---------------------------------------------------------------------------


class TestCheckAblationManifests:
    """Tests for ablation CSV manifest_ref validation."""

    def test_valid_manifest_ref(self, tmp_path):
        """An existing manifest_ref file should produce a NOTE, not a WARNING."""
        ablation_dir = tmp_path / "ablation"
        ablation_dir.mkdir()
        manifest = tmp_path / "manifest.json"
        manifest.write_text("{}")

        csv_path = ablation_dir / "ablation_test.csv"
        csv_path.write_text(f"metric_name,manifest_ref\nfoo,{manifest}\n")

        issues = []
        check_ablation_manifests(str(ablation_dir), issues)
        assert any("manifest_ref OK" in i.message and i.severity == "NOTE" for i in issues)
        assert not any(i.severity == "WARNING" for i in issues)

    def test_missing_manifest_ref(self, tmp_path):
        """A manifest_ref pointing to a non-existent file should produce a WARNING."""
        ablation_dir = tmp_path / "ablation"
        ablation_dir.mkdir()

        csv_path = ablation_dir / "ablation_test.csv"
        csv_path.write_text("metric_name,manifest_ref\nfoo,/nonexistent/manifest.json\n")

        issues = []
        check_ablation_manifests(str(ablation_dir), issues)
        assert any("MISSING" in i.message and i.severity == "WARNING" for i in issues)

    def test_no_manifest_ref_column(self, tmp_path):
        """A CSV without manifest_ref column should produce no issues."""
        ablation_dir = tmp_path / "ablation"
        ablation_dir.mkdir()

        csv_path = ablation_dir / "ablation_test.csv"
        csv_path.write_text("metric_name,value\nfoo,42\n")

        issues = []
        check_ablation_manifests(str(ablation_dir), issues)
        assert issues == []

    def test_nonexistent_dir(self):
        """A non-existent directory should return without error."""
        issues = []
        check_ablation_manifests("/nonexistent/path/xyz", issues)
        assert issues == []

    def test_empty_manifest_ref_skipped(self, tmp_path):
        """Empty manifest_ref values should be skipped (no issue)."""
        ablation_dir = tmp_path / "ablation"
        ablation_dir.mkdir()

        csv_path = ablation_dir / "ablation_test.csv"
        csv_path.write_text("metric_name,manifest_ref\nfoo,\n")

        issues = []
        check_ablation_manifests(str(ablation_dir), issues)
        assert issues == []


# ---------------------------------------------------------------------------
# Additional gpu_micro_row edge cases (ob-8qt.24)
# ---------------------------------------------------------------------------


class TestValidateGpuMicroRowEdgeCases:
    """Cover remaining branches in validate_gpu_micro_row."""

    def test_missing_dim_keys(self):
        """Missing dim1/dim2 keys should produce a parse error and return early."""
        issues = []
        row = _gpu_row()
        del row["dim1"]
        del row["dim2"]
        validate_gpu_micro_row(row, "test.csv", issues, 2)
        assert any("cannot parse dims" in i.message and i.severity == "ERROR" for i in issues)
        # Should have returned early — no p50/bw checks
        assert not any("p50" in i.message for i in issues)
        assert not any("bw_mibs" in i.message for i in issues)

    def test_non_integer_dim1(self):
        """Non-integer dim1 should trigger parse error."""
        issues = []
        validate_gpu_micro_row(_gpu_row(dim1="abc"), "test.csv", issues, 2)
        assert any("cannot parse dims" in i.message for i in issues)

    def test_non_integer_dim2(self):
        """Non-integer dim2 should trigger parse error."""
        issues = []
        validate_gpu_micro_row(_gpu_row(dim2="xyz"), "test.csv", issues, 2)
        assert any("cannot parse dims" in i.message for i in issues)

    def test_non_positive_dim3(self):
        """dim3=0 should produce a non-positive dim3 error."""
        issues = []
        validate_gpu_micro_row(_gpu_row(dim3="0"), "test.csv", issues, 2)
        assert any("non-positive dim3" in i.message and i.severity == "ERROR" for i in issues)

    def test_non_integer_dim3(self):
        """Non-integer dim3 should produce a warning."""
        issues = []
        validate_gpu_micro_row(_gpu_row(dim3="abc"), "test.csv", issues, 2)
        assert any("non-integer dim3" in i.message and i.severity == "WARNING" for i in issues)

    def test_non_positive_p50(self):
        """p50_ms=0 should produce a non-positive p50_ms error."""
        issues = []
        validate_gpu_micro_row(_gpu_row(p50_ms="0.0"), "test.csv", issues, 2)
        assert any("non-positive p50_ms" in i.message and i.severity == "ERROR" for i in issues)

    def test_non_numeric_p95(self):
        """Non-numeric p95_ms should produce a warning."""
        issues = []
        validate_gpu_micro_row(_gpu_row(p95_ms="abc"), "test.csv", issues, 2)
        assert any("non-numeric p95_ms" in i.message and i.severity == "WARNING" for i in issues)

    def test_cannot_parse_bw(self):
        """Non-numeric bw_mibs should produce an error."""
        issues = []
        validate_gpu_micro_row(_gpu_row(bw_mibs="N/A"), "test.csv", issues, 2)
        assert any("cannot parse bw_mibs" in i.message and i.severity == "ERROR" for i in issues)

    def test_missing_bw_key(self):
        """Missing bw_mibs key should produce an error."""
        issues = []
        row = _gpu_row()
        del row["bw_mibs"]
        validate_gpu_micro_row(row, "test.csv", issues, 2)
        assert any("cannot parse bw_mibs" in i.message for i in issues)

    def test_missing_p50_key_returns_early(self):
        """Missing p50_ms key should return early (no p95 or bw checks)."""
        issues = []
        row = _gpu_row()
        del row["p50_ms"]
        validate_gpu_micro_row(row, "test.csv", issues, 2)
        assert any("cannot parse p50_ms" in i.message for i in issues)
        # Should have returned — no bw_mibs check
        assert not any("bw_mibs" in i.message for i in issues)


# ---------------------------------------------------------------------------
# Additional kleidiai_matmul_row edge cases (ob-8qt.24)
# ---------------------------------------------------------------------------


class TestValidateKleidiaiMatmulRowEdgeCases:
    """Cover remaining branches in validate_kleidiai_matmul_row."""

    def test_malformed_gflops(self):
        """Non-numeric GFLOP_s should produce a warning."""
        issues = []
        validate_kleidiai_matmul_row(_kleidiai_row(GFLOP_s="N/A"), "test.csv", issues, 2)
        assert any("cannot parse GFLOP_s" in i.message and i.severity == "WARNING" for i in issues)

    def test_missing_gflops_key(self):
        """Missing GFLOP_s key should produce a warning."""
        issues = []
        row = _kleidiai_row()
        del row["GFLOP_s"]
        validate_kleidiai_matmul_row(row, "test.csv", issues, 2)
        assert any("cannot parse GFLOP_s" in i.message for i in issues)

    def test_malformed_gibs_missing_gflops(self):
        """When both GiB_s and GFLOP_s are missing, both warnings should fire."""
        issues = []
        row = _kleidiai_row()
        del row["GiB_s"]
        del row["GFLOP_s"]
        validate_kleidiai_matmul_row(row, "test.csv", issues, 2)
        assert any("cannot parse GiB_s" in i.message for i in issues)
        assert any("cannot parse GFLOP_s" in i.message for i in issues)


# ---------------------------------------------------------------------------
# Additional kleidiai_gdn_kernel_row edge cases (ob-8qt.24)
# ---------------------------------------------------------------------------


class TestValidateKleidiaiGdnKernelRowEdgeCases:
    """Cover remaining branches in validate_kleidiai_gdn_kernel_row."""

    def test_malformed_gibs(self):
        """Non-numeric gib_per_s_p50 should produce a warning."""
        issues = []
        validate_kleidiai_gdn_kernel_row(
            _gdn_kernel_row(gib_per_s_p50="N/A"), "test.csv", issues, 2
        )
        assert any(
            "cannot parse gib_per_s_p50" in i.message and i.severity == "WARNING" for i in issues
        )

    def test_missing_gibs_key(self):
        """Missing gib_per_s_p50 key should produce a warning."""
        issues = []
        row = _gdn_kernel_row()
        del row["gib_per_s_p50"]
        validate_kleidiai_gdn_kernel_row(row, "test.csv", issues, 2)
        assert any("cannot parse gib_per_s_p50" in i.message for i in issues)


# ---------------------------------------------------------------------------
# Additional layer_profile_row edge cases (ob-8qt.24)
# ---------------------------------------------------------------------------


class TestValidateLayerProfileRowEdgeCases:
    """Cover remaining branches: non-positive mean_us, ctx_len, layer_idx."""

    def test_non_positive_mean_us(self):
        issues = []
        validate_layer_profile_row(_layer_profile_row(mean_us="0.0"), "test.csv", issues, 2)
        assert any("non-positive mean_us" in i.message and i.severity == "ERROR" for i in issues)

    def test_non_positive_ctx_len(self):
        issues = []
        validate_layer_profile_row(_layer_profile_row(ctx_len="0"), "test.csv", issues, 2)
        assert any("non-positive ctx_len" in i.message and i.severity == "ERROR" for i in issues)

    def test_negative_layer_idx(self):
        issues = []
        validate_layer_profile_row(_layer_profile_row(layer_idx="-1"), "test.csv", issues, 2)
        assert any("negative layer_idx" in i.message and i.severity == "ERROR" for i in issues)


# ---------------------------------------------------------------------------
# CSV type dispatch inline sanity checks (ob-8qt.24)
# ---------------------------------------------------------------------------


def _write_csv(path, header, row_data):
    """Write a single-row CSV file."""
    path.write_text(f"{header}\n{row_data}\n")


class TestValidateCsvE2eDecode:
    """Test the inline tok_per_sec_mean sanity check for e2e_decode CSVs."""

    def test_implausible_tok_per_sec_warns(self, tmp_path):
        csv_path = tmp_path / "test_e2e_decode.csv"
        header = "tok_per_sec_mean,gdn_proj_pct,ffn_pct"
        _write_csv(csv_path, header, "2000,50,50")
        issues = []
        validate_csv(str(csv_path), "test_e2e_decode.csv", issues)
        assert any("implausible tok/s" in i.message for i in issues)

    def test_negative_tok_per_sec_warns(self, tmp_path):
        csv_path = tmp_path / "test_e2e_decode.csv"
        header = "tok_per_sec_mean,gdn_proj_pct,ffn_pct"
        _write_csv(csv_path, header, "-5,50,50")
        issues = []
        validate_csv(str(csv_path), "test_e2e_decode.csv", issues)
        assert any("implausible tok/s" in i.message for i in issues)

    def test_valid_tok_per_sec_no_warning(self, tmp_path):
        csv_path = tmp_path / "test_e2e_decode.csv"
        header = "tok_per_sec_mean,gdn_proj_pct,ffn_pct"
        _write_csv(csv_path, header, "42.5,50,50")
        issues = []
        validate_csv(str(csv_path), "test_e2e_decode.csv", issues)
        assert not any("implausible tok/s" in i.message for i in issues)

    def test_non_numeric_tok_per_sec_handled(self, tmp_path):
        """Non-numeric tok_per_sec should not crash (ValueError silently caught)."""
        csv_path = tmp_path / "test_e2e_decode.csv"
        header = "tok_per_sec_mean,gdn_proj_pct,ffn_pct"
        _write_csv(csv_path, header, "N/A,50,50")
        issues = []
        # Should not raise
        validate_csv(str(csv_path), "test_e2e_decode.csv", issues)
        assert not any("implausible" in i.message for i in issues)


class TestValidateCsvPrefillGemm:
    """Test the inline tok_per_sec_prefill sanity check."""

    def test_implausible_prefill_tps_warns(self, tmp_path):
        csv_path = tmp_path / "test_prefill.csv"
        header = "prefill_M,ttft_ms,tok_per_sec_prefill"
        _write_csv(csv_path, header, "1,100,5000")
        issues = []
        validate_csv(str(csv_path), "test_prefill.csv", issues)
        assert any("implausible prefill tok/s" in i.message for i in issues)

    def test_negative_prefill_tps_warns(self, tmp_path):
        csv_path = tmp_path / "test_prefill.csv"
        header = "prefill_M,ttft_ms,tok_per_sec_prefill"
        _write_csv(csv_path, header, "1,100,-1")
        issues = []
        validate_csv(str(csv_path), "test_prefill.csv", issues)
        assert any("implausible prefill tok/s" in i.message for i in issues)

    def test_valid_prefill_tps_no_warning(self, tmp_path):
        csv_path = tmp_path / "test_prefill.csv"
        header = "prefill_M,ttft_ms,tok_per_sec_prefill"
        _write_csv(csv_path, header, "1,100,250.5")
        issues = []
        validate_csv(str(csv_path), "test_prefill.csv", issues)
        assert not any("implausible" in i.message for i in issues)


class TestValidateCsvPrefillAb:
    """Test the inline prefill_tps sanity check."""

    def test_implausible_prefill_tps_warns(self, tmp_path):
        csv_path = tmp_path / "test_prefill_ab.csv"
        header = "variant,prefill_len,ttft_s,prefill_tps"
        _write_csv(csv_path, header, "fp16,1024,0.5,5000")
        issues = []
        validate_csv(str(csv_path), "test_prefill_ab.csv", issues)
        assert any("implausible prefill tok/s" in i.message for i in issues)

    def test_negative_prefill_tps_warns(self, tmp_path):
        csv_path = tmp_path / "test_prefill_ab.csv"
        header = "variant,prefill_len,ttft_s,prefill_tps"
        _write_csv(csv_path, header, "fp16,1024,0.5,-1")
        issues = []
        validate_csv(str(csv_path), "test_prefill_ab.csv", issues)
        assert any("implausible prefill tok/s" in i.message for i in issues)

    def test_valid_prefill_tps_no_warning(self, tmp_path):
        csv_path = tmp_path / "test_prefill_ab.csv"
        header = "variant,prefill_len,ttft_s,prefill_tps"
        _write_csv(csv_path, header, "fp16,1024,0.5,180.0")
        issues = []
        validate_csv(str(csv_path), "test_prefill_ab.csv", issues)
        assert not any("implausible" in i.message for i in issues)


class TestValidateCsvQuantComparison:
    """Test the inline tok_per_sec sanity check."""

    def test_implausible_tok_per_sec_warns(self, tmp_path):
        csv_path = tmp_path / "test_quant.csv"
        header = "variant,tok_per_sec,ffn_pct,gdn_proj_pct"
        _write_csv(csv_path, header, "int4,5000,50,50")
        issues = []
        validate_csv(str(csv_path), "test_quant.csv", issues)
        assert any("implausible tok/s" in i.message for i in issues)

    def test_negative_tok_per_sec_warns(self, tmp_path):
        csv_path = tmp_path / "test_quant.csv"
        header = "variant,tok_per_sec,ffn_pct,gdn_proj_pct"
        _write_csv(csv_path, header, "int4,-1,50,50")
        issues = []
        validate_csv(str(csv_path), "test_quant.csv", issues)
        assert any("implausible tok/s" in i.message for i in issues)

    def test_valid_tok_per_sec_no_warning(self, tmp_path):
        csv_path = tmp_path / "test_quant.csv"
        header = "variant,tok_per_sec,ffn_pct,gdn_proj_pct"
        _write_csv(csv_path, header, "int4,42.0,50,50")
        issues = []
        validate_csv(str(csv_path), "test_quant.csv", issues)
        assert not any("implausible" in i.message for i in issues)


class TestValidateCsvCrossToolComparison:
    """Test the inline avg_ts sanity check."""

    def test_implausible_avg_ts_warns(self, tmp_path):
        csv_path = tmp_path / "test_cross.csv"
        header = "engine,quant,test,n_tokens,avg_ts"
        _write_csv(csv_path, header, "cpu,int8,gdn,1024,5000")
        issues = []
        validate_csv(str(csv_path), "test_cross.csv", issues)
        assert any("implausible tok/s" in i.message for i in issues)

    def test_negative_avg_ts_warns(self, tmp_path):
        csv_path = tmp_path / "test_cross.csv"
        header = "engine,quant,test,n_tokens,avg_ts"
        _write_csv(csv_path, header, "cpu,int8,gdn,1024,-1")
        issues = []
        validate_csv(str(csv_path), "test_cross.csv", issues)
        assert any("implausible tok/s" in i.message for i in issues)

    def test_valid_avg_ts_no_warning(self, tmp_path):
        csv_path = tmp_path / "test_cross.csv"
        header = "engine,quant,test,n_tokens,avg_ts"
        _write_csv(csv_path, header, "cpu,int8,gdn,1024,35.5")
        issues = []
        validate_csv(str(csv_path), "test_cross.csv", issues)
        assert not any("implausible" in i.message for i in issues)


class TestValidateCsvQuantAccuracy:
    """Test the inline cos_sim sanity check."""

    def test_low_cos_sim_warns(self, tmp_path):
        csv_path = tmp_path / "test_qa.csv"
        header = "quant_variant,matrix,cos_sim,rel_err_pct"
        _write_csv(csv_path, header, "int8,wq,0.85,15.0")
        issues = []
        validate_csv(str(csv_path), "test_qa.csv", issues)
        assert any("implausible cosine similarity" in i.message for i in issues)

    def test_high_cos_sim_warns(self, tmp_path):
        """cos_sim > 1.0 is mathematically impossible."""
        csv_path = tmp_path / "test_qa.csv"
        header = "quant_variant,matrix,cos_sim,rel_err_pct"
        _write_csv(csv_path, header, "int8,wq,1.05,5.0")
        issues = []
        validate_csv(str(csv_path), "test_qa.csv", issues)
        assert any("implausible cosine similarity" in i.message for i in issues)

    def test_valid_cos_sim_no_warning(self, tmp_path):
        csv_path = tmp_path / "test_qa.csv"
        header = "quant_variant,matrix,cos_sim,rel_err_pct"
        _write_csv(csv_path, header, "int8,wq,0.99,1.0")
        issues = []
        validate_csv(str(csv_path), "test_qa.csv", issues)
        assert not any("implausible" in i.message for i in issues)


# ---------------------------------------------------------------------------
# check_ablation_manifests exception handling (ob-8qt.24)
# ---------------------------------------------------------------------------


class TestCheckAblationManifestsExceptions:
    """Cover the exception branch when reading a malformed ablation CSV."""

    def test_read_error_produces_error_issue(self, tmp_path):
        """A CSV file that raises during read should produce an ERROR issue."""
        ablation_dir = tmp_path / "ablation"
        ablation_dir.mkdir()

        csv_path = ablation_dir / "bad.csv"
        # Write bytes that crash csv.DictReader
        csv_path.write_bytes(b"\xff\xfeinvalid\x00csv")

        issues = []
        check_ablation_manifests(str(ablation_dir), issues)
        assert any(
            "cannot read ablation CSV" in i.message and i.severity == "ERROR" for i in issues
        )


# ---------------------------------------------------------------------------
# Non-numeric ValueError coverage for CSV type dispatch (ob-8qt.24)
# ---------------------------------------------------------------------------


class TestValidateCsvNonNumericPrefillGemm:
    """Cover the ValueError handler for non-numeric prefill tok/s."""

    def test_non_numeric_prefill_tps_handled(self, tmp_path):
        csv_path = tmp_path / "test_prefill.csv"
        header = "prefill_M,ttft_ms,tok_per_sec_prefill"
        _write_csv(csv_path, header, "1,100,N/A")
        issues = []
        validate_csv(str(csv_path), "test_prefill.csv", issues)
        assert not any("implausible" in i.message for i in issues)


class TestValidateCsvNonNumericPrefillAb:
    """Cover the ValueError handler for non-numeric prefill_ab tok/s."""

    def test_non_numeric_prefill_tps_handled(self, tmp_path):
        csv_path = tmp_path / "test_prefill_ab.csv"
        header = "variant,prefill_len,ttft_s,prefill_tps"
        _write_csv(csv_path, header, "fp16,1024,0.5,N/A")
        issues = []
        validate_csv(str(csv_path), "test_prefill_ab.csv", issues)
        assert not any("implausible" in i.message for i in issues)


class TestValidateCsvNonNumericQuantComparison:
    """Cover the ValueError handler for non-numeric quant_comparison tok/s."""

    def test_non_numeric_tok_per_sec_handled(self, tmp_path):
        csv_path = tmp_path / "test_quant.csv"
        header = "variant,tok_per_sec,ffn_pct,gdn_proj_pct"
        _write_csv(csv_path, header, "int4,N/A,50,50")
        issues = []
        validate_csv(str(csv_path), "test_quant.csv", issues)
        assert not any("implausible" in i.message for i in issues)


class TestValidateCsvNonNumericCrossTool:
    """Cover the ValueError handler for non-numeric cross_tool avg_ts."""

    def test_non_numeric_avg_ts_handled(self, tmp_path):
        csv_path = tmp_path / "test_cross.csv"
        header = "engine,quant,test,n_tokens,avg_ts"
        _write_csv(csv_path, header, "cpu,int8,gdn,1024,N/A")
        issues = []
        validate_csv(str(csv_path), "test_cross.csv", issues)
        assert not any("implausible" in i.message for i in issues)


class TestValidateCsvNonNumericQuantAccuracy:
    """Cover the ValueError handler for non-numeric quant_accuracy cos_sim."""

    def test_non_numeric_cos_sim_handled(self, tmp_path):
        csv_path = tmp_path / "test_qa.csv"
        header = "quant_variant,matrix,cos_sim,rel_err_pct"
        _write_csv(csv_path, header, "int8,wq,N/A,15.0")
        issues = []
        validate_csv(str(csv_path), "test_qa.csv", issues)
        assert not any("implausible" in i.message for i in issues)


# ---------------------------------------------------------------------------
# Dispatch branches through validate_csv for ctx_sweep, gpu_micro, kleidiai_gdn (ob-8qt.24)
# ---------------------------------------------------------------------------


class TestValidateCsvDispatch:
    """Ensure the CSV type dispatch reaches the row validators for
    ctx_sweep, gpu_micro, and kleidiai_gdn_kernel types."""

    def test_ctx_sweep_dispatched(self, tmp_path):
        """A ctx_sweep CSV with bad data should trigger validate_ctx_sweep_row."""
        csv_path = tmp_path / "test_ctx.csv"
        header = "model,ctx_len,gdn_layer_us,full_attn_us,ffn_us,total_us,tok_per_sec,kv_cache_mb"
        _write_csv(csv_path, header, "qwen2.5,0,100,50,30,180,50,10")
        issues = []
        validate_csv(str(csv_path), "test_ctx.csv", issues)
        assert any("non-positive ctx_len" in i.message for i in issues)

    def test_gpu_micro_dispatched(self, tmp_path):
        """A gpu_micro CSV with bad data should trigger validate_gpu_micro_row."""
        csv_path = tmp_path / "test_gpu.csv"
        header = "kernel,dim1,dim2,dim3,p50_ms,p95_ms,bw_mibs"
        _write_csv(csv_path, header, "cumsum,64,64,64,1.0,2.0,-1.0")
        issues = []
        validate_csv(str(csv_path), "test_gpu.csv", issues)
        assert any("bw_mibs" in i.message for i in issues)

    def test_kleidiai_gdn_kernel_dispatched(self, tmp_path):
        """A kleidiai_gdn_kernel CSV with bad data should trigger the validator."""
        csv_path = tmp_path / "test_kgdn.csv"
        header = "kernel,shape,p50_us,gib_per_s_p50"
        _write_csv(csv_path, header, "cumdecay,64x2560,0.0,-1.0")
        issues = []
        validate_csv(str(csv_path), "test_kgdn.csv", issues)
        assert any("negative gib_per_s_p50" in i.message for i in issues)
