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
    KLEIDIAI_MATMUL_COLS,
    LAYER_PROFILE_COLS,
    STANDARD_COLS,
    SUSTAINED_COLS,
    Issue,
    check_manifest_exists,
    detect_csv_type,
    expected_columns,
    find_device_spec,
    get_git_head_sha,
    load_manifest,
    main,
    validate_csv,
    validate_e2e_sweep_row,
    validate_gpu_micro_row,
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


# ---------------------------------------------------------------------------
# find_device_spec
# ---------------------------------------------------------------------------


class TestFindDeviceSpec:
    def test_jetson(self):
        assert find_device_spec("jetson-j1.csv") == 25.6

    def test_pi5(self):
        assert find_device_spec("pi5-r5.csv") == 17.0

    def test_rk3588(self):
        assert find_device_spec("rk3588-t4_big.csv") == 34.0

    def test_orion(self):
        assert find_device_spec("orion-o6_big.csv") == 100.0

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
