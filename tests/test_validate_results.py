"""Tests for scripts/validate_results.py — benchmark CSV and manifest validation.

Covers CSV-type detection, row-level sanity checks (impossible latencies, absurd
throughput, low repeats), manifest linkage, and end-to-end exit codes.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.validate_results import (  # noqa: E402
    ABSURD_THROUGHPUT,
    STANDARD_COLS,
    check_manifest_exists,
    detect_csv_type,
    expected_columns,
    find_device_spec,
    Issue,
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
            "sustained_model", "sustained_kernel", "dispatch_path",
            "elapsed_s", "throughput_gibs", "thermal_c", "vs_first_pct",
        ]
        assert detect_csv_type(cols) == "sustained"

    def test_power_detected(self):
        cols = [
            "timestamp_ms", "power_in_mw", "power_gpu_mw",
            "power_cpu_mw", "temp_milliC",
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
# check_manifest_exists
# ---------------------------------------------------------------------------

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
            "--csv-dir", str(csv_dir),
            "--manifest-dir", str(manifest_dir),
            "--quiet",
        ]
        try:
            return vr.main()
        finally:
            sys.argv = orig_argv

    def test_clean_csv_exit_zero(self, tmp_path):
        """A valid CSV with all good rows exits 0."""
        csv_dir = tmp_path / "raw"
        man_dir = tmp_path / "manifests"
        csv_dir.mkdir()
        man_dir.mkdir()
        _write_std_csv(csv_dir / "jetson-j1.csv", [_std_row()])
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
