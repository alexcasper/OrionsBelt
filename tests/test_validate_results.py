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
    PROFILE_COLS,
    SCHEMA_COLS,
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
    validate_manifest,
    validate_profile_row,
    validate_schema_row,
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


def _schema_row(**overrides):
    """A valid schema-compliant row dict (bench/schema.py format)."""
    base = {col: "" for col in SCHEMA_COLS}
    base.update(
        {
            "run_id": "rk3588-t4_test",
            "timestamp": "2026-08-06T10:00:00Z",
            "git_sha": "abc1234",
            "manifest_ref": "results/manifests/rk3588-t4_test.json",
            "device": "rk3588-t4",
            "engine_gdn": "cpu",
            "engine_full_attention": "cpu",
            "model_checkpoint": "Qwen3.5-0.8B",
            "quantization": "fp32",
            "context_length": "64",
            "phase": "prefill",
            "metric_name": "prefill_tokens_per_sec",
            "metric_component": "",
            "value": "14.93",
            "unit": "tokens_per_sec",
            "repeat_index": "0",
            "repeat_count": "5",
            "layer_class": "all",
        }
    )
    base.update(overrides)
    return base


def _profile_row(**overrides):
    """A valid per-layer profiling row dict."""
    base = {col: "" for col in PROFILE_COLS}
    base.update(
        {
            "phase": "decode",
            "ctx_len": "64",
            "layer_idx": "0",
            "layer_type": "linear_attention",
            "p50_us": "5000.0",
            "p95_us": "6000.0",
            "mean_us": "5200.0",
            "n_samples": "9",
        }
    )
    base.update(overrides)
    return base


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

    def test_schema_detected(self):
        """Canonical 19-column schema CSV (bench/schema.py) is detected."""
        assert detect_csv_type(SCHEMA_COLS) == "schema"

    def test_schema_detected_with_subset(self):
        """Detection uses key columns, not the full 19."""
        assert (
            detect_csv_type(["run_id", "metric_name", "value", "phase", "context_length"])
            == "schema"
        )

    def test_profile_detected(self):
        assert detect_csv_type(PROFILE_COLS) == "profile"

    def test_schema_takes_precedence_over_standard(self):
        """Schema CSVs contain many columns — must not misidentify as standard."""
        assert detect_csv_type(SCHEMA_COLS) == "schema"


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

    def test_schema_columns(self):
        assert "run_id" in expected_columns("schema")
        assert "metric_name" in expected_columns("schema")

    def test_profile_columns(self):
        assert "layer_idx" in expected_columns("profile")


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
# validate_schema_row
# ---------------------------------------------------------------------------


class TestValidateSchemaRow:
    def test_valid_row_no_issues(self):
        issues = []
        validate_schema_row(_schema_row(), "test.csv", issues, 2)
        assert issues == []

    def test_negative_value(self):
        issues = []
        validate_schema_row(_schema_row(value="-1.5"), "test.csv", issues, 2)
        assert any("negative" in i.message for i in issues)

    def test_zero_context_length(self):
        issues = []
        validate_schema_row(_schema_row(context_length="0"), "test.csv", issues, 2)
        assert any("context_length" in i.message and i.severity == "ERROR" for i in issues)

    def test_repeat_count_zero(self):
        issues = []
        validate_schema_row(_schema_row(repeat_count="0"), "test.csv", issues, 2)
        assert any("repeat_count" in i.message and i.severity == "ERROR" for i in issues)

    def test_repeat_index_out_of_range(self):
        issues = []
        validate_schema_row(_schema_row(repeat_index="5", repeat_count="5"), "test.csv", issues, 2)
        assert any("repeat_index" in i.message and i.severity == "WARNING" for i in issues)

    def test_unexpected_phase(self):
        issues = []
        validate_schema_row(_schema_row(phase="weird"), "test.csv", issues, 2)
        assert any("phase" in i.message and i.severity == "WARNING" for i in issues)

    def test_high_throughput_warning(self):
        issues = []
        validate_schema_row(
            _schema_row(value="999999", metric_name="prefill_tokens_per_sec"),
            "test.csv",
            issues,
            2,
        )
        assert any("high throughput" in i.message for i in issues)

    def test_malformed_value(self):
        issues = []
        validate_schema_row(_schema_row(value="not_a_number"), "test.csv", issues, 2)
        assert any("cannot parse" in i.message for i in issues)


# ---------------------------------------------------------------------------
# validate_profile_row
# ---------------------------------------------------------------------------


class TestValidateProfileRow:
    def test_valid_row_no_issues(self):
        issues = []
        validate_profile_row(_profile_row(), "test.csv", issues, 2)
        assert issues == []

    def test_p95_less_than_p50(self):
        issues = []
        validate_profile_row(_profile_row(p50_us="5000", p95_us="3000"), "test.csv", issues, 2)
        assert any("p95" in i.message for i in issues)

    def test_non_positive_latency(self):
        issues = []
        validate_profile_row(_profile_row(p50_us="0"), "test.csv", issues, 2)
        assert any("latency" in i.message for i in issues)

    def test_too_few_samples(self):
        issues = []
        validate_profile_row(_profile_row(n_samples="0"), "test.csv", issues, 2)
        assert any("n_samples" in i.message for i in issues)

    def test_malformed_value(self):
        issues = []
        validate_profile_row(_profile_row(layer_idx="not_a_number"), "test.csv", issues, 2)
        assert any("cannot parse" in i.message for i in issues)


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
# validate_manifest
# ---------------------------------------------------------------------------


class TestValidateManifest:
    """Tests for manifest-level provenance and dirty-tree checks."""

    def _run(
        self,
        manifest_dict,
        tmp_path,
        csv_name="test.csv",
        csv_type="standard",
        row_count=24,
        head_sha="abc1234",
    ):
        """Helper: write manifest JSON and call validate_manifest."""
        import json

        man_path = tmp_path / "test.json"
        man_path.write_text(json.dumps(manifest_dict))
        issues: list = []
        validate_manifest(csv_name, csv_type, row_count, str(man_path), issues, head_sha)
        return issues

    def test_no_manifest_path_warns(self, tmp_path):
        issues: list = []
        validate_manifest("test.csv", "standard", 24, None, issues, "abc1234")
        assert any(i.severity == "WARNING" and "no manifest" in i.message for i in issues)

    def test_clean_manifest_no_warnings(self, tmp_path):
        """A manifest with git.sha and dirty=False produces no WARNINGs."""
        issues = self._run({"git": {"sha": "abc1234", "dirty": False}}, tmp_path)
        warnings = [i for i in issues if i.severity == "WARNING"]
        assert len(warnings) == 0

    def test_missing_git_section_warns(self, tmp_path):
        """A manifest with no git section at all should WARN about missing provenance."""
        issues = self._run({"device": "test"}, tmp_path)
        warnings = [i for i in issues if i.severity == "WARNING"]
        assert len(warnings) == 1
        assert "no provenance" in warnings[0].message

    def test_empty_git_dict_warns(self, tmp_path):
        """A manifest with an empty git dict should WARN."""
        issues = self._run({"device": "test", "git": {}}, tmp_path)
        warnings = [i for i in issues if i.severity == "WARNING"]
        assert any("no provenance" in w.message for w in warnings)

    def test_sha_present_no_provenance_warning(self, tmp_path):
        """If sha is present, the no-provenance warning should NOT fire."""
        issues = self._run({"git": {"sha": "abc1234", "dirty": True}}, tmp_path)
        no_prov = [i for i in issues if "no provenance" in i.message]
        assert len(no_prov) == 0

    def test_dirty_tree_warns(self, tmp_path):
        """dirty=True should produce a DIRTY tree WARNING."""
        issues = self._run({"git": {"sha": "abc1234", "dirty": True}}, tmp_path)
        dirty_warnings = [i for i in issues if "DIRTY" in i.message]
        assert len(dirty_warnings) == 1

    def test_stale_sha_note(self, tmp_path):
        """SHA different from HEAD produces a NOTE (not WARNING)."""
        issues = self._run(
            {"git": {"sha": "different", "dirty": False}}, tmp_path, head_sha="abc1234"
        )
        notes = [i for i in issues if "run-time snapshot" in i.message]
        assert len(notes) == 1
        assert notes[0].severity == "NOTE"

    def test_sha_matching_head_no_note(self, tmp_path):
        """SHA matching HEAD should not produce a staleness NOTE."""
        issues = self._run(
            {"git": {"sha": "abc1234", "dirty": False}}, tmp_path, head_sha="abc1234"
        )
        notes = [i for i in issues if "run-time snapshot" in i.message]
        assert len(notes) == 0

    def test_note_includes_manifest_filename(self, tmp_path):
        """The first NOTE should identify which manifest was found."""
        issues = self._run({"git": {"sha": "abc1234", "dirty": False}}, tmp_path)
        assert any("test.json" in i.message and i.severity == "NOTE" for i in issues)

    def test_low_row_count_note(self, tmp_path):
        """Standard CSVs with < 12 rows get a NOTE about possible missing variants."""
        issues = self._run({"git": {"sha": "abc1234", "dirty": False}}, tmp_path, row_count=8)
        assert any("only 8 rows" in i.message for i in issues)


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


# ---------------------------------------------------------------------------
# validate_csv() — edge cases
# ---------------------------------------------------------------------------


class TestValidateCsv:
    def test_file_not_found(self):
        issues = []
        result = validate_csv("/nonexistent/file.csv", "file.csv", issues)
        assert result == (None, 0)
        assert any("not found" in i.message for i in issues)

    def test_empty_csv(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("")
        issues = []
        result = validate_csv(str(path), "empty.csv", issues)
        assert result == (None, 0)
        assert any(
            "empty" in i.message.lower() or "unreadable" in i.message.lower() for i in issues
        )

    def test_unrecognized_format(self, tmp_path):
        path = tmp_path / "unknown.csv"
        path.write_text("foo,bar,baz\n1,2,3\n")
        issues = []
        result = validate_csv(str(path), "unknown.csv", issues)
        assert result == (None, 0)
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
        header = ",".join(SCHEMA_COLS)
        row = (
            "run1,2026-01-01,a1b2c3d,manifests/run1.json,rk3588,cpu,cpu,"
            "Qwen3.5-4B,fp16,4096,prefill,prefill_tokens_per_sec,,800.0,"
            "tokens_per_sec,0,5,all,"
        )
        path = tmp_path / "schema.csv"
        path.write_text(f"{header}\n{row}\n")
        issues = []
        csv_type, row_count, schema_ref = validate_csv(str(path), "schema.csv", issues)
        assert csv_type == "schema"
        assert schema_ref == "manifests/run1.json"

    def test_profile_csv_validated(self, tmp_path):
        header = "phase,ctx_len,layer_idx,layer_type,p50_us,p95_us,mean_us,n_samples"
        row = "prefill,64,0,linear_attention,100.0,120.0,110.0,3"
        path = tmp_path / "profile.csv"
        path.write_text(f"{header}\n{row}\n")
        issues = []
        csv_type, row_count, _ = validate_csv(str(path), "profile.csv", issues)
        assert csv_type == "profile"
        assert row_count == 1


class TestLoadManifest:
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
    def test_schema_missing_non_key_columns(self, tmp_path):
        """Schema CSV with key cols but missing some non-key schema cols → ERROR."""
        # Include all SCHEMA_KEY_COLS but omit some non-key ones
        from scripts.validate_results import SCHEMA_KEY_COLS

        partial = sorted(SCHEMA_KEY_COLS | {"run_id", "metric_name"})
        path = tmp_path / "partial_schema.csv"
        path.write_text(",".join(partial) + "\n" + ",".join(["x"] * len(partial)) + "\n")
        issues = []
        csv_type, row_count, _ = validate_csv(str(path), "partial_schema.csv", issues)
        assert csv_type == "schema"
        assert any("missing required columns" in i.message for i in issues)


class TestValidateCsvException:
    def test_read_error_returns_none(self, tmp_path):
        """If CSV reader raises, returns (None, 0) with ERROR."""
        from unittest.mock import patch

        path = tmp_path / "bad.csv"
        path.write_text(",".join(STANDARD_COLS) + "\n")
        issues = []
        with patch("csv.DictReader", side_effect=OSError("io error")):
            result = validate_csv(str(path), "bad.csv", issues)
        assert result == (None, 0)
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
        header = ",".join(SCHEMA_COLS)
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
        header = ",".join(SCHEMA_COLS)
        row = (
            f"run1,2026-01-01,a1b2c3d,{man_file},rk3588,cpu,cpu,"
            "Qwen3.5-4B,fp16,4096,prefill,prefill_tokens_per_sec,,800.0,"
            "tokens_per_sec,0,5,all,"
        )
        (csv_dir / "schema.csv").write_text(f"{header}\n{row}\n")
        rc = self._run_main(csv_dir, man_dir, quiet=True)
        assert rc == 0
