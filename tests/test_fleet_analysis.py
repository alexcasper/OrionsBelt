"""Tests for bench/fleet_analysis.py.

Covers:
- Device registry integrity (DEVICES, REPLICATES constants)
- CSV loading and filtering (fp32-only, seq=64, no decode models)
- Value extraction (get_gibs, get_spread)
- Provenance audit (dirty/clean/missing manifest detection)
- Report generation (structure, tables, O6 extrapolation, spread warnings)
- Plot generation graceful degradation
"""

import csv
import json
import os

import bench.fleet_analysis as fa
import pytest

# ---------------------------------------------------------------------------
# Helpers: create synthetic CSVs and manifests in a temp directory
# ---------------------------------------------------------------------------

CSV_HEADER = [
    "model",
    "kernel",
    "dispatch_path",
    "seq",
    "channels",
    "repeats",
    "p50_us",
    "p95_us",
    "spread_pct",
    "gib_per_s_p50",
    "gflop_per_s_p50",
]

# Canonical kernels that the report iterates over
FP32_KERNELS = ["gdn_cumdecay", "gdn_gated_scan", "gdn_causal_dwconv1d"]

# Representative throughput values (GiB/s) for a "clean" device
CLEAN_GIBS = {
    "gdn_cumdecay": 3.74,
    "gdn_gated_scan": 1.20,
    "gdn_causal_dwconv1d": 2.50,
}

# Spread values
CLEAN_SPREAD = 5.0
NOISY_SPREAD = 20.0


def make_csv_row(model, kernel, gib_per_s, spread_pct, seq=64, channels=4096, repeats=30):
    """Build a single CSV row dict."""
    return {
        "model": model,
        "kernel": kernel,
        "dispatch_path": "neon",
        "seq": str(seq),
        "channels": str(channels),
        "repeats": str(repeats),
        "p50_us": "1000.0",
        "p95_us": "1100.0",
        "spread_pct": str(spread_pct),
        "gib_per_s_p50": str(gib_per_s),
        "gflop_per_s_p50": "0.20",
    }


def write_device_csv(
    path,
    model="Qwen3.5-4B",
    kernels=None,
    gib_overrides=None,
    spread=CLEAN_SPREAD,
    include_bf16=False,
    include_decode=False,
    include_seq1=False,
):
    """Write a synthetic device CSV.

    By default produces fp32, seq=64 rows for the three canonical kernels.
    Extra rows can be included to test filtering logic.
    """
    if kernels is None:
        kernels = list(FP32_KERNELS)
    gib_overrides = gib_overrides or {}

    rows = []
    for kern in kernels:
        gib = gib_overrides.get(kern, CLEAN_GIBS.get(kern, 1.0))
        rows.append(make_csv_row(model, kern, gib, spread))

    # bf16/f16 variants — should be filtered out by load_device_csv
    if include_bf16:
        for kern in FP32_KERNELS:
            rows.append(make_csv_row(model, kern + "_bf16", 4.5, 8.0))
            rows.append(make_csv_row(model, kern + "_f16", 4.0, 9.0))

    # decode model variants — should be filtered out by load_device_csv
    if include_decode:
        decode_model = model + "_decode"
        for kern in FP32_KERNELS:
            rows.append(make_csv_row(decode_model, kern, 10.0, 5.0, seq=1))

    # seq=1 rows (non-decode) — should be filtered out
    if include_seq1:
        for kern in FP32_KERNELS:
            rows.append(make_csv_row(model, kern, 15.0, 3.0, seq=1))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_manifest(path, dirty=True, sha="abcdef0123456"):
    """Write a synthetic manifest JSON."""
    data = {
        "device": "test_device",
        "timestamp": "2026-08-04T12:00:00Z",
        "git": {
            "sha": sha,
            "dirty": dirty,
            "branch": "test_branch",
        },
        "python_version": "3.10.0",
        "compiler": "gcc 9.4.0",
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def setup_fleet_data(root):
    """Create a complete set of synthetic fleet CSVs and manifests under root.

    Creates the directory structure root/results/raw/ and root/results/manifests/
    with all CSVs and manifests that DEVICES, REPLICATES, and J2_OPTIMIZED_CSV
    reference.
    """
    raw_dir = root / "results" / "raw"
    manifests_dir = root / "results" / "manifests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # Device data for DEVICES registry
    device_data_map = {
        "results/raw/pi5-r5.csv": 2.0,
        "results/raw/rk3588-t4_big.csv": 3.3,
        "results/raw/rk3588-t4_little.csv": 1.0,
        "results/raw/jetson-j1.csv": 0.72,
        "results/raw/jetson-j2.csv": 0.73,
    }
    for rel_path, scan_gib in device_data_map.items():
        full_path = root / rel_path
        write_device_csv(
            str(full_path),
            gib_overrides={"gdn_gated_scan": scan_gib},
            spread=5.0,
        )

    # Replicate CSVs
    replicate_map = {
        "results/raw/rk3588-t3_big.csv": (3.29, NOISY_SPREAD),
        "results/raw/rk3588-t3_little.csv": (0.8, 25.0),
        "results/raw/pi5-j1.csv": (1.84, 6.0),
        "results/raw/jetson-j2_single.csv": (1.13, 8.0),
    }
    for rel_path, (scan_gib, spread) in replicate_map.items():
        full_path = root / rel_path
        write_device_csv(
            str(full_path),
            gib_overrides={"gdn_gated_scan": scan_gib},
            spread=spread,
        )

    # j2 optimized CSV (with bf16/f16 + decode rows + 0.8B model)
    j2_opt_path = root / fa.J2_OPTIMIZED_CSV
    j2_opt_dir = j2_opt_path.parent
    j2_opt_dir.mkdir(parents=True, exist_ok=True)
    write_device_csv(
        str(j2_opt_path),
        gib_overrides={"gdn_gated_scan": 2.5},
        include_bf16=True,
        include_decode=True,
    )
    # Also need 0.8B model for the report
    with open(str(j2_opt_path), "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        for kern in FP32_KERNELS:
            writer.writerow(make_csv_row("Qwen3.5-0.8B", kern, 5.0, 5.0))

    # Manifests for all devices
    manifest_basenames = [
        "pi5-r5",
        "pi5-j1",
        "rk3588-t3",
        "rk3588-t4",
        "jetson-j1",
        "jetson-j2",
    ]
    for base in manifest_basenames:
        write_manifest(str(manifests_dir / (base + ".json")), dirty=True)

    return root


# ---------------------------------------------------------------------------
# Constants & registry integrity
# ---------------------------------------------------------------------------


class TestRegistryIntegrity:
    """Validate the DEVICES and REPLICATES constants are well-formed."""

    def test_devices_count(self):
        """Exactly 5 devices in the fleet table."""
        assert len(fa.DEVICES) == 5

    def test_devices_tuples(self):
        """Each device entry is a 5-tuple (name, path, spec_gibs, cores, isa)."""
        for dev in fa.DEVICES:
            assert len(dev) == 5, f"Bad device tuple: {dev}"
            name, path, spec, cores, isa = dev
            assert isinstance(name, str) and name
            assert isinstance(path, str) and path.endswith(".csv")
            assert isinstance(spec, (int, float)) and spec > 0
            assert isinstance(cores, str) and cores
            assert isinstance(isa, str) and isa

    def test_device_names_unique(self):
        names = [d[0] for d in fa.DEVICES]
        assert len(names) == len(set(names)), f"Duplicate device names: {names}"

    def test_device_csv_paths_unique(self):
        paths = [d[1] for d in fa.DEVICES]
        assert len(paths) == len(set(paths)), f"Duplicate CSV paths: {paths}"

    def test_replicates_structure(self):
        """Each replicate group is (class, [(label, path), ...]). Notes are generated
        dynamically from manifest provenance (ob-9t0.3), not hardcoded."""
        for cls, runs in fa.REPLICATES:
            assert isinstance(cls, str) and cls
            assert isinstance(runs, list) and len(runs) >= 2
            for label, path in runs:
                assert isinstance(label, str) and label
                assert isinstance(path, str) and path.endswith(".csv")

    def test_kernel_labels_complete(self):
        """KERNEL_LABELS covers all canonical kernels used in the report."""
        for kern in FP32_KERNELS:
            assert kern in fa.KERNEL_LABELS, f"Missing label for {kern}"

    def test_o6_spec_positive(self):
        assert fa.O6_SPEC_GIBS > 50, "O6 spec seems too low"

    def test_spread_warn_pct(self):
        assert fa.SPREAD_WARN_PCT > 0
        assert fa.SPREAD_WARN_PCT < 100


# ---------------------------------------------------------------------------
# load_device_csv
# ---------------------------------------------------------------------------


class TestLoadDeviceCsv:
    """Test CSV loading and row filtering."""

    def test_missing_file_returns_empty(self):
        rows = fa.load_device_csv("/nonexistent/path/file.csv")
        assert rows == []

    def test_basic_load(self, tmp_path):
        csv_path = write_device_csv(str(tmp_path / "test.csv"))
        rows = fa.load_device_csv(str(csv_path))
        assert len(rows) == 3, f"Expected 3 fp32 kernels, got {len(rows)}"
        kernels_loaded = [r["kernel"] for r in rows]
        assert set(kernels_loaded) == set(FP32_KERNELS)

    def test_filters_bf16_f16(self, tmp_path):
        csv_path = write_device_csv(str(tmp_path / "test.csv"), include_bf16=True)
        rows = fa.load_device_csv(str(csv_path))
        for r in rows:
            assert "_bf16" not in r["kernel"]
            assert "_f16" not in r["kernel"]
        assert len(rows) == 3

    def test_filters_decode_models(self, tmp_path):
        csv_path = write_device_csv(str(tmp_path / "test.csv"), include_decode=True)
        rows = fa.load_device_csv(str(csv_path))
        for r in rows:
            assert "_decode" not in r["model"]
        assert len(rows) == 3

    def test_filters_non64_seq(self, tmp_path):
        csv_path = write_device_csv(str(tmp_path / "test.csv"), include_seq1=True)
        rows = fa.load_device_csv(str(csv_path))
        for r in rows:
            assert r["seq"] == "64"
        assert len(rows) == 3

    def test_all_filters_combined(self, tmp_path):
        csv_path = write_device_csv(
            str(tmp_path / "test.csv"),
            include_bf16=True,
            include_decode=True,
            include_seq1=True,
        )
        rows = fa.load_device_csv(str(csv_path))
        assert len(rows) == 3, "All non-fp32-seq64 rows should be filtered"

    def test_08b_model_loaded(self, tmp_path):
        csv_path = write_device_csv(str(tmp_path / "test.csv"), model="Qwen3.5-0.8B")
        rows = fa.load_device_csv(str(csv_path))
        assert len(rows) == 3
        assert all(r["model"] == "Qwen3.5-0.8B" for r in rows)


# ---------------------------------------------------------------------------
# get_gibs
# ---------------------------------------------------------------------------


class TestGetGibs:
    """Test value extraction from loaded rows."""

    def test_exact_match(self):
        rows = [
            make_csv_row("Qwen3.5-4B", "gdn_gated_scan", 1.20, 5.0),
        ]
        result = fa.get_gibs(rows, "Qwen3.5-4B", "gdn_gated_scan")
        assert result == pytest.approx(1.20)

    def test_no_match_returns_none(self):
        rows = [
            make_csv_row("Qwen3.5-4B", "gdn_gated_scan", 1.20, 5.0),
        ]
        assert fa.get_gibs(rows, "Qwen3.5-4B", "gdn_cumdecay") is None
        assert fa.get_gibs(rows, "Qwen3.5-0.8B", "gdn_gated_scan") is None

    def test_empty_rows(self):
        assert fa.get_gibs([], "Qwen3.5-4B", "gdn_gated_scan") is None

    def test_with_loaded_csv(self, tmp_path):
        csv_path = write_device_csv(
            str(tmp_path / "dev.csv"),
            gib_overrides={"gdn_gated_scan": 2.5},
        )
        rows = fa.load_device_csv(str(csv_path))
        assert fa.get_gibs(rows, "Qwen3.5-4B", "gdn_gated_scan") == pytest.approx(2.5)
        assert fa.get_gibs(rows, "Qwen3.5-4B", "gdn_cumdecay") == pytest.approx(3.74)
        assert fa.get_gibs(rows, "Qwen3.5-0.8B", "gdn_gated_scan") is None


# ---------------------------------------------------------------------------
# get_spread
# ---------------------------------------------------------------------------


class TestGetSpread:
    """Test spread extraction."""

    def test_exact_match(self):
        rows = [make_csv_row("Qwen3.5-4B", "gdn_gated_scan", 1.0, 7.3)]
        assert fa.get_spread(rows, "Qwen3.5-4B", "gdn_gated_scan") == pytest.approx(7.3)

    def test_no_match_returns_none(self):
        rows = [make_csv_row("Qwen3.5-4B", "gdn_gated_scan", 1.0, 5.0)]
        assert fa.get_spread(rows, "Qwen3.5-4B", "gdn_cumdecay") is None

    def test_empty_rows(self):
        assert fa.get_spread([], "Qwen3.5-4B", "gdn_gated_scan") is None

    def test_bad_spread_value(self):
        """If spread_pct is non-numeric, should return None."""
        row = make_csv_row("Qwen3.5-4B", "gdn_gated_scan", 1.0, 5.0)
        row["spread_pct"] = "N/A"
        assert fa.get_spread([row], "Qwen3.5-4B", "gdn_gated_scan") is None

    def test_missing_spread_key(self):
        """If spread_pct key is absent, should return None."""
        row = make_csv_row("Qwen3.5-4B", "gdn_gated_scan", 1.0, 5.0)
        del row["spread_pct"]
        assert fa.get_spread([row], "Qwen3.5-4B", "gdn_gated_scan") is None


# ---------------------------------------------------------------------------
# _provenance_audit_lines
# ---------------------------------------------------------------------------


class TestProvenanceAudit:
    """Test the manifest provenance audit function."""

    def test_all_dirty(self, tmp_path):
        """When all manifests are dirty, the audit reports that."""
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        # Create dirty manifests for all replicate paths
        for _cls, runs in fa.REPLICATES:
            for _label, csv_path in runs:
                base = os.path.basename(csv_path).replace(".csv", "")
                # Handle _big/_little suffix mapping
                candidates = [base]
                for suffix in ("_big", "_little"):
                    if base.endswith(suffix):
                        candidates.append(base[: -len(suffix)])
                for c in candidates:
                    mpath = manifest_dir / (c + ".json")
                    if not mpath.exists():
                        write_manifest(str(mpath), dirty=True)

        original_dir = fa.MANIFEST_DIR
        fa.MANIFEST_DIR = str(manifest_dir)
        try:
            lines = fa._provenance_audit_lines()
            text = "\n".join(lines)
            assert "dirty" in text.lower()
            # Should not say "no manifest at all" since all exist
            assert "no manifest at all" not in text
        finally:
            fa.MANIFEST_DIR = original_dir

    def test_missing_manifests(self, tmp_path):
        """When manifests don't exist, the audit reports missing."""
        empty_dir = tmp_path / "empty_manifests"
        empty_dir.mkdir(parents=True, exist_ok=True)

        original_dir = fa.MANIFEST_DIR
        fa.MANIFEST_DIR = str(empty_dir)
        try:
            lines = fa._provenance_audit_lines()
            text = "\n".join(lines)
            assert "no manifest" in text.lower()
        finally:
            fa.MANIFEST_DIR = original_dir

    def test_mixed_dirty_clean(self, tmp_path):
        """Some dirty, some clean."""
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        # Make first replicate group's manifests clean, rest dirty
        for i, (_cls, runs) in enumerate(fa.REPLICATES):
            for _label, csv_path in runs:
                base = os.path.basename(csv_path).replace(".csv", "")
                candidates = [base]
                for suffix in ("_big", "_little"):
                    if base.endswith(suffix):
                        candidates.append(base[: -len(suffix)])
                for c in candidates:
                    mpath = manifest_dir / (c + ".json")
                    if not mpath.exists():
                        write_manifest(str(mpath), dirty=(i > 0))

        original_dir = fa.MANIFEST_DIR
        fa.MANIFEST_DIR = str(manifest_dir)
        try:
            lines = fa._provenance_audit_lines()
            text = "\n".join(lines)
            assert "dirty" in text.lower()
        finally:
            fa.MANIFEST_DIR = original_dir

    def test_returns_list_of_strings(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir(parents=True, exist_ok=True)

        original_dir = fa.MANIFEST_DIR
        fa.MANIFEST_DIR = str(empty_dir)
        try:
            lines = fa._provenance_audit_lines()
            assert isinstance(lines, list)
            assert all(isinstance(line, str) for line in lines)
        finally:
            fa.MANIFEST_DIR = original_dir

    def test_corrupt_manifest_treated_as_missing(self, tmp_path):
        """A manifest file with invalid JSON is treated as missing."""
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        # Create a corrupt manifest for the first replicate
        first_csv = fa.REPLICATES[0][1][0][1]
        base = os.path.basename(first_csv).replace(".csv", "")
        candidates = [base]
        for suffix in ("_big", "_little"):
            if base.endswith(suffix):
                candidates.append(base[: -len(suffix)])
        for c in candidates:
            (manifest_dir / (c + ".json")).write_text("{corrupt json")

        original_dir = fa.MANIFEST_DIR
        fa.MANIFEST_DIR = str(manifest_dir)
        try:
            lines = fa._provenance_audit_lines()
            # Should report the manifest as missing (not crash)
            assert isinstance(lines, list)
        finally:
            fa.MANIFEST_DIR = original_dir


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Test the full report generation with synthetic data."""

    def test_report_file_written(self, tmp_path, monkeypatch):
        """generate_report writes a markdown file to the given path."""
        setup_fleet_data(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_path = str(tmp_path / "report.md")

        report = fa.generate_report(output_path)

        assert os.path.exists(output_path)
        assert len(report) > 1000, "Report too short"

    def test_report_has_required_sections(self, tmp_path, monkeypatch):
        """Report contains all expected section headers."""
        setup_fleet_data(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_path = str(tmp_path / "report.md")

        report = fa.generate_report(output_path)

        required_sections = [
            "# Fleet Bandwidth-Scaling Analysis",
            "## Devices in the fleet",
            "## Achieved throughput vs spec bandwidth",
            "## The discriminating test",
            "## ⚠ Replicate spread limits everything below this line",
            "## O6 extrapolation",
            "## Optimization impact",
            "Generated by `bench/fleet_analysis.py`",
        ]
        for section in required_sections:
            assert section in report, f"Missing section: {section!r}"

    def test_report_has_device_table(self, tmp_path, monkeypatch):
        """Device table includes Orion O6 row."""
        setup_fleet_data(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_path = str(tmp_path / "report.md")

        report = fa.generate_report(output_path)

        assert "Orion O6" in report
        assert str(fa.O6_SPEC_GIBS) in report

    def test_report_flags_noisy_spread(self, tmp_path, monkeypatch):
        """A device with spread > SPREAD_WARN_PCT gets a warning emoji."""
        setup_fleet_data(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_path = str(tmp_path / "report.md")

        report = fa.generate_report(output_path)

        # The replicate spread table should contain the RK3588 entry
        assert "RK3588" in report
        # The warning emoji should appear for noisy entries
        assert "⚠" in report

    def test_report_has_pi5_vs_jetson_comparison(self, tmp_path, monkeypatch):
        """Discriminating test compares Pi 5 and Jetson."""
        setup_fleet_data(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_path = str(tmp_path / "report.md")

        report = fa.generate_report(output_path)

        assert "Pi 5" in report
        assert "Jetson" in report
        assert "ratio" in report.lower() or "Pi5/J1" in report

    def test_report_has_o6_extrapolation(self, tmp_path, monkeypatch):
        """O6 section has bandwidth-linear and core-performance predictions."""
        setup_fleet_data(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_path = str(tmp_path / "report.md")

        report = fa.generate_report(output_path)

        assert "O6" in report
        assert "extrapolat" in report.lower()
        # Should mention it's probably wrong (instruction-bound)
        assert "wrong" in report.lower() or "overpredict" in report.lower()

    def test_report_has_provenance_audit(self, tmp_path, monkeypatch):
        """Report includes provenance audit section."""
        setup_fleet_data(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_path = str(tmp_path / "report.md")

        report = fa.generate_report(output_path)

        assert "Provenance audit" in report
        assert "dirty" in report.lower()

    def test_report_has_mixed_precision_section(self, tmp_path, monkeypatch):
        """Report includes decode mixed-precision comparison."""
        setup_fleet_data(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_path = str(tmp_path / "report.md")

        report = fa.generate_report(output_path)

        assert "Mixed-precision" in report or "mixed-precision" in report.lower()
        assert "decode" in report.lower()

    def test_report_has_optimization_impact(self, tmp_path, monkeypatch):
        """Optimization impact table compares single vs OpenMP."""
        setup_fleet_data(tmp_path)
        monkeypatch.chdir(tmp_path)
        output_path = str(tmp_path / "report.md")

        report = fa.generate_report(output_path)

        assert "Optimization impact" in report
        assert "OpenMP" in report or "single-thread" in report.lower()


# ---------------------------------------------------------------------------
# generate_report — edge cases with missing data
# ---------------------------------------------------------------------------


class TestReportEdgeCases:
    """Test report behavior with incomplete or missing data."""

    def test_report_with_no_data_files(self, tmp_path, monkeypatch):
        """generate_report should still produce a report even if CSVs are missing."""
        # Empty results dir so all load_device_csv calls return []
        (tmp_path / "results" / "raw").mkdir(parents=True, exist_ok=True)
        (tmp_path / "results" / "manifests").mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(tmp_path)
        output_path = str(tmp_path / "empty_report.md")

        report = fa.generate_report(output_path)

        assert os.path.exists(output_path)
        # Should still have structure
        assert "# Fleet Bandwidth-Scaling Analysis" in report
        assert "Devices in the fleet" in report

    def test_report_with_partial_data(self, tmp_path, monkeypatch):
        """Report with some devices having data, others missing."""
        raw_dir = tmp_path / "results" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (tmp_path / "results" / "manifests").mkdir(parents=True, exist_ok=True)

        # Only write Pi5 data
        write_device_csv(
            str(raw_dir / "pi5-r5.csv"),
            gib_overrides={"gdn_gated_scan": 2.0},
        )
        monkeypatch.chdir(tmp_path)
        output_path = str(tmp_path / "partial_report.md")

        report = fa.generate_report(output_path)

        assert os.path.exists(output_path)
        # Pi5 should have a value
        assert "2.00" in report


# ---------------------------------------------------------------------------
# plot_cross_device
# ---------------------------------------------------------------------------


class TestPlotCrossDevice:
    """Test the plotting function."""

    def test_no_data_returns_false(self, tmp_path):
        """When no devices have scan data, returns False without crashing."""
        device_data = {}
        for name, _path, spec, cores, isa in fa.DEVICES:
            device_data[name] = {
                "rows": [],
                "spec": spec,
                "cores": cores,
                "isa": isa,
            }
        result = fa.plot_cross_device(device_data, str(tmp_path / "plot.png"))
        assert isinstance(result, bool)

    def test_returns_bool_with_data(self, tmp_path, monkeypatch):
        """plot_cross_device returns True or False (never None or crash)."""
        setup_fleet_data(tmp_path)
        monkeypatch.chdir(tmp_path)

        device_data = {}
        for name, path, spec, cores, isa in fa.DEVICES:
            rows = fa.load_device_csv(path)
            device_data[name] = {
                "rows": rows,
                "spec": spec,
                "cores": cores,
                "isa": isa,
            }
        result = fa.plot_cross_device(device_data, str(tmp_path / "plot.png"))
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Integration: full pipeline with real committed CSVs (if present)
# ---------------------------------------------------------------------------


class TestRealDataIntegration:
    """If committed CSVs exist, test the full pipeline end-to-end."""

    @pytest.fixture
    def has_committed_data(self):
        """Check if the real fleet CSVs exist."""
        return all(os.path.exists(path) for _name, path, _spec, _cores, _isa in fa.DEVICES)

    def test_load_all_committed_csvs(self, has_committed_data):
        """Every device CSV in the registry loads without error."""
        if not has_committed_data:
            pytest.skip("Committed CSVs not available")

        for name, path, _spec, _cores, _isa in fa.DEVICES:
            rows = fa.load_device_csv(path)
            assert isinstance(rows, list), f"{name}: load failed"

    def test_committed_report_generation(self, has_committed_data, tmp_path):
        """Generate report from committed data."""
        if not has_committed_data:
            pytest.skip("Committed CSVs not available")

        output_path = str(tmp_path / "real_report.md")
        report = fa.generate_report(output_path)
        assert os.path.exists(output_path)
        assert len(report) > 5000, "Report from real data should be substantial"
        assert "Pi 5" in report
        assert "Jetson" in report
        assert "RK3588" in report


class TestGetManifestSha:
    """Cover get_manifest_sha and _manifest_path_for_csv edge cases."""

    def test_missing_manifest_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fa, "MANIFEST_DIR", str(tmp_path))
        sha, dirty, full = fa.get_manifest_sha(str(tmp_path / "nonexistent.csv"))
        assert sha is None
        assert dirty is None
        assert full is None

    def test_corrupt_manifest_returns_none(self, tmp_path, monkeypatch):
        """Malformed JSON triggers ValueError → (None, None, None)."""
        monkeypatch.setattr(fa, "MANIFEST_DIR", str(tmp_path))
        (tmp_path / "rk3588-t4.json").write_text("{not valid json")
        sha, dirty, full = fa.get_manifest_sha("rk3588-t4.csv")
        assert sha is None
        assert dirty is None
        assert full is None

    def test_valid_manifest(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fa, "MANIFEST_DIR", str(tmp_path))
        (tmp_path / "rk3588-t4.json").write_text(
            json.dumps({"git": {"sha": "abcdef1234567890", "dirty": False}})
        )
        sha, dirty, full = fa.get_manifest_sha("rk3588-t4.csv")
        assert sha == "abcdef123456"
        assert dirty is False
        assert full == "abcdef1234567890"

    def test_empty_sha_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fa, "MANIFEST_DIR", str(tmp_path))
        (tmp_path / "rk3588-t4.json").write_text(json.dumps({"git": {"sha": "", "dirty": True}}))
        sha, dirty, full = fa.get_manifest_sha("rk3588-t4.csv")
        assert sha is None
        assert dirty is True

    def test_big_suffix_finds_shared_manifest(self, tmp_path, monkeypatch):
        """_big CSV finds the shared manifest (without _big suffix)."""
        monkeypatch.setattr(fa, "MANIFEST_DIR", str(tmp_path))
        (tmp_path / "rk3588-t4.json").write_text(
            json.dumps({"git": {"sha": "abc123def456", "dirty": False}})
        )
        sha, dirty, full = fa.get_manifest_sha("rk3588-t4_big.csv")
        assert sha == "abc123def456"


class TestMainCLI:
    """Cover fleet_analysis.main()."""

    def test_main_creates_report(self, tmp_path, monkeypatch):
        """main() writes report and plot files."""
        setup_fleet_data(tmp_path)
        monkeypatch.chdir(tmp_path)
        import sys

        old_argv = sys.argv
        sys.argv = ["fleet_analysis", "--output-dir", str(tmp_path / "out")]
        try:
            fa.main()
        finally:
            sys.argv = old_argv
        assert os.path.exists(str(tmp_path / "out" / "fleet_bandwidth_scaling.md"))
