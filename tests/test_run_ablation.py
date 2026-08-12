# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/run_ablation.py — the ablation matrix runner (ob-rqd)."""

import csv
import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.run_ablation import ABLATION_GRID, main, run_ablation  # noqa: E402

# ---------------------------------------------------------------------------
# ABLATION_GRID structure
# ---------------------------------------------------------------------------


class TestAblationGrid:
    """Validate the grid configuration is well-formed."""

    def test_grid_has_six_configs(self):
        assert len(ABLATION_GRID) == 6

    def test_each_entry_has_required_keys(self):
        required = {"name", "engine_gdn", "engine_full_attention", "quantization"}
        for entry in ABLATION_GRID:
            assert required <= set(entry), f"{entry['name']} missing keys"

    def test_names_unique(self):
        names = [e["name"] for e in ABLATION_GRID]
        assert len(names) == len(set(names))

    def test_gdn_always_cpu(self):
        """GDN scan stays on CPU per the heterogeneous mapping hypothesis."""
        for entry in ABLATION_GRID:
            assert entry["engine_gdn"] == "cpu", f"{entry['name']} has gdn on {entry['engine_gdn']}"

    def test_quantization_values(self):
        valid = {"fp16", "int4_w4a16"}
        for entry in ABLATION_GRID:
            assert entry["quantization"] in valid

    def test_engines_cover_cpu_gpu_npu(self):
        attn_engines = {e["engine_full_attention"] for e in ABLATION_GRID}
        assert attn_engines == {"cpu", "gpu_vulkan", "npu"}


# ---------------------------------------------------------------------------
# run_ablation() — end-to-end with SyntheticBackend
# ---------------------------------------------------------------------------


class TestRunAblation:
    """Run the ablation grid and validate the CSV output."""

    @pytest.fixture
    def csv_paths(self, tmp_path):
        return run_ablation(
            context_lengths=[4096],
            warmup=1,
            repeats=5,
            decode_length=5,
            output_dir=str(tmp_path / "ablation"),
        )

    def test_returns_one_csv_per_config(self, csv_paths):
        assert len(csv_paths) == len(ABLATION_GRID)

    def test_csvs_exist(self, csv_paths):
        for p in csv_paths:
            assert Path(p).exists()

    def test_csv_has_schema_columns(self, csv_paths):
        """All CSVs must use the 19-column results schema."""
        with open(csv_paths[0]) as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
        assert len(cols) == 19
        assert "run_id" in cols
        assert "metric_name" in cols
        assert "device" in cols

    def test_rows_have_data(self, csv_paths):
        for p in csv_paths:
            with open(p) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) > 0

    def test_each_csv_matches_config_name(self, csv_paths):
        names = {Path(p).stem.replace("ablation_", "") for p in csv_paths}
        expected = {e["name"] for e in ABLATION_GRID}
        assert names == expected

    def test_engine_assignment_in_csv(self, csv_paths):
        """engine_gdn / engine_full_attention columns reflect the grid config."""
        for p in csv_paths:
            config_name = Path(p).stem.replace("ablation_", "")
            grid_entry = next(e for e in ABLATION_GRID if e["name"] == config_name)
            with open(p) as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                assert row["engine_gdn"] == grid_entry["engine_gdn"]
                assert row["engine_full_attention"] == grid_entry["engine_full_attention"]

    def test_quantization_in_csv(self, csv_paths):
        for p in csv_paths:
            config_name = Path(p).stem.replace("ablation_", "")
            grid_entry = next(e for e in ABLATION_GRID if e["name"] == config_name)
            with open(p) as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                assert row["quantization"] == grid_entry["quantization"]

    def test_context_length_reflected(self, csv_paths):
        with open(csv_paths[0]) as f:
            rows = list(csv.DictReader(f))
        ctx_values = {r["context_length"] for r in rows}
        assert ctx_values == {"4096"}

    def test_output_dir_created_if_missing(self, tmp_path):
        d = tmp_path / "deep" / "nested" / "ablation"
        result = run_ablation([4096], warmup=1, repeats=5, output_dir=str(d))
        assert len(result) == 6
        assert d.exists()


# ---------------------------------------------------------------------------
# main() — CLI entry point
# ---------------------------------------------------------------------------


class TestMain:
    """Test the CLI entry point.

    Tests pass --output-dir to a temp directory so they never overwrite
    the committed ablation CSVs in results/raw/ablation/.
    """

    def test_main_returns_zero(self, tmp_path):
        rc = main(
            [
                "--context",
                "4096",
                "--warmup",
                "1",
                "--repeats",
                "5",
                "--decode-length",
                "5",
                "--output-dir",
                str(tmp_path / "ablation"),
                "--manifest-dir",
                str(tmp_path / "manifests"),
                "--table-output",
                str(tmp_path / "table.md"),
            ]
        )
        assert rc == 0

    def test_main_writes_table(self, tmp_path):
        table_path = tmp_path / "table.md"
        main(
            [
                "--context",
                "4096",
                "--warmup",
                "1",
                "--repeats",
                "5",
                "--output-dir",
                str(tmp_path / "ablation"),
                "--manifest-dir",
                str(tmp_path / "manifests"),
                "--table-output",
                str(table_path),
            ]
        )
        assert table_path.exists()
        content = table_path.read_text()
        assert "Synthetic data" in content  # disclaimer present

    def test_main_multiple_context_lengths(self, tmp_path):
        rc = main(
            [
                "--context",
                "4096,32768",
                "--warmup",
                "1",
                "--repeats",
                "5",
                "--output-dir",
                str(tmp_path / "ablation"),
                "--manifest-dir",
                str(tmp_path / "manifests"),
                "--table-output",
                str(tmp_path / "table.md"),
            ]
        )
        assert rc == 0


# ---------------------------------------------------------------------------
# Coverage gap tests (ob-eek)
# ---------------------------------------------------------------------------


class TestWriteManifestForRows:
    def test_empty_rows_returns_none(self):
        """_write_manifest_for_rows returns None for empty rows list."""
        from bench.harness import SweepConfig
        from scripts.run_ablation import _write_manifest_for_rows

        config = SweepConfig(
            context_lengths=[4096],
            warmup_count=1,
            repeat_count=5,
            decode_length=20,
        )
        result = _write_manifest_for_rows([], config)
        assert result is None


class TestMainEntryRunpy:
    def test_main_via_runpy(self, tmp_path, monkeypatch):
        """Running as __main__ via runpy covers the __main__ guard."""
        import runpy

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_ablation.py",
                "--context",
                "4096",
                "--warmup",
                "1",
                "--repeats",
                "5",
                "--output-dir",
                str(tmp_path / "ablation"),
                "--manifest-dir",
                str(tmp_path / "manifests"),
                "--table-output",
                str(tmp_path / "table.md"),
            ],
        )
        script_path = str(Path(__file__).resolve().parent.parent / "scripts" / "run_ablation.py")
        import pytest as _pt

        with _pt.raises(SystemExit) as exc_info:
            runpy.run_path(script_path, run_name="__main__")
        assert exc_info.value.code == 0
