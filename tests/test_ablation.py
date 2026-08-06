"""Tests for the ablation matrix runner (scripts/run_ablation.py, bead ob-rqd/ob-1lm).

The ablation grid drives the headline comparison table for the submission.
A silent regression here would produce wrong submission numbers. These tests:

  * Verify the ABLATION_GRID structure (all 6 configs, valid engine/quant fields).
  * Run a minimal end-to-end ablation via SyntheticBackend (the "tiny-model
    smoke run" that finishes in CI time per ob-1lm).
  * Assert every produced CSV validates against the frozen schema on readback.
  * Test the CLI main() entry point.

Uses tiny context lengths and low repeat counts to finish well under CI time
limits, matching the ob-1lm requirement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.schema import read_csv, validate_rows  # noqa: E402
from scripts.run_ablation import ABLATION_GRID, main, run_ablation  # noqa: E402

# ---------------------------------------------------------------------------
# ABLATION_GRID structure
# ---------------------------------------------------------------------------


class TestAblationGridStructure:
    """The grid is the data structure the submission table is built from."""

    def test_six_configurations(self):
        assert len(ABLATION_GRID) == 6

    def test_unique_names(self):
        names = [e["name"] for e in ABLATION_GRID]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize(
        "key",
        ["name", "engine_gdn", "engine_full_attention", "quantization"],
    )
    def test_all_entries_have_required_keys(self, key):
        for entry in ABLATION_GRID:
            assert key in entry, f"entry {entry.get('name')} missing {key}"

    def test_all_gdn_engines_are_cpu(self):
        """GDN layers always run on CPU in the ablation grid (the optimization
        is about the full-attention layers and quantization, not GDN placement)."""
        for entry in ABLATION_GRID:
            assert entry["engine_gdn"] == "cpu"

    def test_full_attention_engines(self):
        fa_engines = {e["engine_full_attention"] for e in ABLATION_GRID}
        assert fa_engines == {"cpu", "gpu_vulkan", "npu"}

    def test_quantization_levels(self):
        quants = {e["quantization"] for e in ABLATION_GRID}
        assert quants == {"fp16", "int4_w4a16"}

    def test_each_engine_has_both_quant_levels(self):
        """Every full-attention engine appears with both fp16 and int4."""
        for engine in ("cpu", "gpu_vulkan", "npu"):
            quants = {
                e["quantization"] for e in ABLATION_GRID if e["engine_full_attention"] == engine
            }
            assert quants == {"fp16", "int4_w4a16"}, (
                f"engine {engine} missing a quantization variant"
            )


# ---------------------------------------------------------------------------
# End-to-end: run_ablation with SyntheticBackend
# ---------------------------------------------------------------------------


class TestRunAblationMinimal:
    """Minimal ablation run that exercises the full pipeline in CI time."""

    def test_produces_six_csvs(self, tmp_path):
        csv_paths = run_ablation(
            context_lengths=[64],
            warmup=1,
            repeats=5,
            decode_length=6,
            output_dir=str(tmp_path / "ablation"),
        )
        assert len(csv_paths) == 6
        for p in csv_paths:
            assert Path(p).exists()

    def test_csvs_match_grid_names(self, tmp_path):
        csv_paths = run_ablation(
            context_lengths=[64],
            warmup=1,
            repeats=5,
            decode_length=6,
            output_dir=str(tmp_path / "ablation"),
        )
        # Each CSV is named ablation_<config_name>.csv
        for entry, path in zip(ABLATION_GRID, csv_paths, strict=True):
            assert f"ablation_{entry['name']}.csv" in path

    def test_all_csvs_validate_on_readback(self, tmp_path):
        """Every ablation CSV must round-trip through read_csv + validate_rows."""
        csv_paths = run_ablation(
            context_lengths=[64],
            warmup=1,
            repeats=5,
            decode_length=6,
            output_dir=str(tmp_path / "ablation"),
        )
        for path in csv_paths:
            rows = read_csv(path)
            assert len(rows) > 0, f"no rows in {path}"
            validate_rows(rows)

    def test_row_count_scales_with_contexts(self, tmp_path):
        """Two context lengths should produce ~2x the rows of one."""
        one = run_ablation(
            context_lengths=[64],
            warmup=0,
            repeats=5,
            decode_length=6,
            output_dir=str(tmp_path / "a1"),
        )
        two = run_ablation(
            context_lengths=[64, 128],
            warmup=0,
            repeats=5,
            decode_length=6,
            output_dir=str(tmp_path / "a2"),
        )
        total_one = sum(len(read_csv(p)) for p in one)
        total_two = sum(len(read_csv(p)) for p in two)
        assert total_two == pytest.approx(2 * total_one, rel=0.05)

    def test_engine_config_propagates_to_rows(self, tmp_path):
        """The engine fields from the grid entry must appear in the CSV."""
        csv_paths = run_ablation(
            context_lengths=[64],
            warmup=1,
            repeats=5,
            decode_length=6,
            output_dir=str(tmp_path / "ablation"),
        )
        for entry, path in zip(ABLATION_GRID, csv_paths, strict=True):
            rows = read_csv(path)
            fa_values = {r.engine_full_attention for r in rows}
            assert fa_values == {entry["engine_full_attention"]}
            gdn_values = {r.engine_gdn for r in rows}
            assert gdn_values == {entry["engine_gdn"]}

    def test_quantization_propagates_to_rows(self, tmp_path):
        csv_paths = run_ablation(
            context_lengths=[64],
            warmup=1,
            repeats=5,
            decode_length=6,
            output_dir=str(tmp_path / "ablation"),
        )
        for entry, path in zip(ABLATION_GRID, csv_paths, strict=True):
            rows = read_csv(path)
            quant_values = {r.quantization for r in rows}
            assert quant_values == {entry["quantization"]}

    def test_memory_decomposition_present(self, tmp_path):
        """Each ablation CSV must include the 3-component memory breakdown
        (weights, kv_cache, recurrent_state) — the headline architectural claim."""
        csv_paths = run_ablation(
            context_lengths=[64],
            warmup=1,
            repeats=5,
            decode_length=6,
            output_dir=str(tmp_path / "ablation"),
        )
        path = csv_paths[0]
        rows = read_csv(path)
        mem_components = {r.metric_component for r in rows if r.metric_name == "peak_memory_bytes"}
        assert "weights" in mem_components
        assert "kv_cache" in mem_components
        assert "recurrent_state" in mem_components


# ---------------------------------------------------------------------------
# CLI main()
# ---------------------------------------------------------------------------


class TestAblationMain:
    """Tests for the CLI main() entry point.

    main() calls run_sweep() internally, which resolves the git SHA via
    ``git rev-parse --short HEAD``.  Since these tests chdir to tmp_path
    (outside the repo), we monkeypatch _git_short_sha to avoid the
    un-attributable-run guard.
    """

    @pytest.fixture(autouse=True)
    def _stub_git_sha(self, monkeypatch):
        monkeypatch.setattr("bench.harness._git_short_sha", lambda: "deadbeef")

    def test_main_writes_csvs_and_table(self, tmp_path, monkeypatch):
        """main() should write per-config CSVs and a comparison table."""
        # main() uses relative default paths, so chdir to tmp_path for isolation.
        monkeypatch.chdir(tmp_path)
        table_path = tmp_path / "table.md"
        rc = main(
            [
                "--context",
                "64",
                "--warmup",
                "1",
                "--repeats",
                "5",
                "--decode-length",
                "6",
                "--table-output",
                str(table_path),
            ]
        )
        assert rc == 0
        assert table_path.exists()
        # CSVs go to results/raw/ablation/ (relative to tmp_path)
        csv_dir = tmp_path / "results" / "raw" / "ablation"
        assert csv_dir.exists()
        csvs = list(csv_dir.glob("*.csv"))
        assert len(csvs) == 6

    def test_main_multi_context(self, tmp_path, monkeypatch):
        """Multiple comma-separated context lengths should be accepted."""
        monkeypatch.chdir(tmp_path)
        table_path = tmp_path / "table.md"
        rc = main(
            [
                "--context",
                "64,128",
                "--warmup",
                "0",
                "--repeats",
                "5",
                "--decode-length",
                "6",
                "--table-output",
                str(table_path),
            ]
        )
        assert rc == 0

    def test_main_table_contains_engine_info(self, tmp_path, monkeypatch):
        """The comparison table should reference the engine configurations."""
        monkeypatch.chdir(tmp_path)
        table_path = tmp_path / "table.md"
        main(
            [
                "--context",
                "64",
                "--warmup",
                "1",
                "--repeats",
                "5",
                "--decode-length",
                "6",
                "--table-output",
                str(table_path),
            ]
        )
        table = table_path.read_text()
        assert "cpu" in table
        assert "prefill_tokens_per_sec" in table
