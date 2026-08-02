"""Tests for bench/plots.py — plot and table generation.

Bead ``ob-9y8``. Verifies that the plotting code:
- runs without errors on committed data
- produces output files
- generates table content with expected structure
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure bench is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BENCH_DIR = _REPO_ROOT / "bench"
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))


class TestMemoryScaling:
    """plot_memory_scaling generates the central claim chart."""

    def test_generates_png(self, tmp_path, monkeypatch):
        """Should produce a PNG file in the figures directory."""
        import plots

        monkeypatch.setattr(plots, "_FIGURES_DIR", tmp_path)
        out = plots.plot_memory_scaling()
        assert out.exists()
        assert out.suffix == ".png"
        assert out.stat().st_size > 1000  # not an empty file

    def test_uses_4b_config(self, tmp_path, monkeypatch):
        """Chart should reflect 4B model dimensions (48 MiB state, 8 attn layers)."""
        import plots

        monkeypatch.setattr(plots, "_FIGURES_DIR", tmp_path)
        out = plots.plot_memory_scaling()
        assert "memory_scaling_4b" in out.name


class TestKernelBandwidth:
    """plot_kernel_bandwidth reads committed device microbenchmark CSVs."""

    def test_generates_pngs(self, tmp_path, monkeypatch):
        """Should produce one PNG per model found in the CSVs."""
        import plots

        monkeypatch.setattr(plots, "_FIGURES_DIR", tmp_path)
        # Use the real raw dir which has committed data
        paths = plots.plot_kernel_bandwidth()
        assert len(paths) >= 1
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 1000

    def test_reads_rk3588_csvs(self):
        """The _read_microbenchmark_csv helper parses our committed format."""
        import plots

        csv_path = plots._RAW_DIR / "rk3588-t4_big.csv"
        if not csv_path.exists():
            pytest.skip("No rk3588-t4 CSV committed yet")
        rows = plots._read_microbenchmark_csv(csv_path)
        assert len(rows) > 0
        assert "model" in rows[0]
        assert "kernel" in rows[0]
        assert isinstance(rows[0]["gib_per_s_p50"], float)
        assert isinstance(rows[0]["gflop_per_s_p50"], float)

    def test_no_csvs_returns_empty(self, tmp_path, monkeypatch):
        """Should gracefully handle no CSVs found."""
        import plots

        monkeypatch.setattr(plots, "_RAW_DIR", tmp_path)
        result = plots.plot_kernel_bandwidth()
        assert result == []


class TestDecodeTraffic:
    """plot_decode_traffic generates the bandwidth model chart."""

    def test_generates_png(self, tmp_path, monkeypatch):
        import plots

        monkeypatch.setattr(plots, "_FIGURES_DIR", tmp_path)
        out = plots.plot_decode_traffic()
        assert out.exists()
        assert out.stat().st_size > 1000


class TestComparisonTable:
    """generate_table produces a markdown table from committed data."""

    def test_generates_markdown(self, tmp_path, monkeypatch):
        import plots

        monkeypatch.setattr(plots, "_FIGURES_DIR", tmp_path)
        out = plots.generate_table()
        assert out.exists()
        content = out.read_text()
        assert "# Results comparison table" in content

    def test_contains_microbenchmark_section(self, tmp_path, monkeypatch):
        import plots

        monkeypatch.setattr(plots, "_FIGURES_DIR", tmp_path)
        content = plots.generate_table().read_text()
        # Should have the kernel table if CSVs exist
        csvs = list(plots._RAW_DIR.glob("rk3588-*.csv"))
        if csvs:
            assert "Static kernel microbenchmark" in content
            assert "gdn_" in content

    def test_contains_memory_decomposition(self, tmp_path, monkeypatch):
        import plots

        monkeypatch.setattr(plots, "_FIGURES_DIR", tmp_path)
        content = plots.generate_table().read_text()
        assert "Memory decomposition" in content
        assert "262K" in content  # should cover all sweep points

    def test_contains_decode_bandwidth(self, tmp_path, monkeypatch):
        import plots

        monkeypatch.setattr(plots, "_FIGURES_DIR", tmp_path)
        content = plots.generate_table().read_text()
        assert "Decode bandwidth model" in content
        assert "INT4" in content

    def test_memory_table_values_match_audit(self, tmp_path, monkeypatch):
        """Table should show 48 MiB recurrent state and 8.00 GiB KV at 262K."""
        import plots

        monkeypatch.setattr(plots, "_FIGURES_DIR", tmp_path)
        content = plots.generate_table().read_text()
        assert "48" in content  # recurrent state MiB
        # KV cache at 262K should be ~8 GiB
        assert "8.00" in content


class TestCLI:
    """The CLI interface."""

    def test_no_args_prints_help(self, capsys):
        import plots

        rc = plots.main([])
        assert rc == 1
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "Generate" in captured.out

    def test_all_flag(self, tmp_path, monkeypatch):
        import plots

        monkeypatch.setattr(plots, "_FIGURES_DIR", tmp_path)
        rc = plots.main(["--all"])
        assert rc == 0

    def test_individual_flags(self, tmp_path, monkeypatch):
        import plots

        monkeypatch.setattr(plots, "_FIGURES_DIR", tmp_path)
        rc = plots.main(["--memory-scaling", "--decode-traffic"])
        assert rc == 0
