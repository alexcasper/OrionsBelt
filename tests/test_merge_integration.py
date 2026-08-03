"""Cross-module integration tests after bench/j1 merge.

Exercises the full pipeline: harness → CSV → comparison_table → memory.
Tests that modules from different agents work together correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.comparison_table import generate_comparison, load_and_summarize  # noqa: E402
from bench.harness import HarnessConfig, SyntheticBackend, run_sweep  # noqa: E402
from bench.schema import read_csv, validate_rows, write_csv  # noqa: E402


def _make_config(**overrides):
    """Create a minimal valid HarnessConfig for testing."""
    defaults = dict(
        model_checkpoint="test@synthetic",
        device="generic_aarch64",
        engine_gdn="cpu",
        engine_full_attention="cpu",
        quantization="fp16",
        context_lengths=[64],
        warmups=0,
        repeats=5,
        decode_tokens=4,
    )
    defaults.update(overrides)
    return HarnessConfig(**defaults)


class TestHarnessToComparisonTable:
    """Harness CSV output feeds into the comparison table generator."""

    def test_csv_to_comparison_table(self, tmp_path):
        """The comparison table generator must consume harness CSVs."""
        backend = SyntheticBackend()
        config = _make_config()
        rows = run_sweep(backend, config, progress=False)
        assert len(rows) > 0

        csv_path = str(tmp_path / "sweep.csv")
        write_csv(rows, csv_path)

        table = generate_comparison([csv_path])
        assert "prefill_tokens_per_sec" in table or "decode_tokens_per_sec" in table

    def test_multi_config_comparison(self, tmp_path):
        """Multiple CSVs with different configs produce a multi-row table."""
        csvs = []
        for engine in ("cpu", "gpu_vulkan", "npu"):
            backend = SyntheticBackend()
            config = _make_config(engine_full_attention=engine)
            rows = run_sweep(backend, config, progress=False)
            csv_path = str(tmp_path / f"sweep_{engine}.csv")
            write_csv(rows, csv_path)
            csvs.append(csv_path)

        summaries = load_and_summarize(csvs)
        engines = {s["engine_full_attention"] for s in summaries}
        assert "cpu" in engines
        assert "gpu_vulkan" in engines


class TestHarnessSchemaConsistency:
    """All rows produced by the harness must validate against the frozen schema."""

    def test_all_rows_validate_after_sweep(self):
        """Every row from run_sweep must pass schema.validate_rows."""
        backend = SyntheticBackend()
        config = _make_config(context_lengths=[64, 128])
        rows = run_sweep(backend, config, progress=False)
        assert len(rows) > 0
        validate_rows(rows)  # raises on invalid

    def test_csv_roundtrip_preserves_row_count(self, tmp_path):
        """CSV write → read must preserve the row count."""
        backend = SyntheticBackend()
        config = _make_config()
        rows = run_sweep(backend, config, progress=False)

        csv_path = str(tmp_path / "roundtrip.csv")
        write_csv(rows, csv_path)
        read_back = read_csv(csv_path)
        validate_rows(read_back)
        assert len(read_back) == len(rows)


class TestBackwardCompat:
    """The backward-compatible aliases I added must work."""

    def test_sweepconfig_alias_exists(self):
        """SweepConfig should be an alias for HarnessConfig."""
        from bench.harness import SweepConfig

        assert SweepConfig is HarnessConfig

    def test_model_config_alias_exists(self):
        """ModelConfig and QWEN35_4B should be importable for bench/memory.py."""
        from bench.harness import QWEN35_08B, QWEN35_4B

        assert QWEN35_4B.num_gdn_layers == 24
        assert QWEN35_08B.num_gdn_layers == 18
        assert QWEN35_4B.fa_head_dim == 256
