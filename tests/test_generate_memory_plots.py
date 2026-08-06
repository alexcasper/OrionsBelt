"""Tests for scripts/generate_memory_plots.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "generate_memory_plots.py"


def _load_module():
    """Load the script as a module (it's not in a package)."""
    spec = importlib.util.spec_from_file_location("generate_memory_plots", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generate_memory_plots"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gmp():
    return _load_module()


# ---------------------------------------------------------------------------
# Hypothetical all-attention KV cache
# ---------------------------------------------------------------------------


class TestHypotheticalAllAttention:
    def test_4b_ratio_is_4x(self, gmp):
        """4B has 8 FA layers out of 32 total → 4× multiplier."""
        hyp = gmp._hypothetical_all_attention_kv("4B", 4096)
        actual = gmp.predict_breakdown("4B", 4096)
        ratio = hyp / actual.kv_cache_bytes
        assert abs(ratio - 4.0) < 0.01

    def test_08b_ratio_is_4x(self, gmp):
        """0.8B has 6 FA layers out of 24 total → 4× multiplier."""
        hyp = gmp._hypothetical_all_attention_kv("0.8B", 4096)
        actual = gmp.predict_breakdown("0.8B", 4096)
        ratio = hyp / actual.kv_cache_bytes
        assert abs(ratio - 4.0) < 0.01

    def test_hypothetical_grows_with_context(self, gmp):
        """Hypothetical KV cache grows linearly with context length."""
        small = gmp._hypothetical_all_attention_kv("4B", 4096)
        large = gmp._hypothetical_all_attention_kv("4B", 262144)
        # 262144 / 4096 = 64×
        assert abs(large / small - 64.0) < 0.01


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


class TestComparisonTable:
    def test_table_has_both_checkpoints(self, gmp):
        table = gmp.generate_comparison_table()
        assert "Qwen3.5-4B" in table
        assert "Qwen3.5-0.8B" in table

    def test_table_has_all_context_lengths(self, gmp):
        table = gmp.generate_comparison_table()
        for label in ["4K", "32K", "128K", "262K"]:
            assert label in table

    def test_table_has_key_insight(self, gmp):
        table = gmp.generate_comparison_table()
        assert "Key insight" in table
        assert "GiB" in table

    def test_state_is_constant_across_contexts(self, gmp):
        """GDN recurrent state must be the same at every context length."""
        table = gmp.generate_comparison_table()
        # The table formats state as "51.0 MiB"; it should appear 4 times (one per ctx)
        assert table.count("51.0 MiB") >= 4


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


class TestMain:
    def test_text_only_mode(self, gmp, tmp_path):
        """Text-only mode should produce the markdown table without requiring matplotlib."""
        rc = gmp.main(["--text-only", "--output-dir", str(tmp_path)])
        assert rc == 0
        assert (tmp_path / "memory_comparison.md").exists()
        content = (tmp_path / "memory_comparison.md").read_text()
        assert "Qwen3.5-4B" in content

    def test_custom_output_dir(self, gmp, tmp_path):
        rc = gmp.main(["--text-only", "--output-dir", str(tmp_path / "nested")])
        assert rc == 0
        assert (tmp_path / "nested" / "memory_comparison.md").exists()


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------


class TestUnitConversion:
    def test_gib(self, gmp):
        assert abs(gmp._gib(1024**3) - 1.0) < 1e-9

    def test_mib(self, gmp):
        assert abs(gmp._mib(1024**2) - 1.0) < 1e-9
