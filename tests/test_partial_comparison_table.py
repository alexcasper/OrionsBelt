"""Tests for scripts/partial_comparison_table.py helper functions (ob-9t0.2).

Tests the module-level utility functions that were previously untestable
because all logic ran at import time. After the __main__ refactor, these
functions are importable without side effects.
"""

import os
import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.partial_comparison_table import (  # noqa: E402
    fmt_mem,
    gib,
    kv,
    load_kernel,
    load_sustained,
    main,
    median,
)

RAW = str(Path(_ROOT) / "results" / "raw")

# ---------------------------------------------------------------------------
# median — trivial statistics helper
# ---------------------------------------------------------------------------


class TestMedian:
    def test_odd_length(self):
        assert median([1, 3, 2]) == 2

    def test_even_length(self):
        assert median([1, 2, 3, 4]) == 2.5

    def test_single(self):
        assert median([42]) == 42

    def test_empty_returns_none(self):
        assert median([]) is None

    def test_floats(self):
        assert median([1.5, 2.5, 3.5]) == 2.5


# ---------------------------------------------------------------------------
# fmt_mem — byte formatting
# ---------------------------------------------------------------------------


class TestFmtMem:
    def test_bytes_to_kib(self):
        assert fmt_mem(500) == "0.5 KiB"

    def test_exact_kib(self):
        assert fmt_mem(1024) == "1.0 KiB"

    def test_mib(self):
        assert fmt_mem(1024 * 1024) == "1.0 MiB"

    def test_gib(self):
        assert fmt_mem(1024 * 1024 * 1024) == "1.0 GiB"

    def test_half_gib(self):
        assert fmt_mem(1024 * 1024 * 512) == "512.0 MiB"

    def test_large_value_to_tib(self):
        assert fmt_mem(1024**4 * 2) == "2.0 TiB"

    def test_zero(self):
        assert fmt_mem(0) == "0.0 KiB"


# ---------------------------------------------------------------------------
# kv — CSV row lookup
# ---------------------------------------------------------------------------


class TestKV:
    @pytest.fixture
    def kernel_rows(self):
        return load_kernel("rk3588-t4_big.csv")

    def test_finds_existing_row(self, kernel_rows):
        result = kv(kernel_rows, "Qwen3.5-4B", "gdn_gated_scan", 64)
        assert result is not None
        gib_s, p50_us, spread = result
        assert gib_s > 0
        assert p50_us > 0
        assert spread >= 0

    def test_returns_none_for_missing(self, kernel_rows):
        result = kv(kernel_rows, "Nonexistent", "nonexistent_kernel", 999)
        assert result is None

    def test_returns_tuple_of_three(self, kernel_rows):
        result = kv(kernel_rows, "Qwen3.5-4B", "gdn_cumdecay", 64)
        assert isinstance(result, tuple)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# load_kernel — CSV loading
# ---------------------------------------------------------------------------


class TestLoadKernel:
    def test_returns_list_of_dicts(self):
        rows = load_kernel("rk3588-t4_big.csv")
        assert isinstance(rows, list)
        assert len(rows) > 0
        assert isinstance(rows[0], dict)

    def test_has_expected_columns(self):
        rows = load_kernel("rk3588-t4_big.csv")
        assert "model" in rows[0]
        assert "kernel" in rows[0]
        assert "seq" in rows[0]
        assert "gib_per_s_p50" in rows[0]


# ---------------------------------------------------------------------------
# load_sustained — sustained load CSV
# ---------------------------------------------------------------------------


class TestLoadSustained:
    def test_returns_none_for_missing_file(self):
        result = load_sustained("nonexistent_sustained.csv")
        assert result is None

    def test_returns_tuple_for_valid_file(self):
        # Use a known sustained file if it exists
        path = os.path.join(RAW, "jetson-j1_sustained.csv")
        if not os.path.exists(path):
            pytest.skip("jetson-j1_sustained.csv not present")
        result = load_sustained("jetson-j1_sustained.csv")
        assert result is not None
        first, last, med, count = result
        assert count > 0
        assert isinstance(med, float)


# ---------------------------------------------------------------------------
# gib — convenience GiB/s lookup
# ---------------------------------------------------------------------------


class TestGib:
    def test_returns_float_for_existing(self):
        val = gib("rk3588-t4_big.csv", "gdn_gated_scan")
        assert val is not None
        assert isinstance(val, float)
        assert val > 0

    def test_returns_none_for_missing(self):
        val = gib("rk3588-t4_big.csv", "nonexistent_kernel")
        assert val is None


# ---------------------------------------------------------------------------
# main — integration (runs full script)
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_runs_without_error(self, capsys):
        """main() should produce output and return 0."""
        main()
        captured = capsys.readouterr()
        assert "DONE." in captured.out
        assert "KERNEL FLEET BASELINE" in captured.out
