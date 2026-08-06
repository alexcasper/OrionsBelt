"""Tests for bench/profile_layers.py — per-layer latency profiling.

Focuses on write_csv() pure logic (p50/p95/mean computation, layer-type
assignment, CSV format) since load_model/run_profiling require PyTorch + weights.
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.profile_layers import write_csv  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_times():
    """Build a small all_times dict with 3 layers, 2 phases, 1 ctx."""
    times = defaultdict(list)
    # Layer 0: linear_attention, prefill, ctx=64, 3 samples
    times[(0, "prefill", 64)] = [100.0, 120.0, 110.0]
    # Layer 1: full_attention, prefill, ctx=64, 3 samples
    times[(1, "prefill", 64)] = [200.0, 220.0, 210.0]
    # Layer 0: linear_attention, decode, ctx=64, 3 samples
    times[(0, "decode", 64)] = [50.0, 55.0, 52.0]
    # Layer 1: full_attention, decode, ctx=64, 3 samples
    times[(1, "decode", 64)] = [80.0, 90.0, 85.0]
    return times


def _read_csv(path):
    """Read a CSV file and return (fieldnames, list-of-row-dicts)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


# ---------------------------------------------------------------------------
# CSV output format
# ---------------------------------------------------------------------------


class TestWriteCsvFormat:
    def test_header_columns(self, tmp_path):
        times = _make_times()
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        fieldnames, rows = _read_csv(out)
        assert fieldnames == [
            "phase",
            "ctx_len",
            "layer_idx",
            "layer_type",
            "p50_us",
            "p95_us",
            "mean_us",
            "n_samples",
        ]

    def test_row_count(self, tmp_path):
        times = _make_times()
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        assert len(rows) == 4  # 2 layers × 2 phases

    def test_rows_sorted_by_layer_then_phase(self, tmp_path):
        times = _make_times()
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        # sorted(all_times) orders by (idx, phase, ctx)
        assert rows[0]["layer_idx"] == "0"
        assert rows[0]["phase"] == "decode"  # "decode" < "prefill" alphabetically
        assert rows[1]["layer_idx"] == "0"
        assert rows[1]["phase"] == "prefill"
        assert rows[2]["layer_idx"] == "1"
        assert rows[2]["phase"] == "decode"

    def test_empty_times_produces_empty_csv(self, tmp_path):
        out = tmp_path / "profile.csv"
        write_csv(defaultdict(list), set(), set(), str(out))
        fieldnames, rows = _read_csv(out)
        assert fieldnames is not None
        assert len(rows) == 0


# ---------------------------------------------------------------------------
# Layer-type assignment
# ---------------------------------------------------------------------------


class TestWriteCsvLayerType:
    def test_linear_attention_assigned(self, tmp_path):
        times = _make_times()
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        layer0_rows = [r for r in rows if r["layer_idx"] == "0"]
        assert all(r["layer_type"] == "linear_attention" for r in layer0_rows)

    def test_full_attention_assigned(self, tmp_path):
        times = _make_times()
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        layer1_rows = [r for r in rows if r["layer_idx"] == "1"]
        assert all(r["layer_type"] == "full_attention" for r in layer1_rows)

    def test_layer_not_in_either_set_defaults_to_linear(self, tmp_path):
        """A layer index not in full_attn defaults to linear_attention."""
        times = defaultdict(list)
        times[(5, "prefill", 32)] = [10.0, 20.0, 15.0]
        out = tmp_path / "profile.csv"
        write_csv(times, set(), set(), str(out))  # 5 not in either set
        _, rows = _read_csv(out)
        assert len(rows) == 1
        assert rows[0]["layer_type"] == "linear_attention"


# ---------------------------------------------------------------------------
# Statistics (p50, p95, mean)
# ---------------------------------------------------------------------------


class TestWriteCsvStats:
    def test_p50_is_median(self, tmp_path):
        times = defaultdict(list)
        times[(0, "prefill", 64)] = [100.0, 120.0, 110.0]
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        assert float(rows[0]["p50_us"]) == statistics.median([100.0, 120.0, 110.0])

    def test_mean_is_correct(self, tmp_path):
        times = defaultdict(list)
        times[(0, "prefill", 64)] = [100.0, 120.0, 110.0]
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        assert abs(float(rows[0]["mean_us"]) - statistics.mean([100.0, 120.0, 110.0])) < 0.1

    def test_p95_is_max_for_small_samples(self, tmp_path):
        """With < 20 samples, p95 should be max (not the percentile index)."""
        times = defaultdict(list)
        times[(0, "prefill", 64)] = [100.0, 200.0, 150.0]
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        assert float(rows[0]["p95_us"]) == 200.0

    def test_p95_percentile_for_large_samples(self, tmp_path):
        """With >= 20 samples, p95 should use the index-based percentile."""
        samples = list(range(100, 2100, 100))  # 20 samples: 100..2000
        times = defaultdict(list)
        times[(0, "prefill", 64)] = samples
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        expected_p95 = sorted(samples)[int(len(samples) * 0.95)]
        assert float(rows[0]["p95_us"]) == expected_p95

    def test_n_samples_recorded(self, tmp_path):
        times = _make_times()
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        for row in rows:
            assert int(row["n_samples"]) == 3

    def test_single_sample(self, tmp_path):
        """A single sample: p50 = p95 = mean = that value."""
        times = defaultdict(list)
        times[(0, "prefill", 64)] = [42.0]
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        assert float(rows[0]["p50_us"]) == 42.0
        assert float(rows[0]["p95_us"]) == 42.0
        assert float(rows[0]["mean_us"]) == 42.0


# ---------------------------------------------------------------------------
# Multiple contexts and phases
# ---------------------------------------------------------------------------


class TestWriteCsvMultiContext:
    def test_multiple_contexts(self, tmp_path):
        times = defaultdict(list)
        times[(0, "prefill", 32)] = [10.0, 12.0, 11.0]
        times[(0, "prefill", 64)] = [20.0, 22.0, 21.0]
        times[(0, "decode", 32)] = [5.0, 6.0, 5.5]
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        assert len(rows) == 3
        ctx_values = {r["ctx_len"] for r in rows}
        assert ctx_values == {"32", "64"}

    def test_summary_prints_to_stdout(self, tmp_path, capsys):
        times = _make_times()
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        captured = capsys.readouterr()
        assert "Summary" in captured.out
        assert "linear_attention" in captured.out
        assert "full_attention" in captured.out
        assert "Wrote 4 rows" in captured.out
