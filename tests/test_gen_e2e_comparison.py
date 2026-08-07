"""Tests for scripts/gen_e2e_comparison.py — dedup logic and table generation.

Covers the device-name normalization and stale-entry dedup that prevents
old low-run data from cluttering the fleet comparison table.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Import from scripts/ by adding it to the path
_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from gen_e2e_comparison import (  # noqa: E402
    _dedup_rows,
    _normalize_device,
    fmt_mean_std,
)

# ---------------------------------------------------------------------------
# Schema columns matching the e2e schema CSV format
# ---------------------------------------------------------------------------

COLUMNS = [
    "run_id",
    "timestamp",
    "git_sha",
    "manifest_ref",
    "device",
    "engine_gdn",
    "engine_full_attention",
    "model_checkpoint",
    "quantization",
    "context_length",
    "phase",
    "metric_name",
    "metric_component",
    "value",
    "unit",
    "repeat_index",
    "repeat_count",
    "layer_class",
    "notes",
]


def _row(
    *,
    device="rk3588-t3_big",
    model="Qwen/Qwen3.5-4B",
    quant="fp32",
    metric="decode_tokens_per_sec",
    value=1.04,
    git_sha="8e7403c",
    repeat_count=3,
    notes="tokens=16;cluster=big",
):
    """Build a single e2e schema CSV row."""
    return {
        "run_id": f"{device}_e2e_{git_sha}",
        "timestamp": "2026-08-07T05:00:00Z",
        "git_sha": git_sha,
        "manifest_ref": f"results/manifests/{device}_e2e.json",
        "device": device,
        "engine_gdn": "cpu",
        "engine_full_attention": "cpu",
        "model_checkpoint": model,
        "quantization": quant,
        "context_length": "16",
        "phase": "decode" if "tokens_per_sec" in metric else "prefill",
        "metric_name": metric,
        "metric_component": "",
        "value": str(value),
        "unit": "tokens_per_sec" if "tokens_per_sec" in metric else "seconds",
        "repeat_index": "0",
        "repeat_count": str(repeat_count),
        "layer_class": "all",
        "notes": notes,
    }


class TestNormalizeDevice:
    """Test device-name normalization."""

    def test_already_has_big(self):
        assert _normalize_device("rk3588-t3_big", "cluster=big") == "rk3588-t3_big"

    def test_already_has_little(self):
        assert _normalize_device("rk3588-t3_little", "cluster=little") == "rk3588-t3_little"

    def test_infers_big_from_notes(self):
        assert _normalize_device("rk3588-t3", "tokens=16;cluster=big") == "rk3588-t3_big"

    def test_infers_little_from_notes(self):
        assert _normalize_device("rk3588-t3", "tokens=16;cluster=little") == "rk3588-t3_little"

    def test_no_cluster_in_notes(self):
        assert _normalize_device("jetson-j1", "tokens=16") == "jetson-j1_all"

    def test_empty_notes(self):
        assert _normalize_device("jetson-j1", "") == "jetson-j1_all"


class TestDedupRows:
    """Test that stale low-run entries are removed when a higher-run entry exists."""

    def test_removes_fewer_run_entry(self):
        """Old 2-run entry superseded by new 3-run entry for same device/model/quant."""
        rows = [
            _row(device="rk3588-t3", value=1.04, repeat_count=2, git_sha="2e752af"),
            _row(
                device="rk3588-t3",
                value=0.964,
                repeat_count=2,
                git_sha="2e752af",
                metric="ttft_seconds",
            ),
            _row(device="rk3588-t3_big", value=1.04, repeat_count=3, git_sha="8e7403c"),
            _row(
                device="rk3588-t3_big",
                value=0.960,
                repeat_count=3,
                git_sha="8e7403c",
                metric="ttft_seconds",
            ),
        ]
        _dedup_rows(rows)
        devices = {r["device"] for r in rows}
        assert devices == {"rk3588-t3_big"}, f"Expected only _big, got {devices}"

    def test_keeps_different_clusters(self):
        """big and little cluster entries should NOT be deduped."""
        rows = [
            _row(device="rk3588-t3_big", value=1.04, repeat_count=3),
            _row(device="rk3588-t3_big", value=0.960, repeat_count=3, metric="ttft_seconds"),
            _row(device="rk3588-t3_little", value=0.30, repeat_count=2),
            _row(device="rk3588-t3_little", value=2.5, repeat_count=2, metric="ttft_seconds"),
        ]
        _dedup_rows(rows)
        devices = {r["device"] for r in rows}
        assert devices == {"rk3588-t3_big", "rk3588-t3_little"}

    def test_keeps_different_models(self):
        """4B and 0.8B entries should NOT be deduped."""
        rows = [
            _row(device="rk3588-t3_big", model="Qwen/Qwen3.5-4B", value=1.04, repeat_count=3),
            _row(device="rk3588-t3_big", model="Qwen/Qwen3.5-0.8B", value=7.95, repeat_count=3),
        ]
        _dedup_rows(rows)
        assert len(rows) == 2

    def test_keeps_different_quant(self):
        """fp32 and int8 entries should NOT be deduped."""
        rows = [
            _row(device="rk3588-t3_big", quant="fp32", value=1.04, repeat_count=3),
            _row(device="rk3588-t3_big_int8", quant="int8", value=1.84, repeat_count=3),
        ]
        _dedup_rows(rows)
        assert len(rows) == 2

    def test_no_dedup_needed(self):
        """Single entry per group — nothing removed."""
        rows = [
            _row(device="jetson-j1", value=0.43, repeat_count=3),
            _row(device="jetson-j1", value=2.321, repeat_count=3, metric="ttft_seconds"),
        ]
        _dedup_rows(rows)
        assert len(rows) == 2

    def test_handles_missing_notes(self):
        """Rows with empty/missing notes should not crash."""
        rows = [
            _row(device="jetson-j1", value=0.43, repeat_count=3, notes=""),
        ]
        _dedup_rows(rows)
        assert len(rows) == 1


class TestFmtMeanStd:
    """Test the formatting helper."""

    def test_empty(self):
        assert fmt_mean_std([]) == "—"

    def test_single_value(self):
        assert fmt_mean_std([1.04]) == "1.04"

    def test_identical_values(self):
        assert fmt_mean_std([1.04, 1.04, 1.04]) == "1.04"

    def test_with_spread(self):
        result = fmt_mean_std([1.0, 1.1, 1.2])
        assert "±" in result

    def test_low_spread_no_std(self):
        """<1% spread should not show ± std."""
        result = fmt_mean_std([100.0, 100.5])
        assert "±" not in result
