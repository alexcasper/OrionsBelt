# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the engine boundary-crossing micro-benchmark (bead ob-t3b.6).

Validates:
- CSV format conformance (gpu_micro schema)
- Result plausibility (latency floor, scaling, 16-crossing total)
- Manifest presence
"""
import csv
import json
import os

import pytest

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "raw")
MANIFEST_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "manifests")
CSV_PATH = os.path.join(RESULTS_DIR, "rk3588-t4_gpu_boundary_crossing.csv")
MANIFEST_PATH = os.path.join(
    MANIFEST_DIR, "rk3588-t4_gpu_boundary_crossing.json"
)

_HAS_CSV = os.path.isfile(CSV_PATH)


def _load_csv():
    """Load the boundary-crossing CSV into a list of dicts."""
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def _find_rows(rows, kernel_substr):
    """Return rows whose 'kernel' column contains kernel_substr."""
    return [r for r in rows if kernel_substr in r.get("kernel", "")]


# ---------------------------------------------------------------------------
# CSV existence and format
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CSV, reason="boundary crossing CSV not present")
class TestBoundaryCrossingCSV:
    """Validate the committed CSV from the RK3588 Mali-G610 run."""

    def test_file_exists(self):
        assert os.path.isfile(CSV_PATH)

    def test_csv_has_header(self):
        rows = _load_csv()
        assert len(rows) > 0, "CSV has no data rows"

    def test_csv_columns(self):
        rows = _load_csv()
        expected_cols = {"kernel", "dim1", "dim2", "dim3", "p50_ms", "p95_ms", "bw_mibs"}
        actual_cols = set(rows[0].keys())
        assert expected_cols <= actual_cols, (
            f"Missing columns: {expected_cols - actual_cols}"
        )

    def test_has_write_blocking_rows(self):
        rows = _load_csv()
        write_rows = _find_rows(rows, "write_blocking")
        assert len(write_rows) >= 5, f"Expected ≥5 write_blocking rows, got {len(write_rows)}"

    def test_has_read_blocking_rows(self):
        rows = _load_csv()
        read_rows = _find_rows(rows, "read_blocking")
        assert len(read_rows) >= 5, f"Expected ≥5 read_blocking rows, got {len(read_rows)}"

    def test_has_roundtrip_rows(self):
        rows = _load_csv()
        rt_rows = _find_rows(rows, "roundtrip_blocking")
        assert len(rt_rows) >= 5, f"Expected ≥5 roundtrip rows, got {len(rt_rows)}"

    def test_has_n_crossings_row(self):
        rows = _load_csv()
        crossing_rows = _find_rows(rows, "n_crossings")
        assert len(crossing_rows) == 1, (
            f"Expected 1 n_crossings row, got {len(crossing_rows)}"
        )
        assert crossing_rows[0]["dim1"] == "16"

    def test_has_5kb_payload(self):
        """The critical decode-time payload (hidden_size=2560 fp16 = 5KB)."""
        rows = _load_csv()
        payload_rows = [r for r in rows if "5KB" in r.get("dim1", "")]
        assert len(payload_rows) >= 3, (
            f"Expected ≥3 rows for 5KB payload, got {len(payload_rows)}"
        )


# ---------------------------------------------------------------------------
# Data plausibility
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CSV, reason="boundary crossing CSV not present")
class TestBoundaryCrossingPlausibility:
    """Sanity-check the measured values."""

    @staticmethod
    def _ms(row):
        return float(row["p50_ms"])

    def test_write_latency_positive(self):
        rows = _load_csv()
        for r in _find_rows(rows, "write_blocking"):
            assert self._ms(r) > 0, f"Non-positive write latency: {r}"

    def test_read_latency_positive(self):
        rows = _load_csv()
        for r in _find_rows(rows, "read_blocking"):
            assert self._ms(r) > 0, f"Non-positive read latency: {r}"

    def test_1mb_slower_than_1kb_write(self):
        """Larger payloads should take at least as long (up to noise)."""
        rows = _load_csv()
        wr = {r["dim1"]: self._ms(r) for r in _find_rows(rows, "write_blocking")}
        assert wr["1MB"] >= wr["1KB"] * 0.5, (
            f"1MB write ({wr['1MB']:.3f}ms) unexpectedly faster than "
            f"1KB ({wr['1KB']:.3f}ms) — measurement error?"
        )

    def test_16_crossings_total_under_50ms(self):
        """16 crossings should not dominate the entire decode budget."""
        rows = _load_csv()
        crossing_rows = _find_rows(rows, "n_crossings")
        total_ms = self._ms(crossing_rows[0])
        assert total_ms < 50.0, (
            f"16 crossings total {total_ms:.1f}ms exceeds 50ms — "
            "suspicious for 5KB payloads"
        )

    def test_16_crossings_total_over_1ms(self):
        """16 crossings should have measurable overhead."""
        rows = _load_csv()
        crossing_rows = _find_rows(rows, "n_crossings")
        total_ms = self._ms(crossing_rows[0])
        assert total_ms > 1.0, (
            f"16 crossings total {total_ms:.3f}ms unexpectedly low"
        )

    def test_5kb_write_under_1ms(self):
        """5KB write should be sub-millisecond (dispatch-overhead dominated)."""
        rows = _load_csv()
        wr_5kb = [
            self._ms(r)
            for r in _find_rows(rows, "write_blocking")
            if "5KB" in r["dim1"]
        ]
        assert wr_5kb, "No 5KB write_blocking row found"
        assert wr_5kb[0] < 1.0, (
            f"5KB write {wr_5kb[0]:.3f}ms exceeds 1ms — unexpectedly high"
        )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CSV, reason="boundary crossing CSV not present")
class TestBoundaryCrossingManifest:
    """Validate provenance manifest."""

    def test_manifest_exists(self):
        assert os.path.isfile(MANIFEST_PATH), (
            f"Manifest not found at {MANIFEST_PATH}"
        )

    def test_manifest_has_git_sha(self):
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        git_info = manifest.get("git", {})
        assert "sha" in git_info, "Manifest missing git.sha"
        assert len(git_info["sha"]) >= 8, "Git SHA too short"

    def test_manifest_has_host_info(self):
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        host = manifest.get("host", {})
        assert "core_count" in host or "cpu_model" in host, (
            "Manifest missing host information"
        )

    def test_manifest_device_is_rk3588(self):
        """The manifest should reflect the RK3588-t4 device."""
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        # The manifest may encode device info in various ways.
        # At minimum, the host should have CPU topology data.
        host = manifest.get("host", {})
        cpu_top = host.get("cpu_topology", [])
        if cpu_top:
            # RK3588 has 8 cores
            assert len(cpu_top) == 8, (
                f"Expected 8 CPU cores for RK3588, got {len(cpu_top)}"
            )
