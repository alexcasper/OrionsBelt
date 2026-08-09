#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/gen_optimization_stack.py — data integrity and chart generation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import gen_optimization_stack as gos  # noqa: E402

# ─── Data integrity ────────────────────────────────────────────────────────────


class TestDataIntegrity:
    """Verify the hardcoded throughput data is self-consistent."""

    def test_stage_count(self):
        assert len(gos.STAGES) == 5

    def test_tput_4b_length_matches_stages(self):
        assert len(gos.TPUT_4B) == len(gos.STAGES)

    def test_tput_08b_length_matches_stages(self):
        assert len(gos.TPUT_08B) == len(gos.STAGES)

    def test_commits_4b_length_matches_stages(self):
        assert len(gos.COMMITS_4B) == len(gos.STAGES)

    def test_colors_length_matches_stages(self):
        assert len(gos.COLORS) == len(gos.STAGES)

    def test_all_tput_positive(self):
        for v in gos.TPUT_4B:
            assert v > 0
        for v in gos.TPUT_08B:
            assert v > 0

    def test_tput_4b_monotonically_increasing(self):
        """Each optimization stage must improve throughput."""
        for i in range(1, len(gos.TPUT_4B)):
            assert gos.TPUT_4B[i] > gos.TPUT_4B[i - 1], (
                f"Stage {i} ({gos.TPUT_4B[i]}) not > stage {i - 1} ({gos.TPUT_4B[i - 1]})"
            )

    def test_tput_08b_monotonically_increasing(self):
        for i in range(1, len(gos.TPUT_08B)):
            assert gos.TPUT_08B[i] > gos.TPUT_08B[i - 1]

    def test_commits_are_hex(self):
        for c in gos.COMMITS_4B:
            assert len(c) == 7  # short hash format
            int(c, 16)  # valid hex


class TestCumulativeSpeedup:
    """The headline speedup claims must match the data."""

    def test_4b_speedup_approx_63x(self):
        speedup = gos.TPUT_4B[-1] / gos.TPUT_4B[0]
        assert 55 < speedup < 70  # ~63×

    def test_08b_speedup_approx_55x(self):
        speedup = gos.TPUT_08B[-1] / gos.TPUT_08B[0]
        assert 50 < speedup < 60  # ~55×

    def test_4b_each_stage_speedup(self):
        """Each stage's individual contribution (relative to previous)."""
        for i in range(1, len(gos.TPUT_4B)):
            ratio = gos.TPUT_4B[i] / gos.TPUT_4B[i - 1]
            assert ratio > 1.0  # every stage helps

    def test_08b_each_stage_speedup(self):
        for i in range(1, len(gos.TPUT_08B)):
            ratio = gos.TPUT_08B[i] / gos.TPUT_08B[i - 1]
            assert ratio > 1.0


class TestStageLabels:
    """Stage labels should describe the optimization progression."""

    def test_first_stage_mentions_fp32(self):
        assert "FP32" in gos.STAGES[0]

    def test_last_stage_mentions_int4_and_sdot(self):
        combined = gos.STAGES[-1]
        assert "INT4" in combined.upper() or "int4" in combined
        assert "SDOT" in combined.upper() or "sdot" in combined

    def test_all_stages_nonempty(self):
        for s in gos.STAGES:
            assert len(s.strip()) > 0


# ─── Chart generation ──────────────────────────────────────────────────────────


class TestGenerateChart:
    """Test the chart output."""

    def test_generates_png(self, tmp_path):
        out = tmp_path / "chart.png"
        result = gos.generate_chart(out)
        assert Path(result).exists()

    def test_output_is_valid_image(self, tmp_path):
        out = tmp_path / "chart.png"
        gos.generate_chart(out)
        data = out.read_bytes()
        # PNG magic bytes
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_output_nonempty(self, tmp_path):
        out = tmp_path / "chart.png"
        gos.generate_chart(out)
        assert out.stat().st_size > 1000  # real PNG, not empty

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "chart.png"
        gos.generate_chart(out)
        assert out.exists()

    def test_returns_output_path(self, tmp_path):
        out = tmp_path / "chart.png"
        result = gos.generate_chart(out)
        assert str(out) in result or result == str(out)


class TestMain:
    """Integration test for main()."""

    def test_main_writes_default_path(self, monkeypatch, tmp_path):

        monkeypatch.chdir(tmp_path)
        gos.main()
        expected = tmp_path / "results" / "figures" / "optimization_stack.png"
        assert expected.exists()
