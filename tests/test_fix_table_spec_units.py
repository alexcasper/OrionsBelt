# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/fix_table_spec_units.py — spec bandwidth correction.

Covers device lookup, spec-line repair, percentage recalculation, and
graceful handling of files that are already correct or have unknown devices.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.fix_table_spec_units import (  # noqa: E402
    SPEC_GIBS,
    fix_table_file,
    lookup_spec,
)

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

TABLE_TEMPLATE = """\
# {device} Bandwidth Table

**Device spec bandwidth:** {spec:.1f} GiB/s

## Achieved vs Spec Bandwidth

| Kernel | Achieved (GiB/s) | % of Spec | p50 (μs) | Spread |
|--------|-----------------|-----------|----------|--------|
| gdn_gated_scan | {achieved:.2f} | {pct:.1f}% | 10.5 | 1.2× |
| gdn_delta_update | {achieved2:.2f} | {pct2:.1f}% | 8.3 | 1.1× |

## Notes

Some notes here.
"""


def _make_table(
    tmp_path: Path,
    device: str,
    old_spec: float,
    achieved: float = 10.0,
    achieved2: float = 8.0,
) -> Path:
    """Write a table file and return its path."""
    fpath = tmp_path / f"{device}_table.md"
    pct = achieved / old_spec * 100
    pct2 = achieved2 / old_spec * 100
    fpath.write_text(
        TABLE_TEMPLATE.format(
            device=device,
            spec=old_spec,
            achieved=achieved,
            achieved2=achieved2,
            pct=pct,
            pct2=pct2,
        )
    )
    return fpath


# ---------------------------------------------------------------------------
# lookup_spec
# ---------------------------------------------------------------------------


class TestLookupSpec:
    def test_rk3588(self):
        new, old = lookup_spec("rk3588")
        assert new == SPEC_GIBS["rk3588"]
        assert old is not None

    def test_pi5(self):
        new, old = lookup_spec("pi5")
        assert new == SPEC_GIBS["pi5"]

    def test_jetson(self):
        new, old = lookup_spec("jetson")
        assert new == SPEC_GIBS["jetson"]

    def test_case_insensitive(self):
        new, _ = lookup_spec("RK3588")
        assert new == SPEC_GIBS["rk3588"]

    def test_prefix_match(self):
        new, _ = lookup_spec("rk3588-t4")
        assert new == SPEC_GIBS["rk3588"]

    def test_unknown_device(self):
        new, old = lookup_spec("mystery-device")
        assert new is None
        assert old is None


# ---------------------------------------------------------------------------
# fix_table_file
# ---------------------------------------------------------------------------


class TestFixTableFile:
    def test_fixes_spec_line(self, tmp_path):
        """The stale spec bandwidth line is replaced with the correct value."""
        fpath = _make_table(tmp_path, "rk3588", old_spec=34.0, achieved=10.0)
        ok, msg = fix_table_file(str(fpath))
        assert ok
        text = fpath.read_text()
        assert "**Device spec bandwidth:** 31.7 GiB/s" in text

    def test_recalculates_percentages(self, tmp_path):
        """Percentages are recalculated against the new spec bandwidth."""
        achieved = 15.0
        fpath = _make_table(tmp_path, "rk3588", old_spec=34.0, achieved=achieved)
        fix_table_file(str(fpath))
        text = fpath.read_text()
        # Old % = 15.0 / 34.0 * 100 = 44.1%
        # New % = 15.0 / 31.7 * 100 = 47.3%
        assert "44.1%" not in text
        assert "47.3%" in text

    def test_already_correct(self, tmp_path):
        """A file that already has the correct spec is not modified."""
        fpath = _make_table(tmp_path, "rk3588", old_spec=31.7, achieved=10.0)
        original = fpath.read_text()
        ok, msg = fix_table_file(str(fpath))
        assert not ok
        assert fpath.read_text() == original

    def test_unknown_device(self, tmp_path):
        """A file for an unknown device is skipped."""
        fpath = _make_table(tmp_path, "mystery", old_spec=99.0)
        ok, msg = fix_table_file(str(fpath))
        assert not ok
        assert "unknown" in msg.lower()

    def test_preserves_non_table_content(self, tmp_path):
        """Content outside the bandwidth table is preserved."""
        fpath = _make_table(tmp_path, "jetson", old_spec=25.6, achieved=12.0)
        fix_table_file(str(fpath))
        text = fpath.read_text()
        assert "# jetson Bandwidth Table" in text
        assert "## Notes" in text
        assert "Some notes here." in text

    def test_multiple_rows_all_recalculated(self, tmp_path):
        """All data rows in the bandwidth table get recalculated."""
        fpath = _make_table(tmp_path, "pi5", old_spec=17.0, achieved=10.0, achieved2=8.0)
        fix_table_file(str(fpath))
        text = fpath.read_text()
        # New spec = 15.8
        # Row 1: 10.0 / 15.8 * 100 = 63.3%
        # Row 2:  8.0 / 15.8 * 100 = 50.6%
        assert "63.3%" in text
        assert "50.6%" in text

    def test_no_bandwidth_section(self, tmp_path):
        """A file without a bandwidth table section is handled gracefully."""
        fpath = tmp_path / "rk3588_table.md"
        fpath.write_text(
            "# rk3588 Table\n\n**Device spec bandwidth:** 34.0 GiB/s\n\nNo bandwidth table here.\n"
        )
        ok, msg = fix_table_file(str(fpath))
        # The spec line is still fixed even without the table
        assert ok
        text = fpath.read_text()
        assert "**Device spec bandwidth:** 31.7 GiB/s" in text

    def test_integer_spec_format(self, tmp_path):
        """Spec bandwidth written as integer (no decimal) is also fixed."""
        fpath = tmp_path / "rk3588_table.md"
        fpath.write_text(
            "# rk3588 Table\n\n"
            "**Device spec bandwidth:** 34 GiB/s\n\n"
            "## Achieved vs Spec Bandwidth\n\n"
            "| Kernel | Achieved (GiB/s) | % of Spec | p50 | Spread |\n"
            "|--------|------------------|-----------|-----|--------|\n"
            "| scan | 10.00 | 29.4% | 5.0 | 1.1× |\n"
        )
        ok, msg = fix_table_file(str(fpath))
        assert ok
        text = fpath.read_text()
        assert "31.7 GiB/s" in text
        # Old % = 10.0 / 34.0 * 100 = 29.4% → New % = 10.0 / 31.7 * 100 = 31.5%
        assert "31.5%" in text


class TestFixTableFileEdgeCases:
    """Cover the ValueError exception path and other edge cases."""

    def test_non_numeric_achieved_value_skipped(self, tmp_path):
        """A data row with non-numeric achieved value doesn't crash."""
        fpath = tmp_path / "rk3588_table.md"
        fpath.write_text(
            "# rk3588 Table\n\n"
            "**Device spec bandwidth:** 34.0 GiB/s\n\n"
            "## Achieved vs Spec Bandwidth\n\n"
            "| Kernel | Achieved (GiB/s) | % of Spec | p50 | Spread |\n"
            "|--------|------------------|-----------|-----|--------|\n"
            "| scan | N/A | 29.4% | 5.0 | 1.1× |\n"
        )
        ok, msg = fix_table_file(str(fpath))
        assert ok  # spec line was still fixed
        text = fpath.read_text()
        assert "31.7 GiB/s" in text
        # The N/A row should be preserved (not crashed)
        assert "N/A" in text

    def test_separator_row_not_crashed(self, tmp_path):
        """The markdown separator row should not crash the parser."""
        fpath = tmp_path / "rk3588_table.md"
        fpath.write_text(
            "# rk3588 Table\n\n"
            "**Device spec bandwidth:** 34.0 GiB/s\n\n"
            "## Achieved vs Spec Bandwidth\n\n"
            "| Kernel | Achieved (GiB/s) | % of Spec | p50 | Spread |\n"
            "|--------|------------------|-----------|-----|--------|\n"
        )
        ok, _ = fix_table_file(str(fpath))
        assert ok  # spec line still fixed


class TestMain:
    """Test the main() entry point."""

    def test_main_fixes_files(self, monkeypatch, tmp_path, capsys):
        import scripts.fix_table_spec_units as ftsu

        # Create a table file that needs fixing
        fpath = tmp_path / "rk3588_table.md"
        fpath.write_text(
            "# rk3588 Table\n\n"
            "**Device spec bandwidth:** 34.0 GiB/s\n\n"
            "## Achieved vs Spec Bandwidth\n\n"
            "| Kernel | Achieved (GiB/s) | % of Spec | p50 | Spread |\n"
            "|--------|------------------|-----------|-----|--------|\n"
            "| scan | 10.00 | 29.4% | 5.0 | 1.1× |\n"
        )

        def patched_glob(pattern):
            if "*_table.md" in pattern:
                return [str(fpath)]
            return []

        monkeypatch.setattr(ftsu.glob, "glob", patched_glob)

        ftsu.main()

        captured = capsys.readouterr()
        assert "FIXED" in captured.out
        assert "1 files fixed" in captured.out

    def test_main_skips_unfixable_file(self, monkeypatch, tmp_path, capsys):
        """main() should count files that can't be fixed as skipped."""
        import scripts.fix_table_spec_units as ftsu

        # Table file with unknown device — fix_table_file returns (False, ...)
        fpath = tmp_path / "unknown_table.md"
        fpath.write_text(
            "# unknown Table\n\n"
            "**Device spec bandwidth:** N/A\n\n"
            "## Achieved vs Spec Bandwidth\n\n"
            "| Kernel | Achieved (GiB/s) | % of Spec | p50 | Spread |\n"
            "|--------|------------------|-----------|-----|--------|\n"
            "| scan | 10.00 | N/A | 5.0 | 1.1× |\n"
        )

        def patched_glob(pattern):
            if "*_table.md" in pattern:
                return [str(fpath)]
            return []

        monkeypatch.setattr(ftsu.glob, "glob", patched_glob)

        ftsu.main()

        captured = capsys.readouterr()
        assert "SKIP" in captured.out
        assert "0 files fixed" in captured.out
        assert "1 skipped" in captured.out
