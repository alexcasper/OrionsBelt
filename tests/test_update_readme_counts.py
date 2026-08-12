# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/update_readme_counts.py — README auto-count repair.

Covers file counting, headline repair, directory-layout repair, dry-run,
and graceful handling of README files that lack the expected patterns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import scripts.update_readme_counts as urc  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo(
    tmp_path: Path, csvs: int = 3, manifests: int = 5, figures: int = 2, findings: int = 3
):
    """Create a minimal repo structure under *tmp_path*."""
    (tmp_path / "results" / "raw").mkdir(parents=True)
    (tmp_path / "results" / "manifests").mkdir(parents=True)
    (tmp_path / "results" / "figures").mkdir(parents=True)
    (tmp_path / "docs").mkdir(exist_ok=True)

    for i in range(csvs):
        (tmp_path / "results" / "raw" / f"device-{i}.csv").write_text("data\n")
    for i in range(manifests):
        (tmp_path / "results" / "manifests" / f"device-{i}.json").write_text("{}")
    for i in range(figures):
        (tmp_path / "results" / "figures" / f"fig-{i}.md").write_text("# fig\n")

    # figures/README.md should NOT be counted as a figure
    (tmp_path / "results" / "figures" / "README.md").write_text("# index\n")

    # docs/FINDINGS.md with the requested number of ## sections
    (tmp_path / "docs" / "FINDINGS.md").write_text(
        "".join(f"## Section {i}\nContent.\n\n" for i in range(findings))
    )


README_TEMPLATE = """\
# Test Repo

> **Results so far:** {csvs} CSVs from the device fleet, {manifests} provenance manifests, {figs} generated figures/tables, {findings} FINDINGS sections.

> ```
> results/
>   raw/         <- {csvs} per-run CSVs
>   manifests/   <- {manifests} provenance manifests (git SHA, governor, thermals)
>   figures/     <- fleet analysis, comparison table
> ```
"""


def _write_readme(
    tmp_path: Path,
    csvs: int,
    manifests: int,
    figs: int,
    findings: int = 3,
):
    (tmp_path / "README.md").write_text(
        README_TEMPLATE.format(csvs=csvs, manifests=manifests, figs=figs, findings=findings)
    )


# ---------------------------------------------------------------------------
# _count_files
# ---------------------------------------------------------------------------


class TestCountFiles:
    def test_count_csvs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, csvs=5, manifests=0, figures=0)
        assert urc._count_files("results/raw", suffix=".csv") == 5

    def test_count_manifests(self, tmp_path, monkeypatch):
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, csvs=0, manifests=7, figures=0)
        assert urc._count_files("results/manifests", suffix=".json") == 7

    def test_count_figures_excludes_readme(self, tmp_path, monkeypatch):
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, csvs=0, manifests=0, figures=3)
        # 3 figure files + 1 README.md (excluded)
        assert urc._count_files("results/figures", exclude_name="README.md") == 3

    def test_count_recursive(self, tmp_path, monkeypatch):
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, csvs=2, manifests=0, figures=0)
        # Add files in a subdirectory
        sub = tmp_path / "results" / "raw" / "ablation"
        sub.mkdir()
        (sub / "extra.csv").write_text("data\n")
        assert urc._count_files("results/raw", suffix=".csv") == 3

    def test_count_empty_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        (tmp_path / "results" / "raw").mkdir(parents=True)
        assert urc._count_files("results/raw", suffix=".csv") == 0

    def test_count_findings_sections(self, tmp_path, monkeypatch):
        """_count_findings_sections counts all ## headers in FINDINGS.md."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, findings=5)
        assert urc._count_findings_sections() == 5

    def test_count_findings_sections_no_file(self, tmp_path, monkeypatch):
        """Returns 0 when docs/FINDINGS.md does not exist."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        assert urc._count_findings_sections() == 0


# ---------------------------------------------------------------------------
# update_readme — repair scenarios
# ---------------------------------------------------------------------------


class TestUpdateReadmeRepair:
    def test_already_correct(self, tmp_path, monkeypatch):
        """When counts match, no changes are made."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, csvs=5, manifests=10, figures=3)
        _write_readme(tmp_path, csvs=5, manifests=10, figs=3)

        n = urc.update_readme()
        assert n == 0

    def test_fix_manifest_drift_headline(self, tmp_path, monkeypatch):
        """Stale manifest count in headline is repaired."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, csvs=5, manifests=20, figures=3)
        # README claims only 10 manifests
        _write_readme(tmp_path, csvs=5, manifests=10, figs=3)

        n = urc.update_readme()
        assert n == 2  # headline + dir-layout
        text = (tmp_path / "README.md").read_text()
        assert "20 provenance manifests" in text

    def test_fix_csv_drift_headline(self, tmp_path, monkeypatch):
        """Stale CSV count in headline is repaired."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, csvs=8, manifests=10, figures=3)
        _write_readme(tmp_path, csvs=5, manifests=10, figs=3)

        n = urc.update_readme()
        assert n == 2  # headline + raw dir-layout (manifests match in dir-layout too)
        text = (tmp_path / "README.md").read_text()
        assert "8 CSVs" in text

    def test_fix_figures_drift_headline(self, tmp_path, monkeypatch):
        """Stale figures count in headline is repaired."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, csvs=5, manifests=10, figures=6)
        _write_readme(tmp_path, csvs=5, manifests=10, figs=3)

        n = urc.update_readme()
        assert n == 1  # headline only
        text = (tmp_path / "README.md").read_text()
        assert "6 generated figures/tables" in text

    def test_fix_findings_drift_headline(self, tmp_path, monkeypatch):
        """Stale FINDINGS count in headline is repaired."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, csvs=5, manifests=10, figures=3, findings=7)
        _write_readme(tmp_path, csvs=5, manifests=10, figs=3, findings=3)

        n = urc.update_readme()
        assert n == 1  # headline only (FINDINGS drift)
        text = (tmp_path / "README.md").read_text()
        assert "7 FINDINGS sections" in text

    def test_fix_all_three_drift(self, tmp_path, monkeypatch):
        """All counts drift at once — headline repaired, dir-layout manifest + raw too."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, csvs=10, manifests=15, figures=7)
        _write_readme(tmp_path, csvs=3, manifests=5, figs=2)

        n = urc.update_readme()
        assert n == 3  # headline + manifests dir-layout + raw dir-layout
        text = (tmp_path / "README.md").read_text()
        assert "10 CSVs" in text
        assert "15 provenance manifests" in text
        assert "7 generated figures/tables" in text

    def test_dir_layout_independent(self, tmp_path, monkeypatch):
        """When only the dir-layout manifest count is wrong, it gets fixed."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, csvs=5, manifests=10, figures=3)
        # Write README with correct headline but stale dir-layout
        (tmp_path / "README.md").write_text(
            README_TEMPLATE.format(csvs=5, manifests=10, figs=3, findings=3).replace(
                "10 provenance manifests (git SHA", "5 provenance manifests (git SHA"
            )
        )

        n = urc.update_readme()
        assert n == 1  # dir-layout only
        text = (tmp_path / "README.md").read_text()
        # headline unchanged (still 10)
        assert text.count("10 provenance manifests") == 2  # headline + dir-layout


# ---------------------------------------------------------------------------
# update_readme — dry-run
# ---------------------------------------------------------------------------


class TestUpdateReadmeDryRun:
    def test_dry_run_no_write(self, tmp_path, monkeypatch):
        """Dry-run reports changes but does not modify README."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, csvs=5, manifests=20, figures=3)
        _write_readme(tmp_path, csvs=5, manifests=10, figs=3)

        original = (tmp_path / "README.md").read_text()
        n = urc.update_readme(dry_run=True)
        assert n == 2  # changes detected
        # File unchanged
        assert (tmp_path / "README.md").read_text() == original


# ---------------------------------------------------------------------------
# update_readme — edge cases
# ---------------------------------------------------------------------------


class TestUpdateReadmeEdgeCases:
    def test_no_readme(self, tmp_path, monkeypatch):
        """Missing README.md raises FileNotFoundError (fail-fast)."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path)
        with pytest.raises(FileNotFoundError):
            urc.update_readme()

    def test_readme_without_pattern(self, tmp_path, monkeypatch):
        """README without the expected patterns results in no changes."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, csvs=5, manifests=10, figures=3)
        (tmp_path / "README.md").write_text("# Minimal README\nNo counts here.\n")

        # Should not crash, should report 0 changes
        n = urc.update_readme()
        assert n == 0

    def test_subdirectory_manifests_counted(self, tmp_path, monkeypatch):
        """Manifests in subdirectories are counted recursively."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, csvs=3, manifests=5, figures=2)
        # Add manifests in a subdirectory (like the real repo's structure)
        sub = tmp_path / "results" / "manifests" / "ablation"
        sub.mkdir()
        for i in range(4):
            (sub / f"extra-{i}.json").write_text("{}")

        _write_readme(tmp_path, csvs=3, manifests=5, figs=2)
        n = urc.update_readme()
        assert n == 2  # headline + dir-layout
        text = (tmp_path / "README.md").read_text()
        assert "9 provenance manifests" in text

    def test_figures_readme_not_counted(self, tmp_path, monkeypatch):
        """results/figures/README.md is excluded from figure count."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, csvs=3, manifests=5, figures=4)
        _write_readme(tmp_path, csvs=3, manifests=5, figs=4)
        n = urc.update_readme()
        assert n == 0  # already correct, README.md excluded


# ---------------------------------------------------------------------------
# _count_files — git ls-files success path
# ---------------------------------------------------------------------------


class TestCountFilesGitLsFiles:
    """Cover the git ls-files success path (line 47) and suffix filter (line 56)."""

    def test_git_ls_files_success_path(self, tmp_path, monkeypatch):
        """When git ls-files succeeds, its parsed output is used."""
        import subprocess as sp

        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        mock_result = sp.CompletedProcess(
            args=["git", "ls-files"],
            returncode=0,
            stdout="results/raw/a.csv\nresults/raw/b.csv\n",
        )
        monkeypatch.setattr(sp, "run", lambda *a, **kw: mock_result)
        assert urc._count_files("results/raw", suffix=".csv") == 2

    def test_suffix_filter_skips_non_matching(self, tmp_path, monkeypatch):
        """Files not matching the suffix are skipped."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        raw = tmp_path / "results" / "raw"
        raw.mkdir(parents=True)
        (raw / "a.csv").write_text("data")
        (raw / "b.txt").write_text("text")
        (raw / "c.csv").write_text("data")
        assert urc._count_files("results/raw", suffix=".csv") == 2


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    """Test the main() CLI entry point."""

    def test_main_default_no_dry_run(self, monkeypatch):
        """main() calls update_readme without dry_run."""
        called = {}

        def mock_update(dry_run=False):
            called["dry_run"] = dry_run
            return 0

        monkeypatch.setattr(urc, "update_readme", mock_update)
        monkeypatch.setattr("sys.argv", ["update_readme_counts.py"])
        assert urc.main() == 0
        assert called["dry_run"] is False

    def test_main_dry_run_flag(self, monkeypatch):
        """main() passes --dry-run to update_readme."""
        called = {}

        def mock_update(dry_run=False):
            called["dry_run"] = dry_run
            return 0

        monkeypatch.setattr(urc, "update_readme", mock_update)
        monkeypatch.setattr("sys.argv", ["update_readme_counts.py", "--dry-run"])
        assert urc.main() == 0
        assert called["dry_run"] is True


# ---------------------------------------------------------------------------
# update_test_count()
# ---------------------------------------------------------------------------


class TestUpdateTestCount:
    """Tests for the --test-count flag in update_readme_counts.py."""

    @staticmethod
    def _make_readme(tmp_path):
        """Create minimal README.md and DEVPOST_SUBMISSION.md with test counts."""
        root = tmp_path / "repo"
        root.mkdir()
        docs = root / "docs"
        docs.mkdir()
        (root / "README.md").write_text(
            "| CI: lint + unit tests (100 passed locally, 1 skipped — pending) |\n"
        )
        (docs / "DEVPOST_SUBMISSION.md").write_text(
            "- 100 unit tests (1 skip) covering correctness\n"
        )
        return root

    def test_matching_counts_no_changes(self, tmp_path, monkeypatch):
        """Correct test count produces no changes."""
        root = self._make_readme(tmp_path)
        monkeypatch.setattr(urc, "REPO_ROOT", str(root))
        changes = urc.update_test_count(100)
        assert changes == 0

    def test_stale_readme_test_count_fixed(self, tmp_path, monkeypatch):
        """Stale README.md test count is updated."""
        root = self._make_readme(tmp_path)
        monkeypatch.setattr(urc, "REPO_ROOT", str(root))
        changes = urc.update_test_count(200)
        assert changes >= 1
        readme = (root / "README.md").read_text()
        assert "200 passed locally" in readme

    def test_stale_devpost_test_count_fixed(self, tmp_path, monkeypatch):
        """Stale DEVPOST_SUBMISSION.md test count is updated."""
        root = self._make_readme(tmp_path)
        monkeypatch.setattr(urc, "REPO_ROOT", str(root))
        changes = urc.update_test_count(200)
        assert changes >= 1
        devpost = (root / "docs" / "DEVPOST_SUBMISSION.md").read_text()
        assert "200 unit tests" in devpost

    def test_both_files_updated(self, tmp_path, monkeypatch):
        """Both files are updated in one call."""
        root = self._make_readme(tmp_path)
        monkeypatch.setattr(urc, "REPO_ROOT", str(root))
        changes = urc.update_test_count(300)
        assert changes == 2

    def test_dry_run_no_write(self, tmp_path, monkeypatch):
        """Dry-run does not modify files."""
        root = self._make_readme(tmp_path)
        monkeypatch.setattr(urc, "REPO_ROOT", str(root))
        changes = urc.update_test_count(300, dry_run=True)
        assert changes == 2
        # Files unchanged
        assert "100 passed locally" in (root / "README.md").read_text()
        assert "100 unit tests" in (root / "docs" / "DEVPOST_SUBMISSION.md").read_text()

    def test_main_with_test_count_flag(self, monkeypatch):
        """main() calls update_test_count when --test-count is provided."""
        called = {}

        def mock_test_count(test_count, dry_run=False):
            called["test_count"] = test_count
            called["dry_run"] = dry_run
            return 0

        monkeypatch.setattr(urc, "update_readme", lambda **kw: 0)
        monkeypatch.setattr(urc, "update_test_count", mock_test_count)
        monkeypatch.setattr("sys.argv", ["update_readme_counts.py", "--test-count", "500"])
        assert urc.main() == 0
        assert called["test_count"] == 500


# ---------------------------------------------------------------------------
# _count_findings_lines
# ---------------------------------------------------------------------------


class TestCountFindingsLines:
    """Tests for _count_findings_lines — wc -l equivalent."""

    def test_count_lines(self, tmp_path, monkeypatch):
        """Counts all lines in FINDINGS.md, not just ## headers."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, findings=3)
        # _make_repo creates "## Section N\nContent.\n\n" = 3 lines per section
        assert urc._count_findings_lines() == 9

    def test_no_file(self, tmp_path, monkeypatch):
        """Returns 0 when docs/FINDINGS.md does not exist."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        assert urc._count_findings_lines() == 0

    def test_single_line_file(self, tmp_path, monkeypatch):
        """A file with no trailing newline is still counted correctly."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        (tmp_path / "docs").mkdir(parents=True)
        (tmp_path / "docs" / "FINDINGS.md").write_text("## Only\nBody")
        assert urc._count_findings_lines() == 2


# ---------------------------------------------------------------------------
# update_readme — "Operator analysis findings" line repair
# ---------------------------------------------------------------------------

README_WITH_OPERATOR = """\
# Test Repo

**Operator analysis findings** ([`docs/FINDINGS.md`](./docs/FINDINGS.md), {findings} sections):

> **Results so far:** {csvs} CSVs from the device fleet, {manifests} provenance manifests, {figs} generated figures/tables, {findings2} FINDINGS sections.
"""


class TestOperatorFindingsRepair:
    """Tests for the 'Operator analysis findings' line auto-fix in update_readme."""

    def test_operator_findings_drift(self, tmp_path, monkeypatch):
        """Stale 'Operator analysis findings' section count is repaired."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, csvs=5, manifests=10, figures=3, findings=7)
        # Write README with stale count in both locations
        (tmp_path / "README.md").write_text(
            README_WITH_OPERATOR.format(csvs=5, manifests=10, figs=3, findings=3, findings2=3)
        )
        n = urc.update_readme()
        assert n == 2  # headline FINDINGS + operator-findings
        text = (tmp_path / "README.md").read_text()
        assert "55 sections" not in text  # exact count depends on _make_repo
        assert "7 sections" in text

    def test_operator_findings_only_drift(self, tmp_path, monkeypatch):
        """When headline FINDINGS matches but operator line drifts, only operator is fixed."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, csvs=5, manifests=10, figures=3, findings=7)
        # Headline correct, operator line stale
        (tmp_path / "README.md").write_text(
            README_WITH_OPERATOR.format(csvs=5, manifests=10, figs=3, findings=3, findings2=7)
        )
        n = urc.update_readme()
        assert n == 1  # only operator-findings
        text = (tmp_path / "README.md").read_text()
        assert "7 sections" in text

    def test_no_operator_line_no_crash(self, tmp_path, monkeypatch):
        """README without the operator-findings line does not crash."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, csvs=5, manifests=10, figures=3, findings=3)
        _write_readme(tmp_path, csvs=5, manifests=10, figs=3, findings=3)
        n = urc.update_readme()
        assert n == 0  # nothing wrong, and no operator line to check


# ---------------------------------------------------------------------------
# update_findings_doc_counts
# ---------------------------------------------------------------------------

DEVPOST_SUBMISSION_TEMPLATE = """\
# Devpost Submission

- **Findings ({sections} sections, {lines} lines):** [`docs/FINDINGS.md`](../docs/FINDINGS.md)
"""

DEVPOST_WRITEUP_TEMPLATE = """\
# Devpost Writeup

| [`docs/FINDINGS.md`](./FINDINGS.md) | All measured results with analysis ({sections} sections) |
"""


def _write_devpost_files(tmp_path, sections, lines, writeup_sections=None):
    """Write DEVPOST_SUBMISSION.md and DEVPOST_WRITEUP.md with given counts."""
    if writeup_sections is None:
        writeup_sections = sections
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "DEVPOST_SUBMISSION.md").write_text(
        DEVPOST_SUBMISSION_TEMPLATE.format(sections=sections, lines=lines)
    )
    (docs / "DEVPOST_WRITEUP.md").write_text(
        DEVPOST_WRITEUP_TEMPLATE.format(sections=writeup_sections)
    )


class TestUpdateFindingsDocCounts:
    """Tests for update_findings_doc_counts — DEVPOST FINDINGS auto-repair."""

    def test_devpost_submission_drift(self, tmp_path, monkeypatch):
        """Stale DEVPOST_SUBMISSION sections+lines are repaired."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, findings=7)  # 7 sections = 21 lines
        _write_devpost_files(tmp_path, sections=3, lines=100)
        n = urc.update_findings_doc_counts()
        assert n == 2  # submission + writeup
        text = (tmp_path / "docs" / "DEVPOST_SUBMISSION.md").read_text()
        assert "7 sections" in text
        assert "21 lines" in text

    def test_devpost_writeup_drift(self, tmp_path, monkeypatch):
        """Stale DEVPOST_WRITEUP section count is repaired independently."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, findings=7)
        _write_devpost_files(tmp_path, sections=7, lines=21, writeup_sections=3)
        n = urc.update_findings_doc_counts()
        assert n == 1  # only writeup
        text = (tmp_path / "docs" / "DEVPOST_WRITEUP.md").read_text()
        assert "7 sections" in text

    def test_already_correct(self, tmp_path, monkeypatch):
        """When all counts match, no changes are made."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, findings=7)
        _write_devpost_files(tmp_path, sections=7, lines=21)
        n = urc.update_findings_doc_counts()
        assert n == 0

    def test_dry_run_no_write(self, tmp_path, monkeypatch):
        """Dry-run reports changes but does not modify files."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, findings=7)
        _write_devpost_files(tmp_path, sections=3, lines=100)
        original_sub = (tmp_path / "docs" / "DEVPOST_SUBMISSION.md").read_text()
        original_wri = (tmp_path / "docs" / "DEVPOST_WRITEUP.md").read_text()
        n = urc.update_findings_doc_counts(dry_run=True)
        assert n == 2
        # Files unchanged
        assert (tmp_path / "docs" / "DEVPOST_SUBMISSION.md").read_text() == original_sub
        assert (tmp_path / "docs" / "DEVPOST_WRITEUP.md").read_text() == original_wri

    def test_missing_files_no_crash(self, tmp_path, monkeypatch):
        """Missing DEVPOST files result in 0 changes, not an error."""
        monkeypatch.setattr(urc, "REPO_ROOT", str(tmp_path))
        _make_repo(tmp_path, findings=7)
        # Don't create DEVPOST files
        n = urc.update_findings_doc_counts()
        assert n == 0


# ---------------------------------------------------------------------------
# main() — calls update_findings_doc_counts
# ---------------------------------------------------------------------------


class TestMainFindingsDocCounts:
    """Test that main() calls update_findings_doc_counts."""

    def test_main_calls_findings_doc_counts(self, monkeypatch):
        called = {}

        def mock_findings(dry_run=False):
            called["dry_run"] = dry_run
            return 0

        monkeypatch.setattr(urc, "update_readme", lambda **kw: 0)
        monkeypatch.setattr(urc, "update_findings_doc_counts", mock_findings)
        monkeypatch.setattr("sys.argv", ["update_readme_counts.py"])
        assert urc.main() == 0
        assert called["dry_run"] is False
