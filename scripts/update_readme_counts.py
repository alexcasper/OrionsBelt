#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Auto-update the "Results so far" counts in README.md.

This script exists because the manifest/CSV/figure counts in README.md
drift every time a teammate adds results without updating the docs.
Running this script is a single-command fix; ``validate_results.py``
already *detects* the drift (via ``check_readme_counts``), but this
script *repairs* it.

Usage::

    python3 scripts/update_readme_counts.py [--dry-run]

Exits 0 if README was already up to date (or was corrected), non-zero
on error.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _count_files(dirpath: str, suffix: str | None = None, exclude_name: str | None = None) -> int:
    """Count tracked files — mirrors the git ls-files logic in validate_results.

    Uses ``git ls-files`` to count only git-tracked files (ignoring gitignored
    artifacts like auto-generated ablation manifests). Falls back to os.walk
    when git is unavailable (e.g. in unit tests with temp directories).
    """
    try:
        import subprocess

        result = subprocess.run(  # noqa: S603, S607
            ["git", "ls-files", "--", f"{dirpath}/"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
        files = [os.path.basename(f) for f in result.stdout.strip().splitlines() if f]
    except (subprocess.CalledProcessError, FileNotFoundError):
        files = []
        for _root, _dirs, dirfiles in os.walk(os.path.join(REPO_ROOT, dirpath)):
            files.extend(dirfiles)

    total = 0
    for fname in files:
        if suffix and not fname.endswith(suffix):
            continue
        if exclude_name and fname == exclude_name:
            continue
        total += 1
    return total


# Regex patterns for the two locations we need to patch.
# "Results so far" headline.
RE_HEADLINE = re.compile(
    r"(Results so far:\*\*\s*)(\d+)(\s*CSVs from the device fleet,\s*)"
    r"(\d+)(\s*provenance manifests,\s*)"
    r"(\d+)(\s*generated figures/tables.*)"
)
# Directory-layout manifest count.
RE_DIRLAYOUT = re.compile(r"(manifests/\s*<-\s*)(\d+)(\s*provenance manifests.*)")


def update_readme(dry_run: bool = False) -> int:
    """Update README.md counts. Returns number of changes made."""
    readme_path = os.path.join(REPO_ROOT, "README.md")
    with open(readme_path) as f:
        text = f.read()

    actual_csvs = _count_files("results/raw", suffix=".csv")
    actual_manifests = _count_files("results/manifests", suffix=".json")
    actual_figures = _count_files("results/figures", exclude_name="README.md")

    changes = []

    # --- Headline line ---
    m = RE_HEADLINE.search(text)
    if m:
        old_csvs, old_manifests, old_figures = int(m.group(2)), int(m.group(4)), int(m.group(6))
        if (old_csvs, old_manifests, old_figures) != (
            actual_csvs,
            actual_manifests,
            actual_figures,
        ):
            replacement = (
                m.group(1)
                + str(actual_csvs)
                + m.group(3)
                + str(actual_manifests)
                + m.group(5)
                + str(actual_figures)
                + m.group(7)
            )
            text = text[: m.start()] + replacement + text[m.end() :]
            changes.append(
                f"headline: CSVs {old_csvs}→{actual_csvs}, "
                f"manifests {old_manifests}→{actual_manifests}, "
                f"figures {old_figures}→{actual_figures}"
            )

    # --- Directory layout manifest count ---
    m = RE_DIRLAYOUT.search(text)
    if m:
        old_count = int(m.group(2))
        if old_count != actual_manifests:
            replacement = m.group(1) + str(actual_manifests) + m.group(3)
            text = text[: m.start()] + replacement + text[m.end() :]
            changes.append(f"dir-layout: manifests {old_count}→{actual_manifests}")

    if changes:
        for c in changes:
            print(f"  updated: {c}")
        if not dry_run:
            with open(readme_path, "w") as f:
                f.write(text)
            print(f"README.md updated ({len(changes)} change(s)).")
        else:
            print("[dry-run] README.md not modified.")
    else:
        print("README.md counts already correct — no changes needed.")

    return len(changes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-update README.md result counts.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    args = parser.parse_args()

    update_readme(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
