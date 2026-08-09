# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0
"""Validate manifest git SHA provenance.

Every manifest's git.sha must either resolve to a reachable commit, or
have a git.sha_resolved field pointing to the equivalent valid commit
(for SHAs lost during branch rebases before PR merges).

Bead ob-mrd.18.
"""

import json
import os
import subprocess

import pytest

MANIFESTS_DIR = os.path.join("results", "manifests")


def _git_cat_file(sha: str) -> bool:
    """Return True if *sha* resolves to a valid git object."""
    r = subprocess.run(
        ["git", "cat-file", "-t", sha],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def _load_named_manifests():
    """Load all named (non-generic-retro) manifests."""
    manifests = []
    if not os.path.isdir(MANIFESTS_DIR):
        return manifests
    for fn in sorted(os.listdir(MANIFESTS_DIR)):
        if not fn.endswith(".json"):
            continue
        if fn.startswith("generic_aarch64_"):
            continue
        path = os.path.join(MANIFESTS_DIR, fn)
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        manifests.append((fn, data))
    return manifests


class TestManifestShaProvenance:
    """Every named manifest must have a resolvable git SHA chain."""

    @pytest.fixture
    def named_manifests(self):
        return _load_named_manifests()

    def test_named_manifests_exist(self, named_manifests):
        """Sanity: we have named manifests to check."""
        assert len(named_manifests) > 0, "No named manifests found"

    def test_all_manifests_have_sha_field(self, named_manifests):
        """Every manifest must record a git.sha."""
        missing = []
        for fn, data in named_manifests:
            sha = data.get("git", {}).get("sha", "")
            if not sha:
                missing.append(fn)
        assert not missing, f"Manifests missing git.sha: {missing}"

    def test_stale_shas_have_resolved_equivalent(self, named_manifests):
        """If git.sha is unreachable, git.sha_resolved must be present and valid.

        SHAs from pre-rebase branches are annotated with sha_resolved pointing
        to the equivalent merged commit. This test verifies that chain.
        """
        unresolved = []
        for fn, data in named_manifests:
            git_info = data.get("git", {})
            sha = git_info.get("sha", "")
            if not sha or len(sha) < 8:
                continue

            # If the original SHA is valid, no annotation needed
            if _git_cat_file(sha):
                continue

            # Stale SHA — must have sha_resolved
            resolved = git_info.get("sha_resolved", "")
            if not resolved:
                unresolved.append(f"{fn}: sha {sha[:12]} is unreachable and has no sha_resolved")
                continue

            # Verify sha_resolved is itself valid
            if not _git_cat_file(resolved):
                unresolved.append(
                    f"{fn}: sha_resolved {resolved[:12]} is also unreachable"
                )

        assert not unresolved, (
            "Manifests with unresolvable SHA chains:\n  "
            + "\n  ".join(unresolved)
        )

    def test_sha_note_documented_for_stale_shas(self, named_manifests):
        """Stale SHAs must have a human-readable note explaining the gap."""
        missing_notes = []
        for fn, data in named_manifests:
            git_info = data.get("git", {})
            sha = git_info.get("sha", "")
            if not sha or len(sha) < 8:
                continue

            if _git_cat_file(sha):
                continue

            note = git_info.get("sha_note", "")
            if not note:
                missing_notes.append(fn)

        assert not missing_notes, (
            f"Stale SHA manifests missing sha_note: {missing_notes}"
        )

    def test_all_resolved_shas_valid(self, named_manifests):
        """Every sha_resolved field must point to a valid git object."""
        invalid = []
        for fn, data in named_manifests:
            resolved = data.get("git", {}).get("sha_resolved", "")
            if not resolved:
                continue
            if not _git_cat_file(resolved):
                invalid.append(f"{fn}: {resolved[:12]}")

        assert not invalid, f"Invalid sha_resolved values: {invalid}"
