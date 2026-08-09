#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

#
# purge_sudo_password.sh — Purge the device sudo password from git history.
#
# This script rewrites ALL branches to replace the leaked password string
# with '***REDACTED***'. It then force-pushes all branches.
#
# PREREQUISITES (human action):
#   1. ROTATE the device sudo password first — treat the old value as disclosed.
#   2. Ensure all PRs are merged or closed (history rewrite invalidates them).
#   3. Install git-filter-repo for the recommended path:
#        pip3 install git-filter-repo
#      Otherwise this script falls back to git filter-branch (slower).
#
# WARNING: This REWRITES git history and FORCE-PUSHES all branches.
#          Coordinate with the team before running.
#
# Usage:  bash scripts/purge_sudo_password.sh [--dry-run]
#
set -euo pipefail

DRY_RUN="${1:-}"

cd "$(git rev-parse --show-toplevel)"

echo "=== Sudo Password Purge ==="
echo "Repo: $(git remote get-url origin 2>/dev/null || echo 'no remote')"
echo "Branches: $(git branch --format='%(refname:short)' | tr '\n' ' ')"
echo ""

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "[DRY RUN] No changes will be made."
    echo ""
fi

# Read the password from an environment variable — never hardcode the
# credential in a tracked file (see ob-3i5, SECURITY_REMEDIATION.md §2).
# Usage:  PURGE_PASSWORD='<old-password>' bash scripts/purge_sudo_password.sh [--dry-run]
if [ -z "${PURGE_PASSWORD:-}" ]; then
    echo "ERROR: PURGE_PASSWORD env var is not set."
    echo "  Usage: PURGE_PASSWORD='<old-password>' bash $0 [--dry-run]"
    exit 1
fi
PATTERN="echo ${PURGE_PASSWORD} | sudo -S"
REPLACEMENT='echo ***REDACTED*** | sudo -S'

echo "Pattern to replace: '$PATTERN'"
echo "Replacement:        '$REPLACEMENT'"
echo ""

# Count affected commits
COUNT=$(git log --all -S "$PATTERN" --oneline 2>/dev/null | wc -l)
echo "Affected commits: $COUNT"
echo ""

if [ "$COUNT" -eq 0 ]; then
    echo "No commits found containing the pattern. Nothing to do."
    exit 0
fi

# Show affected commits
echo "Affected commits:"
git log --all -S "$PATTERN" --oneline 2>/dev/null
echo ""

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "[DRY RUN] Would rewrite history on all branches."
    echo "Run without --dry-run to execute."
    exit 0
fi

# --- Method 1: git-filter-repo (preferred) ---
if command -v git-filter-repo &>/dev/null; then
    echo "Using git-filter-repo (recommended)..."
    git filter-repo \
        --replace-text <(echo "$PATTERN==>$REPLACEMENT") \
        --force
    echo ""
    echo "History rewritten. Force-pushing all branches..."
    git push --force --all
    echo "Done."

# --- Method 2: git filter-branch (fallback) ---
else
    echo "git-filter-repo not found. Falling back to git filter-branch."
    echo "(Install with: pip3 install git-filter-repo for better performance)"
    echo ""

    # Get all branches
    BRANCHES=$(git branch --format='%(refname:short)')

    for BRANCH in $BRANCHES; do
        echo "Rewriting branch: $BRANCH"
        FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --force \
            --tree-filter "
                for f in .goose-task.md .goose-loop.log; do
                    if [ -f \"\$f\" ]; then
                        sed -i 's|$PATTERN|$REPLACEMENT|g' \"\$f\" 2>/dev/null || true
                    fi
                done
            " \
            -- "$BRANCH" 2>/dev/null || echo "  (skipped $BRANCH — may have no affected commits)"
    done

    echo ""
    echo "History rewritten. Force-pushing all branches..."
    git push --force --all
    echo "Done."
fi

echo ""
echo "=== POST-PURGE CHECKLIST ==="
echo "1. Verify no remaining matches: git log --all -S '$PATTERN' --oneline"
echo "2. Rotate the device sudo password on all Arm devices."
echo "3. Re-open any PRs that were invalidated by the history rewrite."
echo "4. Notify all collaborators to re-clone (or git pull --rebase)."
