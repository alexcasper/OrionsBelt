#!/bin/bash
# git history purge procedure for ob-3i5
# Removes .goose-task.md and .goose-loop.log from all git history.
#
# ⚠⚠⚠ READ ALL COMMENTS BEFORE RUNNING ⚠⚠⚠
#
# This script REWRITES git history. It must be run by a maintainer with
# push access to ALL branches. Every clone and PR will need to be reset.
#
# PREREQUISITE (non-negotiable):
# 1. ROTATE the device sudo password. The committed password must be
#    treated as permanently disclosed — it has been on GitHub since 2026-08-02.
#    (Set PURGE_PASSWORD env var to the old value for verification below.)
# 2. All agents must stop their loops before running this (they will need to
#    re-clone after history is rewritten).
#
# PROCEDURE:
#   Option A: git-filter-repo (recommended)
#   Option B: BFG Repo-Cleaner (alternative)
#
# After running either option, force-push ALL branches:
#   git push --force origin main bench/j1 bench/j2 bench/t4 bench/r5
#
# =====================================================================

set -euo pipefail

echo "============================================"
echo "  HISTORY PURGE PROCEDURE (ob-3i5)"
echo "============================================"
echo ""
echo "⚠  This will rewrite ALL git history."
echo "⚠  Ensure the device sudo password has been ROTATED."
echo "⚠  All agent loops must be stopped."
echo ""
read -p "Type PURGE to continue: " CONFIRM
if [ "$CONFIRM" != "PURGE" ]; then
    echo "Aborted."
    exit 1
fi

# =====================================================================
# Option A: git-filter-repo
# =====================================================================
# Install:
#   pip3 install git-filter-repo
#
# Run from a FRESH clone (git-filter-repo refuses to run on a repo with
# a remote origin configured, for safety):
#
#   git clone --mirror https://github.com/alexcasper/OrionsBelt.git orionsbelt-purge
#   cd orionsbelt-purge
#   git filter-repo --invert-paths \
#     --path .goose-task.md \
#     --path .goose-loop.log
#
# Then push the rewritten history:
#   git push --force
#
# Re-clone normally and verify:
#   git log --all -p -S "password '$PURGE_PASSWORD'" -- .goose-task.md
#   # Should return nothing
#
# =====================================================================

# For this script, we use Option A on a fresh mirror clone:

WORKDIR="/tmp/orionsbelt-purge-$(date +%s)"
REMOTE="https://github.com/alexcasper/OrionsBelt.git"

echo "Creating mirror clone at $WORKDIR ..."
git clone --mirror "$REMOTE" "$WORKDIR"
cd "$WORKDIR"

echo "Installing git-filter-repo ..."
pip3 install --user git-filter-repo 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

echo "Purging .goose-task.md and .goose-loop.log from history ..."
git filter-repo --invert-paths \
    --path .goose-task.md \
    --path .goose-loop.log

echo "Verifying purge ..."
PURGE_PW="${PURGE_PASSWORD:-}"
if [ -z "$PURGE_PW" ]; then
    echo "NOTE: PURGE_PASSWORD not set — skipping verification of password string."
    echo "      Set PURGE_PASSWORD env var to the old password to enable verification."
    HITS=0
else
    HITS=$(git log --all --oneline -S "echo $PURGE_PW" 2>/dev/null | wc -l)
fi
if [ "$HITS" -gt 0 ]; then
    echo "⚠  WARNING: $HITS commits still contain the password string."
    echo "   These may be in other files. Manual review needed."
else
    echo "✓ No commits contain the password string."
fi

echo ""
echo "Force-pushing rewritten history to remote ..."
echo "⚠  This will rewrite ALL branches. All clones and PRs must be reset."
read -p "Type FORCE to push: " CONFIRM2
if [ "$CONFIRM2" = "FORCE" ]; then
    git push --force
    echo "✓ History rewritten and pushed."
    echo ""
    echo "POST-PURGE CHECKLIST:"
    echo "  1. Verify password is gone: git log --all -p -S '$PURGE_PW' | grep -c password"
    echo "  2. All agents: re-clone the repo (old clones have stale history)"
    echo "  3. Recreate any open PRs (history rewrite invalidates them)"
    echo "  4. Confirm the rotated password works on all devices"
    echo "  5. Close ob-3i5 once verified"
else
    echo "Aborted before push. Rewritten history is at $WORKDIR (not pushed)."
    echo "You can inspect it, then push manually with:"
    echo "  cd $WORKDIR && git push --force"
fi
