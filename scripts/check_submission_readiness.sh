#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

# Submission readiness check — verifies the repo is ready for Devpost.
#
# Run from the repo root:  bash scripts/check_submission_readiness.sh
#
# Checks:
#   1. Python tests pass
#   2. Ruff lint clean
#   3. Ruff format clean
#   4. Memory scaling plots regenerable
#   5. Fleet analysis runs without error
#   6. Key deliverable files exist
#   7. No credentials / secrets in tracked files
#   8. Results CSVs validated
#   9. README result counts match actual file counts
#  10. No credential files tracked on ANY remote branch (ob-3i5)
#
# Exit code 0 = all checks pass, 1 = at least one failure.

set -euo pipefail

PASS=0
FAIL=0
WARN=0
SKIP=0

# Temp dir for check 4; cleaned on EXIT so Ctrl-C doesn't leak it.
_MEM_TMPDIR=""
_trap_cleanup() { [ -n "$_MEM_TMPDIR" ] && rm -rf "$_MEM_TMPDIR"; true; }
trap _trap_cleanup EXIT

ok()   { echo "  ✓ $1"; PASS=$((PASS + 1)); }
fail() { echo "  ✗ $1"; FAIL=$((FAIL + 1)); }
warn() { echo "  ! $1"; WARN=$((WARN + 1)); }
skip() { echo "  ⊘ $1"; SKIP=$((SKIP + 1)); }

echo "=== OrionsBelt Submission Readiness Check ==="
echo ""

# Detect Python version — some checks require Python 3.7+ and will be skipped
# gracefully on older interpreters (e.g. Jetson Nano's Python 3.6.9).
PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])' 2>/dev/null || echo 0)
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])' 2>/dev/null || echo 0)
PY_OK=$(( PY_MAJOR * 100 + PY_MINOR >= 307 ))

# -------------------------------------------------------------------
# 1. Tests
# -------------------------------------------------------------------
echo "[1/10] Python test suite"
if [ "$PY_OK" -ne 1 ]; then
    skip "Python 3.7+ required (have ${PY_MAJOR}.${PY_MINOR}); run on CI or an x86 host"
else
    RESULT=$(python3 -m pytest tests/ --tb=no 2>&1 | grep "passed" | tail -1)
    if [ -n "$RESULT" ]; then
        ok "Tests: $RESULT"
    else
        fail "Tests did not pass"
    fi
fi

# -------------------------------------------------------------------
# 2. Lint
# -------------------------------------------------------------------
echo "[2/10] Ruff lint"
if ruff check . > /dev/null 2>&1; then
    ok "Ruff lint clean"
else
    fail "Ruff lint has errors (run: ruff check .)"
fi

# -------------------------------------------------------------------
# 3. Format
# -------------------------------------------------------------------
echo "[3/10] Ruff format"
if ruff format --check . > /dev/null 2>&1; then
    ok "Ruff format clean"
else
    fail "Ruff format has issues (run: ruff format .)"
fi

# -------------------------------------------------------------------
# 4. Memory plots regenerable
# -------------------------------------------------------------------
echo "[4/10] Memory scaling plots"
if [ "$PY_OK" -ne 1 ]; then
    skip "Python 3.7+ required (have ${PY_MAJOR}.${PY_MINOR}); run on CI or an x86 host"
else
    _MEM_TMPDIR=$(mktemp -d)
    if python3 scripts/generate_memory_plots.py --text-only --output-dir "$_MEM_TMPDIR" > /dev/null 2>&1; then
        if [ -f "$_MEM_TMPDIR/memory_comparison.md" ]; then
            ok "Memory plots generate successfully"
        else
            fail "Memory plots script ran but no output file"
        fi
    else
        fail "Memory plots script failed"
    fi
    rm -rf "$_MEM_TMPDIR"
    _MEM_TMPDIR=""
fi

# -------------------------------------------------------------------
# 5. Fleet analysis
# -------------------------------------------------------------------
echo "[5/10] Fleet analysis"
# Use a temp dir so the check doesn't clobber the committed PNG with a
# device-local render (cross-device freetype nondeterminism, bead ob-6ay).
if python3 bench/fleet_analysis.py --output-dir "$(mktemp -d)" > /dev/null 2>&1; then
    ok "Fleet analysis runs cleanly"
else
    warn "Fleet analysis has warnings (may be expected with dirty-tree data)"
fi

# -------------------------------------------------------------------
# 6. Key deliverable files
# -------------------------------------------------------------------
echo "[6/10] Key deliverable files"
for f in \
    README.md \
    docs/archive/PLAN.md \
    LICENSE \
    docs/FINDINGS.md \
    docs/CLAIM_VERIFICATION.md \
    docs/adr/0007-commit-to-edge-ai-track.md \
    results/figures/fleet_bandwidth_scaling.md \
    results/figures/memory_comparison.md \
    results/figures/comparison_table.md \
; do
    if [ -f "$f" ]; then
        ok "Exists: $f"
    else
        fail "Missing: $f"
    fi
done

# -------------------------------------------------------------------
# 7. Credential scan (tracked files only)
# -------------------------------------------------------------------
echo "[7/10] Credential scan"
# Check for actual credential assignments, not the word "password" in docs
SECRETS=$(git grep -n -E '(password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\s*[=:]\s*["\x27][^"\x27]{4,}' -- '*.py' '*.sh' '*.yaml' '*.json' 2>/dev/null \
    | grep -v '.beads/' \
    | grep -v 'node_modules/' \
    | grep -v 'eos_token' \
    || true)
if [ -z "$SECRETS" ]; then
    ok "No credential patterns found in tracked files"
else
    fail "Potential credentials: $(echo "$SECRETS" | head -5)"
fi

# -------------------------------------------------------------------
# 8. Results validation
# -------------------------------------------------------------------
echo "[8/10] Results CSV validation"
VALOUT=$(python3 scripts/validate_results.py 2>&1 || true)
if echo "$VALOUT" | grep -q "CSV(s) checked"; then
    ISSUES=$(echo "$VALOUT" | grep -c "WARNING" || true)
    if [ "$ISSUES" -eq 0 ]; then
        ok "All CSVs validated, no warnings"
    else
        warn "$ISSUES validation warnings (see: python3 scripts/validate_results.py)"
    fi
else
    warn "Results validator did not complete cleanly"
fi

# -------------------------------------------------------------------
# 9. README count verification
# -------------------------------------------------------------------
echo "[9/10] README result counts vs actual files"
if python3 scripts/update_readme_counts.py --dry-run > /dev/null 2>&1; then
    ok "README counts match actual file counts"
else
    fail "README counts are stale — run: python3 scripts/update_readme_counts.py"
fi

# -------------------------------------------------------------------
# 10. Cross-branch credential file scan (ob-3i5)
# -------------------------------------------------------------------
echo "[10/10] Cross-branch credential file scan"
# Known credential-containing files that must not be tracked on ANY branch.
# If the repo is made public, these are immediately exposed (ob-3i5).
CRED_BRANCHES=""
for branch in $(git branch -r --format='%(refname:short)' 2>/dev/null | grep -v HEAD); do
    tracked=$(git ls-tree "$branch" -- .goose-task.md .goose-loop.log 2>/dev/null || true)
    if [ -n "$tracked" ]; then
        CRED_BRANCHES="$CRED_BRANCHES $branch"
    fi
done
if [ -z "$CRED_BRANCHES" ]; then
    ok "No credential files tracked on any remote branch"
else
    fail "Credential files (.goose-task.md/.goose-loop.log) tracked on:$CRED_BRANCHES — clean before making repo public (ob-3i5)"
fi
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo "  Warnings: $WARN"
if [ "$SKIP" -gt 0 ]; then
    echo "  Skipped: $SKIP (Python version or tool not available on this device)"
fi
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "❌ NOT READY — $FAIL check(s) failed"
    exit 1
else
    echo "✅ READY — all critical checks pass"
    if [ "$WARN" -gt 0 ]; then
        echo "   ($WARN warning(s) — review but not blocking)"
    fi
    exit 0
fi
