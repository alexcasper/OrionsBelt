#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

# Fleet sweep — clean-tree, commit-matched benchmark run for ob-bf7.
#
# The fleet comparison is currently broken by cross-commit and dirty-tree
# variance (bead ob-bf7): same-code runs differ up to 4x, and every manifest
# except rk3588-t3.json records dirty=true. This script enforces the
# preconditions for a valid fleet sweep so every device operator follows the
# same protocol:
#
#   1. Clean working tree (dirty=false in manifest)
#   2. Governor set to performance (recorded)
#   3. Pre/post thermal snapshots
#   4. Correct binary for the detected ISA
#   5. taskset pinning on asymmetric boards (RK3588)
#   6. OMP_NUM_THREADS recorded alongside the CSV
#   7. Manifest captured from the same clean tree
#
# Usage:
#   ./scripts/fleet_sweep.sh                          # auto-detect, single-thread
#   ./scripts/fleet_sweep.sh --threads 4              # 4-core OpenMP run
#   ./scripts/fleet_sweep.sh --binary dist/bench_gdn_jetson_a57
#   ./scripts/fleet_sweep.sh --pin "4-7"              # pin to big cluster
#   ./scripts/fleet_sweep.sh --device jetson-j1       # override output name
#
# Output:
#   results/raw/<device>.csv          — benchmark CSV
#   results/manifests/<device>.json   — provenance manifest (clean tree enforced)
#   results/thermal_<device>_pre.txt  — pre-run thermal snapshot
#   results/thermal_<device>_post.txt — post-run thermal snapshot
#
# Run this on EACH fleet device at the SAME git commit. The clean-tree check
# ensures the manifest's SHA actually identifies the code that ran.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# --- defaults ---------------------------------------------------------------
DEVICE_NAME=""
BINARY=""
THREADS=1
PIN_CPUS=""
REPEATS=30
FORCE=0

# --- arg parsing ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)  DEVICE_NAME="$2"; shift 2 ;;
        --binary)  BINARY="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --pin)     PIN_CPUS="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

export OMP_NUM_THREADS="$THREADS"

# --- 1. clean tree check ----------------------------------------------------
# Exclude results/ and .beads/ — output data from prior runs, not source changes.
STATUS=$(git status --porcelain 2>/dev/null | grep -vE '^[ ?][M?] (results/|\.beads/)' || true)
if [ -n "$STATUS" ] && [ "$FORCE" -eq 0 ]; then
    echo "ERROR: working tree is dirty. A dirty tree means the manifest's git SHA" >&2
    echo "       does not identify the code that produced the numbers (bead ob-bf7)." >&2
    echo "" >&2
    echo "Commit or stash your changes first, or use --force to override:" >&2
    echo "  $STATUS" >&2
    exit 1
fi
if [ -n "$STATUS" ] && [ "$FORCE" -eq 1 ]; then
    echo "WARNING: --force used; manifest will record dirty=true (ob-bf7 sweep invalid)" >&2
fi

GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")
echo "=== Fleet Sweep (ob-bf7) ==="
echo "  Git SHA: $GIT_SHA"
echo "  Threads: $THREADS"
echo ""

# --- 2. device detection ----------------------------------------------------
ARCH=$(uname -m 2>/dev/null || echo "unknown")
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
    echo "WARNING: not running on aarch64 ($ARCH) — this is not a fleet device" >&2
fi

# Auto-detect device name from hostname if not provided
if [ -z "$DEVICE_NAME" ]; then
    HOST=$(hostname 2>/dev/null || echo "unknown")
    # Normalise common hostname patterns to fleet device IDs
    case "$HOST" in
        j1|jetson-j1)  DEVICE_NAME="jetson-j1" ;;
        j2|jetson-j2)  DEVICE_NAME="jetson-j2" ;;
        r5|pi5-r5)     DEVICE_NAME="pi5-r5" ;;
        t3|rk3588-t3)  DEVICE_NAME="rk3588-t3" ;;
        t4|rk3588-t4)  DEVICE_NAME="rk3588-t4" ;;
        *)             DEVICE_NAME="$HOST" ;;
    esac
fi

# Auto-detect binary if not provided
if [ -z "$BINARY" ]; then
    if [ -x "scripts/detect_isa.sh" ]; then
        RECOMMENDED=$(scripts/detect_isa.sh --binary 2>/dev/null || echo "armv8a")
    else
        RECOMMENDED="armv8a"
    fi
    BINARY="dist/bench_gdn_${RECOMMENDED}"
fi

if [ ! -x "$BINARY" ]; then
    echo "ERROR: benchmark binary not found or not executable: $BINARY" >&2
    echo "Build first: ./scripts/build_device_bench.sh" >&2
    exit 1
fi

echo "  Device:  $DEVICE_NAME"
echo "  Binary:  $BINARY"
echo ""

# --- 3. governor ------------------------------------------------------------
GOV_SET=0
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    CURRENT_GOV=$(cat "$c" 2>/dev/null || echo "?")
    if [ "$CURRENT_GOV" != "performance" ]; then
        if echo performance | sudo tee "$c" >/dev/null 2>&1; then
            GOV_SET=1
        else
            echo "WARNING: cannot set governor on $c (current: $CURRENT_GOV)" >&2
            echo "         Set it manually: echo performance | sudo tee \"$c\"" >&2
        fi
    fi
done

if [ "$GOV_SET" -eq 1 ]; then
    echo "  Governor: set to performance"
else
    # Verify all cores are already at performance
    ALL_PERF=1
    for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        CURRENT_GOV=$(cat "$c" 2>/dev/null || echo "?")
        if [ "$CURRENT_GOV" != "performance" ]; then
            ALL_PERF=0
            echo "  Governor: $CURRENT_GOV on $(basename $(dirname "$c")) (NOT performance — numbers will be low)"
        fi
    done
    [ "$ALL_PERF" -eq 1 ] && echo "  Governor: performance (already set)"
fi
echo ""

# --- 4. pre-run thermal snapshot -------------------------------------------
THERMAL_PRE="results/thermal_${DEVICE_NAME}_pre.txt"
mkdir -p results/raw results/manifests
{
    echo "# Pre-run thermal snapshot"
    echo "# Device: $DEVICE_NAME"
    echo "# Git SHA: $GIT_SHA"
    echo "# Timestamp: $(date -u +'Y-%m-%dT%H:%M:%SZ' 2>/dev/null || echo '?')"
    echo "# OMP_NUM_THREADS: $THREADS"
    echo ""
    for z in /sys/class/thermal/thermal_zone*; do
        ZONE_TYPE=$(cat "$z/type" 2>/dev/null || echo "?")
        ZONE_TEMP=$(cat "$z/temp" 2>/dev/null || echo "?")
        echo "$(basename "$z") $ZONE_TYPE ${ZONE_TEMP}"
    done
} > "$THERMAL_PRE"
echo "  Pre-run thermals: $THERMAL_PRE"

# --- 5. run benchmark -------------------------------------------------------
echo ""
echo "=== Running benchmark ==="
PIN_CMD=""
if [ -n "$PIN_CPUS" ]; then
    PIN_CMD="taskset -c $PIN_CPUS"
    echo "  Pinning to CPUs: $PIN_CPUS"
fi

echo "  Command: $PIN_CMD $BINARY --repeats $REPEATS --csv"
echo ""

CSV_OUT="results/raw/${DEVICE_NAME}.csv"
$PIN_CMD "$BINARY" --repeats "$REPEATS" --csv > "$CSV_OUT"

if [ ! -s "$CSV_OUT" ]; then
    echo "ERROR: benchmark produced no output" >&2
    exit 1
fi

ROW_COUNT=$(($(wc -l < "$CSV_OUT") - 1))
echo "  CSV: $CSV_OUT ($ROW_COUNT data rows)"
echo ""

# --- 6. post-run thermal snapshot ------------------------------------------
THERMAL_POST="results/thermal_${DEVICE_NAME}_post.txt"
{
    echo "# Post-run thermal snapshot"
    echo "# Device: $DEVICE_NAME"
    echo "# Git SHA: $GIT_SHA"
    echo "# Timestamp: $(date -u +'Y-%m-%dT%H:%M:%SZ' 2>/dev/null || echo '?')"
    echo "# OMP_NUM_THREADS: $THREADS"
    echo ""
    for z in /sys/class/thermal/thermal_zone*; do
        ZONE_TYPE=$(cat "$z/type" 2>/dev/null || echo "?")
        ZONE_TEMP=$(cat "$z/temp" 2>/dev/null || echo "?")
        echo "$(basename "$z") $ZONE_TYPE ${ZONE_TEMP}"
    done
} > "$THERMAL_POST"
echo "  Post-run thermals: $THERMAL_POST"

# Quick thermal delta check
PRE_MAX=$(grep -oE '[0-9]+$' "$THERMAL_PRE" | sort -n | tail -1 || echo 0)
POST_MAX=$(grep -oE '[0-9]+$' "$THERMAL_POST" | sort -n | tail -1 || echo 0)
if [ -n "$PRE_MAX" ] && [ -n "$POST_MAX" ]; then
    DELTA=$(( (POST_MAX - PRE_MAX) / 1000 ))
    if [ "$DELTA" -gt 10 ]; then
        echo "  ⚠ Thermal delta: +${DELTA}°C — possible throttling, check p95/p50 spread"
    else
        echo "  Thermal delta: +${DELTA}°C"
    fi
fi
echo ""

# --- 7. manifest ------------------------------------------------------------
MANIFEST_OUT="results/manifests/${DEVICE_NAME}.json"
if [ -x "scripts/capture_manifest.sh" ]; then
    scripts/capture_manifest.sh --run-id "${DEVICE_NAME}_sweep_${GIT_SHA}" > "$MANIFEST_OUT"
elif command -v python3 >/dev/null 2>&1; then
    python3 bench/manifest.py > "$MANIFEST_OUT" 2>/dev/null || {
        echo "WARNING: manifest capture failed; creating minimal manifest" >&2
        echo "{\"git\":{\"sha\":\"$GIT_SHA\",\"dirty\":$( [ -n "$STATUS" ] && echo true || echo false )}}" > "$MANIFEST_OUT"
    }
else
    echo "{\"git\":{\"sha\":\"$GIT_SHA\",\"dirty\":$( [ -n "$STATUS" ] && echo true || echo false )}}" > "$MANIFEST_OUT"
fi
echo "  Manifest: $MANIFEST_OUT"

# --- 8. summary ------------------------------------------------------------
echo ""
echo "=== Sweep complete ==="
echo "  Commit:  $GIT_SHA"
echo "  Device:  $DEVICE_NAME"
echo "  Threads: $THREADS"
echo "  Rows:    $ROW_COUNT"
echo "  CSV:     $CSV_OUT"
echo ""
echo "Commit these files and push to your branch:"
echo "  git add results/ && git commit -m \"${DEVICE_NAME}: fleet sweep at ${GIT_SHA} (ob-bf7)\""
echo ""
echo "After ALL devices have run at this commit, regenerate the fleet analysis:"
echo "  python3 bench/fleet_analysis.py"
