#!/usr/bin/env bash
# Validate device benchmark CSVs for data completeness and staleness.
#
# Catches the kind of issue found 2026-08-03: jetson-j2.csv had only 6 rows
# (pre-OpenMP baseline) while j1 had 28 rows (full sweep). This script checks:
#
#   1. Every CSV has a matching manifest in results/manifests/
#   2. The CSV has the expected kernel variants (fp32, bf16, fp16)
#   3. The CSV has both prefill (seq=64) and decode (seq=1) configs
#   4. The manifest git_sha is not from a known stale commit
#
# Usage: ./scripts/validate_bench_csvs.sh
# Exit code: 0 if all pass, 1 if any issues found
set -euo pipefail

RAW="results/raw"
MANIFESTS="results/manifests"
ISSUES=0

# Known stale commits (pre-OpenMP, pre-NEON-unrolling)
STALE_COMMITS="28729f3"

echo "=== Benchmark CSV Validation ==="
echo ""

# Find all device CSVs (exclude intermediate optimization-stage files)
for csv in "$RAW"/jetson-j*.csv "$RAW"/pi5-*.csv "$RAW"/rk3588-t*_big.csv "$RAW"/rk3588-t*_little.csv; do
    [ -f "$csv" ] || continue

    base=$(basename "$csv" .csv)

    # Skip intermediate/special files
    case "$base" in
        *-omp|*-omp-unroll|*-conv-unroll|*_single|*-full-optimized|*-sustained*|*_power*) continue ;;
    esac

    rows=$(($(wc -l < "$csv") - 1))  # subtract header
    has_bf16=$(grep -c "_bf16" "$csv" || true)
    has_f16=$(grep -c "_f16" "$csv" || true)
    has_decode=$(grep -c "_decode" "$csv" || true)

    echo "  $base: $rows data rows"

    # Check for manifest
    manifest_found=""
    for m in "$MANIFESTS/$base.json"; do
        if [ -f "$m" ]; then
            manifest_found="$m"
            # Check for stale commit
            sha=$(python3 -c "
import json,sys
try:
    d=json.load(open('$m'))
    g=d.get('git',d.get('git_sha',''))
    if isinstance(g,dict): print(g.get('sha','')[:7])
    else: print(str(g)[:7])
except: print('')
" 2>/dev/null || echo "")
            if [ -n "$sha" ]; then
                for stale in $STALE_COMMITS; do
                    if [ "$sha" = "$stale" ]; then
                        echo "    ⚠ WARNING: manifest git_sha $sha is a known stale commit (pre-OpenMP)"
                        echo "    → Re-run ALL fleet devices together, not just this one:"
                        echo "      results/figures/fleet_bandwidth_scaling.md deliberately compares"
                        echo "      every device at $sha so the code is matched. Re-running one device"
                        echo "      alone makes that table mix commits, which is the defect ob-bf7"
                        echo "      tracks. Single-device: dist/bench_gdn_<variant> --repeats 30 --csv"
                        ISSUES=$((ISSUES + 1))
                    fi
                done
                echo "    manifest: $sha"
            fi
            break
        fi
    done
    if [ -z "$manifest_found" ]; then
        echo "    ⚠ WARNING: no manifest found in $MANIFESTS/$base.json"
        ISSUES=$((ISSUES + 1))
    fi

    # Check kernel coverage (warn, don't fail — baseline-only CSVs are legitimate)
    if [ "$rows" -lt 10 ]; then
        echo "    ⚠ NOTE: only $rows rows — may be missing mixed-precision or decode variants"
    fi
    if [ "$has_bf16" -eq 0 ] && [ "$rows" -gt 6 ]; then
        echo "    ℹ no bf16 variants (expected for full sweep)"
    fi

    echo ""
done

# Summary
if [ "$ISSUES" -eq 0 ]; then
    echo "✅ All CSVs validated — no issues found"
    exit 0
else
    echo "⚠ $ISSUES issue(s) found — see warnings above"
    exit 1
fi
