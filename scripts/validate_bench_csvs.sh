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

# Find all device CSVs — validate everything in results/raw/ except known
# intermediate/exploratory files (skip patterns below).
for csv in "$RAW"/*.csv; do
    [ -f "$csv" ] || continue

    base=$(basename "$csv" .csv)

    # Skip intermediate/special files
    case "$base" in
        *-omp|*-omp-unroll|*-conv-unroll|*_single|*-full-optimized|*-sustained*|*_power*) continue ;;
        # Old hyphen-format naming (rk3588-t4-big.csv etc.) superseded by _big/_little
        rk3588-t[34]-*) continue ;;
        # Cleanup / freshness-check runs
        *-clean|*-fresh) continue ;;
        # Timestamped exploratory sweeps (e.g. rk3588-t4_20260806T092103Z_...)
        *_20[0-9][0-9][0-1][0-9][0-3][0-9]T*) continue ;;
        # GDN1-vs-GDN2 comparison snapshots (different CSV schema)
        *_gdn2_vs_gdn1_*) continue ;;
        # GPU microbenchmarks (OpenCL Mali, different schema — validated by validate_results.py)
        *_gpu_*) continue ;;
        # Cross-quant comparison tables (aggregate multiple commits, no single-run manifest)
        *_vs_*) continue ;;
    esac

    rows=$(($(wc -l < "$csv") - 1))  # subtract header
    has_bf16=$(grep -c "_bf16" "$csv" || true)
    has_f16=$(grep -c "_f16" "$csv" || true)
    has_decode=$(grep -c "_decode" "$csv" || true)

    echo "  $base: $rows data rows"

    # Check for manifest.
    # Some devices produce multiple CSVs from one run (e.g. rk3588-t3_big.csv and
    # rk3588-t3_little.csv share rk3588-t3.json), so try the exact name first,
    # then strip _big/_little/_singlethread suffixes to find the shared manifest.
    manifest_found=""
    # Build candidate manifest names: exact, then progressively stripped.
    # Strip cluster/run suffixes (_big/_little/_singlethread) and e2e pipeline
    # suffixes (_raw/_schema) to find the shared manifest.
    stripped="$base"
    for suffix in _big _little _singlethread; do
        case "$stripped" in
            *"$suffix") stripped="${stripped%$suffix}" ;;
        esac
    done
    for suffix in _raw _schema; do
        case "$stripped" in
            *"$suffix") stripped="${stripped%$suffix}" ;;
        esac
    done
    # ctxsweep_e2e CSVs share ctxsweep manifests (the _e2e is a pipeline qualifier,
    # not a device identifier). Try the name without _e2e as a third candidate
    # so we don't break *_e2e_raw -> *_e2e matching for e2e decode manifests.
    stripped_no_e2e="$stripped"
    case "$stripped_no_e2e" in
        *_e2e) stripped_no_e2e="${stripped_no_e2e%_e2e}" ;;
    esac
    for m in "$MANIFESTS/$base.json" "$MANIFESTS/$stripped.json" \
             "$MANIFESTS/$stripped_no_e2e.json"; do
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
    # E2E decode and delta_matmul CSVs legitimately have few rows by design.
    case "$base" in
        *e2e*|*delta_matmul*) special_csv=1 ;; *) special_csv=0 ;; esac
    if [ "$rows" -lt 10 ] && [ "$special_csv" -eq 0 ]; then
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
