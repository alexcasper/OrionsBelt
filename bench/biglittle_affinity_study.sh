#!/usr/bin/env bash
# big.LITTLE affinity policy study for RK3588 (ob-dqu)
#
# Compares GDN kernel throughput under different CPU affinity policies:
#   1. big-only:    A76 binary pinned to cpu4-7 (recommended latency path)
#   2. all-cores:   A76 binary, no pinning (default OS scheduler behavior)
#   3. big-on-little: A76 binary pinned to cpu0-3 (cross-cluster test)
#   4. little-only: A55 binary pinned to cpu0-3 (housekeeping path)
#   5. little-on-big: A55 binary pinned to cpu4-7 (tuning mismatch test)
#   6. simultaneous: A76 on big + A55 on little (split workload)
#
# Usage: bash bench/biglittle_affinity_study.sh [--repeats N] [--csv]
#   Default: --repeats 30, human-readable output

set -euo pipefail

# Clean up temp files on exit/interrupt
_trap_cleanup() {
    rm -f "$TMP_BIG" "$TMP_LITTLE" 2>/dev/null || true
}
trap _trap_cleanup EXIT

REPEATS=30
CSV_MODE=false
TMP_BIG=""
TMP_LITTLE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repeats) REPEATS="$2"; shift 2 ;;
        --csv)     CSV_MODE=true; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

BIG="dist/bench_gdn_rk3588_a76"
LITTLE="dist/bench_gdn_rk3588_a55"

# Ensure binaries exist
for b in "$BIG" "$LITTLE"; do
    [[ -x "$b" ]] || { echo "ERROR: $b not found. Run scripts/build_device_bench.sh" >&2; exit 1; }
done

# Thermal baseline
therm_before=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo "N/A")

run_config() {
    local label="$1" binary="$2" affinity="$3"
    local extra="${4:-}"

    if [[ -n "$affinity" ]]; then
        if $CSV_MODE; then
            echo "# config=$label binary=$(basename $binary) affinity=$affinity extra=$extra therm_before=$therm_before"
            taskset -c "$affinity" "$binary" --repeats "$REPEATS" --csv 2>/dev/null
        else
            echo ""
            echo "==================================================================="
            echo "  CONFIG: $label"
            echo "  binary=$(basename $binary)  affinity=$affinity  extra=$extra"
            echo "==================================================================="
            taskset -c "$affinity" "$binary" --repeats "$REPEATS" 2>/dev/null
        fi
    else
        # No pinning — all cores available
        if $CSV_MODE; then
            echo "# config=$label binary=$(basename $binary) affinity=all extra=$extra therm_before=$therm_before"
            "$binary" --repeats "$REPEATS" --csv 2>/dev/null
        else
            echo ""
            echo "==================================================================="
            echo "  CONFIG: $label"
            echo "  binary=$(basename $binary)  affinity=all  extra=$extra"
            echo "==================================================================="
            "$binary" --repeats "$REPEATS" 2>/dev/null
        fi
    fi
}

# Thermal check
echo "# big.LITTLE affinity study (ob-dqu)" >&2
echo "# device: RK3588 (4xA76@2.3GHz + 4xA55@1.8GHz)" >&2
echo "# repeats: $REPEATS" >&2
echo "# thermal_before: $therm_before" >&2
echo "# governor: $(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor)" >&2

# 1. Big-only: A76 binary on big cores
run_config "big_only_a76" "$BIG" "4-7"

# 2. All-cores: A76 binary, no pinning (OS scheduler decides)
run_config "all_cores_a76" "$BIG" ""

# 3. Cross-cluster: A76 binary on little cores
run_config "big_on_little" "$BIG" "0-3"

# 4. Little-only: A55 binary on little cores
run_config "little_only_a55" "$LITTLE" "0-3"

# 5. Tuning mismatch: A55 binary on big cores
run_config "little_on_big" "$LITTLE" "4-7"

# 6. Simultaneous: A76 on big + A55 on little (split workload)
# Both run the full benchmark; we measure wall-clock overlap
if $CSV_MODE; then
    echo "# config=simultaneous_split binary=both affinity=big4-7+little0-3 therm_before=$therm_before"
    echo "# (A76 on big cores, A55 on little cores — launched simultaneously)"
    # Run both in parallel, capture to temp files
    TMP_BIG=$(mktemp)
    TMP_LITTLE=$(mktemp)
    taskset -c 4-7 "$BIG" --repeats "$REPEATS" --csv > "$TMP_BIG" 2>/dev/null &
    PID_BIG=$!
    taskset -c 0-3 "$LITTLE" --repeats "$REPEATS" --csv > "$TMP_LITTLE" 2>/dev/null &
    PID_LITTLE=$!
    wait $PID_BIG $PID_LITTLE
    echo "# --- A76 on big cores (concurrent with A55 on little): ---"
    cat "$TMP_BIG"
    echo "# --- A55 on little cores (concurrent with A76 on big): ---"
    cat "$TMP_LITTLE"
    rm -f "$TMP_BIG" "$TMP_LITTLE"
else
    echo ""
    echo "==================================================================="
    echo "  CONFIG: simultaneous_split"
    echo "  A76 on big cores + A55 on little cores — launched simultaneously"
    echo "==================================================================="
    TMP_BIG=$(mktemp)
    TMP_LITTLE=$(mktemp)
    taskset -c 4-7 "$BIG" --repeats "$REPEATS" > "$TMP_BIG" 2>/dev/null &
    PID_BIG=$!
    taskset -c 0-3 "$LITTLE" --repeats "$REPEATS" > "$TMP_LITTLE" 2>/dev/null &
    PID_LITTLE=$!
    wait $PID_BIG $PID_LITTLE
    echo "--- A76 on big cores (concurrent): ---"
    cat "$TMP_BIG"
    echo "--- A55 on little cores (concurrent): ---"
    cat "$TMP_LITTLE"
    rm -f "$TMP_BIG" "$TMP_LITTLE"
fi

# Thermal after
therm_after=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo "N/A")
echo "# thermal_after: $therm_after" >&2
if [[ "$therm_before" =~ ^[0-9]+$ ]] && [[ "$therm_after" =~ ^[0-9]+$ ]]; then
    echo "# thermal_delta: $(( therm_after - therm_before ))" >&2
else
    echo "# thermal_delta: N/A" >&2
fi
