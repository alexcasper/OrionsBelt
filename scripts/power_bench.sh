#!/usr/bin/env bash
# Power-instrumented benchmark wrapper for Jetson Nano (Jetson-J1).
#
# Samples the onboard INA3221 power monitor at /sys/devices/.../iio:device0/
# while bench_gdn runs, then reports energy-per-GiB alongside throughput.
#
# Rails:
#   POM_5V_IN  (rail 0) — total board input power
#   POM_5V_GPU (rail 1) — GPU subsystem
#   POM_5V_CPU (rail 2) — CPU subsystem
#
# Usage:
#   sudo ./scripts/power_bench.sh [bench_args...]
#   sudo ./scripts/power_bench.sh --repeats 30 --csv
#
# Requires sudo (root) to read the IIO sysfs entries.
# If the binary path isn't given, uses dist/bench_gdn_jetson_a57.

set -euo pipefail

# --- locate the IIO device -------------------------------------------------
IIO_BASE="/sys/devices/50000000.host1x/546c0000.i2c/i2c-6/6-0040/iio:device0"
if [ ! -d "$IIO_BASE" ]; then
    echo "ERROR: INA3221 IIO device not found at $IIO_BASE" >&2
    echo "Try: sudo modprobe ina3221" >&2
    exit 1
fi

read_rail() {
    cat "$IIO_BASE/$1" 2>/dev/null || echo "0"
}

# --- locate binary ---------------------------------------------------------
BINARY="${BENCH_BINARY:-dist/bench_gdn_jetson_a57}"
if [ ! -x "$BINARY" ]; then
    echo "ERROR: benchmark binary not found: $BINARY" >&2
    exit 1
fi

# --- temp files ------------------------------------------------------------
POWER_LOG=$(mktemp /tmp/power_sample_XXXXXX.csv)
BENCH_OUT=$(mktemp /tmp/bench_out_XXXXXX)
trap 'rm -f "$POWER_LOG" "$BENCH_OUT"' EXIT

# --- thermal zone ----------------------------------------------------------
THERMAL="/sys/class/thermal/thermal_zone0/temp"

# --- idle baseline (1 second average) --------------------------------------
echo "# Sampling idle power baseline (1 s)..." >&2
IDLE_SAMPLES=0; IDLE_IN=0; IDLE_GPU=0; IDLE_CPU=0; IDLE_TEMP=0
for i in $(seq 1 10); do
    IN=$(read_rail in_power0_input)
    GPU=$(read_rail in_power1_input)
    CPU=$(read_rail in_power2_input)
    TEMP=$(cat "$THERMAL" 2>/dev/null || echo "0")
    IDLE_IN=$((IDLE_IN + IN))
    IDLE_GPU=$((IDLE_GPU + GPU))
    IDLE_CPU=$((IDLE_CPU + CPU))
    IDLE_TEMP=$((IDLE_TEMP + TEMP))
    IDLE_SAMPLES=$((IDLE_SAMPLES + 1))
    sleep 0.1
done
IDLE_IN=$((IDLE_IN / IDLE_SAMPLES))
IDLE_GPU=$((IDLE_GPU / IDLE_SAMPLES))
IDLE_CPU=$((IDLE_CPU / IDLE_SAMPLES))
IDLE_TEMP=$((IDLE_TEMP / IDLE_SAMPLES))

echo "# Idle: IN=${IDLE_IN}mW GPU=${IDLE_GPU}mW CPU=${IDLE_CPU}mW temp=$((IDLE_TEMP / 1000)).$((IDLE_TEMP % 1000))C" >&2

# --- power sampler (background) --------------------------------------------
# Writes CSV: timestamp_ms,power_in_mw,power_gpu_mw,power_cpu_mw,temp_milliC
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.1}"
echo "timestamp_ms,power_in_mw,power_gpu_mw,power_cpu_mw,temp_milliC" > "$POWER_LOG"

sample_power() {
    local start_ns end_ns elapsed_ms
    start_ns=$(date +%s%N)
    while kill -0 "$BENCH_PID" 2>/dev/null; do
        local IN GPU CPU TEMP now_ns
        IN=$(read_rail in_power0_input)
        GPU=$(read_rail in_power1_input)
        CPU=$(read_rail in_power2_input)
        TEMP=$(cat "$THERMAL" 2>/dev/null || echo "0")
        now_ns=$(date +%s%N)
        elapsed_ms=$(( (now_ns - start_ns) / 1000000))
        echo "${elapsed_ms},${IN},${GPU},${CPU},${TEMP}" >> "$POWER_LOG"
        sleep "$SAMPLE_INTERVAL"
    done
}

# --- run benchmark with power sampling -------------------------------------
echo "# Running: $BINARY $*" >&2
BENCH_START=$(date +%s%N)
"$BINARY" "$@" > "$BENCH_OUT" 2>&1 &
BENCH_PID=$!

# Start sampler
sample_power &
SAMPLER_PID=$!

# Wait for benchmark
wait "$BENCH_PID"
BENCH_RC=$?
BENCH_END=$(date +%s%N)
wait "$SAMPLER_PID" 2>/dev/null || true

BENCH_DURATION_NS=$((BENCH_END - BENCH_START))
BENCH_DURATION_MS=$((BENCH_DURATION_NS / 1000000))
BENCH_DURATION_S=$(awk "BEGIN { printf \"%.3f\", $BENCH_DURATION_NS / 1000000000 }")

# --- output benchmark results ---------------------------------------------
cat "$BENCH_OUT"

# --- compute power statistics ----------------------------------------------
# Skip the header line; compute averages from the sample rows
AVG_IN=0; AVG_GPU=0; AVG_CPU=0; AVG_TEMP=0; N=0; MAX_IN=0; MAX_CPU=0; MAX_TEMP=0
while IFS=, read -r ts p_in p_gpu p_cpu temp; do
    # skip header
    [ "$ts" = "timestamp_ms" ] && continue
    [ -z "$ts" ] && continue
    AVG_IN=$((AVG_IN + p_in))
    AVG_GPU=$((AVG_GPU + p_gpu))
    AVG_CPU=$((AVG_CPU + p_cpu))
    AVG_TEMP=$((AVG_TEMP + temp))
    [ "$p_in" -gt "$MAX_IN" ] && MAX_IN=$p_in
    [ "$p_cpu" -gt "$MAX_CPU" ] && MAX_CPU=$p_cpu
    [ "$temp" -gt "$MAX_TEMP" ] && MAX_TEMP=$temp
    N=$((N + 1))
done < "$POWER_LOG"

if [ "$N" -eq 0 ]; then
    echo "ERROR: no power samples collected" >&2
    exit 1
fi

AVG_IN=$((AVG_IN / N))
AVG_GPU=$((AVG_GPU / N))
AVG_CPU=$((AVG_CPU / N))
AVG_TEMP=$((AVG_TEMP / N))
DELTA_IN=$((AVG_IN - IDLE_IN))
DELTA_CPU=$((AVG_CPU - IDLE_CPU))

# Energy during run (mJ = mW * s)
ENERGY_MJ=$(awk "BEGIN { printf \"%.1f\", $DELTA_IN * $BENCH_DURATION_S }")
ENERGY_CPU_MJ=$(awk "BEGIN { printf \"%.1f\", $DELTA_CPU * $BENCH_DURATION_S }")

echo ""
echo "# ============================================================ "
echo "# Power Summary (INA3221, $N samples @ ${SAMPLE_INTERVAL}s interval)"
echo "# ============================================================ "
echo "# Duration:       ${BENCH_DURATION_S}s"
echo "# Idle power:     IN=${IDLE_IN}mW CPU=${IDLE_CPU}mW"
echo "# Avg load power: IN=${AVG_IN}mW GPU=${AVG_GPU}mW CPU=${AVG_CPU}mW"
echo "# Peak power:     IN=${MAX_IN}mW CPU=${MAX_CPU}mW"
echo "# Delta power:    IN=${DELTA_IN}mW CPU=${DELTA_CPU}mW"
echo "# Energy:         board=${ENERGY_MJ}mJ cpu=${ENERGY_CPU_MJ}mJ"
echo "# Temperature:    idle=$((IDLE_TEMP / 1000))C avg=$((AVG_TEMP / 1000))C peak=$((MAX_TEMP / 1000))C"
echo "#"
echo "# Power log: $POWER_LOG (preserved)"

# Don't clean up the power log — rename it for analysis
PRESERVED="${POWER_LOG}.final"
mv "$POWER_LOG" "$PRESERVED"
echo "# Preserved as: $PRESERVED"
# Prevent trap from deleting it
POWER_LOG=""
