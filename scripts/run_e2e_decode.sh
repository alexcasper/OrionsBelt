#!/usr/bin/env bash
# Fleet e2e decode benchmark — end-to-end Qwen3.5 CPU-only tokens/sec.
#
# Runs the C decode loop (gdn_e2e_decode.c) on the local device with the same
# provenance protocol as fleet_sweep.sh: clean tree, governor pinned to
# performance, pre/post thermal snapshots, taskset pinning on asymmetric boards,
# and a manifest. The raw CSV is then converted to RESULTS_SCHEMA-conformant
# tidy rows by bench/convert_e2e_decode.py.
#
# Bead ob-mrd.8. Run this on EACH fleet device at the SAME git commit.
#
# Usage:
#   ./scripts/run_e2e_decode.sh                          # auto-detect, 8 tokens
#   ./scripts/run_e2e_decode.sh --tokens 16              # more tokens
#   ./scripts/run_e2e_decode.sh --pin "4-7" --cluster big    # RK3588 big cluster
#   ./scripts/run_e2e_decode.sh --pin "0-3" --cluster little # RK3588 little cluster
#   ./scripts/run_e2e_decode.sh --device rk3588-t3       # override output name
#   ./scripts/run_e2e_decode.sh --runs 3                 # 3 independent runs for repeat stats
#   ./scripts/run_e2e_decode.sh --rebuild                # force rebuild even if binary exists
#
# Output:
#   results/raw/<device>_e2e_raw.csv         — raw binary output (simple CSV)
#   results/raw/<device>_e2e_schema_<run>.csv — schema-conformant tidy rows
#   results/manifests/<device>_e2e.json       — provenance manifest
#   results/thermal_<device>_e2e_pre.txt      — pre-run thermal snapshot
#   results/thermal_<device>_e2e_post.txt     — post-run thermal snapshot

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# --- defaults ---------------------------------------------------------------
DEVICE_NAME=""
TOKENS=8
PIN_CPUS=""
CLUSTER="all"
RUNS=1
FORCE=0
REBUILD=0
MODEL="4b"
QUANT="fp32"
KV_QUANT="fp32"
BINARY=""
K=src/orionsbelt/engines/cpu/kernels

# --- arg parsing ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)  DEVICE_NAME="$2"; shift 2 ;;
        --tokens)  TOKENS="$2"; shift 2 ;;
        --pin)     PIN_CPUS="$2"; shift 2 ;;
        --cluster) CLUSTER="$2"; shift 2 ;;
        --runs)    RUNS="$2"; shift 2 ;;
        --binary)  BINARY="$2"; shift 2 ;;
        --model)   MODEL="$2"; shift 2 ;;   # 4b (default) or 08b
        --quant)   QUANT="$2"; shift 2 ;;   # fp32 (default), int8, or q8_0
        --kv-quant) KV_QUANT="$2"; shift 2 ;; # fp32 (default) or int8 (KV cache)
        --force)   FORCE=1; shift ;;
        --rebuild) REBUILD=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

case "$MODEL" in
    4b)   MODEL_DEFINE="" ;;
    08b)  MODEL_DEFINE="-DMODEL_08B" ;;
    *) echo "Unknown --model: $MODEL (expected 4b or 08b)" >&2; exit 1 ;;
esac

case "$QUANT" in
    fp32) QUANT_DEF="" ;;
    int8) QUANT_DEF="-DINT8_WEIGHTS" ;;
    int4) QUANT_DEF="-DINT4_WEIGHTS" ;;
    q8_0) QUANT_DEF="-DQ80_WEIGHTS" ;;
    *) echo "Unknown --quant: $QUANT (expected fp32, int8, int4, or q8_0)" >&2; exit 1 ;;
esac
case "$KV_QUANT" in
    fp32) KV_DEF="" ;;
    int8) KV_DEF="-DKV_INT8" ;;
    *) echo "Unknown --kv-quant: $KV_QUANT (expected fp32 or int8)" >&2; exit 1 ;;
esac
if [ -z "$BINARY" ]; then
    if [ "$MODEL" = "4b" ]; then
        BINARY="dist/bench_gdn_e2e_decode"
    else
        BINARY="dist/bench_gdn_e2e_decode_${MODEL}"
    fi
    if [ "$QUANT" = "int8" ]; then
        BINARY="${BINARY}_int8"
    fi
    if [ "$QUANT" = "int4" ]; then
        BINARY="${BINARY}_int4"
    fi
    if [ "$QUANT" = "q8_0" ]; then
        BINARY="${BINARY}_q80"
    fi
    if [ "$KV_QUANT" = "int8" ]; then
        BINARY="${BINARY}_kvint8"
    fi
fi

# --- 1. clean tree check ----------------------------------------------------
STATUS=$(git status --porcelain 2>/dev/null || true)
if [ -n "$STATUS" ] && [ "$FORCE" -eq 0 ]; then
    echo "ERROR: working tree is dirty. A dirty tree means the manifest's git SHA" >&2
    echo "       does not identify the code that produced the numbers (bead ob-bf7)." >&2
    echo "" >&2
    echo "Commit or stash your changes first, or use --force to override." >&2
    echo "  $STATUS" >&2
    exit 1
fi
if [ -n "$STATUS" ] && [ "$FORCE" -eq 1 ]; then
    echo "WARNING: --force used; manifest will record dirty=true" >&2
fi

GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")
echo "=== E2E Decode Benchmark (ob-mrd.8) ==="
echo "  Git SHA: $GIT_SHA"
echo "  Tokens per run: $TOKENS"
echo "  Runs: $RUNS"
echo ""

# --- 2. device detection ----------------------------------------------------
ARCH=$(uname -m 2>/dev/null || echo "unknown")
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
    echo "WARNING: not running on aarch64 ($ARCH)" >&2
fi

if [ -z "$DEVICE_NAME" ]; then
    HOST=$(hostname 2>/dev/null || echo "unknown")
    case "$HOST" in
        j1|jetson-j1)  DEVICE_NAME="jetson-j1" ;;
        j2|jetson-j2)  DEVICE_NAME="jetson-j2" ;;
        r5|pi5-r5)     DEVICE_NAME="pi5-r5" ;;
        t3|rk3588-t3)  DEVICE_NAME="rk3588-t3" ;;
        t4|rk3588-t4)  DEVICE_NAME="rk3588-t4" ;;
        *)             DEVICE_NAME="$HOST" ;;
    esac
fi
if [ "$MODEL" != "4b" ]; then
    DEVICE_NAME="${DEVICE_NAME}_${MODEL}"
fi
if [ "$CLUSTER" != "all" ]; then
    DEVICE_NAME="${DEVICE_NAME}_${CLUSTER}"
fi
if [ "$QUANT" = "int8" ]; then
    DEVICE_NAME="${DEVICE_NAME}_int8"
fi
if [ "$QUANT" = "int4" ]; then
    DEVICE_NAME="${DEVICE_NAME}_int4"
fi
if [ "$QUANT" = "q8_0" ]; then
    DEVICE_NAME="${DEVICE_NAME}_q80"
fi

echo "  Device:  $DEVICE_NAME"
echo "  Model:   $MODEL"
echo "  Cluster: $CLUSTER"
echo ""

# --- 3. build if needed ----------------------------------------------------
# Detect ISA and use appropriate flags (mirrors detect_isa.sh logic).
# Must handle ARMv8.0-A cores (e.g. Cortex-A57) that lack dotprod —
# the old default of -march=armv8.2-a+dotprod would SIGILL on them.
FEATURES=$(grep -m1 '^Features' /proc/cpuinfo 2>/dev/null || echo "")
has_feat() { echo "$FEATURES" | grep -qw "$1" 2>/dev/null; }
if has_feat sve2; then
    ISA_FLAGS="-march=armv9-a+sve2+i8mm+bf16"
elif has_feat sve; then
    ISA_FLAGS="-march=armv8.2-a+sve+bf16"
elif has_feat asimddp; then
    ISA_FLAGS="-march=armv8.2-a+dotprod"
elif has_feat asimd; then
    ISA_FLAGS="-march=armv8-a"
else
    ISA_FLAGS="-march=armv8-a"
fi

NEED_BUILD=0
if [ ! -x "$BINARY" ]; then
    echo "  Binary not found, building..."
    NEED_BUILD=1
elif [ "$REBUILD" -eq 1 ]; then
    echo "  --rebuild requested, rebuilding..."
    NEED_BUILD=1
else
    # Staleness check: rebuild if any source file is newer than the binary.
    # This prevents the class of bug where run_e2e_decode.sh reuses a stale
    # binary from before a code optimization was merged (see FINDINGS §14
    # correction — a stale binary produced 19x-too-slow numbers).
    for src in "$K/gdn_sve.c" "$K/gdn_delta_matmul.c" "$K/gdn_e2e_decode.c"; do
        if [ "$src" -nt "$BINARY" ]; then
            echo "  Source $src is newer than binary — rebuilding..."
            NEED_BUILD=1
            break
        fi
    done
fi

if [ "$NEED_BUILD" -eq 1 ]; then
    cc -O3 -fopenmp $ISA_FLAGS $MODEL_DEFINE -static \
        -Wno-aggressive-loop-optimizations \
        ${QUANT_DEF:+$QUANT_DEF} ${KV_DEF:+$KV_DEF} \
        "$K/gdn_sve.c" "$K/gdn_delta_matmul.c" "$K/gdn_e2e_decode.c" \
        -I"$K/" -o "$BINARY" -lm 2>&1 || {
        echo "ERROR: build failed" >&2
        exit 1
    }
    echo "  Built: $BINARY"
fi

# --- 4. governor ------------------------------------------------------------
GOV_SET=0
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    CURRENT_GOV=$(cat "$c" 2>/dev/null || echo "?")
    if [ "$CURRENT_GOV" != "performance" ]; then
        if echo performance | sudo tee "$c" >/dev/null 2>&1; then
            GOV_SET=1
        else
            echo "WARNING: cannot set governor on $c (current: $CURRENT_GOV)" >&2
        fi
    fi
done

if [ "$GOV_SET" -eq 1 ]; then
    echo "  Governor: set to performance"
else
    ALL_PERF=1
    for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        CURRENT_GOV=$(cat "$c" 2>/dev/null || echo "?")
        if [ "$CURRENT_GOV" != "performance" ]; then
            ALL_PERF=0
        fi
    done
    if [ "$ALL_PERF" -eq 1 ]; then
        echo "  Governor: performance (already set)"
    else
        echo "  Governor: NOT performance on all cores — numbers will be low" >&2
    fi
fi

# Record OMP threads
if [ -n "${OMP_NUM_THREADS:-}" ]; then
    echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
else
    echo "  OMP_NUM_THREADS: unset (libgomp default)"
fi
echo ""

# --- 5. pre-run thermal snapshot -------------------------------------------
mkdir -p results/raw results/manifests
THERMAL_PRE="results/thermal_${DEVICE_NAME}_e2e_pre.txt"
{
    echo "# Pre-run thermal snapshot — e2e decode"
    echo "# Device: $DEVICE_NAME  Cluster: $CLUSTER"
    echo "# Git SHA: $GIT_SHA"
    echo "# Timestamp: $(date -u +'%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || echo '?')"
    echo ""
    for z in /sys/class/thermal/thermal_zone*; do
        ZONE_TYPE=$(cat "$z/type" 2>/dev/null || echo "?")
        ZONE_TEMP=$(cat "$z/temp" 2>/dev/null || echo "?")
        echo "$(basename "$z") $ZONE_TYPE ${ZONE_TEMP}"
    done
} > "$THERMAL_PRE"
echo "  Pre-run thermals: $THERMAL_PRE"

# --- 6. run benchmark -------------------------------------------------------
echo ""
echo "=== Running e2e decode benchmark ==="
PIN_CMD=""
if [ -n "$PIN_CPUS" ]; then
    PIN_CMD="taskset -c $PIN_CPUS"
    echo "  Pinning to CPUs: $PIN_CPUS"
fi
echo "  Tokens per run: $TOKENS"
echo "  Runs: $RUNS"
echo ""

RAW_OUT="results/raw/${DEVICE_NAME}_e2e_raw.csv"
# Write header for raw CSV (same as binary output)
echo "model,tokens,ttft_ms,tok_per_sec_mean,p50_us,p95_us,p99_us,mean_us,gdn_proj_pct,gdn_conv_pct,gdn_decay_pct,gdn_scan_pct,gdn_oproj_pct,full_pct,ffn_pct" > "$RAW_OUT"

for run_idx in $(seq 0 $((RUNS - 1))); do
    echo "  Run $((run_idx + 1))/$RUNS..."
    # Strip the header line from each run's CSV output (header already written above)
    $PIN_CMD "$BINARY" --tokens "$TOKENS" --csv 2>/dev/null | tail -n +2 >> "$RAW_OUT" || {
        echo "ERROR: run $((run_idx + 1)) failed" >&2
        exit 1
    }
done

RAW_ROWS=$(($(wc -l < "$RAW_OUT") - 1))
echo "  Raw CSV: $RAW_OUT ($RAW_ROWS data rows)"
echo ""

# --- 7. post-run thermal snapshot ------------------------------------------
THERMAL_POST="results/thermal_${DEVICE_NAME}_e2e_post.txt"
{
    echo "# Post-run thermal snapshot — e2e decode"
    echo "# Device: $DEVICE_NAME  Cluster: $CLUSTER"
    echo "# Git SHA: $GIT_SHA"
    echo "# Timestamp: $(date -u +'%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || echo '?')"
    echo ""
    for z in /sys/class/thermal/thermal_zone*; do
        ZONE_TYPE=$(cat "$z/type" 2>/dev/null || echo "?")
        ZONE_TEMP=$(cat "$z/temp" 2>/dev/null || echo "?")
        echo "$(basename "$z") $ZONE_TYPE ${ZONE_TEMP}"
    done
} > "$THERMAL_POST"
echo "  Post-run thermals: $THERMAL_POST"

# Quick thermal delta
PRE_MAX=$(grep -oE '[0-9]+$' "$THERMAL_PRE" | sort -n | tail -1 || echo 0)
POST_MAX=$(grep -oE '[0-9]+$' "$THERMAL_POST" | sort -n | tail -1 || echo 0)
if [ -n "$PRE_MAX" ] && [ -n "$POST_MAX" ]; then
    DELTA=$(( (POST_MAX - PRE_MAX) / 1000 ))
    if [ "$DELTA" -gt 10 ]; then
        echo "  ⚠ Thermal delta: +${DELTA}°C — possible throttling"
    else
        echo "  Thermal delta: +${DELTA}°C"
    fi
fi
echo ""

# --- 8. manifest ------------------------------------------------------------
MANIFEST_OUT="results/manifests/${DEVICE_NAME}_e2e.json"
if [ -x "scripts/capture_manifest.sh" ]; then
    scripts/capture_manifest.sh --run-id "${DEVICE_NAME}_e2e_${GIT_SHA}" > "$MANIFEST_OUT"
elif command -v python3 >/dev/null 2>&1; then
    python3 bench/manifest.py > "$MANIFEST_OUT" 2>/dev/null || {
        echo "{\"git\":{\"sha\":\"$GIT_SHA\",\"dirty\":$( [ -n "$STATUS" ] && echo true || echo false )}}" > "$MANIFEST_OUT"
    }
else
    echo "{\"git\":{\"sha\":\"$GIT_SHA\",\"dirty\":$( [ -n "$STATUS" ] && echo true || echo false )}}" > "$MANIFEST_OUT"
fi
echo "  Manifest: $MANIFEST_OUT"

# --- 9. convert to schema-conformant CSV -----------------------------------
SCHEMA_OUT="results/raw/${DEVICE_NAME}_e2e_schema.csv"
RUN_ID="${DEVICE_NAME}_e2e_${GIT_SHA}"
MANIFEST_REF="results/manifests/${DEVICE_NAME}_e2e.json"

if command -v python3 >/dev/null 2>&1; then
    python3 bench/convert_e2e_decode.py \
        --raw "$RAW_OUT" \
        --device "$DEVICE_NAME" \
        --output "$SCHEMA_OUT" \
        --run-id "$RUN_ID" \
        --git-sha "$GIT_SHA" \
        --manifest-ref "$MANIFEST_REF" \
        --quantization "$QUANT" \
        --cluster "$CLUSTER" || {
        echo "WARNING: schema conversion failed; raw CSV still available at $RAW_OUT" >&2
    }
    echo "  Schema CSV: $SCHEMA_OUT"
else
    echo "  NOTE: python3 not found; schema conversion skipped. Raw CSV: $RAW_OUT"
fi

# --- 10. summary ------------------------------------------------------------
echo ""
echo "=== E2E Decode Complete ==="
echo "  Commit:  $GIT_SHA"
echo "  Device:  $DEVICE_NAME ($CLUSTER cluster)"
echo "  Tokens:  $TOKENS × $RUNS runs"
echo "  Raw CSV:     $RAW_OUT"
echo "  Schema CSV:  $SCHEMA_OUT"
echo "  Manifest:    $MANIFEST_OUT"
echo ""
echo "Human-readable summary:"
$PIN_CMD "$BINARY" --tokens 2 2>/dev/null | grep -E '^(  TTFT|  Tokens|  Per-token|    p)' || true
echo ""
echo "Commit these files and push to your branch:"
echo "  git add results/ && git commit -m \"${DEVICE_NAME}: e2e decode benchmark (ob-mrd.8)\""
