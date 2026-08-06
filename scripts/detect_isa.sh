#!/usr/bin/env bash
# Standalone ISA feature detection for aarch64 devices (bead ob-ng6).
#
# Works on ANY device — no Python required. Mirrors the logic of
# src/orionsbelt/engines/cpu/isa_detect.py but as a shell script so it
# runs on devices with old Python (e.g. Jetson Nano's 3.6.9) or no Python
# at all.
#
# Usage:
#   scripts/detect_isa.sh                  # human-readable
#   scripts/detect_isa.sh --json           # machine-readable JSON
#   scripts/detect_isa.sh --binary         # print only the recommended bench binary name
#
# Output (human-readable):
#   CPU part: 0xd07 (Cortex-A57)
#   Active ISA: asimd
#   Recommended binary: armv8a
#
# Output (JSON):
#   {"cpu_part":"0xd07","cpu_part_name":"Cortex-A57","features":["fp","asimd",...],
#    "dispatch":{"asimd":true,"asimddp":false,"i8mm":false,...},
#    "recommended_binary":"armv8a","core_count":4}
set -euo pipefail

ARCH="${ARCH:-$(uname -m 2>/dev/null || echo "unknown")}"
CPUINFO=""
if [ -r /proc/cpuinfo ]; then
    CPUINFO=$(cat /proc/cpuinfo)
fi

# ---------------------------------------------------------------------------
# Parse features and CPU part from /proc/cpuinfo
# ---------------------------------------------------------------------------
FEATURES_STR=""
CPU_PART=""
CPU_IMPLEMENTER=""
CORE_COUNT=0

if [ -n "$CPUINFO" ]; then
    FEATURES_STR=$(echo "$CPUINFO" | grep -m1 '^Features' 2>/dev/null | sed 's/^Features[[:space:]]*:[[:space:]]*//' || true)
    CPU_PART=$(echo "$CPUINFO" | grep -m1 '^CPU part' 2>/dev/null | sed 's/^CPU part[[:space:]]*:[[:space:]]*//' || true)
    CPU_IMPLEMENTER=$(echo "$CPUINFO" | grep -m1 '^CPU implementer' 2>/dev/null | sed 's/^CPU implementer[[:space:]]*:[[:space:]]*//' || true)
    CORE_COUNT=$(echo "$CPUINFO" | grep -c '^processor' 2>/dev/null || echo 1)
fi
[ -z "$CORE_COUNT" ] && CORE_COUNT=1

# ---------------------------------------------------------------------------
# CPU part number -> common name (matches isa_detect.py _CPU_PARTS)
# ---------------------------------------------------------------------------
cpu_part_name() {
    local part="$1"
    case "$(echo "$part" | tr 'A-F' 'a-f')" in
        0xd01) echo "Cortex-A32" ;;
        0xd03) echo "Cortex-A53" ;;
        0xd04) echo "Cortex-A35" ;;
        0xd05) echo "Cortex-A55" ;;
        0xd07) echo "Cortex-A57" ;;
        0xd08) echo "Cortex-A72" ;;
        0xd09) echo "Cortex-A73" ;;
        0xd0a) echo "Cortex-A75" ;;
        0xd0b) echo "Cortex-A76" ;;
        0xd0c) echo "Neoverse-N1" ;;
        0xd0d) echo "Cortex-A77" ;;
        0xd40) echo "Neoverse-V1" ;;
        0xd41) echo "Cortex-A78" ;;
        0xd44) echo "Cortex-X1C" ;;
        0xd4a) echo "Neoverse-E1" ;;
        0xd4b) echo "Cortex-A78C" ;;
        0xd4d) echo "Cortex-A715" ;;
        0xd4e) echo "Cortex-X4" ;;
        0xd4f) echo "Neoverse-V2" ;;
        0xd80) echo "Cortex-A520" ;;
        0xd81) echo "Cortex-A720" ;;
        *)     echo "${part:-unknown}" ;;
    esac
}

# ---------------------------------------------------------------------------
# Check whether a feature string is present in the Features line
# ---------------------------------------------------------------------------
has_feat() {
    echo "$FEATURES_STR" | grep -qw "$1" 2>/dev/null
}

# Dispatch-critical features (same set as isa_detect.py _DISPATCH_FEATURES)
DISPATCH_FEATS="asimd asimddp i8mm sve sve2 bf16 fphp asimdhp"

# Determine recommended binary
recommend_binary() {
    if has_feat sve2;  then echo "armv9sve2"
    elif has_feat i8mm; then echo "armv8.6i8mm"
    elif has_feat asimddp; then echo "armv8.2dot"
    elif has_feat asimd; then echo "armv8a"
    else echo "scalar"
    fi
}

PART_NAME=$(cpu_part_name "$CPU_PART")
RECOMMENDED=$(recommend_binary)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_MODE="${1:-human}"

if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
    if [ "$OUTPUT_MODE" = "--json" ]; then
        echo "{\"arch\":\"$ARCH\",\"error\":\"not aarch64\",\"recommended_binary\":\"scalar\"}"
    elif [ "$OUTPUT_MODE" = "--binary" ]; then
        echo "scalar"
    else
        echo "Architecture: $ARCH (not aarch64 — ISA detection is aarch64-only)"
        echo "Recommended binary: scalar"
    fi
    exit 0
fi

if [ "$OUTPUT_MODE" = "--binary" ]; then
    echo "$RECOMMENDED"
    exit 0
fi

if [ "$OUTPUT_MODE" = "--json" ]; then
    # Build dispatch JSON
    DISP_JSON=""
    FIRST=1
    for feat in $DISPATCH_FEATS; do
        [ "$FIRST" -eq 0 ] && DISP_JSON+=","
        if has_feat "$feat"; then
            DISP_JSON+="\"$feat\":true"
        else
            DISP_JSON+="\"$feat\":false"
        fi
        FIRST=0
    done

    # Build features array
    FEAT_JSON="["
    FIRST=1
    for feat in $FEATURES_STR; do
        [ "$FIRST" -eq 0 ] && FEAT_JSON+=","
        FEAT_JSON+="\"$feat\""
        FIRST=0
    done
    FEAT_JSON+="]"

    echo "{\"arch\":\"aarch64\",\"cpu_part\":\"$CPU_PART\",\"cpu_part_name\":\"$PART_NAME\",\"cpu_implementer\":\"$CPU_IMPLEMENTER\",\"features\":$FEAT_JSON,\"dispatch\":{$DISP_JSON},\"recommended_binary\":\"$RECOMMENDED\",\"core_count\":$CORE_COUNT}"
    exit 0
fi

# Human-readable output
echo "=== ISA Feature Detection ==="
echo "Architecture: $ARCH"
echo "CPU part: $CPU_PART ($PART_NAME)"
if [ -n "$CPU_IMPLEMENTER" ]; then
    case "$CPU_IMPLEMENTER" in
        0x41) echo "CPU implementer: $CPU_IMPLEMENTER (ARM)" ;;
        *)    echo "CPU implementer: $CPU_IMPLEMENTER" ;;
    esac
fi
echo "Core count: $CORE_COUNT"
echo ""
echo "All features: ${FEATURES_STR:-(none)}"
echo ""
echo "Dispatch-critical features:"
for feat in $DISPATCH_FEATS; do
    if has_feat "$feat"; then
        printf "  %-12s YES\n" "$feat"
    else
        printf "  %-12s no\n" "$feat"
    fi
done
echo ""
echo "Recommended bench binary: $RECOMMENDED"
echo ""
echo "Usage:"
echo "  ./dist/bench_gdn_${RECOMMENDED} --repeats 30            # human-readable"
echo "  ./dist/bench_gdn_${RECOMMENDED} --repeats 30 --csv      # CSV output"
