#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

# Shell-based run manifest — provenance capture without Python.
#
# bench/manifest.py requires Python 3.10+. Edge devices like the Jetson Nano
# run Python 3.6.9 and cannot execute it, forcing manual JSON creation.
# This script captures the same provenance fields using only bash and
# standard /proc /sys reads, outputting schema-compatible JSON.
#
# Bead ob-mrd.4. Usage:
#   ./scripts/capture_manifest.sh > results/manifests/<device>.json
#   ./scripts/capture_manifest.sh --run-id custom_id > results/manifests/custom.json
#
# Same graceful-degradation principle as manifest.py: missing fields become
# null, the script never crashes. Every probe is individually guarded.
set -euo pipefail

# --- helpers ---

json_escape() {
    # Escape a string for JSON: backslash, quote, control chars
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\t'/\\t}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\r'/\\r}"
    printf '%s' "$s"
}

read_file() {
    local f="$1"
    if [ -r "$f" ]; then cat "$f" 2>/dev/null || true; fi
}

read_trim() {
    local f="$1"
    local v
    v=$(read_file "$f")
    printf '%s' "${v//[[:space:]]/}"
}

# --- run identity ---

HOSTNAME_S=$(hostname 2>/dev/null || echo "unknown")
HOSTNAME_S=$(json_escape "$HOSTNAME_S")
TIMESTAMP_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo "unknown")

GIT_SHA="null"
GIT_DIRTY="null"
if git rev-parse --git-dir >/dev/null 2>&1; then
    SHA=$(git rev-parse HEAD 2>/dev/null || true)
    if [ -n "$SHA" ]; then
        GIT_SHA="\"$(json_escape "$SHA")\""
        SHORT_SHA="${SHA:0:7}"
        # Exclude results/ and .beads/ from dirty check — these are output
        # data (bench CSVs, manifests, thermal snapshots, beads export), not
        # source code.  A dirty flag from writing benchmark output is misleading
        # and causes exactly the provenance confusion ob-bf7 documents.
        STATUS=$(git status --porcelain 2>/dev/null | grep -vE '^[ ?][M?] (results/|\.beads/)' || true)
        if [ -n "$STATUS" ]; then
            GIT_DIRTY="true"
        else
            GIT_DIRTY="false"
        fi
    else
        SHORT_SHA="nogit"
    fi
else
    SHORT_SHA="nogit"
fi

# Allow override of run_id
RUN_ID=""
 while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id) RUN_ID="$2"; shift 2 ;;
        *) shift ;;
    esac
done
if [ -z "$RUN_ID" ]; then
    STAMP=$(date -u +"%Y%m%dT%H%M%SZ" 2>/dev/null || echo "unknown")
    RUN_ID="${HOSTNAME_S//\"/}_${STAMP}_${SHORT_SHA}"
fi
RUN_ID_ESC=$(json_escape "$RUN_ID")

# --- host info ---

ARCH=$(uname -m 2>/dev/null || echo "unknown")
KERNEL=$(uname -r 2>/dev/null || echo "unknown")
OS_STR=$(uname -s 2>/dev/null || echo "unknown")
if [ -n "$(uname -v 2>/dev/null)" ]; then
    OS_FULL="${OS_STR} $(uname -v 2>/dev/null)"
else
    OS_FULL="$OS_STR"
fi

# CPU model from /proc/cpuinfo
CPU_MODEL="null"
CPUINFO=$(read_file /proc/cpuinfo)
if [ -n "$CPUINFO" ]; then
    for key in "model name" "Hardware" "Model"; do
        LINE=$(echo "$CPUINFO" | grep -m1 "^$key" 2>/dev/null || true)
        if [ -n "$LINE" ]; then
            VAL="${LINE#*:}"
            VAL="${VAL#"${VAL%%[![:space:]]*}"}"  # ltrim
            CPU_MODEL="\"$(json_escape "$VAL")\""
            break
        fi
    done
fi

# Core count
CORE_COUNT=$(nproc 2>/dev/null || grep -c ^processor /proc/cpuinfo 2>/dev/null || echo "null")

# --- ISA features (aarch64 only) ---

ISA_JSON="null"
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    FEATURES_LINE=$(echo "$CPUINFO" | grep -m1 '^Features' 2>/dev/null || true)
    if [ -n "$FEATURES_LINE" ]; then
        FEATURES_STR="${FEATURES_LINE#*:}"
        ISA_JSON="{"
        FIRST=1
        for feat in sve sve2 i8mm bf16 asimddp sme; do
            if [ "$FIRST" -eq 0 ]; then ISA_JSON+=","; fi
            if echo "$FEATURES_STR" | grep -qw "$feat" 2>/dev/null; then
                ISA_JSON+="\"$feat\":true"
            else
                ISA_JSON+="\"$feat\":false"
            fi
            FIRST=0
        done
        ISA_JSON+="}"
    fi
fi

# --- CPU topology (per-core freq/governor/capacity) ---

TOPOLOGY_JSON="null"
CPU_BASE="/sys/devices/system/cpu"
if [ -d "$CPU_BASE" ]; then
    CPU_DIRS=$(ls -d "$CPU_BASE"/cpu[0-9]* 2>/dev/null | sort -t 'u' -k4 -n || true)
    if [ -n "$CPU_DIRS" ]; then
        TOPOLOGY_JSON="["
        FIRST=1
        for cpu_dir in $CPU_DIRS; do
            if [ "$FIRST" -eq 0 ]; then TOPOLOGY_JSON+=","; fi
            FIRST=0
            CPU_ID=$(basename "$cpu_dir" | sed 's/cpu//')

            MAX_FREQ=$(read_trim "$cpu_dir/cpufreq/scaling_max_freq" 2>/dev/null || true)
            MIN_FREQ=$(read_trim "$cpu_dir/cpufreq/scaling_min_freq" 2>/dev/null || true)
            GOVERNOR=$(read_trim "$cpu_dir/cpufreq/scaling_governor" 2>/dev/null || true)
            CAPACITY=$(read_trim "$cpu_dir/cpu_capacity" 2>/dev/null || true)

            [ -z "$MAX_FREQ" ] && MAX_FREQ_JSON="null" || MAX_FREQ_JSON="$MAX_FREQ"
            [ -z "$MIN_FREQ" ] && MIN_FREQ_JSON="null" || MIN_FREQ_JSON="$MIN_FREQ"
            [ -z "$GOVERNOR" ] && GOV_JSON="null" || GOV_JSON="\"$(json_escape "$GOVERNOR")\""
            [ -z "$CAPACITY" ] && CAP_JSON="null" || CAP_JSON="$CAPACITY"

            TOPOLOGY_JSON+="{\"cpu\":$CPU_ID,\"max_freq_khz\":$MAX_FREQ_JSON,\"min_freq_khz\":$MIN_FREQ_JSON,\"governor\":$GOV_JSON,\"cpu_capacity\":$CAP_JSON}"
        done
        TOPOLOGY_JSON+="]"
    fi
fi

# --- thermal zones ---

THERMAL_JSON="null"
if [ -d /sys/class/thermal ]; then
    ZONES=$(ls -d /sys/class/thermal/thermal_zone* 2>/dev/null | sort || true)
    if [ -n "$ZONES" ]; then
        THERMAL_JSON="["
        FIRST=1
        for zone_dir in $ZONES; do
            if [ "$FIRST" -eq 0 ]; then THERMAL_JSON+=","; fi
            FIRST=0
            ZONE_NAME=$(basename "$zone_dir")
            ZONE_TYPE=$(read_trim "$zone_dir/type" 2>/dev/null || true)
            ZONE_TEMP=$(read_trim "$zone_dir/temp" 2>/dev/null || true)
            [ -z "$ZONE_TYPE" ] && TYPE_JSON="null" || TYPE_JSON="\"$(json_escape "$ZONE_TYPE")\""
            [ -z "$ZONE_TEMP" ] && TEMP_JSON="null" || TEMP_JSON="$ZONE_TEMP"
            THERMAL_JSON+="{\"zone\":\"$ZONE_NAME\",\"type\":$TYPE_JSON,\"temp_millicelsius\":$TEMP_JSON}"
        done
        THERMAL_JSON+="]"
    fi
fi

# --- memory ---

MEM_TOTAL="null"
MEM_AVAIL="null"
MEMINFO=$(read_file /proc/meminfo)
if [ -n "$MEMINFO" ]; then
    LINE=$(echo "$MEMINFO" | grep -m1 '^MemTotal:' 2>/dev/null || true)
    if [ -n "$LINE" ]; then MEM_TOTAL=$(echo "$LINE" | grep -oE '[0-9]+' | head -1); fi
    LINE=$(echo "$MEMINFO" | grep -m1 '^MemAvailable:' 2>/dev/null || true)
    if [ -n "$LINE" ]; then MEM_AVAIL=$(echo "$LINE" | grep -oE '[0-9]+' | head -1); fi
fi

# --- Python version (best effort) ---

PY_VER="null"
if command -v python3 >/dev/null 2>&1; then
    PY_VER_RAW=$(python3 --version 2>&1 || true)
    [ -n "$PY_VER_RAW" ] && PY_VER="\"$(json_escape "$PY_VER_RAW")\""
fi

# --- output JSON ---

# --- parallelism -----------------------------------------------------------
# Thread count became a 4x experimental variable when the kernels gained OpenMP:
# a 1-thread and a 4-core run of the SAME commit on the SAME device differ 3-4x,
# and nothing recorded which one a CSV came from. jetson-j1_clean.csv is exactly
# that trap -- captured to answer ob-bf7, it reads 2.9-4.1x its predecessor
# because it is a 4-core run, not because the tree was clean.
if [ -n "${OMP_NUM_THREADS:-}" ]; then
  OMP_THREADS_JSON="\"$OMP_NUM_THREADS\""
  EFFECTIVE_THREADS="$OMP_NUM_THREADS"
  THREADS_SOURCE="OMP_NUM_THREADS"
else
  OMP_THREADS_JSON="null"
  # libgomp defaults to one thread per available CPU when the var is unset.
  EFFECTIVE_THREADS="$CORE_COUNT"
  THREADS_SOURCE="core_count_default"
fi
[ -n "${OMP_PROC_BIND:-}" ] && OMP_BIND_JSON="\"$OMP_PROC_BIND\"" || OMP_BIND_JSON="null"
[ -n "${OMP_PLACES:-}" ] && OMP_PLACES_JSON="\"$OMP_PLACES\"" || OMP_PLACES_JSON="null"

cat <<JSONEOF
{
  "manifest_version": 1,
  "run_id": "$RUN_ID_ESC",
  "timestamp_utc": "$TIMESTAMP_UTC",
  "source": "scripts/capture_manifest.sh",
  "git": {
    "sha": $GIT_SHA,
    "dirty": $GIT_DIRTY
  },
  "host": {
    "hostname": "$HOSTNAME_S",
    "machine": "$ARCH",
    "kernel": "$KERNEL",
    "os": "$(json_escape "$OS_FULL")",
    "cpu_model": $CPU_MODEL,
    "core_count": $CORE_COUNT,
    "cpu_topology": $TOPOLOGY_JSON
  },
  "isa_features": $ISA_JSON,
  "parallelism": {
    "omp_num_threads": $OMP_THREADS_JSON,
    "omp_proc_bind": $OMP_BIND_JSON,
    "omp_places": $OMP_PLACES_JSON,
    "effective_threads": $EFFECTIVE_THREADS,
    "threads_source": "$THREADS_SOURCE"
  },
  "thermal_zones": $THERMAL_JSON,
  "memory": {
    "mem_total_kb": $MEM_TOTAL,
    "mem_available_kb": $MEM_AVAIL
  },
  "software": {
    "python_version": $PY_VER
  }
}
JSONEOF
