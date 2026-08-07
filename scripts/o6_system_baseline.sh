#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0
#
# System baseline automation for Arm SoCs — written for the Orion O6 (3-cluster:
# 4×A720 big / 4×A720 med / 4×A520 little) but tested on RK3588 (2-cluster) and
# designed to work on any big.LITTLE / tri-cluster Arm device.
#
# Automates ob-41j: CPU topology detection, governor state, thermal baseline,
# memory info, and ready-to-paste taskset commands for each cluster.
#
# Usage:
#   sudo bash scripts/o6_system_baseline.sh                    # human-readable
#   sudo bash scripts/o6_system_baseline.sh --json > baseline.json  # machine-readable
#   bash scripts/o6_system_baseline.sh --dry-run                # preview without sudo changes
#
# Bead: ob-41j.1.  Parent: ob-41j.
set -euo pipefail

JSON=0
DRY_RUN=0
SUDO=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --json)   JSON=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Read a sysfs value safely (returns empty string if file doesn't exist)
read_sysfs() {
    cat "$1" 2>/dev/null || true
}

# Set governor on all CPUs (needs root). In dry-run mode, just report.
set_governor_all() {
    local target_gov="$1"
    for gov_file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        local cpu_num
        cpu_num=$(echo "$gov_file" | grep -oP 'cpu\K[0-9]+')
        local current_gov
        current_gov=$(read_sysfs "$gov_file")
        if [[ $DRY_RUN -eq 0 ]]; then
            echo "$target_gov" | ${SUDO}tee "$gov_file" >/dev/null 2>&1 || true
            local new_gov
            new_gov=$(read_sysfs "$gov_file")
            if [[ $JSON -eq 0 ]]; then
                printf "  cpu%-2s  %s → %s\n" "$cpu_num" "$current_gov" "$new_gov"
            fi
        else
            if [[ $JSON -eq 0 ]]; then
                printf "  cpu%-2s  %s (dry-run: would set %s)\n" "$cpu_num" "$current_gov" "$target_gov"
            fi
        fi
    done
}

# ---------------------------------------------------------------------------
# 1. CPU Topology Detection — group CPUs by max frequency
# ---------------------------------------------------------------------------

# Returns: "freq:cpu_list" lines, sorted by freq descending
detect_clusters() {
    declare -A freq_map
    for cpu_dir in /sys/devices/system/cpu/cpu[0-9]*/; do
        local cpu_num
        cpu_num=$(basename "$cpu_dir" | grep -oP 'cpu\K[0-9]+')
        local max_freq
        max_freq=$(read_sysfs "${cpu_dir}cpufreq/cpuinfo_max_freq")
        [[ -z "$max_freq" ]] && continue
        freq_map["$max_freq"]+="${cpu_num},"
    done

    # Sort frequencies descending and print
    for freq in $(echo "${!freq_map[@]}" | tr ' ' '\n' | sort -rn); do
        local cpus="${freq_map[$freq]}"
        cpus="${cpus%,}"  # trim trailing comma
        # Convert comma-separated to range format (e.g., "4,5,6,7" → "4-7")
        local range
        range=$(echo "$cpus" | tr ',' '\n' | sort -n | awk '
            NR==1 { start=$1; prev=$1; next }
            $1 == prev+1 { prev=$1; next }
            { if (start==prev) print start; else print start"-"prev; start=$1; prev=$1 }
            END { if (start==prev) print start; else print start"-"prev }
        ' | paste -sd,)
        echo "${freq}|${range}|${cpus}"
    done
}

# ---------------------------------------------------------------------------
# 2. Thermal baseline
# ---------------------------------------------------------------------------

read_thermals() {
    for tz in /sys/class/thermal/thermal_zone*; do
        local tz_type temp
        tz_type=$(read_sysfs "$tz/type")
        temp=$(read_sysfs "$tz/temp")
        [[ -n "$temp" ]] && echo "${tz}|${tz_type}|${temp}"
    done
}

# ---------------------------------------------------------------------------
# 3. Memory info
# ---------------------------------------------------------------------------

read_memory_info() {
    echo "=== Memory ==="
    free -m 2>/dev/null | head -3 || true
    echo ""
    grep -E "MemTotal|MemAvailable|SwapTotal" /proc/meminfo 2>/dev/null || true
    echo ""
    # DMC frequency (RK3588-specific, may not exist on O6)
    local dmc_dev="/sys/class/devfreq/fdab0000.dmc"
    if [[ -d "$dmc_dev" ]]; then
        echo "DMC (RK3588 memory controller):"
        echo "  cur_freq: $(read_sysfs "${dmc_dev}/cur_freq")"
        echo "  available_frequencies: $(read_sysfs "${dmc_dev}/available_frequencies")"
        echo "  governor: $(read_sysfs "${dmc_dev}/governor")"
    fi
}

# ---------------------------------------------------------------------------
# 4. ISA features
# ---------------------------------------------------------------------------

read_isa_features() {
    echo "=== ISA Features ==="
    grep -m1 "^Features" /proc/cpuinfo 2>/dev/null || true
    echo ""
    echo "CPU implementer / part:"
    grep -m1 "^CPU implementer\|^CPU part\|^CPU variant\|^BogoMIPS" /proc/cpuinfo 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Check for sudo
if [[ $DRY_RUN -eq 0 && $(id -u) -ne 0 ]]; then
    if sudo -n true 2>/dev/null; then
        SUDO="sudo "
    else
        echo "WARNING: Not running as root. Governor changes will be attempted with sudo." >&2
        echo "         For full automation, run as root or configure passwordless sudo." >&2
        SUDO="sudo "
    fi
fi

# --- Thermal baseline (before) ---
THERMAL_BEFORE=$(read_thermals)

# --- Governor state (before) ---
GOVERNORS_BEFORE=""
for gov_file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    cpu_num=$(echo "$gov_file" | grep -oP 'cpu\K[0-9]+')
    gov=$(read_sysfs "$gov_file")
    GOVERNORS_BEFORE+="cpu${cpu_num}:${gov} "
done

# --- Set performance governor ---
if [[ $JSON -eq 0 ]]; then
    echo ""
    echo "=== Setting governor to 'performance' ==="
fi
set_governor_all "performance"

# Sleep briefly to let frequencies settle
sleep 0.5

# --- Cluster detection ---
CLUSTERS=$(detect_clusters)

# --- Thermal (after governor change) ---
THERMAL_AFTER=$(read_thermals)

# --- Governor state (after) ---
GOVERNORS_AFTER=""
for gov_file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    cpu_num=$(echo "$gov_file" | grep -oP 'cpu\K[0-9]+')
    gov=$(read_sysfs "$gov_file")
    GOVERNORS_AFTER+="cpu${cpu_num}:${gov} "
done

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

if [[ $JSON -eq 1 ]]; then
    # Machine-readable JSON output
    echo "{"
    echo "  \"device\": \"$(hostname)\","
    echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"kernel\": \"$(uname -r)\","
    echo "  \"arch\": \"$(uname -m)\","

    # Clusters
    echo "  \"clusters\": ["
    jfirst=1
    while IFS='|' read -r freq range cpus; do
        freq_mhz=$((freq / 1000))
        jtotal=$(echo "$CLUSTERS" | wc -l)
        if [[ $jfirst -eq 1 ]]; then
            jlabel="big"
        elif [[ $jtotal -eq 2 ]]; then
            jlabel="little"
        elif [[ $jtotal -eq 3 ]]; then
            if [[ $jfirst -eq 2 ]]; then
                jlabel="medium"
            else
                jlabel="little"
            fi
        else
            jlabel="cluster${jfirst}"
        fi
        [[ $jfirst -eq 0 ]] && echo ","
        echo "    {\"label\": \"${jlabel}\", \"max_freq_khz\": ${freq}, \"max_freq_mhz\": ${freq_mhz}, \"cpu_range\": \"${range}\", \"cpus\": \"${cpus}\"}"
        jfirst=0
    done <<< "$CLUSTERS"
    echo "  ],"

    # Governors
    echo "  \"governors_before\": \"${GOVERNORS_BEFORE%\ }\"," 
    echo "  \"governors_after\": \"${GOVERNORS_AFTER%\ }\","

    # Thermals
    echo "  \"thermal_before\": ["
    jfirst=1
    while IFS='|' read -r tz type temp; do
        [[ -z "$temp" ]] && continue
        [[ $jfirst -eq 0 ]] && echo ","
        echo "    {\"zone\": \"$(basename $tz)\", \"type\": \"${type}\", \"temp_millideg\": ${temp}}"
        jfirst=0
    done <<< "$THERMAL_BEFORE"
    echo "  ],"
    echo "  \"thermal_after\": ["
    jfirst=1
    while IFS='|' read -r tz type temp; do
        [[ -z "$temp" ]] && continue
        [[ $jfirst -eq 0 ]] && echo ","
        echo "    {\"zone\": \"$(basename $tz)\", \"type\": \"${type}\", \"temp_millideg\": ${temp}}"
        jfirst=0
    done <<< "$THERMAL_AFTER"
    echo "  ],"

    # Memory
    jmem_total=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    jmem_avail=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    echo "  \"memory_total_kb\": ${jmem_total:-0},"
    echo "  \"memory_available_kb\": ${jmem_avail:-0}"

    echo "}"
    exit 0
fi

# Human-readable output
echo "================================================================"
echo "  System Baseline: $(hostname)"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Kernel: $(uname -r)  Arch: $(uname -m)"
echo "================================================================"
echo ""

# ISA features
read_isa_features
echo ""

# Clusters
echo "=== CPU Clusters (by max frequency, descending) ==="
cluster_idx=0
while IFS='|' read -r freq range cpus; do
    freq_mhz=$((freq / 1000))
    total_clusters=$(echo "$CLUSTERS" | wc -l)

    if [[ $cluster_idx -eq 0 ]]; then
        label="BIG"
    elif [[ $total_clusters -eq 2 ]]; then
        label="LITTLE"
    elif [[ $cluster_idx -eq 1 ]]; then
        label="MEDIUM"
    else
        label="LITTLE"
    fi

    # Get current freq for first CPU in cluster
    first_cpu=$(echo "$cpus" | cut -d, -f1)
    cur_freq=$(read_sysfs "/sys/devices/system/cpu/cpu${first_cpu}/cpufreq/scaling_cur_freq")
    cur_mhz="${cur_freq:+$((cur_freq / 1000))}"
    avail_freqs=$(read_sysfs "/sys/devices/system/cpu/cpu${first_cpu}/cpufreq/scaling_available_frequencies")

    echo ""
    echo "  [$label] CPUs $range  —  max ${freq_mhz} MHz"
    [[ -n "$cur_mhz" ]] && echo "    current: ${cur_mhz} MHz"
    [[ -n "$avail_freqs" ]] && echo "    available OPPs: $(echo $avail_freqs | tr '\n' ' ')"
    echo "    governor: $(read_sysfs /sys/devices/system/cpu/cpu${first_cpu}/cpufreq/scaling_governor)"
    echo ""
    echo "    taskset -c $range ./bench_gdn_<variant> --repeats 30 --csv > results/raw/${label,,}.csv"
    echo ""

    cluster_idx=$((cluster_idx + 1))
done <<< "$CLUSTERS"

# Memory
read_memory_info
echo ""

# Thermals
echo "=== Thermals ==="
echo "  Before governor change:"
while IFS='|' read -r tz type temp; do
    [[ -z "$temp" ]] && continue
    printf "    %-30s %s\n" "$type" "$(awk "BEGIN{printf \"%.1f°C\", $temp/1000}")"
done <<< "$THERMAL_BEFORE"
echo ""
echo "  After governor change:"
while IFS='|' read -r tz type temp; do
    [[ -z "$temp" ]] && continue
    printf "    %-30s %s\n" "$type" "$(awk "BEGIN{printf \"%.1f°C\", $temp/1000}")"
done <<< "$THERMAL_AFTER"
echo ""

# Governor summary
echo "=== Governors ==="
echo "  Before: ${GOVERNORS_BEFORE}"
echo "  After:  ${GOVERNORS_AFTER}"
echo ""

echo "================================================================"
echo "  Next steps:"
echo "    1. Build:    ./scripts/build_device_bench.sh"
echo "    2. Pin+run:  taskset -c <range> ./dist/bench_gdn_<variant> --repeats 30 --csv"
echo "    3. Provenance: python3 bench/manifest.py > results/manifests/$(hostname).json"
echo "================================================================"
