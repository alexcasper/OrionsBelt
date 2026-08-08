#!/usr/bin/env bash
# Fleet orchestration for the Arm device fleet (bead ob-8ms.4).
#
# ONE orchestrator drives every node in a matched configuration, instead of a
# per-node agent loop on each. Those loops each built at their own commit whenever
# they happened to fire, which is why so many committed results are mixed-commit
# and mixed-thread-count (ob-bf7, ob-mrd.12/14, ob-dpl).
#
# Relationship to scripts/fleet_sweep.sh: that script is the ON-DEVICE protocol —
# clean tree, governor, pinning, manifest — and it stays the thing that runs on a
# node. This one builds once locally and drives fleet_sweep.sh's equivalent steps
# across every node at a single SHA and a single thread count, so the matched
# configuration is enforced BETWEEN nodes rather than only within one.
#
# Run this from a dev box or container, NOT from a fleet node. It only needs ssh
# and scp; the device-side footprint is one ~1.1 MB static binary.
#
# Usage:
#   scripts/fleet.sh status                    # which nodes answer
#   scripts/fleet.sh inventory                 # capability sweep -> results/fleet/
#   scripts/fleet.sh run --threads 1           # matched-config benchmark, all nodes
#   scripts/fleet.sh run --threads 4 --only j2
#   scripts/fleet.sh <cmd> --dry-run           # print what would happen, touch nothing
#
# Testability: every remote call goes through _ssh/_scp, which honour $FLEET_SSH
# and $FLEET_SCP. tests/test_fleet.sh substitutes local shims so the control flow
# is exercised without any hardware.
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NODES_FILE=${FLEET_NODES:-$REPO_ROOT/fleet-nodes.conf}
OUT_DIR=${FLEET_OUT:-$REPO_ROOT/results}
SSH_BIN=${FLEET_SSH:-ssh}
SCP_BIN=${FLEET_SCP:-scp}
SSH_OPTS=${FLEET_SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10}
DRY_RUN=0
THREADS=""
ONLY=""
REPEATS=${FLEET_REPEATS:-30}
ALLOW_DIRTY=0

die() { echo "fleet: $*" >&2; exit 1; }
info() { echo "  $*"; }

_ssh() {
    local host=$1; shift
    if [ "$DRY_RUN" = 1 ]; then echo "    [dry-run] ssh $host $*"; return 0; fi
    # shellcheck disable=SC2086
    $SSH_BIN $SSH_OPTS "$host" "$@"
}

_scp() {
    local src=$1 host=$2 dest=$3
    if [ "$DRY_RUN" = 1 ]; then echo "    [dry-run] scp $src $host:$dest"; return 0; fi
    # shellcheck disable=SC2086
    $SCP_BIN $SSH_OPTS "$src" "$host:$dest"
}

# ---------------------------------------------------------------------------
# Node registry
# ---------------------------------------------------------------------------

# Emits "name ssh cluster label" per node, comments and blanks stripped.
nodes() {
    [ -f "$NODES_FILE" ] || die "no node registry at $NODES_FILE"
    grep -vE '^\s*(#|$)' "$NODES_FILE" | while read -r name ssh cluster label rest; do
        [ -n "${label:-}" ] || die "malformed registry line for '${name:-?}': need 4 columns"
        if [ -n "$ONLY" ] && [ "$name" != "$ONLY" ]; then continue; fi
        echo "$name $ssh $cluster $label"
    done
}

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

require_clean_tree() {
    # A dry run writes no CSV and no manifest, so there is nothing whose provenance
    # could be wrong. Gating it on a clean tree just makes the plumbing untestable
    # while you are still editing the thing you want to test.
    [ "$DRY_RUN" = 1 ] && return 0
    local dirty
    # --untracked-files=no is deliberate and load-bearing. What invalidates a
    # manifest's SHA is a modification to a TRACKED source file, because that is
    # what the binary was built from. New untracked files do not — and the runner's
    # own output lands in results/ as untracked files, so a wholesale
    # `git status --porcelain` made the tree dirty as a side effect of the first
    # run and refused every subsequent one. Sweeping --threads 1 then --threads 4
    # was impossible.
    dirty=$(cd "$REPO_ROOT" && git status --porcelain --untracked-files=no 2>/dev/null | head -1)
    if [ -n "$dirty" ]; then
        if [ "$ALLOW_DIRTY" = 1 ]; then
            echo "fleet: WARNING running from a DIRTY tree (--allow-dirty)." >&2
            echo "fleet: the recorded SHA will not identify the code that ran." >&2
        else
            # Every pre-existing manifest on this fleet records dirty=true, which is
            # the single reason no cross-run comparison is trustworthy. Refuse by
            # default rather than silently producing another unusable datapoint.
            die "working tree is dirty; commit or stash first, or pass --allow-dirty.
     Reason: a dirty tree means the manifest's git SHA does not identify the
     binary that produced the numbers, which is defect ob-bf7."
        fi
    fi
}

git_sha() { (cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown); }

# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

cmd_status() {
    echo "== fleet status ($(date -u +%FT%TZ))"
    local n=0 up=0
    while read -r name ssh _cluster _label; do
        n=$((n + 1))
        printf "  %-4s %-12s " "$name" "$ssh"
        if [ "$DRY_RUN" = 1 ]; then echo "[dry-run]"; up=$((up + 1)); continue; fi
        if _ssh "$ssh" true 2>/dev/null; then echo "up"; up=$((up + 1)); else echo "UNREACHABLE"; fi
    done < <(nodes)
    echo "  $up/$n reachable"
    [ "$up" -gt 0 ] || return 1
}

cmd_inventory() {
    local stamp; stamp=$(date -u +%Y%m%dT%H%M%SZ)
    local dest="$OUT_DIR/fleet"
    [ "$DRY_RUN" = 1 ] || mkdir -p "$dest"
    local out="$dest/inventory-$stamp.json"
    echo "== fleet inventory -> $out"

    local first=1
    { echo "{"; echo "  \"captured_utc\": \"$(date -u +%FT%TZ)\","; echo "  \"orchestrator_git_sha\": \"$(git_sha)\","; echo "  \"nodes\": ["; } > "$out.tmp" 2>/dev/null || true

    while read -r name ssh cluster label; do
        info "$name ($ssh)"
        # detect_isa.sh is deliberately reused rather than reimplemented: it already
        # knows the ISA->binary mapping and runs on Python-less / Python 3.6 nodes.
        _scp "$REPO_ROOT/scripts/detect_isa.sh" "$ssh" "/tmp/detect_isa.sh" >/dev/null 2>&1 || true
        local isa binary mem disk py cores govr therm
        isa=$(_ssh "$ssh" "bash /tmp/detect_isa.sh --json 2>/dev/null || echo '{}'" 2>/dev/null || echo '{}')
        binary=$(_ssh "$ssh" "bash /tmp/detect_isa.sh --binary 2>/dev/null || echo unknown" 2>/dev/null || echo unknown)
        # Free disk matters: it decides which checkpoint can run where (ob-8ms.6).
        disk=$(_ssh "$ssh" "df -Pk / | awk 'NR==2{print \$4}'" 2>/dev/null || echo 0)
        mem=$(_ssh "$ssh" "awk '/MemTotal/{t=\$2} /MemAvailable/{a=\$2} END{print t\" \"a}' /proc/meminfo" 2>/dev/null || echo "0 0")
        py=$(_ssh "$ssh" "python3 -c 'import sys;print(\"%d.%d.%d\"%sys.version_info[:3])' 2>/dev/null || echo none" 2>/dev/null || echo none)
        cores=$(_ssh "$ssh" "nproc" 2>/dev/null || echo 0)
        govr=$(_ssh "$ssh" "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown" 2>/dev/null || echo unknown)
        therm=$(_ssh "$ssh" "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo -1" 2>/dev/null || echo -1)

        if [ "$DRY_RUN" = 1 ]; then continue; fi
        [ "$first" = 1 ] || echo "," >> "$out.tmp"
        first=0
        cat >> "$out.tmp" <<JSON
    {
      "name": "$name", "ssh": "$ssh", "cluster_request": "$cluster", "label": "$label",
      "recommended_binary": "$(echo "$binary" | tr -d '\r\n')",
      "cores": ${cores:-0},
      "mem_total_kb": $(echo "$mem" | awk '{print $1+0}'),
      "mem_available_kb": $(echo "$mem" | awk '{print $2+0}'),
      "disk_avail_kb": ${disk:-0},
      "python3": "$(echo "$py" | tr -d '\r\n')",
      "governor_cpu0": "$(echo "$govr" | tr -d '\r\n')",
      "thermal_zone0_milli_c": ${therm:--1},
      "isa": $(echo "$isa" | tr -d '\r\n' | grep -q '^{' && echo "$isa" | tr -d '\r\n' || echo '{}')
    }
JSON
        # Surface the two facts that actually gate work assignment.
        local py_ok="no"; case "$py" in 3.1[0-9]*) py_ok="yes";; esac
        info "  binary=$(echo "$binary" | tr -d '\r\n')  cores=$cores  avail=$(( ${disk:-0} / 1024 ))MiB disk, $(echo "$mem" | awk '{printf "%d", $2/1024}')MiB RAM  python3=$py (harness-capable: $py_ok)"
    done < <(nodes)

    if [ "$DRY_RUN" = 1 ]; then rm -f "$out.tmp"; echo "  [dry-run] no file written"; return 0; fi
    { echo ""; echo "  ]"; echo "}"; } >> "$out.tmp"
    mv "$out.tmp" "$out"
    echo "  wrote $out"
    command -v python3 >/dev/null && python3 -c "import json,sys;json.load(open('$out'));print('  JSON valid')" || true
}

# Resolve which CPU ids belong to the requested cluster, ON THE NODE.
# DEVICE_RUNBOOK is explicit that assuming cpu0-3 = little is wrong and
# board-dependent, so this reads cpuinfo_max_freq and derives it.
remote_cluster_cpus() {
    local ssh=$1 cluster=$2
    [ "$cluster" = "all" ] && { echo ""; return 0; }
    _ssh "$ssh" "
        for c in /sys/devices/system/cpu/cpu[0-9]*; do
            f=\$(cat \$c/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0)
            echo \"\${c##*/cpu} \$f\"
        done | sort -k2 -n | awk '
            {id[NR]=\$1; fr[NR]=\$2}
            END{
                if (NR==0) exit
                lo=fr[1]; hi=fr[NR]
                for (i=1;i<=NR;i++) {
                    want = (\"$cluster\"==\"big\") ? (fr[i]==hi) : (fr[i]==lo)
                    if (want) printf \"%s%s\", (n++?\",\":\"\"), id[i]
                }
            }'
    " 2>/dev/null || echo ""
}

cmd_run() {
    [ -n "$THREADS" ] || die "run requires --threads N (thread count is a 3-4x variable and must be explicit)"
    require_clean_tree
    local sha; sha=$(git_sha)
    echo "== fleet run: OMP_NUM_THREADS=$THREADS repeats=$REPEATS sha=$sha"

    echo "  building device binaries..."
    if [ "$DRY_RUN" = 1 ]; then
        echo "    [dry-run] bash scripts/build_device_bench.sh"
    else
        (cd "$REPO_ROOT" && bash scripts/build_device_bench.sh >/dev/null 2>&1) \
            || die "build_device_bench.sh failed"
    fi

    local raw="$OUT_DIR/raw" man="$OUT_DIR/manifests"
    [ "$DRY_RUN" = 1 ] || mkdir -p "$raw" "$man"

    while read -r name ssh cluster label; do
        info "$name ($ssh) cluster=$cluster"
        # Ship the detector every run. It used to be copied only by `inventory`,
        # so `run` silently depended on inventory having gone first; when it had
        # not, detection returned nothing, the code fell back to a hardcoded
        # bench_gdn_armv8a, and the node was skipped with a misleading
        # "not built" message instead of "could not detect".
        _scp "$REPO_ROOT/scripts/detect_isa.sh" "$ssh" "/tmp/detect_isa.sh" >/dev/null 2>&1 || true
        local binary; binary=$(_ssh "$ssh" "bash /tmp/detect_isa.sh --binary 2>/dev/null" 2>/dev/null | tr -d '\r\n')
        if [ -z "$binary" ] || [ "$binary" = "unknown" ]; then
            echo "    SKIP: could not detect a binary on $name (is detect_isa.sh runnable there?)" >&2
            continue
        fi
        case "$binary" in bench_gdn_*) ;; *) binary="bench_gdn_$binary";; esac
        local local_bin="$REPO_ROOT/dist/$binary"
        if [ "$DRY_RUN" != 1 ] && [ ! -x "$local_bin" ]; then
            echo "    SKIP: $name wants $binary but $local_bin was not built" >&2; continue
        fi

        _scp "$local_bin" "$ssh" "/tmp/$binary" || { echo "    SKIP: scp failed" >&2; continue; }
        _scp "$REPO_ROOT/scripts/capture_manifest.sh" "$ssh" "/tmp/capture_manifest.sh" >/dev/null 2>&1 || true

        local pin="" cpus=""
        cpus=$(remote_cluster_cpus "$ssh" "$cluster")
        if [ -n "$cpus" ]; then pin="taskset -c $cpus"; info "  pinned to cpus $cpus"; fi

        # Thread count goes in the FILENAME as well as the manifest. jetson-j1_clean.csv
        # was mislabelled precisely because only the manifest was consulted, and a
        # 4-core run got read as a single-threaded baseline.
        local suffix; if [ "$THREADS" = 1 ]; then suffix="st"; else suffix="omp$THREADS"; fi
        local csv="$raw/${label}_${suffix}.csv"
        local mf="$man/${label}_${suffix}.json"

        _ssh "$ssh" "chmod +x /tmp/$binary && OMP_NUM_THREADS=$THREADS $pin /tmp/$binary --repeats $REPEATS --csv" > "${csv}.tmp" 2>/dev/null
        if [ "$DRY_RUN" = 1 ]; then rm -f "${csv}.tmp"; continue; fi
        if [ -s "${csv}.tmp" ]; then mv "${csv}.tmp" "$csv"; info "  -> $csv"; else
            rm -f "${csv}.tmp"; echo "    SKIP: benchmark produced no output" >&2; continue
        fi

        # capture_manifest.sh, not bench/manifest.py: the Jetsons run Python 3.6.9.
        _ssh "$ssh" "OMP_NUM_THREADS=$THREADS bash /tmp/capture_manifest.sh 2>/dev/null" > "${mf}.tmp" 2>/dev/null
        if [ -s "${mf}.tmp" ]; then mv "${mf}.tmp" "$mf"; info "  -> $mf"; else
            rm -f "${mf}.tmp"; echo "    WARNING: no manifest captured — PLAN.md §9 says this is not a result" >&2
        fi

        # Flag noisy rows immediately rather than letting them reach a table. The O6
        # extrapolation was anchored on a row with 153% spread.
        if command -v awk >/dev/null && [ -f "$csv" ]; then
            awk -F, 'NR>1 && $9+0 > 10 {print "    WARNING: spread " $9 "% on " $2 " — re-run before publishing"}' "$csv" >&2 || true
        fi
    done < <(nodes)
    echo "  done. Validate with: python3 scripts/validate_results.py"
}

# ---------------------------------------------------------------------------

main() {
    local cmd=${1:-}; shift || true
    while [ $# -gt 0 ]; do
        case "$1" in
            --dry-run) DRY_RUN=1 ;;
            --threads) THREADS=${2:?--threads needs a value}; shift ;;
            --threads=*) THREADS=${1#*=} ;;
            --only) ONLY=${2:?--only needs a node name}; shift ;;
            --only=*) ONLY=${1#*=} ;;
            --repeats) REPEATS=${2:?--repeats needs a value}; shift ;;
            --repeats=*) REPEATS=${1#*=} ;;
            --allow-dirty) ALLOW_DIRTY=1 ;;
            -h|--help) cmd="help" ;;
            *) die "unknown option: $1" ;;
        esac
        shift
    done

    case "$cmd" in
        status) cmd_status ;;
        inventory) cmd_inventory ;;
        run) cmd_run ;;
        help|"") sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
        *) die "unknown subcommand '$cmd' (try: status, inventory, run)" ;;
    esac
}

main "$@"
