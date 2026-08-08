#!/usr/bin/env bash
# Tests for scripts/fleet.sh with NO hardware (bead ob-8ms.4).
#
# fleet.sh routes every remote call through _ssh/_scp, which honour $FLEET_SSH and
# $FLEET_SCP. This substitutes local shims that impersonate a fleet: a Jetson-like
# node (Python 3.6.9, 4 equal-frequency cores) and an RK3588-like node (asymmetric
# 4+4 cores). That exercises the logic that actually tends to be wrong — cluster
# resolution, thread-count labelling, the dirty-tree guard, spread warnings —
# rather than just checking the script parses.
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); echo "  ok   - $1"; }
bad()  { FAIL=$((FAIL+1)); echo "  FAIL - $1"; [ -n "${2:-}" ] && echo "         $2"; }
check(){ if [ "$2" = "$3" ]; then ok "$1"; else bad "$1" "expected '$3', got '$2'"; fi; }
contains(){ if echo "$2" | grep -qF "$3"; then ok "$1"; else bad "$1" "output lacked '$3'"; fi; }
lacks(){ if echo "$2" | grep -qF "$3"; then bad "$1" "output unexpectedly had '$3'"; else ok "$1"; fi; }

# --- fake ssh -------------------------------------------------------------
# Impersonates two node types. Frequencies are the interesting part: the RK3588
# fake reports 4 slow + 4 fast cores so "big"/"little" resolution has something
# real to resolve, and deliberately puts the FAST cores at cpu4-7 and slow at
# cpu0-3 — the layout DEVICE_RUNBOOK warns must not be assumed.
cat > "$TMP/fake_ssh" <<'SH'
#!/usr/bin/env bash
args=(); host=""
for a in "$@"; do
  case "$a" in -o) shift 2>/dev/null;; BatchMode=*|ConnectTimeout=*) ;;
    *) if [ -z "$host" ] && [[ "$a" == fake-* ]]; then host="$a"; else args+=("$a"); fi;;
  esac
done
cmd="${args[*]}"
[ -n "${FAKE_UNREACHABLE:-}" ] && [ "$host" = "$FAKE_UNREACHABLE" ] && exit 255
case "$cmd" in
  true) exit 0 ;;
esac
case "$host" in
  fake-jetson)
    case "$cmd" in
      *detect_isa.sh\ --binary*) echo "bench_gdn_jetson_a57" ;;
      *detect_isa.sh\ --json*)   echo '{"sve":false,"asimddp":false}' ;;
      *nproc*)                   echo 4 ;;
      *meminfo*)                 echo "4055040 2768000" ;;
      *df*)                      echo 1048576 ;;
      *version_info*)            echo "3.6.9" ;;
      *scaling_governor*)        echo performance ;;
      *thermal_zone0/temp*)      echo 41000 ;;
      *cpuinfo_max_freq*)        for i in 0 1 2 3; do echo "$i 1479000"; done | sort -k2 -n | awk '{id[NR]=$1;fr[NR]=$2} END{lo=fr[1];hi=fr[NR];for(i=1;i<=NR;i++){w=(ENVIRON["C"]=="big")?(fr[i]==hi):(fr[i]==lo); if(w) printf "%s%s",(n++?",":""),id[i]}}' ;;
      *bench_gdn*--csv*)
        echo "model,kernel,dispatch_path,seq,channels,repeats,p50_us,p95_us,spread_pct,gib_per_s_p50,gflop_per_s_p50"
        echo "Qwen3.5-4B,gdn_gated_scan,neon,64,4096,30,4084.7,4788.7,17.2,0.72,0.13"
        echo "Qwen3.5-4B,gdn_cumdecay,neon,64,4096,30,1690.7,1750.0,3.5,1.16,0.16" ;;
      *capture_manifest.sh*)
        t=$(echo "$cmd" | sed -n 's/.*OMP_NUM_THREADS=\([0-9]*\).*/\1/p'); t=${t:-null}
        echo "{\"manifest_version\":1,\"git\":{\"sha\":\"deadbee\",\"dirty\":false},\"parallelism\":{\"omp_num_threads\":\"$t\"}}" ;;
      *chmod*) ;;
    esac ;;
  fake-rk1)
    case "$cmd" in
      *detect_isa.sh\ --binary*) echo "bench_gdn_rk3588_a76" ;;
      *detect_isa.sh\ --json*)   echo '{"sve":false,"asimddp":true}' ;;
      *nproc*)                   echo 8 ;;
      *meminfo*)                 echo "8127040 6000000" ;;
      *df*)                      echo 3145728 ;;
      *version_info*)            echo "3.10.12" ;;
      *scaling_governor*)        echo performance ;;
      *thermal_zone0/temp*)      echo 43000 ;;
      *bench_gdn*--csv*)
        echo "model,kernel,dispatch_path,seq,channels,repeats,p50_us,p95_us,spread_pct,gib_per_s_p50,gflop_per_s_p50"
        echo "Qwen3.5-4B,gdn_gated_scan,neon,64,4096,30,898.9,1055.3,17.4,3.29,0.58" ;;
      *capture_manifest.sh*)
        t=$(echo "$cmd" | sed -n 's/.*OMP_NUM_THREADS=\([0-9]*\).*/\1/p'); t=${t:-null}
        echo "{\"manifest_version\":1,\"git\":{\"sha\":\"deadbee\",\"dirty\":false},\"parallelism\":{\"omp_num_threads\":\"$t\"}}" ;;
      *chmod*) ;;
    esac
    # Asymmetric: slow cores at 0-3, FAST at 4-7.
    if [[ "$cmd" == *cpuinfo_max_freq* ]]; then
      { for i in 0 1 2 3; do echo "$i 1800000"; done; for i in 4 5 6 7; do echo "$i 2400000"; done; } \
        | sort -k2 -n | awk -v c="${C:-big}" '{id[NR]=$1;fr[NR]=$2} END{lo=fr[1];hi=fr[NR];for(i=1;i<=NR;i++){w=(c=="big")?(fr[i]==hi):(fr[i]==lo); if(w) printf "%s%s",(n++?",":""),id[i]}}'
    fi ;;
esac
exit 0
SH
chmod +x "$TMP/fake_ssh"
printf '#!/usr/bin/env bash\nexit 0\n' > "$TMP/fake_scp"; chmod +x "$TMP/fake_scp"

cat > "$TMP/nodes.conf" <<'CONF'
# name  ssh           cluster  label
fj      fake-jetson   all      fake-jetson
fr      fake-rk1      big      fake-rk1
CONF

export FLEET_SSH="$TMP/fake_ssh" FLEET_SCP="$TMP/fake_scp"
export FLEET_NODES="$TMP/nodes.conf" FLEET_OUT="$TMP/results"
export FLEET_SSH_OPTS=""
F="$REPO_ROOT/scripts/fleet.sh"

echo "== fleet.sh"

# --- basics ---------------------------------------------------------------
bash -n "$F" && ok "passes bash -n" || bad "passes bash -n"
out=$(bash "$F" help 2>&1); contains "help mentions subcommands" "$out" "inventory"

# --- status ---------------------------------------------------------------
out=$(bash "$F" status 2>&1)
contains "status reports both nodes up" "$out" "2/2 reachable"
out=$(FAKE_UNREACHABLE=fake-rk1 bash "$F" status 2>&1)
contains "status detects an unreachable node" "$out" "UNREACHABLE"
contains "status counts only reachable nodes" "$out" "1/2 reachable"

# --- --only filter --------------------------------------------------------
out=$(bash "$F" status --only fj 2>&1)
contains "--only limits to one node" "$out" "1/1 reachable"
lacks "--only excludes the other node" "$out" "fake-rk1"

# --- inventory ------------------------------------------------------------
out=$(bash "$F" inventory 2>&1)
contains "inventory records free disk" "$out" "MiB disk"
contains "inventory flags the Python floor" "$out" "harness-capable: no"
contains "inventory sees the capable node" "$out" "harness-capable: yes"
inv=$(ls "$TMP/results/fleet/"inventory-*.json 2>/dev/null | head -1)
if [ -n "$inv" ]; then
  ok "inventory wrote a file"
  if command -v python3 >/dev/null; then
    python3 - "$inv" <<'PY' && ok "inventory JSON is valid and complete" || bad "inventory JSON invalid/incomplete"
import json,sys
d=json.load(open(sys.argv[1]))
assert len(d["nodes"])==2, d
j=[n for n in d["nodes"] if n["name"]=="fj"][0]
assert j["python3"].startswith("3.6"), j
assert j["mem_available_kb"]==2768000, j
assert j["disk_avail_kb"]==1048576, j
assert j["recommended_binary"]=="bench_gdn_jetson_a57", j
PY
  fi
else bad "inventory wrote a file"; fi

# --- run: guards ----------------------------------------------------------
out=$(bash "$F" run 2>&1); contains "run refuses without --threads" "$out" "requires --threads"

# Dirty-tree guard: fleet.sh resolves REPO_ROOT from BASH_SOURCE, not cwd.
# Copy it into a scratch repo so we can dirty THAT tree and exercise the real
# guard (ob-7cf: the old test ran the real fleet.sh from a scratch cwd, but
# fleet.sh always checks its own checkout — clean — so the guard never fired
# and cmd_run wrote a stray CSV into the shared FLEET_OUT, breaking the
# later 'dry-run writes no CSV' assertion via cross-test contamination).
mkdir -p "$TMP/dirty/scripts" "$TMP/dirty/dist"
cp "$F" "$TMP/dirty/scripts/fleet.sh"
cp "$REPO_ROOT/scripts/detect_isa.sh" "$TMP/dirty/scripts/" 2>/dev/null || true
cp "$REPO_ROOT/scripts/capture_manifest.sh" "$TMP/dirty/scripts/" 2>/dev/null || true
printf '#!/usr/bin/env bash\nexit 0\n' > "$TMP/dirty/scripts/build_device_bench.sh"
for b in bench_gdn_jetson_a57 bench_gdn_rk3588_a76; do
  printf '#!/usr/bin/env bash\nexit 0\n' > "$TMP/dirty/dist/$b"; chmod +x "$TMP/dirty/dist/$b"
done
(cd "$TMP/dirty" && git init -q . && git add -A && git -c user.email=t@t -c user.name=t commit -q -m stage)
echo x >> "$TMP/dirty/scripts/build_device_bench.sh"   # dirty the tracked tree
out=$(cd "$TMP/dirty" && FLEET_NODES="$TMP/nodes.conf" FLEET_OUT="$TMP/dirty-out" \
      bash scripts/fleet.sh run --threads 1 2>&1)
contains "dirty-tree guard fires and refuses" "$out" "working tree is dirty"
contains "dirty-tree guard cites the provenance defect" "$out" "ob-bf7"
lacks "dirty-tree guard leaves no CSV behind" "$(ls "$TMP/dirty-out/raw" 2>/dev/null || true)" ".csv"

# --- run: dry-run plumbing ------------------------------------------------
out=$(bash "$F" run --threads 1 --dry-run 2>&1)
contains "dry-run shows the build step" "$out" "build_device_bench.sh"
lacks "dry-run writes no CSV" "$(ls "$TMP/results/raw" 2>/dev/null || true)" ".csv"

# --- run: the three things most likely to be wrong ------------------------
# A real (non-dry) run needs a clean tree, so stage one in scratch with just the
# pieces fleet.sh reaches for. This is where cluster resolution, thread-count
# labelling and the spread warning actually get exercised.
STAGE="$TMP/stage"
mkdir -p "$STAGE/scripts" "$STAGE/dist"
cp "$F" "$STAGE/scripts/fleet.sh"
cp "$REPO_ROOT/scripts/detect_isa.sh" "$STAGE/scripts/" 2>/dev/null || true
cp "$REPO_ROOT/scripts/capture_manifest.sh" "$STAGE/scripts/" 2>/dev/null || true
# build_device_bench.sh is stubbed: the real one needs a cross-compiler, and what
# is under test here is the orchestration, not the build.
printf '#!/usr/bin/env bash\nexit 0\n' > "$STAGE/scripts/build_device_bench.sh"
for b in bench_gdn_jetson_a57 bench_gdn_rk3588_a76; do
  printf '#!/usr/bin/env bash\nexit 0\n' > "$STAGE/dist/$b"; chmod +x "$STAGE/dist/$b"
done
(cd "$STAGE" && git init -q . && git add -A && git -c user.email=t@t -c user.name=t commit -q -m stage)

out=$(cd "$STAGE" && FLEET_NODES="$TMP/nodes.conf" FLEET_OUT="$STAGE/results" \
      bash scripts/fleet.sh run --threads 4 --repeats 30 2>&1)

# The RK1 fake reports slow cores at 0-3 and FAST at 4-7 — the layout the runbook
# says must never be assumed. "big" must resolve to 4,5,6,7 from cpuinfo_max_freq.
contains "resolves the big cluster from cpufreq, not assumption" "$out" "pinned to cpus 4,5,6,7"
lacks "does not pin the symmetric Jetson" "$out" "pinned to cpus 0,1,2,3"

# Thread count must reach the FILENAME. jetson-j1_clean.csv was misread precisely
# because only the manifest carried it.
if [ -f "$STAGE/results/raw/fake-rk1_omp4.csv" ]; then ok "thread count lands in the filename (_omp4)"; else bad "thread count lands in the filename (_omp4)" "$(ls "$STAGE/results/raw" 2>/dev/null)"; fi
if [ -f "$STAGE/results/manifests/fake-rk1_omp4.json" ]; then ok "manifest captured alongside" ; else bad "manifest captured alongside"; fi

# OMP_NUM_THREADS must reach the node, not just the local filename.
if grep -q '"omp_num_threads":"4"' "$STAGE/results/manifests/fake-rk1_omp4.json" 2>/dev/null; then
  ok "OMP_NUM_THREADS is exported to the remote"
else bad "OMP_NUM_THREADS is exported to the remote" "$(cat "$STAGE/results/manifests/fake-rk1_omp4.json" 2>/dev/null)"; fi

# Both fakes emit a >10% spread row; it must be flagged, not silently published.
contains "warns on rows above the spread threshold" "$out" "WARNING: spread"

# And single-threaded must label differently.
out=$(cd "$STAGE" && FLEET_NODES="$TMP/nodes.conf" FLEET_OUT="$STAGE/results" \
      bash scripts/fleet.sh run --threads 1 --only fj 2>&1)
if [ -f "$STAGE/results/raw/fake-jetson_st.csv" ]; then ok "single-threaded labelled _st" ; else bad "single-threaded labelled _st" "files: $(ls "$STAGE/results/raw") || run output: $out"; fi
lacks "a second sweep is not blocked by its own output" "$out" "working tree is dirty"

# But a genuinely modified SOURCE file must still block, since that is what makes a
# manifest SHA a lie.
(cd "$STAGE" && echo "# touched" >> scripts/build_device_bench.sh)
out=$(cd "$STAGE" && FLEET_NODES="$TMP/nodes.conf" FLEET_OUT="$STAGE/results" \
      bash scripts/fleet.sh run --threads 1 --only fj 2>&1)
contains "modified tracked source still blocks a run" "$out" "working tree is dirty"
out=$(cd "$STAGE" && FLEET_NODES="$TMP/nodes.conf" FLEET_OUT="$STAGE/results" \
      bash scripts/fleet.sh run --threads 1 --only fj --allow-dirty 2>&1)
contains "--allow-dirty overrides but warns" "$out" "WARNING running from a DIRTY tree"
(cd "$STAGE" && git checkout -- scripts/build_device_bench.sh)

echo
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
