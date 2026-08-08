# Device runbook — getting the first real numbers

Copy-paste procedure for running the GDN kernel benchmark on the Arm fleet (ADR 0005) and
producing the bandwidth-scaling result (bead `ob-8ms.3`).

Everything here is a **static binary**. The devices need no toolchain, no Python, and no shared
libraries — just `scp` and run.

---

## Start here: drive the whole fleet from one orchestrator

`scripts/fleet.sh` does the per-device steps below across every node in one matched
configuration. Prefer it over the manual procedure — the manual route is what produced
the mixed-commit, mixed-thread-count results that made no cross-device comparison
trustworthy (`ob-bf7`, `ob-mrd.12`/`ob-mrd.14`, `ob-dpl`).

It does not replace [`scripts/fleet_sweep.sh`](../scripts/fleet_sweep.sh). That script is
the **on-device** protocol and still is: run it when you are sitting on a node. `fleet.sh`
runs **from a dev box** and enforces the same matched configuration *between* nodes —
one local build, one SHA, one thread count, shipped everywhere.

```bash
scripts/fleet.sh status                 # which nodes answer
scripts/fleet.sh inventory              # capability sweep -> results/fleet/
scripts/fleet.sh run --threads 1        # the cross-device baseline
scripts/fleet.sh run --threads 4        # multi-core, as a SEPARATE set
scripts/fleet.sh run --threads 1 --only j2 --dry-run
```

Run it from a dev box, **not from a fleet node**. Nodes are listed in
`fleet-nodes.conf`, which holds ssh *aliases* only — hostnames, users and keys stay in
your `~/.ssh/config`, so no credential is ever committed (`ob-3i5`).

Three things it enforces, each because they went wrong once:

- **`--threads` is mandatory and lands in the filename**, not just the manifest
  (`jetson-j2_st.csv` vs `jetson-j2_omp4.csv`). A 1-thread and a 4-thread run of the
  same commit differ 3–4×, and `results/raw/jetson-j1_clean.csv` was captured as a
  clean-tree baseline while actually being a 4-core run. Read as the single-threaded
  row it would put the A57 above the Pi 5 and invert this project's central result.
- **It refuses to run from a tree with modified tracked sources**, because then the
  manifest's git SHA does not identify the binary that produced the numbers — which is
  true of every manifest captured before 2026-08-05. `--allow-dirty` overrides and warns.
- **It flags any row with `spread_pct` above ~10%** so a noisy row cannot quietly reach
  a table. The O6 extrapolation was anchored on a row with 153% spread.

Cluster pinning is resolved **on the node** from `cpufreq/cpuinfo_max_freq`, never
assumed — see §6, the mapping is board-dependent.

Everything below remains valid as the manual fallback, and as the explanation of what
`fleet.sh` is doing.

---

## 0. Build on the x86 host

```bash
./scripts/build_device_bench.sh          # writes to dist/
```

| Device | Binary | Notes |
|---|---|---|
| Raspberry Pi 5 | `dist/bench_gdn_pi5_a76` | 4× A76, homogeneous |
| RK3588 big cluster | `dist/bench_gdn_rk3588_a76` | pin to the A76s |
| RK3588 little cluster | `dist/bench_gdn_rk3588_a55` | pin to the A55s |
| Jetson Nano | `dist/bench_gdn_jetson_a57` | Armv8.0, no dotprod |
| Orion O6 (A720 big) | `dist/bench_gdn_orion_a720` | CIX P1, Armv9.2-A, SVE2 — pin to the 4× A720 big cores |
| Orion O6 (A520 little) | `dist/bench_gdn_armv8a` | A520 little cluster (no SVE2 — use armv8a binary) |

## 1. Per device

```bash
scp dist/bench_gdn_<variant> <device>:/tmp/
ssh <device>
```

Then on the device:

```bash
# what does this thing actually have?
uname -m ; grep -m1 Features /proc/cpuinfo ; nproc
free -m | head -2

# human-readable first, to eyeball it
/tmp/bench_gdn_<variant> --repeats 30

# then the machine-readable form
/tmp/bench_gdn_<variant> --repeats 30 --csv > bench_<device>.csv
```

**Pin to a cluster on the RK3588** — this is the whole point of having an asymmetric board, and an
unpinned run on a big.LITTLE SoC is close to meaningless because the scheduler will migrate you
mid-measurement:

```bash
# On most RK3588 boards cpu0-3 are the A55 littles and cpu4-7 the A76 bigs.
# CONFIRM before trusting it — the mapping is board-dependent:
for c in /sys/devices/system/cpu/cpu[0-7]; do
  echo "$c $(cat $c/cpufreq/cpuinfo_max_freq 2>/dev/null)"
done
# higher max freq = the big cluster

taskset -c 4-7 /tmp/bench_gdn_rk3588_a76 --repeats 30 --csv > bench_rk3588_big.csv
taskset -c 0-3 /tmp/bench_gdn_rk3588_a55 --repeats 30 --csv > bench_rk3588_little.csv
```

**Pin on the Orion O6** — it has a **3-cluster** design (unlike the RK3588's 2):

```bash
# Identify clusters by max frequency (4× A720 big + 4× A720 med + 4× A520 little):
for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/cpuinfo_max_freq; do
  echo "$c $(cat "$c")"
done
# Expect three distinct frequencies. Highest = A720 big, mid = A720 med, lowest = A520.

# Pin to the A720 big cluster (highest-frequency cores) for SVE2 kernels:
taskset -c <big-cores> /tmp/bench_gdn_orion_a720 --repeats 30 --csv > bench_orion_o6_big.csv
# Pin to the A520 little cluster for the Armv8-A fallback path:
taskset -c <little-cores> /tmp/bench_gdn_armv8a --repeats 30 --csv > bench_orion_o6_little.csv
```

## 2. Capture provenance

**For the Orion O6** (or any new Arm device), run the automated system baseline first
to detect cluster topology, set the governor, and capture thermal/memory state in one
command — it auto-detects clusters by `cpuinfo_max_freq` and prints ready-to-paste
`taskset` commands:

```bash
sudo bash scripts/o6_system_baseline.sh             # human-readable
sudo bash scripts/o6_system_baseline.sh --json      # machine-readable
```

Tested on RK3588 (2-cluster), works unchanged on the O6 (3-cluster).

A number without a manifest is not a result (`docs/archive/PLAN.md` §9). If the device has Python 3.10+:

```bash
scp bench/manifest.py <device>:/tmp/
ssh <device> 'python3 /tmp/manifest.py' > results/manifests/<device>.json
```

It is stdlib-only and degrades gracefully, so it will not fail on a minimal image.
If the device has Python < 3.10 (e.g. Jetson Nano's 3.6.9), use the shell-based
alternative instead — same fields, no Python dependency:

```bash
scp scripts/capture_manifest.sh <device>:/tmp/
ssh <device> 'bash /tmp/capture_manifest.sh' > results/manifests/<device>.json
```

If there is no Python at all and no bash, capture the equivalent by hand:

```bash
uname -a ; cat /proc/cpuinfo | head -30 ; free -m
for z in /sys/class/thermal/thermal_zone*; do echo "$z $(cat $z/type) $(cat $z/temp)"; done
for c in /sys/devices/system/cpu/cpu*/cpufreq; do
  echo "$c $(cat $c/scaling_governor) $(cat $c/cpuinfo_max_freq)"
done
```

## 3. Things that will quietly ruin the numbers

- **Thermal throttling.** Check temps before and after. On a Pi 5 without a cooler this is the most
  likely source of a bad result. If p95 is far above p50, suspect throttling first.
- **Governor.** `powersave` will halve your figures. Set `performance` if you can, and record which
  you used either way.
- **Background load.** Idle the device. `nproc`-many busy processes will show up as spread.
- **Not pinning on big.LITTLE.** See above.
- **Reading QEMU numbers as real.** Never benchmark under emulation — see `FINDINGS.md` §5.

## 4. Bring the results back

```bash
mkdir -p results/raw results/manifests
scp <device>:bench_<device>.csv results/raw/
git add results/ && git commit -m "Add measured results from <device>"
```

---

## What we are actually testing

Not "how fast is my board". The devices span **~15.8 GiB/s (Pi 5) → 23.8 GiB/s (Jetson) → ~31.7 GiB/s
(RK3588) → ~93.1 GiB/s (Orion O6)** of spec memory bandwidth, and the hypothesis (`METRICS.md`) is that these kernels are
**memory-bandwidth-bound at ~0.25 FLOP/byte**. If that holds, achieved throughput should track
bandwidth roughly linearly and **largely ignore core generation**.

The discriminating case is that the ordering is inverted:

> The **Pi 5 has the newest cores** (A76, Armv8.2, dotprod) but the **lowest bandwidth**.
> The **Jetson Nano has the oldest cores** (A57, Armv8.0, no dotprod) and **more bandwidth**.

**If the Jetson beats the Pi 5 on the scan kernel** despite three ISA generations less capability,
that is real evidence for bandwidth-boundedness rather than a confirmation-biased result.
**If the Pi 5 wins comfortably**, the thesis is wrong or incomplete, and we need to know that —
several downstream decisions (the CPU-first mapping in `docs/archive/PLAN.md` §3.1, deprioritising the bf16
state variant, prioritising weight quantization) rest on it.

Either outcome is publishable. Report whichever happens.

The RK3588's ~31.7 GiB/s figure is **estimated** from its quad-channel 16-bit interface, not vendor-
confirmed — the benchmark's achieved-GiB/s column is there so the spec number can be checked rather
than trusted.
