# Portable aarch64 hedge-target setup (Edge AI track)

**Bead:** `ob-8ms.4` · **Device verified on:** Raspberry Pi 5 Model B Rev 1.0

This guide covers setting up and running the OrionsBelt benchmark suite on a
generic aarch64 edge device. The Raspberry Pi 5 is the reference implementation —
every command below was verified on `pi5-r5` during the Phase 1 device benchmark.

For the full device fleet (Pi 5, RK3588, Jetson Nano), see
[`docs/DEVICE_RUNBOOK.md`](./DEVICE_RUNBOOK.md) and [ADR 0005](./adr/0005-device-fleet-and-bandwidth-study.md).

---

## 1. Prerequisites

### Hardware

Any 64-bit Arm (aarch64) device. The project's fleet spans:

| Device | SoC | Cores | ISA | Spec BW | Binary |
|---|---|---|---|---|---|
| Raspberry Pi 5 | BCM2712 | 4× Cortex-A76 | Armv8.2-A + dotprod | ~15.8 GiB/s | `bench_gdn_pi5_a76` |
| RK3588 (big) | RK3588 | 4× Cortex-A76 | Armv8.2-A + dotprod | ~31.7 GiB/s | `bench_gdn_rk3588_a76` |
| RK3588 (little) | RK3588 | 4× Cortex-A55 | Armv8.2-A + dotprod | — | `bench_gdn_rk3588_a55` |
| Jetson Nano | Tegra X1 | 4× Cortex-A57 | Armv8.0-A (no dotprod) | ~23.8 GiB/s | `bench_gdn_jetson_a57` |

### OS and toolchain

The device benchmark needs only a C compiler and Python 3:

```bash
# Verify the device
uname -m                    # expect: aarch64
grep -m1 Features /proc/cpuinfo   # check for asimd (NEON), asimddp (dotprod)
nproc                       # core count
free -m | head -2           # memory

# Install toolchain (Raspberry Pi OS / Debian):
sudo apt-get install -y build-essential python3 python3-pip

# Verify gcc can cross-compile for aarch64 (it's native on these devices):
gcc --version               # expect: aarch64-linux-gnu-gcc
```

No GPU drivers, NPU SDK, or ML frameworks are required for the device benchmark.
The full benchmark harness (with model inference) needs `transformers` + `torch`
— see [`docs/BACKEND_GUIDE.md`](./BACKEND_GUIDE.md) for that path.

---

## 2. Build the device benchmark binaries

```bash
cd OrionsBelt
./scripts/build_device_bench.sh
```

This produces static binaries in `dist/` — one per ISA variant. **Run only the
binary matching your device's core** (others may illegal-instruction):

```bash
# Check which binary to use:
grep -m1 Features /proc/cpuinfo
#   asimd    → NEON only         → bench_gdn_armv8a
#   asimddp  → dotprod           → bench_gdn_pi5_a76 (or armv8.2dot)
#   i8mm     → int8 matmul       → bench_gdn_armv8.6i8mm
#   sve/sve2 → SVE              → bench_gdn_armv9sve2 (rare on Armv8)

# For the Raspberry Pi 5:
ls -la dist/bench_gdn_pi5_a76
```

---

## 3. Set the CPU governor to performance

Valid benchmark numbers require the CPU governor set to `performance` — otherwise
DVFS downclocking will silently halve your figures (see
[`docs/METHODOLOGY.md`](./METHODOLOGY.md) §3).

```bash
# Set performance governor on all cores (needs sudo):
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance | sudo tee "$c" >/dev/null
done

# Verify:
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "$(basename $(dirname $(dirname $c))): $(cat $c)"
done
# All cores should show: performance

# Read thermals before the run:
cat /sys/class/thermal/thermal_zone*/temp
```

---

## 4. Run the benchmark

### Human-readable (eyeball it first)

```bash
./dist/bench_gdn_pi5_a76 --repeats 30
```

Check: dispatch path (should be `neon` on Pi5, not `sve`), tight spreads (p95
close to p50), no signs of throttling.

### CSV capture

```bash
mkdir -p results/raw results/manifests
./dist/bench_gdn_pi5_a76 --repeats 30 --csv > results/raw/pi5-r5.csv
```

### Capture provenance

```bash
python3 bench/manifest.py > results/manifests/pi5-r5.json
```

A number without a manifest is not a result (docs/archive/PLAN.md §9).

### Read thermals after the run

```bash
cat /sys/class/thermal/thermal_zone*/temp
# Compare to the pre-run reading. A rise > 10% of the starting temp suggests
# thermal throttling may have affected the numbers.
```

---

## 5. Run the Python harness (optional, for full context sweep)

The Python harness sweeps multiple context lengths with warmup and percentiles.
It uses the `SyntheticBackend` by default (no model weights needed):

```bash
# Create a venv (the Pi5's Python is externally managed):
python3 -m venv /tmp/ob-venv
/tmp/ob-venv/bin/pip install pytest ruff matplotlib numpy

# Run a quick smoke test:
python3 bench/harness.py --backend synthetic --model 0.8b \
    --context-lengths 64,128 --warmup 1 --repeats 5 --decode-length 10 --no-csv

# Run the test suite:
/tmp/ob-venv/bin/python -m pytest tests/test_harness.py tests/test_memory.py tests/test_prompts.py -v

# Generate the memory decomposition plot:
/tmp/ob-venv/bin/python bench/plots.py --memory
```

---

## 6. Pin to a cluster (RK3588 only)

On big.LITTLE SoCs, pin to the correct cluster — the scheduler will migrate you
mid-measurement otherwise:

```bash
# Identify the big cluster (higher max freq = big):
for c in /sys/devices/system/cpu/cpu[0-7]; do
    echo "$c $(cat $c/cpufreq/cpuinfo_max_freq 2>/dev/null)"
done

# Pin to big cores (typically cpu4-7 on RK3588):
taskset -c 4-7 ./dist/bench_gdn_rk3588_a76 --repeats 30 --csv > results/raw/rk3588_big.csv

# Pin to little cores (typically cpu0-3):
taskset -c 0-3 ./dist/bench_gdn_rk3588_a55 --repeats 30 --csv > results/raw/rk3588_little.csv
```

The Pi 5 has homogeneous cores (4× A76), so pinning is unnecessary.

---

## 7. Commit and share results

```bash
git add results/
git commit -m "<device>: device-fleet bench (ob-8ms.3)"
git push origin bench/r5
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Illegal instruction` | Wrong binary for the core | Check `/proc/cpuinfo` Features; use the matching variant |
| p95 far above p50 | Thermal throttling | Check thermals; add a cooldown between context lengths |
| Numbers 2× lower than expected | Governor is `ondemand` or `powersave` | Set to `performance` (§3) |
| `python3: No module named pytest` | Externally managed Python | Use a venv (§5) |
| `bd dolt push: divergent histories` | Multiple agents initialized DBs independently | Export to `issues.jsonl` and push via git; do not force-push |

---

## What the device benchmark tells us

The devices span ~15.8 GiB/s (Pi 5) → ~23.8 GiB/s (Jetson) → ~31.7 GiB/s (RK3588) of
spec memory bandwidth. The initial hypothesis was that GDN kernels are
memory-bandwidth-bound at ~0.25 FLOP/byte — if so, achieved throughput should
track bandwidth roughly linearly and **independently of core generation**.

The discriminating case: the Pi 5 has the **newest cores** (A76) but the
**lowest bandwidth**, while the Jetson Nano has the **oldest cores** (A57) and
**more bandwidth**. If the Nano beats the Pi 5 on the scan kernel, that is strong
evidence for bandwidth-boundedness.

**The result:** the Pi 5 robustly wins on CumDecay and DWConv1D (2.2–2.5× margins)
despite having 33% LESS spec bandwidth. The Scan kernel shows only a marginal edge
(1.05×, within the Jetson inter-board replicate spread — not statistically
reliable). On the two kernels where the result is unambiguous, the bandwidth-bound
hypothesis does **not** hold at seq=64 working-set sizes — those kernels are
**instruction-overhead-bound, not DRAM-bandwidth-bound**. See
[`docs/DEVICE_RUNBOOK.md`](./DEVICE_RUNBOOK.md) §"What we are actually testing"
and [`FINDINGS.md`](./FINDINGS.md) §5b for the full analysis.

### Reading the CSV columns

| Column | Meaning |
|---|---|
| `gib_per_s_p50` | Achieved memory bandwidth in GiB/s — compare against the device spec |
| `gflop_per_s_p50` | Achieved compute throughput |
| `spread_pct` | `(p95 - p50) / p50 × 100` — should be under ~10% for a clean run |
| `dispatch_path` | `neon`, `sve` or `scalar` — which compiled path actually ran |

### What the fleet measured (fleet sweep ob-bf7, post-optimization)

Sanity-check your own numbers against these, all `gdn_gated_scan`, 4B, seq=64,
NEON path (see [`FINDINGS.md`](./FINDINGS.md) §5b for the full analysis):

| Device | Spec BW | Achieved | % of spec |
|---|---:|---:|---:|
| Jetson Nano j1 (A57) | 23.8 GiB/s | 1.14 GiB/s | **4.8%** |
| Raspberry Pi 5 (A76) | 15.8 GiB/s | 1.20 GiB/s | 7.6% |
| RK3588 big (A76) | 31.7 GiB/s | 5.67 GiB/s | 17.9% |

The Pi 5 wins on CumDecay and DWConv1D despite having 33% less spec bandwidth
than the Jetson and ~50% less than the RK3588. Its Scan margin (1.05×) is within
inter-board replicate noise and not reliable (ob-5kw). This is **not** a broken
setup — the Jetson's low scan number is real. The bottleneck is the core
microarchitecture (IPC, OoO depth, clock frequency), not the memory system. The
A76's ~1.6× higher clock and substantially better IPC than the A57 explains the
win despite less bandwidth.
Multi-threaded scaling (OpenMP) reveals a bandwidth component that the
single-thread comparison cannot expose — see the optimization-impact section in
[`fleet_bandwidth_scaling.md`](../results/figures/fleet_bandwidth_scaling.md).
