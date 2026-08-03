# Portable Edge AI Setup — Reproduction Guide

This document is the primary reproduction path for the **Edge AI** hedge track
(bead `ob-uyh`, ADR 0005). If the Orion O6 board is unavailable, judges can
follow these steps to reproduce the GDN kernel benchmark on any aarch64 device.

---

## 1. Prerequisites

| Requirement | Details |
|---|---|
| **Device** | Any 64-bit Arm (aarch64) board or SBC — Pi 5, Jetson Nano, RK3588, Graviton, etc. |
| **OS** | Linux (any distro with a kernel exposing `/proc/cpuinfo`) |
| **Toolchain** | `aarch64-linux-gnu-gcc` (or native `gcc` on-device) for building; OR use pre-built static binaries |
| **Python** | 3.6+ (only for manifest capture; the bench binary itself needs no Python) |
| **Sudo** | Required only for setting the CPU governor to `performance` |

No GPU, NPU, or proprietary SDK is required. This is the whole point of the
hedge track: it runs on commodity silicon.

---

## 2. Build the static benchmark binaries

```bash
git clone https://github.com/alexcasper/OrionsBelt.git
cd OrionsBelt
./scripts/build_device_bench.sh
```

This produces **static binaries** in `dist/` with zero runtime dependencies:

| Binary | Target ISA | Example device |
|---|---|---|
| `bench_gdn_armv8a` | Armv8-A baseline (NEON only) | Jetson Nano (A57), oldest phones |
| `bench_gdn_armv8.2dot` | + dot-product instructions | Pi 5 (A76), RK3588, most phones since 2018 |
| `bench_gdn_armv8.6i8mm` | + int8 matrix multiply | Graviton 3, newer SoCs |
| `bench_gdn_armv9sve2` | Armv9-A + SVE2 | Cortex-A710/A720, Orion O6 |
| `bench_gdn_pi5_a76` | Core-tuned for Cortex-A76 | Raspberry Pi 5 |
| `bench_gdn_jetson_a57` | Core-tuned for Cortex-A57 | Jetson Nano |

**Pick the most specific binary your device supports.** Use the ISA detector to
confirm:

```bash
PYTHONPATH=src python3 -m orionsbelt.engines.cpu.isa_detect
```

Or check manually:

```bash
grep -m1 Features /proc/cpuinfo
# asimd    → NEON  (use armv8a)
# asimddp  → dotprod (use armv8.2dot)
# i8mm     → int8 matmul (use armv8.6i8mm)
# sve/sve2 → SVE (use armv9sve2)
```

---

## 3. Set the CPU governor

Valid benchmark numbers require a fixed clock frequency:

```bash
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance | sudo tee "$c"
done
```

Verify:

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# → performance
```

Read thermals before and after:

```bash
cat /sys/class/thermal/thermal_zone*/temp
```

---

## 4. Run the benchmark

### Human-readable (eyeball first)

```bash
dist/bench_gdn_<variant> --repeats 30
```

Example output on a Pi 5 (Cortex-A76):

```
GDN CPU kernel microbenchmark
  dispatch path compiled in : neon
  warmups (discarded)       : 3
  timed repeats             : 30

Qwen3.5-4B  (seq=64, channels=4096, 24 GDN layers)
  gdn_cumdecay           p50   600.66 us   p95   616.55 us   spread   2.6%     3.25 GiB/s
  gdn_gated_scan         p50  1618.42 us   p95  1643.05 us   spread   1.5%     1.83 GiB/s
  gdn_causal_dwconv1d    p50   882.00 us   p95   910.74 us   spread   3.3%     2.34 GiB/s
```

### Machine-readable (CSV)

```bash
mkdir -p results/raw
dist/bench_gdn_<variant> --repeats 30 --csv > results/raw/<device-id>.csv
```

### Pin to a cluster (big.LITTLE SoCs only)

On asymmetric SoCs like RK3588, pin to a homogeneous cluster:

```bash
# Confirm which cores are big vs little
for c in /sys/devices/system/cpu/cpu[0-7]; do
  echo "$c $(cat $c/cpufreq/cpuinfo_max_freq 2>/dev/null)"
done

taskset -c 4-7 dist/bench_gdn_rk3588_a76 --repeats 30 --csv > results/raw/rk3588_big.csv
taskset -c 0-3 dist/bench_gdn_rk3588_a55 --repeats 30 --csv > results/raw/rk3588_little.csv
```

---

## 5. Capture provenance

A number without a manifest is not a result (PLAN.md §9).

```bash
mkdir -p results/manifests

# If Python 3.6+ is available:
python3 bench/manifest.py > results/manifests/<device-id>.json

# If Python is missing or too old:
bash scripts/capture_manifest.sh > results/manifests/<device-id>.json
```

The manifest captures: git SHA, hostname, kernel, CPU model, ISA features,
core topology, governor, thermal zones, and memory.

---

## 6. Commit and share

```bash
git add results/
git commit -m "device: benchmark results for <device-id>"
git push
```

---

## 7. Interpreting the results

Each row reports the p50 (median) and p95 latency for one GDN kernel variant
across 30 timed repetitions (after 3 discarded warmups). The columns:

| Column | Meaning |
|---|---|
| `gib_per_s_p50` | Achieved memory bandwidth in GiB/s — compare against device spec |
| `gflop_per_s_p50` | Achieved compute throughput |
| `spread_pct` | (p95 - p50) / p50 × 100 — should be <10% for a clean run |
| `dispatch_path` | `neon`, `sve`, or `scalar` — the compiled code path |

### Expected ranges by device class

| Device | Cores | Scan kernel GiB/s | Notes |
|---|---|---|---|
| Jetson Nano (A57) | 4× A57 @ 1.48 GHz | ~0.7 | Oldest cores; memory-bound |
| Raspberry Pi 5 (A76) | 4× A76 @ 2.4 GHz | ~1.8 | 2.5× faster scan than A57 |
| RK3588 big (A76) | 4× A76 @ 2.4 GHz | ~1.8 | Same class as Pi 5 |

The GDN gated-scan kernel is **memory-bandwidth-bound** (per PLAN.md §2.4).
Devices with more spec bandwidth achieve higher GiB/s, confirming the thesis.

---

## 8. Device fleet

The current fleet (ADR 0005) covers the Arm ISA spectrum:

| Device | ISA level | Dot product | i8mm | SVE |
|---|---|---|---|---|
| Jetson Nano (A57) | Armv8.0 | ✗ | ✗ | ✗ |
| Raspberry Pi 5 (A76) | Armv8.2 | ✓ | ✗ | ✗ |
| RK3588 (A55 + A76) | Armv8.2 | ✓ | ✗ | ✗ |
| Orion O6 (A720) | Armv9 | ✓ | ✓ | ✓ (SVE2) |

The Jetson is the discriminating case: oldest cores, highest spec memory
bandwidth, and the only device where the scan kernel is strictly NEON-bound.
