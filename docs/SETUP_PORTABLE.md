# Portable aarch64 Setup Guide

How to build and run the OrionsBelt GDN kernel benchmark on any 64-bit Arm Linux
device. Tested on Jetson Nano (Cortex-A57); the same steps work on Raspberry Pi 5,
RK3588, Graviton, Apple silicon (Linux VM), and Android via Termux.

This is the **Edge AI hedge target** — no Orion O6 board, no NPU SDK, no proprietary
toolchain required. Just a C compiler and an Arm core.

---

## 1. Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| CPU | Any AArch64 (Armv8-A+) | NEON is mandatory (part of Armv8-A). SVE/SVE2 optional. |
| OS | Linux (aarch64) | Tested on Ubuntu 18.04+ and Debian. Termux works too. |
| C compiler | GCC 7+ or Clang 10+ | Must support `-static` and your core's `-march`/`-mcpu` |
| OpenMP | libgomp (usually preinstalled) | Enables multi-core; without it, kernels run single-threaded |
| Python 3 | 3.6+ (optional) | Only for `bench/manifest.py` provenance capture |
| RAM | 64 MB free | Benchmark allocates ~20 MB for Qwen3.5-4B shapes |

Check your compiler and ISA:

```bash
gcc --version                    # or: clang --version
uname -m                         # expect: aarch64
grep -m1 Features /proc/cpuinfo  # look for: asimd (NEON), asimddp (dotprod), sve, i8mm
```

## 2. Clone

```bash
git clone https://github.com/alexcasper/OrionsBelt.git
cd OrionsBelt
```

## 3. Build

```bash
./scripts/build_device_bench.sh
```

This produces static binaries in `dist/` for several ISA levels. Only the builds your
toolchain supports will succeed — the rest are silently skipped. Each build produces:

- `dist/bench_gdn_<variant>` — the benchmark binary
- `dist/test_gdn_sve_<variant>` — the correctness test binary

**Pick the build matching your core:**

| Device | Build variant | Flag |
|---|---|---|
| Jetson Nano (Cortex-A57) | `jetson_a57` | `-mcpu=cortex-a57` |
| Raspberry Pi 5 (Cortex-A76) | `pi5_a76` | `-mcpu=cortex-a76` |
| RK3588 big cores (A76) | `rk3588_a76` | `-mcpu=cortex-a76` |
| RK3588 little cores (A55) | `rk3588_a55` | `-mcpu=cortex-a55` |
| Generic Armv8-A (any) | `armv8a` | `-march=armv8-a` |
| Armv8.2+ with dotprod | `armv8.2dot` | `-march=armv8.2-a+dotprod` |

If you're unsure, use `armv8a` — it runs on any 64-bit Arm device.

## 4. Verify correctness

```bash
./dist/test_gdn_sve_<variant>
```

Expected output (key lines):

```
gated_scan bit-identical to matched reference: YES
causal_dwconv1d              max_abs=5.960e-08  (one fp32 ULP — normal)
```

The test checks all 7 kernel variants (fp32, fp16-state, bf16-state) against independently
written scalar references, plus an 8-chunk drift test for mixed precision.

## 5. Set CPU governor (for valid numbers)

For reproducible benchmark numbers, pin all cores to maximum frequency:

```bash
# Requires root (sudo password: your device's password)
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance | sudo tee "$c" >/dev/null
done

# Verify
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# Expected: performance (one per core)
```

Record the governor in your notes — it goes in the results manifest.

**Read thermals** before and after benchmarking:

```bash
cat /sys/class/thermal/thermal_zone*/temp
# Values in millidegrees Celsius (e.g., 47500 = 47.5 °C)
```

If thermals exceed 80 °C, the device will throttle and numbers will be unreliable.

## 6. Run the benchmark

First, a quick human-readable eyeball:

```bash
./dist/bench_gdn_<variant> --repeats 30
```

Then capture to CSV:

```bash
mkdir -p results/raw
./dist/bench_gdn_<variant> --repeats 30 --csv > results/raw/<device-id>.csv
```

The CSV schema is defined in [`docs/RESULTS_SCHEMA.md`](./RESULTS_SCHEMA.md). Each row
reports p50/p95 latency, spread %, achieved GiB/s, and GFLOP/s for one kernel variant
at one model shape.

### big.LITTLE affinity (RK3588 only)

On asymmetric CPUs, pin to a specific cluster:

```bash
# RK3588 big cores (Cortex-A76, cores 4-7):
taskset -c 4-7 ./dist/bench_gdn_rk3588_a76 --repeats 30 --csv > results/raw/rk3588-a76.csv

# RK3588 little cores (Cortex-A55, cores 0-3):
taskset -c 0-3 ./dist/bench_gdn_rk3588_a55 --repeats 30 --csv > results/raw/rk3588-a55.csv
```

## 7. Capture provenance

```bash
mkdir -p results/manifests
python3 bench/manifest.py > results/manifests/<device-id>.json
```

This captures the compiler version, ISA flags, kernel git hash, CPU model, core count,
frequency, governor, and thermal readings. **A number without a manifest is not a result**
(PLAN.md §9).

If Python 3 is not available, record these manually:

```bash
gcc --version | head -1
grep -m1 Features /proc/cpuinfo
cat /proc/cpuinfo | grep -c ^processor
git rev-parse HEAD
```

## 8. Submit results

```bash
git add results/raw/<device-id>.csv results/manifests/<device-id>.json
git commit -m "<device-id>: device-fleet bench (ob-8ms.3)"
git push
```

## Device-specific notes

### Jetson Nano (Cortex-A57)
- 4 cores @ 1479 MHz, 1 MB shared L2, LPDDR4 @ 1600 MHz (12.8 GiB/s peak)
- Governor can be set without issues; thermals stay 43–51 °C under load
- Only `armv8a`, `armv8.2dot`, and `jetson_a57` builds succeed (GCC 7.5)
- Use `jetson_a57` for the most specific optimization

### Raspberry Pi 5 (Cortex-A76)
- 4 cores @ 2.4 GHz, 512 KB L2 per core + 2 MB shared L3
- Supports dotprod; use `pi5_a76` build
- May need `cpufreq` kernel module for governor control

### RK3588 (Cortex-A76 + A55)
- 4× A76 @ 2.4 GHz + 4× A55 @ 1.8 GHz; benchmark each cluster separately
- Use `taskset` to pin (see above)
- NPU and GPU are out of scope for this benchmark

### AWS Graviton (Neoverse-V1/V2)
- Has SVE; use `armv9sve2` build for SVE path, or `armv8a` for NEON
- Governor is managed by the hypervisor; record what you get
