# Setup: portable aarch64 (Edge AI hedge target)

Tested on:
- **Radxa ROCK 5B / Orange Pi 5 (RK3588)** — Cortex-A76 big + Cortex-A55 little, Ubuntu 24.04
- **Raspberry Pi 5** — Cortex-A76, Raspberry Pi OS / Ubuntu
- **NVIDIA Jetson Nano** — Cortex-A57, JetPack (Python 3.6)

## Prerequisites

- Native `gcc` (cross-compilation also works with `aarch64-linux-gnu-gcc`)
- `taskset` (from `util-linux`, present by default on most distros)
- Python 3.6+ (for manifest capture; the benchmark itself has no Python dependency)
- `git`

## Step-by-step

### 1. Clone

```bash
git clone https://github.com/alexcasper/OrionsBelt.git
cd OrionsBelt
```

### 2. Build the benchmark binaries

```bash
CC=gcc ./scripts/build_device_bench.sh
```

This produces **static** binaries in `dist/`, one per ISA variant. No runtime library
dependencies — copy one to any aarch64 device and run it.

Pick the binary matching your core (run `grep -m1 Features /proc/cpuinfo` to check):

| Binary | Target | How to identify |
|---|---|---|
| `bench_gdn_armv8a` | Any Armv8-A (NEON only) | `asimd` in cpuinfo |
| `bench_gdn_armv8.2dot` | Armv8.2-A + dotprod | `asimddp` in cpuinfo |
| `bench_gdn_armv8.6i8mm` | Armv8.6-A + i8mm | `i8mm` in cpuinfo |
| `bench_gdn_armv9sve2` | Armv9-A + SVE2 | `sve` + `sve2` in cpuinfo |
| `bench_gdn_rk3588_a76` | RK3588 big cores (A76) | device-specific tune |
| `bench_gdn_rk3588_a55` | RK3588 little cores (A55) | device-specific tune |
| `bench_gdn_pi5_a76` | Raspberry Pi 5 (A76) | device-specific tune |
| `bench_gdn_jetson_a57` | Jetson Nano (A57) | device-specific tune |

If unsure, use `bench_gdn_armv8a` — it runs on every Armv8+ device.

### 3. Set the CPU governor (for valid benchmark numbers)

```bash
# Requires sudo — set all cores to performance mode
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance | sudo tee "$c" >/dev/null
done

# Record thermals before the run
cat /sys/class/thermal/thermal_zone*/temp
```

### 4. Run the benchmark

```bash
# Human-readable eyeball (quick check)
taskset -c 4-7 ./dist/bench_gdn_rk3588_a76 --repeats 30

# CSV output (schema-conforming, for results/raw/)
taskset -c 4-7 ./dist/bench_gdn_rk3588_a76 --repeats 30 --csv > my_big.csv

# Little cluster
taskset -c 0-3 ./dist/bench_gdn_rk3588_a55 --repeats 30 --csv > my_little.csv

# Record thermals after the run
cat /sys/class/thermal/thermal_zone*/temp
```

On the RK3588, cores 0–3 are Cortex-A55 (little) and cores 4–7 are Cortex-A76 (big).
Adjust `taskset` ranges for your board's topology.

### 5. Capture provenance

```bash
python3 bench/manifest.py > results/manifests/my_run.json
```

The manifest records git SHA, CPU topology, governor, clock speeds, and thermals.
**A number without a manifest is not a result** (PLAN.md §9).

### 6. Verify kernel correctness (optional but recommended)

```bash
bash scripts/verify_kernels_native.sh
```

Runs the full numerical correctness suite (fp32 reference vs bf16/fp16 state narrowing,
drift accumulation check) on the device's real ISA.

### 7. Download model weights (for the full inference harness)

```bash
python3 scripts/fetch_weights.py          # Qwen3.5-4B (~8 GB)
python3 scripts/fetch_weights.py --model 0.8b  # Qwen3.5-0.8B (~1.6 GB)
```

Weights are not vendored in the repo (see `docs/WEIGHT_LICENSE.md`).

## Troubleshooting

- **Illegal instruction**: you're running a binary compiled for a newer ISA than your CPU.
  Fall back to `bench_gdn_armv8a`.
- **`taskset` not found**: install `util-linux` (`apt install util-linux`).
- **Python < 3.10**: `bench/manifest.py` and `scripts/fetch_weights.py` are compatible
  with Python 3.6+. For shell-only provenance, use `scripts/capture_manifest.sh`.
- **Governor won't change**: some boards need the `cpufreq_userspace` governor module loaded.
