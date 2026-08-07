#!/usr/bin/env bash
# Build the standalone GDN kernel microbenchmark for a real aarch64 device.
#
# Produces STATIC binaries with no runtime dependencies — copy one to the device and run it.
# Bead ob-8ms.2. Protocol: docs/METRICS.md. Schema: docs/RESULTS_SCHEMA.md.
#
#   ./scripts/build_device_bench.sh
#   scp dist/bench_gdn_armv8a  device:/tmp/
#   ssh device '/tmp/bench_gdn_armv8a --repeats 30 --csv' > results/raw/device.csv
#
# WARNING: do NOT run these under QEMU for performance numbers. QEMU emulates
# instruction-by-instruction; its timings are meaningless as measurements (it is fine for
# correctness, which is what scripts/verify_cpu_kernels.sh uses it for).
set -euo pipefail

K=src/orionsbelt/engines/cpu/kernels
OUT=${OUT:-dist}
mkdir -p "$OUT"
CC=${CC:-aarch64-linux-gnu-gcc}

# One build per ISA level, because Arm devices vary enormously and the right binary depends on
# the core. Run the most specific one your device supports.
#
#   armv8a       baseline Armv8-A. NEON only. Runs on essentially any 64-bit Arm device.
#   armv8.2dot   Armv8.2-A + dotprod. Cortex-A55/A75 and later — most phones since ~2018.
#   armv8.6i8mm  Armv8.2-A + i8mm. Needed for int8 matmul work later; not used by fp32 kernels.
#   armv9sve2    Armv9-A + SVE2. Cortex-A710/A720, Orion O6. Exercises the SVE path.
build() {
    local name="$1" flags="$2"
    if $CC -O3 -fopenmp $flags -static "$K/gdn_sve.c" "$K/bench_gdn.c" -o "$OUT/bench_gdn_$name" -lm \
        2>/dev/null; then
        printf "  %-14s %-40s %s bytes\n" "$name" "$flags" \
            "$(stat -c%s "$OUT/bench_gdn_$name")"
        # Also build the correctness test binary with the same flags
        $CC -O3 -fopenmp $flags -static "$K/test_gdn_sve.c" "$K/gdn_sve.c" \
            -o "$OUT/test_gdn_sve_$name" -lm 2>/dev/null || true
    else
        printf "  %-14s %-40s SKIPPED (toolchain rejects these flags)\n" "$name" "$flags"
    fi
}

echo "Building static aarch64 benchmarks with $CC:"

# --- GDN kernel microbenchmarks (scan, decay, conv1d) ---
build armv8a      "-march=armv8-a"
build armv8.2dot  "-march=armv8.2-a+dotprod"
build armv8.6i8mm "-march=armv8.2-a+i8mm"
build armv9sve2   "-march=armv9-a+sve2+i8mm+bf16"

# Core-tuned builds for the actual device fleet (ADR 0005). Verified feature sets:
#   Cortex-A57  Armv8.0-A, NO dotprod, no fp16 vector  -> Jetson Nano
#   Cortex-A76  Armv8.2-A, dotprod, fp16, no i8mm/SVE  -> Raspberry Pi 5, RK3588 big cluster
#   Cortex-A55  Armv8.2-A, dotprod, fp16, no i8mm/SVE  -> RK3588 little cluster
# None of these has SVE, so all take the NEON path -- which is why that path matters.
build jetson_a57  "-mcpu=cortex-a57"
build pi5_a76     "-mcpu=cortex-a76"
build rk3588_a76  "-mcpu=cortex-a76"
build rk3588_a55  "-mcpu=cortex-a55"
# Orion O6 uses Cortex-A720 (Armv9.2-A, SVE2 128-bit, i8mm, dotprod, bf16).
# Requires GCC 14+ or clang 17+. Falls back to armv9sve2 on older toolchains.
build orion_a720  "-mcpu=cortex-a720"

# --- Delta-rule matmul microbenchmark (ob-8qt.1) ---
# Single-threaded — no OpenMP needed. Tests M=1 decode and M=64 prefill shapes.
build_matmul() {
    local name="$1" flags="$2"
    if $CC -O3 $flags -static "$K/gdn_delta_matmul.c" bench/bench_delta_matmul.c -I"$K" \
        -o "$OUT/bench_delta_matmul_$name" -lm 2>/dev/null; then
        printf "  %-14s %-40s %s bytes\n" "matmul_$name" "$flags" \
            "$(stat -c%s "$OUT/bench_delta_matmul_$name")"
    else
        printf "  %-14s %-40s SKIPPED\n" "matmul_$name" "$flags"
    fi
}

echo ""
echo "Delta-rule matmul benchmarks:"
build_matmul armv8a      "-march=armv8-a"
build_matmul armv8.2dot  "-march=armv8.2-a+dotprod"
build_matmul armv8.6i8mm "-march=armv8.2-a+i8mm"
build_matmul armv9sve2   "-march=armv9-a+sve2+i8mm+bf16"
build_matmul jetson_a57  "-mcpu=cortex-a57"
build_matmul pi5_a76     "-mcpu=cortex-a76"
build_matmul rk3588_a76  "-mcpu=cortex-a76"
build_matmul rk3588_a55  "-mcpu=cortex-a55"
build_matmul orion_a720  "-mcpu=cortex-a720"

cat <<'NOTE'

Pick the most specific binary your device supports, then:

    ./bench_gdn_<variant> --repeats 30            # human-readable
    ./bench_gdn_<variant> --repeats 30 --csv      # schema-conforming CSV

Delta-rule matmul benchmark (ob-8qt.1):

    ./bench_delta_matmul_<variant> --repeats 30 --csv > results/raw/<device>_delta_matmul.csv

For the known fleet, use the core-tuned build:

    Raspberry Pi 5    -> bench_gdn_pi5_a76
    RK3588 big cores  -> bench_gdn_rk3588_a76   (pin with taskset -c 4-7 on most RK3588 boards)
    RK3588 little     -> bench_gdn_rk3588_a55   (taskset -c 0-3)
    Jetson Nano       -> bench_gdn_jetson_a57
    Orion O6 (A720)   -> bench_gdn_orion_a720    (or armv9sve2 if GCC < 14)

To find out what an unknown device supports:

    uname -m                      # expect aarch64
    grep -m1 Features /proc/cpuinfo
      asimd    -> NEON  (armv8a build is correct)
      asimddp  -> dotprod
      i8mm     -> int8 matmul
      sve/sve2 -> SVE   (use the armv9sve2 build; rare outside Armv9)

Also capture the device's provenance alongside the numbers:

    python3 bench/manifest.py > results/manifests/<run_id>.json   # if python3 is present

The benchmark prints which dispatch path was compiled in (sve / neon / scalar). On Armv8-A
without SVE, 'neon' is expected and correct — that is the path the large installed base of
Armv8 devices actually runs.
NOTE
