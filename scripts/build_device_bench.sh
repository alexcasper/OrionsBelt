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
    if $CC -O3 $flags -static "$K/gdn_sve.c" "$K/bench_gdn.c" -o "$OUT/bench_gdn_$name" -lm \
        2>/dev/null; then
        printf "  %-14s %-40s %s bytes\n" "$name" "$flags" \
            "$(stat -c%s "$OUT/bench_gdn_$name")"
    else
        printf "  %-14s %-40s SKIPPED (toolchain rejects these flags)\n" "$name" "$flags"
    fi
}

echo "Building static aarch64 benchmarks with $CC:"
build armv8a      "-march=armv8-a"
build armv8.2dot  "-march=armv8.2-a+dotprod"
build armv8.6i8mm "-march=armv8.2-a+i8mm"
build armv9sve2   "-march=armv9-a+sve2+i8mm+bf16"

cat <<'NOTE'

Pick the most specific binary your device supports, then:

    ./bench_gdn_<variant> --repeats 30            # human-readable
    ./bench_gdn_<variant> --repeats 30 --csv      # schema-conforming CSV

To find out what a device supports:

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
