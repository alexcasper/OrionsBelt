#!/usr/bin/env bash
# Build the Armv9.2 GDN kernels and verify them numerically under QEMU.
# No Orion O6 board required — QEMU emulates SVE2, so correctness is checkable today.
# Bead ob-8qt.1. See docs/FINDINGS.md §4.
set -euo pipefail

K=src/orionsbelt/engines/cpu/kernels
OUT=${OUT:-/tmp/gdn_kernel_verify}
mkdir -p "$OUT"

# GCC 13 does not know -mcpu=cortex-a720 (added in GCC 14); the arch-level flag is
# equivalent for our purposes and works on 13.
MARCH=${MARCH:--march=armv9.2-a+sve2+i8mm+bf16}

echo "== cross-compiling for aarch64 ($MARCH)"
aarch64-linux-gnu-gcc -O3 $MARCH -static \
    "$K/gdn_sve.c" "$K/test_gdn_sve.c" -I"$K" -o "$OUT/verify" -lm

echo "== running under QEMU with 128-bit vectors (as Cortex-A720)"
QEMU_CPU=max,sve128=on qemu-aarch64 "$OUT/verify"

echo "== cross-compiling the delta-rule matmul (ob-8qt.1) for aarch64 ($MARCH)"
# KleidiAI is deliberately NOT linked in here (evaluation phase, not a submodule --
# see docs/FINDINGS.md §8's Reproducing steps). This build exercises the hand-NEON/SVE
# fallback path only; the ORIONSBELT_WITH_KLEIDIAI dispatch branch is verified
# separately by hand against a real KleidiAI checkout (see docs/FINDINGS.md §11).
aarch64-linux-gnu-gcc -O3 $MARCH -static \
    "$K/gdn_delta_matmul.c" "$K/test_gdn_delta_matmul.c" -I"$K" -o "$OUT/verify_matmul" -lm

echo "== running delta-rule matmul under QEMU with 128-bit vectors (as Cortex-A720)"
QEMU_CPU=max,sve128=on qemu-aarch64 "$OUT/verify_matmul"

echo "== cross-compiling the GDN-2 scan correctness test for aarch64 ($MARCH)"
aarch64-linux-gnu-gcc -O3 $MARCH -static \
    "$K/gdn_sve.c" "$K/test_gdn2_scan.c" -I"$K" -o "$OUT/verify_gdn2" -lm

echo "== running GDN-2 scan test under QEMU with 128-bit vectors (as Cortex-A720)"
QEMU_CPU=max,sve128=on qemu-aarch64 "$OUT/verify_gdn2"

echo "== portability matrix: SVE1 floor, SVE2, and no-SVE fallback"
# The kernels are SVE1-clean; SVE2 is NOT required. Verified across vector lengths and cores.
for spec in \
  "SVE1@128|-march=armv8.2-a+sve|max,sve128=on" \
  "SVE1@256(Graviton3)|-march=armv8.2-a+sve|max,sve256=on" \
  "SVE1@512(A64FX)|-march=armv8.2-a+sve|max,sve512=on" \
  "Neoverse-V1|-mcpu=neoverse-v1|max,sve256=on" \
  "SVE2/Armv9-A|-march=armv9-a|max,sve128=on" \
  "Neoverse-V2(Graviton4)|-mcpu=neoverse-v2|max,sve128=on" \
  "no-SVE scalar|-march=armv8-a|max" \
  "OpenMP 4 threads|-march=armv8.2-a+sve -fopenmp|max,sve128=on" \
  "OpenMP no-SVE|-march=armv8-a -fopenmp|max" ; do
  IFS='|' read -r label march cpu <<< "$spec"
  # Static linking against libgomp emits a benign warning about dlopen, which
  # libgomp only uses for device-offload plugins we do not build. Filter that one
  # line and nothing else, so real diagnostics still surface.
  #
  # Deleting the binary first is load-bearing: the filter pipeline masks gcc's
  # exit status, so without this a failed compile would silently re-run the
  # previous target's binary and report PASS for a target that never built.
  rm -f "$OUT/v"
  aarch64-linux-gnu-gcc -O3 $march -static \
      "$K/gdn_sve.c" "$K/test_gdn_sve.c" -o "$OUT/v" -lm 2>&1 \
    | grep -vE "Using 'dlopen' in statically linked|gomp_load_plugin_for_device" >&2 || true
  printf "  %-24s " "$label"
  if [ ! -x "$OUT/v" ]; then echo "FAIL (compile failed)"; continue; fi
  # Count-agnostic on purpose: the number of "bit-identical" lines grows as kernels are
  # added (fp32, bf16, fp16...). An exact-count check silently stopped reporting when main
  # gained the narrow-precision kernels, so assert on absence-of-failure instead.
  # OMP_NUM_THREADS is forced above 1 for the OpenMP rows: the channel loops are
  # '#pragma omp parallel for', and the device build script compiles with -fopenmp,
  # so a single-threaded-only gate would never exercise the configuration that
  # actually ships. Each channel's scan is independent, which is what makes the
  # parallelization legal — this asserts that rather than assuming it.
  out=$(OMP_NUM_THREADS=4 QEMU_CPU=$cpu qemu-aarch64 "$OUT/v" 2>&1)
  ok=$(printf '%s' "$out" | grep -c "bit-identical to matched reference: YES" || true)
  bad=$(printf '%s' "$out" | grep -ciE "bit-identical to matched reference: no|EXCEEDS TOLERANCE|CAUSALITY VIOLATED|FAIL" || true)
  if [ "$ok" -ge 1 ] && [ "$bad" -eq 0 ]; then echo "PASS ($ok checks)"; else echo "FAIL (ok=$ok bad=$bad)"; fi
done
