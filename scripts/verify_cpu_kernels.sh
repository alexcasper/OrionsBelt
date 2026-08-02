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
    "$K/gdn_sve.c" "$K/test_gdn_sve.c" -o "$OUT/verify" -lm

echo "== running under QEMU with 128-bit vectors (as Cortex-A720)"
QEMU_CPU=max,sve128=on qemu-aarch64 "$OUT/verify"

echo "== mixed-precision f16/bf16 state kernels (ob-8qt.4)"
# The fp32 accumulator stays wide; only persistent state is narrowed.
# Verify under the full SVE2 path (Armv9.2) and the NEON fp16 path.
aarch64-linux-gnu-gcc -O3 $MARCH -static \
    "$K/gdn_sve.c" "$K/gdn_sve_f16.c" "$K/test_gdn_sve_f16.c" -o "$OUT/verify_f16" -lm
echo "== running f16/bf16 tests under QEMU (128-bit SVE2)"
QEMU_CPU=max,sve128=on qemu-aarch64 "$OUT/verify_f16" | tail -1

echo "== portability matrix: SVE1 floor, SVE2, and no-SVE fallback"
# The kernels are SVE1-clean; SVE2 is NOT required. Verified across vector lengths and cores.
for spec in \
  "SVE1@128|-march=armv8.2-a+sve|max,sve128=on" \
  "SVE1@256(Graviton3)|-march=armv8.2-a+sve|max,sve256=on" \
  "SVE1@512(A64FX)|-march=armv8.2-a+sve|max,sve512=on" \
  "Neoverse-V1|-mcpu=neoverse-v1|max,sve256=on" \
  "SVE2/Armv9-A|-march=armv9-a|max,sve128=on" \
  "Neoverse-V2(Graviton4)|-mcpu=neoverse-v2|max,sve128=on" \
  "no-SVE scalar|-march=armv8-a|max" ; do
  IFS='|' read -r label march cpu <<< "$spec"
  aarch64-linux-gnu-gcc -O3 $march -static \
      "$K/gdn_sve.c" "$K/test_gdn_sve.c" -o "$OUT/v" -lm
  printf "  %-24s " "$label"
  QEMU_CPU=$cpu qemu-aarch64 "$OUT/v" | grep -c "bit-identical to matched reference: YES" \
      | sed 's/^1$/PASS/; s/^0$/FAIL/'
done
