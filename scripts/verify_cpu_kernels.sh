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
    "$K/gdn_sve2.c" "$K/test_gdn_sve2.c" -o "$OUT/verify" -lm

echo "== running under QEMU with 128-bit vectors (as Cortex-A720)"
QEMU_CPU=max,sve128=on qemu-aarch64 "$OUT/verify"

echo "== running with 256-bit vectors (vector-length-agnostic check)"
QEMU_CPU=max,sve256=on qemu-aarch64 "$OUT/verify"

echo "== scalar fallback path (no SVE), same reference"
aarch64-linux-gnu-gcc -O3 -march=armv8-a -static \
    "$K/gdn_sve2.c" "$K/test_gdn_sve2.c" -o "$OUT/verify_scalar" -lm
qemu-aarch64 "$OUT/verify_scalar"
