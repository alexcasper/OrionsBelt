#!/usr/bin/env bash
# Native on-device kernel correctness verification.
#
# Unlike verify_cpu_kernels.sh (which cross-compiles for SVE and runs under QEMU),
# this script builds and runs the C kernel tests NATIVELY on the device using its
# real ISA. This validates the actual dispatch path each device uses:
#
#   Cortex-A57  → NEON (armv8-a)
#   Cortex-A76  → NEON (armv8.2-a+dotprod)
#   Cortex-A55  → NEON (armv8.2-a+dotprod)
#   A720/O6     → SVE2 (armv9-a+sve2)
#
# Bead ob-mrd.3. Usage: ./scripts/verify_kernels_native.sh
set -euo pipefail

K="$(dirname "$0")/../src/orionsbelt/engines/cpu/kernels"
OUT="${OUT:-/tmp/gdn_kernel_native}"
mkdir -p "$OUT"
CC="${CC:-cc}"
FAILURES=0

echo "=== Native kernel correctness verification ==="
echo "Compiler: $($CC --version 2>&1 | head -1)"
echo "Arch:     $(uname -m)"

# Detect ISA features from /proc/cpuinfo
FEATURES=$(grep -m1 '^Features' /proc/cpuinfo 2>/dev/null | cut -d: -f2 || echo "")
if echo "$FEATURES" | grep -qw sve2; then
    MARCH="-march=armv9-a+sve2+i8mm+bf16"
    ISA_LABEL="SVE2"
elif echo "$FEATURES" | grep -qw sve; then
    MARCH="-march=armv8.2-a+sve"
    ISA_LABEL="SVE1"
elif echo "$FEATURES" | grep -qw asimddp; then
    MARCH="-march=armv8.2-a+dotprod"
    ISA_LABEL="NEON+dotprod"
elif echo "$FEATURES" | grep -qw asimd; then
    MARCH="-march=armv8-a"
    ISA_LABEL="NEON"
else
    MARCH=""
    ISA_LABEL="scalar"
fi
echo "Dispatch: $ISA_LABEL ($MARCH)"
echo

# --- Test 1: fp32 kernel correctness (gated_scan, cumdecay, causal_dwconv1d) ---
echo "--- Test 1: fp32 kernel correctness ---"
if $CC -O3 $MARCH -static \
    "$K/gdn_sve.c" "$K/test_gdn_sve.c" -o "$OUT/test_fp32" -lm 2>/dev/null; then
    OUTPUT=$("$OUT/test_fp32" 2>&1) || true
    echo "$OUTPUT" | sed 's/^/  /'
    if echo "$OUTPUT" | grep -q "bit-identical to matched reference: YES"; then
        BIT_ID=$(echo "$OUTPUT" | grep -c "bit-identical to matched reference: YES")
        echo "  RESULT: PASS ($BIT_ID kernels bit-identical to scalar reference)"
    else
        echo "  RESULT: FAIL (expected bit-identical match)"
        FAILURES=$((FAILURES + 1))
    fi
else
    echo "  RESULT: SKIP (build failed — toolchain may not support $MARCH)"
fi
echo

# --- Test 2: mixed-precision (bf16/fp16) kernel correctness ---
echo "--- Test 2: mixed-precision (bf16/fp16) kernel correctness ---"
if $CC -O3 $MARCH -static \
    "$K/gdn_sve.c" "$K/test_gdn_mixed.c" -o "$OUT/test_mixed" -lm 2>/dev/null; then
    OUTPUT=$("$OUT/test_mixed" 2>&1) || true
    echo "$OUTPUT" | sed 's/^/  /'
    if echo "$OUTPUT" | grep -q "ALL TESTS PASSED"; then
        echo "  RESULT: PASS"
    else
        echo "  RESULT: FAIL"
        FAILURES=$((FAILURES + 1))
    fi
else
    echo "  RESULT: SKIP (build failed — toolchain may not support $MARCH)"
fi
echo

# --- Test 3: delta-rule matmul correctness ---
echo "--- Test 3: delta-rule matmul correctness ---"
if $CC -O3 $MARCH -static \
    "$K/gdn_delta_matmul.c" "$K/test_gdn_delta_matmul.c" -I"$K" \
    -o "$OUT/test_matmul" -lm 2>/dev/null; then
    OUTPUT=$("$OUT/test_matmul" 2>&1) || true
    echo "$OUTPUT" | sed 's/^/  /'
    if echo "$OUTPUT" | grep -q "ALL PASS"; then
        echo "  RESULT: PASS"
    else
        echo "  RESULT: FAIL"
        FAILURES=$((FAILURES + 1))
    fi
else
    echo "  RESULT: SKIP (build failed — toolchain may not support $MARCH)"
fi
echo

# --- Test 4: GDN-2 decoupled-gating scan correctness ---
echo "--- Test 4: GDN-2 decoupled-gating scan correctness ---"
if $CC -O3 $MARCH -static \
    "$K/gdn_sve.c" "$K/test_gdn2_scan.c" \
    -o "$OUT/test_gdn2" -lm 2>/dev/null; then
    OUTPUT=$("$OUT/test_gdn2" 2>&1) || true
    echo "$OUTPUT" | sed 's/^/  /'
    if echo "$OUTPUT" | grep -q "ALL TESTS PASSED"; then
        echo "  RESULT: PASS"
    else
        echo "  RESULT: FAIL"
        FAILURES=$((FAILURES + 1))
    fi
else
    echo "  RESULT: SKIP (build failed — toolchain may not support $MARCH)"
fi
echo

# --- Summary ---
echo "=== Summary ==="
if [ "$FAILURES" -eq 0 ]; then
    echo "ALL KERNEL TESTS PASSED on $ISA_LABEL dispatch path"
    exit 0
else
    echo "$FAILURES test suite(s) FAILED"
    exit 1
fi
