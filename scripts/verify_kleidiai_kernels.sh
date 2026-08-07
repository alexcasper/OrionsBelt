#!/usr/bin/env bash
# Cross-compile the KleidiAI GDN submission kernels for aarch64 and verify
# them under QEMU.  Mirrors scripts/verify_cpu_kernels.sh but targets the
# kleidiai_submission/ package (the upstream-ready contribution), not the
# project's own src/ kernels.
#
# Usage:
#   bash scripts/verify_kleidiai_kernels.sh
#
# Runs on an x86 host — no Arm hardware required.
set -euo pipefail

SUB=kleidiai_submission
OUT=${OUT:-/tmp/kai_gdn_verify}
mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# Install cross-compiler and QEMU if missing (CI does this separately, but
# the script should be runnable locally too).
# ---------------------------------------------------------------------------
if ! command -v aarch64-linux-gnu-gcc >/dev/null 2>&1; then
  echo "ERROR: aarch64-linux-gnu-gcc not found.  Install: sudo apt-get install gcc-aarch64-linux-gnu"
  exit 1
fi
if ! command -v qemu-aarch64 >/dev/null 2>&1; then
  echo "ERROR: qemu-aarch64 not found.  Install: sudo apt-get install qemu-user"
  exit 1
fi

# ---------------------------------------------------------------------------
# Portability matrix: verify the kernels across ISA levels.
#
# The KleidiAI GDN kernels use a three-tier dispatch:
#   __ARM_FEATURE_SVE  → SVE path (vector-length-agnostic, predicated tails)
#   __ARM_NEON         → NEON path (double-width 8-lane unroll)
#   neither            → portable scalar C
#
# We verify all three tiers + multiple SVE vector widths.
# ---------------------------------------------------------------------------
PASS=0
FAIL=0

run_variant() {
  local label="$1"
  local march="$2"
  local qemu_cpu="$3"

  local bin="$OUT/test_kai_${label// /_}"

  echo ""
  echo "== $label ($march)"

  rm -f "$bin"
  # `|| true` -- under `set -e`, a failed compile here would kill the whole
  # script instead of letting the `[ ! -x "$bin" ]` check below report a
  # graceful per-tier FAIL and continue to the next variant.
  aarch64-linux-gnu-gcc -O3 -Wall -Wextra -std=c11 -static "$march" -I "$SUB" \
    "$SUB"/test_kai_gdn.c \
    "$SUB"/kai/ukernels/gdn/*.c \
    -o "$bin" -lm 2>&1 || true

  if [ ! -x "$bin" ]; then
    echo "  FAIL (compile failed)"
    FAIL=$((FAIL + 1))
    return
  fi

  # Run under QEMU.  The test binary prints "ALL TESTS PASSED" on success
  # and returns exit code 0.
  local out
  if out=$(QEMU_CPU="$qemu_cpu" qemu-aarch64 "$bin" 2>&1); then
    if printf '%s' "$out" | grep -q "ALL TESTS PASSED"; then
      local n
      n=$(printf '%s' "$out" | grep "Tests run:" | grep -oP '\d+' | head -1)
      echo "  PASS ($n tests)"
      PASS=$((PASS + 1))
    else
      echo "  FAIL (unexpected output)"
      echo "$out" | tail -5
      FAIL=$((FAIL + 1))
    fi
  else
    echo "  FAIL (exit code $?)"
    echo "$out" | tail -5
    FAIL=$((FAIL + 1))
  fi
}

echo "=== KleidiAI GDN kernel verification ==="
echo ""

# SVE2 (as Cortex-A720 target — the headline target for the submission)
run_variant "SVE2-128" \
  "-march=armv9-a+sve2" \
  "max,sve128=on"

# SVE1 at 128-bit vectors (Armv8.2+SVE, e.g. Neoverse-V1 / Graviton 3)
run_variant "SVE1-128" \
  "-march=armv8.2-a+sve" \
  "max,sve128=on"

# SVE1 at 256-bit vectors (e.g. Graviton 3)
run_variant "SVE1-256" \
  "-march=armv8.2-a+sve" \
  "max,sve256=on"

# NEON-only (Armv8.0 floor — Cortex-A57, our Jetson Nano)
run_variant "NEON-A57" \
  "-march=armv8-a" \
  "max"

# NEON with dotprod (Armv8.2 — Cortex-A76, Pi 5 / RK3588)
run_variant "NEON-A76" \
  "-march=armv8.2-a+dotprod" \
  "max"

echo ""
echo "=== Summary: $PASS passed, $FAIL failed ==="

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
