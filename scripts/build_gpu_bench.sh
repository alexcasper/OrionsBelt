#!/usr/bin/env bash
# Build the GDN GPU compute kernel benchmark and validation suite.
#
# Bead ob-q44 / ob-gzk. Produces two binaries:
#   gdn_gpu_bench    — benchmark with correctness check (p50/p95, bandwidth)
#   gdn_gpu_validate — comprehensive validation suite (87 tests across configs)
#
# Prerequisites:
#   - OpenCL ICD loader (ocl-icd-libopencl1, ocl-icd-opencl-dev)
#   - Mali GPU userspace driver (libmali-g610-x11 on RK3588, or equivalent)
#   - Kernel source at gpu/gdn_gpu_kernels.cl (loaded at runtime, not compiled in)
#
# Usage:
#   ./scripts/build_gpu_bench.sh
#   ./gpu/gdn_gpu_bench              # validate + benchmark
#   ./gpu/gdn_gpu_validate           # comprehensive 87-test validation suite
set -euo pipefail

SRC_DIR="gpu"
OUT_DIR="gpu"
mkdir -p "$OUT_DIR"

CC=${CC:-gcc}
CFLAGS="-O2 -Wall"

# Check for OpenCL
if ! pkg-config --exists OpenCL 2>/dev/null; then
    if [ ! -f /usr/lib/aarch64-linux-gnu/libOpenCL.so ] && \
       [ ! -f /usr/lib/aarch64-linux-gnu/libOpenCL.so.1 ]; then
        echo "ERROR: OpenCL development library not found."
        echo "Install with: sudo apt-get install ocl-icd-opencl-dev"
        exit 1
    fi
fi

echo "Building GDN GPU benchmark..."
$CC $CFLAGS -o "$OUT_DIR/gdn_gpu_bench" "$SRC_DIR/gdn_gpu_bench.c" -lOpenCL -lm
echo "  → $OUT_DIR/gdn_gpu_bench"

echo "Building GDN GPU validation suite..."
$CC $CFLAGS -o "$OUT_DIR/gdn_gpu_validate" "$SRC_DIR/gdn_gpu_validate.c" -lOpenCL -lm
echo "  → $OUT_DIR/gdn_gpu_validate"

echo ""
echo "Build complete. Run from the repo root so the kernel source path resolves:"
echo "  ./gpu/gdn_gpu_bench --repeats 50"
echo "  ./gpu/gdn_gpu_validate"
