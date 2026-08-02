# Setup and reproduction scripts

Entry points that make the repo reproducible by someone who has never seen it. Judges
score Developer Experience at 15 points, and bead `ob-kdi` verifies these by
following the documented path verbatim on a clean system.

## Scripts

| Script | Purpose | Bead |
|---|---|---|
| `build_device_bench.sh` | Build static GDN kernel benchmark binaries for each Arm ISA variant (ArmV8-A through ArmV9.2-SVE2). Outputs to `dist/`. | `ob-8ms.2` |
| `verify_cpu_kernels.sh` | Cross-compile SVE2 kernels and verify numerical correctness under QEMU. The project's core correctness gate — runs in CI. | `ob-8qt.3` |
| `verify_kernels_native.sh` | Build and run kernel correctness tests natively on the device's real ISA (no QEMU). For fleet devices. | `ob-mrd.3` |
| `capture_manifest.sh` | Shell-based provenance capture — same JSON schema as `bench/manifest.py` but no Python dependency. For devices with Python <3.10. | `ob-mrd.4` |
| `fetch_weights.py` | Download model weights (not vendored in the repo). | `ob-del` |
| `npu_op_probe.py` | Generate minimal per-operator ONNX probe graphs for the NOE op-coverage audit. | `ob-t3b.2` |
| `run_op_probe_audit.py` | Drive the NOE Compiler (cixparse) over the probe graphs and record results. | `ob-t3b.1` |

## Principles

- Scripts should be non-interactive and idempotent.
- Device-side scripts (`verify_kernels_native.sh`, `capture_manifest.sh`) must work without
  Python ≥3.10 — many edge devices ship with older Python (e.g. Jetson Nano = 3.6.9).
- The benchmark binary (`bench_gdn.c`) links statically — one binary copies to any aarch64
  device and runs with no toolchain, no Python, and no shared libraries.
