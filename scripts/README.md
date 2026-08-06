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
| `detect_isa.sh` | Standalone ISA feature detection for aarch64 — no Python required. Reports active dispatch features (NEON, dotprod, i8mm, SVE/SVE2, bf16) and recommends the correct bench binary. Mirrors `isa_detect.py`. | `ob-ng6` |
| `fetch_weights.py` | Download model weights (not vendored in the repo). | `ob-del` |
| `npu_op_probe.py` | Generate minimal per-operator ONNX probe graphs for the NOE op-coverage audit. | `ob-t3b.2` |
| `power_bench.sh` | Power-instrumented benchmark wrapper for Jetson Nano. Samples the onboard INA3221 power monitor while `bench_gdn` runs, reports energy-per-GiB alongside throughput. Requires sudo. | `ob-agf.1` |
| `run_op_probe_audit.py` | Drive the NOE Compiler (cixparse) over the probe graphs and record results. | `ob-t3b.1` |
| `generate_memory_plots.py` | Generate memory scaling figures (stacked bar + area chart + comparison table) from the analytical model in `memory.py`. Shows GDN O(1) state vs attention O(n) KV cache. | `ob-9t0.4` |
| `check_submission_readiness.sh` | Verify the repo is ready for Devpost submission: tests, lint, format, memory plots, fleet analysis, deliverable files, credential scan, CSV validation. | `ob-9t0.5` |

## Principles

- Scripts should be non-interactive and idempotent.
- Device-side scripts (`verify_kernels_native.sh`, `capture_manifest.sh`) must work without
  Python ≥3.10 — many edge devices ship with older Python (e.g. Jetson Nano = 3.6.9).
- Device-side power monitoring (`power_bench.sh`) uses the Jetson Nano's built-in
  INA3221 via IIO sysfs — no perf/ftrace/powertop needed, but does require sudo for
  the IIO entries.
