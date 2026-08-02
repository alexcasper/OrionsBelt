# Setup and reproduction scripts

Entry points that make the repo reproducible by someone who has never seen it. Judges
score Developer Experience at 15 points, and bead `t-repro-rehearsal` verifies these by
following the documented path verbatim on a clean system.

## Available scripts

| Script | Purpose | Bead |
|---|---|---|
| `build_device_bench.sh` | Build static aarch64 device benchmark binaries (8 ISA variants) | `ob-8ms.2` |
| `verify_cpu_kernels.sh` | Cross-compile + QEMU verify GDN kernels (correctness only, never speed) | `ob-8qt.3` |
| `npu_op_probe.py` | Generate ONNX operator-coverage probes for NOE Compiler | `ob-t3b.1` |
| `run_op_probe_audit.py` | Run ONNX probes through onnxruntime and report results | `ob-t3b.1` |
| `fetch_weights.py` | Download Qwen3.5 checkpoints from HuggingFace (not vendored) | `ob-ixt` |
| `generate_prompts.py` | Generate deterministic needle-in-haystack + RULER prompt corpus | `ob-del` |
| `run_ablation.py` | Sweep engine configs across context lengths, produce comparison table | `ob-8qt.5` |

## Pending (hardware-gated)

- Orion O6 bring-up (Debian 12 flash, first boot) — `ob-tjs` follow-on
- NOE Compiler install (Python 3.10, x86 host) — `ob-huw`

Scripts should be non-interactive and idempotent.
