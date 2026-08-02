# Hedge-target bring-up (RK3588)

**Bead:** `ob-ng6` · **ADR:** [0002 — portable hedge target](adr/0002-portable-hedge-target.md)
**Device:** Radxa Rock 5B / RK3588 (4× Cortex-A76 @ 2.3 GHz + 4× Cortex-A55 @ 1.8 GHz, Mali-G610 MP4, RKNPU)
**Date:** 2026-08-02

This document records the runtime bring-up of the RK3588 as the Edge AI hedge target
(ADR 0002). The critical requirement is confirming which ISA extensions and GPU backends
are **actively available at runtime**, not merely compiled into a binary. A binary built
with `-march=armv9-a` will run on an Armv8.2 core by silently taking NEON fallback paths;
`getauxval(AT_HWCAP)` is the ground truth.

---

## Runtime ISA features

Probed with [`src/orionsbelt/engines/cpu/bringup_probe.c`](../src/orionsbelt/engines/cpu/bringup_probe.c):

```
AT_HWCAP  = 0x0000000000119fff
AT_HWCAP2 = 0x0000000000000000
```

| Feature | Runtime | GDN kernel path affected |
|---|---|---|
| **NEON (ASIMD)** | ✅ YES | Baseline 128-bit SIMD — all kernels use this |
| **FP16 (fphp/asimdhp)** | ✅ YES | `gdn_sve_f16.c` NEON fp16 state variant (`ob-8qt.4`) |
| **DOTPROD (asimddp)** | ✅ YES | KleidiAI SDOT matmul micro-kernels (`ob-8qt.2`) |
| SVE | ❌ no | `gdn_sve.c` SVE path — **not exercised**, NEON fallback used |
| SVE2 | ❌ no | Widening MAC, bf16 dot — not available |
| I8MM | ❌ no | INT8/INT4 GEMV decode path — not available |
| BF16 (vector) | ❌ no | `gdn_sve_f16.c` bf16 state variant uses scalar fallback |

**Verdict:** Cortex-A76/A55 class (Armv8.2-A + dotprod). The GDN kernels run on the NEON
path with fp16 + dotprod. The SVE reference kernels and i8mm GEMV are not exercised on this
device — only the portable NEON fallbacks.

### What this means for the project

1. **CPU-first mapping (PLAN.md §3.1) holds.** The dotprod SDOT instruction covers the
   delta-rule small matmuls (`ob-8qt.2`), and the fp16 NEON path handles the narrowed-state
   variants (`ob-8qt.4`). Both verified correct on this device.

2. **i8mm is absent → INT4/INT8 decode GEMV via KleidiAI's i8mm micro-kernels is NOT
   available here.** The dotprod (SDOT) family covers INT8 matmul instead. This is an
   honest limitation: the decode bandwidth win from INT4 quantization (`ob-qpa`,
   `ob-onz`) cannot use hardware i8mm on the RK3588. The O6's Cortex-A720 (Armv9.2) has
   i8mm and would close that gap.

3. **The SVE path is untested on real hardware.** The SVE kernels in `gdn_sve.c` are
   verified under QEMU (128/256/512-bit) by `scripts/verify_cpu_kernels.sh`, but no
   physical SVE core exists in the hedge fleet. The O6's A720 cores will exercise it.

---

## GPU / accelerator

| Device | Driver | Status |
|---|---|---|
| Mali-G610 MP4 | `panfrost` (Mesa, open) | OpenGL ES via `panfrost_dri.so`; **no Vulkan loader** |
| RKNPU | `RKNPU` (vendor) | Present (`/dev/dri/renderD129`); toolchain not installed |
| Vulkan | — | `libvulkan.so.1` absent → **GPU compute via Vulkan NOT available** |

**Vulkan is not available on this device image.** The `panfrost` driver provides OpenGL ES
but the Mesa Vulkan driver for Valhall (`panvk`) is not installed. This means the GPU
compute-shader scan path (PLAN.md §3.1, `t-gpu-scan`) cannot be exercised here. On the O6,
the Immortalis G720's native Vulkan driver would be used instead.

---

## Build and run

### GDN kernel benchmark (static binary)

```bash
# Build (native gcc on device, or cross-compile on x86)
./scripts/build_device_bench.sh

# Pin to big cluster (A76, cpu4-7) — big.LITTLE MUST be pinned
taskset -c 4-7 ./dist/bench_gdn_armv8.2dot --repeats 30 --csv > results/raw/rk3588-t3_big.csv

# Pin to little cluster (A55, cpu0-3)
taskset -c 0-3 ./dist/bench_gdn_armv8a --repeats 30 --csv > results/raw/rk3588-t3_little.csv
```

### Runtime feature probe

```bash
gcc -O2 -o bringup_probe src/orionsbelt/engines/cpu/bringup_probe.c
./bringup_probe
```

### Verify all kernel variants

```bash
gcc -O3 -march=armv8.2-a+dotprod+fp16 -static \
    src/orionsbelt/engines/cpu/kernels/gdn_sve.c \
    src/orionsbelt/engines/cpu/kernels/gdn_sve_f16.c \
    src/orionsbelt/engines/cpu/kernels/test_gdn_sve_f16.c \
    -o verify_f16 -lm
./verify_f16    # → ALL TESTS PASS
```

### Set performance governor (for valid numbers)

```bash
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo cell | sudo -S tee "$c" >/dev/null
done
```

---

## Device specs

| Spec | Value |
|---|---|
| SoC | Rockchip RK3588 |
| CPU | 4× Cortex-A76 @ 2.3 GHz (big) + 4× Cortex-A55 @ 1.8 GHz (little) |
| GPU | Mali-G610 MP4 (Valhall) |
| NPU | 6 TOPS (RKNN) |
| RAM | 32 GB LPDDR4x |
| ISA | Armv8.2-A + dotprod + fp16 (NEON only, no SVE/i8mm/bf16) |
| Kernel | 5.10.160-rockchip |
| Memory bandwidth (spec, est.) | ~34 GB/s (quad-channel 16-bit) |

**Achieved bandwidth (benchmark):** 1.9–5.6 GiB/s single-core NEON (6–16% of spec). See
`bd memories "rk3588-a76-gdn-kernel-benchmark"` and `docs/FINDINGS.md`.
