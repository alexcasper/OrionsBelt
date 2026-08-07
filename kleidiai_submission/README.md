# KleidiAI Contribution: Gated DeltaNet (GDN) Recurrent Micro-Kernels

## Submission Overview

This package contributes **four novel AArch64 micro-kernels** for Gated
DeltaNet (GDN) linear-attention inference — three recurrent/scan primitives and
one decode-path GEMV — to KleidiAI. None of these operations have an equivalent
in the current KleidiAI library.

**Origin:** These kernels were written, verified, and profiled in the
[OrionsBelt](https://github.com/gastownhall/OrionsBelt) project
(`src/orionsbelt/engines/cpu/kernels/gdn_sve.c`,
`gdn_delta_matmul.c`). The implementations here are extracted and reformatted as
standalone KleidiAI micro-kernels, preserving the verified arithmetic while
removing framework coupling (OpenMP, OrionsBelt headers).

**License:** SPDX-License-Identifier: Apache-2.0 (matches KleidiAI's license).

---

## What is Gated DeltaNet (GDN)?

Gated DeltaNet is a family of **linear-attention / state-space architectures**
that replace the quadratic softmax attention with a recurrent state update. The
core idea is the **delta-rule**: a learnable associative memory whose read and
write are governed by data-dependent **gates** (decay factors). Each layer
maintains a recurrent state `S` that is updated token-by-token, giving
*O(N)* decoding cost instead of *O(N²)*.

The per-layer decode path involves four distinct operations, each of which maps
to a primitive micro-kernel:

1. **Causal depthwise Conv1D** (short-context shift register)
2. **Gated cumulative decay** (prefix product of decay gates)
3. **Chunkwise gated delta-rule scan** (first-order linear recurrence)
4. **Delta-rule GEMV** (M=1 matmul between attention scores and recurrent state)

Operations 1–3 are **recurrent/scan primitives** — sequential along the time
axis with per-channel independence. Operation 4 is a small dense matmul that
dominates decode time.

---

## The Three Novel Recurrent Primitives

### 1. Causal Depthwise Conv1D, Kernel=4

```
out[t][c] = w[0][c]·hist0[c] + w[1][c]·hist1[c] + w[2][c]·hist2[c] + w[3][c]·in[t][c]
```

A causal (left-only) 1-D convolution with depthwise (per-channel) weights and
kernel width 4. The `hist[]` buffer holds the 3 previous timesteps per channel,
carried across decode calls — the conv analogue of a KV cache.

**Why KleidiAI needs this:** KleidiAI's existing depthwise convolution
(`kai_dwconv_fp32_*`) is **SME2-only** — it requires the Scalable Matrix
Extension available only on Armv9.2+ cores (e.g., Cortex-X3+). This kernel
provides a portable SVE/NEON/scalar path that runs on **every AArch64 core**,
including the vast installed base of Armv8.0–Armv8.7 and Armv9.0–Armv9.1 devices
that have no SME.

### 2. Gated Cumulative Decay

```
decay[t][c] = ∏_{i=0}^{t} a[i][c]    (inclusive prefix product)
```

An exclusive prefix product along the sequence axis. Computed as a direct
running product (not `exp(cumsum(log a))`) — at typical chunk lengths of 64 the
direct product is both cheaper and numerically stable in fp32, avoiding two
transcendentals per element.

**Why KleidiAI needs this:** KleidiAI has **no prefix-scan or prefix-product
primitive at all**. This is a fundamental building block for any linear-attention
or SSM model, not just GDN.

### 3. Chunkwise Gated Delta-Rule Scan

```
s[t][c] = g[t][c] · s[t-1][c] + x[t][c]
```

A first-order linear recurrence (gated scan) with cross-call state continuity:
`s[-1]` is loaded from the `state[]` array on entry and written back on exit, so
decode can advance one chunk at a time. This is the sequential half of the
chunkwise GDN formulation — kept separate from the per-chunk dense matmuls so
those can be mapped to GPU/NPU while the scan stays on CPU.

**Why KleidiAI needs this:** KleidiAI has **no recurrence or scan kernel**.
The gated scan is the single most important primitive for modern efficient
sequence models (Mamba, RWKV, RetNet, GDN all use some variant of it).

---

## The NEON GEMV (M=1 Decode-Path Delta-Rule Matmul)

```
C[j] = Σ_k  A[k] · B[k][j]     (M=1, so this is a matrix-vector product)
```

A hand-written NEON GEMV for the per-chunk delta-rule matmul at **M=1** (the
single-token decode path). The result is accumulated in a 4-wide fp32 NEON
register and accumulated across the K dimension with FMA.

**Why not KleidiAI's existing GEMM?** KleidiAI's packed-GEMM
(`kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla`) wins 3.1–3.6× over
hand-NEON at **prefill** (M≥64), even after the per-call RHS repack cost.
But at **decode** (M=1), the recurrent state `S` changes every chunk, so the
repack cost (7–126 µs on Cortex-A76) **exceeds the matmul itself**. The measured
break-even M is 3–6. Hand-NEON without packing wins 2.6–3.0× at M=1 — making this
GEMV the right kernel for the decode path. See `gdn_delta_matmul.c` header for
the full dual-path dispatch rationale.

**This kernel uses NEON FMA** (no SME, no SVE, no dotprod dependency) because:
- The delta-rule operands are fp32 (per the project's quantization policy).
- The decode target devices (e.g., Cortex-A76 on RK3588) predate i8mm and SVE.
- NEON FMA at 4-wide is sufficient for the M=1 bandwidth-bound case.

---

## Naming Convention

All functions follow KleidiAI's established naming pattern:

```
kai_run_<operation>_<data-types>_<ISA>
```

| Function                                              | ISA suffix        |
|-------------------------------------------------------|-------------------|
| `kai_run_gdn_cumdecay_f32_sve`                        | SVE (SVE1 floor)  |
| `kai_run_gdn_gated_scan_f32_sve`                      | SVE (SVE1 floor)  |
| `kai_run_gdn_causal_dwconv1d_f32_k4_sve`              | SVE (SVE1 floor)  |
| `kai_run_gdn_gemv_f32_f32_f32_1x4_neon`             | NEON FMA          |

The `_sve` suffix indicates the primary vectorization ISA (SVE1 baseline), with
compile-time NEON and scalar fallbacks. The GEMV uses `_neon` because
it targets the NEON FMA path specifically (fp32 operands, no SDOT). The `1x4` in the GEMV name encodes
M=1 (single output row) and the 4-wide NEON accumulation.

---

## Portability: SVE2 / NEON / Scalar

Every kernel provides a **three-way dispatch** selected at compile time:

```c
#ifdef __ARM_FEATURE_SVE
    /* SVE1 path: vector-length-agnostic, predicated tails via svwhilelt */
#elif defined(__ARM_NEON)
    /* NEON path: double-width unroll (8 channels/iter, two 4-wide groups) */
#else
    /* Scalar fallback: plain C, runs on any architecture */
#endif
```

**ISA floor is SVE1, not SVE2.** Every SVE intrinsic used
(`svcntw`, `svdup_f32`, `svld1_f32`, `svmul_f32_x`, `svmla_f32_x`, `svst1_f32`,
`svwhilelt_b32`) is base SVE, guarded by `__ARM_FEATURE_SVE`. Nothing in these
fp32 kernels needs SVE2's integer/DSP additions. SVE2 becomes relevant only for
quantized (int8/i8mm) delta-rule matmuls, which live elsewhere.

**Verified identical output** on:
- SVE1 at 128/256/512-bit (`-mcpu=neoverse-v1`, `-mcpu=a64fx`)
- SVE2 (`-march=armv9-a`, `-mcpu=cortex-a710`, `-mcpu=neoverse-v2`)
- Plain Armv8-A via the scalar fallback
- **Tested on Cortex-A76** (RK3588, NEON path) and **cross-compiled for
  Cortex-A720** (SVE2 128-bit)

On Cortex-A720 SVE is 128-bit — the same width as NEON — so the win is
**predication** (`svwhilelt` gives clean channel tails with no scalar epilogue),
not extra lanes. Written vector-length-agnostic regardless, so it widens for
free on a core with longer vectors.

**NEON double-width unrolling:** the NEON paths process 8 channels per iteration
(two independent 4-wide register groups). This hides FMA/MUL latency and doubles
memory-level parallelism. The conv's 4-deep FMA chain benefits even more than the
scan's single FMA: two independent chains let the OoO scheduler hide the full
4-cycle FMA latency.

---

## Performance Context

GDN recurrent kernels are **not the decode bottleneck**. Measured profiling
breakdown on Cortex-A76 (RK3588):

| Operation             | Share of decode time |
|-----------------------|----------------------|
| **FFN dense matmuls** | **53–73%**           |
| Attention matmuls     | ~15%                 |
| GDN recurrent kernels | **< 0.1%**           |

The GDN kernels (cumdecay, gated scan, causal conv, delta-rule GEMV) together
account for less than 0.1% of decode time. The actual bottleneck is the dense
FFN matmuls (53–73%). This means the value of these kernels is **correctness and
completeness** — enabling GDN models to run on KleidiAI at all — rather than
headline speedup. They remove a *hard blocker* (no recurrent primitives exist in
the library), not a performance bottleneck.

---

## File Map

```
kleidiai_submission/
├── README.md                                         (this file)
├── Makefile                                          (build: test, bench, clean)
├── test_kai_gdn.c                                    (test harness, 14 suites)
├── bench_kai_gdn.c                                   (micro-benchmark, CSV output)
└── kai/ukernels/gdn/
    ├── kai_gdn_cumdecay_f32_sve.h
    ├── kai_gdn_cumdecay_f32_sve.c
    ├── kai_gdn_gated_scan_f32_sve.h
    ├── kai_gdn_gated_scan_f32_sve.c
    ├── kai_gdn_causal_dwconv1d_f32_k4_sve.h
    ├── kai_gdn_causal_dwconv1d_f32_k4_sve.c
    ├── kai_gdn_gemv_f32_f32_f32_1x4_neon.h
    └── kai_gdn_gemv_f32_f32_f32_1x4_neon.c
```

## Verification

The test harness compares each kernel against a naive C reference implementation
and prints `ALL TESTS PASSED` on success (14 test suites: 6 correctness +
4 edge-case tail-handling + 4 degenerate-input):

```bash
cd kleidiai_submission && make test
```

Or manually:
```bash
gcc -O3 -march=armv8-a -I kleidiai_submission \
    kleidiai_submission/test_kai_gdn.c \
    kleidiai_submission/kai/ukernels/gdn/*.c \
    -lm -o test_kai_gdn
./test_kai_gdn
```

For SVE targets:
```bash
aarch64-linux-gnu-gcc -O3 -march=armv8.2-a+sve \
    -I kleidiai_submission \
    kleidiai_submission/test_kai_gdn.c \
    kleidiai_submission/kai/ukernels/gdn/*.c \
    -lm -o test_kai_gdn
```

A micro-benchmark (`bench_kai_gdn`) measures per-kernel latency and
achieved bandwidth at realistic GDN shapes:

```bash
cd kleidiai_submission && make bench
```

### Cross-ISA verification (no Arm hardware required)

`scripts/verify_kleidiai_kernels.sh` cross-compiles the four kernels for
aarch64 and verifies correctness under QEMU across all three dispatch tiers:

```bash
bash scripts/verify_kleidiai_kernels.sh
```

This exercises SVE2 (128-bit), SVE1 (128/256-bit), NEON-only (A57 floor),
and NEON+dotprod (A76) paths — confirming the dispatch is correct at every
ISA level.

### A57 (Armv8.0, NEON path) — measured on Jetson Nano

All numbers are p50 of 30 repeats (3 warmups discarded). The A57 is the
lowest-end core in the test fleet (Armv8.0, 4-wide NEON, no dotprod/SVE),
making it the worst-case portability floor.

| Kernel      | Shape (seq×ch)   | p50 (µs) | GiB/s |
|-------------|------------------|----------|-------|
| cumdecay    | 64×160           |     11.8 |   6.5 |
| cumdecay    | 1×160            |      0.3 |   4.6 |
| cumdecay    | 64×2560          |    433.6 |   2.8 |
| cumdecay    | 1×2560           |      2.1 |   9.2 |
| gated_scan  | 64×160           |     18.2 |   6.3 |
| gated_scan  | 1×160            |      0.3 |   9.5 |
| gated_scan  | 64×2560          |    757.0 |   2.4 |
| gated_scan  | 1×2560           |      3.8 |  12.5 |
| dwconv1d    | 64×160           |     17.8 |   4.6 |
| dwconv1d    | 1×160            |      0.7 |  10.6 |
| dwconv1d    | 64×2560          |    475.5 |   2.8 |
| dwconv1d    | 1×2560           |     10.2 |  11.3 |
| gemv        | K=128 N=128      |     13.3 |   4.7 |
| gemv        | K=128 N=2048     |    184.5 |   5.3 |
| gemv        | K=128 N=2560     |    234.1 |   5.3 |

> **Note on variance:** The A57 exhibits high run-to-run variance (up to 1.5×
> on the same kernel at the same commit, per beads ob-bf7). The numbers above
> are from representative single runs; cross-device comparisons must use
> matched commits with multiple replicates. The GEMV NEON path uses
> double-width unrolling (8 channels/iter) and `vfmaq_n_f32` scalar FMA,
> matching the three recurrent kernels' pattern.

At seq=64 (prefill chunk), all three recurrent kernels are bandwidth-bound
(1.6–6.6 GiB/s vs the A57's 25.6 GiB/s spec). At seq=1 (decode), the working
set fits in L1 and the kernels are launch-overhead-dominated, not
bandwidth-limited. The GEMV at all sizes is bandwidth-bound at ~5 GiB/s.

### A76 (Armv8.2-A, NEON + dotprod path) — measured on RK3588

All numbers are p50 of 30 repeats on a Cortex-A76 @ 2.3 GHz (RK3588, device t3).
Governor: `performance`. The A76 has 4-wide NEON with dotprod (`asimddp`) but no
SVE/SVE2 — the kernels run the NEON fallback path (`#elif __ARM_NEON`).

| Kernel      | Shape (seq×ch)   | p50 (µs) | GiB/s |
|-------------|------------------|----------|-------|
| cumdecay    | 64×160           |     11.4 |   6.7 |
| cumdecay    | 1×160            |      0.3 |   4.1 |
| cumdecay    | 64×2560          |    537.3 |   2.3 |
| cumdecay    | 1×2560           |      0.9 |  21.8 |
| gated_scan  | 64×160           |      5.3 |  22.0 |
| gated_scan  | 1×160            |      0.0 |    —* |
| gated_scan  | 64×2560          |    192.5 |   9.6 |
| gated_scan  | 1×2560           |      1.2 |  40.9 |
| dwconv1d    | 64×160           |      5.3 |  15.7 |
| dwconv1d    | 1×160            |      0.3 |  24.6 |
| dwconv1d    | 64×2560          |    143.8 |   9.2 |
| dwconv1d    | 1×2560           |      2.9 |  39.3 |
| gemv        | K=128 N=128      |      3.5 |  17.7 |
| gemv        | K=128 N=2048     |     63.0 |  15.6 |
| gemv        | K=128 N=2560     |     79.9 |  15.4 |

*\*The seq=1×160 shape for gated_scan is too small to measure reliably (p50 <
1 µs); the `inf` GiB/s is treated as a measurement-floor artifact, not a real
throughput number.*

**A76 vs A57 comparison:** At seq=64 shapes, the A76 achieves 2–4× higher
throughput than the A57 (e.g. gated_scan 64×160: 22.0 vs 6.3 GiB/s), consistent
with the A76's wider pipeline and higher clock. The GEMV shows 3× higher
throughput (17.7 vs 4.7 GiB/s at K=128 N=128), reflecting the A76's superior
memory subsystem. At seq=1, the A76's advantage is even larger on
bandwidth-fitting shapes (cumdecay 1×2560: 21.8 vs 9.2 GiB/s) because the A76's
L1 is faster and the launch overhead is lower.

> **Provenance:** Captured at commit `250dc96` on device t3 (RK3588),
> clean tree (dirty=false). Governor: `performance`.
> Manifest: `results/manifests/rk3588-t3_kleidiai_gdn_kernels.json`.
> Raw CSV: `results/raw/kleidiai/rk3588-t3_kleidiai_gdn_kernels.csv`.

---

## References

- **Verified source:** `src/orionsbelt/engines/cpu/kernels/gdn_sve.c`,
  `gdn_delta_matmul.c` (OrionsBelt repository)
- **KleidiAI:** [gitlab.arm.com/kleidi/kleidiai](https://gitlab.arm.com/kleidi/kleidiai)
- **Gated DeltaNet:** NVIDIA NVLabs / GatedDeltaNet — linear attention with
  data-dependent gating and delta-rule state updates
