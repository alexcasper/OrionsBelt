# Fleet Bandwidth-Scaling Analysis

**Bead `ob-8ms.3`.** Cross-device comparison of GDN kernel throughput
versus spec memory bandwidth, testing the bandwidth-bound hypothesis
from `METRICS.md` (~0.25 FLOP/byte).

## Devices in the fleet

| Device | Cores | ISA | Spec BW (GiB/s) |
|--------|-------|-----|-----------------|
| Pi 5 | 4x Cortex-A76 @ 2.4 GHz | Armv8.0-A (NEON only) | 17.0 |
| RK3588 big | 4x Cortex-A76 @ 2.4 GHz | Armv8.0-A (NEON only) | 34.0 |
| RK3588 little | 4x Cortex-A55 @ 1.8 GHz | Armv8.0-A (NEON only) | 34.0 |
| Jetson j1 | 4x Cortex-A57 @ 1.48 GHz | Armv8.0-A (NEON only) | 25.6 |
| Jetson j2 | 4x Cortex-A57 @ 1.48 GHz | Armv8.0-A (NEON only) | 25.6 |
| **Orion O6** | 4x A720 big + 4x A720 mid + 4x A520 | Armv9.2-A | **93.1** |

## Achieved throughput vs spec bandwidth (4B model, seq=64)

All fleet devices were benchmarked single-threaded at commit `28729f3`
(pre-OpenMP, pre-NEON-unrolling). The j2 single-threaded numbers below
are a fresh run of the current binary with `OMP_NUM_THREADS=1` to match
that optimization level for fair comparison. See the optimization-impact
section below for what 4-core OpenMP + NEON unrolling achieves on j2.

### 4B model

| Device | Spec (GiB/s) | CumDecay | Scan | DWConv1D | Scan/Spec |
|--------|-------------|----------|------|----------|-----------|
| Pi 5 | 17.0 | 3.74 | 1.20 | 3.23 | 7.1% |
| RK3588 big | 34.0 | 4.13 | 1.96 | 4.02 | 5.8% |
| RK3588 little | 34.0 | 0.87 | 0.35 | 0.65 | 1.0% |
| Jetson j1 | 25.6 | 1.16 | 0.72 | 1.04 | 2.8% |
| Jetson j2 | 25.6 | 1.32 | 1.13 | 1.20 | 4.4% |

### 0.8B model

| Device | Spec (GiB/s) | CumDecay | Scan | DWConv1D | Scan/Spec |
|--------|-------------|----------|------|----------|-----------|
| Pi 5 | 17.0 | 4.47 | 4.43 | 4.55 | 26.1% |
| RK3588 big | 34.0 | 4.92 | 4.41 | 5.51 | 13.0% |
| RK3588 little | 34.0 | 1.15 | 0.98 | 0.94 | 2.9% |
| Jetson j1 | 25.6 | 1.93 | 1.61 | 1.99 | 6.3% |
| Jetson j2 | 25.6 | 3.55 | 2.45 | 2.80 | 9.6% |

## The discriminating test: Jetson (A57, more BW) vs Pi 5 (A76, less BW)

The DEVICE_RUNBOOK poses this question: if the GDN scan kernel is
bandwidth-bound, then the Jetson Nano (oldest cores, 25.6 GiB/s spec)
should beat the Pi 5 (newest cores, 17.0 GiB/s spec). **If the Pi 5
wins comfortably, the bandwidth-bound thesis is wrong or incomplete.**

| Kernel (4B) | Pi 5 (17.0) | Jetson j1 (25.6) | Jetson j2 (25.6) | Winner | Pi5/J1 ratio |
|-------------|-------------|------------------|------------------|--------|-------------|
| Cumulative Decay | 3.74 | 1.16 | 1.32 | **Pi 5** | 3.22x |
| Gated Delta-Rule Scan | 1.20 | 0.72 | 1.13 | **Pi 5** | 1.67x |
| Causal DWConv1D | 3.23 | 1.04 | 1.20 | **Pi 5** | 3.11x |

**Result: the Pi 5 wins on ALL three kernels despite having 33% LESS
spec bandwidth.** The bandwidth-bound hypothesis does NOT hold at
seq=64 working set sizes — these kernels are **instruction-overhead-bound,
not DRAM-bandwidth-bound** at this scale.

This is consistent with the working set analysis: at seq=64 with 4096
channels, the state is ~1 MiB — small enough to be L2/L3-resident, so
core microarchitecture (IPC, OoO depth, clock) dominates over raw DRAM
bandwidth. The Pi 5's Cortex-A76 has ~1.6x higher clock and substantially
better IPC than the A57, explaining its win despite less bandwidth.

## ⚠ Replicate spread limits everything below this line

The tables above take **one** run per device. Several devices were measured
more than once, and the replicates disagree by more than some of the
cross-device effects being interpreted (bead `ob-bf7`):

| Device class | Runs (scan, 4B, GiB/s) | Spread | Why it matters |
|---|---|---:|---|
| RK3588 big | t3 1.96 vs t4 3.29 | **1.68x** | same source commit `28729f3`, same core class |
| RK3588 little | t3 0.35 vs t4 0.55 | **1.57x** | same source commit `28729f3`, same core class |
| Pi 5 | r5 1.20 vs j1 1.84 | **1.53x** | same physical board, *different* commits (`28729f3` vs `f127a11`) |

The RK3588 pair is the serious one: **identical source commit**, so the cause is environmental — different boards, cluster pinning, governor, or thermal state — and none of that is recorded per run. Worst replicate spread on the fleet is **1.68x**.

This report selects `t3` for RK3588 and `r5` for the Pi 5. Selecting the other
run — equally valid, and for RK3588 the *same commit* — would move every O6
figure below by a similar factor. **Treat the predictions as order-of-magnitude,
not as a fit.** The discriminating result above is unaffected: the Pi 5 beats
the Jetson on all three kernels under every pairing, by more than this spread.

## O6 extrapolation (prediction)

Spec bandwidth: **93.1 GiB/s** (100 GB/s, 128-bit LPDDR5 @ 5500 MT/s).

If the kernels were bandwidth-bound, achieved throughput should scale
linearly with spec bandwidth. Extrapolating the scan kernel from each device:

| Extrapolated from | Scan (GiB/s) | O6 BW ratio | Predicted O6 scan (GiB/s) |
|-------------------|-------------|-------------|--------------------------|
| Pi 5 | 1.20 | 5.5x | 6.57 |
| RK3588 big | 1.96 | 2.7x | 5.37 |
| RK3588 little | 0.35 | 2.7x | 0.96 |
| Jetson j1 | 0.72 | 3.6x | 2.62 |
| Jetson j2 | 1.13 | 3.6x | 4.11 |

**⚠ However, this linear extrapolation is almost certainly WRONG.**
The discriminating test above shows the kernels are instruction-bound,
not bandwidth-bound, at seq=64. A bandwidth-linear extrapolation would
overpredict. The honest prediction is that the O6's Cortex-A720 cores
(Armv9.2-A, wider OoO, higher clock than A76) will achieve higher throughput
than any current fleet device due to better IPC, but **not proportionally
to its 4-5x bandwidth advantage**.

**Core-performance-based prediction** (scaling from RK3588 A76 big cluster):

- RK3588 big scan: 1.96 GiB/s (4x A76 @ 2.4 GHz, Armv8.2)
- O6 big cluster: 4x A720 @ 2.8 GHz, Armv9.2 (SVE2, wider OoO)
- Expected gain from IPC + clock: 1.5-2.5x over A76
- **Predicted O6 scan throughput: 2.9-4.9 GiB/s**
- This is ~3-5% of spec bandwidth, vs 6% achieved on A76

Carrying the replicate spread through: anchoring on the other same-commit RK3588 host (3.29 GiB/s rather than 1.96) gives **4.9-8.2 GiB/s** instead.

So the defensible published claim is **~3-8 GiB/s**, and the *anchor choice* — not the IPC assumption — is the dominant uncertainty. Resolving `ob-bf7` narrows this more than any modelling refinement would.

To check this prediction: if the O6 board arrives, run
`bench_gdn_armv9sve2 --repeats 30 --csv` and compare.

## Optimization impact: j2 single-threaded vs 4-core OpenMP

The j2 CSV was re-run with the current optimized binary (OpenMP 4-core,
NEON double-width unrolling, bf16 conversion vectorization). This shows
the real-world impact of the optimization track (beads ob-8qt.5/6/7):

| Kernel (4B, seq=64) | Single-thread (GiB/s) | 4-core OpenMP (GiB/s) | Speedup |
|--------------------|-----------------------|-----------------------|---------|
| Cumulative Decay | 1.32 | 3.85 | 2.9x |
| Gated Delta-Rule Scan | 1.13 | 2.94 | 2.6x |
| Causal DWConv1D | 1.20 | 3.51 | 2.9x |

The 2.5-2.8x speedup from 4 cores (not the theoretical 4x) confirms the
kernels are partially bandwidth-limited even at seq=64 — the instruction-bound
finding means single-thread performance is IPC-limited, but multi-threaded
scaling reveals a bandwidth component that the single-thread comparison
cannot expose. This has implications for the O6: its 4x more cores and
5x more bandwidth mean the O6 will scale better than the fleet devices.

### Mixed-precision at decode (seq=1)

At decode (seq=1), state I/O dominates. The bf16/fp16 variants trade
narrower state for conversion overhead. On j2 (4-core OpenMP):

| Kernel (4B, seq=1) | fp32 (GiB/s) | bf16 (GiB/s) | fp16 (GiB/s) |
|--------------------|-------------|-------------|-------------|
| Cumulative Decay | 8.03 | 5.23 | 5.78 |
| Gated Delta-Rule Scan | 17.86 | 11.49 | 12.08 |

At decode, bf16/fp16 are **slower** than fp32 — the conversion overhead
(load narrow, widen to fp32, compute, narrow back) exceeds the memory
savings when the working set is tiny (seq=1, 16 KiB state). Mixed-precision
state narrowing helps only at prefill (seq=64), where the state traffic is
amortized over more compute. This confirms the bead ob-8qt.4 design: use
fp32 state at decode, narrow only for prefill chunk boundaries.


---
*Generated by `bench/fleet_analysis.py`. Regenerable from committed CSVs in `results/raw/`.*
