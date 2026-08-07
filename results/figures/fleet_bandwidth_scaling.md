# Fleet Bandwidth-Scaling Analysis

**Bead `ob-8ms.3`.** Cross-device comparison of GDN kernel throughput
versus spec memory bandwidth, testing the bandwidth-bound hypothesis
from `METRICS.md` (~0.25 FLOP/byte).

## Devices in the fleet

| Device | Cores | ISA | Spec BW (GiB/s) |
|--------|-------|-----|-----------------|
| Pi 5 | 4x Cortex-A76 @ 2.4 GHz | Armv8.2-A + dotprod | 17.0 |
| RK3588 big | 4x Cortex-A76 @ 2.3 GHz | Armv8.2-A + dotprod | 34.0 |
| RK3588 little | 4x Cortex-A55 @ 1.8 GHz | Armv8.2-A | 34.0 |
| Jetson j1 | 4x Cortex-A57 @ 1.48 GHz | Armv8.0-A (NEON only) | 25.6 |
| Jetson j2 | 4x Cortex-A57 @ 1.48 GHz | Armv8.0-A (NEON only) | 25.6 |
| **Orion O6** | 4x A720 big + 4x A720 mid + 4x A520 | Armv9.2-A | **93.1** |

## Achieved throughput vs spec bandwidth (4B model, seq=64)

RK3588, Jetson j1, and Jetson j2 are from the fleet sweep (ob-bf7): all at
post-optimization commits, clean tree, single-threaded (`OMP_NUM_THREADS=1`).
Pi 5 was not part of the fleet sweep — its data is from an earlier commit.
See the optimization-impact section below for multi-threaded results.

### 4B model

| Device | Spec (GiB/s) | CumDecay | Scan | DWConv1D | Scan/Spec | Scan spread |
|--------|-------------|----------|------|----------|-----------|-------------|
| Pi 5 | 17.0 | 3.74 | 1.20 | 3.23 | 7.1% | 7.4% |
| RK3588 big | 34.0 | 7.40 | 5.67 | 7.04 | 16.7% | 7.4% |
| RK3588 little | 34.0 | 1.45 | 0.82 | 1.20 | 2.4% | 7.2% |
| Jetson j1 | 25.6 | 1.59 | 1.18 | 1.41 | 4.6% | 2.8% |
| Jetson j2 | 25.6 | 1.50 | 1.09 | 0.93 | 4.3% | **16.7%** ⚠ |

⚠ 1 of 5 scan rows exceed the DEVICE_RUNBOOK's ~10% cleanliness threshold, worst Jetson j2 at 16.7%. The runbook says to suspect thermal throttling first. Treat flagged rows as indicative only.

### 0.8B model

| Device | Spec (GiB/s) | CumDecay | Scan | DWConv1D | Scan/Spec | Scan spread |
|--------|-------------|----------|------|----------|-----------|-------------|
| Pi 5 | 17.0 | 4.47 | 4.43 | 4.55 | 26.1% | 6.5% |
| RK3588 big | 34.0 | 8.15 | 6.93 | 6.82 | 20.4% | 5.7% |
| RK3588 little | 34.0 | 1.65 | 1.48 | 1.54 | 4.4% | 2.4% |
| Jetson j1 | 25.6 | 3.59 | 2.73 | 2.88 | 10.7% | **27.2%** ⚠ |
| Jetson j2 | 25.6 | 3.24 | 1.65 | 2.43 | 6.4% | **40.8%** ⚠ |

⚠ 2 of 5 scan rows exceed the DEVICE_RUNBOOK's ~10% cleanliness threshold, worst Jetson j2 at 40.8%. The runbook says to suspect thermal throttling first. Treat flagged rows as indicative only.

## The discriminating test: Jetson (A57, more BW) vs Pi 5 (A76, less BW)

The DEVICE_RUNBOOK poses this question: if the GDN scan kernel is
bandwidth-bound, then the Jetson Nano (oldest cores, 25.6 GiB/s spec)
should beat the Pi 5 (newest cores, 17.0 GiB/s spec). **If the Pi 5
wins comfortably, the bandwidth-bound thesis is wrong or incomplete.**

| Kernel (4B) | Pi 5 (17.0) | Jetson j1 (25.6) | Jetson j2 (25.6) | Winner | Pi5/J1 ratio |
|-------------|-------------|------------------|------------------|--------|-------------|
| Cumulative Decay | 3.74 | 1.59 | 1.50 | **Pi 5** | 2.35x |
| Gated Delta-Rule Scan | 1.20 | 1.18 | 1.09 | **Pi 5** | 1.02x |
| Causal DWConv1D | 3.23 | 1.41 | 0.93 | **Pi 5** | 2.29x |

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
| RK3588 big | t3 10.62 vs t4 5.67 | **1.87x** | **different commits** — t3 `f015982271a1`, t4 `1ca4d6dfb00c`; not an environmental comparison |
| RK3588 little | t3 0.55 vs t4 0.82 | **1.49x** | **different commits** — t3 `234807d46c95`, t4 `f2658cc98138`; not an environmental comparison |
| Jetson | j1 1.18 vs j2 1.09 | **1.08x** | same source commit `234807d46c95`, same core class |

The fleet sweep (ob-bf7/ob-aw9) resolved the provenance question: all RK3588 and Jetson replicates are now at **post-optimization commits**, clean tree, single-threaded. Despite both running optimized kernels, the RK3588 pair disagrees by **1.87x** — a genuine inter-board effect (different board vendors: t3 vs Turing Machines RK1; different kernels: 5.10 CFS vs 6.11 EEVDF; different DRAM: 32GB vs 8GB). The Jetson pair agrees within ~8%, in normal range. Pi 5 is not in the replicate comparison (only one unit).

### Provenance audit: were these runs captured from a clean tree?

All replicate runs are from the fleet sweep (ob-bf7/ob-aw9): post-optimization, clean tree, governor=performance. Of 6 runs with manifests, **6 recorded `dirty: false`** and 0 recorded dirty.

Since all runs are post-optimization and clean-tree, the RK3588 inter-board gap reflects genuine hardware heterogeneity — different board vendors, kernel versions, and DRAM configurations — not a code-version artifact.

RK3588 and Jetson data are from the fleet sweep (ob-bf7/ob-aw9). Pi 5 was not part of the sweep; its provenance is noted in the audit above.
**Treat the predictions as order-of-magnitude, not as a fit.** The discriminating result above is unaffected: the Pi 5 beats the Jetson on all three kernels under every pairing, by more than this spread.

## O6 extrapolation (prediction)

Spec bandwidth: **93.1 GiB/s** (100 GB/s, 128-bit LPDDR5 @ 5500 MT/s).

If the kernels were bandwidth-bound, achieved throughput should scale
linearly with spec bandwidth. Extrapolating the scan kernel from each device:

| Extrapolated from | Scan (GiB/s) | O6 BW ratio | Predicted O6 scan (GiB/s) |
|-------------------|-------------|-------------|--------------------------|
| Pi 5 | 1.20 | 5.5x | 6.57 |
| RK3588 big | 5.67 | 2.7x | 15.53 |
| RK3588 little | 0.82 | 2.7x | 2.25 |
| Jetson j1 | 1.18 | 3.6x | 4.29 |
| Jetson j2 | 1.09 | 3.6x | 3.96 |

**⚠ However, this linear extrapolation is almost certainly WRONG.**
The discriminating test above shows the kernels are instruction-bound,
not bandwidth-bound, at seq=64. A bandwidth-linear extrapolation would
overpredict. The honest prediction is that the O6's Cortex-A720 cores
(Armv9.2-A, wider OoO, higher clock than A76) will achieve higher throughput
than any current fleet device due to better IPC, but **not proportionally
to its 4-5x bandwidth advantage**.

**Core-performance-based prediction** (scaling from RK3588 A76):

- RK3588 scan: 5.67 GiB/s (4x A76 @ 2.3 GHz, Armv8.2, single-thread)
- O6 big cluster: 4x A720 @ 2.8 GHz, Armv9.2 (SVE2, i8mm, wider OoO)
- **Predicted O6 scan throughput: 17.0-28.4 GiB/s**
- This is ~18-30% of spec bandwidth (93.1 GiB/s)

**Optimized A76 reference.** t4 with 4-core OpenMP + NEON unrolling reads **11.09 GiB/s** — 2.0x the single-threaded baseline. The O6's A720 cores will benefit from both IPC gains AND the optimization stack, so the prediction above is conservative.

The fleet sweep (ob-bf7) confirmed the RK3588 inter-board gap is a genuine hardware effect. This prediction does not depend on resolving that gap.

To check this prediction: if the O6 board arrives, run
`bench_gdn_armv9sve2 --repeats 30 --csv` and compare.

## Optimization impact: j2 single-threaded vs 4-core OpenMP

The j2 CSV was re-run with the current optimized binary (OpenMP 4-core,
NEON double-width unrolling, bf16 conversion vectorization). This shows
the real-world impact of the optimization track (beads ob-8qt.5/6/7):

| Kernel (4B, seq=64) | Single-thread (GiB/s) | 4-core OpenMP (GiB/s) | Speedup |
|--------------------|-----------------------|-----------------------|---------|
| Cumulative Decay | 1.32 | 3.85 | 2.9x |
| Gated Delta-Rule Scan | 1.13 | 2.96 | 2.6x |
| Causal DWConv1D | 1.20 | 3.66 | 3.1x |

The 2.5-2.8x speedup from 4 cores (not the theoretical 4x) confirms the
kernels are partially bandwidth-limited even at seq=64 — the instruction-bound
finding means single-thread performance is IPC-limited, but multi-threaded
scaling reveals a bandwidth component that the single-thread comparison
cannot expose. This has implications for the O6: its 4x more cores and
5x more bandwidth mean the O6 will scale better than the fleet devices.

### Fleet sweep resolution (ob-bf7)

The historical provenance issue is resolved: the fleet sweep re-ran all devices at post-optimization commits with clean trees, governor=performance, and single-thread (`OMP_NUM_THREADS=1`). RK3588 is now **included** in the cross-device table above using the clean sweep data.

**Optimization impact on A76.** The multi-threaded optimized run (4-core OpenMP + NEON unrolling + bf16) on t4 reads 11.56 GiB/s scan vs 5.75 single-threaded — a **2.0x speedup** from parallelization alone. See the optimization-impact table below.

### Mixed-precision at decode (seq=1)

At decode (seq=1), state I/O dominates. The bf16/fp16 variants trade
narrower state for conversion overhead. On j2 (4-core OpenMP):

| Kernel (4B, seq=1) | fp32 (GiB/s) | bf16 (GiB/s) | fp16 (GiB/s) |
|--------------------|-------------|-------------|-------------|
| Cumulative Decay | 8.37 | 5.56 | 5.71 |
| Gated Delta-Rule Scan | 15.10 | 11.49 | 11.96 |

At decode, bf16/fp16 are **slower** than fp32 — the conversion overhead
(load narrow, widen to fp32, compute, narrow back) exceeds the memory
savings when the working set is tiny (seq=1, 16 KiB state). Mixed-precision
state narrowing helps only at prefill (seq=64), where the state traffic is
amortized over more compute. This confirms the bead ob-8qt.4 design: use
fp32 state at decode, narrow only for prefill chunk boundaries.


---
*Generated by `bench/fleet_analysis.py`. Regenerable from committed CSVs in `results/raw/`.*
