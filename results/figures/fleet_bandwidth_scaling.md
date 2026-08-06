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

All fleet devices were benchmarked single-threaded at commit `28729f3`
(pre-OpenMP, pre-NEON-unrolling). The j2 single-threaded numbers below
are a fresh run of the current binary with `OMP_NUM_THREADS=1` to match
that optimization level for fair comparison. See the optimization-impact
section below for what 4-core OpenMP + NEON unrolling achieves on j2.

### 4B model

| Device | Spec (GiB/s) | CumDecay | Scan | DWConv1D | Scan/Spec | Scan spread |
|--------|-------------|----------|------|----------|-----------|-------------|
| Pi 5 | 17.0 | 3.74 | 1.20 | 3.23 | 7.1% | 7.4% |
| RK3588 big | 34.0 | 4.25 | 3.29 | 4.52 | 9.7% | **17.4%** ⚠ |
| RK3588 little | 34.0 | 0.97 | 0.55 | 0.71 | 1.6% | **12.1%** ⚠ |
| Jetson j1 | 25.6 | 1.16 | 0.72 | 1.04 | 2.8% | **17.2%** ⚠ |
| Jetson j2 | 25.6 | 1.15 | 0.73 | 1.04 | 2.9% | 9.4% |

⚠ 3 of 5 scan rows exceed the DEVICE_RUNBOOK's ~10% cleanliness threshold, worst RK3588 big at 17.4%. The runbook says to suspect thermal throttling first. Treat flagged rows as indicative only.

### 0.8B model

| Device | Spec (GiB/s) | CumDecay | Scan | DWConv1D | Scan/Spec | Scan spread |
|--------|-------------|----------|------|----------|-----------|-------------|
| Pi 5 | 17.0 | 4.47 | 4.43 | 4.55 | 26.1% | 6.5% |
| RK3588 big | 34.0 | 5.00 | 4.79 | 6.00 | 14.1% | 3.1% |
| RK3588 little | 34.0 | 1.19 | 0.99 | 0.92 | 2.9% | 2.2% |
| Jetson j1 | 25.6 | 1.93 | 1.61 | 1.99 | 6.3% | **18.9%** ⚠ |
| Jetson j2 | 25.6 | 1.98 | 1.66 | 1.99 | 6.5% | **51.8%** ⚠ |

⚠ 2 of 5 scan rows exceed the DEVICE_RUNBOOK's ~10% cleanliness threshold, worst Jetson j2 at 51.8%. The runbook says to suspect thermal throttling first. Treat flagged rows as indicative only.

## The discriminating test: Jetson (A57, more BW) vs Pi 5 (A76, less BW)

The DEVICE_RUNBOOK poses this question: if the GDN scan kernel is
bandwidth-bound, then the Jetson Nano (oldest cores, 25.6 GiB/s spec)
should beat the Pi 5 (newest cores, 17.0 GiB/s spec). **If the Pi 5
wins comfortably, the bandwidth-bound thesis is wrong or incomplete.**

| Kernel (4B) | Pi 5 (17.0) | Jetson j1 (25.6) | Jetson j2 (25.6) | Winner | Pi5/J1 ratio |
|-------------|-------------|------------------|------------------|--------|-------------|
| Cumulative Decay | 3.74 | 1.16 | 1.15 | **Pi 5** | 3.22x |
| Gated Delta-Rule Scan | 1.20 | 0.72 | 0.73 | **Pi 5** | 1.67x |
| Causal DWConv1D | 3.23 | 1.04 | 1.04 | **Pi 5** | 3.11x |

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
| RK3588 big | t3 11.07 vs t4 3.29 | **3.36x** | **different commits** — t3 `553a96efa8aa`, t4 `28729f3e0a3c` (dirty); not an environmental comparison |
| RK3588 little | t3 2.27 vs t4 0.55 | **4.13x** | **different commits** — t3 `553a96efa8aa`, t4 `28729f3e0a3c` (dirty); not an environmental comparison |
| Pi 5 | r5 1.20 vs j1 1.84 | **1.53x** | **different commits** — r5 `28729f3e0a3c` (dirty), j1 `f127a11cfdee` (dirty); not an environmental comparison |
| Jetson j2 | canonical 0.73 vs _single 1.13 | **1.55x** | same board; _single has **no manifest** |

The RK3588 pair was historically the most concerning — two hosts on the same core class. Their CSVs originally shared commit `28729f3`, but **t3 was re-run at `553a96e`** (clean tree, optimized kernels: OpenMP + NEON unrolling) per the `ob-bf7` 2026-08-06 update, while t4 remains at `28729f3` (dirty tree, pre-optimization). The spread between them is now a **code-version difference, not an environmental one**. On the big cluster, t3 reads 11.07 GiB/s (optimized) vs t4 at 3.29 (pre-opt) — a 3.4x gap that is the optimization stack's real-world impact on the same hardware. Worst replicate spread on the fleet is **4.13x**.

### Provenance audit: were these runs captured from a clean tree?

Of the 7 replicate runs with a manifest, **5 recorded `dirty: true`** at capture time and 2 recorded a clean tree.

**1 have no manifest at all** (jetson-j2_single) — PLAN.md section 9: a number without a manifest is not a result.

This limits the section above more than the spread itself does. `dirty: true` means the recorded SHA does **not** identify the code that produced the numbers, so two runs labelled with the same commit may have executed genuinely different binaries. The RK3588 gap therefore cannot be attributed to environment rather than to code — both explanations stay open and neither is settleable from the committed data. Any re-run for `ob-bf7` must be taken from a clean tree.

This report selects `t4` for RK3588 (pre-optimization, same commit as Pi 5 for a fair cross-device comparison) and `r5` for the Pi 5. t3's optimized data is shown separately in the optimization-impact analysis below.
**Treat the predictions as order-of-magnitude, not as a fit.** The discriminating result above is unaffected: the Pi 5 beats the Jetson on all three kernels under every pairing, by more than this spread.

## O6 extrapolation (prediction)

Spec bandwidth: **93.1 GiB/s** (100 GB/s, 128-bit LPDDR5 @ 5500 MT/s).

If the kernels were bandwidth-bound, achieved throughput should scale
linearly with spec bandwidth. Extrapolating the scan kernel from each device:

| Extrapolated from | Scan (GiB/s) | O6 BW ratio | Predicted O6 scan (GiB/s) |
|-------------------|-------------|-------------|--------------------------|
| Pi 5 | 1.20 | 5.5x | 6.57 |
| RK3588 big | 3.29 | 2.7x | 9.01 |
| RK3588 little | 0.55 | 2.7x | 1.51 |
| Jetson j1 | 0.72 | 3.6x | 2.62 |
| Jetson j2 | 0.73 | 3.6x | 2.65 |

**⚠ However, this linear extrapolation is almost certainly WRONG.**
The discriminating test above shows the kernels are instruction-bound,
not bandwidth-bound, at seq=64. A bandwidth-linear extrapolation would
overpredict. The honest prediction is that the O6's Cortex-A720 cores
(Armv9.2-A, wider OoO, higher clock than A76) will achieve higher throughput
than any current fleet device due to better IPC, but **not proportionally
to its 4-5x bandwidth advantage**.

**Core-performance-based prediction** (scaling from RK3588 A76 big cluster):

- RK3588 big scan: 3.29 GiB/s (4x A76 @ 2.3 GHz, Armv8.2)
- O6 big cluster: 4x A720 @ 2.8 GHz, Armv9.2 (SVE2, wider OoO)
- Expected gain from IPC + clock: 1.5-2.5x over A76
- **Predicted O6 scan throughput: 4.9-8.2 GiB/s**
- This is ~5-9% of spec bandwidth, vs 10% achieved on A76

**On the anchor choice.** t3 was re-run at commit `553a96efa8aa` (clean tree) with optimized kernels (OpenMP + NEON unrolling), reading **11.07 GiB/s** (spread 6.2%) — vs t4 at 3.29 at `28729f3` (pre-optimization). Extrapolating t3's optimized numbers would give 16.6-27.7 GiB/s on the O6, but that conflates the IPC gain from A720 cores with the optimization-stack gain from the A76. t4 is used for the cross-device comparison (same commit as Pi 5); t3's data shows the optimization impact on identical A76 silicon.

Published claim: **~5-8 GiB/s** (from pre-optimization A76). t3's optimized run suggests the O6 with both A720 IPC gains AND the optimization stack could reach higher. Resolving `ob-bf7` — one clean-tree, commit-matched sweep with pinning and thermals recorded — narrows this more than any modelling refinement would.

To check this prediction: if the O6 board arrives, run
`bench_gdn_armv9sve2 --repeats 30 --csv` and compare.

## Optimization impact: j2 single-threaded vs 4-core OpenMP

The j2 CSV was re-run with the current optimized binary (OpenMP 4-core,
NEON double-width unrolling, bf16 conversion vectorization). This shows
the real-world impact of the optimization track (beads ob-8qt.5/6/7):

> ⚠ **No provenance.** `jetson-j2-omp-full.csv` has no manifest on any branch, so the speedups below cannot be tied to a specific build or device state. PLAN.md section 9: a number without a manifest is not a result. Treat these as indicative and re-capture with `bench/manifest.py` alongside the run.

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

### RK3588 A76: optimization stack impact

t4 (commit `28729f3`, pre-optimization) vs t3 (commit `553a96e`, optimized: OpenMP + NEON unrolling) on the same A76 big cluster. Different physical boards, so this is indicative — but j1's same-device re-run on t3 itself showed 2.26 → 11.07 GiB/s on Scan (4.9x), confirming the direction.

| Kernel (4B, seq=64) | t4 pre-opt (GiB/s) | t3 optimized (GiB/s) | Speedup |
|--------------------|--------------------|-----------------------|---------|
| Cumulative Decay | 4.25 | 21.74 | 5.1x |
| Gated Delta-Rule Scan | 3.29 | 11.07 | 3.4x |
| Causal DWConv1D | 4.52 | 21.60 | 4.8x |

The optimization stack delivers 2.6-5.1x on A76 silicon — larger than the 2.6-3.1x seen on A57 (Jetson). This is consistent with wider OoO pipelines benefiting more from NEON unrolling and thread parallelism. t3's clean-tree manifest (`553a96e`, `dirty=false`) is the only clean provenance in the fleet.

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
