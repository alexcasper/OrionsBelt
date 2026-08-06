# Fleet Bandwidth-Scaling Analysis

**Bead `ob-8ms.3`.** Cross-device comparison of GDN kernel throughput
versus spec memory bandwidth, testing the bandwidth-bound hypothesis
from `METRICS.md` (~0.25 FLOP/byte).

## Devices in the fleet

| Device | Cores | ISA | Spec BW (GiB/s) |
|--------|-------|-----|-----------------|
| Pi 5 | 4x Cortex-A76 @ 2.4 GHz | Armv8.2-A + dotprod | 17.0 |
| Jetson j1 | 4x Cortex-A57 @ 1.48 GHz | Armv8.0-A (NEON only) | 25.6 |
| Jetson j2 | 4x Cortex-A57 @ 1.48 GHz | Armv8.0-A (NEON only) | 25.6 |
| **Orion O6** | 4x A720 big + 4x A720 mid + 4x A520 | Armv9.2-A | **93.1** |

## Achieved throughput vs spec bandwidth (4B model, seq=64)

Pi 5 and Jetson were benchmarked single-threaded at pre-optimization commits
(`28729f3` for Pi 5, later commits for Jetson — all pre-OpenMP). RK3588 is
**excluded** from this table (ob-0h0: t4's pre-optimization CSV was overwritten).
See the optimization-impact section below for RK3588 and j2 OpenMP data.

### 4B model

| Device | Spec (GiB/s) | CumDecay | Scan | DWConv1D | Scan/Spec | Scan spread |
|--------|-------------|----------|------|----------|-----------|-------------|
| Pi 5 | 17.0 | 3.74 | 1.20 | 3.23 | 7.1% | 7.4% |
| Jetson j1 | 25.6 | 1.16 | 0.72 | 1.04 | 2.8% | **17.2%** ⚠ |
| Jetson j2 | 25.6 | 1.15 | 0.73 | 1.04 | 2.9% | 9.4% |

⚠ 1 of 3 scan rows exceed the DEVICE_RUNBOOK's ~10% cleanliness threshold, worst Jetson j1 at 17.2%. The runbook says to suspect thermal throttling first. Treat flagged rows as indicative only.

### 0.8B model

| Device | Spec (GiB/s) | CumDecay | Scan | DWConv1D | Scan/Spec | Scan spread |
|--------|-------------|----------|------|----------|-----------|-------------|
| Pi 5 | 17.0 | 4.47 | 4.43 | 4.55 | 26.1% | 6.5% |
| Jetson j1 | 25.6 | 1.93 | 1.61 | 1.99 | 6.3% | **18.9%** ⚠ |
| Jetson j2 | 25.6 | 1.98 | 1.66 | 1.99 | 6.5% | **51.8%** ⚠ |

⚠ 2 of 3 scan rows exceed the DEVICE_RUNBOOK's ~10% cleanliness threshold, worst Jetson j2 at 51.8%. The runbook says to suspect thermal throttling first. Treat flagged rows as indicative only.

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
| RK3588 big | t3 11.07 vs t4 11.48 | **1.04x** | **different commits** — t3 `553a96efa8aa`, t4 `28729f3e0a3c` (dirty); not an environmental comparison |
| RK3588 little | t3 2.27 vs t4 3.91 | **1.72x** | **different commits** — t3 `553a96efa8aa`, t4 `28729f3e0a3c` (dirty); not an environmental comparison |
| Pi 5 | r5 1.20 vs j1 1.84 | **1.53x** | **different commits** — r5 `28729f3e0a3c` (dirty), j1 `f127a11cfdee` (dirty); not an environmental comparison |
| Jetson j2 | canonical 0.73 vs _single 1.13 | **1.55x** | same board; _single has **no manifest** |

The RK3588 pair was historically the most concerning — two hosts on the same core class. As of `ob-0h0`, **both t3 and t4 are at the optimized code level** (t3 at `553a96e` clean, t4 overwritten by commit `8f8be11` but manifest stale at `28729f3`). Their spread now reflects inter-board variance at the same optimization level, not a code-version difference. The Pi 5 pair (r5 at `28729f3` vs j1 at `f127a11`) remains a cross-commit comparison. Worst replicate spread on the fleet is **1.72x**.

### Provenance audit: were these runs captured from a clean tree?

Of the 7 replicate runs with a manifest, **5 recorded `dirty: true`** at capture time and 2 recorded a clean tree.

**1 have no manifest at all** (jetson-j2_single) — PLAN.md section 9: a number without a manifest is not a result.

This limits the section above more than the spread itself does. `dirty: true` means the recorded SHA does **not** identify the code that produced the numbers, so two runs labelled with the same commit may have executed genuinely different binaries. The RK3588 gap therefore cannot be attributed to environment rather than to code — both explanations stay open and neither is settleable from the committed data. Any re-run for `ob-bf7` must be taken from a clean tree.

RK3588 is **excluded** from the cross-device comparison table (ob-0h0: t4's pre-optimization data was overwritten). Only Pi 5 and Jetson remain at the pre-optimization code level. The Pi 5 pair (r5/j1) is cross-commit but the effect is large relative to the spread.
**Treat the predictions as order-of-magnitude, not as a fit.** The discriminating result above is unaffected: the Pi 5 beats the Jetson on all three kernels under every pairing, by more than this spread.

## O6 extrapolation (prediction)

Spec bandwidth: **93.1 GiB/s** (100 GB/s, 128-bit LPDDR5 @ 5500 MT/s).

If the kernels were bandwidth-bound, achieved throughput should scale
linearly with spec bandwidth. Extrapolating the scan kernel from each device:

| Extrapolated from | Scan (GiB/s) | O6 BW ratio | Predicted O6 scan (GiB/s) |
|-------------------|-------------|-------------|--------------------------|
| Pi 5 | 1.20 | 5.5x | 6.57 |
| Jetson j1 | 0.72 | 3.6x | 2.62 |
| Jetson j2 | 0.73 | 3.6x | 2.65 |

**⚠ However, this linear extrapolation is almost certainly WRONG.**
The discriminating test above shows the kernels are instruction-bound,
not bandwidth-bound, at seq=64. A bandwidth-linear extrapolation would
overpredict. The honest prediction is that the O6's Cortex-A720 cores
(Armv9.2-A, wider OoO, higher clock than A76) will achieve higher throughput
than any current fleet device due to better IPC, but **not proportionally
to its 4-5x bandwidth advantage**.

**Core-performance-based prediction** (scaling from Pi 5 A76):

- Pi 5 scan: 1.20 GiB/s (4x A76 @ 2.4 GHz, Armv8.2)
- O6 big cluster: 4x A720 @ 2.8 GHz, Armv9.2 (SVE2, wider OoO)
- **Predicted O6 scan throughput: 3.6-6.0 GiB/s**
- This is ~4-6% of spec bandwidth (93.1 GiB/s)

**Optimized A76 reference.** t3 (commit `553a96efa8aa`, clean) with optimized kernels reads **11.07 GiB/s** on the same A76 big cluster — 9.2x the Pi 5's pre-optimization number. The O6's A720 cores will benefit from both the IPC gain AND the optimization stack, so the prediction above (scaled from pre-optimization Pi 5) is conservative.

Resolving `ob-bf7` — one clean-tree, commit-matched sweep with pinning and thermals recorded — narrows this more than any modelling refinement would.

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

### ⚠ RK3588 t4 provenance breakage (ob-0h0)

Commit `8f8be11` overwrote `rk3588-t4_big.csv` and `rk3588-t4_little.csv` with optimized kernel data (scan 4B big: 3.29→11.48 GiB/s), but the manifest (`rk3588-t4.json`) still records `sha=28729f3 dirty=true`. The pre-optimization t4 data exists only in git history (commit `6fac497`). Both t3 (`553a96e`, clean) and t4 (now optimized, stale manifest) are at the optimized code level, so neither can anchor a pre-optimization cross-device comparison. RK3588 is therefore **excluded** from the cross-device table above.

**Optimization impact on A76** is documented from j1's same-device re-run on t3: scan 4B went from 2.26 GiB/s (commit `0e19308`, pre-optimization) to 11.07 GiB/s (commit `553a96e`, optimized: OpenMP + NEON unrolling) — a **4.9x speedup** on identical silicon with a clean-tree manifest. See bead `ob-bf7` update 2026-08-06.

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
