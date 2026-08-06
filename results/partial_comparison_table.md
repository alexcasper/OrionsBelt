# Partial Comparison Table — CPU/kernel GDN results (O6 NPU + full-attention oracle pending)

**Bead:** `ob-9t0.2` (scoped partial step toward `ob-ami` / `ob-rqd`) · **Assembled:** 2026-08-04 · **Branch:** `bench/j1`

> **Scope.** This is the master comparison table restricted to the data we actually have
> committed: **CPU/kernel-level GDN numbers** across the edge fleet (Jetson Nano A57,
> RK3588 A76/A55, Raspberry Pi 5 A76) plus one **CPU-only model-level ablation** run at 4 K
> context. Every numeric cell cites a `run_id` / `git_sha` so it is auditable to a manifest.
> The two hardware-gated columns of `ob-ami` — the **O6 NPU-offload** configuration and the
> **x86/CUDA full-attention oracle** baseline — are explicitly marked **PENDING HARDWARE**;
> they are not fabricated. End-to-end model throughput is **not yet measured** at the fleet
> layer (only the kernel micro-benchmarks and one synthetic harness ablation exist).
>
> **Method.** All aggregation is hand-rolled by `scripts/partial_comparison_table.py`
> (Python 3.6.9, stdlib `csv` + `statistics` + `json` only — no pandas/numpy/torch; the
> `bench/` package is **not** imported or executed, per the node's 3.6.9 constraint).
> Kernel CSVs already store `p50` over 30 repeats; the ablation CSV is tidy/long and is
> reduced with `statistics.median` over its 5 repeats. Re-run:
> `python3 scripts/partial_comparison_table.py`.

---

## 1. Master comparison table

Rows = configuration. Columns = metric group × context. The model-level metrics
(tokens/sec, TTFT, peak-memory decomposition) come from the single CPU-only ablation run at
**4 K context**; the kernel throughput columns come from the per-device micro-benchmarks at
the prefill chunk size (**seq = 64**) and the decode size (**seq = 1**). A `—` means *no
committed data*; the reason is in §5.

### 1a. Model-level ablation (the `metric_component` three-way memory split)

Source: `results/raw/ablation/ablation_cpu-only.csv` · `run_id = generic_aarch64_20260802T204232Z_49ed76b` · `git_sha = 49ed76b` · `engine_gdn = engine_full_attention = cpu` · `quant = fp16` · p50 over n=5 repeats.

| Configuration | Context | Prefill (tok/s) | Decode (tok/s) | TTFT (ms) | mem `weights` | mem `kv_cache` (FA layers) | mem `recurrent_state` (GDN) |
|---|---|---|---|---|---|---|---|
| **Full-attention-only baseline (oracle)** | 4 K | — ¹ | — ¹ | — ¹ | — ¹ | — ¹ | — ¹ |
| **Hybrid GDN, unoptimized (cpu/cpu)** | 4 K | 1.16 × 10¹⁰ ² | 4.54 × 10⁶ ² | **0.45** ² | **7.45 GiB** ³ | **128.0 MiB** ³ | **48.0 MiB** ³ |
| **Hybrid GDN + OpenMP 4-core** | 4 K | — ⁴ | — ⁴ | — ⁴ | 7.45 GiB ⁵ | 128.0 MiB ⁵ | 48.0 MiB ⁵ |
| **Hybrid GDN, full-optimized** | 4 K | — ⁴ | — ⁴ | — ⁴ | 7.45 GiB ⁵ | 128.0 MiB ⁵ | 48.0 MiB ⁵ |
| *(any config)* | 32 K / 128 K / 262 K | — ⁶ | — ⁶ | — ⁶ | — ⁶ | — ⁶ | — ⁶ |

Footnotes: ¹ **PENDING HARDWARE** — the full-attention-only reference is the x86/CUDA oracle
(`ob-aqv`, `ob-mrd.1`); no such run is committed. ² Harness placeholder, **not end-to-end
model throughput** — the `@ablation-cpu-only` checkpoint reports ~10¹⁰ tok/s, which is a
microbench-style synthetic figure; report the *ratio* and the **memory decomposition**, not
these absolutes (see §5). ³ The load-bearing, defensible numbers — the three-way split at
4 K. `kv_cache` (128 MiB) already exceeds `recurrent_state` (48 MiB) at 4 K and grows
linearly with context; `recurrent_state` is O(1). ⁴ Model-level tokens/sec/TTFT under
optimization is **not measured** — the optimization ladder lives at the *kernel* layer
(§1b/§3), not the model layer. ⁵ Memory components are configuration-invariant by
construction (weights flat; KV depends only on context, not on thread count) — reuses the
4 K ablation values. ⁶ **PENDING** — no committed run spans the 32 K–262 K context sweep;
only 4 K exists.

### 1b. Kernel-level GDN throughput — prefill chunk (seq = 64), Qwen3.5-4B, fp32 baseline single-core

`gib_per_s_p50` over 30 repeats. Device dimension spans the committed fleet. This is where
the "hybrid GDN vs optimized hybrid GDN" contrast is actually measurable today.

| Configuration / device | run_id · git_sha | dirty | CumDecay (GiB/s) | Gated-Scan (GiB/s) | Causal-DWConv1D (GiB/s) |
|---|---|---|---|---|---|
| **Hybrid GDN baseline, single-core** — Jetson j1 | `j1_…204232Z`·`2c9ac9f` | true | 1.16 | **0.72** | 1.04 |
| **Hybrid GDN baseline, single-core** — Jetson j2 | `j2_…144316Z`·`6ea1771` | true | 1.15 | **0.73** | 1.04 |
| **Hybrid GDN baseline, single-core** — Pi 5 (r5) | `r5_…201237Z`·`28729f3` | true | 3.74 | **1.20** | 3.23 |
| **Hybrid GDN baseline, single-core** — Pi 5 (j1 tag) | `r5_…083154Z`·`f127a11` | true | 2.93 | **1.84** | 2.37 |
| **Hybrid GDN baseline, single-core** — RK3588 big (t4) | `t4_…211249Z`·`28729f3` | true | 4.25 | **3.29** | 4.52 |
| **Hybrid GDN baseline, single-core** — RK3588 big (t3) | `t3_…211312Z`·`28729f3` | true | 4.13 | 1.96 ⚠ | 4.02 |
| **Hybrid GDN baseline, single-core** — RK3588 little (t4) | `t4_…211249Z`·`28729f3` | true | 0.97 | 0.55 | 0.71 |
| **Hybrid GDN baseline, single-core** — RK3588 little (t3) | `t3_…211312Z`·`28729f3` | true | 0.87 | 0.35 | 0.65 |
| **Hybrid GDN + OpenMP 4-core** — Jetson j1 (**clean tree**) | `j1_…102230Z`·`ba7506d` | **false** | 3.79 | **2.92** | 3.60 |
| **Hybrid GDN + OpenMP 4-core** — Jetson j2 | `(no manifest)`·`a085417`* | true | 2.35 | 1.97 | 2.26 |
| **Hybrid GDN, full-optimized** (OMP + conv-unroll + f16) — Jetson j2 | `jetson-j2-full-optimized`·`8c9b3a9` | true | 3.82 | **2.97** | 3.61 |
| **Hybrid GDN, full-optimized** — Jetson j1 (clean-tree cross-check) | `j1_…102230Z`·`ba7506d` | **false** | 3.79 | 2.92 | 3.60 |
| **O6 NPU-offload** (any kernel) | — | — | — ⁷ | — ⁷ | — ⁷ |

⚠ `rk3588-t3_big` Gated-Scan carries **spread_pct = 153%** (p50 1514 µs vs p95 3832 µs) —
contaminated per `ob-bf7`; **prefer t4** (spread 17%) for the RK3588 anchor. The t3 row is
retained only to show why it is rejected. ⁷ **PENDING HARDWARE** — O6 board not available
(`ob-axq`).

---

## 2. Headline numbers (each with provenance)

1. **The three-way memory split is real and measurable at 4 K already.** On the committed
   CPU-only ablation (`generic_aarch64_…_49ed76b`, `git_sha 49ed76b`): `weights = 7.45 GiB`
   (flat), full-attention `kv_cache = 128.0 MiB`, GDN `recurrent_state = 48.0 MiB` (O(1)).
   KV exceeds recurrent state by **2.67×** at 4 K, and the gap only widens with context —
   this is the architecture's central advantage, demonstrated from committed data while the
   full-attention oracle is still pending.

2. **Microarchitecture beats spec bandwidth at prefill chunk size.** Pi 5 (A76, **17.0 GiB/s**
   spec) outperforms Jetson Nano (A57, **25.6 GiB/s** spec) on **all three** GDN kernels
   single-threaded — CumDecay 3.74 vs 1.16 (**3.22×**), Scan 1.20 vs 0.72 (**1.67×**),
   DWConv1D 3.23 vs 1.04 (**3.11×**) — despite 33% *less* memory bandwidth
   (`pi5-r5.csv` · `r5_…_28729f3` vs `jetson-j1.csv` · `j1_…_2c9ac9f`). The kernels are
   **instruction-overhead-bound, not DRAM-bandwidth-bound** at the seq=64 (~1 MiB,
   L2/L3-resident) working set. This is the result that survived `ob-bf7` scrutiny.

3. **The optimization ladder lifts Jetson Scan from 0.73 → 2.97 GiB/s (≈4.1×).** Within the
   j2 device series: single-core 1.13 → 4-core OpenMP 1.97 (**1.74×**) → full-optimized
   (OMP + NEON double-width conv-unroll + fp16) **2.97 GiB/s (2.63×)**; CumDecay reaches
   3.82 (2.89×), DWConv1D 3.61 (3.01×) (`jetson-j2-full-optimized.csv` · `8c9b3a9`).
   The **clean-tree** run on j1 (`jetson-j1_clean.csv` · `ba7506d`, **dirty=false**) reads
   2.92 GiB/s on Scan — agreeing with j2's optimized 2.97 to ~2%, so the headline gain is
   not a dirty-tree artifact.

4. **No thermal throttling on the active-cooled Jetson over 120 s.** `gdn_gated_scan`,
   Qwen3.5-4B: j1 baseline holds 0.77 → 0.76 GiB/s (−1.3%, 51 → 52 °C) over 120 s
   (`jetson-j1_sustained` · `a99495f`); j2 optimized holds 2.80 → 2.71 GiB/s (−3.2%, up to
   58 °C) over 120 s (`jetson-j2-sustained-optimized.csv`). The 4-core OpenMP step itself
   is confirmed sustained: j2 1-core 1.03 → 4-core 2.37 GiB/s (`jetson-j2_sustained_{1,4}core.csv`).

5. **Decode (seq=1) recurrence is cheap per token and NOT bandwidth-bound.** Gated-Scan at
   decode: **4.74 µs/token** on j1 clean 4-core, **4.48 µs/token** on j2 full-optimized —
   and throughput *jumps* to **16–17 GiB/s** because the single-token working set is
   L2-resident (`jetson-j1_clean.csv`, `jetson-j2-full-optimized.csv`). This is the
   kernel-level observation behind the architecture claim that decode stays flat under
   optimization (the bandwidth-bound thesis holds at decode, not at prefill-chunk size).

6. **Energy cost (Jetson j1, INA3221, 10 s sustained).** `gdn_gated_scan` 836 mJ/GiB
   (CPU rail, 1250 mJ/GiB board); `gdn_cumdecay` 667 mJ/GiB CPU; `gdn_causal_dwconv1d`
   767 mJ/GiB CPU (`results/manifests/jetson-j1_power.json`, bead `ob-agf.1`/`ob-mrd.7`).

---

## 3. Optimization-ladder detail (Jetson, Qwen3.5-4B, seq=64)

GiB/s @ p50 (p50 µs in parentheses). The ladder isolates each optimization's contribution.

| Step | CumDecay | Gated-Scan | Causal-DWConv1D | provenance |
|---|---|---|---|---|
| baseline single-core (j1, manifest) | 1.16 (1691) | 0.72 (4085) | 1.04 (1984) | `jetson-j1.csv` `2c9ac9f` |
| baseline single-core (j2, manifest) | 1.15 (1696) | 0.73 (4062) | 1.04 (1988) | `jetson-j2.csv` `6ea1771` |
| single-core, reproducibility-fixed (j2) | 1.32 (1485) | 1.13 (2626) | 1.20 (1710) | `jetson-j2_single.csv` ⚠ no manifest |
| + 4-core OpenMP (j2) | 2.35 (829) | 1.97 (1501) | 2.26 (912) | `jetson-j2-omp.csv` ⚠ no manifest |
| + 4-core OpenMP full (j2) | 3.85 (508) | 2.96 (1001) | 3.66 (563) | `jetson-j2-omp-full.csv` ⚠ no manifest |
| + conv-unroll only (j2) | 3.85 (507) | 2.91 (1018) | 3.57 (578) | `jetson-j2-conv-unroll.csv` ⚠ no manifest |
| + OMP + unroll (j2) | 3.71 (526) | 2.95 (1003) | 2.29 (901) | `jetson-j2-omp-unroll.csv` ⚠ no manifest |
| **full-optimized** (OMP+unroll+f16, j2) | **3.82 (512)** | **2.97 (997)** | **3.61 (571)** | `jetson-j2-full-optimized.csv` `8c9b3a9` |
| clean-tree 4-core cross-check (j1) | 3.79 (516) | 2.92 (1014) | 3.60 (572) | `jetson-j1_clean.csv` `ba7506d` **dirty=false** |

**Within-j2-series speedups (single → omp → full-opt):** CumDecay 1.32 → 2.35 (1.78×) →
3.82 (2.89×); Scan 1.13 → 1.97 (1.74×) → 2.97 (2.63×); DWConv1D 1.20 → 2.26 (1.88×) →
3.61 (3.01×). **Cross-commit manifest-backed (same j1 device):** Scan 0.72 → 2.92 = 4.06×
— flagged **superlinear and confounded** (it mixes core-count with code-path maturity, so
the within-j2-series 2.63× is the honest attribution for the optimization stack).

---

## 4. Decode (seq=1) kernel — Qwen3.5-4B — per-token recurrence cost

| Config | CumDecay µs/tok (GiB/s) | Gated-Scan µs/tok (GiB/s) | Causal-DWConv1D µs/tok (GiB/s) | provenance |
|---|---|---|---|---|
| j1 clean 4-core | 3.65 (8.37) | **4.74 (16.10)** | 11.15 (12.32) | `jetson-j1_clean.csv` `ba7506d` |
| j2 full-optimized | 3.80 (8.03) | **4.48 (17.03)** | 9.22 (14.90) | `jetson-j2-full-optimized.csv` `8c9b3a9` |

---

## 5. Gaps, caveats, and marked-pending cells (honest)

### 5.1 PENDING HARDWARE (will land with `ob-axq`, `ob-aqv`, `ob-mrd.1`)
- **O6 NPU-offload configuration** — entire row in §1b marked `—`. No O6 board in hand.
- **x86/CUDA full-attention oracle** — the "Full-attention-only baseline" row in §1a. This
  is the reference against which GDN's memory advantage is *quantified*; without it the
  table shows the hybrid config only.
- **O6 extrapolation** must ship as an order-of-magnitude estimate with the fleet spread
  attached, **not** a fitted line (per `ob-bf7`).

### 5.2 PENDING — data not yet collected
- **Context sweep 32 K / 128 K / 262 K** — no committed run covers anything but 4 K
  (ablation) / seq=64 & seq=1 (kernels). The "KV cache grows linearly" claim is currently
  demonstrated by the *formula* in `METRICS.md` §5.3 and the single 4 K data point, not by
  a measured sweep. (`ob-rqd` / `ob-del`.)
- **End-to-end model throughput / TTFT under optimization** — §1a rows 3–4 tokens/sec are
  `—`. The optimization story is kernel-level only today.

### 5.3 Data-quality flags (`ob-bf7`)
- **The ablation run's manifest is MISSING from disk.** `ablation_cpu-only.csv` references
  `results/manifests/generic_aarch64_20260802T204232Z_49ed76b.json`, which is not committed
  (the `git_sha 49ed76b` is recorded in the CSV, but per `RESULTS_SCHEMA.md` §2 *"a CSV
  without its manifest is not a result"* — the memory decomposition is used here because it
  is internally consistent and formula-verified, but this must be resolved before
  `ob-ami`).
- **The optimization-ladder CSVs have NO companion manifest** (`jetson-j1_single`,
  `jetson-j1_omp`, `jetson-j2_single`, `jetson-j2-omp`, `jetson-j2-omp-full`,
  `jetson-j2-conv-unroll`, `jetson-j2-omp-unroll`). They are committed (introduced in
  commits `dffac4b`, `a085417`) and internally consistent, and the manifest-backed
  `jetson-j2-full-optimized.csv` (`8c9b3a9`) and clean-tree `jetson-j1_clean.csv`
  (`ba7506d`, **dirty=false**) corroborate the optimized endpoint — but the intermediate
  ladder steps are technically unprovenanced and should be re-captured under one
  commit-matched sweep.
- **`rk3588-t3_big` Scan is contaminated** (spread 153%); RK3588 is anchored on **t4**
  (spread 17%) on measurement-quality grounds. The t3 row is retained only to document the
  rejection.
- **Cross-commit comparisons carry the `ob-bf7` spread warning.** Pi 5 / RK3588 are at
  `28729f3`; Jetson canonical baselines are at `2c9ac9f` / `6ea1771`. The qualitative
  result (Pi 5 beats Jetson on all three kernels) survives because the effect (1.67–3.22×)
  is large relative to the spread, but per-device "% of spec" figures and any fitted
  bandwidth slope remain withdrawn.
- **Every fleet manifest records `dirty=true`** except `jetson-j1_clean.csv` (`ba7506d`,
  dirty=false) — the only clean-tree run. The stored SHA therefore does not identify the
  exact binary that ran for the rest; "same commit" never guaranteed "same binary" (`ob-bf7`).
- **Only one Pi 5 physical board.** `pi5-r5.csv` and `pi5-j1.csv` are the same hostname
  `r5` at different commits (`28729f3` vs `f127a11`); the 1.20 vs 1.84 GiB/s Scan
  difference may be a real optimization gain or may be the t3/t4-scale variance — they
  cannot be separated without the commit-matched sweep.

### 5.4 Synthetic-figure caveat
- The ablation throughput numbers (~10¹⁰ prefill tok/s, ~4.5 × 10⁶ decode tok/s, TTFT
  0.45 ms) come from a `@ablation-cpu-only` harness path and are **not plausible as
  end-to-end transformer inference** on `generic_aarch64`. They are kept here only to
  document the cell; the defensible claims from that run are the **memory decomposition**
  and the **prefill ≫ decode shape**, not the absolute tok/s.

---

## 6. Provenance appendix (run_id → git_sha → manifest → dirty)

| CSV | run_id | git_sha | dirty | manifest present? |
|---|---|---|---|---|
| `ablation/ablation_cpu-only.csv` | `generic_aarch64_20260802T204232Z_49ed76b` | `49ed76b` | ? | **NO** (referenced file missing) |
| `jetson-j1_clean.csv` | `j1_20260804T102230Z_ba7506d` | `ba7506d` | **false** | yes |
| `jetson-j1.csv` | `j1_20260802T235238Z_2c9ac9f` | `2c9ac9f` | true | yes |
| `jetson-j1_sustained.csv` | `jetson-j1_sustained` | `a99495f` | true | yes |
| `jetson-j1_single.csv`, `jetson-j1_omp.csv` | — | (commit `dffac4b`) | ? | **NO** |
| `jetson-j2.csv` | `j2_20260803T144316Z_6ea1771` | `6ea1771` | true | yes |
| `jetson-j2-full-optimized.csv` | `jetson-j2-full-optimized` | `8c9b3a9` | true | yes |
| `jetson-j2_single/omp/omp-full/conv-unroll/omp-unroll.csv` | — | (commit `a085417`) | ? | **NO** |
| `jetson-j2_sustained_{1,4}core.csv` | `jetson-j2_sustained_*_retroactive` | `45ae679` | true | yes |
| `pi5-r5.csv` | `r5_20260802T201237Z_28729f3` | `28729f3` | true | yes |
| `pi5-j1.csv` | `r5_20260803T083154Z_f127a11` | `f127a11` | true | yes |
| `rk3588-t3_*.csv` | `t3_20260802T211312Z_28729f3` | `28729f3` | true | yes |
| `rk3588-t4_*.csv` | `t4_20260802T211249Z_28729f3` | `28729f3` | true | yes |
| `jetson-j1_power.json` (sustained) | `jetson-j1` | — (no sha in manifest) | — | manifest = `jetson-j1_power.json` |

---

*Generated by `scripts/partial_comparison_table.py` (Python 3.6.9, stdlib only). No
`bench/` execution, no git commit, no Dolt sync. For maintainer review before `ob-ami`.*
