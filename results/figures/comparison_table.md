# Master comparison table

_Generated from committed CSVs and manifests. Every measurement states its source commit and spread.
See per-device tables (`*_table.md`) for full kernel-level detail._

## Provenance

| Device | CSV | Git SHA | Dirty | Governor | Threads | Manifest |
|---|---|---|---|---|---|---|
| rk3588-t3 (big, **8-thread**) | `rk3588-t3-clean.csv` | `854c6f1` | **false** | performance | 8 | `rk3588-t3-clean.json` |
| rk3588-t3 (big, **1-thread**) | `rk3588-t3-clean-singlethread.csv` | `d72eaa1` | **false** | performance | 1 | `rk3588-t3-clean-singlethread.json` |
| rk3588-t4 (big, 1-thread) | `rk3588-t4-clean.csv` | `1ca4d6d` | **false** | performance | 1 | `rk3588-t4-clean.json` |

> **⚠ Correction (ob-mrd.12, ob-mrd.13):** The original provenance note claimed
> both boards were "single-thread." This was **false for t3**: manifests show
> `rk3588-t3-clean.csv` ran with `effective_threads=8` (thread count defaulted to
> core count). t4's `rk3588-t4-clean.csv` was genuinely 1-thread
> (`OMP_NUM_THREADS=1`). They are therefore **not directly comparable**.
>
> A genuine 1-thread t3 run was captured for ob-mrd.13
> (`rk3588-t3-clean-singlethread.csv`, `OMP_NUM_THREADS=1`, taskset-pinned to
> one A76 core, commit `d72eaa1`). The like-for-like 1-thread comparison is in
> §1a below.
>
> The equal-thread-count 8v8 comparison (`rk3588-t3-clean.csv` vs
> `rk3588-t4_big.csv`, both `effective_threads=8`) gives cumdecay 21.39 vs 27.33,
> gated scan 10.56 vs 11.45, Conv1D 20.59 vs 21.93 — scan and Conv1D agree
> within ~8%, consistent with t4's higher 2400 MHz clock. The cumdecay pair
> agreed within 1% on the 2026-08-12 t4 run (21.39 vs 21.67); the 2026-08-14
> re-run reads 27.33 — cumdecay has ±25% session-to-session variance on t4
> (FINDINGS §43), so it is excluded from the agreement claim. Both CSVs are
> clean-tree (dirty=false); t4 re-run at `4169648` (manifest), commit
> `a5595ab8` (2026-08-14, ob-8ms.3), replacing the prior clean run at
> `aa61e20` (PR #313).
>
> t3 is an unknown RK3588 board (kernel 5.10, CFS, 32 GB); t4 is a Turing
> Machines RK1 (kernel 6.11, EEVDF, 8 GB). Both use Cortex-A76 big cores at
> ~2.3 GHz (t4: 2400 MHz; t3: 2304 MHz).

## 1. GDN kernel bandwidth — RK3588 Cortex-A76 (big, 8-thread vs 1-thread)

### 1a. Like-for-like single-thread comparison (ob-mrd.13)

Qwen3.5-4B, prefill (seq=64), fp32. Both devices `OMP_NUM_THREADS=1`, one A76 core, governor=performance, clean tree.

| Kernel | t3 GiB/s | t3 spread | t4 GiB/s | t4 spread | t3÷t4 ratio |
|---|---:|---:|---:|---:|---:|
| gdn_cumdecay | 7.01 | 6.7% | 7.40 | 5.0% | 0.95× |
| gdn_gated_scan | 3.07 | 7.5% | 5.67 | 7.4% | 0.54× |
| gdn_causal_dwconv1d | 6.02 | 4.3% | 7.04 | 4.3% | 0.86× |

> **The apparent 2.85× "cross-board gap" (§1b) collapses for cumdecay at matched
> thread counts** (0.95×, within noise). Gated scan remains lower on t3 (0.54×,
> t4 is 1.87× faster); the cause is likely a board-level memory-subsystem
> difference (not clock — both are ~2.3 GHz A76). dwconv1d is 0.86×, also within
> the known run-to-run variance (ob-bf7: 1.68× on this fleet).
>
> Sources: `rk3588-t3-clean-singlethread.csv` (commit `d72eaa1`) vs
> `rk3588-t4-clean.csv` (commit `1ca4d6d`). Different commits, but both
> clean-tree post-NEON-optimization with identical kernel code.

### 1b. 8-thread (t3) vs 1-thread (t4) — confounded, kept for historical reference

Qwen3.5-4B, prefill (seq=64), fp32. ⚠ t3 was 8-thread (threads_source=core_count_default); t4 was 1-thread. **Not comparable** — see §1a for the valid comparison.

| Kernel | t3 GiB/s (8T) | t4 GiB/s (1T) | t3÷t4 ratio |
|---|---:|---:|---:|
| gdn_cumdecay | 21.39 | 7.40 | 2.89× |
| gdn_gated_scan | 10.56 | 5.67 | 1.86× |
| gdn_causal_dwconv1d | 20.59 | 7.04 | 2.93× |

> The 8-thread t3 numbers reflect OpenMP scaling across 4 big cores. The
> 8-thread vs 8-thread comparison (`rk3588-t3-clean.csv` vs `rk3588-t4_big.csv`,
> both effective_threads=8) gives cumdecay 21.39 vs 27.33, gated scan 10.56 vs 11.45,
> Conv1D 20.59 vs 21.93 — scan and Conv1D agree within ~8%; cumdecay swung +28%
> on the 2026-08-14 t4 re-run (session variance, FINDINGS §43; the 2026-08-12
> run had agreed within 1%). Both CSVs re-run clean
> (dirty=false): t3 at `854c6f1`, t4 at `4169648`/commit `a5595ab8`
> (2026-08-14, ob-8ms.3), each replacing a prior
> run. Non-GDN-2 numbers are within the documented run-to-run
> variance (ob-bf7).
> See §8 for OpenMP scaling analysis.

### 1c. 0.8B model (t3 8-thread, t4 1-thread — also confounded)

Qwen3.5-0.8B, prefill (seq=64), fp32 baseline (t3 only; t4 shows similar ratios):

| Kernel | t3 GiB/s | t3 spread |
|---|---:|---:|
| gdn_cumdecay | 24.98 | 2.2% |
| gdn_gated_scan | 11.53 | 10.7% |
| gdn_causal_dwconv1d | 15.98 | 2.7% |

> Note: 0.8B prefill numbers shift significantly across measurement sessions on
> this fleet (ob-bf7: gated_scan ranged 97–204 µs across runs). dwconv1d at the
> 0.8B shape is especially noisy (16.0–22.5 GiB/s across 3 back-to-back runs at
> `854c6f1`). See FINDINGS.md §10 for GDN-2 aliasing correction.

## 2. Mixed-precision optimization impact

Qwen3.5-4B, prefill (seq=64), A76 big. Both devices clean, post-optimization.

| Kernel | t3 fp32 | t3 fp16 | t3 bf16 | t4 fp32 | t4 fp16 | t4 bf16 | fp16 speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| gdn_cumdecay | 21.39 | 35.12 | 25.11 | 7.40 | 8.61 | 6.01 | **1.16–1.64×** |
| gdn_gated_scan | 10.56 | 10.43 | 10.45 | 5.67 | 5.66 | 5.67 | 0.99–1.00× |
| gdn_causal_dwconv1d | 20.59 | — | — | 7.04 | — | — | — |

> fp16 halves memory traffic for the decay chain (elementwise gate/decay ops),
> yielding 1.64× on t3 cumdecay but only 1.16× on t4 (the absolute throughput
> ceiling differs by board). Gated scan shows **no fp16 benefit** because it is
> compute-bound on the delta-rule matmul, not bandwidth-bound — consistent with
> the instruction-overhead finding on both boards.
> bf16 on A76 has no hardware path (software emulation), so it underperforms
> fp32 on t4 and only modestly exceeds it on t3.

## 3. Decode-phase kernel performance

Qwen3.5-4B, decode (seq=1), A76 big. Both devices clean, post-optimization.

| Kernel | t3 µs/tok | t3 GiB/s | t3 spread | t4 µs/tok | t4 GiB/s | t4 spread |
|---|---:|---:|---:|---:|---:|---:|
| gdn_cumdecay | 1.035 | 29.47 | 0.8% | 1.750 | 17.44 | 16.7% |
| gdn_gated_scan | 1.386 | 55.07 | 1.3% | 2.333 | 32.70 | 0.0% |
| gdn_causal_dwconv1d | 3.293 | 41.70 | 1.0% | 9.043 | 15.19 | 3.2% |

> Decode GiB/s exceeds the 31.7 GiB/s DRAM spec on t3 because at seq=1 the working
> set fits in L1/L2 cache. t4 achieves lower cache-resident throughput, consistent
> with the cross-board gap. The **per-token latency** (µs/token) is the load-bearing
> decode metric — these are the actual costs a decode loop pays per GDN layer per token.

## 4. Per-layer latency: GDN vs full-attention

From t4 per-layer profiling (Qwen3.5-0.8B, Python `transformers`, commit `80dc1fc5`, dirty=false).
Clean re-run on a clean tree — this is model-level timing through the actual Qwen3.5 forward
pass. See [`rk3588-t4_layer_profile.csv`](../raw/rk3588-t4_layer_profile.csv).

| Phase | Context | Layer type | Mean p50 (µs) | Layers | Total (µs) |
|---|---|---|---:|---:|---:|
| Decode | 32 | linear_attention (GDN) | 50,485 | 18 | 908,732 |
| Decode | 32 | full_attention | 46,037 | 6 | 276,224 |
| Decode | 64 | linear_attention (GDN) | 48,876 | 18 | 879,775 |
| Decode | 64 | full_attention | 49,558 | 6 | 297,349 |
| Decode | 128 | linear_attention (GDN) | 52,493 | 18 | 944,877 |
| Decode | 128 | full_attention | 53,914 | 6 | 323,483 |
| Prefill | 32 | linear_attention (GDN) | 126,586 | 18 | 2,278,540 |
| Prefill | 32 | full_attention | 76,521 | 6 | 459,124 |
| Prefill | 64 | linear_attention (GDN) | 166,478 | 18 | 2,996,599 |
| Prefill | 64 | full_attention | 106,189 | 6 | 637,134 |
| Prefill | 128 | linear_attention (GDN) | 239,104 | 18 | 4,303,867 |
| Prefill | 128 | full_attention | 145,741 | 6 | 874,444 |

> **At decode**, GDN and full-attention layers have similar per-layer cost (~49 ms) and neither
> grows significantly with context — GDN because its state is fixed-size, full-attention because
> the KV cache at ctx≤128 is still small.
>
> **At prefill**, GDN layers are 1.5–1.7× slower per-layer than full-attention, and both scale
> roughly linearly with context. The GDN cost is dominated by the chunkwise WY recurrence
> (sequential scan), not by matmul throughput. See FINDINGS.md §"Chunkwise WY bottleneck".

## 5. Memory decomposition: GDN O(1) state vs attention O(n) KV cache

_Analytical model from `src/orionsbelt/engines/memory.py`. Shapes verified against primary sources.
Full table including 0.8B: [`memory_comparison.md`](memory_comparison.md)._

### Qwen3.5-4B (24 GDN + 8 full-attention layers, 3:1 hybrid)

| Context | Weights (fp16) | KV cache (8 attn) | GDN state | Total (hybrid) | If all-attn | Savings |
|---:|---:|---:|---:|---:|---:|---:|
| 4K | 7.83 GiB | 0.12 GiB | 51 MiB | 8.01 GiB | 8.33 GiB | 0.33 GiB |
| 32K | 7.83 GiB | 1.00 GiB | 51 MiB | 8.88 GiB | 11.83 GiB | 2.95 GiB |
| 128K | 7.83 GiB | 4.00 GiB | 51 MiB | 11.88 GiB | 23.83 GiB | 11.95 GiB |
| 262K | 7.83 GiB | 8.00 GiB | 51 MiB | 15.88 GiB | 39.83 GiB | **23.95 GiB** |

> At **262K context** the GDN hybrid saves **23.95 GiB** versus a hypothetical all-attention model.
> The recurrent state is **51 MiB** regardless of context length, while the KV cache alone reaches
> **8.0 GiB** — exceeding the fp16 weight footprint (7.83 GiB).

## 6. Decode bandwidth ceilings (analytical)

Qwen3.5-4B at 93.1 GiB/s (O6 LPDDR5 spec; RK3588 shared pool is 31.7 GiB/s per cluster).

| Quantization | Weight traffic/token | State traffic/token | Total | Ceiling tok/s |
|---|---:|---:|---:|---:|
| fp16 | 7.83 GiB | 51 MiB | 7.88 GiB | ≈12 |
| INT8 | 3.92 GiB | 51 MiB | 3.96 GiB | ≈23 |
| INT4 (W4A16) | 1.96 GiB | 51 MiB | 2.01 GiB | ≈46 |

> These are **upper bounds** assuming the decode loop is purely memory-bandwidth-bound (no compute
> overhead, no cache effects). Real throughput will be lower. The point is the scaling: INT4
> quantization roughly doubles decode throughput versus fp16 because it halves the dominant
> weight-traffic cost. The GDN recurrent state (51 MiB) is negligible at every precision.

## 7. End-to-end model decode: tokens/sec (measured)

Qwen3.5-4B + 0.8B, A76 big cluster (cpu4-7), governor=performance.
Token counts differ by device/generation: t3 FP32 = 20 tokens, t3 INT8 = 16 tokens,
t4 = 8 tokens (re-run, see per-row manifest for details).

**After GEMV row-sweep optimization (commit `2e752af`, §FINDINGS-15):**

| Device | Model | tok/s | TTFT (ms) | Git SHA | Manifest |
|---|---|---:|---:|---|---|
| rk3588-t3 | 4B   | 1.04 | 964 | `2e752af` | `rk3588-t3_e2e.json` |
| rk3588-t4 | 4B   | 1.11 | 900 | `3d914b6` | `rk3588-t4_big_e2e.json` |
| rk3588-t3 | 0.8B | 7.98 | 125 | `7962968` | `rk3588-t3_08b_e2e.json` |
| rk3588-t4 | 0.8B | 8.32 | 120 | `79668e3` | `rk3588-t4_08b_big_e2e.json` |

> **10–15× speedup** from fixing the GEMV memory access pattern (column-sweep →
> row-sweep + OpenMP tiles). The two boards agree within 7%, consistent with the
> known cross-board gap (§1). See FINDINGS.md §15 for the full analysis.
>
> **Before optimization** (commit `a756662`): t3=0.07, t4=0.09 tok/s — the gap
> to the ~12 tok/s analytical ceiling (§6) was from a pathological column-sweep
> GEMV that achieved <1 GiB/s of ~25 GiB/s bandwidth, not from the GDN kernels.
> **Bottleneck breakdown** (4B, after optimization): FFN 72%, GDN proj 14%,
> full-attn 6%, GDN conv/decay/scan <0.1%. The GDN recurrent kernels are
> negligible — the next high-impact optimization is INT8 weight quantization
> (halves FFN memory traffic, the dominant phase).

**After INT8 weight-only quantization (commit `bdca994`, §FINDINGS-INT8):**

| Device | Model | Quant | tok/s | TTFT (ms) | Git SHA | Manifest |
|---|---|---|---:|---:|---|---|
| rk3588-t3 | 4B   | INT8 | 1.84 | 542 | `d5d867d` | `rk3588-t3_big_int8_e2e.json` |
| rk3588-t4 | 4B   | INT8 | 1.83 | 545 | `861bdf2` | `rk3588-t4_big_int8_e2e.json` |
| rk3588-t3 | 0.8B | INT8 | 10.58 | 94 | `a8e2319` | `rk3588-t3_08b_big_int8_e2e.json` |
| rk3588-t4 | 0.8B | INT8 | 10.03 | 100 | `d2262f1` | `rk3588-t4_08b_big_int8_e2e.json` |

> INT8 weight-only quantization (per-column symmetric, NEON dequantize-on-the-fly)
> cuts weight memory traffic 4×. t3 reported 1.77× speedup (4B) and 1.33× (0.8B);
> t4 confirms the optimisation direction with 1.65× (4B) and 1.21× (0.8B) over its
> own FP32 baseline. Cross-board agreement at INT8 is ~0.5% (4B) and ~5% (0.8B),
> tighter than the FP32 comparison because the bandwidth-bound decode is less
> sensitive to board-level compute differences.
> See FINDINGS.md "INT8 weight-only quantization" section for full analysis.

**After SDOT INT8 GEMV kernel (commit `dccee52`, §33/§38):**

| Device | Model | Quant | tok/s | TTFT (ms) | Git SHA | Manifest |
|---|---|---|---:|---:|---|---|
| rk3588-t3 | 4B   | INT8+SDOT | 2.80 | 357 | `c880887` | `rk3588-t3_big_int8_sdot_e2e.json` |
| rk3588-t4 | 4B   | INT8+SDOT | 3.37 | 296 | `be4d3ca` | `rk3588-t4_e2e_ctxsweep_int8_puregdn_4t.json` |
| rk3588-t3 | 0.8B | INT8+SDOT | 25.60 | 39 | `c880887` | `rk3588-t3_08b_big_int8_sdot_e2e.json` |
| rk3588-t4 | 0.8B | INT8+SDOT | 29.25 | 34 | `be4d3ca` | `rk3588-t4_e2e_ctxsweep_08b_int8_puregdn_4t.json` |

> The SDOT (`vdotq_lane_s32`) INT8×INT8→int32 dot-product kernel replaces the NEON
> dequant→float32 FMA path, cutting GDN projection and FFN decode time by
> 1.63× and 1.85× respectively. t3's pre-SDOT INT8 was 1.84 tok/s (4B) / 10.58
> tok/s (0.8B); with SDOT, 2.80 / 25.60 — a 1.52× and 2.42× speedup. **Provenance
> note (t3 re-run at clean HEAD `c880887`, 2026-08-10):** t3's previous SDOT
> manifests were dirty=true (binary built from uncommitted tree). Re-running at
> clean HEAD yields ~15% lower INT8 numbers; 4 runs agree within 1.8%. INT4
> numbers are within 2-5% of the dirty values. **t4 microbench re-run
> (2026-08-10, clean `d6b77b2`):** INT8+SDOT and INT4+SDOT microbench CSVs
> (sdot_4b_big, sdot_08b_big, int4sdot_4b_big, int4sdot_08b_big) re-run at
> clean HEAD, dirty=false. Numbers within 2% of original dirty runs. The e2e
> ctxsweep manifests were previously dirty=true but are now confirmed clean
> (false positives from the output-file counting bug fixed in ob-mrd.24).
> See FINDINGS §33 and §38.

**After INT4+SDOT hybrid GEMV kernel (commit `3bff376`, §34):**

| Device | Model | Quant | tok/s | TTFT (ms) | Git SHA | Manifest |
|---|---|---|---:|---:|---|---|
| rk3588-t3 | 4B   | INT4+SDOT | 4.21 | 238 | `c880887` | `rk3588-t3_big_int4_sdot_e2e.json` |
| rk3588-t4 | 4B   | INT4+SDOT | 4.52 | 221 | `d6b77b2` | `rk3588-t4_int4sdot_4b.json` |
| rk3588-t3 | 0.8B | INT4+SDOT | 35.05 | 28 | `c880887` | `rk3588-t3_08b_big_int4_sdot_e2e.json` |
| rk3588-t4 | 0.8B | INT4+SDOT | 36.36 | 28 | `d6b77b2` | `rk3588-t4_int4sdot_08b.json` |

> INT4+SDOT combines K-grouped nibble repack with `vdotq_lane_s32` integer
> dot-product, achieving 2× memory advantage of INT4 with the compute efficiency
> of SDOT. vs INT8+SDOT (t3 clean): **1.50×** (4B) and **1.37×** (0.8B) on A76.
> The A55 little cluster does NOT benefit (0.96× — compute-bound, nibble-unpack
> overhead dominates). Cross-device agreement: 4B at **7.4%** gap (4.21 vs 4.52),
> 0.8B at **3.7%** gap (35.05 vs 36.36). See FINDINGS §34.
>
> **Cumulative optimization stack (4B A76, t3 clean):** 0.07 → 1.04 → 1.84 → 2.80 → **4.21 tok/s**
> (~60× over naive FP32 baseline). t4 confirms at 4.52 tok/s (~65×, clean manifest `d6b77b2`).

## 8. OpenMP multi-threading scaling (t4)

Qwen3.5-4B, prefill (seq=64), fp32, A76 big cluster.

| Config | gated_scan GiB/s | vs single-thread |
|---|---:|---:|
| Single-thread | 5.67 | 1.00× |
| 4-thread OpenMP | 11.04 | **1.95×** |

> Source: `rk3588-t4-clean.csv` (single) vs `rk3588-t4-big-omp4.csv` (OMP).
> The 1.95× scaling (not the ideal 4×) reflects the sequential nature of the
> delta-rule recurrence: channels are parallelizable, but the per-channel scan
> is serial. Amdahl's law limits the speedup to the fraction of work that
> parallelizes across channels. This is consistent with the instruction-overhead
> bottleneck identified in FINDINGS.md.
