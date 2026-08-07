# Master comparison table

_Generated from committed CSVs and manifests. Every measurement states its source commit and spread.
See per-device tables (`*_table.md`) for full kernel-level detail._

## Provenance

| Device | CSV | Git SHA | Dirty | Governor | Manifest |
|---|---|---|---|---|---|
| rk3588-t3 (big) | `rk3588-t3-clean.csv` | `f015982` | **false** | performance | `rk3588-t3-clean.json` |
| rk3588-t4 (big) | `rk3588-t4-clean.csv` | `1ca4d6d` | **false** | performance | `rk3588-t4-clean.json` |

> Both t3 and t4 are clean-tree, single-thread, post-optimization runs (ob-aw9).
> **Cross-board gap:** t3 is 1.6–2.9× faster than t4 despite t4 clocked higher
> (2400 vs 2304 MHz). t3 is an unknown RK3588 board (kernel 5.10, CFS, 32 GB);
> t4 is a Turing Machines RK1 (kernel 6.11, EEVDF, 8 GB). The gap is a genuine
> hardware/environment difference, not a measurement artifact (see
> [`FINDINGS.md`](../../docs/FINDINGS.md) §"Cross-Board Gap").

## 1. Headline: GDN kernel bandwidth — RK3588 Cortex-A76 (big)

Qwen3.5-4B, prefill (seq=64), fp32 baseline, spec bandwidth 34.0 GiB/s.

| Kernel | t3 GiB/s | t3 spread | t4 GiB/s | t4 spread | t3÷t4 ratio |
|---|---:|---:|---:|---:|---:|
| gdn_cumdecay | 21.06 | 3.5% | 7.40 | 5.0% | 2.85× |
| gdn_gated_scan | 10.62 | 5.4% | 5.67 | 7.4% | 1.87× |
| gdn_causal_dwconv1d | 18.73 | 4.8% | 7.04 | 4.3% | 2.66× |

> Both boards achieve well under the 34 GiB/s spec bandwidth per cluster. t3's
> cumdecay reaches 62% of spec; t4 reaches 22%. The gap is systematic across
> all kernels — see FINDINGS.md for the cross-board analysis (board vendor,
> kernel scheduler, and DRAM differences).
> Gated scan runs at a lower fraction of spec than cumdecay/dwconv because its
> sequential recurrence is instruction-overhead-bound, not DRAM-bandwidth-bound
> (see [`fleet_bandwidth_scaling.md`](fleet_bandwidth_scaling.md)).

Qwen3.5-0.8B, prefill (seq=64), fp32 baseline (t3 only; t4 shows similar ratios):

| Kernel | t3 GiB/s | t3 spread |
|---|---:|---:|
| gdn_cumdecay | 28.62 | 7.7% |
| gdn_gated_scan | 15.24 | 2.4% |
| gdn_causal_dwconv1d | 28.48 | 1.6% |

## 2. Mixed-precision optimization impact

Qwen3.5-4B, prefill (seq=64), A76 big. Both devices clean, post-optimization.

| Kernel | t3 fp32 | t3 fp16 | t3 bf16 | t4 fp32 | t4 fp16 | t4 bf16 | fp16 speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| gdn_cumdecay | 21.06 | 37.20 | 25.49 | 7.40 | 8.61 | 6.01 | **1.16–1.77×** |
| gdn_gated_scan | 10.62 | 10.47 | 10.51 | 5.67 | 5.66 | 5.67 | 0.99–1.00× |
| gdn_causal_dwconv1d | 18.73 | — | — | 7.04 | — | — | — |

> fp16 halves memory traffic for the decay chain (elementwise gate/decay ops),
> yielding 1.77× on t3 cumdecay but only 1.16× on t4 (the absolute throughput
> ceiling differs by board). Gated scan shows **no fp16 benefit** because it is
> compute-bound on the delta-rule matmul, not bandwidth-bound — consistent with
> the instruction-overhead finding on both boards.
> bf16 on A76 has no hardware path (software emulation), so it underperforms
> fp32 on t4 and only modestly exceeds it on t3.

## 3. Decode-phase kernel performance

Qwen3.5-4B, decode (seq=1), A76 big. Both devices clean, post-optimization.

| Kernel | t3 µs/tok | t3 GiB/s | t3 spread | t4 µs/tok | t4 GiB/s | t4 spread |
|---|---:|---:|---:|---:|---:|---:|
| gdn_cumdecay | 1.166 | 26.17 | 0.1% | 1.750 | 17.44 | 16.7% |
| gdn_gated_scan | 1.458 | 52.33 | 20.0% | 2.333 | 32.70 | 0.0% |
| gdn_causal_dwconv1d | 2.625 | 52.31 | 11.1% | 9.043 | 15.19 | 3.2% |

> Decode GiB/s exceeds the 34 GiB/s DRAM spec on t3 because at seq=1 the working
> set fits in L1/L2 cache. t4 achieves lower cache-resident throughput, consistent
> with the cross-board gap. The **per-token latency** (µs/token) is the load-bearing
> decode metric — these are the actual costs a decode loop pays per GDN layer per token.

## 4. Per-layer latency: GDN vs full-attention

From t4 per-layer profiling (Qwen3.5-0.8B, Python `transformers`, commit `fb578e1`, dirty=true).
This is model-level timing through the actual Qwen3.5 forward pass — the closest available
measurement to end-to-end inference. See [`rk3588-t4_layer_profile.csv`](../raw/rk3588-t4_layer_profile.csv).

| Phase | Context | Layer type | Mean p50 (µs) | Layers | Total (µs) |
|---|---|---|---:|---:|---:|
| Decode | 32 | linear_attention (GDN) | 47,492 | 18 | 854,856 |
| Decode | 32 | full_attention | 42,286 | 6 | 253,716 |
| Decode | 64 | linear_attention (GDN) | 48,667 | 18 | 876,006 |
| Decode | 64 | full_attention | 48,737 | 6 | 292,422 |
| Decode | 128 | linear_attention (GDN) | 51,806 | 18 | 932,508 |
| Decode | 128 | full_attention | 56,971 | 6 | 341,826 |
| Prefill | 32 | linear_attention (GDN) | 129,160 | 18 | 2,324,880 |
| Prefill | 32 | full_attention | 78,081 | 6 | 468,486 |
| Prefill | 64 | linear_attention (GDN) | 171,120 | 18 | 3,080,160 |
| Prefill | 64 | full_attention | 112,658 | 6 | 675,948 |
| Prefill | 128 | linear_attention (GDN) | 238,802 | 18 | 4,298,436 |
| Prefill | 128 | full_attention | 158,013 | 6 | 948,078 |

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

Qwen3.5-4B at 100 GB/s (O6 LPDDR5 spec; RK3588 shared pool is 34 GiB/s per cluster).

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

Qwen3.5-4B, fp32, A76 big cluster (cpu4-7), governor=performance, 8 decode tokens.

| Device | tok/s | TTFT (s) | Git SHA | Manifest |
|---|---:|---:|---|---|
| rk3588-t3 | 0.08 | 13.32 | `a086a50` | `rk3588-t3_e2e.json` |
| rk3588-t4 | 0.09 | 11.76 | `def3f29` | `rk3588-t4_e2e.json` |

> This is the model-level headline number: a full Qwen3.5-4B forward pass through
> all 32 layers (24 GDN + 8 full-attention) with the optimized GDN kernels. The
> two boards agree within 12%, which is expected given the cross-board gap (§1).
> t4 is slightly faster at e2e despite slower kernel-level throughput — likely
> because the e2e binary exercises different code paths (full matmul/FFN blocks
> where t4's higher clock helps). At fp32 on RK3588, 0.09 tok/s = ~11 s/token is
> too slow for interactive use but establishes the CPU-only baseline against
> which INT4 quantization and heterogeneous dispatch must improve. See FINDINGS.md
> §"End-to-end Qwen3.5-0.8B tokens/sec" for the 0.8B variant (0.47 tok/s).
>
> **Context:** the analytical ceiling (§6) predicts ≈12 tok/s at fp16 on a
> 100 GB/s bus — we are 130× below that ceiling because the current decode loop
> is unoptimized reference code. This gap is the optimization opportunity the
> project targets.

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
