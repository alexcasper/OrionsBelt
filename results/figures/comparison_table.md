# Master comparison table

_Generated from committed CSVs and manifests. Every measurement states its source commit and spread.
See per-device tables (`*_table.md`) for full kernel-level detail._

## Provenance

| Device | CSV | Git SHA | Dirty | Governor | Manifest |
|---|---|---|---|---|---|
| rk3588-t3 (big) | `rk3588-t3_big.csv` | `553a96e` | **false** | performance | `rk3588-t3.json` |
| rk3588-t3 (little) | `rk3588-t3_little.csv` | `553a96e` | **false** | performance | `rk3588-t3.json` |
| rk3588-t4 (big) | `rk3588-t4_big.csv` | `28729f3` | true | performance | `rk3588-t4.json` |
| rk3588-t4 (little) | `rk3588-t4_little.csv` | `28729f3` | true | performance | `rk3588-t4.json` |
| rk3588-t4 (big, baseline) | `rk3588-t4_big_singlethread.csv` | `28729f3` | true | performance | `rk3588-t4.json` |

> t3 is the **primary data source** (dirty=false, clean tree at `553a96e`).
> t4 serves as a **cross-check** on the same hardware class (same RK3588 SoC, different node).

## 1. Headline: GDN kernel bandwidth — RK3588 Cortex-A76 (big)

Qwen3.5-4B, prefill (seq=64), fp32 baseline, spec bandwidth 34.0 GiB/s.

| Kernel | t3 GiB/s | t3 spread | t4 GiB/s | t4 spread | t3↔t4 agreement |
|---|---:|---:|---:|---:|---:|
| gdn_cumdecay | 21.74 | 5.2% | 24.26 | 10.9% | 10% |
| gdn_gated_scan | 11.07 | 6.2% | 11.48 | 7.5% | 4% |
| gdn_causal_dwconv1d | 21.60 | 4.3% | 21.02 | 3.6% | 3% |

> Cumulative-decay and Conv1D achieve **64% and 63% of spec bandwidth** respectively — close to the
> memory-bandwidth ceiling for a single A76 core. Gated scan runs at 33% of spec because its
> sequential recurrence is instruction-overhead-bound, not DRAM-bandwidth-bound (see
> [`fleet_bandwidth_scaling.md`](fleet_bandwidth_scaling.md)).

Qwen3.5-0.8B, prefill (seq=64), fp32 baseline:

| Kernel | t3 GiB/s | t3 spread |
|---|---:|---:|
| gdn_cumdecay | 27.67 | 7.4% |
| gdn_gated_scan | 15.38 | 2.7% |
| gdn_causal_dwconv1d | 29.92 | 2.5% |

## 2. Mixed-precision optimization impact

Qwen3.5-4B, prefill (seq=64), A76 big, t3 (`553a96e`, clean tree).

| Kernel | fp32 GiB/s | fp16 GiB/s | bf16 GiB/s | fp16 speedup | bf16 speedup |
|---|---:|---:|---:|---:|---:|
| gdn_cumdecay | 21.74 | 34.87 | 24.62 | **1.60×** | 1.13× |
| gdn_gated_scan | 11.07 | 10.65 | 10.94 | 0.96× | 0.99× |
| gdn_causal_dwconv1d | 21.60 | — | — | — | — |

> fp16 halves memory traffic for the decay chain (elementwise gate/decay ops), yielding 1.6× on
> cumdecay. Gated scan shows **no fp16 benefit** because it is compute-bound on the delta-rule
> matmul, not bandwidth-bound — consistent with the instruction-overhead finding. bf16 on A76 has
> no hardware path (software emulation), so it only matches fp32 despite halving traffic.

## 3. Decode-phase kernel performance

Qwen3.5-4B, decode (seq=1), A76 big, t3 (`553a96e`, clean tree).

| Kernel | µs/token | GiB/s | Spread | % of spec BW |
|---|---:|---:|---:|---:|
| gdn_cumdecay | 1.166 | 26.17 | 0.1% | 77% |
| gdn_gated_scan | 1.458 | 52.33 | 0.1% | 154% |
| gdn_causal_dwconv1d | 2.625 | 52.32 | 11.1% | 154% |

> Decode GiB/s exceeds the 34 GiB/s DRAM spec because at seq=1 the working set (single-token
> vectors) fits in L1/L2 cache. The **per-token latency** (µs/token) is the load-bearing decode
> metric — these are the actual costs a decode loop pays per GDN layer per token.

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
