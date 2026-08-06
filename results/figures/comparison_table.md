# Results comparison table

Memory figures regenerable via `scripts/generate_memory_plots.py`. Kernel data from committed CSVs in `results/raw/`.

## Static kernel microbenchmark (rk3588-t4)

| Device | Model | Kernel | Cluster | GiB/s (p50) | GFLOP/s | Spread % |
|---|---|---|---|---:|---:|---:|
| rk3588-t4 | Qwen3.5-4B | `gdn_cumdecay` | A76 (big) | 4.25 | 0.57 | 3.4 |
| rk3588-t4 | Qwen3.5-4B | `gdn_gated_scan` | A76 (big) | 3.29 | 0.58 | 17.4 |
| rk3588-t4 | Qwen3.5-4B | `gdn_causal_dwconv1d` | A76 (big) | 4.52 | 4.60 | 8.8 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_cumdecay` | A76 (big) | 5.00 | 0.67 | 4.2 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_gated_scan` | A76 (big) | 4.79 | 0.85 | 3.1 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_causal_dwconv1d` | A76 (big) | 6.00 | 6.10 | 5.6 |
| rk3588-t4 | Qwen3.5-4B | `gdn_cumdecay` | A55 (little) | 0.97 | 0.13 | 10.3 |
| rk3588-t4 | Qwen3.5-4B | `gdn_gated_scan` | A55 (little) | 0.55 | 0.10 | 12.1 |
| rk3588-t4 | Qwen3.5-4B | `gdn_causal_dwconv1d` | A55 (little) | 0.71 | 0.73 | 9.9 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_cumdecay` | A55 (little) | 1.19 | 0.16 | 5.3 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_gated_scan` | A55 (little) | 0.99 | 0.17 | 2.2 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_causal_dwconv1d` | A55 (little) | 0.92 | 0.94 | 63.9 |

## Memory decomposition — Qwen3.5-4B (analytical, from verified config)

_Regenerable: `python3 scripts/generate_memory_plots.py`. See [`memory_comparison.md`](memory_comparison.md) for full table including 0.8B._

| Context | Weights (GiB) | KV cache (GiB) | Recurrent state (MiB) | Total (GiB) | If all-attn (GiB) | Savings |
|---:|---:|---:|---:|---:|---:|---:|
| 4K | 10.42 | 0.12 | 51 | 10.60 | 10.92 | 0.33 GiB |
| 32K | 10.42 | 1.00 | 51 | 11.47 | 14.42 | 2.95 GiB |
| 128K | 10.42 | 4.00 | 51 | 14.47 | 26.42 | 11.95 GiB |
| 262K | 10.42 | 8.00 | 51 | 18.47 | 42.42 | 23.95 GiB |

## Decode bandwidth model — Qwen3.5-4B at 100 GB/s (O6 stretch target)

| Quant | Weight traffic/token | State traffic/token | Total | Ceiling tok/s |
|---|---:|---:|---:|---:|
| fp16 | 10.42 GiB | 51 MiB | 10.47 GiB | ≈9 |
| INT8 | 5.21 GiB | 51 MiB | 5.26 GiB | ≈18 |
| INT4 (W4A16) | 2.61 GiB | 51 MiB | 2.66 GiB | ≈35 |
