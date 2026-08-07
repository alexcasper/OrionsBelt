# rk3588-t4-big-omp4 — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 82.0 | 95.7 | 16.7% | 23.83 | 3.20 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 268.1 | 287.0 | 7.1% | 11.04 | 1.96 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 98.6 | 113.2 | 14.8% | 20.89 | 21.27 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 40.8 | 41.7 | 2.1% | 35.87 | 6.42 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 263.1 | 284.7 | 8.2% | 11.19 | 1.99 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 59.2 | 61.5 | 3.9% | 24.74 | 4.43 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 261.9 | 277.1 | 5.8% | 11.24 | 2.00 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 707.6 | 806.5 | 14.0% | 6.94 | 1.48 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 32.1 | 32.7 | 1.8% | 30.44 | 4.09 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 125.7 | 133.3 | 6.0% | 11.77 | 2.09 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 61.0 | 62.7 | 2.9% | 16.89 | 17.20 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 17.8 | 18.1 | 1.6% | 41.16 | 7.37 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 129.2 | 135.1 | 4.5% | 11.40 | 2.03 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 28.0 | 28.3 | 1.0% | 26.16 | 4.68 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 129.8 | 135.3 | 4.3% | 11.34 | 2.02 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 280.9 | 297.8 | 6.0% | 8.75 | 1.87 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.5 | 1.8 | 20.0% | 20.92 | 2.81 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 1.8 | 2.0 | 16.6% | 43.57 | 4.68 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 3.8 | 4.1 | 7.7% | 36.22 | 8.64 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 1.8 | 1.8 | 0.1% | 13.08 | 2.34 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 1.8 | 2.0 | 16.7% | 34.87 | 4.68 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 1.8 | 2.0 | 16.6% | 13.07 | 2.34 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 2.0 | 2.3 | 14.3% | 29.89 | 4.01 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 2.3 | 2.6 | 12.5% | 45.78 | 7.02 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 1.5 | 1.5 | 0.1% | 10.46 | 1.40 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.5 | 1.8 | 20.0% | 26.15 | 2.81 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 2.3 | 2.6 | 12.5% | 29.42 | 7.02 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 1.5 | 1.5 | 0.1% | 7.85 | 1.40 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.5 | 1.8 | 19.9% | 20.91 | 2.81 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.5 | 1.8 | 19.9% | 7.84 | 1.40 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 1.8 | 2.0 | 16.6% | 17.44 | 2.34 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.8 | 1.8 | 0.1% | 30.52 | 4.68 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 23.83 | 75.2% | 82.0 | 16.7% |
| Gated Delta-Rule Scan | 11.04 | 34.8% | 268.1 | 7.1% |
| Causal Depthwise Conv1D | 20.89 | 65.9% | 98.6 | 14.8% |
| gdn_cumdecay_f16 | 35.87 | 113.2% | 40.8 | 2.1% |
| gdn_gated_scan_f16 | 11.19 | 35.3% | 263.1 | 8.2% |
| gdn_cumdecay_bf16 | 24.74 | 78.0% | 59.2 | 3.9% |
| gdn_gated_scan_bf16 | 11.24 | 35.5% | 261.9 | 5.8% |
| gdn2_gated_scan | 6.94 | 21.9% | 707.6 | 14.0% |
| Gated Cumulative Decay | 30.44 | 96.0% | 32.1 | 1.8% |
| Gated Delta-Rule Scan | 11.77 | 37.1% | 125.7 | 6.0% |
| Causal Depthwise Conv1D | 16.89 | 53.3% | 61.0 | 2.9% |
| gdn_cumdecay_f16 | 41.16 | 129.8% | 17.8 | 1.6% |
| gdn_gated_scan_f16 | 11.40 | 36.0% | 129.2 | 4.5% |
| gdn_cumdecay_bf16 | 26.16 | 82.5% | 28.0 | 1.0% |
| gdn_gated_scan_bf16 | 11.34 | 35.8% | 129.8 | 4.3% |
| gdn2_gated_scan | 8.75 | 27.6% | 280.9 | 6.0% |
| Gated Cumulative Decay | 20.92 | 66.0% | 1.5 | 20.0% |
| Gated Delta-Rule Scan | 43.57 | 137.4% | 1.8 | 16.6% |
| Causal Depthwise Conv1D | 36.22 | 114.3% | 3.8 | 7.7% |
| gdn_cumdecay_f16 | 13.08 | 41.3% | 1.8 | 0.1% |
| gdn_gated_scan_f16 | 34.87 | 110.0% | 1.8 | 16.7% |
| gdn_cumdecay_bf16 | 13.07 | 41.2% | 1.8 | 16.6% |
| gdn_gated_scan_bf16 | 29.89 | 94.3% | 2.0 | 14.3% |
| gdn2_gated_scan | 45.78 | 144.4% | 2.3 | 12.5% |
| Gated Cumulative Decay | 10.46 | 33.0% | 1.5 | 0.1% |
| Gated Delta-Rule Scan | 26.15 | 82.5% | 1.5 | 20.0% |
| Causal Depthwise Conv1D | 29.42 | 92.8% | 2.3 | 12.5% |
| gdn_cumdecay_f16 | 7.85 | 24.8% | 1.5 | 0.1% |
| gdn_gated_scan_f16 | 20.91 | 66.0% | 1.5 | 19.9% |
| gdn_cumdecay_bf16 | 7.84 | 24.7% | 1.5 | 19.9% |
| gdn_gated_scan_bf16 | 17.44 | 55.0% | 1.8 | 16.6% |
| gdn2_gated_scan | 30.52 | 96.3% | 1.8 | 0.1% |
