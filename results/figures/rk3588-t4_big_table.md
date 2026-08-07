# rk3588-t4_big — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 86.9 | 103.5 | 19.1% | 22.47 | 3.02 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 266.9 | 280.9 | 5.2% | 11.09 | 1.96 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 89.5 | 97.1 | 8.5% | 23.00 | 23.42 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 41.7 | 44.0 | 5.6% | 35.12 | 6.28 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 265.1 | 273.6 | 3.2% | 11.11 | 1.98 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 57.8 | 58.3 | 1.0% | 25.36 | 4.54 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 263.4 | 275.1 | 4.4% | 11.18 | 1.99 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 717.8 | 809.1 | 12.7% | 6.84 | 1.46 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 38.5 | 40.8 | 6.1% | 25.36 | 3.40 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 133.6 | 137.7 | 3.1% | 11.08 | 1.96 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 61.3 | 65.3 | 6.7% | 16.81 | 17.12 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 17.2 | 17.5 | 1.7% | 42.56 | 7.62 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 135.1 | 140.0 | 3.7% | 10.90 | 1.94 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 26.5 | 29.2 | 9.9% | 27.59 | 4.94 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 135.1 | 143.5 | 6.3% | 10.90 | 1.94 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 301.3 | 313.0 | 3.9% | 8.15 | 1.74 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.5 | 2.0 | 40.0% | 20.93 | 2.81 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 2.0 | 2.0 | 0.0% | 37.36 | 4.01 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 3.8 | 4.1 | 7.7% | 36.21 | 8.64 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 1.8 | 1.8 | 0.1% | 13.08 | 2.34 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 2.0 | 2.0 | 0.1% | 29.90 | 4.01 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 2.0 | 2.0 | 0.0% | 11.21 | 2.01 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 2.3 | 2.3 | 0.0% | 26.16 | 3.51 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 2.6 | 2.6 | 0.0% | 40.69 | 6.24 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 1.5 | 1.8 | 20.0% | 10.46 | 1.40 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.5 | 1.5 | 0.1% | 26.16 | 2.81 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 2.3 | 2.6 | 12.6% | 29.43 | 7.02 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 1.5 | 2.6 | 80.0% | 7.85 | 1.40 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.8 | 1.8 | 0.1% | 17.44 | 2.34 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.8 | 1.8 | 0.1% | 6.54 | 1.17 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 1.8 | 2.0 | 16.7% | 17.44 | 2.34 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.8 | 2.0 | 16.7% | 30.52 | 4.68 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 34.0 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 22.47 | 66.1% | 86.9 | 19.1% |
| Gated Delta-Rule Scan | 11.09 | 32.6% | 266.9 | 5.2% |
| Causal Depthwise Conv1D | 23.00 | 67.6% | 89.5 | 8.5% |
| gdn_cumdecay_f16 | 35.12 | 103.3% | 41.7 | 5.6% |
| gdn_gated_scan_f16 | 11.11 | 32.7% | 265.1 | 3.2% |
| gdn_cumdecay_bf16 | 25.36 | 74.6% | 57.8 | 1.0% |
| gdn_gated_scan_bf16 | 11.18 | 32.9% | 263.4 | 4.4% |
| gdn2_gated_scan | 6.84 | 20.1% | 717.8 | 12.7% |
| Gated Cumulative Decay | 25.36 | 74.6% | 38.5 | 6.1% |
| Gated Delta-Rule Scan | 11.08 | 32.6% | 133.6 | 3.1% |
| Causal Depthwise Conv1D | 16.81 | 49.4% | 61.3 | 6.7% |
| gdn_cumdecay_f16 | 42.56 | 125.2% | 17.2 | 1.7% |
| gdn_gated_scan_f16 | 10.90 | 32.1% | 135.1 | 3.7% |
| gdn_cumdecay_bf16 | 27.59 | 81.1% | 26.5 | 9.9% |
| gdn_gated_scan_bf16 | 10.90 | 32.1% | 135.1 | 6.3% |
| gdn2_gated_scan | 8.15 | 24.0% | 301.3 | 3.9% |
| Gated Cumulative Decay | 20.93 | 61.6% | 1.5 | 40.0% |
| Gated Delta-Rule Scan | 37.36 | 109.9% | 2.0 | 0.0% |
| Causal Depthwise Conv1D | 36.21 | 106.5% | 3.8 | 7.7% |
| gdn_cumdecay_f16 | 13.08 | 38.5% | 1.8 | 0.1% |
| gdn_gated_scan_f16 | 29.90 | 87.9% | 2.0 | 0.1% |
| gdn_cumdecay_bf16 | 11.21 | 33.0% | 2.0 | 0.0% |
| gdn_gated_scan_bf16 | 26.16 | 76.9% | 2.3 | 0.0% |
| gdn2_gated_scan | 40.69 | 119.7% | 2.6 | 0.0% |
| Gated Cumulative Decay | 10.46 | 30.8% | 1.5 | 20.0% |
| Gated Delta-Rule Scan | 26.16 | 76.9% | 1.5 | 0.1% |
| Causal Depthwise Conv1D | 29.43 | 86.6% | 2.3 | 12.6% |
| gdn_cumdecay_f16 | 7.85 | 23.1% | 1.5 | 80.0% |
| gdn_gated_scan_f16 | 17.44 | 51.3% | 1.8 | 0.1% |
| gdn_cumdecay_bf16 | 6.54 | 19.2% | 1.8 | 0.1% |
| gdn_gated_scan_bf16 | 17.44 | 51.3% | 1.8 | 16.7% |
| gdn2_gated_scan | 30.52 | 89.8% | 1.8 | 16.7% |
