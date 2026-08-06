# rk3588-t4_big — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 80.5 | 89.3 | 10.9% | 24.26 | 3.26 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 257.9 | 277.1 | 7.5% | 11.48 | 2.03 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 98.0 | 101.5 | 3.6% | 21.02 | 21.40 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 42.3 | 43.2 | 2.1% | 34.63 | 6.20 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 254.6 | 277.7 | 9.0% | 11.57 | 2.06 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 58.3 | 59.2 | 1.5% | 25.11 | 4.49 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 254.4 | 274.2 | 7.8% | 11.58 | 2.06 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 453.9 | 474.9 | 4.6% | 10.83 | 2.31 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 33.3 | 33.8 | 1.8% | 29.37 | 3.94 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 124.6 | 130.1 | 4.4% | 11.88 | 2.10 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 47.3 | 50.2 | 6.2% | 21.80 | 22.19 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 17.8 | 18.1 | 1.6% | 41.16 | 7.37 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 135.3 | 142.6 | 5.4% | 10.88 | 1.94 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 28.0 | 28.3 | 1.0% | 26.16 | 4.68 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 137.7 | 143.8 | 4.4% | 10.70 | 1.90 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 227.8 | 234.8 | 3.1% | 10.78 | 2.30 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.5 | 1.8 | 20.0% | 20.92 | 2.81 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 2.0 | 2.0 | 0.0% | 37.36 | 4.01 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 3.2 | 3.2 | 0.0% | 42.81 | 10.21 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 1.8 | 1.8 | 0.1% | 13.08 | 2.34 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 1.8 | 2.0 | 16.7% | 34.88 | 4.68 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 1.8 | 2.0 | 16.6% | 13.07 | 2.34 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 2.0 | 2.3 | 14.3% | 29.89 | 4.01 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 2.3 | 2.3 | 0.0% | 45.78 | 7.02 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 1.5 | 1.5 | 0.1% | 10.46 | 1.40 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.5 | 1.8 | 20.0% | 26.15 | 2.81 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 2.3 | 2.6 | 12.5% | 29.43 | 7.02 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 1.5 | 1.8 | 20.0% | 7.84 | 1.40 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.5 | 1.8 | 20.0% | 20.91 | 2.81 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.5 | 1.8 | 20.0% | 7.84 | 1.40 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 1.8 | 2.0 | 16.7% | 17.44 | 2.34 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.5 | 1.8 | 20.0% | 36.61 | 5.62 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 34.0 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 24.26 | 71.4% | 80.5 | 10.9% |
| Gated Delta-Rule Scan | 11.48 | 33.8% | 257.9 | 7.5% |
| Causal Depthwise Conv1D | 21.02 | 61.8% | 98.0 | 3.6% |
| gdn_cumdecay_f16 | 34.63 | 101.9% | 42.3 | 2.1% |
| gdn_gated_scan_f16 | 11.57 | 34.0% | 254.6 | 9.0% |
| gdn_cumdecay_bf16 | 25.11 | 73.9% | 58.3 | 1.5% |
| gdn_gated_scan_bf16 | 11.58 | 34.1% | 254.4 | 7.8% |
| gdn2_gated_scan | 10.83 | 31.9% | 453.9 | 4.6% |
| Gated Cumulative Decay | 29.37 | 86.4% | 33.3 | 1.8% |
| Gated Delta-Rule Scan | 11.88 | 34.9% | 124.6 | 4.4% |
| Causal Depthwise Conv1D | 21.80 | 64.1% | 47.3 | 6.2% |
| gdn_cumdecay_f16 | 41.16 | 121.1% | 17.8 | 1.6% |
| gdn_gated_scan_f16 | 10.88 | 32.0% | 135.3 | 5.4% |
| gdn_cumdecay_bf16 | 26.16 | 76.9% | 28.0 | 1.0% |
| gdn_gated_scan_bf16 | 10.70 | 31.5% | 137.7 | 4.4% |
| gdn2_gated_scan | 10.78 | 31.7% | 227.8 | 3.1% |
| Gated Cumulative Decay | 20.92 | 61.5% | 1.5 | 20.0% |
| Gated Delta-Rule Scan | 37.36 | 109.9% | 2.0 | 0.0% |
| Causal Depthwise Conv1D | 42.81 | 125.9% | 3.2 | 0.0% |
| gdn_cumdecay_f16 | 13.08 | 38.5% | 1.8 | 0.1% |
| gdn_gated_scan_f16 | 34.88 | 102.6% | 1.8 | 16.7% |
| gdn_cumdecay_bf16 | 13.07 | 38.4% | 1.8 | 16.6% |
| gdn_gated_scan_bf16 | 29.89 | 87.9% | 2.0 | 14.3% |
| gdn2_gated_scan | 45.78 | 134.6% | 2.3 | 0.0% |
| Gated Cumulative Decay | 10.46 | 30.8% | 1.5 | 0.1% |
| Gated Delta-Rule Scan | 26.15 | 76.9% | 1.5 | 20.0% |
| Causal Depthwise Conv1D | 29.43 | 86.6% | 2.3 | 12.5% |
| gdn_cumdecay_f16 | 7.84 | 23.1% | 1.5 | 20.0% |
| gdn_gated_scan_f16 | 20.91 | 61.5% | 1.5 | 20.0% |
| gdn_cumdecay_bf16 | 7.84 | 23.1% | 1.5 | 20.0% |
| gdn_gated_scan_bf16 | 17.44 | 51.3% | 1.8 | 16.7% |
| gdn2_gated_scan | 36.61 | 107.7% | 1.5 | 20.0% |
