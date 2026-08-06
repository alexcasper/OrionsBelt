# rk3588-t4_little_singlethread — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 1,422.0 | 1,765.6 | 24.2% | 1.37 | 0.18 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 3,875.3 | 5,476.7 | 41.3% | 0.76 | 0.14 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 1,664.7 | 2,482.8 | 49.2% | 1.24 | 1.26 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 932.2 | 1,690.9 | 81.4% | 1.57 | 0.28 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 3,642.6 | 4,823.6 | 32.4% | 0.81 | 0.14 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 970.1 | 1,015.7 | 4.7% | 1.51 | 0.27 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 3,665.6 | 4,391.3 | 19.8% | 0.80 | 0.14 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 6,704.1 | 7,007.8 | 4.5% | 0.73 | 0.16 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 621.0 | 1,057.4 | 70.3% | 1.57 | 0.21 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 1,022.7 | 1,121.0 | 9.6% | 1.45 | 0.26 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 668.8 | 698.9 | 4.5% | 1.54 | 1.57 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 395.2 | 425.3 | 7.6% | 1.85 | 0.33 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 1,004.3 | 1,080.4 | 7.6% | 1.47 | 0.26 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 453.9 | 654.3 | 44.2% | 1.61 | 0.29 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 1,016.2 | 1,414.7 | 39.2% | 1.45 | 0.26 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 1,709.6 | 2,096.1 | 22.6% | 1.44 | 0.31 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 7.0 | 7.3 | 4.2% | 4.36 | 0.59 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 16.0 | 16.3 | 1.8% | 4.76 | 0.51 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 51.6 | 52.2 | 1.1% | 2.66 | 0.63 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 7.6 | 8.5 | 11.5% | 3.02 | 0.54 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 14.0 | 14.3 | 2.1% | 4.36 | 0.59 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 9.6 | 9.6 | 0.0% | 2.38 | 0.43 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 16.3 | 16.6 | 1.8% | 3.74 | 0.50 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 27.4 | 28.0 | 2.1% | 3.90 | 0.60 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 4.4 | 4.4 | 0.0% | 3.49 | 0.47 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 6.4 | 6.7 | 4.5% | 5.94 | 0.64 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 28.0 | 30.6 | 9.4% | 2.45 | 0.59 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 5.0 | 5.0 | 0.0% | 2.31 | 0.41 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 6.4 | 6.7 | 4.5% | 4.76 | 0.64 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 5.8 | 5.8 | 0.0% | 1.96 | 0.35 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 7.9 | 8.2 | 3.7% | 3.87 | 0.52 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 11.7 | 12.0 | 2.5% | 4.58 | 0.70 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 34.0 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 1.37 | 4.0% | 1,422.0 | 24.2% |
| Gated Delta-Rule Scan | 0.76 | 2.2% | 3,875.3 | 41.3% |
| Causal Depthwise Conv1D | 1.24 | 3.6% | 1,664.7 | 49.2% |
| gdn_cumdecay_f16 | 1.57 | 4.6% | 932.2 | 81.4% |
| gdn_gated_scan_f16 | 0.81 | 2.4% | 3,642.6 | 32.4% |
| gdn_cumdecay_bf16 | 1.51 | 4.4% | 970.1 | 4.7% |
| gdn_gated_scan_bf16 | 0.80 | 2.4% | 3,665.6 | 19.8% |
| gdn2_gated_scan | 0.73 | 2.1% | 6,704.1 | 4.5% |
| Gated Cumulative Decay | 1.57 | 4.6% | 621.0 | 70.3% |
| Gated Delta-Rule Scan | 1.45 | 4.3% | 1,022.7 | 9.6% |
| Causal Depthwise Conv1D | 1.54 | 4.5% | 668.8 | 4.5% |
| gdn_cumdecay_f16 | 1.85 | 5.4% | 395.2 | 7.6% |
| gdn_gated_scan_f16 | 1.47 | 4.3% | 1,004.3 | 7.6% |
| gdn_cumdecay_bf16 | 1.61 | 4.7% | 453.9 | 44.2% |
| gdn_gated_scan_bf16 | 1.45 | 4.3% | 1,016.2 | 39.2% |
| gdn2_gated_scan | 1.44 | 4.2% | 1,709.6 | 22.6% |
| Gated Cumulative Decay | 4.36 | 12.8% | 7.0 | 4.2% |
| Gated Delta-Rule Scan | 4.76 | 14.0% | 16.0 | 1.8% |
| Causal Depthwise Conv1D | 2.66 | 7.8% | 51.6 | 1.1% |
| gdn_cumdecay_f16 | 3.02 | 8.9% | 7.6 | 11.5% |
| gdn_gated_scan_f16 | 4.36 | 12.8% | 14.0 | 2.1% |
| gdn_cumdecay_bf16 | 2.38 | 7.0% | 9.6 | 0.0% |
| gdn_gated_scan_bf16 | 3.74 | 11.0% | 16.3 | 1.8% |
| gdn2_gated_scan | 3.90 | 11.5% | 27.4 | 2.1% |
| Gated Cumulative Decay | 3.49 | 10.3% | 4.4 | 0.0% |
| Gated Delta-Rule Scan | 5.94 | 17.5% | 6.4 | 4.5% |
| Causal Depthwise Conv1D | 2.45 | 7.2% | 28.0 | 9.4% |
| gdn_cumdecay_f16 | 2.31 | 6.8% | 5.0 | 0.0% |
| gdn_gated_scan_f16 | 4.76 | 14.0% | 6.4 | 4.5% |
| gdn_cumdecay_bf16 | 1.96 | 5.8% | 5.8 | 0.0% |
| gdn_gated_scan_bf16 | 3.87 | 11.4% | 7.9 | 3.7% |
| gdn2_gated_scan | 4.58 | 13.5% | 11.7 | 2.5% |
