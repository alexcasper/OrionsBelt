# rk3588-t4_big_singlethread — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 259.0 | 276.5 | 6.8% | 7.54 | 1.01 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 504.6 | 554.5 | 9.9% | 5.87 | 1.04 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 289.4 | 306.9 | 6.0% | 7.12 | 7.25 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 169.5 | 177.9 | 5.0% | 8.64 | 1.55 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 505.2 | 532.3 | 5.4% | 5.83 | 1.04 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 240.6 | 247.1 | 2.7% | 6.09 | 1.09 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 510.5 | 549.2 | 7.6% | 5.77 | 1.03 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 1,472.7 | 1,502.5 | 2.0% | 3.34 | 0.71 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 128.9 | 129.5 | 0.5% | 7.57 | 1.02 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 184.3 | 194.8 | 5.7% | 8.03 | 1.42 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 128.6 | 130.4 | 1.4% | 8.01 | 8.15 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 78.2 | 78.8 | 0.7% | 9.37 | 1.68 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 184.3 | 193.4 | 4.9% | 7.99 | 1.42 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 113.5 | 114.6 | 1.0% | 6.45 | 1.16 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 185.5 | 189.9 | 2.4% | 7.94 | 1.41 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 399.9 | 417.7 | 4.4% | 6.14 | 1.31 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.8 | 1.8 | 0.2% | 17.01 | 2.28 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 2.3 | 2.3 | 2.4% | 33.36 | 3.58 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 9.3 | 9.4 | 0.9% | 14.80 | 3.53 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 2.0 | 2.1 | 3.0% | 11.37 | 2.04 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 2.7 | 2.7 | 1.8% | 22.74 | 3.05 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 2.8 | 2.9 | 1.8% | 8.14 | 1.46 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 4.0 | 4.1 | 1.2% | 15.17 | 2.04 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 3.5 | 3.5 | 1.2% | 30.57 | 4.69 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 1.2 | 1.2 | 0.2% | 12.64 | 1.70 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.5 | 1.5 | 0.2% | 26.05 | 2.80 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 2.6 | 2.6 | 2.0% | 26.54 | 6.33 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 1.3 | 1.3 | 0.2% | 8.64 | 1.55 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.6 | 1.7 | 2.8% | 18.62 | 2.50 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.7 | 1.8 | 2.5% | 6.63 | 1.19 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 2.3 | 2.4 | 1.9% | 13.06 | 1.75 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.7 | 1.8 | 2.7% | 31.03 | 4.76 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 7.54 | 23.8% | 259.0 | 6.8% |
| Gated Delta-Rule Scan | 5.87 | 18.5% | 504.6 | 9.9% |
| Causal Depthwise Conv1D | 7.12 | 22.5% | 289.4 | 6.0% |
| gdn_cumdecay_f16 | 8.64 | 27.3% | 169.5 | 5.0% |
| gdn_gated_scan_f16 | 5.83 | 18.4% | 505.2 | 5.4% |
| gdn_cumdecay_bf16 | 6.09 | 19.2% | 240.6 | 2.7% |
| gdn_gated_scan_bf16 | 5.77 | 18.2% | 510.5 | 7.6% |
| gdn2_gated_scan | 3.34 | 10.5% | 1,472.7 | 2.0% |
| Gated Cumulative Decay | 7.57 | 23.9% | 128.9 | 0.5% |
| Gated Delta-Rule Scan | 8.03 | 25.3% | 184.3 | 5.7% |
| Causal Depthwise Conv1D | 8.01 | 25.3% | 128.6 | 1.4% |
| gdn_cumdecay_f16 | 9.37 | 29.6% | 78.2 | 0.7% |
| gdn_gated_scan_f16 | 7.99 | 25.2% | 184.3 | 4.9% |
| gdn_cumdecay_bf16 | 6.45 | 20.3% | 113.5 | 1.0% |
| gdn_gated_scan_bf16 | 7.94 | 25.0% | 185.5 | 2.4% |
| gdn2_gated_scan | 6.14 | 19.4% | 399.9 | 4.4% |
| Gated Cumulative Decay | 17.01 | 53.7% | 1.8 | 0.2% |
| Gated Delta-Rule Scan | 33.36 | 105.2% | 2.3 | 2.4% |
| Causal Depthwise Conv1D | 14.80 | 46.7% | 9.3 | 0.9% |
| gdn_cumdecay_f16 | 11.37 | 35.9% | 2.0 | 3.0% |
| gdn_gated_scan_f16 | 22.74 | 71.7% | 2.7 | 1.8% |
| gdn_cumdecay_bf16 | 8.14 | 25.7% | 2.8 | 1.8% |
| gdn_gated_scan_bf16 | 15.17 | 47.9% | 4.0 | 1.2% |
| gdn2_gated_scan | 30.57 | 96.4% | 3.5 | 1.2% |
| Gated Cumulative Decay | 12.64 | 39.9% | 1.2 | 0.2% |
| Gated Delta-Rule Scan | 26.05 | 82.2% | 1.5 | 0.2% |
| Causal Depthwise Conv1D | 26.54 | 83.7% | 2.6 | 2.0% |
| gdn_cumdecay_f16 | 8.64 | 27.3% | 1.3 | 0.2% |
| gdn_gated_scan_f16 | 18.62 | 58.7% | 1.6 | 2.8% |
| gdn_cumdecay_bf16 | 6.63 | 20.9% | 1.7 | 2.5% |
| gdn_gated_scan_bf16 | 13.06 | 41.2% | 2.3 | 1.9% |
| gdn2_gated_scan | 31.03 | 97.9% | 1.7 | 2.7% |
