# rk3588-t3_little — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 351.2 | 374.5 | 6.6% | 5.56 | 0.75 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 1,094.1 | 1,541.3 | 40.9% | 2.71 | 0.48 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 385.3 | 417.1 | 8.3% | 5.35 | 5.44 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 233.6 | 249.1 | 6.6% | 6.27 | 1.12 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 1,007.2 | 1,132.0 | 12.4% | 2.92 | 0.52 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 257.3 | 291.1 | 13.2% | 5.69 | 1.02 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 1,023.8 | 1,255.1 | 22.6% | 2.88 | 0.51 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 3,950.0 | 4,438.6 | 12.4% | 1.24 | 0.27 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 155.5 | 170.3 | 9.6% | 6.28 | 0.84 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 241.8 | 251.1 | 3.9% | 6.12 | 1.08 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 147.3 | 149.9 | 1.8% | 6.99 | 7.12 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 114.6 | 115.5 | 0.8% | 6.39 | 1.14 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 223.1 | 238.6 | 6.9% | 6.60 | 1.17 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 128.3 | 130.1 | 1.4% | 5.71 | 1.02 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 222.8 | 236.3 | 6.0% | 6.61 | 1.18 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 388.2 | 412.7 | 6.3% | 6.33 | 1.35 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 2.6 | 2.9 | 11.1% | 11.63 | 1.56 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 4.4 | 5.0 | 13.3% | 17.43 | 1.87 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 9.9 | 10.5 | 5.9% | 13.85 | 3.30 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 2.9 | 2.9 | 0.0% | 7.85 | 1.40 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 4.1 | 4.1 | 0.0% | 14.95 | 2.01 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 3.5 | 3.5 | 0.0% | 6.54 | 1.17 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 4.7 | 4.7 | 0.0% | 13.08 | 1.76 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 6.7 | 7.0 | 4.4% | 15.92 | 2.44 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 2.0 | 2.3 | 14.3% | 7.47 | 1.00 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 3.2 | 3.2 | 0.0% | 11.89 | 1.28 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 8.5 | 8.8 | 3.4% | 8.12 | 1.94 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 2.3 | 2.3 | 0.0% | 4.91 | 0.88 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 2.9 | 2.9 | 0.0% | 10.46 | 1.40 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 2.6 | 2.6 | 0.0% | 4.36 | 0.78 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 3.2 | 3.2 | 0.0% | 9.51 | 1.28 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 4.1 | 4.4 | 7.1% | 13.08 | 2.01 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 5.56 | 17.5% | 351.2 | 6.6% |
| Gated Delta-Rule Scan | 2.71 | 8.5% | 1,094.1 | 40.9% |
| Causal Depthwise Conv1D | 5.35 | 16.9% | 385.3 | 8.3% |
| gdn_cumdecay_f16 | 6.27 | 19.8% | 233.6 | 6.6% |
| gdn_gated_scan_f16 | 2.92 | 9.2% | 1,007.2 | 12.4% |
| gdn_cumdecay_bf16 | 5.69 | 17.9% | 257.3 | 13.2% |
| gdn_gated_scan_bf16 | 2.88 | 9.1% | 1,023.8 | 22.6% |
| gdn2_gated_scan | 1.24 | 3.9% | 3,950.0 | 12.4% |
| Gated Cumulative Decay | 6.28 | 19.8% | 155.5 | 9.6% |
| Gated Delta-Rule Scan | 6.12 | 19.3% | 241.8 | 3.9% |
| Causal Depthwise Conv1D | 6.99 | 22.1% | 147.3 | 1.8% |
| gdn_cumdecay_f16 | 6.39 | 20.2% | 114.6 | 0.8% |
| gdn_gated_scan_f16 | 6.60 | 20.8% | 223.1 | 6.9% |
| gdn_cumdecay_bf16 | 5.71 | 18.0% | 128.3 | 1.4% |
| gdn_gated_scan_bf16 | 6.61 | 20.9% | 222.8 | 6.0% |
| gdn2_gated_scan | 6.33 | 20.0% | 388.2 | 6.3% |
| Gated Cumulative Decay | 11.63 | 36.7% | 2.6 | 11.1% |
| Gated Delta-Rule Scan | 17.43 | 55.0% | 4.4 | 13.3% |
| Causal Depthwise Conv1D | 13.85 | 43.7% | 9.9 | 5.9% |
| gdn_cumdecay_f16 | 7.85 | 24.8% | 2.9 | 0.0% |
| gdn_gated_scan_f16 | 14.95 | 47.2% | 4.1 | 0.0% |
| gdn_cumdecay_bf16 | 6.54 | 20.6% | 3.5 | 0.0% |
| gdn_gated_scan_bf16 | 13.08 | 41.3% | 4.7 | 0.0% |
| gdn2_gated_scan | 15.92 | 50.2% | 6.7 | 4.4% |
| Gated Cumulative Decay | 7.47 | 23.6% | 2.0 | 14.3% |
| Gated Delta-Rule Scan | 11.89 | 37.5% | 3.2 | 0.0% |
| Causal Depthwise Conv1D | 8.12 | 25.6% | 8.5 | 3.4% |
| gdn_cumdecay_f16 | 4.91 | 15.5% | 2.3 | 0.0% |
| gdn_gated_scan_f16 | 10.46 | 33.0% | 2.9 | 0.0% |
| gdn_cumdecay_bf16 | 4.36 | 13.8% | 2.6 | 0.0% |
| gdn_gated_scan_bf16 | 9.51 | 30.0% | 3.2 | 0.0% |
| gdn2_gated_scan | 13.08 | 41.3% | 4.1 | 7.1% |
