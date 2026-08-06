# jetson-j1_omp — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 502.2 | 556.4 | 10.8% | 3.89 | 0.52 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 993.2 | 1,102.6 | 11.0% | 2.98 | 0.53 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 560.7 | 766.5 | 36.7% | 3.67 | 3.74 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 352.9 | 396.5 | 12.4% | 4.15 | 0.74 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 990.5 | 1,110.9 | 12.1% | 2.97 | 0.53 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 366.5 | 396.4 | 8.2% | 4.00 | 0.72 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 1,017.0 | 1,165.0 | 14.5% | 2.90 | 0.52 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 1,671.6 | 1,908.5 | 14.2% | 2.94 | 0.63 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 154.0 | 178.1 | 15.6% | 6.34 | 0.85 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 240.1 | 325.4 | 35.5% | 6.17 | 1.09 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 135.8 | 147.0 | 8.2% | 7.59 | 7.72 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 185.2 | 200.2 | 8.1% | 3.95 | 0.71 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 237.3 | 277.6 | 17.0% | 6.21 | 1.10 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 182.9 | 195.7 | 7.0% | 4.01 | 0.72 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 236.5 | 249.3 | 5.4% | 6.23 | 1.11 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 430.7 | 477.7 | 10.9% | 5.70 | 1.22 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 4.4 | 4.9 | 10.6% | 6.89 | 0.93 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 4.7 | 4.8 | 2.2% | 16.27 | 1.75 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 11.4 | 14.1 | 23.9% | 12.10 | 2.89 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 4.2 | 44.7 | 960.7% | 5.43 | 0.97 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 5.4 | 5.5 | 1.9% | 11.27 | 1.51 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 4.7 | 4.8 | 2.2% | 4.88 | 0.87 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 5.7 | 7.2 | 26.6% | 10.75 | 1.44 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 5.6 | 5.8 | 2.8% | 18.99 | 2.91 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 3.8 | 4.0 | 4.1% | 4.01 | 0.54 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 3.9 | 4.1 | 4.0% | 9.77 | 1.05 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 7.4 | 7.7 | 3.5% | 9.28 | 2.22 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 3.7 | 3.8 | 1.4% | 3.09 | 0.55 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 4.4 | 4.5 | 3.6% | 6.98 | 0.94 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 4.0 | 4.1 | 2.6% | 2.85 | 0.51 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 4.5 | 4.6 | 3.5% | 6.81 | 0.91 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 4.5 | 4.7 | 4.6% | 11.78 | 1.81 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 25.6 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 3.89 | 15.2% | 502.2 | 10.8% |
| Gated Delta-Rule Scan | 2.98 | 11.6% | 993.2 | 11.0% |
| Causal Depthwise Conv1D | 3.67 | 14.3% | 560.7 | 36.7% |
| gdn_cumdecay_f16 | 4.15 | 16.2% | 352.9 | 12.4% |
| gdn_gated_scan_f16 | 2.97 | 11.6% | 990.5 | 12.1% |
| gdn_cumdecay_bf16 | 4.00 | 15.6% | 366.5 | 8.2% |
| gdn_gated_scan_bf16 | 2.90 | 11.3% | 1,017.0 | 14.5% |
| gdn2_gated_scan | 2.94 | 11.5% | 1,671.6 | 14.2% |
| Gated Cumulative Decay | 6.34 | 24.8% | 154.0 | 15.6% |
| Gated Delta-Rule Scan | 6.17 | 24.1% | 240.1 | 35.5% |
| Causal Depthwise Conv1D | 7.59 | 29.6% | 135.8 | 8.2% |
| gdn_cumdecay_f16 | 3.95 | 15.4% | 185.2 | 8.1% |
| gdn_gated_scan_f16 | 6.21 | 24.3% | 237.3 | 17.0% |
| gdn_cumdecay_bf16 | 4.01 | 15.7% | 182.9 | 7.0% |
| gdn_gated_scan_bf16 | 6.23 | 24.3% | 236.5 | 5.4% |
| gdn2_gated_scan | 5.70 | 22.3% | 430.7 | 10.9% |
| Gated Cumulative Decay | 6.89 | 26.9% | 4.4 | 10.6% |
| Gated Delta-Rule Scan | 16.27 | 63.6% | 4.7 | 2.2% |
| Causal Depthwise Conv1D | 12.10 | 47.3% | 11.4 | 23.9% |
| gdn_cumdecay_f16 | 5.43 | 21.2% | 4.2 | 960.7% |
| gdn_gated_scan_f16 | 11.27 | 44.0% | 5.4 | 1.9% |
| gdn_cumdecay_bf16 | 4.88 | 19.1% | 4.7 | 2.2% |
| gdn_gated_scan_bf16 | 10.75 | 42.0% | 5.7 | 26.6% |
| gdn2_gated_scan | 18.99 | 74.2% | 5.6 | 2.8% |
| Gated Cumulative Decay | 4.01 | 15.7% | 3.8 | 4.1% |
| Gated Delta-Rule Scan | 9.77 | 38.2% | 3.9 | 4.0% |
| Causal Depthwise Conv1D | 9.28 | 36.2% | 7.4 | 3.5% |
| gdn_cumdecay_f16 | 3.09 | 12.1% | 3.7 | 1.4% |
| gdn_gated_scan_f16 | 6.98 | 27.3% | 4.4 | 3.6% |
| gdn_cumdecay_bf16 | 2.85 | 11.1% | 4.0 | 2.6% |
| gdn_gated_scan_bf16 | 6.81 | 26.6% | 4.5 | 3.5% |
| gdn2_gated_scan | 11.78 | 46.0% | 4.5 | 4.6% |
