# rk3588-t4_little — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 333.1 | 358.2 | 7.5% | 5.86 | 0.79 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 809.4 | 947.4 | 17.0% | 3.66 | 0.65 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 394.1 | 432.3 | 9.7% | 5.23 | 5.32 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 221.1 | 238.3 | 7.8% | 6.63 | 1.19 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 763.6 | 817.9 | 7.1% | 3.86 | 0.69 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 247.1 | 263.4 | 6.6% | 5.93 | 1.06 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 768.9 | 854.6 | 11.2% | 3.83 | 0.68 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 2,906.1 | 3,316.8 | 14.1% | 1.69 | 0.36 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 152.8 | 182.0 | 19.1% | 6.39 | 0.86 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 253.8 | 273.0 | 7.6% | 5.83 | 1.03 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 143.2 | 150.2 | 4.9% | 7.19 | 7.32 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 112.9 | 113.8 | 0.8% | 6.49 | 1.16 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 253.8 | 269.2 | 6.1% | 5.80 | 1.03 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 126.3 | 142.3 | 12.7% | 5.80 | 1.04 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 253.8 | 267.2 | 5.3% | 5.80 | 1.03 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 570.5 | 649.0 | 13.8% | 4.31 | 0.92 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 3.8 | 4.1 | 7.7% | 8.05 | 1.08 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 5.7 | 5.9 | 3.2% | 13.31 | 1.43 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 14.9 | 15.5 | 3.9% | 9.23 | 2.20 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 4.7 | 4.8 | 3.7% | 4.92 | 0.88 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 5.4 | 5.7 | 5.4% | 11.22 | 1.51 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 5.1 | 5.3 | 2.7% | 4.45 | 0.80 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 6.1 | 6.2 | 2.6% | 10.06 | 1.35 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 9.8 | 10.0 | 1.6% | 10.87 | 1.67 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 3.7 | 3.8 | 3.9% | 4.14 | 0.56 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 4.5 | 4.6 | 3.3% | 8.56 | 0.92 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 9.7 | 9.9 | 2.4% | 7.07 | 1.69 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 3.5 | 4.0 | 12.8% | 3.24 | 0.58 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 4.3 | 4.5 | 3.5% | 7.07 | 0.95 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 4.2 | 4.3 | 3.4% | 2.76 | 0.49 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 4.5 | 4.7 | 3.0% | 6.72 | 0.90 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 6.5 | 6.7 | 2.1% | 8.20 | 1.26 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 5.86 | 18.5% | 333.1 | 7.5% |
| Gated Delta-Rule Scan | 3.66 | 11.5% | 809.4 | 17.0% |
| Causal Depthwise Conv1D | 5.23 | 16.5% | 394.1 | 9.7% |
| gdn_cumdecay_f16 | 6.63 | 20.9% | 221.1 | 7.8% |
| gdn_gated_scan_f16 | 3.86 | 12.2% | 763.6 | 7.1% |
| gdn_cumdecay_bf16 | 5.93 | 18.7% | 247.1 | 6.6% |
| gdn_gated_scan_bf16 | 3.83 | 12.1% | 768.9 | 11.2% |
| gdn2_gated_scan | 1.69 | 5.3% | 2,906.1 | 14.1% |
| Gated Cumulative Decay | 6.39 | 20.2% | 152.8 | 19.1% |
| Gated Delta-Rule Scan | 5.83 | 18.4% | 253.8 | 7.6% |
| Causal Depthwise Conv1D | 7.19 | 22.7% | 143.2 | 4.9% |
| gdn_cumdecay_f16 | 6.49 | 20.5% | 112.9 | 0.8% |
| gdn_gated_scan_f16 | 5.80 | 18.3% | 253.8 | 6.1% |
| gdn_cumdecay_bf16 | 5.80 | 18.3% | 126.3 | 12.7% |
| gdn_gated_scan_bf16 | 5.80 | 18.3% | 253.8 | 5.3% |
| gdn2_gated_scan | 4.31 | 13.6% | 570.5 | 13.8% |
| Gated Cumulative Decay | 8.05 | 25.4% | 3.8 | 7.7% |
| Gated Delta-Rule Scan | 13.31 | 42.0% | 5.7 | 3.2% |
| Causal Depthwise Conv1D | 9.23 | 29.1% | 14.9 | 3.9% |
| gdn_cumdecay_f16 | 4.92 | 15.5% | 4.7 | 3.7% |
| gdn_gated_scan_f16 | 11.22 | 35.4% | 5.4 | 5.4% |
| gdn_cumdecay_bf16 | 4.45 | 14.0% | 5.1 | 2.7% |
| gdn_gated_scan_bf16 | 10.06 | 31.7% | 6.1 | 2.6% |
| gdn2_gated_scan | 10.87 | 34.3% | 9.8 | 1.6% |
| Gated Cumulative Decay | 4.14 | 13.1% | 3.7 | 3.9% |
| Gated Delta-Rule Scan | 8.56 | 27.0% | 4.5 | 3.3% |
| Causal Depthwise Conv1D | 7.07 | 22.3% | 9.7 | 2.4% |
| gdn_cumdecay_f16 | 3.24 | 10.2% | 3.5 | 12.8% |
| gdn_gated_scan_f16 | 7.07 | 22.3% | 4.3 | 3.5% |
| gdn_cumdecay_bf16 | 2.76 | 8.7% | 4.2 | 3.4% |
| gdn_gated_scan_bf16 | 6.72 | 21.2% | 4.5 | 3.0% |
| gdn2_gated_scan | 8.20 | 25.9% | 6.5 | 2.1% |
