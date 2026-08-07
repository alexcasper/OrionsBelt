# rk3588-t4-big — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 264.0 | 277.1 | 5.0% | 7.40 | 0.99 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 522.4 | 560.9 | 7.4% | 5.67 | 1.00 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 292.6 | 305.1 | 4.3% | 7.04 | 7.17 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 170.1 | 172.7 | 1.5% | 8.61 | 1.54 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 520.4 | 545.7 | 4.9% | 5.66 | 1.01 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 243.6 | 248.2 | 1.9% | 6.01 | 1.08 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 519.5 | 544.9 | 4.9% | 5.67 | 1.01 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 1,534.3 | 1,559.9 | 1.7% | 3.20 | 0.68 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 119.9 | 120.8 | 0.7% | 8.15 | 1.09 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 213.5 | 225.8 | 5.7% | 6.93 | 1.23 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 151.1 | 157.8 | 4.4% | 6.82 | 6.94 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 78.2 | 79.3 | 1.5% | 9.37 | 1.68 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 213.5 | 227.5 | 6.6% | 6.90 | 1.23 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 133.6 | 137.7 | 3.1% | 5.48 | 0.98 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 215.0 | 217.9 | 1.4% | 6.85 | 1.22 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 390.3 | 403.7 | 3.4% | 6.29 | 1.34 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.8 | 2.0 | 16.7% | 17.44 | 2.34 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 2.3 | 2.3 | 0.0% | 32.70 | 3.51 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 9.0 | 9.3 | 3.2% | 15.19 | 3.62 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 2.0 | 2.0 | 0.0% | 11.21 | 2.01 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 2.6 | 2.9 | 11.1% | 23.25 | 3.12 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 2.9 | 2.9 | 0.0% | 7.85 | 1.40 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 4.1 | 4.1 | 0.0% | 14.95 | 2.01 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 3.5 | 3.8 | 8.3% | 30.52 | 4.68 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 1.2 | 1.5 | 24.9% | 13.08 | 1.76 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.5 | 1.8 | 19.9% | 26.15 | 2.81 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 2.6 | 2.9 | 11.1% | 26.16 | 6.24 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 1.5 | 1.5 | 0.1% | 7.85 | 1.40 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.8 | 1.8 | 0.1% | 17.44 | 2.34 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.8 | 1.8 | 0.1% | 6.54 | 1.17 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 2.3 | 2.6 | 12.5% | 13.08 | 1.76 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.8 | 1.8 | 0.1% | 30.52 | 4.68 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 34.0 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 7.40 | 21.8% | 264.0 | 5.0% |
| Gated Delta-Rule Scan | 5.67 | 16.7% | 522.4 | 7.4% |
| Causal Depthwise Conv1D | 7.04 | 20.7% | 292.6 | 4.3% |
| gdn_cumdecay_f16 | 8.61 | 25.3% | 170.1 | 1.5% |
| gdn_gated_scan_f16 | 5.66 | 16.6% | 520.4 | 4.9% |
| gdn_cumdecay_bf16 | 6.01 | 17.7% | 243.6 | 1.9% |
| gdn_gated_scan_bf16 | 5.67 | 16.7% | 519.5 | 4.9% |
| gdn2_gated_scan | 3.20 | 9.4% | 1,534.3 | 1.7% |
| Gated Cumulative Decay | 8.15 | 24.0% | 119.9 | 0.7% |
| Gated Delta-Rule Scan | 6.93 | 20.4% | 213.5 | 5.7% |
| Causal Depthwise Conv1D | 6.82 | 20.1% | 151.1 | 4.4% |
| gdn_cumdecay_f16 | 9.37 | 27.6% | 78.2 | 1.5% |
| gdn_gated_scan_f16 | 6.90 | 20.3% | 213.5 | 6.6% |
| gdn_cumdecay_bf16 | 5.48 | 16.1% | 133.6 | 3.1% |
| gdn_gated_scan_bf16 | 6.85 | 20.1% | 215.0 | 1.4% |
| gdn2_gated_scan | 6.29 | 18.5% | 390.3 | 3.4% |
| Gated Cumulative Decay | 17.44 | 51.3% | 1.8 | 16.7% |
| Gated Delta-Rule Scan | 32.70 | 96.2% | 2.3 | 0.0% |
| Causal Depthwise Conv1D | 15.19 | 44.7% | 9.0 | 3.2% |
| gdn_cumdecay_f16 | 11.21 | 33.0% | 2.0 | 0.0% |
| gdn_gated_scan_f16 | 23.25 | 68.4% | 2.6 | 11.1% |
| gdn_cumdecay_bf16 | 7.85 | 23.1% | 2.9 | 0.0% |
| gdn_gated_scan_bf16 | 14.95 | 44.0% | 4.1 | 0.0% |
| gdn2_gated_scan | 30.52 | 89.8% | 3.5 | 8.3% |
| Gated Cumulative Decay | 13.08 | 38.5% | 1.2 | 24.9% |
| Gated Delta-Rule Scan | 26.15 | 76.9% | 1.5 | 19.9% |
| Causal Depthwise Conv1D | 26.16 | 76.9% | 2.6 | 11.1% |
| gdn_cumdecay_f16 | 7.85 | 23.1% | 1.5 | 0.1% |
| gdn_gated_scan_f16 | 17.44 | 51.3% | 1.8 | 0.1% |
| gdn_cumdecay_bf16 | 6.54 | 19.2% | 1.8 | 0.1% |
| gdn_gated_scan_bf16 | 13.08 | 38.5% | 2.3 | 12.5% |
| gdn2_gated_scan | 30.52 | 89.8% | 1.8 | 0.1% |
