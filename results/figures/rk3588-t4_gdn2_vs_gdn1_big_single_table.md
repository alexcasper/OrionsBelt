# rk3588-t4_gdn2_vs_gdn1_big_single — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 270.4 | 287.6 | 6.4% | 7.22 | 0.97 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 548.4 | 590.7 | 7.7% | 5.40 | 0.96 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 297.8 | 322.0 | 8.1% | 6.92 | 7.04 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 173.8 | 193.4 | 11.2% | 8.43 | 1.51 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 552.5 | 608.8 | 10.2% | 5.33 | 0.95 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 251.7 | 301.9 | 19.9% | 5.82 | 1.04 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 547.2 | 578.4 | 5.7% | 5.38 | 0.96 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 1,429.0 | 1,544.2 | 8.1% | 3.44 | 0.73 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 122.5 | 128.1 | 4.5% | 7.97 | 1.07 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 204.2 | 214.4 | 5.0% | 7.25 | 1.28 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 142.9 | 148.5 | 3.9% | 7.21 | 7.34 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 78.2 | 83.1 | 6.3% | 9.37 | 1.68 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 203.3 | 220.5 | 8.5% | 7.24 | 1.29 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 132.4 | 141.8 | 7.0% | 5.53 | 0.99 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 205.1 | 218.8 | 6.7% | 7.18 | 1.28 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 447.4 | 478.7 | 7.0% | 5.49 | 1.17 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.8 | 2.0 | 16.7% | 17.44 | 2.34 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 2.3 | 2.3 | 0.0% | 32.69 | 3.51 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 9.0 | 9.3 | 3.2% | 15.19 | 3.62 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 2.0 | 2.0 | 0.0% | 11.21 | 2.01 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 2.9 | 2.9 | 0.0% | 20.93 | 2.81 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 2.9 | 2.9 | 0.0% | 7.85 | 1.40 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 4.1 | 4.1 | 0.0% | 14.95 | 2.01 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 3.5 | 3.8 | 8.3% | 30.52 | 4.68 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 1.2 | 1.5 | 25.0% | 13.08 | 1.76 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.5 | 1.8 | 19.9% | 26.15 | 2.81 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 2.6 | 2.9 | 11.1% | 26.16 | 6.24 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 1.5 | 1.5 | 0.1% | 7.85 | 1.40 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.5 | 1.8 | 20.0% | 20.92 | 2.81 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.8 | 1.8 | 0.1% | 6.54 | 1.17 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 2.3 | 2.3 | 0.0% | 13.08 | 1.76 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.8 | 1.8 | 0.1% | 30.52 | 4.68 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 7.22 | 22.8% | 270.4 | 6.4% |
| Gated Delta-Rule Scan | 5.40 | 17.0% | 548.4 | 7.7% |
| Causal Depthwise Conv1D | 6.92 | 21.8% | 297.8 | 8.1% |
| gdn_cumdecay_f16 | 8.43 | 26.6% | 173.8 | 11.2% |
| gdn_gated_scan_f16 | 5.33 | 16.8% | 552.5 | 10.2% |
| gdn_cumdecay_bf16 | 5.82 | 18.4% | 251.7 | 19.9% |
| gdn_gated_scan_bf16 | 5.38 | 17.0% | 547.2 | 5.7% |
| gdn2_gated_scan | 3.44 | 10.9% | 1,429.0 | 8.1% |
| Gated Cumulative Decay | 7.97 | 25.1% | 122.5 | 4.5% |
| Gated Delta-Rule Scan | 7.25 | 22.9% | 204.2 | 5.0% |
| Causal Depthwise Conv1D | 7.21 | 22.7% | 142.9 | 3.9% |
| gdn_cumdecay_f16 | 9.37 | 29.6% | 78.2 | 6.3% |
| gdn_gated_scan_f16 | 7.24 | 22.8% | 203.3 | 8.5% |
| gdn_cumdecay_bf16 | 5.53 | 17.4% | 132.4 | 7.0% |
| gdn_gated_scan_bf16 | 7.18 | 22.6% | 205.1 | 6.7% |
| gdn2_gated_scan | 5.49 | 17.3% | 447.4 | 7.0% |
| Gated Cumulative Decay | 17.44 | 55.0% | 1.8 | 16.7% |
| Gated Delta-Rule Scan | 32.69 | 103.1% | 2.3 | 0.0% |
| Causal Depthwise Conv1D | 15.19 | 47.9% | 9.0 | 3.2% |
| gdn_cumdecay_f16 | 11.21 | 35.4% | 2.0 | 0.0% |
| gdn_gated_scan_f16 | 20.93 | 66.0% | 2.9 | 0.0% |
| gdn_cumdecay_bf16 | 7.85 | 24.8% | 2.9 | 0.0% |
| gdn_gated_scan_bf16 | 14.95 | 47.2% | 4.1 | 0.0% |
| gdn2_gated_scan | 30.52 | 96.3% | 3.5 | 8.3% |
| Gated Cumulative Decay | 13.08 | 41.3% | 1.2 | 25.0% |
| Gated Delta-Rule Scan | 26.15 | 82.5% | 1.5 | 19.9% |
| Causal Depthwise Conv1D | 26.16 | 82.5% | 2.6 | 11.1% |
| gdn_cumdecay_f16 | 7.85 | 24.8% | 1.5 | 0.1% |
| gdn_gated_scan_f16 | 20.92 | 66.0% | 1.5 | 20.0% |
| gdn_cumdecay_bf16 | 6.54 | 20.6% | 1.8 | 0.1% |
| gdn_gated_scan_bf16 | 13.08 | 41.3% | 2.3 | 0.0% |
| gdn2_gated_scan | 30.52 | 96.3% | 1.8 | 0.1% |
