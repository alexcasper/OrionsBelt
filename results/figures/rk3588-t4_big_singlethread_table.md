# rk3588-t4_big_singlethread — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 271.0 | 292.3 | 7.9% | 7.21 | 0.97 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 559.5 | 622.2 | 11.2% | 5.29 | 0.94 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 324.4 | 352.6 | 8.7% | 6.35 | 6.47 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 171.5 | 176.2 | 2.7% | 8.54 | 1.53 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 558.6 | 689.0 | 23.3% | 5.27 | 0.94 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 247.1 | 315.9 | 27.9% | 5.93 | 1.06 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 548.7 | 660.1 | 20.3% | 5.37 | 0.96 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 1,179.9 | 1,318.1 | 11.7% | 4.16 | 0.89 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 124.0 | 127.2 | 2.6% | 7.88 | 1.06 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 208.3 | 221.4 | 6.3% | 7.11 | 1.26 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 131.3 | 140.6 | 7.1% | 7.85 | 7.99 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 78.2 | 80.2 | 2.6% | 9.37 | 1.68 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 209.7 | 222.0 | 5.8% | 7.02 | 1.25 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 129.5 | 131.0 | 1.1% | 5.66 | 1.01 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 212.1 | 225.2 | 6.2% | 6.94 | 1.24 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 324.1 | 340.1 | 5.0% | 7.58 | 1.62 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.8 | 2.0 | 16.7% | 17.44 | 2.34 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 2.3 | 2.3 | 0.0% | 32.70 | 3.51 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 7.0 | 7.3 | 4.2% | 19.62 | 4.68 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 2.0 | 2.0 | 0.0% | 11.21 | 2.01 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 2.9 | 2.9 | 0.0% | 20.93 | 2.81 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 2.6 | 2.9 | 11.1% | 8.72 | 1.56 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 4.1 | 4.1 | 0.0% | 14.95 | 2.01 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 2.9 | 3.2 | 10.0% | 36.62 | 5.62 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 1.2 | 1.5 | 25.0% | 13.08 | 1.76 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.5 | 1.8 | 19.9% | 26.15 | 2.81 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 2.9 | 3.2 | 10.0% | 23.54 | 5.62 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 1.5 | 1.5 | 0.1% | 7.85 | 1.40 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.8 | 1.8 | 0.1% | 17.44 | 2.34 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.8 | 1.8 | 0.1% | 6.54 | 1.17 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 2.3 | 2.3 | 0.0% | 13.08 | 1.76 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.8 | 1.8 | 0.1% | 30.52 | 4.68 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 34.0 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 7.21 | 21.2% | 271.0 | 7.9% |
| Gated Delta-Rule Scan | 5.29 | 15.6% | 559.5 | 11.2% |
| Causal Depthwise Conv1D | 6.35 | 18.7% | 324.4 | 8.7% |
| gdn_cumdecay_f16 | 8.54 | 25.1% | 171.5 | 2.7% |
| gdn_gated_scan_f16 | 5.27 | 15.5% | 558.6 | 23.3% |
| gdn_cumdecay_bf16 | 5.93 | 17.4% | 247.1 | 27.9% |
| gdn_gated_scan_bf16 | 5.37 | 15.8% | 548.7 | 20.3% |
| gdn2_gated_scan | 4.16 | 12.2% | 1,179.9 | 11.7% |
| Gated Cumulative Decay | 7.88 | 23.2% | 124.0 | 2.6% |
| Gated Delta-Rule Scan | 7.11 | 20.9% | 208.3 | 6.3% |
| Causal Depthwise Conv1D | 7.85 | 23.1% | 131.3 | 7.1% |
| gdn_cumdecay_f16 | 9.37 | 27.6% | 78.2 | 2.6% |
| gdn_gated_scan_f16 | 7.02 | 20.6% | 209.7 | 5.8% |
| gdn_cumdecay_bf16 | 5.66 | 16.6% | 129.5 | 1.1% |
| gdn_gated_scan_bf16 | 6.94 | 20.4% | 212.1 | 6.2% |
| gdn2_gated_scan | 7.58 | 22.3% | 324.1 | 5.0% |
| Gated Cumulative Decay | 17.44 | 51.3% | 1.8 | 16.7% |
| Gated Delta-Rule Scan | 32.70 | 96.2% | 2.3 | 0.0% |
| Causal Depthwise Conv1D | 19.62 | 57.7% | 7.0 | 4.2% |
| gdn_cumdecay_f16 | 11.21 | 33.0% | 2.0 | 0.0% |
| gdn_gated_scan_f16 | 20.93 | 61.6% | 2.9 | 0.0% |
| gdn_cumdecay_bf16 | 8.72 | 25.6% | 2.6 | 11.1% |
| gdn_gated_scan_bf16 | 14.95 | 44.0% | 4.1 | 0.0% |
| gdn2_gated_scan | 36.62 | 107.7% | 2.9 | 10.0% |
| Gated Cumulative Decay | 13.08 | 38.5% | 1.2 | 25.0% |
| Gated Delta-Rule Scan | 26.15 | 76.9% | 1.5 | 19.9% |
| Causal Depthwise Conv1D | 23.54 | 69.2% | 2.9 | 10.0% |
| gdn_cumdecay_f16 | 7.85 | 23.1% | 1.5 | 0.1% |
| gdn_gated_scan_f16 | 17.44 | 51.3% | 1.8 | 0.1% |
| gdn_cumdecay_bf16 | 6.54 | 19.2% | 1.8 | 0.1% |
| gdn_gated_scan_bf16 | 13.08 | 38.5% | 2.3 | 0.0% |
| gdn2_gated_scan | 30.52 | 89.8% | 1.8 | 0.1% |
