# rk3588-t4_big_singlethread — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 277.4 | 305.7 | 10.2% | 7.04 | 0.95 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 515.4 | 557.7 | 8.2% | 5.74 | 1.02 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 297.2 | 324.9 | 9.3% | 6.93 | 7.06 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 175.3 | 184.3 | 5.2% | 8.36 | 1.50 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 508.4 | 543.7 | 6.9% | 5.79 | 1.03 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 250.0 | 262.8 | 5.1% | 5.86 | 1.05 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 508.4 | 551.9 | 8.5% | 5.79 | 1.03 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 1,587.9 | 1,616.8 | 1.8% | 3.09 | 0.66 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 121.3 | 130.7 | 7.7% | 8.05 | 1.08 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 214.7 | 227.5 | 6.0% | 6.89 | 1.22 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 145.8 | 153.7 | 5.4% | 7.06 | 7.19 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 77.3 | 80.2 | 3.8% | 9.48 | 1.70 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 214.7 | 233.9 | 9.0% | 6.86 | 1.22 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 131.3 | 153.7 | 17.1% | 5.58 | 1.00 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 215.8 | 234.2 | 8.5% | 6.82 | 1.21 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 446.9 | 476.6 | 6.7% | 5.50 | 1.17 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.8 | 2.0 | 16.7% | 17.44 | 2.34 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 2.3 | 2.3 | 0.0% | 32.70 | 3.51 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 9.3 | 9.6 | 3.1% | 14.71 | 3.51 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 2.0 | 2.0 | 0.0% | 11.21 | 2.01 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 2.6 | 2.9 | 11.1% | 23.25 | 3.12 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 2.9 | 2.9 | 0.0% | 7.85 | 1.40 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 4.1 | 4.1 | 0.0% | 14.95 | 2.01 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 3.5 | 3.8 | 8.3% | 30.52 | 4.68 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 1.2 | 1.5 | 24.9% | 13.08 | 1.76 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.5 | 1.5 | 0.1% | 26.16 | 2.81 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 2.6 | 2.9 | 11.1% | 26.16 | 6.24 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 1.5 | 1.5 | 0.1% | 7.85 | 1.40 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.8 | 1.8 | 0.1% | 17.44 | 2.34 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.8 | 1.8 | 0.1% | 6.54 | 1.17 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 2.3 | 2.6 | 12.5% | 13.08 | 1.76 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.8 | 1.8 | 0.1% | 30.52 | 4.68 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 7.04 | 22.2% | 277.4 | 10.2% |
| Gated Delta-Rule Scan | 5.74 | 18.1% | 515.4 | 8.2% |
| Causal Depthwise Conv1D | 6.93 | 21.9% | 297.2 | 9.3% |
| gdn_cumdecay_f16 | 8.36 | 26.4% | 175.3 | 5.2% |
| gdn_gated_scan_f16 | 5.79 | 18.3% | 508.4 | 6.9% |
| gdn_cumdecay_bf16 | 5.86 | 18.5% | 250.0 | 5.1% |
| gdn_gated_scan_bf16 | 5.79 | 18.3% | 508.4 | 8.5% |
| gdn2_gated_scan | 3.09 | 9.7% | 1,587.9 | 1.8% |
| Gated Cumulative Decay | 8.05 | 25.4% | 121.3 | 7.7% |
| Gated Delta-Rule Scan | 6.89 | 21.7% | 214.7 | 6.0% |
| Causal Depthwise Conv1D | 7.06 | 22.3% | 145.8 | 5.4% |
| gdn_cumdecay_f16 | 9.48 | 29.9% | 77.3 | 3.8% |
| gdn_gated_scan_f16 | 6.86 | 21.6% | 214.7 | 9.0% |
| gdn_cumdecay_bf16 | 5.58 | 17.6% | 131.3 | 17.1% |
| gdn_gated_scan_bf16 | 6.82 | 21.5% | 215.8 | 8.5% |
| gdn2_gated_scan | 5.50 | 17.4% | 446.9 | 6.7% |
| Gated Cumulative Decay | 17.44 | 55.0% | 1.8 | 16.7% |
| Gated Delta-Rule Scan | 32.70 | 103.2% | 2.3 | 0.0% |
| Causal Depthwise Conv1D | 14.71 | 46.4% | 9.3 | 3.1% |
| gdn_cumdecay_f16 | 11.21 | 35.4% | 2.0 | 0.0% |
| gdn_gated_scan_f16 | 23.25 | 73.3% | 2.6 | 11.1% |
| gdn_cumdecay_bf16 | 7.85 | 24.8% | 2.9 | 0.0% |
| gdn_gated_scan_bf16 | 14.95 | 47.2% | 4.1 | 0.0% |
| gdn2_gated_scan | 30.52 | 96.3% | 3.5 | 8.3% |
| Gated Cumulative Decay | 13.08 | 41.3% | 1.2 | 24.9% |
| Gated Delta-Rule Scan | 26.16 | 82.5% | 1.5 | 0.1% |
| Causal Depthwise Conv1D | 26.16 | 82.5% | 2.6 | 11.1% |
| gdn_cumdecay_f16 | 7.85 | 24.8% | 1.5 | 0.1% |
| gdn_gated_scan_f16 | 17.44 | 55.0% | 1.8 | 0.1% |
| gdn_cumdecay_bf16 | 6.54 | 20.6% | 1.8 | 0.1% |
| gdn_gated_scan_bf16 | 13.08 | 41.3% | 2.3 | 12.5% |
| gdn2_gated_scan | 30.52 | 96.3% | 1.8 | 0.1% |
