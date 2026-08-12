# rk3588-t4_little — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 336.0 | 371.9 | 10.7% | 5.81 | 0.78 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 730.1 | 813.8 | 11.5% | 4.05 | 0.72 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 387.4 | 404.6 | 4.4% | 5.32 | 5.41 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 222.6 | 254.1 | 14.2% | 6.58 | 1.18 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 727.5 | 838.9 | 15.3% | 4.05 | 0.72 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 248.5 | 270.7 | 8.9% | 5.89 | 1.05 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 721.6 | 801.0 | 11.0% | 4.08 | 0.73 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 2,931.5 | 3,103.0 | 5.9% | 1.68 | 0.36 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 152.3 | 152.8 | 0.4% | 6.41 | 0.86 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 253.2 | 272.4 | 7.6% | 5.85 | 1.04 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 143.8 | 145.8 | 1.4% | 7.16 | 7.29 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 112.3 | 116.7 | 3.9% | 6.52 | 1.17 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 253.2 | 268.6 | 6.1% | 5.82 | 1.04 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 125.7 | 126.6 | 0.7% | 5.83 | 1.04 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 253.2 | 269.5 | 6.5% | 5.82 | 1.04 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 613.7 | 645.8 | 5.2% | 4.00 | 0.85 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 3.8 | 4.1 | 7.7% | 8.05 | 1.08 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 5.8 | 6.0 | 3.3% | 13.23 | 1.42 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 15.1 | 15.3 | 1.3% | 9.09 | 2.17 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 4.7 | 4.8 | 3.2% | 4.91 | 0.88 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 5.5 | 5.7 | 3.4% | 11.11 | 1.49 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 5.2 | 5.3 | 2.8% | 4.43 | 0.79 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 6.1 | 6.3 | 2.3% | 9.97 | 1.34 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 10.0 | 10.2 | 1.6% | 10.64 | 1.63 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 3.7 | 3.8 | 2.9% | 4.13 | 0.55 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 4.5 | 4.7 | 4.2% | 8.49 | 0.91 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 9.9 | 10.1 | 1.5% | 6.92 | 1.65 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 3.9 | 4.0 | 3.7% | 2.93 | 0.53 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 4.3 | 4.5 | 3.5% | 7.03 | 0.94 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 4.2 | 4.3 | 3.2% | 2.75 | 0.49 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 4.6 | 4.7 | 3.1% | 6.67 | 0.90 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 6.6 | 6.7 | 2.2% | 8.15 | 1.25 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 5.81 | 18.3% | 336.0 | 10.7% |
| Gated Delta-Rule Scan | 4.05 | 12.8% | 730.1 | 11.5% |
| Causal Depthwise Conv1D | 5.32 | 16.8% | 387.4 | 4.4% |
| gdn_cumdecay_f16 | 6.58 | 20.8% | 222.6 | 14.2% |
| gdn_gated_scan_f16 | 4.05 | 12.8% | 727.5 | 15.3% |
| gdn_cumdecay_bf16 | 5.89 | 18.6% | 248.5 | 8.9% |
| gdn_gated_scan_bf16 | 4.08 | 12.9% | 721.6 | 11.0% |
| gdn2_gated_scan | 1.68 | 5.3% | 2,931.5 | 5.9% |
| Gated Cumulative Decay | 6.41 | 20.2% | 152.3 | 0.4% |
| Gated Delta-Rule Scan | 5.85 | 18.5% | 253.2 | 7.6% |
| Causal Depthwise Conv1D | 7.16 | 22.6% | 143.8 | 1.4% |
| gdn_cumdecay_f16 | 6.52 | 20.6% | 112.3 | 3.9% |
| gdn_gated_scan_f16 | 5.82 | 18.4% | 253.2 | 6.1% |
| gdn_cumdecay_bf16 | 5.83 | 18.4% | 125.7 | 0.7% |
| gdn_gated_scan_bf16 | 5.82 | 18.4% | 253.2 | 6.5% |
| gdn2_gated_scan | 4.00 | 12.6% | 613.7 | 5.2% |
| Gated Cumulative Decay | 8.05 | 25.4% | 3.8 | 7.7% |
| Gated Delta-Rule Scan | 13.23 | 41.7% | 5.8 | 3.3% |
| Causal Depthwise Conv1D | 9.09 | 28.7% | 15.1 | 1.3% |
| gdn_cumdecay_f16 | 4.91 | 15.5% | 4.7 | 3.2% |
| gdn_gated_scan_f16 | 11.11 | 35.0% | 5.5 | 3.4% |
| gdn_cumdecay_bf16 | 4.43 | 14.0% | 5.2 | 2.8% |
| gdn_gated_scan_bf16 | 9.97 | 31.5% | 6.1 | 2.3% |
| gdn2_gated_scan | 10.64 | 33.6% | 10.0 | 1.6% |
| Gated Cumulative Decay | 4.13 | 13.0% | 3.7 | 2.9% |
| Gated Delta-Rule Scan | 8.49 | 26.8% | 4.5 | 4.2% |
| Causal Depthwise Conv1D | 6.92 | 21.8% | 9.9 | 1.5% |
| gdn_cumdecay_f16 | 2.93 | 9.2% | 3.9 | 3.7% |
| gdn_gated_scan_f16 | 7.03 | 22.2% | 4.3 | 3.5% |
| gdn_cumdecay_bf16 | 2.75 | 8.7% | 4.2 | 3.2% |
| gdn_gated_scan_bf16 | 6.67 | 21.0% | 4.6 | 3.1% |
| gdn2_gated_scan | 8.15 | 25.7% | 6.6 | 2.2% |
