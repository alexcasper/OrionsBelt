# rk3588-t4_little — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 335.7 | 375.7 | 11.9% | 5.82 | 0.78 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 802.4 | 1,107.0 | 37.9% | 3.69 | 0.65 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 399.9 | 416.8 | 4.2% | 5.15 | 5.24 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 219.6 | 238.9 | 8.8% | 6.67 | 1.19 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 774.4 | 833.1 | 7.6% | 3.80 | 0.68 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 246.2 | 267.2 | 8.5% | 5.95 | 1.06 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 793.4 | 849.1 | 7.0% | 3.71 | 0.66 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 2,983.1 | 3,191.3 | 7.0% | 1.65 | 0.35 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 152.6 | 177.6 | 16.4% | 6.40 | 0.86 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 256.1 | 295.2 | 15.3% | 5.78 | 1.02 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 143.8 | 144.4 | 0.4% | 7.16 | 7.29 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 112.6 | 114.0 | 1.3% | 6.51 | 1.16 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 255.5 | 270.7 | 5.9% | 5.76 | 1.03 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 126.0 | 127.2 | 0.9% | 5.81 | 1.04 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 255.8 | 270.1 | 5.6% | 5.76 | 1.02 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 605.0 | 620.1 | 2.5% | 4.06 | 0.87 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 3.8 | 4.1 | 7.7% | 8.05 | 1.08 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 5.8 | 6.1 | 5.0% | 13.12 | 1.41 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 14.9 | 15.2 | 1.8% | 9.21 | 2.20 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 4.6 | 4.8 | 3.3% | 4.93 | 0.88 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 5.5 | 5.7 | 2.7% | 11.04 | 1.48 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 5.2 | 5.3 | 2.7% | 4.44 | 0.79 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 6.2 | 6.3 | 2.2% | 9.91 | 1.33 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 10.0 | 10.2 | 2.3% | 10.67 | 1.64 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 3.7 | 3.9 | 4.4% | 4.10 | 0.55 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 4.0 | 4.6 | 14.7% | 9.53 | 1.02 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 9.8 | 10.0 | 1.7% | 6.97 | 1.66 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 3.9 | 4.0 | 3.5% | 2.94 | 0.53 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 4.4 | 4.7 | 6.2% | 6.88 | 0.92 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 3.7 | 4.2 | 11.2% | 3.07 | 0.55 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 4.7 | 4.8 | 3.2% | 6.56 | 0.88 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 6.7 | 6.8 | 2.3% | 8.00 | 1.23 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 5.82 | 18.4% | 335.7 | 11.9% |
| Gated Delta-Rule Scan | 3.69 | 11.6% | 802.4 | 37.9% |
| Causal Depthwise Conv1D | 5.15 | 16.2% | 399.9 | 4.2% |
| gdn_cumdecay_f16 | 6.67 | 21.0% | 219.6 | 8.8% |
| gdn_gated_scan_f16 | 3.80 | 12.0% | 774.4 | 7.6% |
| gdn_cumdecay_bf16 | 5.95 | 18.8% | 246.2 | 8.5% |
| gdn_gated_scan_bf16 | 3.71 | 11.7% | 793.4 | 7.0% |
| gdn2_gated_scan | 1.65 | 5.2% | 2,983.1 | 7.0% |
| Gated Cumulative Decay | 6.40 | 20.2% | 152.6 | 16.4% |
| Gated Delta-Rule Scan | 5.78 | 18.2% | 256.1 | 15.3% |
| Causal Depthwise Conv1D | 7.16 | 22.6% | 143.8 | 0.4% |
| gdn_cumdecay_f16 | 6.51 | 20.5% | 112.6 | 1.3% |
| gdn_gated_scan_f16 | 5.76 | 18.2% | 255.5 | 5.9% |
| gdn_cumdecay_bf16 | 5.81 | 18.3% | 126.0 | 0.9% |
| gdn_gated_scan_bf16 | 5.76 | 18.2% | 255.8 | 5.6% |
| gdn2_gated_scan | 4.06 | 12.8% | 605.0 | 2.5% |
| Gated Cumulative Decay | 8.05 | 25.4% | 3.8 | 7.7% |
| Gated Delta-Rule Scan | 13.12 | 41.4% | 5.8 | 5.0% |
| Causal Depthwise Conv1D | 9.21 | 29.1% | 14.9 | 1.8% |
| gdn_cumdecay_f16 | 4.93 | 15.6% | 4.6 | 3.3% |
| gdn_gated_scan_f16 | 11.04 | 34.8% | 5.5 | 2.7% |
| gdn_cumdecay_bf16 | 4.44 | 14.0% | 5.2 | 2.7% |
| gdn_gated_scan_bf16 | 9.91 | 31.3% | 6.2 | 2.2% |
| gdn2_gated_scan | 10.67 | 33.7% | 10.0 | 2.3% |
| Gated Cumulative Decay | 4.10 | 12.9% | 3.7 | 4.4% |
| Gated Delta-Rule Scan | 9.53 | 30.1% | 4.0 | 14.7% |
| Causal Depthwise Conv1D | 6.97 | 22.0% | 9.8 | 1.7% |
| gdn_cumdecay_f16 | 2.94 | 9.3% | 3.9 | 3.5% |
| gdn_gated_scan_f16 | 6.88 | 21.7% | 4.4 | 6.2% |
| gdn_cumdecay_bf16 | 3.07 | 9.7% | 3.7 | 11.2% |
| gdn_gated_scan_bf16 | 6.56 | 20.7% | 4.7 | 3.2% |
| gdn2_gated_scan | 8.00 | 25.2% | 6.7 | 2.3% |
