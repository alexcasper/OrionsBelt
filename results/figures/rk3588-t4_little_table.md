# rk3588-t4_little — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 336.0 | 371.3 | 10.5% | 5.81 | 0.78 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 908.0 | 1,070.2 | 17.9% | 3.26 | 0.58 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 400.8 | 439.0 | 9.5% | 5.14 | 5.23 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 221.7 | 255.8 | 15.4% | 6.61 | 1.18 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 895.8 | 1,015.9 | 13.4% | 3.29 | 0.59 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 247.9 | 267.8 | 8.0% | 5.91 | 1.06 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 893.1 | 974.2 | 9.1% | 3.30 | 0.59 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 3,129.5 | 3,521.8 | 12.5% | 1.57 | 0.34 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 152.6 | 153.4 | 0.6% | 6.40 | 0.86 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 257.6 | 301.0 | 16.9% | 5.75 | 1.02 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 143.8 | 146.4 | 1.8% | 7.16 | 7.29 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 110.8 | 112.6 | 1.6% | 6.61 | 1.18 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 257.9 | 277.7 | 7.7% | 5.71 | 1.02 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 124.6 | 127.5 | 2.3% | 5.88 | 1.05 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 257.9 | 278.9 | 8.1% | 5.71 | 1.02 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 595.3 | 681.1 | 14.4% | 4.13 | 0.88 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 3.8 | 4.1 | 7.7% | 8.05 | 1.08 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 4.7 | 5.5 | 15.4% | 16.15 | 1.73 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 14.6 | 14.9 | 2.0% | 9.42 | 2.25 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 4.5 | 4.7 | 3.8% | 5.05 | 0.90 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 5.4 | 5.6 | 2.9% | 11.21 | 1.51 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 4.6 | 5.0 | 8.6% | 4.95 | 0.89 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 6.1 | 6.3 | 2.6% | 10.00 | 1.34 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 9.8 | 9.9 | 1.4% | 10.91 | 1.67 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 3.7 | 3.9 | 3.7% | 4.08 | 0.55 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 4.4 | 4.6 | 3.6% | 8.62 | 0.93 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 9.7 | 9.9 | 1.6% | 7.07 | 1.69 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 3.8 | 3.9 | 1.6% | 2.98 | 0.53 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 4.4 | 4.5 | 3.5% | 6.96 | 0.93 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 4.1 | 4.2 | 3.6% | 2.80 | 0.50 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 4.6 | 4.8 | 3.2% | 6.59 | 0.88 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 6.5 | 6.7 | 2.5% | 8.16 | 1.25 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 5.81 | 18.3% | 336.0 | 10.5% |
| Gated Delta-Rule Scan | 3.26 | 10.3% | 908.0 | 17.9% |
| Causal Depthwise Conv1D | 5.14 | 16.2% | 400.8 | 9.5% |
| gdn_cumdecay_f16 | 6.61 | 20.9% | 221.7 | 15.4% |
| gdn_gated_scan_f16 | 3.29 | 10.4% | 895.8 | 13.4% |
| gdn_cumdecay_bf16 | 5.91 | 18.6% | 247.9 | 8.0% |
| gdn_gated_scan_bf16 | 3.30 | 10.4% | 893.1 | 9.1% |
| gdn2_gated_scan | 1.57 | 5.0% | 3,129.5 | 12.5% |
| Gated Cumulative Decay | 6.40 | 20.2% | 152.6 | 0.6% |
| Gated Delta-Rule Scan | 5.75 | 18.1% | 257.6 | 16.9% |
| Causal Depthwise Conv1D | 7.16 | 22.6% | 143.8 | 1.8% |
| gdn_cumdecay_f16 | 6.61 | 20.9% | 110.8 | 1.6% |
| gdn_gated_scan_f16 | 5.71 | 18.0% | 257.9 | 7.7% |
| gdn_cumdecay_bf16 | 5.88 | 18.5% | 124.6 | 2.3% |
| gdn_gated_scan_bf16 | 5.71 | 18.0% | 257.9 | 8.1% |
| gdn2_gated_scan | 4.13 | 13.0% | 595.3 | 14.4% |
| Gated Cumulative Decay | 8.05 | 25.4% | 3.8 | 7.7% |
| Gated Delta-Rule Scan | 16.15 | 50.9% | 4.7 | 15.4% |
| Causal Depthwise Conv1D | 9.42 | 29.7% | 14.6 | 2.0% |
| gdn_cumdecay_f16 | 5.05 | 15.9% | 4.5 | 3.8% |
| gdn_gated_scan_f16 | 11.21 | 35.4% | 5.4 | 2.9% |
| gdn_cumdecay_bf16 | 4.95 | 15.6% | 4.6 | 8.6% |
| gdn_gated_scan_bf16 | 10.00 | 31.5% | 6.1 | 2.6% |
| gdn2_gated_scan | 10.91 | 34.4% | 9.8 | 1.4% |
| Gated Cumulative Decay | 4.08 | 12.9% | 3.7 | 3.7% |
| Gated Delta-Rule Scan | 8.62 | 27.2% | 4.4 | 3.6% |
| Causal Depthwise Conv1D | 7.07 | 22.3% | 9.7 | 1.6% |
| gdn_cumdecay_f16 | 2.98 | 9.4% | 3.8 | 1.6% |
| gdn_gated_scan_f16 | 6.96 | 22.0% | 4.4 | 3.5% |
| gdn_cumdecay_bf16 | 2.80 | 8.8% | 4.1 | 3.6% |
| gdn_gated_scan_bf16 | 6.59 | 20.8% | 4.6 | 3.2% |
| gdn2_gated_scan | 8.16 | 25.7% | 6.5 | 2.5% |
