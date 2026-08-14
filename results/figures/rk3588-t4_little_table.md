# rk3588-t4_little — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 329.6 | 369.9 | 12.2% | 5.93 | 0.80 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 755.5 | 854.9 | 13.2% | 3.92 | 0.69 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 397.0 | 426.7 | 7.5% | 5.19 | 5.28 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 221.4 | 235.4 | 6.3% | 6.62 | 1.18 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 736.8 | 777.6 | 5.5% | 4.00 | 0.71 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 247.4 | 267.2 | 8.0% | 5.92 | 1.06 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 741.8 | 781.4 | 5.3% | 3.97 | 0.71 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 2,975.8 | 3,180.3 | 6.9% | 1.65 | 0.35 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 152.8 | 181.1 | 18.5% | 6.39 | 0.86 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 252.3 | 271.9 | 7.7% | 5.87 | 1.04 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 144.1 | 144.7 | 0.4% | 7.15 | 7.28 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 112.6 | 114.6 | 1.8% | 6.51 | 1.16 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 252.0 | 268.9 | 6.7% | 5.84 | 1.04 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 126.6 | 129.5 | 2.3% | 5.79 | 1.04 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 252.6 | 268.4 | 6.2% | 5.83 | 1.04 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 619.0 | 711.7 | 15.0% | 3.97 | 0.85 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 3.8 | 4.1 | 7.7% | 8.05 | 1.08 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 5.8 | 6.0 | 4.7% | 13.24 | 1.42 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 14.9 | 15.2 | 2.0% | 9.23 | 2.20 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 4.7 | 4.9 | 5.5% | 4.89 | 0.88 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 5.5 | 5.7 | 2.9% | 11.06 | 1.48 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 5.2 | 5.3 | 2.9% | 4.42 | 0.79 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 6.2 | 6.3 | 2.7% | 9.92 | 1.33 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 9.8 | 10.1 | 2.9% | 10.89 | 1.67 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 3.7 | 3.9 | 3.8% | 4.07 | 0.55 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 4.6 | 4.8 | 3.3% | 8.25 | 0.89 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 9.8 | 9.9 | 1.5% | 7.01 | 1.67 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 3.5 | 3.9 | 11.4% | 3.24 | 0.58 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 4.5 | 4.6 | 3.1% | 6.85 | 0.92 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 4.2 | 4.3 | 3.6% | 2.74 | 0.49 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 4.7 | 4.8 | 3.1% | 6.54 | 0.88 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 6.8 | 6.9 | 2.2% | 7.90 | 1.21 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 5.93 | 18.7% | 329.6 | 12.2% |
| Gated Delta-Rule Scan | 3.92 | 12.4% | 755.5 | 13.2% |
| Causal Depthwise Conv1D | 5.19 | 16.4% | 397.0 | 7.5% |
| gdn_cumdecay_f16 | 6.62 | 20.9% | 221.4 | 6.3% |
| gdn_gated_scan_f16 | 4.00 | 12.6% | 736.8 | 5.5% |
| gdn_cumdecay_bf16 | 5.92 | 18.7% | 247.4 | 8.0% |
| gdn_gated_scan_bf16 | 3.97 | 12.5% | 741.8 | 5.3% |
| gdn2_gated_scan | 1.65 | 5.2% | 2,975.8 | 6.9% |
| Gated Cumulative Decay | 6.39 | 20.2% | 152.8 | 18.5% |
| Gated Delta-Rule Scan | 5.87 | 18.5% | 252.3 | 7.7% |
| Causal Depthwise Conv1D | 7.15 | 22.6% | 144.1 | 0.4% |
| gdn_cumdecay_f16 | 6.51 | 20.5% | 112.6 | 1.8% |
| gdn_gated_scan_f16 | 5.84 | 18.4% | 252.0 | 6.7% |
| gdn_cumdecay_bf16 | 5.79 | 18.3% | 126.6 | 2.3% |
| gdn_gated_scan_bf16 | 5.83 | 18.4% | 252.6 | 6.2% |
| gdn2_gated_scan | 3.97 | 12.5% | 619.0 | 15.0% |
| Gated Cumulative Decay | 8.05 | 25.4% | 3.8 | 7.7% |
| Gated Delta-Rule Scan | 13.24 | 41.8% | 5.8 | 4.7% |
| Causal Depthwise Conv1D | 9.23 | 29.1% | 14.9 | 2.0% |
| gdn_cumdecay_f16 | 4.89 | 15.4% | 4.7 | 5.5% |
| gdn_gated_scan_f16 | 11.06 | 34.9% | 5.5 | 2.9% |
| gdn_cumdecay_bf16 | 4.42 | 13.9% | 5.2 | 2.9% |
| gdn_gated_scan_bf16 | 9.92 | 31.3% | 6.2 | 2.7% |
| gdn2_gated_scan | 10.89 | 34.4% | 9.8 | 2.9% |
| Gated Cumulative Decay | 4.07 | 12.8% | 3.7 | 3.8% |
| Gated Delta-Rule Scan | 8.25 | 26.0% | 4.6 | 3.3% |
| Causal Depthwise Conv1D | 7.01 | 22.1% | 9.8 | 1.5% |
| gdn_cumdecay_f16 | 3.24 | 10.2% | 3.5 | 11.4% |
| gdn_gated_scan_f16 | 6.85 | 21.6% | 4.5 | 3.1% |
| gdn_cumdecay_bf16 | 2.74 | 8.6% | 4.2 | 3.6% |
| gdn_gated_scan_bf16 | 6.54 | 20.6% | 4.7 | 3.1% |
| gdn2_gated_scan | 7.90 | 24.9% | 6.8 | 2.2% |
