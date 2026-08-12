# rk3588-t4_big — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 90.1 | 101.8 | 12.9% | 21.67 | 2.91 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 259.3 | 276.5 | 6.6% | 11.42 | 2.02 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 99.5 | 103.8 | 4.4% | 20.71 | 21.08 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 38.5 | 40.3 | 4.5% | 38.04 | 6.81 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 261.9 | 269.8 | 3.0% | 11.24 | 2.00 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 54.5 | 54.8 | 0.5% | 26.86 | 4.81 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 264.6 | 271.9 | 2.8% | 11.13 | 1.98 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 697.7 | 756.9 | 8.5% | 7.04 | 1.50 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 38.2 | 38.5 | 0.8% | 25.56 | 3.43 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 131.6 | 134.8 | 2.4% | 11.25 | 1.99 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 41.1 | 44.6 | 8.5% | 25.04 | 25.50 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 21.9 | 22.2 | 1.3% | 33.48 | 5.99 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 133.3 | 139.1 | 4.4% | 11.05 | 1.97 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 32.7 | 33.3 | 1.8% | 22.42 | 4.01 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 129.5 | 141.5 | 9.2% | 11.37 | 2.02 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 321.1 | 327.6 | 2.0% | 7.65 | 1.63 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.5 | 1.8 | 19.9% | 20.91 | 2.81 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 1.8 | 1.8 | 1.3% | 43.09 | 4.63 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 3.7 | 3.7 | 2.0% | 37.37 | 8.92 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 1.6 | 1.7 | 3.2% | 13.96 | 2.50 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 1.8 | 1.9 | 2.7% | 33.53 | 4.50 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 1.8 | 1.9 | 0.3% | 12.40 | 2.22 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 2.1 | 2.2 | 2.0% | 28.43 | 3.82 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 2.3 | 2.3 | 1.8% | 47.19 | 7.24 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 1.4 | 1.4 | 0.2% | 10.83 | 1.45 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.5 | 1.6 | 2.7% | 25.01 | 2.68 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 1.9 | 2.1 | 7.6% | 35.61 | 8.50 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 1.5 | 1.5 | 0.2% | 7.85 | 1.40 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.6 | 1.6 | 3.0% | 19.45 | 2.61 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.6 | 1.6 | 0.2% | 7.33 | 1.31 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 1.8 | 1.8 | 2.7% | 17.35 | 2.33 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.6 | 1.6 | 1.8% | 33.05 | 5.07 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 21.67 | 68.4% | 90.1 | 12.9% |
| Gated Delta-Rule Scan | 11.42 | 36.0% | 259.3 | 6.6% |
| Causal Depthwise Conv1D | 20.71 | 65.3% | 99.5 | 4.4% |
| gdn_cumdecay_f16 | 38.04 | 120.0% | 38.5 | 4.5% |
| gdn_gated_scan_f16 | 11.24 | 35.5% | 261.9 | 3.0% |
| gdn_cumdecay_bf16 | 26.86 | 84.7% | 54.5 | 0.5% |
| gdn_gated_scan_bf16 | 11.13 | 35.1% | 264.6 | 2.8% |
| gdn2_gated_scan | 7.04 | 22.2% | 697.7 | 8.5% |
| Gated Cumulative Decay | 25.56 | 80.6% | 38.2 | 0.8% |
| Gated Delta-Rule Scan | 11.25 | 35.5% | 131.6 | 2.4% |
| Causal Depthwise Conv1D | 25.04 | 79.0% | 41.1 | 8.5% |
| gdn_cumdecay_f16 | 33.48 | 105.6% | 21.9 | 1.3% |
| gdn_gated_scan_f16 | 11.05 | 34.9% | 133.3 | 4.4% |
| gdn_cumdecay_bf16 | 22.42 | 70.7% | 32.7 | 1.8% |
| gdn_gated_scan_bf16 | 11.37 | 35.9% | 129.5 | 9.2% |
| gdn2_gated_scan | 7.65 | 24.1% | 321.1 | 2.0% |
| Gated Cumulative Decay | 20.91 | 66.0% | 1.5 | 19.9% |
| Gated Delta-Rule Scan | 43.09 | 135.9% | 1.8 | 1.3% |
| Causal Depthwise Conv1D | 37.37 | 117.9% | 3.7 | 2.0% |
| gdn_cumdecay_f16 | 13.96 | 44.0% | 1.6 | 3.2% |
| gdn_gated_scan_f16 | 33.53 | 105.8% | 1.8 | 2.7% |
| gdn_cumdecay_bf16 | 12.40 | 39.1% | 1.8 | 0.3% |
| gdn_gated_scan_bf16 | 28.43 | 89.7% | 2.1 | 2.0% |
| gdn2_gated_scan | 47.19 | 148.9% | 2.3 | 1.8% |
| Gated Cumulative Decay | 10.83 | 34.2% | 1.4 | 0.2% |
| Gated Delta-Rule Scan | 25.01 | 78.9% | 1.5 | 2.7% |
| Causal Depthwise Conv1D | 35.61 | 112.3% | 1.9 | 7.6% |
| gdn_cumdecay_f16 | 7.85 | 24.8% | 1.5 | 0.2% |
| gdn_gated_scan_f16 | 19.45 | 61.4% | 1.6 | 3.0% |
| gdn_cumdecay_bf16 | 7.33 | 23.1% | 1.6 | 0.2% |
| gdn_gated_scan_bf16 | 17.35 | 54.7% | 1.8 | 2.7% |
| gdn2_gated_scan | 33.05 | 104.3% | 1.6 | 1.8% |
