# jetson-j1_clean — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 515.6 | 747.6 | 45.0% | 3.79 | 0.51 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 1,013.7 | 1,170.4 | 15.5% | 2.92 | 0.52 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 571.9 | 635.7 | 11.1% | 3.60 | 3.67 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 353.4 | 385.2 | 9.0% | 4.14 | 0.74 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 1,008.9 | 1,098.7 | 8.9% | 2.92 | 0.52 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 348.9 | 496.1 | 42.2% | 4.20 | 0.75 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 993.5 | 1,040.3 | 4.7% | 2.96 | 0.53 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 1,731.0 | 3,381.6 | 95.4% | 2.84 | 0.61 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 197.2 | 219.9 | 11.5% | 4.95 | 0.66 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 283.0 | 337.8 | 19.4% | 5.23 | 0.93 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 196.2 | 240.2 | 22.5% | 5.25 | 5.35 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 160.3 | 164.8 | 2.9% | 4.57 | 0.82 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 286.9 | 338.8 | 18.1% | 5.13 | 0.91 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 159.4 | 178.8 | 12.2% | 4.60 | 0.82 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 285.3 | 315.4 | 10.6% | 5.16 | 0.92 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 437.1 | 1,493.2 | 241.6% | 5.62 | 1.20 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 3.6 | 4.1 | 11.4% | 8.37 | 1.12 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 4.7 | 4.8 | 2.2% | 16.10 | 1.73 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 11.1 | 11.6 | 3.7% | 12.32 | 2.94 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 4.2 | 4.3 | 3.7% | 5.49 | 0.98 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 5.4 | 5.5 | 2.9% | 11.38 | 1.53 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 4.6 | 4.8 | 3.4% | 4.94 | 0.88 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 5.5 | 5.7 | 2.8% | 11.06 | 1.48 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 5.2 | 5.4 | 4.0% | 20.71 | 3.18 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 3.1 | 3.3 | 5.0% | 4.88 | 0.66 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 3.9 | 3.9 | 1.4% | 9.90 | 1.06 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 8.1 | 8.3 | 3.2% | 8.51 | 2.03 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 3.3 | 4.0 | 20.3% | 3.43 | 0.61 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 4.2 | 4.3 | 2.5% | 7.23 | 0.97 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 3.8 | 3.9 | 4.2% | 3.05 | 0.55 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 4.2 | 4.4 | 3.7% | 7.23 | 0.97 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 4.3 | 4.3 | 1.2% | 12.50 | 1.92 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 25.6 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 3.79 | 14.8% | 515.6 | 45.0% |
| Gated Delta-Rule Scan | 2.92 | 11.4% | 1,013.7 | 15.5% |
| Causal Depthwise Conv1D | 3.60 | 14.1% | 571.9 | 11.1% |
| gdn_cumdecay_f16 | 4.14 | 16.2% | 353.4 | 9.0% |
| gdn_gated_scan_f16 | 2.92 | 11.4% | 1,008.9 | 8.9% |
| gdn_cumdecay_bf16 | 4.20 | 16.4% | 348.9 | 42.2% |
| gdn_gated_scan_bf16 | 2.96 | 11.6% | 993.5 | 4.7% |
| gdn2_gated_scan | 2.84 | 11.1% | 1,731.0 | 95.4% |
| Gated Cumulative Decay | 4.95 | 19.3% | 197.2 | 11.5% |
| Gated Delta-Rule Scan | 5.23 | 20.4% | 283.0 | 19.4% |
| Causal Depthwise Conv1D | 5.25 | 20.5% | 196.2 | 22.5% |
| gdn_cumdecay_f16 | 4.57 | 17.9% | 160.3 | 2.9% |
| gdn_gated_scan_f16 | 5.13 | 20.0% | 286.9 | 18.1% |
| gdn_cumdecay_bf16 | 4.60 | 18.0% | 159.4 | 12.2% |
| gdn_gated_scan_bf16 | 5.16 | 20.2% | 285.3 | 10.6% |
| gdn2_gated_scan | 5.62 | 22.0% | 437.1 | 241.6% |
| Gated Cumulative Decay | 8.37 | 32.7% | 3.6 | 11.4% |
| Gated Delta-Rule Scan | 16.10 | 62.9% | 4.7 | 2.2% |
| Causal Depthwise Conv1D | 12.32 | 48.1% | 11.1 | 3.7% |
| gdn_cumdecay_f16 | 5.49 | 21.4% | 4.2 | 3.7% |
| gdn_gated_scan_f16 | 11.38 | 44.5% | 5.4 | 2.9% |
| gdn_cumdecay_bf16 | 4.94 | 19.3% | 4.6 | 3.4% |
| gdn_gated_scan_bf16 | 11.06 | 43.2% | 5.5 | 2.8% |
| gdn2_gated_scan | 20.71 | 80.9% | 5.2 | 4.0% |
| Gated Cumulative Decay | 4.88 | 19.1% | 3.1 | 5.0% |
| Gated Delta-Rule Scan | 9.90 | 38.7% | 3.9 | 1.4% |
| Causal Depthwise Conv1D | 8.51 | 33.2% | 8.1 | 3.2% |
| gdn_cumdecay_f16 | 3.43 | 13.4% | 3.3 | 20.3% |
| gdn_gated_scan_f16 | 7.23 | 28.2% | 4.2 | 2.5% |
| gdn_cumdecay_bf16 | 3.05 | 11.9% | 3.8 | 4.2% |
| gdn_gated_scan_bf16 | 7.23 | 28.2% | 4.2 | 3.7% |
| gdn2_gated_scan | 12.50 | 48.8% | 4.3 | 1.2% |
