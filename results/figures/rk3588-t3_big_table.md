# rk3588-t3_big — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 84.3 | 89.5 | 6.2% | 23.17 | 3.11 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 286.4 | 310.1 | 8.2% | 10.33 | 1.83 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 96.5 | 103.8 | 7.6% | 21.34 | 21.72 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 40.5 | 41.7 | 2.9% | 36.13 | 6.47 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 293.4 | 317.1 | 8.1% | 10.04 | 1.79 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 60.1 | 63.3 | 5.3% | 24.38 | 4.36 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 287.9 | 307.1 | 6.7% | 10.23 | 1.82 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 543.4 | 590.1 | 8.6% | 9.04 | 1.93 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 37.9 | 40.5 | 6.9% | 25.75 | 3.46 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 96.0 | 100.9 | 5.2% | 15.42 | 2.73 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 34.4 | 35.0 | 1.7% | 29.92 | 30.46 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 21.6 | 21.9 | 1.4% | 33.93 | 6.07 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 96.0 | 100.3 | 4.6% | 15.34 | 2.73 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 29.2 | 30.9 | 6.0% | 25.11 | 4.49 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 96.8 | 98.9 | 2.1% | 15.21 | 2.71 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 237.4 | 251.1 | 5.8% | 10.35 | 2.21 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.2 | 1.2 | 0.1% | 26.17 | 3.51 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 1.5 | 1.5 | 0.1% | 52.33 | 5.62 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 2.6 | 2.9 | 11.1% | 52.32 | 12.48 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 1.2 | 1.2 | 0.0% | 19.61 | 3.51 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 1.5 | 1.5 | 0.1% | 41.86 | 5.62 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 1.5 | 1.5 | 0.1% | 15.70 | 2.81 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 1.8 | 1.8 | 0.1% | 34.88 | 4.68 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 1.8 | 1.8 | 0.0% | 61.04 | 9.36 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 0.9 | 1.2 | 33.4% | 17.44 | 2.34 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.2 | 1.2 | 0.1% | 32.72 | 3.51 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 2.0 | 2.0 | 0.1% | 33.64 | 8.03 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 0.9 | 1.2 | 33.4% | 13.08 | 2.34 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.2 | 1.2 | 0.0% | 26.15 | 3.51 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.2 | 1.2 | 0.0% | 9.81 | 1.76 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 1.2 | 1.5 | 25.0% | 26.15 | 3.51 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.2 | 1.2 | 0.0% | 45.77 | 7.02 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 34.0 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 23.17 | 68.1% | 84.3 | 6.2% |
| Gated Delta-Rule Scan | 10.33 | 30.4% | 286.4 | 8.2% |
| Causal Depthwise Conv1D | 21.34 | 62.8% | 96.5 | 7.6% |
| gdn_cumdecay_f16 | 36.13 | 106.3% | 40.5 | 2.9% |
| gdn_gated_scan_f16 | 10.04 | 29.5% | 293.4 | 8.1% |
| gdn_cumdecay_bf16 | 24.38 | 71.7% | 60.1 | 5.3% |
| gdn_gated_scan_bf16 | 10.23 | 30.1% | 287.9 | 6.7% |
| gdn2_gated_scan | 9.04 | 26.6% | 543.4 | 8.6% |
| Gated Cumulative Decay | 25.75 | 75.7% | 37.9 | 6.9% |
| Gated Delta-Rule Scan | 15.42 | 45.4% | 96.0 | 5.2% |
| Causal Depthwise Conv1D | 29.92 | 88.0% | 34.4 | 1.7% |
| gdn_cumdecay_f16 | 33.93 | 99.8% | 21.6 | 1.4% |
| gdn_gated_scan_f16 | 15.34 | 45.1% | 96.0 | 4.6% |
| gdn_cumdecay_bf16 | 25.11 | 73.9% | 29.2 | 6.0% |
| gdn_gated_scan_bf16 | 15.21 | 44.7% | 96.8 | 2.1% |
| gdn2_gated_scan | 10.35 | 30.4% | 237.4 | 5.8% |
| Gated Cumulative Decay | 26.17 | 77.0% | 1.2 | 0.1% |
| Gated Delta-Rule Scan | 52.33 | 153.9% | 1.5 | 0.1% |
| Causal Depthwise Conv1D | 52.32 | 153.9% | 2.6 | 11.1% |
| gdn_cumdecay_f16 | 19.61 | 57.7% | 1.2 | 0.0% |
| gdn_gated_scan_f16 | 41.86 | 123.1% | 1.5 | 0.1% |
| gdn_cumdecay_bf16 | 15.70 | 46.2% | 1.5 | 0.1% |
| gdn_gated_scan_bf16 | 34.88 | 102.6% | 1.8 | 0.1% |
| gdn2_gated_scan | 61.04 | 179.5% | 1.8 | 0.0% |
| Gated Cumulative Decay | 17.44 | 51.3% | 0.9 | 33.4% |
| Gated Delta-Rule Scan | 32.72 | 96.2% | 1.2 | 0.1% |
| Causal Depthwise Conv1D | 33.64 | 98.9% | 2.0 | 0.1% |
| gdn_cumdecay_f16 | 13.08 | 38.5% | 0.9 | 33.4% |
| gdn_gated_scan_f16 | 26.15 | 76.9% | 1.2 | 0.0% |
| gdn_cumdecay_bf16 | 9.81 | 28.9% | 1.2 | 0.0% |
| gdn_gated_scan_bf16 | 26.15 | 76.9% | 1.2 | 25.0% |
| gdn2_gated_scan | 45.77 | 134.6% | 1.2 | 0.0% |
