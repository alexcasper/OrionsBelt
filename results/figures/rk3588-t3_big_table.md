# rk3588-t3_big — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 89.8 | 94.5 | 5.2% | 21.74 | 2.92 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 267.5 | 284.1 | 6.2% | 11.07 | 1.96 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 95.4 | 99.5 | 4.3% | 21.60 | 21.99 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 42.0 | 42.9 | 2.1% | 34.87 | 6.24 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 276.5 | 296.4 | 7.2% | 10.65 | 1.90 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 59.5 | 62.1 | 4.4% | 24.62 | 4.41 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 269.2 | 298.7 | 10.9% | 10.94 | 1.95 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 533.5 | 591.5 | 10.9% | 9.21 | 1.97 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 35.3 | 37.9 | 7.4% | 27.67 | 3.71 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 96.3 | 98.9 | 2.7% | 15.38 | 2.72 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 34.4 | 35.3 | 2.5% | 29.92 | 30.46 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 19.0 | 19.3 | 1.5% | 38.63 | 6.91 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 96.8 | 99.8 | 3.0% | 15.21 | 2.71 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 26.3 | 26.8 | 2.2% | 27.90 | 4.99 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 97.4 | 100.0 | 2.7% | 15.11 | 2.69 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 208.6 | 214.4 | 2.8% | 11.78 | 2.51 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.2 | 1.2 | 0.1% | 26.17 | 3.51 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 1.5 | 1.5 | 0.1% | 52.33 | 5.62 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 2.6 | 2.9 | 11.1% | 52.32 | 12.48 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 1.2 | 1.2 | 0.0% | 19.61 | 3.51 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 1.5 | 1.5 | 0.1% | 41.86 | 5.62 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 1.5 | 1.5 | 0.1% | 15.70 | 2.81 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 1.8 | 1.8 | 0.1% | 34.88 | 4.68 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 1.5 | 1.8 | 20.0% | 73.20 | 11.23 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 0.9 | 1.2 | 33.3% | 17.44 | 2.34 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.2 | 1.2 | 0.1% | 32.72 | 3.51 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 1.8 | 2.0 | 16.6% | 39.22 | 9.36 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 0.9 | 1.2 | 33.4% | 13.08 | 2.34 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.2 | 1.2 | 0.0% | 26.15 | 3.51 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.2 | 1.2 | 0.0% | 9.81 | 1.76 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 1.2 | 1.5 | 25.0% | 26.15 | 3.51 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.2 | 1.2 | 0.0% | 45.77 | 7.02 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 34.0 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 21.74 | 63.9% | 89.8 | 5.2% |
| Gated Delta-Rule Scan | 11.07 | 32.6% | 267.5 | 6.2% |
| Causal Depthwise Conv1D | 21.60 | 63.5% | 95.4 | 4.3% |
| gdn_cumdecay_f16 | 34.87 | 102.6% | 42.0 | 2.1% |
| gdn_gated_scan_f16 | 10.65 | 31.3% | 276.5 | 7.2% |
| gdn_cumdecay_bf16 | 24.62 | 72.4% | 59.5 | 4.4% |
| gdn_gated_scan_bf16 | 10.94 | 32.2% | 269.2 | 10.9% |
| gdn2_gated_scan | 9.21 | 27.1% | 533.5 | 10.9% |
| Gated Cumulative Decay | 27.67 | 81.4% | 35.3 | 7.4% |
| Gated Delta-Rule Scan | 15.38 | 45.2% | 96.3 | 2.7% |
| Causal Depthwise Conv1D | 29.92 | 88.0% | 34.4 | 2.5% |
| gdn_cumdecay_f16 | 38.63 | 113.6% | 19.0 | 1.5% |
| gdn_gated_scan_f16 | 15.21 | 44.7% | 96.8 | 3.0% |
| gdn_cumdecay_bf16 | 27.90 | 82.1% | 26.3 | 2.2% |
| gdn_gated_scan_bf16 | 15.11 | 44.4% | 97.4 | 2.7% |
| gdn2_gated_scan | 11.78 | 34.6% | 208.6 | 2.8% |
| Gated Cumulative Decay | 26.17 | 77.0% | 1.2 | 0.1% |
| Gated Delta-Rule Scan | 52.33 | 153.9% | 1.5 | 0.1% |
| Causal Depthwise Conv1D | 52.32 | 153.9% | 2.6 | 11.1% |
| gdn_cumdecay_f16 | 19.61 | 57.7% | 1.2 | 0.0% |
| gdn_gated_scan_f16 | 41.86 | 123.1% | 1.5 | 0.1% |
| gdn_cumdecay_bf16 | 15.70 | 46.2% | 1.5 | 0.1% |
| gdn_gated_scan_bf16 | 34.88 | 102.6% | 1.8 | 0.1% |
| gdn2_gated_scan | 73.20 | 215.3% | 1.5 | 20.0% |
| Gated Cumulative Decay | 17.44 | 51.3% | 0.9 | 33.3% |
| Gated Delta-Rule Scan | 32.72 | 96.2% | 1.2 | 0.1% |
| Causal Depthwise Conv1D | 39.22 | 115.4% | 1.8 | 16.6% |
| gdn_cumdecay_f16 | 13.08 | 38.5% | 0.9 | 33.4% |
| gdn_gated_scan_f16 | 26.15 | 76.9% | 1.2 | 0.0% |
| gdn_cumdecay_bf16 | 9.81 | 28.9% | 1.2 | 0.0% |
| gdn_gated_scan_bf16 | 26.15 | 76.9% | 1.2 | 25.0% |
| gdn2_gated_scan | 45.77 | 134.6% | 1.2 | 0.0% |
