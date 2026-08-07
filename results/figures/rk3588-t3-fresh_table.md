# rk3588-t3-fresh — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 92.8 | 96.0 | 3.5% | 21.06 | 2.83 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 278.9 | 294.0 | 5.4% | 10.62 | 1.88 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 110.0 | 115.2 | 4.8% | 18.73 | 19.07 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 39.4 | 40.5 | 3.0% | 37.20 | 6.66 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 281.2 | 305.4 | 8.6% | 10.47 | 1.86 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 57.5 | 58.9 | 2.5% | 25.49 | 4.56 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 280.3 | 296.6 | 5.8% | 10.51 | 1.87 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 547.8 | 577.2 | 5.4% | 8.97 | 1.91 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 34.1 | 36.8 | 7.7% | 28.62 | 3.84 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 97.1 | 99.5 | 2.4% | 15.24 | 2.70 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 36.2 | 36.8 | 1.6% | 28.48 | 28.99 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 19.3 | 19.5 | 1.5% | 38.05 | 6.81 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 98.6 | 100.9 | 2.4% | 14.94 | 2.66 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 27.4 | 27.4 | 0.0% | 26.71 | 4.78 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 98.0 | 103.0 | 5.1% | 15.02 | 2.67 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 212.3 | 222.3 | 4.7% | 11.57 | 2.47 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.2 | 1.2 | 0.1% | 26.17 | 3.51 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 1.5 | 1.8 | 20.0% | 52.33 | 5.62 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 2.6 | 2.9 | 11.1% | 52.31 | 12.48 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 1.2 | 1.5 | 24.9% | 19.61 | 3.51 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 1.5 | 1.5 | 0.1% | 41.86 | 5.62 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 1.5 | 1.5 | 0.1% | 15.70 | 2.81 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 1.8 | 1.8 | 0.1% | 34.88 | 4.68 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 1.8 | 1.8 | 0.0% | 61.04 | 9.36 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 0.9 | 1.2 | 33.4% | 17.44 | 2.34 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.2 | 1.2 | 0.0% | 32.69 | 3.51 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 1.8 | 2.0 | 16.6% | 39.22 | 9.36 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 0.9 | 1.2 | 33.4% | 13.08 | 2.34 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.2 | 1.2 | 0.0% | 26.15 | 3.51 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.2 | 1.2 | 0.0% | 9.81 | 1.76 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 1.2 | 1.5 | 25.0% | 26.15 | 3.51 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.2 | 1.5 | 25.0% | 45.77 | 7.02 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 21.06 | 66.4% | 92.8 | 3.5% |
| Gated Delta-Rule Scan | 10.62 | 33.5% | 278.9 | 5.4% |
| Causal Depthwise Conv1D | 18.73 | 59.1% | 110.0 | 4.8% |
| gdn_cumdecay_f16 | 37.20 | 117.4% | 39.4 | 3.0% |
| gdn_gated_scan_f16 | 10.47 | 33.0% | 281.2 | 8.6% |
| gdn_cumdecay_bf16 | 25.49 | 80.4% | 57.5 | 2.5% |
| gdn_gated_scan_bf16 | 10.51 | 33.2% | 280.3 | 5.8% |
| gdn2_gated_scan | 8.97 | 28.3% | 547.8 | 5.4% |
| Gated Cumulative Decay | 28.62 | 90.3% | 34.1 | 7.7% |
| Gated Delta-Rule Scan | 15.24 | 48.1% | 97.1 | 2.4% |
| Causal Depthwise Conv1D | 28.48 | 89.8% | 36.2 | 1.6% |
| gdn_cumdecay_f16 | 38.05 | 120.0% | 19.3 | 1.5% |
| gdn_gated_scan_f16 | 14.94 | 47.1% | 98.6 | 2.4% |
| gdn_cumdecay_bf16 | 26.71 | 84.3% | 27.4 | 0.0% |
| gdn_gated_scan_bf16 | 15.02 | 47.4% | 98.0 | 5.1% |
| gdn2_gated_scan | 11.57 | 36.5% | 212.3 | 4.7% |
| Gated Cumulative Decay | 26.17 | 82.6% | 1.2 | 0.1% |
| Gated Delta-Rule Scan | 52.33 | 165.1% | 1.5 | 20.0% |
| Causal Depthwise Conv1D | 52.31 | 165.0% | 2.6 | 11.1% |
| gdn_cumdecay_f16 | 19.61 | 61.9% | 1.2 | 24.9% |
| gdn_gated_scan_f16 | 41.86 | 132.1% | 1.5 | 0.1% |
| gdn_cumdecay_bf16 | 15.70 | 49.5% | 1.5 | 0.1% |
| gdn_gated_scan_bf16 | 34.88 | 110.0% | 1.8 | 0.1% |
| gdn2_gated_scan | 61.04 | 192.6% | 1.8 | 0.0% |
| Gated Cumulative Decay | 17.44 | 55.0% | 0.9 | 33.4% |
| Gated Delta-Rule Scan | 32.69 | 103.1% | 1.2 | 0.0% |
| Causal Depthwise Conv1D | 39.22 | 123.7% | 1.8 | 16.6% |
| gdn_cumdecay_f16 | 13.08 | 41.3% | 0.9 | 33.4% |
| gdn_gated_scan_f16 | 26.15 | 82.5% | 1.2 | 0.0% |
| gdn_cumdecay_bf16 | 9.81 | 30.9% | 1.2 | 0.0% |
| gdn_gated_scan_bf16 | 26.15 | 82.5% | 1.2 | 25.0% |
| gdn2_gated_scan | 45.77 | 144.4% | 1.2 | 25.0% |
