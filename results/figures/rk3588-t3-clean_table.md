# rk3588-t3-clean — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 92.2 | 98.3 | 6.6% | 21.19 | 2.84 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 282.1 | 298.4 | 5.8% | 10.49 | 1.86 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 98.0 | 105.9 | 8.0% | 21.02 | 21.40 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 42.3 | 44.3 | 4.8% | 34.63 | 6.20 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 280.3 | 288.2 | 2.8% | 10.51 | 1.87 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 59.2 | 59.5 | 0.5% | 24.74 | 4.43 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 273.6 | 280.6 | 2.6% | 10.76 | 1.92 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 741.2 | 843.3 | 13.8% | 6.63 | 1.41 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 36.5 | 36.8 | 0.8% | 26.78 | 3.59 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 127.8 | 130.4 | 2.1% | 11.59 | 2.05 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 51.0 | 51.9 | 1.7% | 20.18 | 20.54 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 20.1 | 20.4 | 1.5% | 36.39 | 6.51 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 128.9 | 132.1 | 2.5% | 11.42 | 2.03 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 30.0 | 30.6 | 1.9% | 24.38 | 4.36 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 128.9 | 131.6 | 2.0% | 11.42 | 2.03 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 321.4 | 326.4 | 1.5% | 7.64 | 1.63 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.0 | 1.0 | 0.8% | 29.47 | 3.96 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 1.4 | 1.4 | 1.3% | 55.07 | 5.91 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 3.3 | 3.3 | 1.0% | 41.70 | 9.95 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 1.1 | 1.1 | 1.6% | 21.38 | 3.83 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 1.3 | 1.3 | 0.4% | 45.49 | 6.11 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 1.3 | 1.3 | 0.2% | 17.28 | 3.09 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 1.7 | 1.7 | 1.6% | 36.52 | 4.90 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 1.9 | 1.9 | 0.9% | 57.40 | 8.80 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 0.9 | 0.9 | 0.7% | 17.26 | 2.32 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.0 | 1.0 | 1.4% | 37.69 | 4.05 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 2.1 | 2.1 | 1.1% | 32.43 | 7.74 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 0.9 | 0.9 | 0.3% | 12.78 | 2.29 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.1 | 1.1 | 1.1% | 28.05 | 3.76 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.1 | 1.1 | 0.8% | 10.66 | 1.91 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 1.2 | 1.2 | 0.7% | 25.03 | 3.36 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.2 | 1.2 | 0.8% | 46.24 | 7.09 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 21.19 | 66.8% | 92.2 | 6.6% |
| Gated Delta-Rule Scan | 10.49 | 33.1% | 282.1 | 5.8% |
| Causal Depthwise Conv1D | 21.02 | 66.3% | 98.0 | 8.0% |
| gdn_cumdecay_f16 | 34.63 | 109.2% | 42.3 | 4.8% |
| gdn_gated_scan_f16 | 10.51 | 33.2% | 280.3 | 2.8% |
| gdn_cumdecay_bf16 | 24.74 | 78.0% | 59.2 | 0.5% |
| gdn_gated_scan_bf16 | 10.76 | 33.9% | 273.6 | 2.6% |
| gdn2_gated_scan | 6.63 | 20.9% | 741.2 | 13.8% |
| Gated Cumulative Decay | 26.78 | 84.5% | 36.5 | 0.8% |
| Gated Delta-Rule Scan | 11.59 | 36.6% | 127.8 | 2.1% |
| Causal Depthwise Conv1D | 20.18 | 63.7% | 51.0 | 1.7% |
| gdn_cumdecay_f16 | 36.39 | 114.8% | 20.1 | 1.5% |
| gdn_gated_scan_f16 | 11.42 | 36.0% | 128.9 | 2.5% |
| gdn_cumdecay_bf16 | 24.38 | 76.9% | 30.0 | 1.9% |
| gdn_gated_scan_bf16 | 11.42 | 36.0% | 128.9 | 2.0% |
| gdn2_gated_scan | 7.64 | 24.1% | 321.4 | 1.5% |
| Gated Cumulative Decay | 29.47 | 93.0% | 1.0 | 0.8% |
| Gated Delta-Rule Scan | 55.07 | 173.7% | 1.4 | 1.3% |
| Causal Depthwise Conv1D | 41.70 | 131.5% | 3.3 | 1.0% |
| gdn_cumdecay_f16 | 21.38 | 67.4% | 1.1 | 1.6% |
| gdn_gated_scan_f16 | 45.49 | 143.5% | 1.3 | 0.4% |
| gdn_cumdecay_bf16 | 17.28 | 54.5% | 1.3 | 0.2% |
| gdn_gated_scan_bf16 | 36.52 | 115.2% | 1.7 | 1.6% |
| gdn2_gated_scan | 57.40 | 181.1% | 1.9 | 0.9% |
| Gated Cumulative Decay | 17.26 | 54.4% | 0.9 | 0.7% |
| Gated Delta-Rule Scan | 37.69 | 118.9% | 1.0 | 1.4% |
| Causal Depthwise Conv1D | 32.43 | 102.3% | 2.1 | 1.1% |
| gdn_cumdecay_f16 | 12.78 | 40.3% | 0.9 | 0.3% |
| gdn_gated_scan_f16 | 28.05 | 88.5% | 1.1 | 1.1% |
| gdn_cumdecay_bf16 | 10.66 | 33.6% | 1.1 | 0.8% |
| gdn_gated_scan_bf16 | 25.03 | 79.0% | 1.2 | 0.7% |
| gdn2_gated_scan | 46.24 | 145.9% | 1.2 | 0.8% |
