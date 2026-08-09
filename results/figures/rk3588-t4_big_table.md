# rk3588-t4_big — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 87.8 | 100.6 | 14.6% | 22.25 | 2.99 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 256.7 | 280.3 | 9.2% | 11.53 | 2.04 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 108.2 | 112.3 | 3.8% | 19.04 | 19.38 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 40.5 | 40.8 | 0.7% | 36.13 | 6.47 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 258.1 | 275.1 | 6.6% | 11.41 | 2.03 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 57.5 | 58.0 | 1.0% | 25.49 | 4.56 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 257.8 | 273.6 | 6.1% | 11.42 | 2.03 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 688.1 | 747.9 | 8.7% | 7.14 | 1.52 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 38.5 | 39.7 | 3.0% | 25.36 | 3.40 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 131.3 | 140.0 | 6.7% | 11.28 | 2.00 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 45.5 | 46.1 | 1.3% | 22.64 | 23.04 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 20.4 | 20.7 | 1.4% | 35.87 | 6.42 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 128.9 | 132.7 | 2.9% | 11.42 | 2.03 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 28.3 | 28.6 | 1.0% | 25.89 | 4.63 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 130.7 | 139.7 | 6.9% | 11.27 | 2.01 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 297.5 | 315.3 | 6.0% | 8.26 | 1.76 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.5 | 1.8 | 19.9% | 20.91 | 2.81 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 1.8 | 1.8 | 1.3% | 41.78 | 4.49 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 3.4 | 3.5 | 1.4% | 40.10 | 9.57 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 1.6 | 1.7 | 2.7% | 13.99 | 2.50 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 1.8 | 1.8 | 0.2% | 33.75 | 4.53 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 1.8 | 1.8 | 0.2% | 12.56 | 2.25 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 2.1 | 2.1 | 0.3% | 28.59 | 3.84 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 2.3 | 2.3 | 1.6% | 47.43 | 7.28 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 1.4 | 1.4 | 0.2% | 10.90 | 1.46 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.5 | 1.5 | 0.4% | 25.05 | 2.69 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 2.1 | 2.1 | 0.6% | 33.25 | 7.93 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 1.4 | 1.4 | 0.2% | 7.91 | 1.42 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.6 | 1.6 | 0.2% | 19.56 | 2.62 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.5 | 1.5 | 0.2% | 7.40 | 1.32 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 1.8 | 1.8 | 2.5% | 17.35 | 2.33 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.6 | 1.6 | 0.9% | 33.84 | 5.19 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 22.25 | 70.2% | 87.8 | 14.6% |
| Gated Delta-Rule Scan | 11.53 | 36.4% | 256.7 | 9.2% |
| Causal Depthwise Conv1D | 19.04 | 60.1% | 108.2 | 3.8% |
| gdn_cumdecay_f16 | 36.13 | 114.0% | 40.5 | 0.7% |
| gdn_gated_scan_f16 | 11.41 | 36.0% | 258.1 | 6.6% |
| gdn_cumdecay_bf16 | 25.49 | 80.4% | 57.5 | 1.0% |
| gdn_gated_scan_bf16 | 11.42 | 36.0% | 257.8 | 6.1% |
| gdn2_gated_scan | 7.14 | 22.5% | 688.1 | 8.7% |
| Gated Cumulative Decay | 25.36 | 80.0% | 38.5 | 3.0% |
| Gated Delta-Rule Scan | 11.28 | 35.6% | 131.3 | 6.7% |
| Causal Depthwise Conv1D | 22.64 | 71.4% | 45.5 | 1.3% |
| gdn_cumdecay_f16 | 35.87 | 113.2% | 20.4 | 1.4% |
| gdn_gated_scan_f16 | 11.42 | 36.0% | 128.9 | 2.9% |
| gdn_cumdecay_bf16 | 25.89 | 81.7% | 28.3 | 1.0% |
| gdn_gated_scan_bf16 | 11.27 | 35.6% | 130.7 | 6.9% |
| gdn2_gated_scan | 8.26 | 26.1% | 297.5 | 6.0% |
| Gated Cumulative Decay | 20.91 | 66.0% | 1.5 | 19.9% |
| Gated Delta-Rule Scan | 41.78 | 131.8% | 1.8 | 1.3% |
| Causal Depthwise Conv1D | 40.10 | 126.5% | 3.4 | 1.4% |
| gdn_cumdecay_f16 | 13.99 | 44.1% | 1.6 | 2.7% |
| gdn_gated_scan_f16 | 33.75 | 106.5% | 1.8 | 0.2% |
| gdn_cumdecay_bf16 | 12.56 | 39.6% | 1.8 | 0.2% |
| gdn_gated_scan_bf16 | 28.59 | 90.2% | 2.1 | 0.3% |
| gdn2_gated_scan | 47.43 | 149.6% | 2.3 | 1.6% |
| Gated Cumulative Decay | 10.90 | 34.4% | 1.4 | 0.2% |
| Gated Delta-Rule Scan | 25.05 | 79.0% | 1.5 | 0.4% |
| Causal Depthwise Conv1D | 33.25 | 104.9% | 2.1 | 0.6% |
| gdn_cumdecay_f16 | 7.91 | 25.0% | 1.4 | 0.2% |
| gdn_gated_scan_f16 | 19.56 | 61.7% | 1.6 | 0.2% |
| gdn_cumdecay_bf16 | 7.40 | 23.3% | 1.5 | 0.2% |
| gdn_gated_scan_bf16 | 17.35 | 54.7% | 1.8 | 2.5% |
| gdn2_gated_scan | 33.84 | 106.8% | 1.6 | 0.9% |
