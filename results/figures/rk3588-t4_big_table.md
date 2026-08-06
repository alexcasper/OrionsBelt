# rk3588-t4_big — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 86.6 | 108.2 | 24.9% | 22.55 | 3.03 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 256.1 | 273.0 | 6.6% | 11.56 | 2.05 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 100.3 | 110.3 | 9.9% | 20.53 | 20.90 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 41.4 | 41.7 | 0.7% | 35.37 | 6.33 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 257.3 | 275.4 | 7.0% | 11.45 | 2.04 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 58.3 | 58.6 | 0.5% | 25.11 | 4.49 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 257.3 | 268.1 | 4.2% | 11.45 | 2.04 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 458.8 | 478.4 | 4.3% | 10.71 | 2.29 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 37.0 | 37.6 | 1.6% | 26.36 | 3.54 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 123.1 | 137.1 | 11.4% | 12.02 | 2.13 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 52.5 | 55.7 | 6.1% | 19.62 | 19.97 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 17.8 | 17.8 | 0.0% | 41.16 | 7.37 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 135.1 | 137.7 | 1.9% | 10.90 | 1.94 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 28.3 | 29.8 | 5.2% | 25.89 | 4.63 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 131.0 | 135.3 | 3.3% | 11.24 | 2.00 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 200.1 | 217.3 | 8.6% | 12.28 | 2.62 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.5 | 1.8 | 20.0% | 20.92 | 2.81 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 2.0 | 2.3 | 14.3% | 37.36 | 4.01 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 2.9 | 3.2 | 10.0% | 47.08 | 11.23 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 1.8 | 1.8 | 0.1% | 13.08 | 2.34 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 1.8 | 2.0 | 16.6% | 34.85 | 4.68 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 1.8 | 2.0 | 16.7% | 13.08 | 2.34 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 2.0 | 2.3 | 14.3% | 29.89 | 4.01 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 2.3 | 2.3 | 0.0% | 45.78 | 7.02 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 1.5 | 1.5 | 0.1% | 10.47 | 1.40 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.5 | 1.8 | 20.0% | 26.15 | 2.81 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 2.3 | 2.3 | 0.0% | 29.43 | 7.02 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 1.5 | 1.8 | 20.0% | 7.85 | 1.40 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.8 | 1.8 | 0.1% | 17.44 | 2.34 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.5 | 1.8 | 20.0% | 7.84 | 1.40 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 1.8 | 1.8 | 0.1% | 17.44 | 2.34 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.8 | 1.8 | 0.1% | 30.52 | 4.68 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 34.0 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 22.55 | 66.3% | 86.6 | 24.9% |
| Gated Delta-Rule Scan | 11.56 | 34.0% | 256.1 | 6.6% |
| Causal Depthwise Conv1D | 20.53 | 60.4% | 100.3 | 9.9% |
| gdn_cumdecay_f16 | 35.37 | 104.0% | 41.4 | 0.7% |
| gdn_gated_scan_f16 | 11.45 | 33.7% | 257.3 | 7.0% |
| gdn_cumdecay_bf16 | 25.11 | 73.9% | 58.3 | 0.5% |
| gdn_gated_scan_bf16 | 11.45 | 33.7% | 257.3 | 4.2% |
| gdn2_gated_scan | 10.71 | 31.5% | 458.8 | 4.3% |
| Gated Cumulative Decay | 26.36 | 77.5% | 37.0 | 1.6% |
| Gated Delta-Rule Scan | 12.02 | 35.4% | 123.1 | 11.4% |
| Causal Depthwise Conv1D | 19.62 | 57.7% | 52.5 | 6.1% |
| gdn_cumdecay_f16 | 41.16 | 121.1% | 17.8 | 0.0% |
| gdn_gated_scan_f16 | 10.90 | 32.1% | 135.1 | 1.9% |
| gdn_cumdecay_bf16 | 25.89 | 76.1% | 28.3 | 5.2% |
| gdn_gated_scan_bf16 | 11.24 | 33.1% | 131.0 | 3.3% |
| gdn2_gated_scan | 12.28 | 36.1% | 200.1 | 8.6% |
| Gated Cumulative Decay | 20.92 | 61.5% | 1.5 | 20.0% |
| Gated Delta-Rule Scan | 37.36 | 109.9% | 2.0 | 14.3% |
| Causal Depthwise Conv1D | 47.08 | 138.5% | 2.9 | 10.0% |
| gdn_cumdecay_f16 | 13.08 | 38.5% | 1.8 | 0.1% |
| gdn_gated_scan_f16 | 34.85 | 102.5% | 1.8 | 16.6% |
| gdn_cumdecay_bf16 | 13.08 | 38.5% | 1.8 | 16.7% |
| gdn_gated_scan_bf16 | 29.89 | 87.9% | 2.0 | 14.3% |
| gdn2_gated_scan | 45.78 | 134.6% | 2.3 | 0.0% |
| Gated Cumulative Decay | 10.47 | 30.8% | 1.5 | 0.1% |
| Gated Delta-Rule Scan | 26.15 | 76.9% | 1.5 | 20.0% |
| Causal Depthwise Conv1D | 29.43 | 86.6% | 2.3 | 0.0% |
| gdn_cumdecay_f16 | 7.85 | 23.1% | 1.5 | 20.0% |
| gdn_gated_scan_f16 | 17.44 | 51.3% | 1.8 | 0.1% |
| gdn_cumdecay_bf16 | 7.84 | 23.1% | 1.5 | 20.0% |
| gdn_gated_scan_bf16 | 17.44 | 51.3% | 1.8 | 0.1% |
| gdn2_gated_scan | 30.52 | 89.8% | 1.8 | 0.1% |
