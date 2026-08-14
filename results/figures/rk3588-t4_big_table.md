# rk3588-t4_big — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 71.5 | 89.5 | 25.3% | 27.33 | 3.67 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 258.4 | 278.9 | 7.9% | 11.45 | 2.03 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 93.9 | 105.0 | 11.8% | 21.93 | 22.33 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 40.3 | 41.7 | 3.6% | 36.39 | 6.51 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 258.1 | 275.4 | 6.7% | 11.41 | 2.03 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 54.5 | 55.1 | 1.1% | 26.86 | 4.81 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 261.1 | 273.9 | 4.9% | 11.28 | 2.01 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 711.4 | 768.9 | 8.1% | 6.91 | 1.47 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 39.7 | 40.3 | 1.5% | 24.62 | 3.30 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 128.6 | 141.2 | 9.8% | 11.51 | 2.04 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 40.0 | 41.7 | 4.4% | 25.77 | 26.24 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 20.7 | 21.0 | 1.4% | 35.37 | 6.33 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 129.2 | 133.3 | 3.2% | 11.40 | 2.03 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 28.6 | 28.9 | 1.0% | 25.62 | 4.59 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 130.1 | 135.9 | 4.5% | 11.32 | 2.02 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 320.9 | 327.0 | 1.9% | 7.66 | 1.63 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.5 | 1.8 | 19.9% | 20.91 | 2.81 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 1.8 | 1.8 | 3.0% | 43.16 | 4.63 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 3.5 | 3.5 | 1.5% | 39.60 | 9.45 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 1.6 | 1.6 | 0.2% | 13.99 | 2.50 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 1.8 | 1.8 | 0.5% | 33.48 | 4.49 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 1.8 | 1.9 | 4.5% | 12.51 | 2.24 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 2.1 | 2.1 | 0.3% | 28.55 | 3.83 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 2.3 | 2.3 | 1.4% | 47.31 | 7.26 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 1.4 | 1.4 | 0.2% | 10.85 | 1.46 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.5 | 1.5 | 0.4% | 25.15 | 2.70 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 2.1 | 2.1 | 0.7% | 32.92 | 7.86 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 1.5 | 1.5 | 0.2% | 7.86 | 1.41 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.6 | 1.6 | 0.4% | 19.56 | 2.62 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.6 | 1.6 | 0.2% | 7.36 | 1.32 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 1.8 | 1.8 | 2.5% | 17.38 | 2.33 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.6 | 1.6 | 1.7% | 33.72 | 5.17 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 27.33 | 86.2% | 71.5 | 25.3% |
| Gated Delta-Rule Scan | 11.45 | 36.1% | 258.4 | 7.9% |
| Causal Depthwise Conv1D | 21.93 | 69.2% | 93.9 | 11.8% |
| gdn_cumdecay_f16 | 36.39 | 114.8% | 40.3 | 3.6% |
| gdn_gated_scan_f16 | 11.41 | 36.0% | 258.1 | 6.7% |
| gdn_cumdecay_bf16 | 26.86 | 84.7% | 54.5 | 1.1% |
| gdn_gated_scan_bf16 | 11.28 | 35.6% | 261.1 | 4.9% |
| gdn2_gated_scan | 6.91 | 21.8% | 711.4 | 8.1% |
| Gated Cumulative Decay | 24.62 | 77.7% | 39.7 | 1.5% |
| Gated Delta-Rule Scan | 11.51 | 36.3% | 128.6 | 9.8% |
| Causal Depthwise Conv1D | 25.77 | 81.3% | 40.0 | 4.4% |
| gdn_cumdecay_f16 | 35.37 | 111.6% | 20.7 | 1.4% |
| gdn_gated_scan_f16 | 11.40 | 36.0% | 129.2 | 3.2% |
| gdn_cumdecay_bf16 | 25.62 | 80.8% | 28.6 | 1.0% |
| gdn_gated_scan_bf16 | 11.32 | 35.7% | 130.1 | 4.5% |
| gdn2_gated_scan | 7.66 | 24.2% | 320.9 | 1.9% |
| Gated Cumulative Decay | 20.91 | 66.0% | 1.5 | 19.9% |
| Gated Delta-Rule Scan | 43.16 | 136.2% | 1.8 | 3.0% |
| Causal Depthwise Conv1D | 39.60 | 124.9% | 3.5 | 1.5% |
| gdn_cumdecay_f16 | 13.99 | 44.1% | 1.6 | 0.2% |
| gdn_gated_scan_f16 | 33.48 | 105.6% | 1.8 | 0.5% |
| gdn_cumdecay_bf16 | 12.51 | 39.5% | 1.8 | 4.5% |
| gdn_gated_scan_bf16 | 28.55 | 90.1% | 2.1 | 0.3% |
| gdn2_gated_scan | 47.31 | 149.2% | 2.3 | 1.4% |
| Gated Cumulative Decay | 10.85 | 34.2% | 1.4 | 0.2% |
| Gated Delta-Rule Scan | 25.15 | 79.3% | 1.5 | 0.4% |
| Causal Depthwise Conv1D | 32.92 | 103.8% | 2.1 | 0.7% |
| gdn_cumdecay_f16 | 7.86 | 24.8% | 1.5 | 0.2% |
| gdn_gated_scan_f16 | 19.56 | 61.7% | 1.6 | 0.4% |
| gdn_cumdecay_bf16 | 7.36 | 23.2% | 1.6 | 0.2% |
| gdn_gated_scan_bf16 | 17.38 | 54.8% | 1.8 | 2.5% |
| gdn2_gated_scan | 33.72 | 106.4% | 1.6 | 1.7% |
