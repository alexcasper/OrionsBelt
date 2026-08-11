# rk3588-t4_big — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 91.0 | 93.9 | 3.2% | 21.46 | 2.88 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 247.9 | 265.7 | 7.2% | 11.94 | 2.11 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 106.5 | 115.8 | 8.8% | 19.35 | 19.70 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 41.4 | 43.2 | 4.2% | 35.37 | 6.33 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 249.4 | 272.7 | 9.4% | 11.81 | 2.10 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 57.8 | 58.6 | 1.5% | 25.36 | 4.54 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 249.4 | 280.6 | 12.5% | 11.81 | 2.10 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 689.8 | 749.6 | 8.7% | 7.12 | 1.52 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 38.8 | 44.3 | 14.3% | 25.17 | 3.38 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 120.8 | 126.6 | 4.8% | 12.26 | 2.17 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 39.1 | 39.7 | 1.5% | 26.35 | 26.83 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 19.5 | 20.4 | 4.5% | 37.48 | 6.71 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 125.1 | 126.9 | 1.4% | 11.77 | 2.09 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 27.1 | 27.4 | 1.1% | 27.00 | 4.83 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 122.8 | 127.2 | 3.6% | 11.99 | 2.13 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 294.6 | 301.6 | 2.4% | 8.34 | 1.78 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.5 | 1.8 | 19.9% | 20.91 | 2.81 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 1.8 | 1.9 | 6.5% | 42.74 | 4.59 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 3.6 | 3.6 | 1.4% | 38.31 | 9.14 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 1.6 | 1.7 | 3.0% | 13.99 | 2.50 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 1.8 | 1.8 | 0.5% | 33.43 | 4.49 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 1.6 | 1.8 | 11.8% | 14.01 | 2.51 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 2.1 | 2.1 | 0.3% | 28.59 | 3.84 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 2.2 | 2.3 | 1.9% | 47.50 | 7.29 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 1.4 | 1.4 | 0.2% | 10.85 | 1.46 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.5 | 1.5 | 0.4% | 25.10 | 2.70 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 2.1 | 2.1 | 0.7% | 32.74 | 7.81 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 1.5 | 1.5 | 0.2% | 7.88 | 1.41 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.6 | 1.6 | 0.2% | 19.45 | 2.61 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.6 | 1.6 | 0.2% | 7.37 | 1.32 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 1.8 | 1.8 | 1.0% | 17.44 | 2.34 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.6 | 1.6 | 1.7% | 33.72 | 5.17 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 21.46 | 67.7% | 91.0 | 3.2% |
| Gated Delta-Rule Scan | 11.94 | 37.7% | 247.9 | 7.2% |
| Causal Depthwise Conv1D | 19.35 | 61.0% | 106.5 | 8.8% |
| gdn_cumdecay_f16 | 35.37 | 111.6% | 41.4 | 4.2% |
| gdn_gated_scan_f16 | 11.81 | 37.3% | 249.4 | 9.4% |
| gdn_cumdecay_bf16 | 25.36 | 80.0% | 57.8 | 1.5% |
| gdn_gated_scan_bf16 | 11.81 | 37.3% | 249.4 | 12.5% |
| gdn2_gated_scan | 7.12 | 22.5% | 689.8 | 8.7% |
| Gated Cumulative Decay | 25.17 | 79.4% | 38.8 | 14.3% |
| Gated Delta-Rule Scan | 12.26 | 38.7% | 120.8 | 4.8% |
| Causal Depthwise Conv1D | 26.35 | 83.1% | 39.1 | 1.5% |
| gdn_cumdecay_f16 | 37.48 | 118.2% | 19.5 | 4.5% |
| gdn_gated_scan_f16 | 11.77 | 37.1% | 125.1 | 1.4% |
| gdn_cumdecay_bf16 | 27.00 | 85.2% | 27.1 | 1.1% |
| gdn_gated_scan_bf16 | 11.99 | 37.8% | 122.8 | 3.6% |
| gdn2_gated_scan | 8.34 | 26.3% | 294.6 | 2.4% |
| Gated Cumulative Decay | 20.91 | 66.0% | 1.5 | 19.9% |
| Gated Delta-Rule Scan | 42.74 | 134.8% | 1.8 | 6.5% |
| Causal Depthwise Conv1D | 38.31 | 120.9% | 3.6 | 1.4% |
| gdn_cumdecay_f16 | 13.99 | 44.1% | 1.6 | 3.0% |
| gdn_gated_scan_f16 | 33.43 | 105.5% | 1.8 | 0.5% |
| gdn_cumdecay_bf16 | 14.01 | 44.2% | 1.6 | 11.8% |
| gdn_gated_scan_bf16 | 28.59 | 90.2% | 2.1 | 0.3% |
| gdn2_gated_scan | 47.50 | 149.8% | 2.2 | 1.9% |
| Gated Cumulative Decay | 10.85 | 34.2% | 1.4 | 0.2% |
| Gated Delta-Rule Scan | 25.10 | 79.2% | 1.5 | 0.4% |
| Causal Depthwise Conv1D | 32.74 | 103.3% | 2.1 | 0.7% |
| gdn_cumdecay_f16 | 7.88 | 24.9% | 1.5 | 0.2% |
| gdn_gated_scan_f16 | 19.45 | 61.4% | 1.6 | 0.2% |
| gdn_cumdecay_bf16 | 7.37 | 23.2% | 1.6 | 0.2% |
| gdn_gated_scan_bf16 | 17.44 | 55.0% | 1.8 | 1.0% |
| gdn2_gated_scan | 33.72 | 106.4% | 1.6 | 1.7% |
