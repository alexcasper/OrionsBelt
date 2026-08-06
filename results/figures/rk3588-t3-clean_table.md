# rk3588-t3-clean — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 317.4 | 373.4 | 17.6% | 6.15 | 0.83 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 1,016.8 | 1,320.5 | 29.9% | 2.91 | 0.52 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 345.4 | 382.1 | 10.6% | 5.96 | 6.07 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 212.6 | 220.2 | 3.6% | 6.89 | 1.23 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 1,010.7 | 1,200.3 | 18.8% | 2.91 | 0.52 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 253.5 | 268.9 | 6.1% | 5.78 | 1.03 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 927.9 | 1,118.3 | 20.5% | 3.17 | 0.57 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 2,458.6 | 2,582.9 | 5.1% | 2.00 | 0.43 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 119.6 | 120.8 | 1.0% | 8.17 | 1.10 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 204.5 | 207.7 | 1.6% | 7.24 | 1.28 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 142.3 | 144.1 | 1.2% | 7.24 | 7.37 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 78.5 | 81.1 | 3.3% | 9.33 | 1.67 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 205.6 | 212.1 | 3.1% | 7.16 | 1.27 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 135.6 | 138.3 | 1.9% | 5.40 | 0.97 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 208.0 | 212.1 | 2.0% | 7.08 | 1.26 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 283.5 | 291.7 | 2.9% | 8.66 | 1.85 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.5 | 1.8 | 20.0% | 20.92 | 2.81 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 2.0 | 2.0 | 0.0% | 37.36 | 4.01 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 7.0 | 7.0 | 0.0% | 19.62 | 4.68 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 1.8 | 1.8 | 0.1% | 13.08 | 2.34 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 2.6 | 2.6 | 0.0% | 23.25 | 3.12 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 2.9 | 2.9 | 0.0% | 7.85 | 1.40 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 3.8 | 3.8 | 0.0% | 16.10 | 2.16 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 2.9 | 3.2 | 10.0% | 36.62 | 5.62 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 0.9 | 1.2 | 33.4% | 17.44 | 2.34 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.2 | 1.2 | 0.0% | 32.69 | 3.51 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 2.6 | 2.9 | 11.1% | 26.16 | 6.24 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 1.2 | 1.2 | 0.1% | 9.81 | 1.76 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.5 | 1.8 | 19.9% | 20.92 | 2.81 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.8 | 1.8 | 0.0% | 6.54 | 1.17 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 2.0 | 2.3 | 14.3% | 14.95 | 2.01 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.5 | 1.8 | 19.9% | 36.60 | 5.61 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 34.0 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 6.15 | 18.1% | 317.4 | 17.6% |
| Gated Delta-Rule Scan | 2.91 | 8.6% | 1,016.8 | 29.9% |
| Causal Depthwise Conv1D | 5.96 | 17.5% | 345.4 | 10.6% |
| gdn_cumdecay_f16 | 6.89 | 20.3% | 212.6 | 3.6% |
| gdn_gated_scan_f16 | 2.91 | 8.6% | 1,010.7 | 18.8% |
| gdn_cumdecay_bf16 | 5.78 | 17.0% | 253.5 | 6.1% |
| gdn_gated_scan_bf16 | 3.17 | 9.3% | 927.9 | 20.5% |
| gdn2_gated_scan | 2.00 | 5.9% | 2,458.6 | 5.1% |
| Gated Cumulative Decay | 8.17 | 24.0% | 119.6 | 1.0% |
| Gated Delta-Rule Scan | 7.24 | 21.3% | 204.5 | 1.6% |
| Causal Depthwise Conv1D | 7.24 | 21.3% | 142.3 | 1.2% |
| gdn_cumdecay_f16 | 9.33 | 27.4% | 78.5 | 3.3% |
| gdn_gated_scan_f16 | 7.16 | 21.1% | 205.6 | 3.1% |
| gdn_cumdecay_bf16 | 5.40 | 15.9% | 135.6 | 1.9% |
| gdn_gated_scan_bf16 | 7.08 | 20.8% | 208.0 | 2.0% |
| gdn2_gated_scan | 8.66 | 25.5% | 283.5 | 2.9% |
| Gated Cumulative Decay | 20.92 | 61.5% | 1.5 | 20.0% |
| Gated Delta-Rule Scan | 37.36 | 109.9% | 2.0 | 0.0% |
| Causal Depthwise Conv1D | 19.62 | 57.7% | 7.0 | 0.0% |
| gdn_cumdecay_f16 | 13.08 | 38.5% | 1.8 | 0.1% |
| gdn_gated_scan_f16 | 23.25 | 68.4% | 2.6 | 0.0% |
| gdn_cumdecay_bf16 | 7.85 | 23.1% | 2.9 | 0.0% |
| gdn_gated_scan_bf16 | 16.10 | 47.4% | 3.8 | 0.0% |
| gdn2_gated_scan | 36.62 | 107.7% | 2.9 | 10.0% |
| Gated Cumulative Decay | 17.44 | 51.3% | 0.9 | 33.4% |
| Gated Delta-Rule Scan | 32.69 | 96.1% | 1.2 | 0.0% |
| Causal Depthwise Conv1D | 26.16 | 76.9% | 2.6 | 11.1% |
| gdn_cumdecay_f16 | 9.81 | 28.9% | 1.2 | 0.1% |
| gdn_gated_scan_f16 | 20.92 | 61.5% | 1.5 | 19.9% |
| gdn_cumdecay_bf16 | 6.54 | 19.2% | 1.8 | 0.0% |
| gdn_gated_scan_bf16 | 14.95 | 44.0% | 2.0 | 14.3% |
| gdn2_gated_scan | 36.60 | 107.6% | 1.5 | 19.9% |
