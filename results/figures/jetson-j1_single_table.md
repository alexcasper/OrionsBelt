# jetson-j1_single — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 1,139.1 | 1,316.4 | 15.6% | 1.71 | 0.23 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 2,624.9 | 2,775.1 | 5.7% | 1.13 | 0.20 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 1,633.1 | 1,747.6 | 7.0% | 1.26 | 1.28 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 674.5 | 823.2 | 22.0% | 2.17 | 0.39 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 2,711.9 | 3,224.4 | 18.9% | 1.09 | 0.19 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 893.2 | 1,073.3 | 20.2% | 1.64 | 0.29 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 2,590.7 | 2,725.8 | 5.2% | 1.14 | 0.20 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 4,380.5 | 4,440.5 | 1.4% | 1.12 | 0.24 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 276.7 | 329.6 | 19.1% | 3.53 | 0.47 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 554.3 | 736.3 | 32.8% | 2.67 | 0.47 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 378.0 | 440.6 | 16.6% | 2.72 | 2.77 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 297.8 | 318.9 | 7.1% | 2.46 | 0.44 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 585.5 | 743.3 | 26.9% | 2.51 | 0.45 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 363.4 | 426.4 | 17.3% | 2.02 | 0.36 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 504.9 | 610.9 | 21.0% | 2.92 | 0.52 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 1,212.7 | 1,366.9 | 12.7% | 2.03 | 0.43 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 5.8 | 6.0 | 3.6% | 5.23 | 0.70 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 10.2 | 10.6 | 4.1% | 7.47 | 0.80 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 36.2 | 39.2 | 8.3% | 3.79 | 0.91 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 6.8 | 7.1 | 4.6% | 3.35 | 0.60 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 11.2 | 11.7 | 4.2% | 5.45 | 0.73 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 8.4 | 8.6 | 3.1% | 2.73 | 0.49 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 11.9 | 12.6 | 5.7% | 5.14 | 0.69 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 13.3 | 17.9 | 34.5% | 8.04 | 1.23 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 4.2 | 4.3 | 2.5% | 3.62 | 0.49 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 6.0 | 6.2 | 3.5% | 6.37 | 0.68 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 21.4 | 22.2 | 3.7% | 3.21 | 0.77 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 4.4 | 4.5 | 2.4% | 2.62 | 0.47 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 6.7 | 6.9 | 3.9% | 4.58 | 0.61 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 5.3 | 5.4 | 2.0% | 2.18 | 0.39 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 7.1 | 7.4 | 4.4% | 4.28 | 0.57 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 7.3 | 7.6 | 3.6% | 7.32 | 1.12 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 25.6 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 1.71 | 6.7% | 1,139.1 | 15.6% |
| Gated Delta-Rule Scan | 1.13 | 4.4% | 2,624.9 | 5.7% |
| Causal Depthwise Conv1D | 1.26 | 4.9% | 1,633.1 | 7.0% |
| gdn_cumdecay_f16 | 2.17 | 8.5% | 674.5 | 22.0% |
| gdn_gated_scan_f16 | 1.09 | 4.3% | 2,711.9 | 18.9% |
| gdn_cumdecay_bf16 | 1.64 | 6.4% | 893.2 | 20.2% |
| gdn_gated_scan_bf16 | 1.14 | 4.5% | 2,590.7 | 5.2% |
| gdn2_gated_scan | 1.12 | 4.4% | 4,380.5 | 1.4% |
| Gated Cumulative Decay | 3.53 | 13.8% | 276.7 | 19.1% |
| Gated Delta-Rule Scan | 2.67 | 10.4% | 554.3 | 32.8% |
| Causal Depthwise Conv1D | 2.72 | 10.6% | 378.0 | 16.6% |
| gdn_cumdecay_f16 | 2.46 | 9.6% | 297.8 | 7.1% |
| gdn_gated_scan_f16 | 2.51 | 9.8% | 585.5 | 26.9% |
| gdn_cumdecay_bf16 | 2.02 | 7.9% | 363.4 | 17.3% |
| gdn_gated_scan_bf16 | 2.92 | 11.4% | 504.9 | 21.0% |
| gdn2_gated_scan | 2.03 | 7.9% | 1,212.7 | 12.7% |
| Gated Cumulative Decay | 5.23 | 20.4% | 5.8 | 3.6% |
| Gated Delta-Rule Scan | 7.47 | 29.2% | 10.2 | 4.1% |
| Causal Depthwise Conv1D | 3.79 | 14.8% | 36.2 | 8.3% |
| gdn_cumdecay_f16 | 3.35 | 13.1% | 6.8 | 4.6% |
| gdn_gated_scan_f16 | 5.45 | 21.3% | 11.2 | 4.2% |
| gdn_cumdecay_bf16 | 2.73 | 10.7% | 8.4 | 3.1% |
| gdn_gated_scan_bf16 | 5.14 | 20.1% | 11.9 | 5.7% |
| gdn2_gated_scan | 8.04 | 31.4% | 13.3 | 34.5% |
| Gated Cumulative Decay | 3.62 | 14.1% | 4.2 | 2.5% |
| Gated Delta-Rule Scan | 6.37 | 24.9% | 6.0 | 3.5% |
| Causal Depthwise Conv1D | 3.21 | 12.5% | 21.4 | 3.7% |
| gdn_cumdecay_f16 | 2.62 | 10.2% | 4.4 | 2.4% |
| gdn_gated_scan_f16 | 4.58 | 17.9% | 6.7 | 3.9% |
| gdn_cumdecay_bf16 | 2.18 | 8.5% | 5.3 | 2.0% |
| gdn_gated_scan_bf16 | 4.28 | 16.7% | 7.1 | 4.4% |
| gdn2_gated_scan | 7.32 | 28.6% | 7.3 | 3.6% |
