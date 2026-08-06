# jetson-j1_omp — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 514.1 | 561.1 | 9.1% | 3.80 | 0.51 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 1,010.1 | 1,169.3 | 15.8% | 2.93 | 0.52 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 581.8 | 637.5 | 9.6% | 3.54 | 3.60 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 364.7 | 444.8 | 22.0% | 4.02 | 0.72 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 1,002.5 | 1,190.3 | 18.7% | 2.94 | 0.52 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 362.7 | 390.6 | 7.7% | 4.04 | 0.72 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 1,005.9 | 1,065.4 | 5.9% | 2.93 | 0.52 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 1,662.2 | 1,756.6 | 5.7% | 2.96 | 0.63 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 208.5 | 239.4 | 14.8% | 4.68 | 0.63 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 285.9 | 955.6 | 234.3% | 5.18 | 0.92 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 201.4 | 223.2 | 10.9% | 5.12 | 5.21 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 165.4 | 170.5 | 3.1% | 4.43 | 0.79 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 263.6 | 320.9 | 21.8% | 5.59 | 0.99 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 167.1 | 173.3 | 3.7% | 4.38 | 0.78 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 263.0 | 303.6 | 15.4% | 5.60 | 1.00 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 409.4 | 465.8 | 13.8% | 6.00 | 1.28 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 3.8 | 5.0 | 31.5% | 8.03 | 1.08 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 5.6 | 5.8 | 2.8% | 13.56 | 1.46 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 10.9 | 11.3 | 3.8% | 12.62 | 3.01 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 3.9 | 4.0 | 1.3% | 5.86 | 1.05 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 5.1 | 5.1 | 1.0% | 12.08 | 1.62 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 4.1 | 4.3 | 3.8% | 5.56 | 1.00 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 5.4 | 5.6 | 3.9% | 11.38 | 1.53 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 6.9 | 7.1 | 2.3% | 15.42 | 2.37 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 3.1 | 3.2 | 3.3% | 4.88 | 0.66 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 3.8 | 3.9 | 2.7% | 10.03 | 1.08 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 9.3 | 10.2 | 8.9% | 7.37 | 1.76 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 3.1 | 3.2 | 3.3% | 3.66 | 0.66 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 4.1 | 4.2 | 2.6% | 7.51 | 1.01 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 3.7 | 3.8 | 2.8% | 3.09 | 0.55 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 4.2 | 4.3 | 3.7% | 7.32 | 0.98 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 4.8 | 5.0 | 4.3% | 11.15 | 1.71 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 25.6 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 3.80 | 14.8% | 514.1 | 9.1% |
| Gated Delta-Rule Scan | 2.93 | 11.4% | 1,010.1 | 15.8% |
| Causal Depthwise Conv1D | 3.54 | 13.8% | 581.8 | 9.6% |
| gdn_cumdecay_f16 | 4.02 | 15.7% | 364.7 | 22.0% |
| gdn_gated_scan_f16 | 2.94 | 11.5% | 1,002.5 | 18.7% |
| gdn_cumdecay_bf16 | 4.04 | 15.8% | 362.7 | 7.7% |
| gdn_gated_scan_bf16 | 2.93 | 11.4% | 1,005.9 | 5.9% |
| gdn2_gated_scan | 2.96 | 11.6% | 1,662.2 | 5.7% |
| Gated Cumulative Decay | 4.68 | 18.3% | 208.5 | 14.8% |
| Gated Delta-Rule Scan | 5.18 | 20.2% | 285.9 | 234.3% |
| Causal Depthwise Conv1D | 5.12 | 20.0% | 201.4 | 10.9% |
| gdn_cumdecay_f16 | 4.43 | 17.3% | 165.4 | 3.1% |
| gdn_gated_scan_f16 | 5.59 | 21.8% | 263.6 | 21.8% |
| gdn_cumdecay_bf16 | 4.38 | 17.1% | 167.1 | 3.7% |
| gdn_gated_scan_bf16 | 5.60 | 21.9% | 263.0 | 15.4% |
| gdn2_gated_scan | 6.00 | 23.4% | 409.4 | 13.8% |
| Gated Cumulative Decay | 8.03 | 31.4% | 3.8 | 31.5% |
| Gated Delta-Rule Scan | 13.56 | 53.0% | 5.6 | 2.8% |
| Causal Depthwise Conv1D | 12.62 | 49.3% | 10.9 | 3.8% |
| gdn_cumdecay_f16 | 5.86 | 22.9% | 3.9 | 1.3% |
| gdn_gated_scan_f16 | 12.08 | 47.2% | 5.1 | 1.0% |
| gdn_cumdecay_bf16 | 5.56 | 21.7% | 4.1 | 3.8% |
| gdn_gated_scan_bf16 | 11.38 | 44.5% | 5.4 | 3.9% |
| gdn2_gated_scan | 15.42 | 60.2% | 6.9 | 2.3% |
| Gated Cumulative Decay | 4.88 | 19.1% | 3.1 | 3.3% |
| Gated Delta-Rule Scan | 10.03 | 39.2% | 3.8 | 2.7% |
| Causal Depthwise Conv1D | 7.37 | 28.8% | 9.3 | 8.9% |
| gdn_cumdecay_f16 | 3.66 | 14.3% | 3.1 | 3.3% |
| gdn_gated_scan_f16 | 7.51 | 29.3% | 4.1 | 2.6% |
| gdn_cumdecay_bf16 | 3.09 | 12.1% | 3.7 | 2.8% |
| gdn_gated_scan_bf16 | 7.32 | 28.6% | 4.2 | 3.7% |
| gdn2_gated_scan | 11.15 | 43.6% | 4.8 | 4.3% |
