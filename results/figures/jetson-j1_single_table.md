# jetson-j1_single — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 1,285.9 | 1,381.8 | 7.5% | 1.52 | 0.20 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 2,833.2 | 3,017.3 | 6.5% | 1.04 | 0.19 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 1,772.1 | 2,041.4 | 15.2% | 1.16 | 1.18 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 797.3 | 944.0 | 18.4% | 1.84 | 0.33 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 2,844.5 | 3,308.4 | 16.3% | 1.04 | 0.18 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 971.3 | 1,137.9 | 17.2% | 1.51 | 0.27 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 2,832.7 | 3,104.5 | 9.6% | 1.04 | 0.19 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 4,470.5 | 4,583.1 | 2.5% | 1.10 | 0.23 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 307.6 | 452.7 | 47.2% | 3.17 | 0.43 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 767.3 | 1,308.1 | 70.5% | 1.93 | 0.34 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 382.4 | 505.1 | 32.1% | 2.69 | 2.74 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 299.3 | 331.5 | 10.7% | 2.45 | 0.44 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 589.5 | 780.3 | 32.4% | 2.50 | 0.44 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 375.9 | 448.9 | 19.4% | 1.95 | 0.35 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 670.9 | 1,027.5 | 53.1% | 2.19 | 0.39 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 1,401.2 | 1,568.0 | 11.9% | 1.75 | 0.37 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 5.8 | 6.0 | 2.7% | 5.23 | 0.70 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 9.7 | 10.0 | 3.2% | 7.88 | 0.85 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 28.5 | 29.2 | 2.4% | 4.82 | 1.15 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 6.6 | 6.8 | 3.2% | 3.49 | 0.62 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 10.3 | 10.6 | 2.5% | 5.92 | 0.79 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 8.3 | 9.8 | 18.1% | 2.75 | 0.49 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 11.5 | 11.8 | 2.3% | 5.30 | 0.71 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 12.3 | 12.7 | 2.5% | 8.65 | 1.33 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 4.0 | 4.1 | 2.6% | 3.80 | 0.51 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 5.5 | 5.6 | 2.9% | 6.98 | 0.75 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 18.4 | 19.1 | 3.7% | 3.72 | 0.89 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 4.3 | 4.4 | 2.4% | 2.65 | 0.47 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 6.6 | 6.8 | 2.4% | 4.61 | 0.62 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 5.1 | 5.3 | 4.1% | 2.24 | 0.40 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 7.4 | 7.5 | 1.4% | 4.13 | 0.55 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 6.8 | 6.9 | 2.3% | 7.89 | 1.21 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 25.6 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 1.52 | 5.9% | 1,285.9 | 7.5% |
| Gated Delta-Rule Scan | 1.04 | 4.1% | 2,833.2 | 6.5% |
| Causal Depthwise Conv1D | 1.16 | 4.5% | 1,772.1 | 15.2% |
| gdn_cumdecay_f16 | 1.84 | 7.2% | 797.3 | 18.4% |
| gdn_gated_scan_f16 | 1.04 | 4.1% | 2,844.5 | 16.3% |
| gdn_cumdecay_bf16 | 1.51 | 5.9% | 971.3 | 17.2% |
| gdn_gated_scan_bf16 | 1.04 | 4.1% | 2,832.7 | 9.6% |
| gdn2_gated_scan | 1.10 | 4.3% | 4,470.5 | 2.5% |
| Gated Cumulative Decay | 3.17 | 12.4% | 307.6 | 47.2% |
| Gated Delta-Rule Scan | 1.93 | 7.5% | 767.3 | 70.5% |
| Causal Depthwise Conv1D | 2.69 | 10.5% | 382.4 | 32.1% |
| gdn_cumdecay_f16 | 2.45 | 9.6% | 299.3 | 10.7% |
| gdn_gated_scan_f16 | 2.50 | 9.8% | 589.5 | 32.4% |
| gdn_cumdecay_bf16 | 1.95 | 7.6% | 375.9 | 19.4% |
| gdn_gated_scan_bf16 | 2.19 | 8.6% | 670.9 | 53.1% |
| gdn2_gated_scan | 1.75 | 6.8% | 1,401.2 | 11.9% |
| Gated Cumulative Decay | 5.23 | 20.4% | 5.8 | 2.7% |
| Gated Delta-Rule Scan | 7.88 | 30.8% | 9.7 | 3.2% |
| Causal Depthwise Conv1D | 4.82 | 18.8% | 28.5 | 2.4% |
| gdn_cumdecay_f16 | 3.49 | 13.6% | 6.6 | 3.2% |
| gdn_gated_scan_f16 | 5.92 | 23.1% | 10.3 | 2.5% |
| gdn_cumdecay_bf16 | 2.75 | 10.7% | 8.3 | 18.1% |
| gdn_gated_scan_bf16 | 5.30 | 20.7% | 11.5 | 2.3% |
| gdn2_gated_scan | 8.65 | 33.8% | 12.3 | 2.5% |
| Gated Cumulative Decay | 3.80 | 14.8% | 4.0 | 2.6% |
| Gated Delta-Rule Scan | 6.98 | 27.3% | 5.5 | 2.9% |
| Causal Depthwise Conv1D | 3.72 | 14.5% | 18.4 | 3.7% |
| gdn_cumdecay_f16 | 2.65 | 10.4% | 4.3 | 2.4% |
| gdn_gated_scan_f16 | 4.61 | 18.0% | 6.6 | 2.4% |
| gdn_cumdecay_bf16 | 2.24 | 8.8% | 5.1 | 4.1% |
| gdn_gated_scan_bf16 | 4.13 | 16.1% | 7.4 | 1.4% |
| gdn2_gated_scan | 7.89 | 30.8% | 6.8 | 2.3% |
