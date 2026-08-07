# rk3588-t4_little — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 332.8 | 376.9 | 13.2% | 5.87 | 0.79 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 838.3 | 1,073.7 | 28.1% | 3.53 | 0.63 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 401.7 | 430.2 | 7.1% | 5.13 | 5.22 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 219.1 | 246.8 | 12.7% | 6.69 | 1.20 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 818.8 | 934.0 | 14.1% | 3.60 | 0.64 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 245.9 | 268.6 | 9.3% | 5.96 | 1.07 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 809.7 | 941.0 | 16.2% | 3.64 | 0.65 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 3,234.2 | 3,369.6 | 4.2% | 1.52 | 0.32 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 155.5 | 160.4 | 3.2% | 6.28 | 0.84 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 262.2 | 277.1 | 5.7% | 5.64 | 1.00 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 174.4 | 192.8 | 10.5% | 5.90 | 6.01 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 100.6 | 102.1 | 1.4% | 7.28 | 1.30 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 262.8 | 275.1 | 4.7% | 5.60 | 1.00 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 114.6 | 115.5 | 0.8% | 6.39 | 1.14 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 263.1 | 282.9 | 7.5% | 5.60 | 1.00 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 589.5 | 717.6 | 21.7% | 4.17 | 0.89 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 3.8 | 4.1 | 7.7% | 8.05 | 1.08 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 4.7 | 5.0 | 6.3% | 16.35 | 1.76 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 14.6 | 14.9 | 2.0% | 9.42 | 2.25 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 4.7 | 5.0 | 6.3% | 4.90 | 0.88 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 5.8 | 5.8 | 0.0% | 10.46 | 1.40 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 5.2 | 5.5 | 5.6% | 4.36 | 0.78 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 6.4 | 6.7 | 4.5% | 9.51 | 1.28 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 10.2 | 10.8 | 5.7% | 10.46 | 1.60 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 3.8 | 4.1 | 7.7% | 4.02 | 0.54 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 3.8 | 4.1 | 7.7% | 10.06 | 1.08 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 9.0 | 9.3 | 3.2% | 7.59 | 1.81 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 4.1 | 4.1 | 0.0% | 2.80 | 0.50 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 4.7 | 4.7 | 0.0% | 6.54 | 0.88 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 4.1 | 4.4 | 7.1% | 2.80 | 0.50 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 5.0 | 5.0 | 0.0% | 6.16 | 0.83 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 7.0 | 7.3 | 4.2% | 7.63 | 1.17 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 34.0 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 5.87 | 17.3% | 332.8 | 13.2% |
| Gated Delta-Rule Scan | 3.53 | 10.4% | 838.3 | 28.1% |
| Causal Depthwise Conv1D | 5.13 | 15.1% | 401.7 | 7.1% |
| gdn_cumdecay_f16 | 6.69 | 19.7% | 219.1 | 12.7% |
| gdn_gated_scan_f16 | 3.60 | 10.6% | 818.8 | 14.1% |
| gdn_cumdecay_bf16 | 5.96 | 17.5% | 245.9 | 9.3% |
| gdn_gated_scan_bf16 | 3.64 | 10.7% | 809.7 | 16.2% |
| gdn2_gated_scan | 1.52 | 4.5% | 3,234.2 | 4.2% |
| Gated Cumulative Decay | 6.28 | 18.5% | 155.5 | 3.2% |
| Gated Delta-Rule Scan | 5.64 | 16.6% | 262.2 | 5.7% |
| Causal Depthwise Conv1D | 5.90 | 17.4% | 174.4 | 10.5% |
| gdn_cumdecay_f16 | 7.28 | 21.4% | 100.6 | 1.4% |
| gdn_gated_scan_f16 | 5.60 | 16.5% | 262.8 | 4.7% |
| gdn_cumdecay_bf16 | 6.39 | 18.8% | 114.6 | 0.8% |
| gdn_gated_scan_bf16 | 5.60 | 16.5% | 263.1 | 7.5% |
| gdn2_gated_scan | 4.17 | 12.3% | 589.5 | 21.7% |
| Gated Cumulative Decay | 8.05 | 23.7% | 3.8 | 7.7% |
| Gated Delta-Rule Scan | 16.35 | 48.1% | 4.7 | 6.3% |
| Causal Depthwise Conv1D | 9.42 | 27.7% | 14.6 | 2.0% |
| gdn_cumdecay_f16 | 4.90 | 14.4% | 4.7 | 6.3% |
| gdn_gated_scan_f16 | 10.46 | 30.8% | 5.8 | 0.0% |
| gdn_cumdecay_bf16 | 4.36 | 12.8% | 5.2 | 5.6% |
| gdn_gated_scan_bf16 | 9.51 | 28.0% | 6.4 | 4.5% |
| gdn2_gated_scan | 10.46 | 30.8% | 10.2 | 5.7% |
| Gated Cumulative Decay | 4.02 | 11.8% | 3.8 | 7.7% |
| Gated Delta-Rule Scan | 10.06 | 29.6% | 3.8 | 7.7% |
| Causal Depthwise Conv1D | 7.59 | 22.3% | 9.0 | 3.2% |
| gdn_cumdecay_f16 | 2.80 | 8.2% | 4.1 | 0.0% |
| gdn_gated_scan_f16 | 6.54 | 19.2% | 4.7 | 0.0% |
| gdn_cumdecay_bf16 | 2.80 | 8.2% | 4.1 | 7.1% |
| gdn_gated_scan_bf16 | 6.16 | 18.1% | 5.0 | 0.0% |
| gdn2_gated_scan | 7.63 | 22.4% | 7.0 | 4.2% |
