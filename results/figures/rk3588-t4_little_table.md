# rk3588-t4_little — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 329.9 | 374.5 | 13.5% | 5.92 | 0.79 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 757.8 | 900.4 | 18.8% | 3.91 | 0.69 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 398.2 | 444.5 | 11.6% | 5.17 | 5.27 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 217.6 | 233.3 | 7.2% | 6.73 | 1.20 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 772.7 | 857.6 | 11.0% | 3.81 | 0.68 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 243.6 | 260.8 | 7.1% | 6.01 | 1.08 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 743.2 | 885.9 | 19.2% | 3.96 | 0.71 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 1,881.4 | 2,023.7 | 7.6% | 2.61 | 0.56 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 156.1 | 156.9 | 0.6% | 6.26 | 0.84 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 266.0 | 283.8 | 6.7% | 5.56 | 0.99 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 174.4 | 188.7 | 8.2% | 5.90 | 6.01 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 102.1 | 106.5 | 4.3% | 7.17 | 1.28 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 265.1 | 289.4 | 9.1% | 5.55 | 0.99 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 116.1 | 118.4 | 2.0% | 6.31 | 1.13 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 266.6 | 291.1 | 9.2% | 5.52 | 0.98 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 455.9 | 528.2 | 15.9% | 5.39 | 1.15 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 4.1 | 4.1 | 0.0% | 7.47 | 1.00 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 5.8 | 6.1 | 5.0% | 13.08 | 1.40 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 11.7 | 12.0 | 2.5% | 11.77 | 2.81 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 4.7 | 5.0 | 6.3% | 4.90 | 0.88 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 5.8 | 6.7 | 15.0% | 10.46 | 1.40 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 5.2 | 5.3 | 0.0% | 4.36 | 0.78 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 6.1 | 6.4 | 4.8% | 9.96 | 1.34 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 8.8 | 9.0 | 3.3% | 12.21 | 1.87 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 4.1 | 4.1 | 0.0% | 3.74 | 0.50 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 3.8 | 4.1 | 7.7% | 10.06 | 1.08 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 10.2 | 10.5 | 2.9% | 6.73 | 1.60 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 4.1 | 4.1 | 0.0% | 2.80 | 0.50 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 4.7 | 5.0 | 6.3% | 6.54 | 0.88 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 4.4 | 4.4 | 0.0% | 2.62 | 0.47 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 5.0 | 5.0 | 0.0% | 6.15 | 0.83 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 6.4 | 6.7 | 4.5% | 8.32 | 1.28 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 34.0 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 5.92 | 17.4% | 329.9 | 13.5% |
| Gated Delta-Rule Scan | 3.91 | 11.5% | 757.8 | 18.8% |
| Causal Depthwise Conv1D | 5.17 | 15.2% | 398.2 | 11.6% |
| gdn_cumdecay_f16 | 6.73 | 19.8% | 217.6 | 7.2% |
| gdn_gated_scan_f16 | 3.81 | 11.2% | 772.7 | 11.0% |
| gdn_cumdecay_bf16 | 6.01 | 17.7% | 243.6 | 7.1% |
| gdn_gated_scan_bf16 | 3.96 | 11.6% | 743.2 | 19.2% |
| gdn2_gated_scan | 2.61 | 7.7% | 1,881.4 | 7.6% |
| Gated Cumulative Decay | 6.26 | 18.4% | 156.1 | 0.6% |
| Gated Delta-Rule Scan | 5.56 | 16.4% | 266.0 | 6.7% |
| Causal Depthwise Conv1D | 5.90 | 17.4% | 174.4 | 8.2% |
| gdn_cumdecay_f16 | 7.17 | 21.1% | 102.1 | 4.3% |
| gdn_gated_scan_f16 | 5.55 | 16.3% | 265.1 | 9.1% |
| gdn_cumdecay_bf16 | 6.31 | 18.6% | 116.1 | 2.0% |
| gdn_gated_scan_bf16 | 5.52 | 16.2% | 266.6 | 9.2% |
| gdn2_gated_scan | 5.39 | 15.9% | 455.9 | 15.9% |
| Gated Cumulative Decay | 7.47 | 22.0% | 4.1 | 0.0% |
| Gated Delta-Rule Scan | 13.08 | 38.5% | 5.8 | 5.0% |
| Causal Depthwise Conv1D | 11.77 | 34.6% | 11.7 | 2.5% |
| gdn_cumdecay_f16 | 4.90 | 14.4% | 4.7 | 6.3% |
| gdn_gated_scan_f16 | 10.46 | 30.8% | 5.8 | 15.0% |
| gdn_cumdecay_bf16 | 4.36 | 12.8% | 5.2 | 0.0% |
| gdn_gated_scan_bf16 | 9.96 | 29.3% | 6.1 | 4.8% |
| gdn2_gated_scan | 12.21 | 35.9% | 8.8 | 3.3% |
| Gated Cumulative Decay | 3.74 | 11.0% | 4.1 | 0.0% |
| Gated Delta-Rule Scan | 10.06 | 29.6% | 3.8 | 7.7% |
| Causal Depthwise Conv1D | 6.73 | 19.8% | 10.2 | 2.9% |
| gdn_cumdecay_f16 | 2.80 | 8.2% | 4.1 | 0.0% |
| gdn_gated_scan_f16 | 6.54 | 19.2% | 4.7 | 6.3% |
| gdn_cumdecay_bf16 | 2.62 | 7.7% | 4.4 | 0.0% |
| gdn_gated_scan_bf16 | 6.15 | 18.1% | 5.0 | 0.0% |
| gdn2_gated_scan | 8.32 | 24.5% | 6.4 | 4.5% |
