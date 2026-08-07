# rk3588-t4_little_singlethread — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 1,494.6 | 1,650.1 | 10.4% | 1.31 | 0.18 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 3,619.8 | 3,735.6 | 3.2% | 0.82 | 0.14 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 1,623.2 | 1,724.2 | 6.2% | 1.27 | 1.29 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 963.4 | 1,018.3 | 5.7% | 1.52 | 0.27 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 3,604.4 | 3,672.6 | 1.9% | 0.82 | 0.15 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 1,079.0 | 1,126.8 | 4.4% | 1.36 | 0.24 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 3,610.2 | 3,734.2 | 3.4% | 0.82 | 0.15 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 9,277.4 | 9,391.5 | 1.2% | 0.53 | 0.11 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 591.5 | 619.3 | 4.7% | 1.65 | 0.22 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 1,015.7 | 1,054.2 | 3.8% | 1.46 | 0.26 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 667.4 | 694.8 | 4.1% | 1.54 | 1.57 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 393.8 | 413.0 | 4.9% | 1.86 | 0.33 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 1,015.9 | 1,048.0 | 3.2% | 1.45 | 0.26 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 451.5 | 468.4 | 3.7% | 1.62 | 0.29 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 1,021.5 | 1,094.7 | 7.2% | 1.44 | 0.26 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 2,310.5 | 2,599.2 | 12.5% | 1.06 | 0.23 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 7.0 | 7.3 | 4.2% | 4.36 | 0.59 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 16.6 | 17.2 | 3.5% | 4.59 | 0.49 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 53.7 | 54.5 | 1.6% | 2.56 | 0.61 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 8.2 | 8.2 | 0.0% | 2.80 | 0.50 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 14.0 | 14.3 | 2.1% | 4.36 | 0.59 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 9.9 | 10.2 | 2.9% | 2.31 | 0.41 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 16.0 | 16.3 | 1.8% | 3.80 | 0.51 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 34.1 | 34.7 | 1.7% | 3.13 | 0.48 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 4.4 | 4.4 | 0.0% | 3.49 | 0.47 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 7.0 | 7.3 | 4.2% | 5.45 | 0.59 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 27.7 | 29.5 | 6.3% | 2.48 | 0.59 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 5.0 | 5.3 | 5.9% | 2.31 | 0.41 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 6.7 | 7.0 | 4.4% | 4.55 | 0.61 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 6.1 | 6.1 | 0.0% | 1.87 | 0.33 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 7.9 | 8.2 | 3.7% | 3.88 | 0.52 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 15.8 | 16.3 | 3.7% | 3.39 | 0.52 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 1.31 | 4.1% | 1,494.6 | 10.4% |
| Gated Delta-Rule Scan | 0.82 | 2.6% | 3,619.8 | 3.2% |
| Causal Depthwise Conv1D | 1.27 | 4.0% | 1,623.2 | 6.2% |
| gdn_cumdecay_f16 | 1.52 | 4.8% | 963.4 | 5.7% |
| gdn_gated_scan_f16 | 0.82 | 2.6% | 3,604.4 | 1.9% |
| gdn_cumdecay_bf16 | 1.36 | 4.3% | 1,079.0 | 4.4% |
| gdn_gated_scan_bf16 | 0.82 | 2.6% | 3,610.2 | 3.4% |
| gdn2_gated_scan | 0.53 | 1.7% | 9,277.4 | 1.2% |
| Gated Cumulative Decay | 1.65 | 5.2% | 591.5 | 4.7% |
| Gated Delta-Rule Scan | 1.46 | 4.6% | 1,015.7 | 3.8% |
| Causal Depthwise Conv1D | 1.54 | 4.9% | 667.4 | 4.1% |
| gdn_cumdecay_f16 | 1.86 | 5.9% | 393.8 | 4.9% |
| gdn_gated_scan_f16 | 1.45 | 4.6% | 1,015.9 | 3.2% |
| gdn_cumdecay_bf16 | 1.62 | 5.1% | 451.5 | 3.7% |
| gdn_gated_scan_bf16 | 1.44 | 4.5% | 1,021.5 | 7.2% |
| gdn2_gated_scan | 1.06 | 3.3% | 2,310.5 | 12.5% |
| Gated Cumulative Decay | 4.36 | 13.8% | 7.0 | 4.2% |
| Gated Delta-Rule Scan | 4.59 | 14.5% | 16.6 | 3.5% |
| Causal Depthwise Conv1D | 2.56 | 8.1% | 53.7 | 1.6% |
| gdn_cumdecay_f16 | 2.80 | 8.8% | 8.2 | 0.0% |
| gdn_gated_scan_f16 | 4.36 | 13.8% | 14.0 | 2.1% |
| gdn_cumdecay_bf16 | 2.31 | 7.3% | 9.9 | 2.9% |
| gdn_gated_scan_bf16 | 3.80 | 12.0% | 16.0 | 1.8% |
| gdn2_gated_scan | 3.13 | 9.9% | 34.1 | 1.7% |
| Gated Cumulative Decay | 3.49 | 11.0% | 4.4 | 0.0% |
| Gated Delta-Rule Scan | 5.45 | 17.2% | 7.0 | 4.2% |
| Causal Depthwise Conv1D | 2.48 | 7.8% | 27.7 | 6.3% |
| gdn_cumdecay_f16 | 2.31 | 7.3% | 5.0 | 5.9% |
| gdn_gated_scan_f16 | 4.55 | 14.4% | 6.7 | 4.4% |
| gdn_cumdecay_bf16 | 1.87 | 5.9% | 6.1 | 0.0% |
| gdn_gated_scan_bf16 | 3.88 | 12.2% | 7.9 | 3.7% |
| gdn2_gated_scan | 3.39 | 10.7% | 15.8 | 3.7% |
