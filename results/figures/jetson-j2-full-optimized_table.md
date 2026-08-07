# jetson-j2-full-optimized — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 511.8 | 588.2 | 14.9% | 3.82 | 0.51 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 996.8 | 1,050.1 | 5.3% | 2.97 | 0.53 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 570.7 | 659.5 | 15.6% | 3.61 | 3.67 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 362.3 | 413.2 | 14.0% | 4.04 | 0.72 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 1,007.8 | 1,114.5 | 10.6% | 2.92 | 0.52 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 341.2 | 375.7 | 10.1% | 4.29 | 0.77 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 1,001.0 | 1,059.9 | 5.9% | 2.94 | 0.52 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 249.1 | 1,223.1 | 391.0% | 3.92 | 0.53 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 287.7 | 961.8 | 234.4% | 5.15 | 0.91 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 191.3 | 213.1 | 11.4% | 5.39 | 5.48 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 169.8 | 172.8 | 1.7% | 4.31 | 0.77 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 272.2 | 370.9 | 36.3% | 5.41 | 0.96 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 166.5 | 172.6 | 3.7% | 4.40 | 0.79 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 273.3 | 305.8 | 11.9% | 5.39 | 0.96 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 3.8 | 4.3 | 13.7% | 8.03 | 1.08 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 4.5 | 5.1 | 12.8% | 17.03 | 1.83 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 9.2 | 9.3 | 1.1% | 14.90 | 3.55 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 4.1 | 4.2 | 1.2% | 5.56 | 1.00 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 5.3 | 5.5 | 2.9% | 11.49 | 1.54 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 4.2 | 4.8 | 13.6% | 5.42 | 0.97 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 5.6 | 5.7 | 2.8% | 10.95 | 1.47 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 3.2 | 3.8 | 19.7% | 4.80 | 0.64 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 4.1 | 4.2 | 2.6% | 9.39 | 1.01 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 7.6 | 7.9 | 3.4% | 9.03 | 2.15 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 3.2 | 3.6 | 13.1% | 3.60 | 0.64 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 4.4 | 4.4 | 1.2% | 6.98 | 0.94 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 3.9 | 4.0 | 2.7% | 2.93 | 0.52 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 4.4 | 4.5 | 1.2% | 6.89 | 0.93 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 23.8 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 3.82 | 16.1% | 511.8 | 14.9% |
| Gated Delta-Rule Scan | 2.97 | 12.5% | 996.8 | 5.3% |
| Causal Depthwise Conv1D | 3.61 | 15.2% | 570.7 | 15.6% |
| gdn_cumdecay_f16 | 4.04 | 17.0% | 362.3 | 14.0% |
| gdn_gated_scan_f16 | 2.92 | 12.3% | 1,007.8 | 10.6% |
| gdn_cumdecay_bf16 | 4.29 | 18.0% | 341.2 | 10.1% |
| gdn_gated_scan_bf16 | 2.94 | 12.4% | 1,001.0 | 5.9% |
| Gated Cumulative Decay | 3.92 | 16.5% | 249.1 | 391.0% |
| Gated Delta-Rule Scan | 5.15 | 21.6% | 287.7 | 234.4% |
| Causal Depthwise Conv1D | 5.39 | 22.6% | 191.3 | 11.4% |
| gdn_cumdecay_f16 | 4.31 | 18.1% | 169.8 | 1.7% |
| gdn_gated_scan_f16 | 5.41 | 22.7% | 272.2 | 36.3% |
| gdn_cumdecay_bf16 | 4.40 | 18.5% | 166.5 | 3.7% |
| gdn_gated_scan_bf16 | 5.39 | 22.6% | 273.3 | 11.9% |
| Gated Cumulative Decay | 8.03 | 33.7% | 3.8 | 13.7% |
| Gated Delta-Rule Scan | 17.03 | 71.6% | 4.5 | 12.8% |
| Causal Depthwise Conv1D | 14.90 | 62.6% | 9.2 | 1.1% |
| gdn_cumdecay_f16 | 5.56 | 23.4% | 4.1 | 1.2% |
| gdn_gated_scan_f16 | 11.49 | 48.3% | 5.3 | 2.9% |
| gdn_cumdecay_bf16 | 5.42 | 22.8% | 4.2 | 13.6% |
| gdn_gated_scan_bf16 | 10.95 | 46.0% | 5.6 | 2.8% |
| Gated Cumulative Decay | 4.80 | 20.2% | 3.2 | 19.7% |
| Gated Delta-Rule Scan | 9.39 | 39.5% | 4.1 | 2.6% |
| Causal Depthwise Conv1D | 9.03 | 37.9% | 7.6 | 3.4% |
| gdn_cumdecay_f16 | 3.60 | 15.1% | 3.2 | 13.1% |
| gdn_gated_scan_f16 | 6.98 | 29.3% | 4.4 | 1.2% |
| gdn_cumdecay_bf16 | 2.93 | 12.3% | 3.9 | 2.7% |
| gdn_gated_scan_bf16 | 6.89 | 28.9% | 4.4 | 1.2% |
