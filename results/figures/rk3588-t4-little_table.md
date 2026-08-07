# rk3588-t4-little — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 1,349.9 | 1,412.4 | 4.6% | 1.45 | 0.19 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 3,604.1 | 3,864.3 | 7.2% | 0.82 | 0.15 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 1,722.4 | 1,874.4 | 8.8% | 1.20 | 1.22 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 882.6 | 924.6 | 4.8% | 1.66 | 0.30 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 3,588.9 | 3,750.2 | 4.5% | 0.82 | 0.15 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 963.2 | 983.0 | 2.1% | 1.52 | 0.27 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 3,578.7 | 3,814.7 | 6.6% | 0.82 | 0.15 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 9,267.5 | 9,412.8 | 1.6% | 0.53 | 0.11 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 591.8 | 619.3 | 4.6% | 1.65 | 0.22 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 1,001.4 | 1,025.3 | 2.4% | 1.48 | 0.26 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 668.8 | 696.3 | 4.1% | 1.54 | 1.57 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 393.2 | 419.4 | 6.7% | 1.86 | 0.33 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 1,014.2 | 1,056.5 | 4.2% | 1.45 | 0.26 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 451.2 | 471.9 | 4.6% | 1.62 | 0.29 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 1,011.9 | 1,052.1 | 4.0% | 1.46 | 0.26 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 2,566.6 | 2,904.6 | 13.2% | 0.96 | 0.20 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 7.0 | 7.3 | 4.2% | 4.36 | 0.59 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 16.6 | 16.9 | 1.8% | 4.59 | 0.49 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 54.0 | 54.5 | 1.1% | 2.54 | 0.61 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 8.2 | 9.3 | 14.3% | 2.80 | 0.50 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 14.0 | 14.3 | 2.1% | 4.36 | 0.59 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 9.9 | 10.2 | 2.9% | 2.31 | 0.41 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 16.6 | 16.9 | 1.8% | 3.67 | 0.49 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 33.5 | 33.8 | 0.9% | 3.18 | 0.49 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 4.4 | 4.4 | 0.0% | 3.49 | 0.47 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 7.3 | 7.3 | 0.0% | 5.23 | 0.56 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 28.3 | 29.8 | 5.2% | 2.43 | 0.58 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 5.0 | 5.3 | 5.9% | 2.31 | 0.41 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 6.7 | 6.7 | 0.0% | 4.55 | 0.61 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 6.1 | 6.1 | 0.0% | 1.87 | 0.33 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 7.9 | 8.2 | 3.7% | 3.87 | 0.52 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 16.0 | 16.3 | 1.8% | 3.33 | 0.51 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 1.45 | 4.6% | 1,349.9 | 4.6% |
| Gated Delta-Rule Scan | 0.82 | 2.6% | 3,604.1 | 7.2% |
| Causal Depthwise Conv1D | 1.20 | 3.8% | 1,722.4 | 8.8% |
| gdn_cumdecay_f16 | 1.66 | 5.2% | 882.6 | 4.8% |
| gdn_gated_scan_f16 | 0.82 | 2.6% | 3,588.9 | 4.5% |
| gdn_cumdecay_bf16 | 1.52 | 4.8% | 963.2 | 2.1% |
| gdn_gated_scan_bf16 | 0.82 | 2.6% | 3,578.7 | 6.6% |
| gdn2_gated_scan | 0.53 | 1.7% | 9,267.5 | 1.6% |
| Gated Cumulative Decay | 1.65 | 5.2% | 591.8 | 4.6% |
| Gated Delta-Rule Scan | 1.48 | 4.7% | 1,001.4 | 2.4% |
| Causal Depthwise Conv1D | 1.54 | 4.9% | 668.8 | 4.1% |
| gdn_cumdecay_f16 | 1.86 | 5.9% | 393.2 | 6.7% |
| gdn_gated_scan_f16 | 1.45 | 4.6% | 1,014.2 | 4.2% |
| gdn_cumdecay_bf16 | 1.62 | 5.1% | 451.2 | 4.6% |
| gdn_gated_scan_bf16 | 1.46 | 4.6% | 1,011.9 | 4.0% |
| gdn2_gated_scan | 0.96 | 3.0% | 2,566.6 | 13.2% |
| Gated Cumulative Decay | 4.36 | 13.8% | 7.0 | 4.2% |
| Gated Delta-Rule Scan | 4.59 | 14.5% | 16.6 | 1.8% |
| Causal Depthwise Conv1D | 2.54 | 8.0% | 54.0 | 1.1% |
| gdn_cumdecay_f16 | 2.80 | 8.8% | 8.2 | 14.3% |
| gdn_gated_scan_f16 | 4.36 | 13.8% | 14.0 | 2.1% |
| gdn_cumdecay_bf16 | 2.31 | 7.3% | 9.9 | 2.9% |
| gdn_gated_scan_bf16 | 3.67 | 11.6% | 16.6 | 1.8% |
| gdn2_gated_scan | 3.18 | 10.0% | 33.5 | 0.9% |
| Gated Cumulative Decay | 3.49 | 11.0% | 4.4 | 0.0% |
| Gated Delta-Rule Scan | 5.23 | 16.5% | 7.3 | 0.0% |
| Causal Depthwise Conv1D | 2.43 | 7.7% | 28.3 | 5.2% |
| gdn_cumdecay_f16 | 2.31 | 7.3% | 5.0 | 5.9% |
| gdn_gated_scan_f16 | 4.55 | 14.4% | 6.7 | 0.0% |
| gdn_cumdecay_bf16 | 1.87 | 5.9% | 6.1 | 0.0% |
| gdn_gated_scan_bf16 | 3.87 | 12.2% | 7.9 | 3.7% |
| gdn2_gated_scan | 3.33 | 10.5% | 16.0 | 1.8% |
