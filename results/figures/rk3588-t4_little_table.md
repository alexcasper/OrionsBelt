# rk3588-t4_little — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 332.8 | 395.8 | 18.9% | 5.87 | 0.79 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 757.2 | 1,024.1 | 35.2% | 3.91 | 0.69 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 388.8 | 418.9 | 7.7% | 5.30 | 5.39 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 221.7 | 244.1 | 10.1% | 6.61 | 1.18 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 765.1 | 941.6 | 23.1% | 3.85 | 0.69 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 248.5 | 271.6 | 9.3% | 5.89 | 1.05 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 754.3 | 962.6 | 27.6% | 3.90 | 0.70 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 1,938.0 | 2,165.5 | 11.7% | 2.54 | 0.54 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 155.5 | 156.9 | 0.9% | 6.28 | 0.84 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 256.1 | 276.2 | 7.9% | 5.78 | 1.02 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 174.4 | 189.0 | 8.4% | 5.90 | 6.01 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 102.1 | 105.0 | 2.9% | 7.17 | 1.28 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 257.3 | 285.0 | 10.8% | 5.72 | 1.02 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 115.8 | 117.8 | 1.8% | 6.32 | 1.13 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 257.3 | 274.8 | 6.8% | 5.72 | 1.02 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 408.9 | 440.2 | 7.6% | 6.01 | 1.28 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 4.1 | 4.1 | 0.0% | 7.47 | 1.00 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 5.0 | 5.3 | 5.9% | 15.38 | 1.65 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 11.1 | 11.7 | 5.3% | 12.39 | 2.96 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 4.4 | 5.0 | 13.3% | 5.23 | 0.94 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 5.8 | 6.1 | 5.0% | 10.46 | 1.40 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 5.3 | 5.5 | 5.5% | 4.36 | 0.78 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 6.1 | 6.4 | 4.8% | 9.96 | 1.34 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 8.8 | 9.0 | 3.3% | 12.21 | 1.87 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 4.1 | 4.1 | 0.0% | 3.74 | 0.50 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 4.1 | 4.1 | 0.0% | 9.34 | 1.00 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 10.2 | 10.5 | 2.9% | 6.73 | 1.60 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 4.1 | 4.4 | 7.1% | 2.80 | 0.50 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 4.7 | 5.0 | 6.3% | 6.54 | 0.88 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 4.4 | 4.4 | 0.0% | 2.62 | 0.47 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 5.0 | 5.0 | 0.0% | 6.15 | 0.83 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 6.1 | 6.4 | 4.8% | 8.72 | 1.34 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 34.0 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 5.87 | 17.3% | 332.8 | 18.9% |
| Gated Delta-Rule Scan | 3.91 | 11.5% | 757.2 | 35.2% |
| Causal Depthwise Conv1D | 5.30 | 15.6% | 388.8 | 7.7% |
| gdn_cumdecay_f16 | 6.61 | 19.4% | 221.7 | 10.1% |
| gdn_gated_scan_f16 | 3.85 | 11.3% | 765.1 | 23.1% |
| gdn_cumdecay_bf16 | 5.89 | 17.3% | 248.5 | 9.3% |
| gdn_gated_scan_bf16 | 3.90 | 11.5% | 754.3 | 27.6% |
| gdn2_gated_scan | 2.54 | 7.5% | 1,938.0 | 11.7% |
| Gated Cumulative Decay | 6.28 | 18.5% | 155.5 | 0.9% |
| Gated Delta-Rule Scan | 5.78 | 17.0% | 256.1 | 7.9% |
| Causal Depthwise Conv1D | 5.90 | 17.4% | 174.4 | 8.4% |
| gdn_cumdecay_f16 | 7.17 | 21.1% | 102.1 | 2.9% |
| gdn_gated_scan_f16 | 5.72 | 16.8% | 257.3 | 10.8% |
| gdn_cumdecay_bf16 | 6.32 | 18.6% | 115.8 | 1.8% |
| gdn_gated_scan_bf16 | 5.72 | 16.8% | 257.3 | 6.8% |
| gdn2_gated_scan | 6.01 | 17.7% | 408.9 | 7.6% |
| Gated Cumulative Decay | 7.47 | 22.0% | 4.1 | 0.0% |
| Gated Delta-Rule Scan | 15.38 | 45.2% | 5.0 | 5.9% |
| Causal Depthwise Conv1D | 12.39 | 36.4% | 11.1 | 5.3% |
| gdn_cumdecay_f16 | 5.23 | 15.4% | 4.4 | 13.3% |
| gdn_gated_scan_f16 | 10.46 | 30.8% | 5.8 | 5.0% |
| gdn_cumdecay_bf16 | 4.36 | 12.8% | 5.3 | 5.5% |
| gdn_gated_scan_bf16 | 9.96 | 29.3% | 6.1 | 4.8% |
| gdn2_gated_scan | 12.21 | 35.9% | 8.8 | 3.3% |
| Gated Cumulative Decay | 3.74 | 11.0% | 4.1 | 0.0% |
| Gated Delta-Rule Scan | 9.34 | 27.5% | 4.1 | 0.0% |
| Causal Depthwise Conv1D | 6.73 | 19.8% | 10.2 | 2.9% |
| gdn_cumdecay_f16 | 2.80 | 8.2% | 4.1 | 7.1% |
| gdn_gated_scan_f16 | 6.54 | 19.2% | 4.7 | 6.3% |
| gdn_cumdecay_bf16 | 2.62 | 7.7% | 4.4 | 0.0% |
| gdn_gated_scan_bf16 | 6.15 | 18.1% | 5.0 | 0.0% |
| gdn2_gated_scan | 8.72 | 25.6% | 6.1 | 4.8% |
