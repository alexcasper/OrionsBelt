# rk3588-t3_little — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 357.0 | 383.3 | 7.4% | 5.47 | 0.73 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 1,303.8 | 1,913.2 | 46.7% | 2.27 | 0.40 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 377.2 | 402.8 | 6.8% | 5.46 | 5.56 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 230.1 | 243.6 | 5.8% | 6.37 | 1.14 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 1,448.2 | 1,724.7 | 19.1% | 2.03 | 0.36 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 259.9 | 275.4 | 5.9% | 5.64 | 1.01 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 1,343.2 | 1,720.4 | 28.1% | 2.19 | 0.39 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 4,046.9 | 5,449.6 | 34.7% | 1.21 | 0.26 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 156.1 | 166.3 | 6.5% | 6.26 | 0.84 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 245.6 | 257.0 | 4.6% | 6.03 | 1.07 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 147.6 | 149.6 | 1.4% | 6.98 | 7.10 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 115.2 | 116.4 | 1.0% | 6.36 | 1.14 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 225.5 | 235.4 | 4.4% | 6.53 | 1.16 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 128.9 | 129.8 | 0.7% | 5.68 | 1.02 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 226.1 | 235.7 | 4.3% | 6.51 | 1.16 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 371.6 | 387.7 | 4.3% | 6.61 | 1.41 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 2.9 | 2.9 | 0.0% | 10.46 | 1.40 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 4.7 | 5.0 | 6.2% | 16.35 | 1.76 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 10.2 | 10.8 | 5.7% | 13.45 | 3.21 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 2.9 | 3.2 | 10.0% | 7.85 | 1.40 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 4.4 | 4.4 | 0.0% | 13.95 | 1.87 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 3.5 | 3.8 | 8.3% | 6.54 | 1.17 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 4.7 | 5.0 | 6.3% | 13.08 | 1.76 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 6.7 | 7.0 | 4.3% | 15.92 | 2.44 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 2.3 | 2.3 | 0.0% | 6.54 | 0.88 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 3.2 | 3.5 | 9.1% | 11.89 | 1.28 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 8.2 | 8.5 | 3.6% | 8.41 | 2.01 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 2.3 | 2.3 | 0.0% | 4.90 | 0.88 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 2.9 | 3.2 | 10.0% | 10.46 | 1.40 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 2.6 | 2.6 | 0.0% | 4.36 | 0.78 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 3.2 | 3.5 | 9.1% | 9.51 | 1.28 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 4.4 | 4.4 | 0.0% | 12.21 | 1.87 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 34.0 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 5.47 | 16.1% | 357.0 | 7.4% |
| Gated Delta-Rule Scan | 2.27 | 6.7% | 1,303.8 | 46.7% |
| Causal Depthwise Conv1D | 5.46 | 16.1% | 377.2 | 6.8% |
| gdn_cumdecay_f16 | 6.37 | 18.7% | 230.1 | 5.8% |
| gdn_gated_scan_f16 | 2.03 | 6.0% | 1,448.2 | 19.1% |
| gdn_cumdecay_bf16 | 5.64 | 16.6% | 259.9 | 5.9% |
| gdn_gated_scan_bf16 | 2.19 | 6.4% | 1,343.2 | 28.1% |
| gdn2_gated_scan | 1.21 | 3.6% | 4,046.9 | 34.7% |
| Gated Cumulative Decay | 6.26 | 18.4% | 156.1 | 6.5% |
| Gated Delta-Rule Scan | 6.03 | 17.7% | 245.6 | 4.6% |
| Causal Depthwise Conv1D | 6.98 | 20.5% | 147.6 | 1.4% |
| gdn_cumdecay_f16 | 6.36 | 18.7% | 115.2 | 1.0% |
| gdn_gated_scan_f16 | 6.53 | 19.2% | 225.5 | 4.4% |
| gdn_cumdecay_bf16 | 5.68 | 16.7% | 128.9 | 0.7% |
| gdn_gated_scan_bf16 | 6.51 | 19.1% | 226.1 | 4.3% |
| gdn2_gated_scan | 6.61 | 19.4% | 371.6 | 4.3% |
| Gated Cumulative Decay | 10.46 | 30.8% | 2.9 | 0.0% |
| Gated Delta-Rule Scan | 16.35 | 48.1% | 4.7 | 6.2% |
| Causal Depthwise Conv1D | 13.45 | 39.6% | 10.2 | 5.7% |
| gdn_cumdecay_f16 | 7.85 | 23.1% | 2.9 | 10.0% |
| gdn_gated_scan_f16 | 13.95 | 41.0% | 4.4 | 0.0% |
| gdn_cumdecay_bf16 | 6.54 | 19.2% | 3.5 | 8.3% |
| gdn_gated_scan_bf16 | 13.08 | 38.5% | 4.7 | 6.3% |
| gdn2_gated_scan | 15.92 | 46.8% | 6.7 | 4.3% |
| Gated Cumulative Decay | 6.54 | 19.2% | 2.3 | 0.0% |
| Gated Delta-Rule Scan | 11.89 | 35.0% | 3.2 | 9.1% |
| Causal Depthwise Conv1D | 8.41 | 24.7% | 8.2 | 3.6% |
| gdn_cumdecay_f16 | 4.90 | 14.4% | 2.3 | 0.0% |
| gdn_gated_scan_f16 | 10.46 | 30.8% | 2.9 | 10.0% |
| gdn_cumdecay_bf16 | 4.36 | 12.8% | 2.6 | 0.0% |
| gdn_gated_scan_bf16 | 9.51 | 28.0% | 3.2 | 9.1% |
| gdn2_gated_scan | 12.21 | 35.9% | 4.4 | 0.0% |
