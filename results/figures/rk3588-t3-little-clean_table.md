# rk3588-t3-little-clean — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 1,638.4 | 2,659.6 | 62.3% | 1.19 | 0.16 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 5,358.6 | 5,574.1 | 4.0% | 0.55 | 0.10 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 1,836.5 | 2,017.6 | 9.9% | 1.12 | 1.14 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 1,272.0 | 1,396.3 | 9.8% | 1.15 | 0.21 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 4,807.3 | 5,142.7 | 7.0% | 0.61 | 0.11 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 1,047.4 | 1,063.5 | 1.5% | 1.40 | 0.25 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 4,895.1 | 5,776.3 | 18.0% | 0.60 | 0.11 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 9,964.6 | 10,463.7 | 5.0% | 0.49 | 0.11 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 627.1 | 642.3 | 2.4% | 1.56 | 0.21 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 967.2 | 1,016.2 | 5.1% | 1.53 | 0.27 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 670.6 | 688.1 | 2.6% | 1.54 | 1.56 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 423.8 | 436.4 | 3.0% | 1.73 | 0.31 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 981.5 | 1,001.4 | 2.0% | 1.50 | 0.27 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 483.6 | 497.9 | 3.0% | 1.51 | 0.27 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 1,156.8 | 1,524.4 | 31.8% | 1.27 | 0.23 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 2,099.0 | 2,454.8 | 17.0% | 1.17 | 0.25 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 6.1 | 6.4 | 4.8% | 4.98 | 0.67 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 16.6 | 17.2 | 3.5% | 4.59 | 0.49 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 52.5 | 53.4 | 1.7% | 2.62 | 0.62 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 7.0 | 7.0 | 0.0% | 3.27 | 0.59 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 14.0 | 14.3 | 2.1% | 4.36 | 0.59 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 9.0 | 9.0 | 0.0% | 2.53 | 0.45 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 16.3 | 16.6 | 1.8% | 3.74 | 0.50 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 23.9 | 24.5 | 2.4% | 4.47 | 0.68 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 3.5 | 3.5 | 0.0% | 4.36 | 0.59 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 6.4 | 6.7 | 4.5% | 5.94 | 0.64 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 29.8 | 31.5 | 5.9% | 2.31 | 0.55 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 4.1 | 4.1 | 0.0% | 2.80 | 0.50 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 6.1 | 6.4 | 4.8% | 4.98 | 0.67 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 5.0 | 5.0 | 0.0% | 2.31 | 0.41 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 7.6 | 7.6 | 0.0% | 4.02 | 0.54 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 10.5 | 10.8 | 2.8% | 5.09 | 0.78 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 1.19 | 3.8% | 1,638.4 | 62.3% |
| Gated Delta-Rule Scan | 0.55 | 1.7% | 5,358.6 | 4.0% |
| Causal Depthwise Conv1D | 1.12 | 3.5% | 1,836.5 | 9.9% |
| gdn_cumdecay_f16 | 1.15 | 3.6% | 1,272.0 | 9.8% |
| gdn_gated_scan_f16 | 0.61 | 1.9% | 4,807.3 | 7.0% |
| gdn_cumdecay_bf16 | 1.40 | 4.4% | 1,047.4 | 1.5% |
| gdn_gated_scan_bf16 | 0.60 | 1.9% | 4,895.1 | 18.0% |
| gdn2_gated_scan | 0.49 | 1.5% | 9,964.6 | 5.0% |
| Gated Cumulative Decay | 1.56 | 4.9% | 627.1 | 2.4% |
| Gated Delta-Rule Scan | 1.53 | 4.8% | 967.2 | 5.1% |
| Causal Depthwise Conv1D | 1.54 | 4.9% | 670.6 | 2.6% |
| gdn_cumdecay_f16 | 1.73 | 5.5% | 423.8 | 3.0% |
| gdn_gated_scan_f16 | 1.50 | 4.7% | 981.5 | 2.0% |
| gdn_cumdecay_bf16 | 1.51 | 4.8% | 483.6 | 3.0% |
| gdn_gated_scan_bf16 | 1.27 | 4.0% | 1,156.8 | 31.8% |
| gdn2_gated_scan | 1.17 | 3.7% | 2,099.0 | 17.0% |
| Gated Cumulative Decay | 4.98 | 15.7% | 6.1 | 4.8% |
| Gated Delta-Rule Scan | 4.59 | 14.5% | 16.6 | 3.5% |
| Causal Depthwise Conv1D | 2.62 | 8.3% | 52.5 | 1.7% |
| gdn_cumdecay_f16 | 3.27 | 10.3% | 7.0 | 0.0% |
| gdn_gated_scan_f16 | 4.36 | 13.8% | 14.0 | 2.1% |
| gdn_cumdecay_bf16 | 2.53 | 8.0% | 9.0 | 0.0% |
| gdn_gated_scan_bf16 | 3.74 | 11.8% | 16.3 | 1.8% |
| gdn2_gated_scan | 4.47 | 14.1% | 23.9 | 2.4% |
| Gated Cumulative Decay | 4.36 | 13.8% | 3.5 | 0.0% |
| Gated Delta-Rule Scan | 5.94 | 18.7% | 6.4 | 4.5% |
| Causal Depthwise Conv1D | 2.31 | 7.3% | 29.8 | 5.9% |
| gdn_cumdecay_f16 | 2.80 | 8.8% | 4.1 | 0.0% |
| gdn_gated_scan_f16 | 4.98 | 15.7% | 6.1 | 4.8% |
| gdn_cumdecay_bf16 | 2.31 | 7.3% | 5.0 | 0.0% |
| gdn_gated_scan_bf16 | 4.02 | 12.7% | 7.6 | 0.0% |
| gdn2_gated_scan | 5.09 | 16.1% | 10.5 | 2.8% |
