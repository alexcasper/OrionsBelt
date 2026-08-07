# jetson-j1 — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 528.2 | 1,352.4 | 156.0% | 3.70 | 0.50 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 1,082.2 | 1,965.5 | 81.6% | 2.74 | 0.48 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 634.0 | 1,520.1 | 139.8% | 3.25 | 3.31 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 361.0 | 550.6 | 52.5% | 4.06 | 0.73 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 1,131.4 | 3,227.8 | 185.3% | 2.60 | 0.46 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 370.2 | 513.7 | 38.8% | 3.96 | 0.71 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 1,068.7 | 1,806.4 | 69.0% | 2.76 | 0.49 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 2,367.2 | 3,421.9 | 44.6% | 2.08 | 0.44 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 202.1 | 619.7 | 206.6% | 4.83 | 0.65 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 280.9 | 546.7 | 94.6% | 5.27 | 0.93 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 194.0 | 222.3 | 14.6% | 5.31 | 5.41 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 158.4 | 182.7 | 15.3% | 4.62 | 0.83 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 280.0 | 464.1 | 65.8% | 5.26 | 0.94 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 165.7 | 320.6 | 93.4% | 4.42 | 0.79 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 275.0 | 473.6 | 72.2% | 5.36 | 0.95 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 635.8 | 1,395.9 | 119.5% | 3.86 | 0.82 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 3.5 | 3.8 | 5.9% | 8.62 | 1.16 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 4.5 | 4.9 | 9.2% | 16.84 | 1.81 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 11.2 | 11.7 | 4.2% | 12.21 | 2.91 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 4.0 | 4.1 | 3.9% | 5.78 | 1.03 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 5.2 | 5.4 | 4.0% | 11.72 | 1.57 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 4.4 | 4.5 | 3.6% | 5.23 | 0.94 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 5.6 | 5.7 | 2.8% | 10.95 | 1.47 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 5.9 | 6.2 | 4.4% | 17.99 | 2.76 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 3.1 | 3.2 | 1.7% | 4.88 | 0.66 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 4.4 | 4.6 | 3.5% | 8.62 | 0.93 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 8.0 | 10.4 | 30.7% | 8.62 | 2.06 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 3.1 | 3.3 | 5.0% | 3.66 | 0.66 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 4.0 | 4.1 | 2.7% | 7.71 | 1.03 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 3.5 | 3.9 | 10.4% | 3.28 | 0.59 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 4.2 | 4.4 | 3.7% | 7.23 | 0.97 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 5.5 | 5.7 | 4.8% | 9.76 | 1.50 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 25.6 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 3.70 | 14.5% | 528.2 | 156.0% |
| Gated Delta-Rule Scan | 2.74 | 10.7% | 1,082.2 | 81.6% |
| Causal Depthwise Conv1D | 3.25 | 12.7% | 634.0 | 139.8% |
| gdn_cumdecay_f16 | 4.06 | 15.9% | 361.0 | 52.5% |
| gdn_gated_scan_f16 | 2.60 | 10.2% | 1,131.4 | 185.3% |
| gdn_cumdecay_bf16 | 3.96 | 15.5% | 370.2 | 38.8% |
| gdn_gated_scan_bf16 | 2.76 | 10.8% | 1,068.7 | 69.0% |
| gdn2_gated_scan | 2.08 | 8.1% | 2,367.2 | 44.6% |
| Gated Cumulative Decay | 4.83 | 18.9% | 202.1 | 206.6% |
| Gated Delta-Rule Scan | 5.27 | 20.6% | 280.9 | 94.6% |
| Causal Depthwise Conv1D | 5.31 | 20.7% | 194.0 | 14.6% |
| gdn_cumdecay_f16 | 4.62 | 18.0% | 158.4 | 15.3% |
| gdn_gated_scan_f16 | 5.26 | 20.5% | 280.0 | 65.8% |
| gdn_cumdecay_bf16 | 4.42 | 17.3% | 165.7 | 93.4% |
| gdn_gated_scan_bf16 | 5.36 | 20.9% | 275.0 | 72.2% |
| gdn2_gated_scan | 3.86 | 15.1% | 635.8 | 119.5% |
| Gated Cumulative Decay | 8.62 | 33.7% | 3.5 | 5.9% |
| Gated Delta-Rule Scan | 16.84 | 65.8% | 4.5 | 9.2% |
| Causal Depthwise Conv1D | 12.21 | 47.7% | 11.2 | 4.2% |
| gdn_cumdecay_f16 | 5.78 | 22.6% | 4.0 | 3.9% |
| gdn_gated_scan_f16 | 11.72 | 45.8% | 5.2 | 4.0% |
| gdn_cumdecay_bf16 | 5.23 | 20.4% | 4.4 | 3.6% |
| gdn_gated_scan_bf16 | 10.95 | 42.8% | 5.6 | 2.8% |
| gdn2_gated_scan | 17.99 | 70.3% | 5.9 | 4.4% |
| Gated Cumulative Decay | 4.88 | 19.1% | 3.1 | 1.7% |
| Gated Delta-Rule Scan | 8.62 | 33.7% | 4.4 | 3.5% |
| Causal Depthwise Conv1D | 8.62 | 33.7% | 8.0 | 30.7% |
| gdn_cumdecay_f16 | 3.66 | 14.3% | 3.1 | 5.0% |
| gdn_gated_scan_f16 | 7.71 | 30.1% | 4.0 | 2.7% |
| gdn_cumdecay_bf16 | 3.28 | 12.8% | 3.5 | 10.4% |
| gdn_gated_scan_bf16 | 7.23 | 28.2% | 4.2 | 3.7% |
| gdn2_gated_scan | 9.76 | 38.1% | 5.5 | 4.8% |
