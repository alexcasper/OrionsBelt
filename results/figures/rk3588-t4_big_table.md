# rk3588-t4_big — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 81.7 | 82.8 | 1.4% | 23.91 | 3.21 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 248.8 | 271.9 | 9.3% | 11.90 | 2.11 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 99.2 | 106.8 | 7.6% | 20.77 | 21.15 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 41.7 | 42.9 | 2.8% | 35.12 | 6.28 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 247.9 | 274.2 | 10.6% | 11.88 | 2.11 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 56.3 | 58.0 | 3.1% | 26.02 | 4.66 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 257.3 | 265.4 | 3.2% | 11.45 | 2.04 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 667.4 | 717.0 | 7.4% | 7.36 | 1.57 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 35.6 | 36.2 | 1.6% | 27.44 | 3.68 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 126.0 | 133.6 | 6.0% | 11.75 | 2.08 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 36.8 | 37.6 | 2.4% | 28.02 | 28.53 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 21.0 | 21.3 | 1.4% | 34.87 | 6.24 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 129.8 | 134.2 | 3.4% | 11.34 | 2.02 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 29.2 | 29.5 | 1.0% | 25.11 | 4.49 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 128.1 | 132.1 | 3.2% | 11.50 | 2.05 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 312.4 | 323.5 | 3.5% | 7.86 | 1.68 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.8 | 1.8 | 0.1% | 17.44 | 2.34 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 1.9 | 2.0 | 2.8% | 40.18 | 4.31 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 3.7 | 3.8 | 1.9% | 36.75 | 8.77 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 1.7 | 1.8 | 5.2% | 13.60 | 2.43 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 1.9 | 1.9 | 1.7% | 32.00 | 4.29 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 1.9 | 1.9 | 0.3% | 11.93 | 2.13 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 2.2 | 2.2 | 0.3% | 27.61 | 3.71 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 2.4 | 2.4 | 0.9% | 45.04 | 6.91 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 1.5 | 1.5 | 3.1% | 10.22 | 1.37 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.6 | 1.6 | 0.6% | 24.13 | 2.59 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 2.1 | 2.2 | 1.6% | 32.03 | 7.64 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 1.5 | 1.6 | 3.3% | 7.62 | 1.36 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.6 | 1.6 | 0.4% | 18.82 | 2.53 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.6 | 1.6 | 0.4% | 6.98 | 1.25 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 1.8 | 1.9 | 2.6% | 16.69 | 2.24 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.6 | 1.7 | 1.1% | 32.41 | 4.97 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 23.91 | 75.4% | 81.7 | 1.4% |
| Gated Delta-Rule Scan | 11.90 | 37.5% | 248.8 | 9.3% |
| Causal Depthwise Conv1D | 20.77 | 65.5% | 99.2 | 7.6% |
| gdn_cumdecay_f16 | 35.12 | 110.8% | 41.7 | 2.8% |
| gdn_gated_scan_f16 | 11.88 | 37.5% | 247.9 | 10.6% |
| gdn_cumdecay_bf16 | 26.02 | 82.1% | 56.3 | 3.1% |
| gdn_gated_scan_bf16 | 11.45 | 36.1% | 257.3 | 3.2% |
| gdn2_gated_scan | 7.36 | 23.2% | 667.4 | 7.4% |
| Gated Cumulative Decay | 27.44 | 86.6% | 35.6 | 1.6% |
| Gated Delta-Rule Scan | 11.75 | 37.1% | 126.0 | 6.0% |
| Causal Depthwise Conv1D | 28.02 | 88.4% | 36.8 | 2.4% |
| gdn_cumdecay_f16 | 34.87 | 110.0% | 21.0 | 1.4% |
| gdn_gated_scan_f16 | 11.34 | 35.8% | 129.8 | 3.4% |
| gdn_cumdecay_bf16 | 25.11 | 79.2% | 29.2 | 1.0% |
| gdn_gated_scan_bf16 | 11.50 | 36.3% | 128.1 | 3.2% |
| gdn2_gated_scan | 7.86 | 24.8% | 312.4 | 3.5% |
| Gated Cumulative Decay | 17.44 | 55.0% | 1.8 | 0.1% |
| Gated Delta-Rule Scan | 40.18 | 126.8% | 1.9 | 2.8% |
| Causal Depthwise Conv1D | 36.75 | 115.9% | 3.7 | 1.9% |
| gdn_cumdecay_f16 | 13.60 | 42.9% | 1.7 | 5.2% |
| gdn_gated_scan_f16 | 32.00 | 100.9% | 1.9 | 1.7% |
| gdn_cumdecay_bf16 | 11.93 | 37.6% | 1.9 | 0.3% |
| gdn_gated_scan_bf16 | 27.61 | 87.1% | 2.2 | 0.3% |
| gdn2_gated_scan | 45.04 | 142.1% | 2.4 | 0.9% |
| Gated Cumulative Decay | 10.22 | 32.2% | 1.5 | 3.1% |
| Gated Delta-Rule Scan | 24.13 | 76.1% | 1.6 | 0.6% |
| Causal Depthwise Conv1D | 32.03 | 101.0% | 2.1 | 1.6% |
| gdn_cumdecay_f16 | 7.62 | 24.0% | 1.5 | 3.3% |
| gdn_gated_scan_f16 | 18.82 | 59.4% | 1.6 | 0.4% |
| gdn_cumdecay_bf16 | 6.98 | 22.0% | 1.6 | 0.4% |
| gdn_gated_scan_bf16 | 16.69 | 52.6% | 1.8 | 2.6% |
| gdn2_gated_scan | 32.41 | 102.2% | 1.6 | 1.1% |
