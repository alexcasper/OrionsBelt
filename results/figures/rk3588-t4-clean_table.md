# rk3588-t4-clean — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 261.9 | 276.8 | 5.7% | 7.46 | 1.00 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 514.5 | 576.4 | 12.0% | 5.75 | 1.02 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 294.6 | 312.4 | 6.0% | 6.99 | 7.12 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 172.4 | 179.4 | 4.1% | 8.50 | 1.52 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 509.0 | 561.2 | 10.3% | 5.79 | 1.03 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 245.3 | 251.7 | 2.6% | 5.97 | 1.07 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 517.7 | 571.4 | 10.4% | 5.69 | 1.01 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 1,163.5 | 1,354.6 | 16.4% | 4.22 | 0.90 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 130.1 | 131.8 | 1.3% | 7.51 | 1.01 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 186.7 | 198.1 | 6.1% | 7.93 | 1.40 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 126.3 | 127.8 | 1.2% | 8.15 | 8.30 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 79.9 | 82.3 | 2.9% | 9.16 | 1.64 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 188.7 | 201.8 | 7.0% | 7.80 | 1.39 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 119.9 | 121.6 | 1.5% | 6.11 | 1.09 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 188.7 | 191.6 | 1.5% | 7.80 | 1.39 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 298.1 | 310.6 | 4.2% | 8.24 | 1.76 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.8 | 2.0 | 16.7% | 17.44 | 2.34 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 2.3 | 2.3 | 0.0% | 32.69 | 3.51 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 6.4 | 6.4 | 0.0% | 21.40 | 5.11 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 2.0 | 2.0 | 0.0% | 11.21 | 2.01 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 2.9 | 2.9 | 0.0% | 20.92 | 2.81 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 2.9 | 2.9 | 0.0% | 7.85 | 1.40 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 4.1 | 4.1 | 0.0% | 14.95 | 2.01 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 2.9 | 3.2 | 10.0% | 36.62 | 5.62 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 1.2 | 1.5 | 24.9% | 13.08 | 1.76 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.5 | 1.5 | 0.1% | 26.16 | 2.81 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 2.9 | 3.2 | 10.0% | 23.54 | 5.62 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 1.2 | 1.5 | 25.0% | 9.80 | 1.75 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.8 | 1.8 | 0.1% | 17.44 | 2.34 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.8 | 1.8 | 0.1% | 6.54 | 1.17 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 2.3 | 2.3 | 0.0% | 13.08 | 1.76 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.8 | 1.8 | 0.1% | 30.52 | 4.68 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 34.0 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 7.46 | 21.9% | 261.9 | 5.7% |
| Gated Delta-Rule Scan | 5.75 | 16.9% | 514.5 | 12.0% |
| Causal Depthwise Conv1D | 6.99 | 20.6% | 294.6 | 6.0% |
| gdn_cumdecay_f16 | 8.50 | 25.0% | 172.4 | 4.1% |
| gdn_gated_scan_f16 | 5.79 | 17.0% | 509.0 | 10.3% |
| gdn_cumdecay_bf16 | 5.97 | 17.6% | 245.3 | 2.6% |
| gdn_gated_scan_bf16 | 5.69 | 16.7% | 517.7 | 10.4% |
| gdn2_gated_scan | 4.22 | 12.4% | 1,163.5 | 16.4% |
| Gated Cumulative Decay | 7.51 | 22.1% | 130.1 | 1.3% |
| Gated Delta-Rule Scan | 7.93 | 23.3% | 186.7 | 6.1% |
| Causal Depthwise Conv1D | 8.15 | 24.0% | 126.3 | 1.2% |
| gdn_cumdecay_f16 | 9.16 | 26.9% | 79.9 | 2.9% |
| gdn_gated_scan_f16 | 7.80 | 22.9% | 188.7 | 7.0% |
| gdn_cumdecay_bf16 | 6.11 | 18.0% | 119.9 | 1.5% |
| gdn_gated_scan_bf16 | 7.80 | 22.9% | 188.7 | 1.5% |
| gdn2_gated_scan | 8.24 | 24.2% | 298.1 | 4.2% |
| Gated Cumulative Decay | 17.44 | 51.3% | 1.8 | 16.7% |
| Gated Delta-Rule Scan | 32.69 | 96.1% | 2.3 | 0.0% |
| Causal Depthwise Conv1D | 21.40 | 62.9% | 6.4 | 0.0% |
| gdn_cumdecay_f16 | 11.21 | 33.0% | 2.0 | 0.0% |
| gdn_gated_scan_f16 | 20.92 | 61.5% | 2.9 | 0.0% |
| gdn_cumdecay_bf16 | 7.85 | 23.1% | 2.9 | 0.0% |
| gdn_gated_scan_bf16 | 14.95 | 44.0% | 4.1 | 0.0% |
| gdn2_gated_scan | 36.62 | 107.7% | 2.9 | 10.0% |
| Gated Cumulative Decay | 13.08 | 38.5% | 1.2 | 24.9% |
| Gated Delta-Rule Scan | 26.16 | 76.9% | 1.5 | 0.1% |
| Causal Depthwise Conv1D | 23.54 | 69.2% | 2.9 | 10.0% |
| gdn_cumdecay_f16 | 9.80 | 28.8% | 1.2 | 25.0% |
| gdn_gated_scan_f16 | 17.44 | 51.3% | 1.8 | 0.1% |
| gdn_cumdecay_bf16 | 6.54 | 19.2% | 1.8 | 0.1% |
| gdn_gated_scan_bf16 | 13.08 | 38.5% | 2.3 | 0.0% |
| gdn2_gated_scan | 30.52 | 89.8% | 1.8 | 0.1% |
