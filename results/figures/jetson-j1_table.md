# jetson-j1 — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 537.7 | 709.8 | 32.0% | 3.63 | 0.49 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 997.3 | 1,321.8 | 32.5% | 2.97 | 0.53 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 576.2 | 659.0 | 14.4% | 3.57 | 3.64 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 355.2 | 509.2 | 43.3% | 4.12 | 0.74 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 1,018.8 | 1,250.2 | 22.7% | 2.89 | 0.51 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 374.4 | 430.7 | 15.0% | 3.91 | 0.70 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 1,003.2 | 1,162.9 | 15.9% | 2.94 | 0.52 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 1,674.7 | 1,875.7 | 12.0% | 2.93 | 0.63 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 210.6 | 243.5 | 15.7% | 4.64 | 0.62 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 286.8 | 360.9 | 25.9% | 5.16 | 0.91 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 203.9 | 228.5 | 12.1% | 5.05 | 5.14 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 171.5 | 187.2 | 9.2% | 4.27 | 0.76 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 288.0 | 739.5 | 156.8% | 5.11 | 0.91 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 165.1 | 1,372.1 | 731.0% | 4.44 | 0.79 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 272.1 | 332.2 | 22.1% | 5.41 | 0.96 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 451.2 | 660.5 | 46.4% | 5.45 | 1.16 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 3.6 | 4.0 | 10.1% | 8.49 | 1.14 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 5.6 | 5.8 | 2.8% | 13.56 | 1.46 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 10.7 | 11.2 | 4.4% | 12.80 | 3.05 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 3.9 | 4.0 | 2.7% | 5.86 | 1.05 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 5.1 | 5.3 | 4.1% | 12.08 | 1.62 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 4.0 | 4.1 | 2.6% | 5.71 | 1.02 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 5.3 | 5.5 | 4.0% | 11.60 | 1.56 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 6.4 | 6.6 | 3.2% | 16.67 | 2.56 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 3.0 | 3.1 | 3.5% | 5.14 | 0.69 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 3.9 | 4.1 | 5.3% | 9.76 | 1.05 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 12.1 | 12.5 | 3.4% | 5.68 | 1.36 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 3.2 | 3.3 | 4.9% | 3.60 | 0.64 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 4.0 | 4.1 | 4.0% | 7.71 | 1.03 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 3.6 | 3.8 | 4.3% | 3.14 | 0.56 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 4.2 | 4.4 | 3.7% | 7.23 | 0.97 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 4.8 | 5.0 | 4.3% | 11.14 | 1.71 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 25.6 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 3.63 | 14.2% | 537.7 | 32.0% |
| Gated Delta-Rule Scan | 2.97 | 11.6% | 997.3 | 32.5% |
| Causal Depthwise Conv1D | 3.57 | 13.9% | 576.2 | 14.4% |
| gdn_cumdecay_f16 | 4.12 | 16.1% | 355.2 | 43.3% |
| gdn_gated_scan_f16 | 2.89 | 11.3% | 1,018.8 | 22.7% |
| gdn_cumdecay_bf16 | 3.91 | 15.3% | 374.4 | 15.0% |
| gdn_gated_scan_bf16 | 2.94 | 11.5% | 1,003.2 | 15.9% |
| gdn2_gated_scan | 2.93 | 11.4% | 1,674.7 | 12.0% |
| Gated Cumulative Decay | 4.64 | 18.1% | 210.6 | 15.7% |
| Gated Delta-Rule Scan | 5.16 | 20.2% | 286.8 | 25.9% |
| Causal Depthwise Conv1D | 5.05 | 19.7% | 203.9 | 12.1% |
| gdn_cumdecay_f16 | 4.27 | 16.7% | 171.5 | 9.2% |
| gdn_gated_scan_f16 | 5.11 | 20.0% | 288.0 | 156.8% |
| gdn_cumdecay_bf16 | 4.44 | 17.3% | 165.1 | 731.0% |
| gdn_gated_scan_bf16 | 5.41 | 21.1% | 272.1 | 22.1% |
| gdn2_gated_scan | 5.45 | 21.3% | 451.2 | 46.4% |
| Gated Cumulative Decay | 8.49 | 33.2% | 3.6 | 10.1% |
| Gated Delta-Rule Scan | 13.56 | 53.0% | 5.6 | 2.8% |
| Causal Depthwise Conv1D | 12.80 | 50.0% | 10.7 | 4.4% |
| gdn_cumdecay_f16 | 5.86 | 22.9% | 3.9 | 2.7% |
| gdn_gated_scan_f16 | 12.08 | 47.2% | 5.1 | 4.1% |
| gdn_cumdecay_bf16 | 5.71 | 22.3% | 4.0 | 2.6% |
| gdn_gated_scan_bf16 | 11.60 | 45.3% | 5.3 | 4.0% |
| gdn2_gated_scan | 16.67 | 65.1% | 6.4 | 3.2% |
| Gated Cumulative Decay | 5.14 | 20.1% | 3.0 | 3.5% |
| Gated Delta-Rule Scan | 9.76 | 38.1% | 3.9 | 5.3% |
| Causal Depthwise Conv1D | 5.68 | 22.2% | 12.1 | 3.4% |
| gdn_cumdecay_f16 | 3.60 | 14.1% | 3.2 | 4.9% |
| gdn_gated_scan_f16 | 7.71 | 30.1% | 4.0 | 4.0% |
| gdn_cumdecay_bf16 | 3.14 | 12.3% | 3.6 | 4.3% |
| gdn_gated_scan_bf16 | 7.23 | 28.2% | 4.2 | 3.7% |
| gdn2_gated_scan | 11.14 | 43.5% | 4.8 | 4.3% |
