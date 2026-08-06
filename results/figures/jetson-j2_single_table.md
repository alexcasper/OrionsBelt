# jetson-j2_single — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 1,485.0 | 2,160.6 | 45.5% | 1.32 | 0.18 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 2,625.9 | 2,873.0 | 9.4% | 1.13 | 0.20 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 1,709.8 | 2,097.2 | 22.7% | 1.20 | 1.23 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 831.0 | 1,025.2 | 23.4% | 1.76 | 0.32 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 2,550.9 | 2,667.2 | 4.6% | 1.15 | 0.21 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 963.0 | 1,131.8 | 17.5% | 1.52 | 0.27 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 2,605.5 | 3,330.5 | 27.8% | 1.13 | 0.20 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 275.4 | 367.0 | 33.3% | 3.55 | 0.48 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 603.0 | 741.3 | 22.9% | 2.45 | 0.43 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 368.0 | 396.3 | 7.7% | 2.80 | 2.85 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 297.6 | 358.0 | 20.3% | 2.46 | 0.44 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 531.8 | 673.6 | 26.7% | 2.77 | 0.49 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 364.0 | 422.9 | 16.2% | 2.01 | 0.36 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 542.0 | 659.8 | 21.7% | 2.72 | 0.48 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 5.7 | 5.9 | 3.6% | 5.33 | 0.71 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 9.2 | 9.4 | 2.8% | 8.32 | 0.89 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 27.8 | 33.4 | 20.3% | 4.95 | 1.18 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 6.3 | 6.6 | 4.1% | 3.63 | 0.65 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 10.2 | 10.6 | 3.6% | 5.98 | 0.80 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 8.1 | 8.8 | 7.7% | 2.82 | 0.50 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 11.7 | 12.0 | 3.1% | 5.23 | 0.70 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 3.9 | 4.1 | 4.0% | 3.91 | 0.52 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 6.0 | 6.1 | 2.6% | 6.37 | 0.68 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 20.4 | 22.3 | 9.7% | 3.37 | 0.80 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 4.2 | 4.4 | 3.7% | 2.71 | 0.49 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 6.5 | 6.7 | 4.0% | 4.72 | 0.63 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 5.1 | 5.3 | 3.1% | 2.24 | 0.40 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 7.1 | 7.7 | 8.1% | 4.31 | 0.58 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 25.6 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 1.32 | 5.2% | 1,485.0 | 45.5% |
| Gated Delta-Rule Scan | 1.13 | 4.4% | 2,625.9 | 9.4% |
| Causal Depthwise Conv1D | 1.20 | 4.7% | 1,709.8 | 22.7% |
| gdn_cumdecay_f16 | 1.76 | 6.9% | 831.0 | 23.4% |
| gdn_gated_scan_f16 | 1.15 | 4.5% | 2,550.9 | 4.6% |
| gdn_cumdecay_bf16 | 1.52 | 5.9% | 963.0 | 17.5% |
| gdn_gated_scan_bf16 | 1.13 | 4.4% | 2,605.5 | 27.8% |
| Gated Cumulative Decay | 3.55 | 13.9% | 275.4 | 33.3% |
| Gated Delta-Rule Scan | 2.45 | 9.6% | 603.0 | 22.9% |
| Causal Depthwise Conv1D | 2.80 | 10.9% | 368.0 | 7.7% |
| gdn_cumdecay_f16 | 2.46 | 9.6% | 297.6 | 20.3% |
| gdn_gated_scan_f16 | 2.77 | 10.8% | 531.8 | 26.7% |
| gdn_cumdecay_bf16 | 2.01 | 7.9% | 364.0 | 16.2% |
| gdn_gated_scan_bf16 | 2.72 | 10.6% | 542.0 | 21.7% |
| Gated Cumulative Decay | 5.33 | 20.8% | 5.7 | 3.6% |
| Gated Delta-Rule Scan | 8.32 | 32.5% | 9.2 | 2.8% |
| Causal Depthwise Conv1D | 4.95 | 19.3% | 27.8 | 20.3% |
| gdn_cumdecay_f16 | 3.63 | 14.2% | 6.3 | 4.1% |
| gdn_gated_scan_f16 | 5.98 | 23.4% | 10.2 | 3.6% |
| gdn_cumdecay_bf16 | 2.82 | 11.0% | 8.1 | 7.7% |
| gdn_gated_scan_bf16 | 5.23 | 20.4% | 11.7 | 3.1% |
| Gated Cumulative Decay | 3.91 | 15.3% | 3.9 | 4.0% |
| Gated Delta-Rule Scan | 6.37 | 24.9% | 6.0 | 2.6% |
| Causal Depthwise Conv1D | 3.37 | 13.2% | 20.4 | 9.7% |
| gdn_cumdecay_f16 | 2.71 | 10.6% | 4.2 | 3.7% |
| gdn_gated_scan_f16 | 4.72 | 18.4% | 6.5 | 4.0% |
| gdn_cumdecay_bf16 | 2.24 | 8.8% | 5.1 | 3.1% |
| gdn_gated_scan_bf16 | 4.31 | 16.8% | 7.1 | 8.1% |
