# jetson-j1 — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 1,690.8 | 1,868.6 | 10.5% | 1.16 | 0.16 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 4,084.7 | 4,788.7 | 17.2% | 0.72 | 0.13 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 1,983.6 | 2,337.4 | 17.8% | 1.04 | 1.06 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 1,167.6 | 1,359.8 | 16.5% | 1.25 | 0.22 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 3,957.7 | 4,222.6 | 6.7% | 0.74 | 0.13 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 1,012.6 | 1,182.7 | 16.8% | 1.45 | 0.26 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 3,969.1 | 4,305.1 | 8.5% | 0.74 | 0.13 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 504.7 | 1,036.5 | 105.3% | 1.93 | 0.26 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 919.1 | 1,092.8 | 18.9% | 1.61 | 0.29 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 517.9 | 563.1 | 8.7% | 1.99 | 2.02 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 511.2 | 590.5 | 15.5% | 1.43 | 0.26 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 1,027.8 | 1,734.8 | 68.8% | 1.43 | 0.26 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 444.2 | 517.7 | 16.5% | 1.65 | 0.30 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 921.7 | 1,102.5 | 19.6% | 1.60 | 0.28 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 6.6 | 6.8 | 3.2% | 4.65 | 0.62 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 8.2 | 8.4 | 2.5% | 9.27 | 1.00 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 27.6 | 28.7 | 4.2% | 4.98 | 1.19 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 10.7 | 10.9 | 2.4% | 2.14 | 0.38 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 13.5 | 13.9 | 2.7% | 4.52 | 0.61 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 7.4 | 7.6 | 2.1% | 3.09 | 0.55 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 10.8 | 11.2 | 3.4% | 5.63 | 0.76 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 3.8 | 3.9 | 2.7% | 4.01 | 0.54 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 5.0 | 5.2 | 3.1% | 7.63 | 0.82 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 16.7 | 17.4 | 4.4% | 4.12 | 0.98 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 5.8 | 5.9 | 0.9% | 1.96 | 0.35 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 7.2 | 7.4 | 2.9% | 4.25 | 0.57 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 4.2 | 4.2 | 1.3% | 2.75 | 0.49 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 6.0 | 6.1 | 1.7% | 5.09 | 0.68 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 25.6 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 1.16 | 4.5% | 1,690.8 | 10.5% |
| Gated Delta-Rule Scan | 0.72 | 2.8% | 4,084.7 | 17.2% |
| Causal Depthwise Conv1D | 1.04 | 4.1% | 1,983.6 | 17.8% |
| gdn_cumdecay_bf16 | 1.25 | 4.9% | 1,167.6 | 16.5% |
| gdn_gated_scan_bf16 | 0.74 | 2.9% | 3,957.7 | 6.7% |
| gdn_cumdecay_f16 | 1.45 | 5.7% | 1,012.6 | 16.8% |
| gdn_gated_scan_f16 | 0.74 | 2.9% | 3,969.1 | 8.5% |
| Gated Cumulative Decay | 1.93 | 7.5% | 504.7 | 105.3% |
| Gated Delta-Rule Scan | 1.61 | 6.3% | 919.1 | 18.9% |
| Causal Depthwise Conv1D | 1.99 | 7.8% | 517.9 | 8.7% |
| gdn_cumdecay_bf16 | 1.43 | 5.6% | 511.2 | 15.5% |
| gdn_gated_scan_bf16 | 1.43 | 5.6% | 1,027.8 | 68.8% |
| gdn_cumdecay_f16 | 1.65 | 6.4% | 444.2 | 16.5% |
| gdn_gated_scan_f16 | 1.60 | 6.2% | 921.7 | 19.6% |
| Gated Cumulative Decay | 4.65 | 18.2% | 6.6 | 3.2% |
| Gated Delta-Rule Scan | 9.27 | 36.2% | 8.2 | 2.5% |
| Causal Depthwise Conv1D | 4.98 | 19.5% | 27.6 | 4.2% |
| gdn_cumdecay_bf16 | 2.14 | 8.4% | 10.7 | 2.4% |
| gdn_gated_scan_bf16 | 4.52 | 17.7% | 13.5 | 2.7% |
| gdn_cumdecay_f16 | 3.09 | 12.1% | 7.4 | 2.1% |
| gdn_gated_scan_f16 | 5.63 | 22.0% | 10.8 | 3.4% |
| Gated Cumulative Decay | 4.01 | 15.7% | 3.8 | 2.7% |
| Gated Delta-Rule Scan | 7.63 | 29.8% | 5.0 | 3.1% |
| Causal Depthwise Conv1D | 4.12 | 16.1% | 16.7 | 4.4% |
| gdn_cumdecay_bf16 | 1.96 | 7.7% | 5.8 | 0.9% |
| gdn_gated_scan_bf16 | 4.25 | 16.6% | 7.2 | 2.9% |
| gdn_cumdecay_f16 | 2.75 | 10.7% | 4.2 | 1.3% |
| gdn_gated_scan_f16 | 5.09 | 19.9% | 6.0 | 1.7% |
