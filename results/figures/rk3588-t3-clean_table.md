# rk3588-t3-clean — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 91.3 | 98.0 | 7.3% | 21.39 | 2.87 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 280.3 | 298.1 | 6.3% | 10.56 | 1.87 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 100.0 | 103.5 | 3.5% | 20.59 | 20.96 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 41.7 | 42.0 | 0.7% | 35.12 | 6.28 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 282.4 | 288.2 | 2.1% | 10.43 | 1.86 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 58.3 | 60.1 | 3.0% | 25.11 | 4.49 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 281.8 | 296.1 | 5.1% | 10.45 | 1.86 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 760.7 | 862.8 | 13.4% | 6.46 | 1.38 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 39.1 | 40.0 | 2.2% | 24.98 | 3.35 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 128.3 | 142.1 | 10.7% | 11.53 | 2.04 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 64.5 | 66.2 | 2.7% | 15.98 | 16.27 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 18.4 | 18.7 | 1.6% | 39.86 | 7.13 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 130.7 | 137.4 | 5.1% | 11.27 | 2.01 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 29.5 | 29.8 | 1.0% | 24.86 | 4.45 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 131.0 | 133.0 | 1.6% | 11.24 | 2.00 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 322.6 | 330.2 | 2.4% | 7.62 | 1.63 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.0 | 1.0 | 0.6% | 29.72 | 3.99 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 1.4 | 1.4 | 1.9% | 56.37 | 6.05 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 3.5 | 3.6 | 0.9% | 38.78 | 9.25 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 1.1 | 1.1 | 1.4% | 21.62 | 3.87 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 1.3 | 1.3 | 0.2% | 45.79 | 6.15 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 1.3 | 1.3 | 0.2% | 17.32 | 3.10 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 1.7 | 1.7 | 1.4% | 36.65 | 4.92 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 1.9 | 1.9 | 1.2% | 57.22 | 8.78 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 0.9 | 0.9 | 0.7% | 17.32 | 2.32 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.0 | 1.0 | 0.9% | 37.58 | 4.04 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 2.1 | 2.2 | 1.8% | 32.16 | 7.67 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 0.9 | 0.9 | 1.0% | 12.78 | 2.29 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.1 | 1.1 | 0.8% | 27.97 | 3.75 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.1 | 1.1 | 0.8% | 10.63 | 1.90 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 1.2 | 1.2 | 1.0% | 25.03 | 3.36 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.2 | 1.2 | 1.3% | 46.12 | 7.07 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 21.39 | 67.5% | 91.3 | 7.3% |
| Gated Delta-Rule Scan | 10.56 | 33.3% | 280.3 | 6.3% |
| Causal Depthwise Conv1D | 20.59 | 65.0% | 100.0 | 3.5% |
| gdn_cumdecay_f16 | 35.12 | 110.8% | 41.7 | 0.7% |
| gdn_gated_scan_f16 | 10.43 | 32.9% | 282.4 | 2.1% |
| gdn_cumdecay_bf16 | 25.11 | 79.2% | 58.3 | 3.0% |
| gdn_gated_scan_bf16 | 10.45 | 33.0% | 281.8 | 5.1% |
| gdn2_gated_scan | 6.46 | 20.4% | 760.7 | 13.4% |
| Gated Cumulative Decay | 24.98 | 78.8% | 39.1 | 2.2% |
| Gated Delta-Rule Scan | 11.53 | 36.4% | 128.3 | 10.7% |
| Causal Depthwise Conv1D | 15.98 | 50.4% | 64.5 | 2.7% |
| gdn_cumdecay_f16 | 39.86 | 125.7% | 18.4 | 1.6% |
| gdn_gated_scan_f16 | 11.27 | 35.6% | 130.7 | 5.1% |
| gdn_cumdecay_bf16 | 24.86 | 78.4% | 29.5 | 1.0% |
| gdn_gated_scan_bf16 | 11.24 | 35.5% | 131.0 | 1.6% |
| gdn2_gated_scan | 7.62 | 24.0% | 322.6 | 2.4% |
| Gated Cumulative Decay | 29.72 | 93.8% | 1.0 | 0.6% |
| Gated Delta-Rule Scan | 56.37 | 177.8% | 1.4 | 1.9% |
| Causal Depthwise Conv1D | 38.78 | 122.3% | 3.5 | 0.9% |
| gdn_cumdecay_f16 | 21.62 | 68.2% | 1.1 | 1.4% |
| gdn_gated_scan_f16 | 45.79 | 144.4% | 1.3 | 0.2% |
| gdn_cumdecay_bf16 | 17.32 | 54.6% | 1.3 | 0.2% |
| gdn_gated_scan_bf16 | 36.65 | 115.6% | 1.7 | 1.4% |
| gdn2_gated_scan | 57.22 | 180.5% | 1.9 | 1.2% |
| Gated Cumulative Decay | 17.32 | 54.6% | 0.9 | 0.7% |
| Gated Delta-Rule Scan | 37.58 | 118.5% | 1.0 | 0.9% |
| Causal Depthwise Conv1D | 32.16 | 101.5% | 2.1 | 1.8% |
| gdn_cumdecay_f16 | 12.78 | 40.3% | 0.9 | 1.0% |
| gdn_gated_scan_f16 | 27.97 | 88.2% | 1.1 | 0.8% |
| gdn_cumdecay_bf16 | 10.63 | 33.5% | 1.1 | 0.8% |
| gdn_gated_scan_bf16 | 25.03 | 79.0% | 1.2 | 1.0% |
| gdn2_gated_scan | 46.12 | 145.5% | 1.2 | 1.3% |
