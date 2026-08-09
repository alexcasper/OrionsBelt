# jetson-j1-clean — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 1,161.0 | 1,266.5 | 9.1% | 1.68 | 0.23 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 2,601.5 | 2,709.5 | 4.2% | 1.14 | 0.20 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 1,566.0 | 1,708.9 | 9.1% | 1.32 | 1.34 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 782.7 | 922.4 | 17.8% | 1.87 | 0.33 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 2,597.2 | 2,800.9 | 7.8% | 1.13 | 0.20 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 899.1 | 1,132.8 | 26.0% | 1.63 | 0.29 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 2,630.8 | 2,744.1 | 4.3% | 1.12 | 0.20 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 5,706.4 | 6,029.0 | 5.7% | 0.86 | 0.18 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 280.1 | 377.7 | 34.9% | 3.49 | 0.47 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 636.8 | 799.2 | 25.5% | 2.32 | 0.41 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 384.6 | 437.9 | 13.9% | 2.68 | 2.73 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 299.2 | 413.2 | 38.1% | 2.45 | 0.44 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 605.3 | 784.1 | 29.5% | 2.43 | 0.43 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 362.9 | 421.1 | 16.0% | 2.02 | 0.36 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 590.3 | 648.7 | 9.9% | 2.49 | 0.44 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 1,773.2 | 2,011.7 | 13.5% | 1.39 | 0.30 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 4.7 | 4.9 | 3.2% | 6.43 | 0.86 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 7.6 | 7.8 | 1.9% | 9.99 | 1.07 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 32.5 | 35.2 | 8.3% | 4.23 | 1.01 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 5.4 | 5.5 | 2.1% | 4.25 | 0.76 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 9.3 | 9.5 | 1.5% | 6.53 | 0.88 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 7.0 | 7.1 | 1.5% | 3.27 | 0.58 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 10.5 | 10.6 | 0.7% | 5.79 | 0.78 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 11.9 | 12.0 | 1.1% | 8.98 | 1.38 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 2.9 | 3.1 | 4.4% | 5.22 | 0.70 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 5.1 | 5.2 | 1.9% | 7.54 | 0.81 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 17.1 | 17.4 | 2.1% | 4.02 | 0.96 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 3.3 | 3.4 | 3.1% | 3.49 | 0.62 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 5.4 | 5.5 | 2.0% | 5.63 | 0.76 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 4.1 | 4.2 | 2.4% | 2.77 | 0.50 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 6.4 | 6.5 | 1.7% | 4.75 | 0.64 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 7.0 | 7.1 | 2.1% | 7.64 | 1.17 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 23.8 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 1.68 | 7.1% | 1,161.0 | 9.1% |
| Gated Delta-Rule Scan | 1.14 | 4.8% | 2,601.5 | 4.2% |
| Causal Depthwise Conv1D | 1.32 | 5.5% | 1,566.0 | 9.1% |
| gdn_cumdecay_f16 | 1.87 | 7.9% | 782.7 | 17.8% |
| gdn_gated_scan_f16 | 1.13 | 4.7% | 2,597.2 | 7.8% |
| gdn_cumdecay_bf16 | 1.63 | 6.8% | 899.1 | 26.0% |
| gdn_gated_scan_bf16 | 1.12 | 4.7% | 2,630.8 | 4.3% |
| gdn2_gated_scan | 0.86 | 3.6% | 5,706.4 | 5.7% |
| Gated Cumulative Decay | 3.49 | 14.7% | 280.1 | 34.9% |
| Gated Delta-Rule Scan | 2.32 | 9.7% | 636.8 | 25.5% |
| Causal Depthwise Conv1D | 2.68 | 11.3% | 384.6 | 13.9% |
| gdn_cumdecay_f16 | 2.45 | 10.3% | 299.2 | 38.1% |
| gdn_gated_scan_f16 | 2.43 | 10.2% | 605.3 | 29.5% |
| gdn_cumdecay_bf16 | 2.02 | 8.5% | 362.9 | 16.0% |
| gdn_gated_scan_bf16 | 2.49 | 10.5% | 590.3 | 9.9% |
| gdn2_gated_scan | 1.39 | 5.8% | 1,773.2 | 13.5% |
| Gated Cumulative Decay | 6.43 | 27.0% | 4.7 | 3.2% |
| Gated Delta-Rule Scan | 9.99 | 42.0% | 7.6 | 1.9% |
| Causal Depthwise Conv1D | 4.23 | 17.8% | 32.5 | 8.3% |
| gdn_cumdecay_f16 | 4.25 | 17.9% | 5.4 | 2.1% |
| gdn_gated_scan_f16 | 6.53 | 27.4% | 9.3 | 1.5% |
| gdn_cumdecay_bf16 | 3.27 | 13.7% | 7.0 | 1.5% |
| gdn_gated_scan_bf16 | 5.79 | 24.3% | 10.5 | 0.7% |
| gdn2_gated_scan | 8.98 | 37.7% | 11.9 | 1.1% |
| Gated Cumulative Decay | 5.22 | 21.9% | 2.9 | 4.4% |
| Gated Delta-Rule Scan | 7.54 | 31.7% | 5.1 | 1.9% |
| Causal Depthwise Conv1D | 4.02 | 16.9% | 17.1 | 2.1% |
| gdn_cumdecay_f16 | 3.49 | 14.7% | 3.3 | 3.1% |
| gdn_gated_scan_f16 | 5.63 | 23.7% | 5.4 | 2.0% |
| gdn_cumdecay_bf16 | 2.77 | 11.6% | 4.1 | 2.4% |
| gdn_gated_scan_bf16 | 4.75 | 20.0% | 6.4 | 1.7% |
| gdn2_gated_scan | 7.64 | 32.1% | 7.0 | 2.1% |
