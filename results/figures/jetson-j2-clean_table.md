# jetson-j2-clean — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 1,301.4 | 1,565.3 | 20.3% | 1.50 | 0.20 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 2,714.0 | 3,166.5 | 16.7% | 1.09 | 0.19 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 2,210.7 | 2,596.5 | 17.4% | 0.93 | 0.95 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 797.0 | 977.5 | 22.7% | 1.84 | 0.33 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 3,023.4 | 3,240.6 | 7.2% | 0.97 | 0.17 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 1,634.7 | 1,892.5 | 15.8% | 0.90 | 0.16 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 3,061.9 | 3,203.2 | 4.6% | 0.96 | 0.17 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 4,595.8 | 4,930.9 | 7.3% | 1.07 | 0.23 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 301.3 | 376.5 | 24.9% | 3.24 | 0.44 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 896.9 | 1,262.7 | 40.8% | 1.65 | 0.29 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 423.4 | 968.8 | 128.8% | 2.43 | 2.48 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 532.5 | 703.2 | 32.1% | 1.38 | 0.25 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 636.1 | 815.9 | 28.3% | 2.32 | 0.41 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 472.2 | 740.0 | 56.7% | 1.55 | 0.28 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 728.6 | 1,305.9 | 79.2% | 2.02 | 0.36 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 1,522.9 | 2,031.3 | 33.4% | 1.61 | 0.34 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 5.8 | 5.9 | 2.7% | 5.28 | 0.71 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 10.3 | 10.5 | 2.5% | 7.44 | 0.80 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 24.8 | 26.2 | 5.9% | 5.54 | 1.32 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 6.4 | 6.6 | 3.2% | 3.57 | 0.64 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 10.5 | 11.1 | 6.5% | 5.83 | 0.78 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 8.1 | 8.3 | 2.6% | 2.82 | 0.50 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 12.1 | 22.3 | 84.5% | 5.05 | 0.68 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 14.1 | 23.0 | 63.7% | 7.60 | 1.17 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 4.0 | 4.1 | 2.6% | 3.85 | 0.52 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 5.9 | 6.0 | 2.7% | 6.48 | 0.70 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 20.3 | 21.8 | 7.2% | 3.38 | 0.81 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 4.3 | 4.4 | 2.4% | 2.68 | 0.48 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 6.6 | 6.7 | 2.4% | 4.65 | 0.62 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 5.0 | 5.2 | 4.2% | 2.29 | 0.41 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 7.1 | 7.3 | 3.0% | 4.31 | 0.58 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 8.0 | 8.2 | 2.0% | 6.66 | 1.02 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 23.8 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 1.50 | 6.3% | 1,301.4 | 20.3% |
| Gated Delta-Rule Scan | 1.09 | 4.6% | 2,714.0 | 16.7% |
| Causal Depthwise Conv1D | 0.93 | 3.9% | 2,210.7 | 17.4% |
| gdn_cumdecay_f16 | 1.84 | 7.7% | 797.0 | 22.7% |
| gdn_gated_scan_f16 | 0.97 | 4.1% | 3,023.4 | 7.2% |
| gdn_cumdecay_bf16 | 0.90 | 3.8% | 1,634.7 | 15.8% |
| gdn_gated_scan_bf16 | 0.96 | 4.0% | 3,061.9 | 4.6% |
| gdn2_gated_scan | 1.07 | 4.5% | 4,595.8 | 7.3% |
| Gated Cumulative Decay | 3.24 | 13.6% | 301.3 | 24.9% |
| Gated Delta-Rule Scan | 1.65 | 6.9% | 896.9 | 40.8% |
| Causal Depthwise Conv1D | 2.43 | 10.2% | 423.4 | 128.8% |
| gdn_cumdecay_f16 | 1.38 | 5.8% | 532.5 | 32.1% |
| gdn_gated_scan_f16 | 2.32 | 9.7% | 636.1 | 28.3% |
| gdn_cumdecay_bf16 | 1.55 | 6.5% | 472.2 | 56.7% |
| gdn_gated_scan_bf16 | 2.02 | 8.5% | 728.6 | 79.2% |
| gdn2_gated_scan | 1.61 | 6.8% | 1,522.9 | 33.4% |
| Gated Cumulative Decay | 5.28 | 22.2% | 5.8 | 2.7% |
| Gated Delta-Rule Scan | 7.44 | 31.3% | 10.3 | 2.5% |
| Causal Depthwise Conv1D | 5.54 | 23.3% | 24.8 | 5.9% |
| gdn_cumdecay_f16 | 3.57 | 15.0% | 6.4 | 3.2% |
| gdn_gated_scan_f16 | 5.83 | 24.5% | 10.5 | 6.5% |
| gdn_cumdecay_bf16 | 2.82 | 11.8% | 8.1 | 2.6% |
| gdn_gated_scan_bf16 | 5.05 | 21.2% | 12.1 | 84.5% |
| gdn2_gated_scan | 7.60 | 31.9% | 14.1 | 63.7% |
| Gated Cumulative Decay | 3.85 | 16.2% | 4.0 | 2.6% |
| Gated Delta-Rule Scan | 6.48 | 27.2% | 5.9 | 2.7% |
| Causal Depthwise Conv1D | 3.38 | 14.2% | 20.3 | 7.2% |
| gdn_cumdecay_f16 | 2.68 | 11.3% | 4.3 | 2.4% |
| gdn_gated_scan_f16 | 4.65 | 19.5% | 6.6 | 2.4% |
| gdn_cumdecay_bf16 | 2.29 | 9.6% | 5.0 | 4.2% |
| gdn_gated_scan_bf16 | 4.31 | 18.1% | 7.1 | 3.0% |
| gdn2_gated_scan | 6.66 | 28.0% | 8.0 | 2.0% |
