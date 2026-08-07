# rk3588-t4_gdn2_vs_gdn1_little_single — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 1,376.8 | 1,442.1 | 4.7% | 1.42 | 0.19 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 3,421.2 | 3,597.1 | 5.1% | 0.87 | 0.15 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 1,587.7 | 1,662.6 | 4.7% | 1.30 | 1.32 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 841.5 | 864.3 | 2.7% | 1.74 | 0.31 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 3,474.0 | 3,604.7 | 3.8% | 0.85 | 0.15 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 974.5 | 1,010.7 | 3.7% | 1.50 | 0.27 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 3,468.4 | 3,598.8 | 3.8% | 0.85 | 0.15 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 9,201.3 | 9,344.8 | 1.6% | 0.53 | 0.11 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 591.5 | 635.6 | 7.4% | 1.65 | 0.22 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 1,009.2 | 1,042.8 | 3.3% | 1.47 | 0.26 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 668.8 | 686.6 | 2.7% | 1.54 | 1.57 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 392.9 | 412.7 | 5.0% | 1.86 | 0.33 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 1,003.7 | 1,049.5 | 4.6% | 1.47 | 0.26 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 452.1 | 480.1 | 6.2% | 1.62 | 0.29 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 1,012.2 | 1,033.4 | 2.1% | 1.45 | 0.26 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 2,753.2 | 2,943.4 | 6.9% | 0.89 | 0.19 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 7.0 | 7.0 | 0.0% | 4.36 | 0.59 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 15.5 | 16.0 | 3.8% | 4.93 | 0.53 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 54.0 | 54.8 | 1.6% | 2.54 | 0.61 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 7.6 | 7.9 | 3.8% | 3.02 | 0.54 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 14.6 | 14.9 | 2.0% | 4.19 | 0.56 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 9.6 | 9.6 | 0.0% | 2.38 | 0.43 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 16.6 | 16.9 | 1.8% | 3.67 | 0.49 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 33.5 | 34.1 | 1.7% | 3.18 | 0.49 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 4.4 | 4.4 | 0.0% | 3.49 | 0.47 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 6.4 | 6.7 | 4.5% | 5.94 | 0.64 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 27.7 | 27.7 | 0.0% | 2.48 | 0.59 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 5.0 | 5.0 | 0.0% | 2.31 | 0.41 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 6.7 | 6.7 | 0.0% | 4.55 | 0.61 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 5.8 | 5.8 | 0.0% | 1.96 | 0.35 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 7.6 | 7.9 | 3.8% | 4.02 | 0.54 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 15.2 | 15.5 | 1.9% | 3.52 | 0.54 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 1.42 | 4.5% | 1,376.8 | 4.7% |
| Gated Delta-Rule Scan | 0.87 | 2.7% | 3,421.2 | 5.1% |
| Causal Depthwise Conv1D | 1.30 | 4.1% | 1,587.7 | 4.7% |
| gdn_cumdecay_f16 | 1.74 | 5.5% | 841.5 | 2.7% |
| gdn_gated_scan_f16 | 0.85 | 2.7% | 3,474.0 | 3.8% |
| gdn_cumdecay_bf16 | 1.50 | 4.7% | 974.5 | 3.7% |
| gdn_gated_scan_bf16 | 0.85 | 2.7% | 3,468.4 | 3.8% |
| gdn2_gated_scan | 0.53 | 1.7% | 9,201.3 | 1.6% |
| Gated Cumulative Decay | 1.65 | 5.2% | 591.5 | 7.4% |
| Gated Delta-Rule Scan | 1.47 | 4.6% | 1,009.2 | 3.3% |
| Causal Depthwise Conv1D | 1.54 | 4.9% | 668.8 | 2.7% |
| gdn_cumdecay_f16 | 1.86 | 5.9% | 392.9 | 5.0% |
| gdn_gated_scan_f16 | 1.47 | 4.6% | 1,003.7 | 4.6% |
| gdn_cumdecay_bf16 | 1.62 | 5.1% | 452.1 | 6.2% |
| gdn_gated_scan_bf16 | 1.45 | 4.6% | 1,012.2 | 2.1% |
| gdn2_gated_scan | 0.89 | 2.8% | 2,753.2 | 6.9% |
| Gated Cumulative Decay | 4.36 | 13.8% | 7.0 | 0.0% |
| Gated Delta-Rule Scan | 4.93 | 15.6% | 15.5 | 3.8% |
| Causal Depthwise Conv1D | 2.54 | 8.0% | 54.0 | 1.6% |
| gdn_cumdecay_f16 | 3.02 | 9.5% | 7.6 | 3.8% |
| gdn_gated_scan_f16 | 4.19 | 13.2% | 14.6 | 2.0% |
| gdn_cumdecay_bf16 | 2.38 | 7.5% | 9.6 | 0.0% |
| gdn_gated_scan_bf16 | 3.67 | 11.6% | 16.6 | 1.8% |
| gdn2_gated_scan | 3.18 | 10.0% | 33.5 | 1.7% |
| Gated Cumulative Decay | 3.49 | 11.0% | 4.4 | 0.0% |
| Gated Delta-Rule Scan | 5.94 | 18.7% | 6.4 | 4.5% |
| Causal Depthwise Conv1D | 2.48 | 7.8% | 27.7 | 0.0% |
| gdn_cumdecay_f16 | 2.31 | 7.3% | 5.0 | 0.0% |
| gdn_gated_scan_f16 | 4.55 | 14.4% | 6.7 | 0.0% |
| gdn_cumdecay_bf16 | 1.96 | 6.2% | 5.8 | 0.0% |
| gdn_gated_scan_bf16 | 4.02 | 12.7% | 7.6 | 3.8% |
| gdn2_gated_scan | 3.52 | 11.1% | 15.2 | 1.9% |
