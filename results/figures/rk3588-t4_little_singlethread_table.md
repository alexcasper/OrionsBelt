# rk3588-t4_little_singlethread — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 1,274.4 | 1,321.9 | 3.7% | 1.53 | 0.21 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 3,369.9 | 3,589.5 | 6.5% | 0.88 | 0.16 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 1,587.9 | 1,671.1 | 5.2% | 1.30 | 1.32 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 844.7 | 882.1 | 4.4% | 1.73 | 0.31 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 3,361.1 | 3,579.0 | 6.5% | 0.88 | 0.16 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 966.9 | 1,032.0 | 6.7% | 1.51 | 0.27 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 3,390.3 | 3,901.6 | 15.1% | 0.87 | 0.15 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 9,108.5 | 9,370.5 | 2.9% | 0.54 | 0.12 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 581.6 | 617.2 | 6.1% | 1.68 | 0.23 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 990.6 | 1,040.2 | 5.0% | 1.49 | 0.26 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 551.9 | 579.0 | 4.9% | 1.87 | 1.90 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 417.7 | 437.8 | 4.8% | 1.75 | 0.31 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 990.0 | 1,027.3 | 3.8% | 1.49 | 0.26 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 477.2 | 495.3 | 3.8% | 1.53 | 0.27 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 1,001.4 | 1,036.1 | 3.5% | 1.47 | 0.26 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 2,444.0 | 2,717.7 | 11.2% | 1.01 | 0.21 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 6.6 | 6.8 | 4.0% | 4.65 | 0.62 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 15.8 | 17.5 | 11.1% | 4.84 | 0.52 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 53.1 | 53.7 | 1.1% | 2.59 | 0.62 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 8.2 | 8.4 | 2.5% | 2.80 | 0.50 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 13.4 | 14.0 | 4.3% | 4.55 | 0.61 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 10.1 | 10.3 | 2.0% | 2.28 | 0.41 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 15.8 | 16.6 | 5.6% | 3.88 | 0.52 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 32.4 | 33.0 | 1.8% | 3.30 | 0.51 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 4.3 | 4.5 | 4.7% | 3.55 | 0.48 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 6.8 | 7.0 | 2.5% | 5.60 | 0.60 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 27.7 | 28.3 | 2.1% | 2.48 | 0.59 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 5.1 | 5.2 | 2.6% | 2.26 | 0.40 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 6.4 | 6.6 | 2.6% | 4.77 | 0.64 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 6.0 | 6.1 | 2.3% | 1.91 | 0.34 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 7.7 | 7.9 | 2.2% | 3.96 | 0.53 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 15.1 | 15.3 | 1.2% | 3.53 | 0.54 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 1.53 | 4.8% | 1,274.4 | 3.7% |
| Gated Delta-Rule Scan | 0.88 | 2.8% | 3,369.9 | 6.5% |
| Causal Depthwise Conv1D | 1.30 | 4.1% | 1,587.9 | 5.2% |
| gdn_cumdecay_f16 | 1.73 | 5.5% | 844.7 | 4.4% |
| gdn_gated_scan_f16 | 0.88 | 2.8% | 3,361.1 | 6.5% |
| gdn_cumdecay_bf16 | 1.51 | 4.8% | 966.9 | 6.7% |
| gdn_gated_scan_bf16 | 0.87 | 2.7% | 3,390.3 | 15.1% |
| gdn2_gated_scan | 0.54 | 1.7% | 9,108.5 | 2.9% |
| Gated Cumulative Decay | 1.68 | 5.3% | 581.6 | 6.1% |
| Gated Delta-Rule Scan | 1.49 | 4.7% | 990.6 | 5.0% |
| Causal Depthwise Conv1D | 1.87 | 5.9% | 551.9 | 4.9% |
| gdn_cumdecay_f16 | 1.75 | 5.5% | 417.7 | 4.8% |
| gdn_gated_scan_f16 | 1.49 | 4.7% | 990.0 | 3.8% |
| gdn_cumdecay_bf16 | 1.53 | 4.8% | 477.2 | 3.8% |
| gdn_gated_scan_bf16 | 1.47 | 4.6% | 1,001.4 | 3.5% |
| gdn2_gated_scan | 1.01 | 3.2% | 2,444.0 | 11.2% |
| Gated Cumulative Decay | 4.65 | 14.7% | 6.6 | 4.0% |
| Gated Delta-Rule Scan | 4.84 | 15.3% | 15.8 | 11.1% |
| Causal Depthwise Conv1D | 2.59 | 8.2% | 53.1 | 1.1% |
| gdn_cumdecay_f16 | 2.80 | 8.8% | 8.2 | 2.5% |
| gdn_gated_scan_f16 | 4.55 | 14.4% | 13.4 | 4.3% |
| gdn_cumdecay_bf16 | 2.28 | 7.2% | 10.1 | 2.0% |
| gdn_gated_scan_bf16 | 3.88 | 12.2% | 15.8 | 5.6% |
| gdn2_gated_scan | 3.30 | 10.4% | 32.4 | 1.8% |
| Gated Cumulative Decay | 3.55 | 11.2% | 4.3 | 4.7% |
| Gated Delta-Rule Scan | 5.60 | 17.7% | 6.8 | 2.5% |
| Causal Depthwise Conv1D | 2.48 | 7.8% | 27.7 | 2.1% |
| gdn_cumdecay_f16 | 2.26 | 7.1% | 5.1 | 2.6% |
| gdn_gated_scan_f16 | 4.77 | 15.0% | 6.4 | 2.6% |
| gdn_cumdecay_bf16 | 1.91 | 6.0% | 6.0 | 2.3% |
| gdn_gated_scan_bf16 | 3.96 | 12.5% | 7.7 | 2.2% |
| gdn2_gated_scan | 3.53 | 11.1% | 15.1 | 1.2% |
