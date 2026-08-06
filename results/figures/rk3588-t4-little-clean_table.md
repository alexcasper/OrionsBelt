# rk3588-t4-little-clean — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 1,321.3 | 1,437.1 | 8.8% | 1.48 | 0.20 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 4,128.5 | 4,956.0 | 20.0% | 0.72 | 0.13 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 2,954.2 | 3,441.6 | 16.5% | 0.70 | 0.71 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 1,150.7 | 1,572.2 | 36.6% | 1.27 | 0.23 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 4,289.5 | 4,370.0 | 1.9% | 0.69 | 0.12 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 1,082.2 | 2,074.8 | 91.7% | 1.35 | 0.24 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 4,285.5 | 4,353.1 | 1.6% | 0.69 | 0.12 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 6,787.5 | 6,916.8 | 1.9% | 0.72 | 0.15 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 585.1 | 628.6 | 7.4% | 1.67 | 0.22 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 1,088.9 | 1,184.8 | 8.8% | 1.36 | 0.24 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 558.9 | 600.9 | 7.5% | 1.84 | 1.88 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 419.4 | 447.7 | 6.7% | 1.75 | 0.31 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 1,025.3 | 1,298.0 | 26.6% | 1.44 | 0.26 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 479.2 | 508.1 | 6.0% | 1.53 | 0.27 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 1,016.2 | 1,072.2 | 5.5% | 1.45 | 0.26 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 1,952.8 | 2,444.6 | 25.2% | 1.26 | 0.27 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 6.7 | 7.6 | 13.0% | 4.55 | 0.61 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 16.6 | 17.2 | 3.5% | 4.59 | 0.49 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 50.2 | 51.3 | 2.3% | 2.74 | 0.65 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 7.6 | 7.9 | 3.8% | 3.02 | 0.54 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 14.3 | 14.9 | 4.1% | 4.27 | 0.57 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 9.6 | 10.5 | 9.1% | 2.38 | 0.43 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 16.3 | 16.9 | 3.6% | 3.74 | 0.50 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 25.7 | 26.3 | 2.3% | 4.16 | 0.64 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 4.4 | 4.4 | 0.0% | 3.49 | 0.47 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 6.4 | 6.7 | 4.5% | 5.94 | 0.64 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 27.4 | 27.7 | 1.1% | 2.50 | 0.60 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 5.0 | 5.0 | 0.0% | 2.31 | 0.41 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 6.4 | 7.3 | 13.6% | 4.76 | 0.64 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 5.8 | 5.8 | 0.0% | 1.96 | 0.35 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 7.6 | 7.6 | 0.0% | 4.02 | 0.54 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 12.0 | 12.3 | 2.4% | 4.47 | 0.69 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 34.0 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 1.48 | 4.4% | 1,321.3 | 8.8% |
| Gated Delta-Rule Scan | 0.72 | 2.1% | 4,128.5 | 20.0% |
| Causal Depthwise Conv1D | 0.70 | 2.1% | 2,954.2 | 16.5% |
| gdn_cumdecay_f16 | 1.27 | 3.7% | 1,150.7 | 36.6% |
| gdn_gated_scan_f16 | 0.69 | 2.0% | 4,289.5 | 1.9% |
| gdn_cumdecay_bf16 | 1.35 | 4.0% | 1,082.2 | 91.7% |
| gdn_gated_scan_bf16 | 0.69 | 2.0% | 4,285.5 | 1.6% |
| gdn2_gated_scan | 0.72 | 2.1% | 6,787.5 | 1.9% |
| Gated Cumulative Decay | 1.67 | 4.9% | 585.1 | 7.4% |
| Gated Delta-Rule Scan | 1.36 | 4.0% | 1,088.9 | 8.8% |
| Causal Depthwise Conv1D | 1.84 | 5.4% | 558.9 | 7.5% |
| gdn_cumdecay_f16 | 1.75 | 5.1% | 419.4 | 6.7% |
| gdn_gated_scan_f16 | 1.44 | 4.2% | 1,025.3 | 26.6% |
| gdn_cumdecay_bf16 | 1.53 | 4.5% | 479.2 | 6.0% |
| gdn_gated_scan_bf16 | 1.45 | 4.3% | 1,016.2 | 5.5% |
| gdn2_gated_scan | 1.26 | 3.7% | 1,952.8 | 25.2% |
| Gated Cumulative Decay | 4.55 | 13.4% | 6.7 | 13.0% |
| Gated Delta-Rule Scan | 4.59 | 13.5% | 16.6 | 3.5% |
| Causal Depthwise Conv1D | 2.74 | 8.1% | 50.2 | 2.3% |
| gdn_cumdecay_f16 | 3.02 | 8.9% | 7.6 | 3.8% |
| gdn_gated_scan_f16 | 4.27 | 12.6% | 14.3 | 4.1% |
| gdn_cumdecay_bf16 | 2.38 | 7.0% | 9.6 | 9.1% |
| gdn_gated_scan_bf16 | 3.74 | 11.0% | 16.3 | 3.6% |
| gdn2_gated_scan | 4.16 | 12.2% | 25.7 | 2.3% |
| Gated Cumulative Decay | 3.49 | 10.3% | 4.4 | 0.0% |
| Gated Delta-Rule Scan | 5.94 | 17.5% | 6.4 | 4.5% |
| Causal Depthwise Conv1D | 2.50 | 7.4% | 27.4 | 1.1% |
| gdn_cumdecay_f16 | 2.31 | 6.8% | 5.0 | 0.0% |
| gdn_gated_scan_f16 | 4.76 | 14.0% | 6.4 | 13.6% |
| gdn_cumdecay_bf16 | 1.96 | 5.8% | 5.8 | 0.0% |
| gdn_gated_scan_bf16 | 4.02 | 11.8% | 7.6 | 0.0% |
| gdn2_gated_scan | 4.47 | 13.1% | 12.0 | 2.4% |
