# jetson-j1-clean — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 1,228.2 | 1,367.5 | 11.3% | 1.59 | 0.21 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 2,511.3 | 2,581.4 | 2.8% | 1.18 | 0.21 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 1,462.7 | 1,570.8 | 7.4% | 1.41 | 1.43 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 734.9 | 908.6 | 23.6% | 1.99 | 0.36 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 2,496.2 | 2,592.4 | 3.9% | 1.18 | 0.21 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 800.2 | 1,581.4 | 97.6% | 1.83 | 0.33 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 2,501.0 | 2,606.8 | 4.2% | 1.18 | 0.21 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 4,346.9 | 4,414.6 | 1.6% | 1.13 | 0.24 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 271.9 | 322.6 | 18.7% | 3.59 | 0.48 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 542.0 | 689.4 | 27.2% | 2.73 | 0.48 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 357.6 | 372.1 | 4.1% | 2.88 | 2.93 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 305.7 | 347.4 | 13.6% | 2.40 | 0.43 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 582.5 | 855.5 | 46.9% | 2.53 | 0.45 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 368.0 | 393.6 | 7.0% | 1.99 | 0.36 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 573.0 | 715.7 | 24.9% | 2.57 | 0.46 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 1,325.8 | 1,507.3 | 13.7% | 1.85 | 0.40 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 5.9 | 6.0 | 2.7% | 5.19 | 0.70 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 13.0 | 14.9 | 15.3% | 5.88 | 0.63 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 28.6 | 29.3 | 2.4% | 4.80 | 1.15 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 6.5 | 6.7 | 3.2% | 3.54 | 0.63 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 11.5 | 11.8 | 2.7% | 5.33 | 0.71 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 8.3 | 8.5 | 1.9% | 2.75 | 0.49 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 13.0 | 13.3 | 2.4% | 4.71 | 0.63 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 15.8 | 16.1 | 2.3% | 6.77 | 1.04 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 4.2 | 4.2 | 1.3% | 3.66 | 0.49 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 5.6 | 5.7 | 2.8% | 6.84 | 0.73 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 13.6 | 15.7 | 15.3% | 5.05 | 1.21 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 4.4 | 4.6 | 4.8% | 2.62 | 0.47 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 6.6 | 6.7 | 2.4% | 4.65 | 0.62 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 5.4 | 5.5 | 1.9% | 2.13 | 0.38 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 7.2 | 7.4 | 2.9% | 4.22 | 0.57 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 6.6 | 6.8 | 3.1% | 8.07 | 1.24 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 23.8 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 1.59 | 6.7% | 1,228.2 | 11.3% |
| Gated Delta-Rule Scan | 1.18 | 5.0% | 2,511.3 | 2.8% |
| Causal Depthwise Conv1D | 1.41 | 5.9% | 1,462.7 | 7.4% |
| gdn_cumdecay_f16 | 1.99 | 8.4% | 734.9 | 23.6% |
| gdn_gated_scan_f16 | 1.18 | 5.0% | 2,496.2 | 3.9% |
| gdn_cumdecay_bf16 | 1.83 | 7.7% | 800.2 | 97.6% |
| gdn_gated_scan_bf16 | 1.18 | 5.0% | 2,501.0 | 4.2% |
| gdn2_gated_scan | 1.13 | 4.7% | 4,346.9 | 1.6% |
| Gated Cumulative Decay | 3.59 | 15.1% | 271.9 | 18.7% |
| Gated Delta-Rule Scan | 2.73 | 11.5% | 542.0 | 27.2% |
| Causal Depthwise Conv1D | 2.88 | 12.1% | 357.6 | 4.1% |
| gdn_cumdecay_f16 | 2.40 | 10.1% | 305.7 | 13.6% |
| gdn_gated_scan_f16 | 2.53 | 10.6% | 582.5 | 46.9% |
| gdn_cumdecay_bf16 | 1.99 | 8.4% | 368.0 | 7.0% |
| gdn_gated_scan_bf16 | 2.57 | 10.8% | 573.0 | 24.9% |
| gdn2_gated_scan | 1.85 | 7.8% | 1,325.8 | 13.7% |
| Gated Cumulative Decay | 5.19 | 21.8% | 5.9 | 2.7% |
| Gated Delta-Rule Scan | 5.88 | 24.7% | 13.0 | 15.3% |
| Causal Depthwise Conv1D | 4.80 | 20.2% | 28.6 | 2.4% |
| gdn_cumdecay_f16 | 3.54 | 14.9% | 6.5 | 3.2% |
| gdn_gated_scan_f16 | 5.33 | 22.4% | 11.5 | 2.7% |
| gdn_cumdecay_bf16 | 2.75 | 11.6% | 8.3 | 1.9% |
| gdn_gated_scan_bf16 | 4.71 | 19.8% | 13.0 | 2.4% |
| gdn2_gated_scan | 6.77 | 28.4% | 15.8 | 2.3% |
| Gated Cumulative Decay | 3.66 | 15.4% | 4.2 | 1.3% |
| Gated Delta-Rule Scan | 6.84 | 28.7% | 5.6 | 2.8% |
| Causal Depthwise Conv1D | 5.05 | 21.2% | 13.6 | 15.3% |
| gdn_cumdecay_f16 | 2.62 | 11.0% | 4.4 | 4.8% |
| gdn_gated_scan_f16 | 4.65 | 19.5% | 6.6 | 2.4% |
| gdn_cumdecay_bf16 | 2.13 | 8.9% | 5.4 | 1.9% |
| gdn_gated_scan_bf16 | 4.22 | 17.7% | 7.2 | 2.9% |
| gdn2_gated_scan | 8.07 | 33.9% | 6.6 | 3.1% |
