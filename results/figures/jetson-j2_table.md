# jetson-j2 — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 1,696.3 | 1,911.8 | 12.7% | 1.15 | 0.15 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 4,062.4 | 4,443.4 | 9.4% | 0.73 | 0.13 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 1,987.6 | 2,143.3 | 7.8% | 1.04 | 1.06 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 1,262.2 | 1,395.9 | 10.6% | 1.16 | 0.21 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 4,156.1 | 4,741.1 | 14.1% | 0.71 | 0.13 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 1,061.3 | 1,224.6 | 15.4% | 1.38 | 0.25 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 4,019.4 | 4,430.7 | 10.2% | 0.73 | 0.13 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 492.4 | 577.4 | 17.3% | 1.98 | 0.27 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 889.7 | 1,350.1 | 51.8% | 1.66 | 0.29 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 518.2 | 814.3 | 57.1% | 1.99 | 2.02 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 511.1 | 542.0 | 6.1% | 1.43 | 0.26 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 898.8 | 1,426.4 | 58.7% | 1.64 | 0.29 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 475.2 | 670.1 | 41.0% | 1.54 | 0.28 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 879.3 | 1,089.9 | 24.0% | 1.67 | 0.30 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 6.6 | 6.8 | 3.1% | 4.61 | 0.62 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 10.2 | 10.5 | 2.5% | 7.47 | 0.80 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 29.4 | 30.7 | 4.3% | 4.67 | 1.11 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 10.7 | 11.3 | 5.3% | 2.13 | 0.38 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 13.9 | 14.2 | 2.3% | 4.39 | 0.59 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 7.3 | 7.6 | 3.5% | 3.12 | 0.56 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 11.4 | 11.7 | 3.2% | 5.38 | 0.72 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 4.2 | 4.3 | 3.7% | 3.66 | 0.49 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 4.8 | 5.2 | 7.6% | 7.96 | 0.85 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 12.2 | 12.5 | 2.1% | 5.61 | 1.34 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 5.8 | 6.0 | 3.5% | 1.96 | 0.35 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 7.2 | 7.4 | 2.9% | 4.22 | 0.57 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 4.2 | 4.3 | 2.5% | 2.75 | 0.49 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 6.1 | 6.2 | 1.7% | 5.01 | 0.67 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 23.8 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 1.15 | 4.8% | 1,696.3 | 12.7% |
| Gated Delta-Rule Scan | 0.73 | 3.1% | 4,062.4 | 9.4% |
| Causal Depthwise Conv1D | 1.04 | 4.4% | 1,987.6 | 7.8% |
| gdn_cumdecay_bf16 | 1.16 | 4.9% | 1,262.2 | 10.6% |
| gdn_gated_scan_bf16 | 0.71 | 3.0% | 4,156.1 | 14.1% |
| gdn_cumdecay_f16 | 1.38 | 5.8% | 1,061.3 | 15.4% |
| gdn_gated_scan_f16 | 0.73 | 3.1% | 4,019.4 | 10.2% |
| Gated Cumulative Decay | 1.98 | 8.3% | 492.4 | 17.3% |
| Gated Delta-Rule Scan | 1.66 | 7.0% | 889.7 | 51.8% |
| Causal Depthwise Conv1D | 1.99 | 8.4% | 518.2 | 57.1% |
| gdn_cumdecay_bf16 | 1.43 | 6.0% | 511.1 | 6.1% |
| gdn_gated_scan_bf16 | 1.64 | 6.9% | 898.8 | 58.7% |
| gdn_cumdecay_f16 | 1.54 | 6.5% | 475.2 | 41.0% |
| gdn_gated_scan_f16 | 1.67 | 7.0% | 879.3 | 24.0% |
| Gated Cumulative Decay | 4.61 | 19.4% | 6.6 | 3.1% |
| Gated Delta-Rule Scan | 7.47 | 31.4% | 10.2 | 2.5% |
| Causal Depthwise Conv1D | 4.67 | 19.6% | 29.4 | 4.3% |
| gdn_cumdecay_bf16 | 2.13 | 8.9% | 10.7 | 5.3% |
| gdn_gated_scan_bf16 | 4.39 | 18.4% | 13.9 | 2.3% |
| gdn_cumdecay_f16 | 3.12 | 13.1% | 7.3 | 3.5% |
| gdn_gated_scan_f16 | 5.38 | 22.6% | 11.4 | 3.2% |
| Gated Cumulative Decay | 3.66 | 15.4% | 4.2 | 3.7% |
| Gated Delta-Rule Scan | 7.96 | 33.4% | 4.8 | 7.6% |
| Causal Depthwise Conv1D | 5.61 | 23.6% | 12.2 | 2.1% |
| gdn_cumdecay_bf16 | 1.96 | 8.2% | 5.8 | 3.5% |
| gdn_gated_scan_bf16 | 4.22 | 17.7% | 7.2 | 2.9% |
| gdn_cumdecay_f16 | 2.75 | 11.6% | 4.2 | 2.5% |
| gdn_gated_scan_f16 | 5.01 | 21.1% | 6.1 | 1.7% |
