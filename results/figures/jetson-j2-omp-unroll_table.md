# jetson-j2-omp-unroll — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 526.4 | 613.3 | 16.5% | 3.71 | 0.50 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 1,002.9 | 1,137.8 | 13.5% | 2.95 | 0.52 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 901.3 | 970.2 | 7.6% | 2.29 | 2.33 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 344.8 | 443.6 | 28.6% | 4.25 | 0.76 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 979.6 | 1,099.6 | 12.3% | 3.01 | 0.54 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 366.0 | 396.5 | 8.3% | 4.00 | 0.72 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 976.0 | 1,053.5 | 7.9% | 3.02 | 0.54 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 199.8 | 230.7 | 15.4% | 4.89 | 0.66 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 261.5 | 321.3 | 22.9% | 5.66 | 1.00 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 271.2 | 294.5 | 8.6% | 3.80 | 3.87 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 159.7 | 165.3 | 3.5% | 4.59 | 0.82 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 258.8 | 297.6 | 15.0% | 5.69 | 1.01 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 162.5 | 167.3 | 3.0% | 4.51 | 0.81 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 258.2 | 286.1 | 10.8% | 5.70 | 1.02 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 23.8 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 3.71 | 15.6% | 526.4 | 16.5% |
| Gated Delta-Rule Scan | 2.95 | 12.4% | 1,002.9 | 13.5% |
| Causal Depthwise Conv1D | 2.29 | 9.6% | 901.3 | 7.6% |
| gdn_cumdecay_f16 | 4.25 | 17.9% | 344.8 | 28.6% |
| gdn_gated_scan_f16 | 3.01 | 12.6% | 979.6 | 12.3% |
| gdn_cumdecay_bf16 | 4.00 | 16.8% | 366.0 | 8.3% |
| gdn_gated_scan_bf16 | 3.02 | 12.7% | 976.0 | 7.9% |
| Gated Cumulative Decay | 4.89 | 20.5% | 199.8 | 15.4% |
| Gated Delta-Rule Scan | 5.66 | 23.8% | 261.5 | 22.9% |
| Causal Depthwise Conv1D | 3.80 | 16.0% | 271.2 | 8.6% |
| gdn_cumdecay_f16 | 4.59 | 19.3% | 159.7 | 3.5% |
| gdn_gated_scan_f16 | 5.69 | 23.9% | 258.8 | 15.0% |
| gdn_cumdecay_bf16 | 4.51 | 18.9% | 162.5 | 3.0% |
| gdn_gated_scan_bf16 | 5.70 | 23.9% | 258.2 | 10.8% |
