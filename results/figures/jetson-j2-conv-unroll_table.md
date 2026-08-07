# jetson-j2-conv-unroll — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 507.3 | 576.0 | 13.5% | 3.85 | 0.52 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 1,017.5 | 1,829.5 | 79.8% | 2.91 | 0.52 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 577.7 | 889.6 | 54.0% | 3.57 | 3.63 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 360.2 | 1,433.1 | 297.8% | 4.07 | 0.73 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 979.0 | 1,044.6 | 6.7% | 3.01 | 0.54 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 378.6 | 471.6 | 24.6% | 3.87 | 0.69 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 1,004.5 | 1,063.3 | 5.9% | 2.93 | 0.52 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 210.2 | 1,624.1 | 672.8% | 4.65 | 0.62 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 280.2 | 328.0 | 17.1% | 5.28 | 0.94 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 191.5 | 197.8 | 3.3% | 5.38 | 5.48 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 168.4 | 171.6 | 1.9% | 4.35 | 0.78 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 283.2 | 326.9 | 15.4% | 5.20 | 0.93 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 168.1 | 184.2 | 9.6% | 4.36 | 0.78 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 284.1 | 346.7 | 22.0% | 5.18 | 0.92 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 23.8 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 3.85 | 16.2% | 507.3 | 13.5% |
| Gated Delta-Rule Scan | 2.91 | 12.2% | 1,017.5 | 79.8% |
| Causal Depthwise Conv1D | 3.57 | 15.0% | 577.7 | 54.0% |
| gdn_cumdecay_f16 | 4.07 | 17.1% | 360.2 | 297.8% |
| gdn_gated_scan_f16 | 3.01 | 12.6% | 979.0 | 6.7% |
| gdn_cumdecay_bf16 | 3.87 | 16.3% | 378.6 | 24.6% |
| gdn_gated_scan_bf16 | 2.93 | 12.3% | 1,004.5 | 5.9% |
| Gated Cumulative Decay | 4.65 | 19.5% | 210.2 | 672.8% |
| Gated Delta-Rule Scan | 5.28 | 22.2% | 280.2 | 17.1% |
| Causal Depthwise Conv1D | 5.38 | 22.6% | 191.5 | 3.3% |
| gdn_cumdecay_f16 | 4.35 | 18.3% | 168.4 | 1.9% |
| gdn_gated_scan_f16 | 5.20 | 21.8% | 283.2 | 15.4% |
| gdn_cumdecay_bf16 | 4.36 | 18.3% | 168.1 | 9.6% |
| gdn_gated_scan_bf16 | 5.18 | 21.8% | 284.1 | 22.0% |
