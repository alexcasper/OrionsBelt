# jetson-j2-omp — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 829.4 | 939.9 | 13.3% | 2.35 | 0.32 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 1,501.0 | 1,682.6 | 12.1% | 1.97 | 0.35 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 911.6 | 965.7 | 5.9% | 2.26 | 2.30 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 606.9 | 1,896.6 | 212.5% | 2.41 | 0.43 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 1,548.9 | 3,232.7 | 108.7% | 1.90 | 0.34 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 641.1 | 726.8 | 13.4% | 2.29 | 0.41 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 1,553.1 | 1,680.7 | 8.2% | 1.90 | 0.34 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 335.1 | 380.5 | 13.6% | 2.91 | 0.39 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 454.0 | 527.6 | 16.2% | 3.26 | 0.58 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 321.2 | 373.0 | 16.1% | 3.21 | 3.26 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 327.4 | 346.6 | 5.9% | 2.24 | 0.40 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 449.9 | 492.3 | 9.4% | 3.27 | 0.58 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 329.6 | 342.4 | 3.9% | 2.22 | 0.40 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 451.3 | 1,412.0 | 212.9% | 3.26 | 0.58 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 25.6 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 2.35 | 9.2% | 829.4 | 13.3% |
| Gated Delta-Rule Scan | 1.97 | 7.7% | 1,501.0 | 12.1% |
| Causal Depthwise Conv1D | 2.26 | 8.8% | 911.6 | 5.9% |
| gdn_cumdecay_f16 | 2.41 | 9.4% | 606.9 | 212.5% |
| gdn_gated_scan_f16 | 1.90 | 7.4% | 1,548.9 | 108.7% |
| gdn_cumdecay_bf16 | 2.29 | 8.9% | 641.1 | 13.4% |
| gdn_gated_scan_bf16 | 1.90 | 7.4% | 1,553.1 | 8.2% |
| Gated Cumulative Decay | 2.91 | 11.4% | 335.1 | 13.6% |
| Gated Delta-Rule Scan | 3.26 | 12.7% | 454.0 | 16.2% |
| Causal Depthwise Conv1D | 3.21 | 12.5% | 321.2 | 16.1% |
| gdn_cumdecay_f16 | 2.24 | 8.8% | 327.4 | 5.9% |
| gdn_gated_scan_f16 | 3.27 | 12.8% | 449.9 | 9.4% |
| gdn_cumdecay_bf16 | 2.22 | 8.7% | 329.6 | 3.9% |
| gdn_gated_scan_bf16 | 3.26 | 12.7% | 451.3 | 212.9% |
