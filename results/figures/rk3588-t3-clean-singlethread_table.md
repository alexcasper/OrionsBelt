# rk3588-t3-clean-singlethread — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 278.6 | 297.2 | 6.7% | 7.01 | 0.94 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 962.9 | 1,034.6 | 7.5% | 3.07 | 0.54 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 342.4 | 357.0 | 4.3% | 6.02 | 6.12 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 168.9 | 225.5 | 33.5% | 8.67 | 1.55 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 947.1 | 1,049.8 | 10.8% | 3.11 | 0.55 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 264.3 | 385.0 | 45.7% | 5.54 | 0.99 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 957.0 | 1,005.4 | 5.1% | 3.08 | 0.55 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 2,843.4 | 3,006.4 | 5.7% | 1.73 | 0.37 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 128.6 | 141.8 | 10.2% | 7.59 | 1.02 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 181.1 | 192.2 | 6.1% | 8.17 | 1.45 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 136.5 | 142.1 | 4.1% | 7.55 | 7.68 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 82.3 | 86.0 | 4.6% | 8.90 | 1.59 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 180.3 | 187.6 | 4.0% | 8.17 | 1.45 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 123.7 | 125.7 | 1.7% | 5.92 | 1.06 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 180.8 | 190.5 | 5.3% | 8.14 | 1.45 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 336.0 | 346.8 | 3.2% | 7.31 | 1.56 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 1.5 | 1.8 | 20.0% | 20.92 | 2.81 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 2.0 | 2.3 | 14.3% | 37.36 | 4.01 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 6.1 | 6.4 | 4.8% | 22.42 | 5.35 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 1.8 | 1.8 | 0.1% | 13.08 | 2.34 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 2.6 | 2.9 | 11.1% | 23.25 | 3.12 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 2.9 | 2.9 | 0.0% | 7.85 | 1.40 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 3.8 | 4.1 | 7.7% | 16.10 | 2.16 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 3.2 | 3.2 | 0.0% | 33.30 | 5.11 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 0.9 | 1.2 | 33.4% | 17.44 | 2.34 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 1.2 | 1.5 | 25.0% | 32.69 | 3.51 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 2.9 | 2.9 | 0.0% | 23.54 | 5.62 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 1.2 | 1.2 | 0.1% | 9.82 | 1.76 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 1.5 | 1.8 | 20.0% | 20.92 | 2.81 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 1.8 | 1.8 | 0.1% | 6.54 | 1.17 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 2.0 | 2.3 | 14.3% | 14.95 | 2.01 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 1.8 | 1.8 | 0.1% | 30.52 | 4.68 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 31.7 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 7.01 | 22.1% | 278.6 | 6.7% |
| Gated Delta-Rule Scan | 3.07 | 9.7% | 962.9 | 7.5% |
| Causal Depthwise Conv1D | 6.02 | 19.0% | 342.4 | 4.3% |
| gdn_cumdecay_f16 | 8.67 | 27.4% | 168.9 | 33.5% |
| gdn_gated_scan_f16 | 3.11 | 9.8% | 947.1 | 10.8% |
| gdn_cumdecay_bf16 | 5.54 | 17.5% | 264.3 | 45.7% |
| gdn_gated_scan_bf16 | 3.08 | 9.7% | 957.0 | 5.1% |
| gdn2_gated_scan | 1.73 | 5.5% | 2,843.4 | 5.7% |
| Gated Cumulative Decay | 7.59 | 23.9% | 128.6 | 10.2% |
| Gated Delta-Rule Scan | 8.17 | 25.8% | 181.1 | 6.1% |
| Causal Depthwise Conv1D | 7.55 | 23.8% | 136.5 | 4.1% |
| gdn_cumdecay_f16 | 8.90 | 28.1% | 82.3 | 4.6% |
| gdn_gated_scan_f16 | 8.17 | 25.8% | 180.3 | 4.0% |
| gdn_cumdecay_bf16 | 5.92 | 18.7% | 123.7 | 1.7% |
| gdn_gated_scan_bf16 | 8.14 | 25.7% | 180.8 | 5.3% |
| gdn2_gated_scan | 7.31 | 23.1% | 336.0 | 3.2% |
| Gated Cumulative Decay | 20.92 | 66.0% | 1.5 | 20.0% |
| Gated Delta-Rule Scan | 37.36 | 117.9% | 2.0 | 14.3% |
| Causal Depthwise Conv1D | 22.42 | 70.7% | 6.1 | 4.8% |
| gdn_cumdecay_f16 | 13.08 | 41.3% | 1.8 | 0.1% |
| gdn_gated_scan_f16 | 23.25 | 73.3% | 2.6 | 11.1% |
| gdn_cumdecay_bf16 | 7.85 | 24.8% | 2.9 | 0.0% |
| gdn_gated_scan_bf16 | 16.10 | 50.8% | 3.8 | 7.7% |
| gdn2_gated_scan | 33.30 | 105.0% | 3.2 | 0.0% |
| Gated Cumulative Decay | 17.44 | 55.0% | 0.9 | 33.4% |
| Gated Delta-Rule Scan | 32.69 | 103.1% | 1.2 | 25.0% |
| Causal Depthwise Conv1D | 23.54 | 74.3% | 2.9 | 0.0% |
| gdn_cumdecay_f16 | 9.82 | 31.0% | 1.2 | 0.1% |
| gdn_gated_scan_f16 | 20.92 | 66.0% | 1.5 | 20.0% |
| gdn_cumdecay_bf16 | 6.54 | 20.6% | 1.8 | 0.1% |
| gdn_gated_scan_bf16 | 14.95 | 47.2% | 2.0 | 14.3% |
| gdn2_gated_scan | 30.52 | 96.3% | 1.8 | 0.1% |
