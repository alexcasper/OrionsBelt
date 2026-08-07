# jetson-j2-omp-full — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 507.9 | 1,036.5 | 104.1% | 3.85 | 0.52 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 1,001.0 | 1,049.6 | 4.8% | 2.96 | 0.52 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 562.7 | 737.7 | 31.1% | 3.66 | 3.73 |
| Qwen3.5-4B | gdn_cumdecay_f16 | neon | 4,096 | 348.2 | 384.2 | 10.3% | 4.21 | 0.75 |
| Qwen3.5-4B | gdn_gated_scan_f16 | neon | 4,096 | 1,001.1 | 1,092.1 | 9.1% | 2.94 | 0.52 |
| Qwen3.5-4B | gdn_cumdecay_bf16 | neon | 4,096 | 352.5 | 439.5 | 24.7% | 4.16 | 0.74 |
| Qwen3.5-4B | gdn_gated_scan_bf16 | neon | 4,096 | 987.9 | 1,042.7 | 5.5% | 2.98 | 0.53 |
| Qwen3.5-4B | gdn2_gated_scan | neon | 4,096 | 1,665.8 | 1,754.7 | 5.3% | 2.95 | 0.63 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 203.7 | 297.7 | 46.2% | 4.80 | 0.64 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 267.0 | 309.4 | 15.9% | 5.54 | 0.98 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 192.3 | 207.9 | 8.1% | 5.36 | 5.45 |
| Qwen3.5-0.8B | gdn_cumdecay_f16 | neon | 2,048 | 171.6 | 175.2 | 2.1% | 4.27 | 0.76 |
| Qwen3.5-0.8B | gdn_gated_scan_f16 | neon | 2,048 | 265.2 | 294.1 | 10.9% | 5.55 | 0.99 |
| Qwen3.5-0.8B | gdn_cumdecay_bf16 | neon | 2,048 | 168.3 | 264.7 | 57.3% | 4.35 | 0.78 |
| Qwen3.5-0.8B | gdn_gated_scan_bf16 | neon | 2,048 | 270.3 | 290.0 | 7.3% | 5.45 | 0.97 |
| Qwen3.5-0.8B | gdn2_gated_scan | neon | 2,048 | 442.1 | 471.2 | 6.6% | 5.56 | 1.19 |
| Qwen3.5-4B_decode | Gated Cumulative Decay | neon | 4,096 | 3.6 | 3.9 | 7.1% | 8.37 | 1.12 |
| Qwen3.5-4B_decode | Gated Delta-Rule Scan | neon | 4,096 | 5.1 | 5.2 | 3.1% | 15.10 | 1.62 |
| Qwen3.5-4B_decode | Causal Depthwise Conv1D | neon | 4,096 | 9.9 | 10.3 | 3.1% | 13.80 | 3.29 |
| Qwen3.5-4B_decode | gdn_cumdecay_f16 | neon | 4,096 | 4.0 | 4.1 | 1.3% | 5.71 | 1.02 |
| Qwen3.5-4B_decode | gdn_gated_scan_f16 | neon | 4,096 | 5.1 | 5.2 | 1.0% | 11.96 | 1.61 |
| Qwen3.5-4B_decode | gdn_cumdecay_bf16 | neon | 4,096 | 4.1 | 4.5 | 10.1% | 5.56 | 1.00 |
| Qwen3.5-4B_decode | gdn_gated_scan_bf16 | neon | 4,096 | 5.3 | 24.2 | 354.9% | 11.49 | 1.54 |
| Qwen3.5-4B_decode | gdn2_gated_scan | neon | 4,096 | 6.3 | 6.5 | 3.3% | 16.95 | 2.60 |
| Qwen3.5-0.8B_decode | Gated Cumulative Decay | neon | 2,048 | 3.2 | 3.2 | 1.7% | 4.80 | 0.64 |
| Qwen3.5-0.8B_decode | Gated Delta-Rule Scan | neon | 2,048 | 4.4 | 36.2 | 727.4% | 8.72 | 0.94 |
| Qwen3.5-0.8B_decode | Causal Depthwise Conv1D | neon | 2,048 | 8.0 | 8.3 | 3.2% | 8.56 | 2.04 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_f16 | neon | 2,048 | 3.1 | 3.3 | 5.0% | 3.66 | 0.66 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_f16 | neon | 2,048 | 4.2 | 4.2 | 1.2% | 7.32 | 0.98 |
| Qwen3.5-0.8B_decode | gdn_cumdecay_bf16 | neon | 2,048 | 3.8 | 3.9 | 4.2% | 3.05 | 0.55 |
| Qwen3.5-0.8B_decode | gdn_gated_scan_bf16 | neon | 2,048 | 4.2 | 4.3 | 1.2% | 7.23 | 0.97 |
| Qwen3.5-0.8B_decode | gdn2_gated_scan | neon | 2,048 | 4.7 | 4.8 | 3.3% | 11.39 | 1.75 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 23.8 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 3.85 | 16.2% | 507.9 | 104.1% |
| Gated Delta-Rule Scan | 2.96 | 12.4% | 1,001.0 | 4.8% |
| Causal Depthwise Conv1D | 3.66 | 15.4% | 562.7 | 31.1% |
| gdn_cumdecay_f16 | 4.21 | 17.7% | 348.2 | 10.3% |
| gdn_gated_scan_f16 | 2.94 | 12.4% | 1,001.1 | 9.1% |
| gdn_cumdecay_bf16 | 4.16 | 17.5% | 352.5 | 24.7% |
| gdn_gated_scan_bf16 | 2.98 | 12.5% | 987.9 | 5.5% |
| gdn2_gated_scan | 2.95 | 12.4% | 1,665.8 | 5.3% |
| Gated Cumulative Decay | 4.80 | 20.2% | 203.7 | 46.2% |
| Gated Delta-Rule Scan | 5.54 | 23.3% | 267.0 | 15.9% |
| Causal Depthwise Conv1D | 5.36 | 22.5% | 192.3 | 8.1% |
| gdn_cumdecay_f16 | 4.27 | 17.9% | 171.6 | 2.1% |
| gdn_gated_scan_f16 | 5.55 | 23.3% | 265.2 | 10.9% |
| gdn_cumdecay_bf16 | 4.35 | 18.3% | 168.3 | 57.3% |
| gdn_gated_scan_bf16 | 5.45 | 22.9% | 270.3 | 7.3% |
| gdn2_gated_scan | 5.56 | 23.4% | 442.1 | 6.6% |
| Gated Cumulative Decay | 8.37 | 35.2% | 3.6 | 7.1% |
| Gated Delta-Rule Scan | 15.10 | 63.4% | 5.1 | 3.1% |
| Causal Depthwise Conv1D | 13.80 | 58.0% | 9.9 | 3.1% |
| gdn_cumdecay_f16 | 5.71 | 24.0% | 4.0 | 1.3% |
| gdn_gated_scan_f16 | 11.96 | 50.3% | 5.1 | 1.0% |
| gdn_cumdecay_bf16 | 5.56 | 23.4% | 4.1 | 10.1% |
| gdn_gated_scan_bf16 | 11.49 | 48.3% | 5.3 | 354.9% |
| gdn2_gated_scan | 16.95 | 71.2% | 6.3 | 3.3% |
| Gated Cumulative Decay | 4.80 | 20.2% | 3.2 | 1.7% |
| Gated Delta-Rule Scan | 8.72 | 36.6% | 4.4 | 727.4% |
| Causal Depthwise Conv1D | 8.56 | 36.0% | 8.0 | 3.2% |
| gdn_cumdecay_f16 | 3.66 | 15.4% | 3.1 | 5.0% |
| gdn_gated_scan_f16 | 7.32 | 30.8% | 4.2 | 1.2% |
| gdn_cumdecay_bf16 | 3.05 | 12.8% | 3.8 | 4.2% |
| gdn_gated_scan_bf16 | 7.23 | 30.4% | 4.2 | 1.2% |
| gdn2_gated_scan | 11.39 | 47.9% | 4.7 | 3.3% |
