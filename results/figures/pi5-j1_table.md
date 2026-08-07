# pi5-j1 — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 667.2 | 760.9 | 14.0% | 2.93 | 0.39 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 1,610.7 | 1,693.0 | 5.1% | 1.84 | 0.33 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 869.6 | 892.5 | 2.6% | 2.37 | 2.41 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 216.3 | 217.9 | 0.7% | 4.51 | 0.61 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 331.3 | 335.7 | 1.3% | 4.47 | 0.79 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 218.7 | 222.7 | 1.8% | 4.71 | 4.80 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 15.8 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 2.93 | 18.5% | 667.2 | 14.0% |
| Gated Delta-Rule Scan | 1.84 | 11.6% | 1,610.7 | 5.1% |
| Causal Depthwise Conv1D | 2.37 | 15.0% | 869.6 | 2.6% |
| Gated Cumulative Decay | 4.51 | 28.5% | 216.3 | 0.7% |
| Gated Delta-Rule Scan | 4.47 | 28.3% | 331.3 | 1.3% |
| Causal Depthwise Conv1D | 4.71 | 29.8% | 218.7 | 1.8% |
