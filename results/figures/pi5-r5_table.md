# pi5-r5 — Microbenchmark Results

_Source: committed CSVs in results/raw/_

| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Gated Cumulative Decay | neon | 4,096 | 522.4 | 544.7 | 4.3% | 3.74 | 0.50 |
| Qwen3.5-4B | Gated Delta-Rule Scan | neon | 4,096 | 2,471.3 | 2,654.5 | 7.4% | 1.20 | 0.21 |
| Qwen3.5-4B | Causal Depthwise Conv1D | neon | 4,096 | 637.4 | 646.4 | 1.4% | 3.23 | 3.29 |
| Qwen3.5-0.8B | Gated Cumulative Decay | neon | 2,048 | 218.4 | 220.6 | 1.0% | 4.47 | 0.60 |
| Qwen3.5-0.8B | Gated Delta-Rule Scan | neon | 2,048 | 333.9 | 355.7 | 6.5% | 4.43 | 0.79 |
| Qwen3.5-0.8B | Causal Depthwise Conv1D | neon | 2,048 | 226.6 | 229.5 | 1.3% | 4.55 | 4.63 |

## Achieved vs Spec Bandwidth

**Device spec bandwidth:** 15.8 GiB/s


| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |
|---|---:|---:|---:|---:|
| Gated Cumulative Decay | 3.74 | 23.7% | 522.4 | 4.3% |
| Gated Delta-Rule Scan | 1.20 | 7.6% | 2,471.3 | 7.4% |
| Causal Depthwise Conv1D | 3.23 | 20.4% | 637.4 | 1.4% |
| Gated Cumulative Decay | 4.47 | 28.3% | 218.4 | 1.0% |
| Gated Delta-Rule Scan | 4.43 | 28.0% | 333.9 | 6.5% |
| Causal Depthwise Conv1D | 4.55 | 28.8% | 226.6 | 1.3% |
