# Harness Results — Schema-Conformant

_Source: committed CSVs from bench/harness.py_

## Throughput vs Context Length


**Qwen3.5-0.8B — rk3588-t4 — Decode**

| Context | p50 (tok/s) | p95 (tok/s) | Spread | Repeats |
|---:|---:|---:|---:|---:|
| 32 | 0.7 | 0.7 | 0.0 | 5 |
| 64 | 0.7 | 0.7 | 0.0 | 5 |
| 128 | 0.7 | 0.7 | 0.0 | 5 |
| 256 | 0.7 | 0.7 | 0.0 | 5 |
| 512 | 0.8 | 0.8 | 0.0 | 5 |
| 1,024 | 0.8 | 0.8 | 0.0 | 5 |

**Qwen3.5-0.8B — rk3588-t4 — Prefill**

| Context | p50 (tok/s) | p95 (tok/s) | Spread | Repeats |
|---:|---:|---:|---:|---:|
| 32 | 9.6 | 9.8 | 0.2 | 5 |
| 64 | 15.0 | 15.4 | 0.4 | 5 |
| 128 | 21.1 | 21.3 | 0.2 | 5 |
| 256 | 27.9 | 28.2 | 0.3 | 5 |
| 512 | 10.2 | 10.6 | 0.5 | 5 |
| 1,024 | 11.1 | 11.2 | 0.1 | 5 |

## Memory Decomposition (p50)

| Model | Device | Context | Weights (MiB) | KV Cache (MiB) | Recurrent State (MiB) | Total (MiB) |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | rk3588-t4 | 32 | 2,870.2 | 0.8 | 0.6 | 2,871.5 |
| Qwen3.5-0.8B | rk3588-t4 | 64 | 2,870.2 | 1.5 | 0.6 | 2,872.2 |
| Qwen3.5-0.8B | rk3588-t4 | 128 | 2,870.2 | 3.0 | 0.6 | 2,873.7 |
| Qwen3.5-0.8B | rk3588-t4 | 256 | 2,870.2 | 6.0 | 0.6 | 2,876.7 |
| Qwen3.5-0.8B | rk3588-t4 | 512 | 2,870.2 | 1.5 | 0.6 | 2,872.2 |
| Qwen3.5-0.8B | rk3588-t4 | 1,024 | 2,870.2 | 3.0 | 0.6 | 2,873.7 |
