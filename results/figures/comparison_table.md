# Results comparison table

Memory figures regenerable via `scripts/generate_memory_plots.py`. Kernel data from committed CSVs in `results/raw/`.

## Static kernel microbenchmark (rk3588-t4)

| Device | Model | Kernel | Cluster | GiB/s (p50) | GFLOP/s | Spread % |
|---|---|---|---|---:|---:|---:|
| rk3588-t3 | Qwen3.5-4B | `gdn_cumdecay` | A76 (big) | 4.13 | 0.55 | 14.7 |
| rk3588-t3 | Qwen3.5-4B | `gdn_gated_scan` | A76 (big) | 1.96 | 0.35 | 153.1 |
| rk3588-t3 | Qwen3.5-4B | `gdn_causal_dwconv1d` | A76 (big) | 4.02 | 4.09 | 35.6 |
| rk3588-t3 | Qwen3.5-0.8B | `gdn_cumdecay` | A76 (big) | 4.92 | 0.66 | 3.1 |
| rk3588-t3 | Qwen3.5-0.8B | `gdn_gated_scan` | A76 (big) | 4.41 | 0.78 | 5.5 |
| rk3588-t3 | Qwen3.5-0.8B | `gdn_causal_dwconv1d` | A76 (big) | 5.51 | 5.61 | 3.9 |
| rk3588-t3 | Qwen3.5-4B | `gdn_cumdecay` | A55 (little) | 0.87 | 0.12 | 57.9 |
| rk3588-t3 | Qwen3.5-4B | `gdn_gated_scan` | A55 (little) | 0.35 | 0.06 | 29.3 |
| rk3588-t3 | Qwen3.5-4B | `gdn_causal_dwconv1d` | A55 (little) | 0.65 | 0.66 | 38.7 |
| rk3588-t3 | Qwen3.5-0.8B | `gdn_cumdecay` | A55 (little) | 1.15 | 0.15 | 1.6 |
| rk3588-t3 | Qwen3.5-0.8B | `gdn_gated_scan` | A55 (little) | 0.98 | 0.17 | 266.8 |
| rk3588-t3 | Qwen3.5-0.8B | `gdn_causal_dwconv1d` | A55 (little) | 0.94 | 0.95 | 4.5 |
| rk3588-t4 | Qwen3.5-4B | `gdn_cumdecay` | A76 (big) | 24.26 | 3.26 | 10.9 |
| rk3588-t4 | Qwen3.5-4B | `gdn_gated_scan` | A76 (big) | 11.48 | 2.03 | 7.5 |
| rk3588-t4 | Qwen3.5-4B | `gdn_causal_dwconv1d` | A76 (big) | 21.02 | 21.40 | 3.6 |
| rk3588-t4 | Qwen3.5-4B | `gdn_cumdecay_f16` | A76 (big) | 34.63 | 6.20 | 2.1 |
| rk3588-t4 | Qwen3.5-4B | `gdn_gated_scan_f16` | A76 (big) | 11.57 | 2.06 | 9.0 |
| rk3588-t4 | Qwen3.5-4B | `gdn_cumdecay_bf16` | A76 (big) | 25.11 | 4.49 | 1.5 |
| rk3588-t4 | Qwen3.5-4B | `gdn_gated_scan_bf16` | A76 (big) | 11.58 | 2.06 | 7.8 |
| rk3588-t4 | Qwen3.5-4B | `gdn2_gated_scan` | A76 (big) | 10.83 | 2.31 | 4.6 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_cumdecay` | A76 (big) | 29.37 | 3.94 | 1.8 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_gated_scan` | A76 (big) | 11.88 | 2.10 | 4.4 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_causal_dwconv1d` | A76 (big) | 21.80 | 22.19 | 6.2 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_cumdecay_f16` | A76 (big) | 41.16 | 7.37 | 1.6 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_gated_scan_f16` | A76 (big) | 10.88 | 1.94 | 5.4 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_cumdecay_bf16` | A76 (big) | 26.16 | 4.68 | 1.0 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_gated_scan_bf16` | A76 (big) | 10.70 | 1.90 | 4.4 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn2_gated_scan` | A76 (big) | 10.78 | 2.30 | 3.1 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_cumdecay` | A76 (big) | 20.92 | 2.81 | 20.0 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_gated_scan` | A76 (big) | 37.36 | 4.01 | 0.0 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_causal_dwconv1d` | A76 (big) | 42.81 | 10.21 | 0.0 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_cumdecay_f16` | A76 (big) | 13.08 | 2.34 | 0.1 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_gated_scan_f16` | A76 (big) | 34.88 | 4.68 | 16.7 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_cumdecay_bf16` | A76 (big) | 13.07 | 2.34 | 16.6 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_gated_scan_bf16` | A76 (big) | 29.89 | 4.01 | 14.3 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn2_gated_scan` | A76 (big) | 45.78 | 7.02 | 0.0 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_cumdecay` | A76 (big) | 10.46 | 1.40 | 0.1 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_gated_scan` | A76 (big) | 26.15 | 2.81 | 20.0 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_causal_dwconv1d` | A76 (big) | 29.43 | 7.02 | 12.5 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_cumdecay_f16` | A76 (big) | 7.84 | 1.40 | 20.0 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_gated_scan_f16` | A76 (big) | 20.91 | 2.81 | 20.0 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_cumdecay_bf16` | A76 (big) | 7.84 | 1.40 | 20.0 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_gated_scan_bf16` | A76 (big) | 17.44 | 2.34 | 16.7 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn2_gated_scan` | A76 (big) | 36.61 | 5.62 | 20.0 |
| rk3588-t4 | Qwen3.5-4B | `gdn_cumdecay` | A76 (big) | 7.21 | 0.97 | 7.9 |
| rk3588-t4 | Qwen3.5-4B | `gdn_gated_scan` | A76 (big) | 5.29 | 0.94 | 11.2 |
| rk3588-t4 | Qwen3.5-4B | `gdn_causal_dwconv1d` | A76 (big) | 6.35 | 6.47 | 8.7 |
| rk3588-t4 | Qwen3.5-4B | `gdn_cumdecay_f16` | A76 (big) | 8.54 | 1.53 | 2.7 |
| rk3588-t4 | Qwen3.5-4B | `gdn_gated_scan_f16` | A76 (big) | 5.27 | 0.94 | 23.3 |
| rk3588-t4 | Qwen3.5-4B | `gdn_cumdecay_bf16` | A76 (big) | 5.93 | 1.06 | 27.9 |
| rk3588-t4 | Qwen3.5-4B | `gdn_gated_scan_bf16` | A76 (big) | 5.37 | 0.96 | 20.3 |
| rk3588-t4 | Qwen3.5-4B | `gdn2_gated_scan` | A76 (big) | 4.16 | 0.89 | 11.7 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_cumdecay` | A76 (big) | 7.88 | 1.06 | 2.6 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_gated_scan` | A76 (big) | 7.11 | 1.26 | 6.3 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_causal_dwconv1d` | A76 (big) | 7.85 | 7.99 | 7.1 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_cumdecay_f16` | A76 (big) | 9.37 | 1.68 | 2.6 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_gated_scan_f16` | A76 (big) | 7.02 | 1.25 | 5.8 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_cumdecay_bf16` | A76 (big) | 5.66 | 1.01 | 1.1 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_gated_scan_bf16` | A76 (big) | 6.94 | 1.24 | 6.2 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn2_gated_scan` | A76 (big) | 7.58 | 1.62 | 5.0 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_cumdecay` | A76 (big) | 17.44 | 2.34 | 16.7 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_gated_scan` | A76 (big) | 32.70 | 3.51 | 0.0 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_causal_dwconv1d` | A76 (big) | 19.62 | 4.68 | 4.2 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_cumdecay_f16` | A76 (big) | 11.21 | 2.01 | 0.0 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_gated_scan_f16` | A76 (big) | 20.93 | 2.81 | 0.0 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_cumdecay_bf16` | A76 (big) | 8.72 | 1.56 | 11.1 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_gated_scan_bf16` | A76 (big) | 14.95 | 2.01 | 0.0 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn2_gated_scan` | A76 (big) | 36.62 | 5.62 | 10.0 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_cumdecay` | A76 (big) | 13.08 | 1.76 | 25.0 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_gated_scan` | A76 (big) | 26.15 | 2.81 | 19.9 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_causal_dwconv1d` | A76 (big) | 23.54 | 5.62 | 10.0 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_cumdecay_f16` | A76 (big) | 7.85 | 1.40 | 0.1 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_gated_scan_f16` | A76 (big) | 17.44 | 2.34 | 0.1 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_cumdecay_bf16` | A76 (big) | 6.54 | 1.17 | 0.1 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_gated_scan_bf16` | A76 (big) | 13.08 | 1.76 | 0.0 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn2_gated_scan` | A76 (big) | 30.52 | 4.68 | 0.1 |
| rk3588-t4 | Qwen3.5-4B | `gdn_cumdecay` | A55 (little) | 5.87 | 0.79 | 18.9 |
| rk3588-t4 | Qwen3.5-4B | `gdn_gated_scan` | A55 (little) | 3.91 | 0.69 | 35.2 |
| rk3588-t4 | Qwen3.5-4B | `gdn_causal_dwconv1d` | A55 (little) | 5.30 | 5.39 | 7.7 |
| rk3588-t4 | Qwen3.5-4B | `gdn_cumdecay_f16` | A55 (little) | 6.61 | 1.18 | 10.1 |
| rk3588-t4 | Qwen3.5-4B | `gdn_gated_scan_f16` | A55 (little) | 3.85 | 0.69 | 23.1 |
| rk3588-t4 | Qwen3.5-4B | `gdn_cumdecay_bf16` | A55 (little) | 5.89 | 1.05 | 9.3 |
| rk3588-t4 | Qwen3.5-4B | `gdn_gated_scan_bf16` | A55 (little) | 3.90 | 0.70 | 27.6 |
| rk3588-t4 | Qwen3.5-4B | `gdn2_gated_scan` | A55 (little) | 2.54 | 0.54 | 11.7 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_cumdecay` | A55 (little) | 6.28 | 0.84 | 0.9 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_gated_scan` | A55 (little) | 5.78 | 1.02 | 7.9 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_causal_dwconv1d` | A55 (little) | 5.90 | 6.01 | 8.4 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_cumdecay_f16` | A55 (little) | 7.17 | 1.28 | 2.9 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_gated_scan_f16` | A55 (little) | 5.72 | 1.02 | 10.8 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_cumdecay_bf16` | A55 (little) | 6.32 | 1.13 | 1.8 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_gated_scan_bf16` | A55 (little) | 5.72 | 1.02 | 6.8 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn2_gated_scan` | A55 (little) | 6.01 | 1.28 | 7.6 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_cumdecay` | A55 (little) | 7.47 | 1.00 | 0.0 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_gated_scan` | A55 (little) | 15.38 | 1.65 | 5.9 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_causal_dwconv1d` | A55 (little) | 12.39 | 2.96 | 5.3 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_cumdecay_f16` | A55 (little) | 5.23 | 0.94 | 13.3 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_gated_scan_f16` | A55 (little) | 10.46 | 1.40 | 5.0 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_cumdecay_bf16` | A55 (little) | 4.36 | 0.78 | 5.5 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_gated_scan_bf16` | A55 (little) | 9.96 | 1.34 | 4.8 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn2_gated_scan` | A55 (little) | 12.21 | 1.87 | 3.3 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_cumdecay` | A55 (little) | 3.74 | 0.50 | 0.0 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_gated_scan` | A55 (little) | 9.34 | 1.00 | 0.0 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_causal_dwconv1d` | A55 (little) | 6.73 | 1.60 | 2.9 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_cumdecay_f16` | A55 (little) | 2.80 | 0.50 | 7.1 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_gated_scan_f16` | A55 (little) | 6.54 | 0.88 | 6.3 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_cumdecay_bf16` | A55 (little) | 2.62 | 0.47 | 0.0 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_gated_scan_bf16` | A55 (little) | 6.15 | 0.83 | 0.0 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn2_gated_scan` | A55 (little) | 8.72 | 1.34 | 4.8 |
| rk3588-t4 | Qwen3.5-4B | `gdn_cumdecay` | A55 (little) | 1.37 | 0.18 | 24.2 |
| rk3588-t4 | Qwen3.5-4B | `gdn_gated_scan` | A55 (little) | 0.76 | 0.14 | 41.3 |
| rk3588-t4 | Qwen3.5-4B | `gdn_causal_dwconv1d` | A55 (little) | 1.24 | 1.26 | 49.2 |
| rk3588-t4 | Qwen3.5-4B | `gdn_cumdecay_f16` | A55 (little) | 1.57 | 0.28 | 81.4 |
| rk3588-t4 | Qwen3.5-4B | `gdn_gated_scan_f16` | A55 (little) | 0.81 | 0.14 | 32.4 |
| rk3588-t4 | Qwen3.5-4B | `gdn_cumdecay_bf16` | A55 (little) | 1.51 | 0.27 | 4.7 |
| rk3588-t4 | Qwen3.5-4B | `gdn_gated_scan_bf16` | A55 (little) | 0.80 | 0.14 | 19.8 |
| rk3588-t4 | Qwen3.5-4B | `gdn2_gated_scan` | A55 (little) | 0.73 | 0.16 | 4.5 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_cumdecay` | A55 (little) | 1.57 | 0.21 | 70.3 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_gated_scan` | A55 (little) | 1.45 | 0.26 | 9.6 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_causal_dwconv1d` | A55 (little) | 1.54 | 1.57 | 4.5 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_cumdecay_f16` | A55 (little) | 1.85 | 0.33 | 7.6 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_gated_scan_f16` | A55 (little) | 1.47 | 0.26 | 7.6 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_cumdecay_bf16` | A55 (little) | 1.61 | 0.29 | 44.2 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn_gated_scan_bf16` | A55 (little) | 1.45 | 0.26 | 39.2 |
| rk3588-t4 | Qwen3.5-0.8B | `gdn2_gated_scan` | A55 (little) | 1.44 | 0.31 | 22.6 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_cumdecay` | A55 (little) | 4.36 | 0.59 | 4.2 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_gated_scan` | A55 (little) | 4.76 | 0.51 | 1.8 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_causal_dwconv1d` | A55 (little) | 2.66 | 0.63 | 1.1 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_cumdecay_f16` | A55 (little) | 3.02 | 0.54 | 11.5 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_gated_scan_f16` | A55 (little) | 4.36 | 0.59 | 2.1 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_cumdecay_bf16` | A55 (little) | 2.38 | 0.43 | 0.0 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn_gated_scan_bf16` | A55 (little) | 3.74 | 0.50 | 1.8 |
| rk3588-t4 | Qwen3.5-4B_decode | `gdn2_gated_scan` | A55 (little) | 3.90 | 0.60 | 2.1 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_cumdecay` | A55 (little) | 3.49 | 0.47 | 0.0 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_gated_scan` | A55 (little) | 5.94 | 0.64 | 4.5 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_causal_dwconv1d` | A55 (little) | 2.45 | 0.59 | 9.4 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_cumdecay_f16` | A55 (little) | 2.31 | 0.41 | 0.0 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_gated_scan_f16` | A55 (little) | 4.76 | 0.64 | 4.5 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_cumdecay_bf16` | A55 (little) | 1.96 | 0.35 | 0.0 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn_gated_scan_bf16` | A55 (little) | 3.87 | 0.52 | 3.7 |
| rk3588-t4 | Qwen3.5-0.8B_decode | `gdn2_gated_scan` | A55 (little) | 4.58 | 0.70 | 2.5 |

## Memory decomposition — Qwen3.5-4B (analytical, from verified config)

_Regenerable: `python3 scripts/generate_memory_plots.py`. See [`memory_comparison.md`](memory_comparison.md) for full table including 0.8B._

| Context | Weights (GiB) | KV cache (GiB) | Recurrent state (MiB) | Total (GiB) | If all-attn (GiB) | Savings |
|---:|---:|---:|---:|---:|---:|---:|
| 4K | 7.83 | 0.12 | 51 | 8.01 | 8.33 | 0.33 GiB |
| 32K | 7.83 | 1.00 | 51 | 8.88 | 11.83 | 2.95 GiB |
| 128K | 7.83 | 4.00 | 51 | 11.88 | 23.83 | 11.95 GiB |
| 262K | 7.83 | 8.00 | 51 | 15.88 | 39.83 | 23.95 GiB |

## Decode bandwidth model — Qwen3.5-4B at 100 GB/s (O6 stretch target)

| Quant | Weight traffic/token | State traffic/token | Total | Ceiling tok/s |
|---|---:|---:|---:|---:|
| fp16 | 7.83 GiB | 51 MiB | 7.88 GiB | ≈12 |
| INT8 | 3.92 GiB | 51 MiB | 3.96 GiB | ≈23 |
| INT4 (W4A16) | 1.96 GiB | 51 MiB | 2.01 GiB | ≈46 |
