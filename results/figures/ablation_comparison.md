> ⚠ **Synthetic data.** These numbers are produced by `SyntheticBackend`, a deterministic analytical model — not measured on real hardware. They exist to validate the ablation pipeline end-to-end and to define the table structure for `ob-ami` (master comparison table). Real numbers require wiring optimized GDN kernels into a Qwen3.5 forward pass (`ob-8qt.9`) and running on the target device.

| Config | Quant | Ctx | decode/decode_tokens_per_sec | decode/peak_memory_bytes[kv_cache] | decode/peak_memory_bytes[recurrent_state] | decode/peak_memory_bytes[weights] | prefill/peak_memory_bytes[kv_cache] | prefill/peak_memory_bytes[recurrent_state] | prefill/peak_memory_bytes[weights] | prefill/prefill_tokens_per_sec | prefill/ttft_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu/cpu | fp16 | 4K | 499.8 | 128.6 MiB | 48.0 MiB | 7.8 GiB | 128.0 MiB | 48.0 MiB | 7.8 GiB | 19608 | 209.4ms |
| cpu/cpu | fp16 | 32K | 499.8 | 1.0 GiB | 48.0 MiB | 7.8 GiB | 1.0 GiB | 48.0 MiB | 7.8 GiB | 19608 | 1675ms |
| cpu/cpu | int4_w4a16 | 4K | 499.8 | 128.6 MiB | 48.0 MiB | 7.8 GiB | 128.0 MiB | 48.0 MiB | 7.8 GiB | 19608 | 209.4ms |
| cpu/cpu | int4_w4a16 | 32K | 499.8 | 1.0 GiB | 48.0 MiB | 7.8 GiB | 1.0 GiB | 48.0 MiB | 7.8 GiB | 19608 | 1675ms |
| cpu/gpu_vulkan | fp16 | 4K | 499.8 | 128.6 MiB | 48.0 MiB | 7.8 GiB | 128.0 MiB | 48.0 MiB | 7.8 GiB | 19608 | 209.4ms |
| cpu/gpu_vulkan | fp16 | 32K | 499.8 | 1.0 GiB | 48.0 MiB | 7.8 GiB | 1.0 GiB | 48.0 MiB | 7.8 GiB | 19608 | 1675ms |
| cpu/gpu_vulkan | int4_w4a16 | 4K | 499.8 | 128.6 MiB | 48.0 MiB | 7.8 GiB | 128.0 MiB | 48.0 MiB | 7.8 GiB | 19608 | 209.4ms |
| cpu/gpu_vulkan | int4_w4a16 | 32K | 499.8 | 1.0 GiB | 48.0 MiB | 7.8 GiB | 1.0 GiB | 48.0 MiB | 7.8 GiB | 19608 | 1675ms |
| cpu/npu | fp16 | 4K | 499.8 | 128.6 MiB | 48.0 MiB | 7.8 GiB | 128.0 MiB | 48.0 MiB | 7.8 GiB | 19608 | 209.4ms |
| cpu/npu | fp16 | 32K | 499.8 | 1.0 GiB | 48.0 MiB | 7.8 GiB | 1.0 GiB | 48.0 MiB | 7.8 GiB | 19608 | 1675ms |
| cpu/npu | int4_w4a16 | 4K | 499.8 | 128.6 MiB | 48.0 MiB | 7.8 GiB | 128.0 MiB | 48.0 MiB | 7.8 GiB | 19608 | 209.4ms |
| cpu/npu | int4_w4a16 | 32K | 499.8 | 1.0 GiB | 48.0 MiB | 7.8 GiB | 1.0 GiB | 48.0 MiB | 7.8 GiB | 19608 | 1675ms |
