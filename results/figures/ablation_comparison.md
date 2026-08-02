| Config | Quant | Ctx | decode/decode_tokens_per_sec | decode/peak_memory_bytes[kv_cache] | decode/peak_memory_bytes[recurrent_state] | decode/peak_memory_bytes[weights] | prefill/peak_memory_bytes[kv_cache] | prefill/peak_memory_bytes[recurrent_state] | prefill/peak_memory_bytes[weights] | prefill/prefill_tokens_per_sec | prefill/ttft_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu/cpu | fp16 | 4K | 4542802.4 | 128.3 MiB | 48.0 MiB | 7.5 GiB | 128.0 MiB | 48.0 MiB | 7.5 GiB | 11635043680.2 | 0.5ms |
| cpu/cpu | int4_w4a16 | 4K | 4859782.0 | 128.3 MiB | 48.0 MiB | 7.5 GiB | 128.0 MiB | 48.0 MiB | 7.5 GiB | 15820311191.0 | 0.4ms |
| cpu/gpu_vulkan | fp16 | 4K | 5010980.8 | 128.3 MiB | 48.0 MiB | 7.5 GiB | 128.0 MiB | 48.0 MiB | 7.5 GiB | 16997281202.3 | 0.4ms |
| cpu/gpu_vulkan | int4_w4a16 | 4K | 4958274.2 | 128.3 MiB | 48.0 MiB | 7.5 GiB | 128.0 MiB | 48.0 MiB | 7.5 GiB | 18459796478.9 | 0.4ms |
| cpu/npu | fp16 | 4K | 4909781.0 | 128.3 MiB | 48.0 MiB | 7.5 GiB | 128.0 MiB | 48.0 MiB | 7.5 GiB | 20082404160.3 | 0.4ms |
| cpu/npu | int4_w4a16 | 4K | 5008383.7 | 128.3 MiB | 48.0 MiB | 7.5 GiB | 128.0 MiB | 48.0 MiB | 7.5 GiB | 14733824157.8 | 0.4ms |
