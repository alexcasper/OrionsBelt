# GDN layer structure audit — Qwen3.5-4B / Qwen3.5-0.8B

**Bead:** `ob-37v` · **Date:** 2026-08-02 · **Source:** HuggingFace `config.json` + transformers `modeling_qwen3_5.py` (main branch, commit retrieved 2026-08-02)

This document confirms the GDN layer structure by reading the actual modeling code, not secondary sources. Every figure below is traceable to either the checkpoint's `config.json` or the `Qwen3_5GatedDeltaNet` class in `transformers`. It supersedes any quoted figure from a survey, paper, or earlier document, per `PLAN.md` §3.

---

## 1. Layer composition and placement

Both checkpoints use `full_attention_interval: 4` — every 4th layer (0-indexed: layers 3, 7, 11, …) is full attention; the rest are Gated DeltaNet (linear attention) layers.

| Checkpoint | Total layers | GDN layers | Full-attn layers | Ratio |
|---|---:|---:|---:|---|
| Qwen3.5-4B | 32 | 24 | 8 | 3:1 |
| Qwen3.5-0.8B | 24 | 18 | 6 | 3:1 |

The `layer_types` array in `config.json` is an explicit per-layer list, read directly — not inferred from the interval. The full-attention layers are at indices {3, 7, 11, 15, 19, 23, 27, 31} (4B) and {3, 7, 11, 15, 19, 23} (0.8B).

**This is the ground truth** for `PLAN.md` §3.1's "16 engine boundary crossings per token" — 8 full-attention layers alternate with GDN blocks, so a CPU/accelerator split crosses 16 times (8 out, 8 back) per token during decode.

---

## 2. GDN layer dimensions

All values read from `config.json` `text_config`:

| Parameter | 4B | 0.8B | Config field |
|---|---:|---:|---|
| Hidden size | 2560 | 1024 | `hidden_size` |
| Key heads | 16 | 16 | `linear_num_key_heads` |
| Value heads | 32 | 16 | `linear_num_value_heads` |
| Key head dim | 128 | 128 | `linear_key_head_dim` |
| Value head dim | 128 | 128 | `linear_value_head_dim` |
| Conv kernel width | 4 | 4 | `linear_conv_kernel_dim` |
| Head ratio (V:K) | 2:1 | 1:1 | computed |

**Key consequence of the V:K ratio:** Q and K have `num_k_heads` heads, but V has `num_v_heads`. When V:K > 1 (4B: 32:16), Q and K are `repeat_interleave`d by the ratio so every value head gets its own query/key pair. At 0.8B (1:1), no replication is needed.

---

## 3. Recurrent state shape — **confirmed**

**Source:** `torch_chunk_gated_delta_rule` and `torch_recurrent_gated_delta_rule` in `modeling_qwen3_5.py`.

The recurrent state is allocated as:

```python
torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, ...)
```

This is `(batch, n_value_heads, d_k, d_v)` — confirming the assumption in ADR 0003. The state is a rank-2 matrix per head, not a vector.

| Checkpoint | State shape per layer | Elements per layer | Bytes (fp32) | MiB |
|---|---|---:|---:|---:|
| 4B | (1, 32, 128, 128) | 524,288 | 2,097,152 | 2.0 |
| 0.8B | (1, 16, 128, 128) | 262,144 | 1,048,576 | 1.0 |

**Total recurrent state across all GDN layers:**

| Checkpoint | GDN layers | Total state (fp32) | Flat vs context |
|---|---:|---:|---|
| 4B | 24 | 48 MiB | **O(1), independent of context length** |
| 0.8B | 18 | 18 MiB | **O(1), independent of context length** |

The state dtype is fp32 (`mamba_ssm_dtype: "float32"` in config, and the torch implementations cast all computation to `torch.float32`).

---

## 4. Conv1D state (decode only)

The causal depthwise Conv1D maintains a sliding-window state during decode:

```
conv_state shape = (batch, conv_dim, kernel_size)
conv_dim = key_dim * 2 + value_dim  # = num_k_heads * d_k * 2 + num_v_heads * d_v
```

| Checkpoint | conv_dim | kernel_size | Conv state (fp32) per layer | Across GDN layers |
|---|---:|---:|---:|---:|
| 4B | 12,288 | 4 | 49,152 floats = 192 KiB | 4.5 MiB |
| 0.8B | 6,144 | 4 | 24,576 floats = 96 KiB | 1.7 MiB |

This is a small, fixed cost per layer — not context-dependent. It adds to the recurrent state in the total "GDN decode footprint" but is O(1) like the recurrent state.

---

## 5. The delta-rule recurrence

**Source:** `torch_recurrent_gated_delta_rule` — the per-step decode path.

Per token, per value head:

```python
# 1. Decay: apply gated decay factor
S = S * exp(g_t)  # S: (d_k, d_v), g_t: scalar

# 2. Retrieve: what does the current key "remember"?
kv_mem = sum(S * k_t, dim=-2)  # (d_v,) — state projected by key

# 3. Delta: how far is the new value from what's remembered?
delta = (v_t - kv_mem) * beta_t  # beta_t = sigmoid(b_t)

# 4. Update: write the correction
S = S + outer(k_t, delta)  # rank-1 update

# 5. Output: retrieve via query
output = sum(S * q_t, dim=-2)  # (d_v,)
```

Where:
- `g_t = -exp(A_log) * softplus(a_t + dt_bias)` — the **data-dependent decay gate**. Always negative (A_log is initialized as `log(uniform(0, 16))`), so `exp(g_t) ∈ (0, 1)` — a proper multiplicative decay. Input-dependent via `a_t` (projected from hidden states).
- `beta_t = sigmoid(b_t)` — the **write gate**. Controls how aggressively the delta correction is applied. Also input-dependent via `b_t`.
- `A_log`, `dt_bias` are learnable per-head parameters (shape `num_v_heads`).
- Q and K are L2-normalized before use (`use_qk_l2norm_in_kernel=True`).
- Scale: `1/sqrt(d_k)` applied to Q.

**This is the sequential dependency chain** that makes the kernel latency-bound (see `FINDINGS.md` §6): step 2 depends on step 1, step 3 on step 2, step 4 on step 3, step 5 on step 4. Each token's output depends on all previous tokens through `S`.

---

## 6. Chunkwise prefill

**Source:** `torch_chunk_gated_delta_rule` — the prefill path.

During prefill, the sequence is split into chunks of `chunk_size=64` tokens:

1. **Within-chunk:** attention matrix computed via gated delta-rule (Q·K^T with decay mask + intra-chunk correction)
2. **Cross-chunk:** sequential scan over chunks, each carrying the recurrent state forward
3. **Decay:** cumulative sum of `g` within each chunk → exponential decay mask

The chunkwise path processes 64 tokens in parallel within each chunk, then chains chunks sequentially. This is where the `fla` library's optimized `chunk_gated_delta_rule` kernel provides the 1.38–1.49× speedup over the naive path (`PLAN.md` §2.4).

**Chunk size 64** is hardcoded as the default in the torch fallback. The `fla` library kernel may use a different internal chunk size.

---

## 7. Decay gate mechanics

The decay gate `g` is the most architecturally significant component — it controls how fast old information is forgotten:

```python
# Per-head learnable parameters
A_log = nn.Parameter(torch.log(torch.empty(num_v_heads).uniform_(0, 16)))
dt_bias = nn.Parameter(torch.ones(num_v_heads))

# Per-token, per-head computation
g = -torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias)
```

- `A_log`: initialized so `exp(A_log) ∈ [1, 16]` — the decay rate range.
- `a`: input-dependent, projected from hidden states via `in_proj_a` (hidden_size → num_v_heads).
- `dt_bias`: learnable bias, initialized to 1.
- `softplus` ensures the multiplier is always positive, so `g` is always negative.
- `exp(g)` is the actual decay factor applied to the state — always in (0, 1).

For the chunkwise path, `g` is cumulatively summed (`g.cumsum(dim=-1)`) to produce position-dependent cumulative decay factors.

---

## 8. Projections and parameter count

Per GDN layer, the projections are:

| Projection | Shape (4B) | Shape (0.8B) | Purpose |
|---|---|---|---|
| `in_proj_qkv` | 2560 → 12,288 | 1024 → 6,144 | Q, K, V combined |
| `in_proj_z` | 2560 → 4,096 | 1024 → 2,048 | Gate for RMSNorm |
| `in_proj_b` | 2560 → 32 | 1024 → 16 | Write gate (beta) |
| `in_proj_a` | 2560 → 32 | 1024 → 16 | Decay gate (alpha) |
| `out_proj` | 4,096 → 2560 | 2,048 → 1024 | Output projection |
| `conv1d` weight | 12,288 × 1 × 4 | 6,144 × 1 × 4 | Depthwise Conv1D |
| `A_log` | 32 | 16 | Learnable decay |
| `dt_bias` | 32 | 16 | Learnable bias |
| `norm` weight | 128 | 128 | RMSNorm scale |

All projections use `bias=False` (except `dt_bias` which is a standalone parameter, not a linear bias).

---

## 9. Full-attention layer (for contrast)

| Parameter | 4B | 0.8B | Config field |
|---|---:|---:|---|
| Attention heads | 16 | 8 | `num_attention_heads` |
| KV heads (GQA) | 4 | 2 | `num_key_value_heads` |
| Head dim | 256 | 256 | `head_dim` |
| KV cache per token | 4×2×256×2B = 4 KiB | 2×2×256×2B = 2 KiB | (per layer, fp16) |

The KV cache grows linearly: at 262K context, 4B's 8 full-attn layers hold 8 × 262,144 × 4 KiB ≈ 8 GiB — versus 48 MiB of GDN state that stays flat. This is the project's central memory claim (`ADR 0003`).

---

## 10. Summary of confirmed facts

| Fact | Status | Source |
|---|---|---|
| 3:1 GDN:full-attn ratio | ✓ confirmed | `layer_types` array in `config.json` |
| State shape `(n_v_heads, d_k, d_v)` | ✓ confirmed | `torch.zeros(batch, num_heads, k_head_dim, v_head_dim)` in modeling code |
| State is fp32 | ✓ confirmed | `mamba_ssm_dtype: "float32"` + `.to(torch.float32)` in all computation |
| 4B state = 524,288 floats/layer | ✓ confirmed | 32 × 128 × 128 |
| Total 4B state = 48 MiB | ✓ confirmed | 24 layers × 2 MiB |
| State is O(1) vs context | ✓ confirmed by construction | state shape has no sequence dimension |
| Chunk size = 64 | ✓ confirmed | default parameter in `torch_chunk_gated_delta_rule` |
| Conv kernel width = 4 | ✓ confirmed | `linear_conv_kernel_dim: 4` |
| Decay is data-dependent | ✓ confirmed | `g = -exp(A_log) * softplus(a + dt_bias)`, `a` is input-projected |
| Write gate is data-dependent | ✓ confirmed | `beta = sigmoid(b)`, `b` is input-projected |
| Q/K L2-normalized | ✓ confirmed | `use_qk_l2norm_in_kernel=True` passed to kernels |
| Scale = 1/sqrt(d_k) | ✓ confirmed | `scale = 1 / (query.shape[-1] ** 0.5)` |
| Fast path requires `fla` + `causal_conv1d` | ✓ confirmed | `is_flash_linear_attention_available()`, `is_causal_conv1d_available()` checks |
| Slow path is pure PyTorch | ✓ confirmed | `torch_chunk_gated_delta_rule`, `torch_recurrent_gated_delta_rule` fallbacks |
| All GDN math in fp32 | ✓ confirmed | `.to(torch.float32)` on Q, K, V, beta, g before computation |
| KV head ratio (GQA) = 4:1 (4B) | ✓ confirmed | `num_attention_heads: 16`, `num_key_value_heads: 4` |

### What this means for the project

1. **The memory claim is structural**, not a tuning parameter — the recurrent state's shape has no sequence dimension by construction.
2. **The fast-path kernel gap is real** — `fla` and `causal_conv1d` are the optimized implementations; without them, the PyTorch fallback runs, but slower (`PLAN.md` §3 "the kernel gap").
3. **All GDN computation is in fp32** — quantization must keep the recurrent state and accumulation in fp32; only weights and activations can be quantized (per `ob-qpa` quantization policy).
4. **The delta rule has 5 sequential steps per token** — decay, retrieve, delta, update, output — explaining the latency-bound behavior observed on RK3588 (`FINDINGS.md` §6).
