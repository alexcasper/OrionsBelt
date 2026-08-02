# GDN layer structure audit — ground truth from modeling code

**Bead:** `ob-37v` · **Status:** Complete 2026-08-02 · **Parent:** `ob-xh3` (E4)

This document is the **ground truth** for the Qwen3.5 Gated DeltaNet layer structure, read
directly from the transformers library's `modeling_qwen3_5.py` (v4.57.0.dev0, fetched from
the `huggingface/transformers` `main` branch). It supersedes any figure quoted from a secondary
source. Everything the mapping strategy (E6), quantization policy (E7), and memory
instrumentation (E5) depends on is defined here.

The config fields come from `Qwen/Qwen3.5-4B` `config.json` (text_config), read live from
HuggingFace. The implementation details come from the actual Python source.

---

## 1. Layer placement: confirmed 3:1 hybrid

The `layer_types` array in `config.json` is explicit and read per-layer by
`Qwen3_5DecoderLayer.__init__`:

```python
self.block_type = config.layer_types[layer_idx]
if self.block_type == "linear_attention":
    self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
elif self.block_type == "full_attention":
    self.self_attn = Qwen3_5Attention(config, layer_idx)
```

| | **Qwen3.5-4B** | **Qwen3.5-0.8B** |
|---|---|---|
| Total layers | 32 | 24 |
| GDN (`linear_attention`) | 24 | 18 |
| Full attention | 8 | 6 |
| Ratio | exactly 3:1 | exactly 3:1 |
| `full_attention_interval` | 4 | 4 |
| Attention layer indices | 3, 7, 11, 15, 19, 23, 27, 31 | 3, 7, 11, 15, 19, 23 |

Every decoder layer (GDN or attention) is followed by a shared MLP block (SwiGLU FFN). The
attention and GDN layers are mutually exclusive token mixers — a given layer is one or the
other, never both.

---

## 2. Recurrent state shape — the load-bearing finding

**State shape: `[batch, num_v_heads, head_k_dim, head_v_dim]`** — a set of independent
`[d_k × d_v]` matrices, one per value head. This is NOT a flat vector.

From `torch_chunk_gated_delta_rule` and `torch_recurrent_gated_delta_rule`:

```python
last_recurrent_state = torch.zeros(
    batch_size, num_heads, k_head_dim, v_head_dim, dtype=torch.float32, ...
)
```

For **Qwen3.5-4B**:

| Field | Config value | Source |
|---|---|---|
| `num_v_heads` | **32** | `config.linear_num_value_heads` |
| `head_k_dim` | **128** | `config.linear_key_head_dim` |
| `head_v_dim` | **128** | `config.linear_value_head_dim` |
| Elements/layer | 32 × 128 × 128 = **524,288** | computed |
| Bytes/layer (fp32) | **2,097,152 (2 MiB)** | `mamba_ssm_dtype = 'float32'` |
| **Total across 24 GDN layers** | **48 MiB** | flat — O(1) per token |

For **Qwen3.5-0.8B** (16 value heads, 18 GDN layers): 262,144 elements/layer = 1 MiB →
**18 MiB total**.

**The `mamba_ssm_dtype: 'float32'` config field confirms the state is always stored in fp32
regardless of the model's compute dtype.** This is critical for numerical stability — the
rank-1 delta-rule updates accumulate over the full sequence and would lose precision in
fp16/bf16. The quantization policy (ob-qpa) must carve out the recurrent state as fp32.

The key/value head count is asymmetric: **16 key heads vs 32 value heads**. Query and key are
`repeat_interleave`d by the ratio (2× for 4B) before entering the delta rule, so the
recurrence runs over `num_v_heads` independent state matrices, each updated by its own key
vector. This is an important implementation detail for the CPU kernel — the state is NOT
`[n_v_heads, d_k, d_v]` contiguous in the sense of a single large matrix; it's 32 separate
`[128, 128]` matrices, each receiving a rank-1 update per token.

---

## 3. Delta-rule update — the decode recurrence

This is the operation the CPU kernel must implement (and the NPU cannot, per
`docs/FINDINGS.md` §1). Read from `torch_recurrent_gated_delta_rule`:

```python
for i in range(sequence_length):
    # 1. Gated decay
    last_recurrent_state = last_recurrent_state * g_t          # element-wise × exp(g)

    # 2. Retrieve (matrix-vector product)
    kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)

    # 3. Delta (prediction error)
    delta = (v_t - kv_mem) * beta_t

    # 4. Rank-1 update (outer product)
    last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)

    # 5. Output (matrix-vector product)
    core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
```

Per token, per value head, this is:
- One element-wise multiply on `[d_k × d_v]` (decay)
- One reduce-sum over `d_k` (retrieve)
- One outer-product add on `[d_k × d_v]` (update)
- One reduce-sum over `d_k` (output)

That is **1 MAC per state element** for the dominant gated-decay + update, confirming the
arithmetic-intensity calculation in `docs/METRICS.md` §9: **0.25 FLOP/byte** — bandwidth-bound.

---

## 4. Gates — confirmed GDN (not GDN-2)

Two **separate, scalar-per-head** gates control the recurrence:

### Decay gate `g`

```python
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
```

- `A_log` is a learned parameter initialised `log(uniform(0, 16))` — one scalar per value head
- `a` is the input-dependent gate from `in_proj_a(hidden_states)` — one scalar per value head per token
- `dt_bias` is a learned bias — one scalar per value head
- In the recurrence, `g_t.exp()` produces the actual decay factor ∈ (0, 1)
- This is a **single scalar per head** that ties erase + write decay — the defining property of
  GDN (vs GDN-2's separate channel-wise erase `b_t` and write `w_t` gates)

### Write gate `beta`

```python
beta = b.sigmoid()     # b from in_proj_b(hidden_states), one scalar per value head per token
```

Controls the delta-rule write strength — how aggressively the prediction error corrects the state.

---

## 5. Causal Conv1D — depthwise, kernel 4

```python
self.conv1d = nn.Conv1d(
    in_channels=self.conv_dim,    # key_dim * 2 + value_dim = 8192 (4B)
    out_channels=self.conv_dim,
    bias=False,
    kernel_size=self.conv_kernel_size,  # = config.linear_conv_kernel_dim = 4
    groups=self.conv_dim,         # depthwise — one filter per channel
    padding=self.conv_kernel_size - 1,  # causal: pad on the left
)
```

- Applied to the **mixed QKV** projection before splitting into Q/K/V (not after)
- Depthwise (groups = channels) — each channel has its own 4-tap filter
- Left-padded → causal
- Followed by SiLU activation (`config.hidden_act = 'silu'`)
- At decode (seq_len=1), uses `causal_conv1d_update` with an in-place conv state of width
  `kernel_size - 1 = 3` past tokens

---

## 6. Chunk size for prefill

The chunkwise scan (`torch_chunk_gated_delta_rule`) uses **`chunk_size = 64`** by default.

Prefill processes the prompt in 64-token chunks:
1. Intra-chunk: attention-like computation with causal masking + cumulative decay
2. Inter-chunk: state propagation via the recurrent delta rule, chunk by chunk

This is where the 1.38–1.49× kernel optimization opportunity lives (PLAN.md §2.4) — the
chunkwise matmuls are the NPU-friendly part, while the inter-chunk state propagation is the
sequential scan the NPU cannot express.

---

## 7. L2 normalization

```python
use_qk_l2norm_in_kernel=True
```

Query and key vectors are L2-normalized before entering the delta rule. This is applied
inside both the chunk and recurrent paths:

```python
if use_qk_l2norm_in_kernel:
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
```

The CPU kernel must replicate this — it affects the numerical contract of the delta rule.

---

## 8. Input projections

Four linear projections from `hidden_size` (2560):

| Projection | Output dim | 4B size | Purpose |
|---|---|---|---|
| `in_proj_qkv` | `key_dim × 2 + value_dim` = 8192 | 2560×8192 | Q, K, V (concatenated) |
| `in_proj_z` | `value_dim` = 4096 | 2560×4096 | Output gate for RMSNormGated |
| `in_proj_b` | `num_v_heads` = 32 | 2560×32 | Write gate (beta) |
| `in_proj_a` | `num_v_heads` = 32 | 2560×32 | Decay gate input (alpha) |
| `out_proj` | `hidden_size` = 2560 | 4096×2560 | Output projection |

The gate projections (`b`, `a`) are tiny (2560×32) compared to the QKV projection (2560×8192).

---

## 9. Full-attention layers (for comparison)

The 8 full-attention layers use standard GQA with an **output gate**:

| Field | Config value |
|---|---|
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 4 (GQA ratio 4:1) |
| `head_dim` | 256 |
| RoPE | yes, `rope_theta = 10,000,000`, `partial_rotary_factor = 0.25` |
| Q/K normalization | RMSNorm on head_dim |
| Output gate | `sigmoid(gate)` on the attention output (from a doubled q_proj) |

These are the **NPU-friendly** layers — large dense matmuls over a growing KV cache.

---

## 10. The kernel gap — confirmed in source

The modeling code explicitly depends on two optional libraries:

```python
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
```

And checks:

```python
if not is_flash_linear_attention_available() or not is_causal_conv1d_available():
    logger.warning_once(
        "The fast path is not available because one of the required library is not installed. "
        "Falling back to torch implementation."
    )
```

The torch fallbacks (`torch_chunk_gated_delta_rule`, `torch_recurrent_gated_delta_rule`) are
correct but slow — they use Python-level loops over tokens for the recurrent path and naive
matmuls for the chunk path. Neither `fla` nor `causal_conv1d` ships an Arm/aarch64 build.

**This is the contribution statement made concrete in source:** the model silently falls back
to slow PyTorch ops on Arm, and our CPU kernel (NEON/SVE2 path in `src/orionsbelt/engines/cpu/`)
is what replaces that fallback.

---

## 11. Implications for downstream work

### For the quantization policy (ob-qpa)
- Recurrent state **must stay fp32** — `mamba_ssm_dtype` confirms it, and the delta-rule
  accumulation requires the precision
- Gates (`a`, `b`) are tiny (2560×32) — quantizing them saves nothing and risks numerical
  instability; keep fp16/bf16
- QKV and output projections are the quantization targets (they are 90%+ of the weight mass)

### For the memory instrumentation (ob-vfp)
- Recurrent state per layer = `num_v_heads × head_k_dim × head_v_dim × 4` bytes
- Read `num_v_heads`, `head_k_dim`, `head_v_dim` from `config.linear_*` fields, NOT hardcoded
- The state is `O(1)` — it does not grow with context length, by construction

### For the CPU kernel (ob-8qt)
- The state is 32 independent `[128, 128]` matrices — each gets a rank-1 update per token
- The dominant per-token operations are element-wise on `[128, 128]` and reduce-sums over `d_k=128`
- L2-normalize Q and K before the delta rule
- The gated decay uses `exp(g)` where `g` is negative — always a decay factor ∈ (0, 1)

### For the heterogeneous mapping (ob-o4g / E6)
- 16 engine-boundary crossings per token (8 GDN→attn transitions, 8 attn→GDN)
- The payload at each crossing is `hidden_size = 2560` floats (5 KiB at fp16)
- Cost is invocation latency, not bandwidth
