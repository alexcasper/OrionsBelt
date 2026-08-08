# GDN layer-structure audit (from modeling code)

**Bead:** `ob-37v` · **Status:** Complete 2026-08-02
**Source:** `modeling_qwen3_5.py` and `modular_qwen3_5.py` from
[huggingface/transformers](https://github.com/huggingface/transformers) `main`
(sha fetched 2026-08-02), cross-referenced with `config.json` for both the 0.8 B
and 4 B checkpoints. This supersedes any figure quoted from a secondary source
(PER docs/archive/PLAN.md §2.3).

---

## 1. Layer placement — verified from `layer_types`

| Model | `num_hidden_layers` | GDN (`linear_attention`) | Full attention | Pattern |
|---|---:|---:|---:|---|
| **Qwen3.5-4B** | 32 | **24** | **8** | 8 × (3 GDN → 1 full) |
| **Qwen3.5-0.8B** | 24 | **18** | **6** | 6 × (3 GDN → 1 full) |

The pattern is confirmed from the explicit `layer_types` array in `config.json`
(not inferred from a formula). Every 4th layer (indices 3, 7, 11, …) is full
attention; all others are GDN. The `DecoderLayer.__init__` reads
`config.layer_types[layer_idx]` and instantiates either
`Qwen3_5GatedDeltaNet` or `Qwen3_5Attention` — never both.

---

## 2. GDN layer internals (`Qwen3_5GatedDeltaNet`)

### 2.1 Config-derived dimensions

| Parameter | 4 B | 0.8 B | Config key |
|---|---:|---:|---|
| `hidden_size` | 2560 | 1024 | `hidden_size` |
| Key head dim | 128 | 128 | `linear_key_head_dim` |
| Value head dim | 128 | 128 | `linear_value_head_dim` |
| Num key heads | 16 | 16 | `linear_num_key_heads` |
| Num value heads | 32 | 16 | `linear_num_value_heads` |
| `key_dim` (total) | 2048 | 2048 | `head_k_dim × num_k_heads` |
| `value_dim` (total) | 4096 | 2048 | `head_v_dim × num_v_heads` |
| Conv kernel dim | 4 | 4 | `linear_conv_kernel_dim` |
| KV head ratio (v/k) | 2 | 1 | `num_v_heads // num_k_heads` |

> **Note on the 4B's 32 value heads.** The 4B checkpoint has twice as many value
> heads as key heads (32 vs 16). Each key head is `repeat_interleave`'d to serve
> 2 value heads — so the key and query tensors are replicated by factor 2 in the
> head dimension before entering the delta rule. The 0.8 B has equal heads (16/16)
> so no replication occurs.

### 2.2 Input projections

Qwen3.5 splits the Qwen3-Next combined projections into four separate linears
(this is the main architectural difference from Qwen3-Next, confirmed in
`modular_qwen3_5.py` which deletes `in_proj_qkvz` and `in_proj_ba` from the
parent):

| Projection | Input | Output | Role |
|---|---|---|---|
| `in_proj_qkv` | `hidden_size` | `key_dim × 2 + value_dim` | Q, K, V (concatenated, pre-conv) |
| `in_proj_z` | `hidden_size` | `value_dim` | Output gate (SiLU) |
| `in_proj_b` | `hidden_size` | `num_v_heads` | Beta (write gate, per-value-head) |
| `in_proj_a` | `hidden_size` | `num_v_heads` | Decay-gate input (per-value-head) |

For 4B: `in_proj_qkv` maps 2560 → 8192; `in_proj_z` maps 2560 → 4096;
`in_proj_b` and `in_proj_a` map 2560 → 32.

### 2.3 Causal depthwise Conv1D

```python
self.conv1d = nn.Conv1d(
    in_channels=key_dim * 2 + value_dim,  # 8192 for 4B
    out_channels=key_dim * 2 + value_dim,
    bias=False,
    kernel_size=4,  # linear_conv_kernel_dim
    groups=key_dim * 2 + value_dim,  # full depthwise
    padding=3,  # kernel_size - 1 (causal)
)
```

- Applied to the concatenated QKV **before** splitting into Q/K/V.
- Activation: **SiLU** (`config.hidden_act = "silu"`).
- **Decode path** (`seq_len == 1`): uses `causal_conv1d_update` — shifts the
  conv state buffer in-place and applies the filter to a single token.
- **Prefill path** (`seq_len > 1`): uses `causal_conv1d_fn` — full sequence
  convolution with left-padding.
- Fast kernels: `causal_conv1d` package (Dao-AILab). **Fallback**: PyTorch
  `F.conv1d` with groups — functionally correct but slower.

**Conv state shape** (decode): `(batch, conv_dim, kernel_size)` =
`(B, 8192, 4)` for 4B = 131 072 elements per layer = 512 KiB in FP32.

### 2.4 Decay gate (`g`) — data-dependent, not a fixed scalar

```python
# Per-value-head parameters:
A_log = nn.Parameter(...)  # init: uniform(0, 16).log_()  → shape (num_v_heads,)
dt_bias = nn.Parameter(...)  # init: ones                  → shape (num_v_heads,)

# Forward:
beta = sigmoid(b)  # write gate
g = -exp(A_log) * softplus(a + dt_bias)  # decay gate (negative ⇒ decay)
```

- `g` is **input-dependent** (depends on `a` from `in_proj_a`) and
  **per-value-head** (32 independent decay rates for the 4B model).
- It is a **negative scalar** — `exp(g)` produces a decay factor in `(0, 1]`.
- In the chunk algorithm, `g` is **cumulatively summed** within each chunk:
  `g_cum = g.cumsum(dim=-1)`, then the decay between positions *i* and *j* is
  `exp(g_cum[j] - g_cum[i])`. This is the Mamba-2 / linear-attention decay
  formulation.

### 2.5 Delta-rule recurrence

**Recurrent state shape** (resolves the open question from ADR 0003 / `ob-eae`):

```
(batch, num_v_heads, head_k_dim, head_v_dim)
```

For 4B: `(B, 32, 128, 128)` = **524 288 elements** per layer = **2 MiB in FP32**.
For 0.8B: `(B, 16, 128, 128)` = **262 144 elements** per layer = **1 MiB in FP32**.

> **Confirmed:** the state's second dimension is `head_v_dim` (128), **not**
> `head_k_dim` again. The value dimension is the second inner axis because the
> state stores an outer product `k ⊗ v`, and `v` has its own head dimension. The
> memory figure in ADR 0003 (48 MiB for 24 layers at 4B) was correct.

**Per-token update** (decode path, `torch_recurrent_gated_delta_rule`):

```
S_t = S_{t-1} * exp(g_t)                          # decay
kv_mem = (S_{t-1} · k_t^T)                        # retrieve
delta  = (v_t - kv_mem) * beta_t                  # correction
S_t    = S_t + k_t ⊗ delta                        # write (rank-1 update)
output_t = S_t · q_t                              # read
```

This is the **delta rule** (a.k.a. delta-net): the state is updated by the
*error* between the stored value and the new value, weighted by `beta`. The
`exp(g_t)` decay is the Mamba-2-style forget gate. Q and K are L2-normalised
before entering the recurrence (`use_qk_l2norm_in_kernel=True`, always).

**Chunked prefill** (`torch_chunk_gated_delta_rule`):

1. Pad sequence to a multiple of `chunk_size = 64`.
2. Within each chunk: compute intra-chunk attention with the cumulative-decay
   mask, then apply a **WY-style Neumann-series correction** (the `for i in
   range(1, chunk_size)` loop that builds `attn += row + (row ⊗ sub).sum`).
3. Between chunks: sequential scan updating the recurrent state
   `S ← S * exp(g_last) + k^T @ v_corrected`.

> **Chunk size is hard-coded at 64** in the PyTorch fallback
> (`torch_chunk_gated_delta_rule(... chunk_size=64)`). The FLA fast-path kernel
> (`chunk_gated_delta_rule`) may choose its own chunk size internally.

### 2.6 Output norm and projection

```python
core_attn_out = Qwen3_5RMSNormGated(core_attn_out, z)  # RMSNorm then SiLU(z)
output = self.out_proj(core_attn_out)  # value_dim → hidden_size
```

`RMSNormGated` applies RMS-normalisation, multiplies by a learnable weight,
then gates by `SiLU(z)` where `z` comes from `in_proj_z`.

---

## 3. Full-attention layer (`Qwen3_5Attention`) — for contrast

| Property | Value (4B) |
|---|---|
| Attention type | GQA with gating |
| Num attention heads | 16 |
| Num KV heads | 4 |
| Head dim | 256 |
| Partial rotary factor | 0.25 (only first 64 dims get RoPE) |
| Q/K normalisation | RMSNorm per head dim |
| **Gating** | q_proj outputs 2× (query + gate); output × `sigmoid(gate)` |
| KV cache (per layer) | `(B, 4, seq_len, 256) × 2` (K + V) |

The gating on full attention is notable: it is not standard GQA but
**gated-GQA**, the same SiLU-gate pattern as the GDN output norm.

---

## 4. Kernel-gap confirmation (critical for this project)

The GDN layer has **two execution paths**, selected at runtime:

| Path | Condition | Fast kernel | Fallback |
|---|---|---|---|
| Chunked (prefill) | `seq_len > 1` | `fla.ops.gated_delta_rule.chunk_gated_delta_rule` | `torch_chunk_gated_delta_rule` |
| Recurrent (decode) | `seq_len == 1` with cache | `fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule` | `torch_recurrent_gated_delta_rule` |
| Conv1D | always | `causal_conv1d_fn` / `causal_conv1d_update` | `F.conv1d` (PyTorch) |

The code checks for the packages and logs a warning on fallback:

```python
if not is_flash_linear_attention_available() or not is_causal_conv1d_available():
    logger.warning_once("The fast path is not available … Falling back to torch implementation.")
```

**Neither `fla` nor `causal_conv1d` ships a build for Arm (AArch64) or Vulkan.**
On NVIDIA GB10 (SM121) the same gap already occurs. **Arm is the same hole,
wider** — this is the project's central contribution (docs/archive/PLAN.md §3).

The PyTorch fallbacks are functionally correct but:
- The chunked fallback has a Python-level `for i in range(0, num_chunks)` loop
  over chunks and a `for i in range(1, chunk_size)` Neumann-series loop — both
  are sequential and unparallelisable.
- The recurrent fallback has a Python-level `for i in range(sequence_length)`
  token-by-token loop — catastrophically slow for prefill, and the reason decode
  is memory-bandwidth-bound rather than compute-bound.

---

## 5. Memory decomposition (per model)

### GDN recurrent state (O(1) per token, constant across context length)

| Model | State per layer | GDN layers | Total state | FP16 | FP32 |
|---|---:|---:|---:|---:|---:|
| 4B | 524 288 floats | 24 | 12 582 912 floats | **24 MiB** | **48 MiB** |
| 0.8B | 262 144 floats | 18 | 4 718 592 floats | **9 MiB** | **18 MiB** |

### GDN conv state (O(1), decode only)

| Model | Per layer | GDN layers | Total (FP32) |
|---|---:|---:|---:|
| 4B | 131 072 floats | 24 | **12 MiB** |
| 0.8B | 98 304 floats | 18 | **6.75 MiB** |

### Full-attention KV cache (grows linearly with context length)

| Model | Per layer per token | FA layers | @ 4K | @ 32K | @ 262K |
|---|---:|---:|---:|---:|---:|
| 4B | 4 heads × 256 × 2 (K+V) = 2048 floats | 8 | **128 MB** | **1 GB** | **8 GB** |
| 0.8B | 2 heads × 256 × 2 = 1024 floats | 6 | **24 MB** | **192 MB** | **1.5 GB** |

(FP16; halve for INT8 quantised cache, quarter for INT4.)

> This table is the empirical basis for the project's central claim: **GDN state
> is flat at every context length while the KV cache grows linearly.** At 262K
> context the 4B model's GDN state is 48 MiB (FP32) while its KV cache is 8 GB
> (FP16) — a 170:1 ratio.

---

## 6. Summary of differences from Qwen3-Next

Qwen3.5's GDN layer inherits from `Qwen3NextGatedDeltaNet` but:

1. **Splits projections**: `in_proj_qkvz` → `in_proj_qkv` + `in_proj_z`;
   `in_proj_ba` → `in_proj_b` + `in_proj_a`. (4 separate projections vs 2.)
2. **Removes `fix_query_key_value_ordering`** — no interleaved QKV layout.
3. **Always L2-normalises Q and K** (`use_qk_l2norm_in_kernel=True`).
4. Uses the same `A_log` / `dt_bias` / `softplus` decay-gate formulation.
5. Same chunk size (64), same recurrent state shape, same delta rule.

---

## 7. Implications for the heterogeneous mapping (E6)

1. **The sequential scan (chunk loop + token loop) is the NPU-rejected part.**
   Each chunk iteration is a small matmul over the recurrent state — but the
   state must be threaded across iterations, and the NPU has no runtime-length
   loop construct (FINDINGS.md §1). The CPU or GPU must own this.

2. **The conv1d is depthwise with groups = 8192** (4B). This is a pure
   elementwise-multiply-and-accumulate along the kernel dimension — trivially
   parallelisable on NEON/SVE, no matmul engine needed.

3. **The projections (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`) are
   dense matmuls** — ideal for the NPU or GPU. The output projection
   (`out_proj`) likewise.

4. **The decay gate (`g = -exp(A_log) * softplus(a + dt_bias)`) involves `exp`
   and `softplus`** — elementwise, precision-sensitive. Per ADR policy, the gate
   values should stay FP16 even under INT8/INT4 weight quantisation.

5. **L2 normalisation of Q and K** happens inside the delta rule — another
   elementwise op that must stay with the scan kernel, not the matmul engine.
