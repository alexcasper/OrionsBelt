# Quantization policy — FP16 carve-outs for GDN state and gates

**Bead:** `ob-qpa` · **Status:** Complete 2026-08-02
**Executable form:** [`src/orionsbelt/quant/policy.py`](../src/orionsbelt/quant/policy.py)

---

## 1. The problem: GDN state error compounds

Standard attention's KV cache stores each token's K and V independently. If a
single entry has quantization error *ε*, that error affects only that token's
attention contribution — it does **not** propagate to future tokens.

GDN's recurrent state is different. The delta-rule update is:

```
S_t = S_{t-1} * exp(g_t) + k_t ⊗ (v_t - S_{t-1}^T k_t) * beta_t
```

The state `S_t` carries **all** previous information, compressed into a fixed
matrix. If `S_{t-1}` has quantization error *ε*, then `S_t` inherits
`ε * exp(g_t)` plus new quantization error from the current step. Over a long
sequence (262K tokens), even a per-step *ε* = 10⁻⁴ compounds multiplicatively
into systematic drift — the model forgets early context, or retrieves the wrong
content from state.

This is the same reason RNNs and LSTMs are notoriously hard to quantize: the
recurrent loop amplifies numerical errors. The KV cache does not have this
property because it has no recurrence.

---

## 2. The decode-bandwidth argument for weight quantization

From [`METRICS.md`](./METRICS.md) appendix, verified against the selected
4B checkpoint (ADR 0003):

| Traffic source | Per token | Share of decode bandwidth |
|---|---:|---:|
| Weights, FP16 (4B) | 7.5 GiB | ~99% |
| Weights, INT4 (4B) | 1.9 GiB | ~95% |
| GDN state, FP32 (24 layers) | 96 MiB | ~5% at INT4 |
| GDN state, BF16 (24 layers) | 48 MiB | ~2.4% at INT4 |

**INT4 weight quantization is the dominant decode-throughput lever** — a ~4×
reduction in weight traffic, taking the 4B from ~12.5 to ~50 tok/s at 100 GB/s.

Narrowing the recurrent state from FP32 to BF16 buys only ~2–3% of decode
traffic. It is worth doing as a **memory-footprint** optimization (halves the
resident state, which matters for fitting long context) but not as a
decode-throughput one.

---

## 3. The policy

### 3.1 INT4 weight-only (the dominant lever)

Applied to all dense matmul weights that have **no recurrence interaction**:

| Tensor group | Layers | Why INT4 is safe |
|---|---|---|
| `in_proj_qkv.weight` | GDN | Output feeds conv → delta-rule, but the projection itself is a one-shot matmul. Weight error is local. |
| `in_proj_z.weight` | GDN | Gate projection; gate value is not fed back. |
| `out_proj.weight` | GDN | Output projection, no recurrence. |
| `q/k/v/o_proj.weight` | Full attn | Standard GQA, no recurrence. |
| `gate/up/down_proj.weight` | MLP (FFN) | SwiGLU MLP — largest weight block, biggest INT4 win. |

Activation stays FP16 (W4A16 scheme). KleidiAI's 109 A720-usable INT4/i8mm
GEMV micro-kernels apply directly here.

### 3.2 INT8 weight-only (conservative for precision-sensitive projections)

| Tensor group | Layers | Why INT8, not INT4 |
|---|---|---|
| `in_proj_b.weight` | GDN | Beta (write-gate) controls state-write magnitude via the delta rule. |
| `in_proj_a.weight` | GDN | Decay-gate input enters `exp()` — amplified error. |
| `conv1d.weight` | GDN | Depthwise conv feeding Q/K/V; short, stable computation. |
| `embed_tokens.weight` | Global | Rare-token representation sensitivity. |
| `lm_head.weight` | Global | Logit resolution across 248K vocabulary. |

### 3.3 FP16+ carve-outs (must NOT be quantized)

| Tensor / computation | Precision | Why |
|---|---|---|
| **Recurrent state** (S) | **FP32** | Error compounds multiplicatively through the recurrence loop. At 262K tokens, per-step *ε* = 10⁻⁴ becomes systematic drift. **This is the critical carve-out.** |
| **A_log** parameter | FP16 | Learnable decay rate, used inside `exp(A_log)`. Any quantization error is exponentially amplified. |
| **dt_bias** parameter | FP16 | Added to decay-gate input before `softplus`. Same `exp()` amplification path. |
| **RMSNorm** weights | FP16 (computed FP32) | All normalizations are numerically sensitive; compute in FP32, store FP16. |
| **KV cache** | FP16 | Unlike GDN state, entries are independent (no compounding). FP16 sufficient. |

### 3.4 Activation quantization (W8A8): deferred

W8A8 (quantizing activations in addition to weights) is **not recommended** for
GDN layers — the recurrent path means activation quantization error compounds
through the loop, same as state error. It may be viable for full-attention and
FFN layers where error is local, but this requires empirical validation against
the correctness oracle and is deferred to a later bead.

---

## 4. Estimated memory footprint (Qwen3.5-4B)

Using `policy.estimate_weight_footprint_mib(total_params=4_020_000_000)`:

| Tier | Params (est.) | Bytes/param | Footprint |
|---|---:|---:|---:|
| INT4 (projections, MLP) | ~85% | 0.5 | ~1.6 GiB |
| INT8 (embeddings, sensitive proj) | ~10% | 1.0 | ~0.4 GiB |
| FP16 (norms, A_log, dt_bias) | ~5% | 2.0 | ~0.4 GiB |
| **Total weights** | | | **~2.4 GiB** |
| FP32 recurrent state (24 layers) | | | 48 MiB |
| FP16 KV cache @ 4K context | | | 128 MiB |
| **Total resident** | | | **~2.6 GiB** |

vs. ~7.5 GiB at pure FP16 — a **3.1× compression** with the recurrent state
preserved at full precision.

---

## 5. Implementation paths

| Component | INT4/INT8 path | FP16 carve-out path |
|---|---|---|
| **NPU (CIX NOE)** | NOE Compiler INT4/INT8 weight quantization for attention + FFN subgraphs | GDN scan stays on CPU (NPU can't express recurrence — FINDINGS.md §1) |
| **CPU (Arm)** | KleidiAI INT4/i8mm GEMV micro-kernels for weight-only quantized matmuls | State update in FP32 NEON/SVE; gates in FP16 |
| **GPU (Vulkan)** | Not the primary quantization target (Vulkan scan is FP16) | N/A — scan kernel is the GPU's job, stays FP16 |

---

## 6. Open questions (for empirical validation)

1. **BF16 recurrent state (ob-8qt.4):** Does BF16 state pass the correctness
   oracle at 262K context? If yes, halve the state footprint (48→24 MiB). If
   no, document the accuracy regression and keep FP32.

2. **INT8 KV cache:** Does quantizing the KV cache to INT8 pass the oracle at
   long context? This would halve KV memory but is secondary to weight
   quantization (KV is 128 MiB @ 4K vs 2.4 GiB weights).

3. **W8A8 for FFN only:** Quantizing FFN activations (not GDN/attention) could
   speed up the MLP matmuls further, but requires the oracle gate.
