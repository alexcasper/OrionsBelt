# Quantization policy for Qwen3.5 GDN on Arm (ob-qpa)

**Status:** Accepted 2026-08-02 · **Bead:** `ob-qpa` · **ADR:** [0006](./adr/0006-quantization-policy.md)

Grounded in the confirmed GDN layer structure ([ob-37v](./FINDINGS.md#6)), the
arithmetic-intensity analysis ([METRICS.md §9](./METRICS.md)), the KleidiAI
coverage audit ([FINDINGS.md §3.3](./FINDINGS.md)), and the checkpoint
dimensions ([ADR 0003](./adr/0003-model-checkpoint-selection.md)).

---

## The one-sentence summary

**Quantize weights aggressively (INT4 primary, INT8 conservative); keep recurrent
state in FP32 and gates in FP16/FP32.** Weight streaming is 95–99% of decode
bandwidth (METRICS.md §9 appendix), so weight quantization is the dominant lever —
not kernel work, not state narrowing.

---

## 1. Per-tensor precision assignment

| Tensor class | Quantization | Rationale |
|---|---|---|
| **All projection weights** (GDN in_proj_qkv/z/b/a, out_proj; FA q/k/v/o; MLP gate/up/down) | **INT4** (primary) or **INT8** (conservative) | The dominant decode-bandwidth term. INT4 is a 4× traffic reduction, taking the 4B from ~12.5 to ~50 tok/s at 100 GB/s. KleidiAI ships 109 A720-usable quantized matmul kernels (FINDINGS.md §3.3). |
| **Recurrent state S** (GDN layers) | **FP32** — carve-out | State is fed back through every token via `S_t = S_{t-1} * exp(g_t) + correction`. Quantization error compounds across the sequence, unlike a KV-cache entry that is read once and discarded. The state is also small (48 MiB for 4B across 24 layers — METRICS.md §9), so keeping it in FP32 costs ~5% of INT4-weight decode traffic. |
| **Decay gate parameters** (A_log, dt_bias) | **FP32** — carve-out | The exponential decay `exp(-A_log * softplus(a + dt_bias))` is the model's memory mechanism. Small errors in the decay rate compound multiplicatively over hundreds of tokens. FINDINGS.md §4 documents: "The decay accumulator is fp32 even when surrounding state is fp16." |
| **Input-dependent gate a** (from in_proj_a) | **FP16** | Computed from weights (can be INT4-quantized) but the gate output `softplus(a + dt_bias)` should accumulate in FP16. Less sensitive than the state because it is recomputed every token from a fresh projection. |
| **Beta gate b** (from in_proj_b) | **FP16** | `sigmoid(b)` is bounded in [0, 1] and recomputed per token. Less precision-sensitive than decay. |
| **KV cache** (full-attention layers) | **FP16** (default), INT8 candidate | Standard FP16 cache. INT8 KV cache quantization is a future optimization gated on accuracy regression against the oracle (ob-3uh). The KV cache grows linearly with context, so at 262K it is the largest single memory consumer — but it is also the architecture's advantage (only 8 layers, not 32), so quantizing it is optional, not necessary. |
| **RMSNorm** (all layers) | **FP16** — carve-out | `rms_norm_eps = 1e-6` in config. FP16 has ~3 decimal digits of precision; FP32 is safer for the epsilon but FP16 is standard practice. The norm weights can be INT8-quantized without issue. |
| **Embeddings / output projection** (tied) | **FP16** | Tied weights (`tie_word_embeddings: true`). Vocabulary-sensitive; the output projection produces logits over 248K tokens. Keep FP16. |

---

## 2. Mixed-precision compute strategy

```
                    ┌─────────────────────────────────────────┐
  INT4/INT8 weights │  KleidiAI INT4/i8mm GEMV micro-kernels   │  FP32 accumulator
  ──────────────────▶│  (109 A720-usable kernels, FINDINGS §3.3)│──────────────────▶
                    └─────────────────────────────────────────┘
                    ┌─────────────────────────────────────────┐
  FP16 activations  │  Standard NEON/SVE FP16 ops              │  FP32 accumulate
  ──────────────────▶│  (no special kernel needed)             │──────────────────▶
                    └─────────────────────────────────────────┘
                    ┌─────────────────────────────────────────┐
  FP32 recurrent    │  Our SVE/NEON gated-scan kernels        │  FP32 throughout
  state + gates     │  (ob-8qt.3, FINDINGS §4)                │──────────────────▶
  ──────────────────▶│  (NOT quantized — the carve-out)        │
                    └─────────────────────────────────────────┘
```

**Three lanes, three precisions, by design.** The weight-quantized matmuls use
KleidiAI's micro-kernels. The recurrent scan uses our own FP32 kernels. Activations
bridge the two in FP16. This maps directly to the heterogeneous engine assignment
(PLAN.md §3.1): GDN scan on CPU (FP32 state), dense matmuls (weights) wherever
KleidiAI or the NPU can accelerate them.

---

## 3. Expected impact (Qwen3.5-4B, 100 GB/s LPDDR5)

| Configuration | Weight traffic | State traffic | Decode tok/s (est.) |
|---|---:|---:|---:|
| FP16 weights, FP32 state (baseline) | 7.5 GiB | 96 MiB (~1.3%) | ~12.5 |
| INT8 weights, FP32 state | 3.75 GiB | 96 MiB (~2.5%) | ~25 |
| **INT4 weights, FP32 state** (recommended) | **1.9 GiB** | 96 MiB (~5%) | **~50** |
| INT4 weights, BF16 state | 1.9 GiB | 48 MiB (~2.5%) | ~50 (negligible gain) |

Source model: weight_bytes / 100 GB/s → tok/s ceiling. The state-traffic column
confirms METRICS.md §9's conclusion: **narrowing the recurrent state to BF16 buys
~2–3% of decode traffic at INT4 weights — not a step change.** The bf16 state
variant remains worth doing for its memory-footprint benefit (fitting longer
context alongside weights), not for decode throughput (ob-8qt.4 re-scoped accordingly).

---

## 4. What must NOT be quantized

Stated plainly so it is never accidentally violated:

1. **The recurrent state S** — accumulates error across the entire sequence.
   This is not a theoretical concern: the delta-rule correction
   `S += k ⊗ (v - S@k) * beta` means a corrupted S produces wrong retrievals at
   every subsequent step, compounding rather than staying local.

2. **The decay rate** — `A_log` and `dt_bias` parameters. A 1% error in the decay
   rate means the effective memory horizon shifts; over 262K tokens this is the
   difference between remembering and forgetting.

3. **The RMSNorm epsilon** — at `1e-6`, this is near the FP16 precision floor.
   The norm computation itself must accumulate in FP32.

These carve-outs are cheap: the recurrent state is 48 MiB (0.6% of INT4 weights),
the gate parameters are `num_v_heads` scalars per layer (negligible), and RMSNorm
is standard FP16 practice.

---

## 5. KleidiAI applicability

| GDN operation | KleidiAI kernel? | Precision |
|---|---|---|
| Projection matmuls (in_proj, out_proj, MLP) | ✅ 109 `matmul` kernels, incl. i8mm/dotprod | INT4/INT8 weights → FP16/FP32 output |
| Delta-rule small matmuls (k⊗delta, S@k, S@q) | ✅ same `matmul` family | Could use INT8 for k,v,q; but S stays FP32 |
| Causal depthwise Conv1D | ❌ KleidiAI's is SME2-only | Our SVE/NEON kernel (ob-8qt.3) |
| Gated scan recurrence | ❌ Not a matmul | Our FP32 kernel (ob-8qt.3) |

The KleidiAI micro-kernels apply to the **dense matmuls** (weight-quantized
projections), which is exactly where weight quantization matters. The recurrent
scan is covered by our own FP32 kernels and is not quantized.

---

## 6. Validation gate

Every quantization configuration must pass the correctness oracle (ob-3uh) before
its numbers enter a results table. Per PLAN.md §9: "speed that changes outputs is
not speed." The tolerance policy:

- **Token-level:** generated text must match the FP16 reference on ≥95% of test
  prompts at each context length (4K, 32K, 128K).
- **Logit-level:** KL divergence of the output distribution vs the FP16 reference
  must be < 0.1 on the test corpus.
- **Long-context:** at 128K+, where recurrent-state drift compounds, the oracle
  must explicitly check multi-key retrieval accuracy, not just perplexity.

If INT4 fails the oracle, fall back to INT8 (which is more conservative and still
a 2× decode speedup). If INT8 also fails, investigate per-layer sensitivity
(quantize only the largest layers first, leave small layers in FP16).

---

## 7. Summary table for the write-up

| Layer class | Weight precision | State/cache precision | Engine |
|---|---|---|---|
| GDN (24 layers) | INT4 (via KleidiAI i8mm GEMV) | **FP32** (recurrent state, carved out) | CPU (SVE/NEON scan) |
| Full attention (8 layers) | INT4 (via KleidiAI i8mm GEMV) | FP16 (KV cache) | NPU or GPU |
| MLP/FFN (32 layers) | INT4 (via KleidiAI i8mm GEMV) | — | NPU or GPU |
| Norms | FP16 weights | — | CPU |
| Embeddings | FP16 (tied) | — | CPU |
