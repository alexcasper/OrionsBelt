# ADR 0006: Quantization policy — INT4 weights with FP16 carve-outs and fp32 recurrent state

- **Status:** Proposed
- **Date:** 2026-08-02
- **Bead:** `ob-qpa`
- **Deciders:** maintainer + agent (rk3588-t4)
- **Supersedes nothing.** Depends on [ADR 0003](./0003-model-checkpoint-selection.md) (checkpoint
  selection), [ADR 0004](./0004-descope-ladder.md) (descope tiers), and the ground-truth layer
  audit ([`docs/GDN_LAYER_AUDIT.md`](../GDN_LAYER_AUDIT.md), `ob-37v`).

## Context

Decode on this hardware is **weight-bandwidth-bound** (METRICS.md §9 appendix). At batch=1, every
weight byte is re-fetched from DRAM per token because there is no cross-token amortization. The
numbers for Qwen3.5-4B at the verified 100 GB/s LPDDR5 bandwidth:

| Precision | Weight traffic/token | Ceiling tok/s | GDN state share |
|---|---:|---:|---:|
| fp16 (baseline) | 7.56 GiB | ~13 | 1.3% |
| INT8 weights | 3.78 GiB | ~26 | 2.6% |
| **INT4 weights** | **1.89 GiB** | **~53** | **5.1%** |

(INT4 = 4-bit weight-only, W4A16: weights stored compressed, dequantized to fp16 before the GEMV.
Compute stays fp16 throughout — only storage and traffic are reduced.)

Weight quantization is the single highest-value optimization lever in the project
(`ob-qpa` notes): INT4 takes the 4B from ~13 to ~53 tok/s by this model — a genuine 4× win.
The GDN state at 48 MiB fp32 is real traffic but is only ~5% of INT4 decode traffic and is
fixed by the recurrence definition, so it cannot be moved by weight quantization.

The question this ADR answers is **not whether** to quantize weights (that is settled by the
bandwidth arithmetic) but **which tensors are safe to quantize and which must stay high-precision**.
Unlike a conventional KV cache where quantization error is local to the cached token, GDN
recurrent state is **fed back through every decode step**, so errors compound over the sequence.
The audit (`ob-37v`) identified the precision-sensitive tensors; this ADR turns that into a
concrete per-tensor policy.

### Evidence base

All per-tensor parameter counts below are computed analytically from the checkpoint config
(`bench/metrics.py`, `ob-vfp`) and cross-checked against the modeling-code audit (`ob-37v`). The
4B checkpoint has `tie_word_embeddings: false`.

**GDN layer parameter breakdown** (one of 24 GDN layers, 4B):

| Tensor | Params | Share of GDN layer | Precision sensitivity |
|---|---:|---:|---|
| `in_proj_qkv` (2560→8192) | 20,971,520 | 49.8% | Low — dense matmul, no feedback |
| `in_proj_z` (2560→4096) | 10,485,760 | 24.9% | Low — output gate projection |
| `out_proj` (4096→2560) | 10,485,760 | 24.9% | Low — dense matmul |
| `in_proj_b` (2560→32) | 81,920 | 0.2% | **High** — write gate β controls delta-rule strength |
| `in_proj_a` (2560→32) | 81,920 | 0.2% | **High** — decay gate input, compounds via exp(g) |
| `conv1d` (dw, 4-tap) | 32,768 | 0.1% | Medium — gates QKV before delta rule |
| `A_log` | 32 | ~0% | **Critical** — decay magnitude, exp() applied |
| `dt_bias` | 32 | ~0% | **High** — added to decay gate input |
| norm weight | 4,096 | ~0% | Medium — RMSNorm on value head dim |
| layernorms (×2) | 5,120 | ~0% | Medium — RMSNorm before/after |

**Aggregate** (across all 32 layers + embeddings):

| Component | Params | Share of total |
|---|---:|---:|
| Quantizable weights (all large projections + MLP + lm_head) | 3,777,542,656 | **99.87%** |
| FP16 carve-outs (gates + conv + norms + embeddings) | 4,941,312 | **0.13%** |
| **Total** | **3,782,483,968** | 100% |

The carve-out is under 5M params (10 MiB fp16) — **0.13% of the model**. Keeping it in fp16 costs
negligible decode bandwidth but protects every precision-sensitive signal path.

## Decision

**INT4 weight-only (W4A16) for all large weight matrices, with three FP16 carve-out classes and an
unchangeable fp32 floor on recurrent state.**

### Per-tensor assignment

| Tier | Tensors | Precision | Rationale |
|---|---|---|---|
| **INT4 (W4A16)** | `in_proj_qkv`, `in_proj_z`, `out_proj` (GDN); `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention); `gate_proj`, `up_proj`, `down_proj` (MLP); `lm_head` | 4-bit weight, fp16 compute | 99.87% of params. These are dense GEMVs with no feedback path — quantization error is local to the projection output, not accumulated across tokens. KleidiAI provides 109 A720-usable INT4/i8mm GEMV micro-kernels for these (FINDINGS.md §3.3). |
| **FP16 carve-out** | `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias` (GDN gates); `conv1d` weights; all RMSNorm weights; `embed_tokens` | fp16 | Gates directly control the recurrence: `g = -A_log.exp() × softplus(a + dt_bias)` — errors in `A_log` or `a` compound through every token via the exponential. Conv1d gates QKV before the delta rule. At 0.13% of params, the bandwidth cost of keeping fp16 is ~0.024 GiB/token — rounding error. `embed_tokens` is a lookup table, not a GEMV at decode (one row read = 5 KB), so quantizing it gains nothing and risks semantic precision loss. |
| **fp32 floor** | Recurrent state (`S` matrices); decay accumulator; attention softmax | fp32 | `mamba_ssm_dtype = 'float32'` in config — the rank-1 delta-rule updates accumulate over the full sequence and would lose precision in fp16/bf16. A decay of 0.5 compounded over 64 steps is ~5e-20, which underflows fp16 entirely (FINDINGS.md §4). The decay accumulator is kept fp32 even when the surrounding state could theoretically be narrowed. This is not a decision — it is a constraint imposed by the model definition. |

### Deployment path

1. **Primary:** KleidiAI INT4 GEMV micro-kernels (109 A720-usable, FINDINGS.md §3.3) for batch=1
   decode. These are the only production-quality INT4/i8mm GEMV library for non-SME Armv9.
2. **Prefill:** KleidiAI INT8/i8mm GEMM kernels for the chunkwise matmuls (batch>1 within chunks).
3. **Carve-outs:** Standard fp16 NEON/SVE matmuls for the tiny gate projections.
4. **NPU path:** When the Orion O6 is available, the same per-tensor policy applies to NOE export
   (`ob-onz`): NPU-resident subgraphs use INT8 (NOE's INT4 path quality is unverified), with the
   same FP16 carve-outs and fp32 state.

### Validation protocol

The x86/CUDA reference (`ob-aqv`) is the correctness oracle. Quantization quality is validated by:

1. **Token-level cosine similarity** between INT4-dequantized and fp16 reference outputs at each
   projection, for prompt lengths 4K, 32K, 128K.
2. **Perplexity** on a held-out eval set, reported with the minimum reportable difference rule from
   METRICS.md §7.
3. **Long-sequence drift test:** decode 256 tokens at 32K context, compare output-token
   distribution to fp16 reference at tokens 1, 64, 128, 256 — this catches accumulation errors that
   short-sequence tests miss.

If any tier-1 tensor fails the acceptance threshold (perplexity within 2× the METRICS.md §7
minimum reportable difference, or cosine > 0.995 at all sampled positions), the fallback is to
drop that tensor (or layer-class) to INT8, not to abandon quantization.

## Alternatives considered

| Option | Why not | What would change our mind |
|---|---|---|
| **INT8 weights only (W8A16)** | Only 2× traffic reduction (3.78 GiB → ~26 tok/s). Leaves a 2× performance gap on the table vs INT4 for a model where the projections are dense GEMVs — exactly the operation INT4 GEMV handles well. | If INT4 quality regression proves unacceptable even with per-layer fallback, INT8 becomes the ceiling. It is the automatic fallback before any tensor is allowed to stay fp16. |
| **fp16 everywhere (no quantization)** | 7.56 GiB/token → ~13 tok/s ceiling at 100 GB/s. This is the baseline, not the target. Acceptable for correctness validation but not for the headline number or a credible Physical AI demo. | Never — fp16 is the reference, not the deployment. |
| **INT4 for gates too** | Gates are 0.2% of params and feed directly into the exponential decay path. An INT4 error of ±0.1 in `a` produces exp(±0.1) ≈ ±10% in the decay factor — compounding over 64 tokens means the state could drift arbitrarily. The bandwidth saving is ~0.005 GiB/token. | Only if empirical testing shows the gated scan is insensitive — but the theoretical argument is strong enough that we should not spend experiment time confirming it. |
| **bf16 recurrent state** | `mamba_ssm_dtype` is explicitly fp32 in config. bf16 halves the 48 MiB state but saves only ~2.4% of INT4 decode traffic (METRICS.md §9 appendix). The state accumulates rank-1 updates over the full sequence — bf16 has only 8 mantissa bits vs fp16's 11, making accumulation error worse. | Re-evaluated under `ob-8qt.4` as a memory optimization (halving resident state for long-context fitting), not a throughput one. Lower priority than INT4 weights. |
| **fp16 recurrent state** | Same as bf16 but even worse mantissa (10 bits). fp16 cannot represent decay factors below ~6e-8 — the recurrence would silently saturate to zero or one. | Never. This is the one tensor where precision is a hard constraint, not a trade-off. |
| **Per-layer mixed INT4/INT8 (sensitivity-ranked)** | Requires a calibration pass we cannot run without a loaded model and the x86 oracle. Adds complexity for marginal gain when the FP16 carve-outs already protect the sensitive tensors. | If the validation protocol (§) reveals specific INT4-sensitive large projections, a per-layer sensitivity ranking becomes worthwhile. This is the escalation path, not the starting point. |

## Consequences

**Accepted costs.** INT4 introduces accuracy risk in 99.87% of weights. The validation protocol
above is mandatory before any INT4 result is reported as a headline number. INT4 dequantization
adds a small compute overhead per GEMV, but decode is bandwidth-bound (0.25 FLOP/byte,
METRICS.md §9), so the extra FLOPs are absorbed by slack compute capacity.

**Follow-on work.**
- `ob-onz` — NPU-resident subgraph export with the same per-tensor policy (unblocked by this ADR).
- `ob-8qt.4` — bf16 state variant, re-scoped as a memory optimization (not throughput).
- `ob-aqv` — x86/CUDA reference inference must be running before INT4 validation can proceed.
- New bead suggested: implement the INT4 quantization calibration pipeline once `ob-aqv` is live.

**Reversal cost.** Low-medium. The per-tensor assignment is a policy file (PLAN.md line 388:
`quant/` directory), not baked into the kernel code. Switching a tier from INT4 to INT8 is a
config change plus a re-calibration. Switching the entire model back to fp16 is a config flag.
The one irreversible decision is the KleidiAI dependency for INT4 GEMV — but KleidiAI is
upstream-able and replaceable with a custom kernel if needed, so even that is not truly locked in.

**Trigger for re-evaluation.** If the validation protocol shows perplexity regression beyond the
METRICS.md §7 minimum reportable difference at any sampled position, the specific failing tensor(s)
drop to INT8. If INT8 also fails for a tensor, it joins the FP16 carve-out list. The carve-out
list is expected to stay small (gates, norms, conv, embeddings) — if it grows past 2% of params,
that indicates a structural problem with INT4 on this model, not a per-tensor fix.
