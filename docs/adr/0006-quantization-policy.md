# ADR 0006: Quantization policy — INT4/INT8 weights, FP32 recurrent state and gates

- **Status:** Accepted
- **Date:** 2026-08-02
- **Bead:** `ob-qpa`
- **Deciders:** agent (grounded in `GDN_LAYER_AUDIT.md`, `METRICS.md` appendix, `FINDINGS.md` §4)
- **Supersedes nothing.** Operationalises the precision note in ADR 0003 and the FP32 carve-out first stated in `FINDINGS.md` §4.

## Context

The METRICS.md appendix ("State traffic is real but secondary — weights dominate decode") established that weight streaming is ~95–99% of decode bandwidth at 100 GB/s. INT4 weights cut the 4B model's weight traffic from ~7.5 GiB to ~1.9 GiB per token, moving decode from ~12.5 tok/s to ~50 tok/s — the single largest throughput lever in the project. KleidiAI provides 109 INT4/i8mm GEMV micro-kernels that are directly applicable (FINDINGS.md §3).

The question is not *whether* to quantize weights, but *which tensors must stay high-precision* to avoid accuracy regression. The GDN layer audit (`docs/GDN_LAYER_AUDIT.md`) provides the structural answer.

## Decision

**Quantize projection weights to INT8 (phase 1) or INT4 (phase 2). Keep the recurrent state, decay gate, and write gate in FP32. Keep the KV cache in FP16 (or INT8 if validated).**

### Per-tensor policy

| Tensor | Precision | Rationale | Source |
|---|---|---|---|
| **Projection weights** (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`) | **INT8** (target INT4) | Dominant decode traffic term (~95%+). KleidiAI GEMV micro-kernels apply directly. Must pass correctness oracle (METRICS.md §5). | METRICS.md appendix |
| **FFN weights** (gate, up, down) | **INT8** (target INT4) | Same rationale — dense matmuls, NPU/GPU friendly. | METRICS.md appendix |
| **Embedding** | **FP16** | Tied input/output embedding; quantizing vocabulary projections risks output distribution shift. Low traffic (read once per token). | Standard practice |
| **Recurrent state** (`S`) | **FP32** (no exception) | Fed back through every step. Quantization error accumulates over the sequence — unlike KV cache where error stays local to one position. The audit confirms all GDN math already runs in FP32 (`mamba_ssm_dtype: "float32"`). | GDN_LAYER_AUDIT.md §3, §5 |
| **Decay gate** (`A_log`, `dt_bias`) | **FP32** | Controls exponential decay. A 1% error in the decay factor compounds over 262K tokens. `A_log` is only `num_v_heads` parameters — negligible traffic. | GDN_LAYER_AUDIT.md §7 |
| **Decay gate activation** (`a`, projected from hidden) | **FP16** | Input-dependent but recomputed per token from weights; not stored. FP16 sufficient for the projection output. | Architectural: gate is transient |
| **Write gate activation** (`beta = sigmoid(b)`) | **FP16** | Same: recomputed per token, transient. The sigmoid saturates and is not precision-sensitive in its output range. | Architectural |
| **Conv1D weight** | **INT8** | 4-tap depthwise conv; standard quantization target. Small tensor, but quantizable. | Standard practice |
| **Conv1D state** (decode) | **FP32** | Fed back per token (sliding window). Same accumulation argument as recurrent state. | GDN_LAYER_AUDIT.md §4 |
| **KV cache** (full-attn layers) | **FP16** (try INT8) | Standard for attention KV cache. INT8 KV cache is well-supported; validate via oracle. Not precision-critical for the GDN claim. | Standard practice |
| **Attention Q/K/V/O weights** | **INT8** (target INT4) | Dense matmuls, same as GDN projections. Only 8 layers, but they carry the full attention computation. | Standard practice |
| **RMSNorm weights** | **FP16** | Per-element scale; tiny tensor, high sensitivity. | Standard practice |

### Summary: what stays FP32

Only three categories of tensor are carved out of quantization:

1. **The recurrent state** — `(n_v_heads, d_k, d_v)` per GDN layer. 48 MiB at 4B. This is the project's central O(1) memory claim; keeping it in FP32 costs ~0.6% of total decode traffic at INT4 weights (METRICS.md appendix table).
2. **Learnable gate parameters** (`A_log`, `dt_bias`) — negligible size (~32 + 32 floats per layer), but exponential compounding makes them precision-critical.
3. **Conv1D state** — tiny (4.5 MiB at 4B total), same accumulation argument.

### INT8 vs INT4 phasing

| Phase | Weight precision | Expected decode tok/s (4B, 100 GB/s) | Risk |
|---|---|---|---|
| Baseline (FP16) | fp16 | ~12.5 | — |
| **Phase 1: INT8** | int8 weights, fp32 state/gates | ~25 | Low — INT8 is well-validated for LLM weights |
| **Phase 2: INT4** | int4 weights, fp32 state/gates | ~50 | Medium — INT4 may need per-channel scales and GDN gate/FFN may need higher precision |

Phase 1 (INT8) is the first target because it is lower-risk and KleidiAI's INT8/i8mm kernels are the most mature. Phase 2 (INT4) is attempted only after the correctness oracle validates Phase 1.

## Alternatives considered

| Option | Why not | What would change our mind |
|---|---|---|
| **BF16 recurrent state** | Halves state traffic but only saves ~2–3% of decode bandwidth at INT4 weights (METRICS.md appendix). Risk of accumulation error at 262K context. | If memory footprint (not throughput) is the binding constraint — bf16 state halves the 48 MiB to 24 MiB, which matters for fitting long context alongside weights. Still worth doing as a memory optimization, but not as a throughput one. |
| **Quantize the recurrent state to INT8** | The delta rule feeds state back every step. `S_t = decay × S_{t-1} + rank-1 update`. INT8 rounding per step accumulates over 262K tokens. The correctness oracle (ob-3uh) would need to validate this at long context — and the drift is structural, not fixable by better scales. | Only if the oracle shows <5% perplexity regression at 262K context with INT8 state. Unlikely given the compounding argument. |
| **FP16 everything (no quantization)** | Leaves ~4× decode throughput on the table. The competition rewards Arm-specific optimization (40 pts) — KleidiAI INT4/i8mm is exactly that. | If no Arm-tuned INT4 kernel is available for the target SoC. KleidiAI already provides this for Cortex-A720. |
| **Per-layer mixed precision (some layers INT4, some INT8)** | Adds complexity. The 3:1 GDN:full-attn split means most layers are GDN, and GDN projections are the same shape as attention projections — no reason to differentiate. | If profiling shows specific layers are more sensitive (the correctness oracle could detect this). |

## Consequences

- **KleidiAI integration (ob-8qt.2)** targets the projection weight matrices, not the state or gates. The micro-kernel selection is driven by this policy.
- **NPU export (ob-onz)** exports only the quantizable subgraphs (projections, FFN, attention matmuls). The GDN recurrent scan stays on the CPU in FP32 regardless — this ADR makes that split explicit by precision, not just by operator.
- **Correctness oracle (ob-3uh)** must validate at the longest context length in the sweep (262K), not just short prompts, because that is where state accumulation error would manifest.
- **BF16 state variant (ob-8qt.4)** is reprioritised as a *memory* optimization (halves the 48 MiB state footprint), not a throughput one — consistent with METRICS.md appendix.

**Reversal cost:** Low. Changing precision for a tensor category is a config change + re-export + oracle validation, not an architectural redesign. The carve-out boundary (state and gates stay FP32) is structural and unlikely to change; the weight precision (INT8 vs INT4) is tunable per phase.
