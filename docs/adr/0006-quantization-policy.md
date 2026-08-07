# ADR 0006: Quantize weights to INT4, carve out recurrent state and gates

- **Status:** Accepted
- **Date:** 2026-08-02
- **Bead:** `ob-qpa`
- **Depends on:** [ADR 0003](./0003-model-checkpoint-selection.md) (checkpoint dimensions), [ob-37v](../FINDINGS.md#6) (confirmed state layout), [METRICS.md §9](../METRICS.md) (traffic breakdown)
- **Policy document:** [`docs/QUANTIZATION_POLICY.md`](../QUANTIZATION_POLICY.md)

## Context

Weight streaming is 95–99% of decode bandwidth on this class of hardware
(METRICS.md §9 appendix). At the O6's 100 GB/s, FP16 weights for the 4B checkpoint
(7.5 GiB) impose a decode ceiling of ~12.5 tok/s. INT4 weights (1.9 GiB) raise that
to ~50 tok/s — a 4× improvement from weight quantization alone. No kernel rewrite
of the recurrent scan changes this, because the scan's 96 MiB of state traffic is
~5% of the INT4-weight total.

The question is not *whether* to quantize weights, but *what must be carved out*
to preserve model quality — and specifically whether the GDN recurrent state and
gating signals can survive quantization.

## Decision

**Quantize all projection weights to INT4** (INT8 as a conservative fallback).
**Keep the recurrent state in FP32 and the decay gate parameters in FP16/FP32.**

| Tensor | Precision | Why |
|---|---|---|
| All projection weights | **INT4** | Dominant decode-bandwidth term; 4× reduction |
| Recurrent state S | **FP32** | Accumulates error across sequence; 48 MiB total (cheap to keep) |
| Decay gate (A_log, dt_bias) | **FP32** | Exponential decay compounds multiplicatively |
| Beta gate (b) | **FP16** | Bounded sigmoid, recomputed per token |
| KV cache | **FP16** | Standard; INT8 is a future option gated on oracle |

See [`docs/QUANTIZATION_POLICY.md`](../QUANTIZATION_POLICY.md) for the full
per-tensor table, KleidiAI mapping, and validation gate specification.

## Why the recurrent state is a carve-out

The delta-rule update `S_t = S_{t-1} * exp(g_t) + k_t ⊗ (v_t - S_{t-1}@k_t) * beta_t`
feeds S back through every token. A quantization error in S is not local — it
propagates forward and compounds, because the same corrupted state is used to
compute every subsequent retrieval `S @ q`. This is fundamentally different from a
KV-cache entry, which is read once and never modified. FINDINGS.md §4 documents
that "the decay accumulator is fp32 even when surrounding state is fp16," and the
upstream code (`mamba_ssm_dtype: "float32"` in config) confirms the model itself
keeps the state in FP32.

The cost of this carve-out is negligible: 48 MiB of state traffic against 1.9 GiB
of INT4 weight traffic is ~5% — well within the margin where preserving model
quality is worth it.

## Alternatives considered

| Option | Why not |
|---|---|
| **INT8 weights everywhere, no carve-outs** | 2× decode speedup instead of 4×; still viable if INT4 fails the oracle |
| **BF16 recurrent state** | Saves ~2–3% of decode traffic at INT4 weights — negligible throughput gain. Worth doing for memory footprint (ob-8qt.4), not for speed. **And on A76-class cores it is the wrong narrow format — see the amendment below.** |
| **Quantize the state (INT8)** | High risk of accuracy collapse on long-context retrieval; the state is the model's memory, and corrupting it is not recoverable. |
| **No weight quantization (FP16 baseline)** | Leaves ~4× decode performance on the table — the single largest available optimization. |

## Consequences

**Accepted:** INT4 weight quantization requires KleidiAI INT4/i8mm GEMV micro-kernels
(ob-8qt.2) or an equivalent dequantize-on-the-fly path. The correctness oracle
(ob-3uh) gates every quantized configuration before its numbers enter a results table.

**Reversal cost:** Low. Each tensor's precision is independently configurable. If
INT4 fails the oracle, the policy degrades gracefully: INT8 weights with FP32 state
is still a 2× speedup, and individual layers can be kept in FP16 if per-layer
sensitivity analysis identifies outliers.

---

## Amendment (2026-08-04): the narrow format is core-class-dependent

Measured on RK3588 (Cortex-A76 big / A55 little) — the first cross-device data for
the narrow-format question, since everything prior was Cortex-A57 only. The RK3588
has **hardware fp16 (`asimdhp`) but no hardware bf16** (that needs Armv8.6-A), so
it exercises the software bf16 conversion path on a fast core, which is the case
most of the installed Armv8.2 base is in. Single-threaded, governor `performance`,
thermals flat (bead `ob-8qt.4`, raw data in
`results/raw/rk3588-t4-clean.csv`).

| cumdecay output format | Jetson A57 | RK3588 A76 | RK3588 A55 |
|---|---:|---:|---:|
| fp16 | 1.45× | **1.55×** | 1.53× |
| bf16 | 1.42× | **1.08×** | 1.40× |

**Prefer FP16 over BF16 for narrowed state and output on A76-class cores.** On the
A76, software bf16 conversion drops the gain to 1.10× at 4B and goes *negative* at
0.8B (129.5 µs against 124.0 fp32, spreads 1.1%/2.6% — real, not noise): the
integer-NEON round-to-nearest-even conversion costs about what the saved bytes buy.
On the slower A55 and A57 the bandwidth saving still dominates and bf16 keeps 1.42–1.47×.

So the fastest core in the fleet is the one where bf16 stops paying. BF16 is worth
it only with **hardware** support — Armv8.6-A and later, which includes the O6's
Cortex-A720 — or on a core slow enough to amortise the conversion.

This does not change the headline decision (INT4 weights, FP32 recurrent state,
FP16 gates); it constrains the *implementation* of the narrow-format carve-out.
A dispatcher that picks a narrow format must key off the ISA at runtime rather
than compiling one choice in — `src/orionsbelt/engines/cpu/isa_detect.py` already
reports `bf16` and `asimdhp` for exactly this.

Unchanged and still the dominant lever: weight quantization, at ~95–99% of decode
bandwidth. Narrow state remains a memory-footprint optimisation, not a throughput one.
