# ob-mrd.16 — ONNX Runtime CPU EP audit for Gated DeltaNet

**Bead:** `ob-mrd.16` · **Date:** 2026-08-08
**Device:** rk3588-t4 (RK3588, Cortex-A76)
**ORT version:** 1.28.0 (pip wheel, aarch64)
**Companion doc:** [`ob-mrd.9-ggml-op-enum-gdn-audit.md`](ob-mrd.9-ggml-op-enum-gdn-audit.md) (llama.cpp audit)

---

## VERDICT (one line)

**ONNX Runtime's CPU EP can execute GDN's delta-rule recurrence via the generic `Loop` op, and produces correct results — but there is no dedicated GDN primitive, and the Loop body is evaluated as a graph per iteration with no fusion, making it a correctness reference rather than a competitive inference path.**

---

## 1. Does ONNX have a GDN primitive?

No. ONNX opset 27 contains no `GatedDeltaNet`, `DeltaRule`, or `GatedScan` op.
The recurrence-related ops available are:

| Op | Purpose | GDN relevance |
|---|---|---|
| `Loop` | Generic iterative computation with loop-carried state | Can express GDN's per-token recurrence |
| `Scan` | Slide over an input sequence, carry state, accumulate outputs | Alternative to Loop; ORT has scan-input type-inference issues with multiple scan inputs |
| `CumSum` | Cumulative sum | Used for gated decay in some formulations |
| `LSTM` / `GRU` | Fixed-form recurrent cells | Wrong recurrence structure (no delta rule) |

This contrasts with llama.cpp, which has `GGML_OP_GATED_DELTA_NET` as a first-class op (§ob-mrd.9).

## 2. Can GDN's recurrence be expressed via Loop?

**Yes — verified empirically.** A minimal ONNX model was constructed that implements
the full delta-rule recurrence inside a `Loop` body:

```
for each token t:
    S *= exp(g[t])                      # gate decay
    delta = (v[t] - S @ k[t]) * beta[t] # error correction
    S += outer(k[t], delta)             # state update (rank-1)
    attn[t] = (S @ q[t]) * scale        # output projection
```

The body uses 7 ops per iteration: `Gather`, `Exp`, `Mul`, `MatMul`, `Sub`,
`Transpose`, `Add`. State `S` [V,V] is the sole loop-carried variable. Per-token
data (q,k,v,g,beta) is baked as body initializers and indexed via `Gather(iter_num)`.

**Implementation note:** ORT 1.28 has a type-inference limitation with `Loop`
when using multiple scan inputs (5 scan inputs with mixed scalar/vector shapes
triggers "output index out of range" during type inference). The workaround is
to bake data as initializers and use Gather — functionally equivalent, just less
flexible for dynamic inputs. This is an ORT implementation issue, not an ONNX
spec limitation.

## 3. Correctness

ORT output matches a NumPy reference implementation to **2.3 × 10⁻⁷ relative error**
(float32 precision floor). Verified at V=128, seq_len=8 and 32.

## 4. Performance (single-head scan only)

| Metric | Value |
|--------|-------|
| Head dim (V) | 128 |
| Per-token latency | ~49 µs |
| Per-sequence (8 tokens) | ~417 µs |
| Per-sequence (32 tokens) | ~1,574 µs |
| Throughput | ~20,300 tok/s |

**Context:** This measures ONE head's recurrence only. The real Qwen3.5-4B model
has 16 key heads × 24 GDN layers, plus projection matmuls, conv, FFN, and
full-attention layers. The scan is ~2% of total decode time in the C benchmark
(§4). ORT's per-iteration Loop overhead (graph evaluation, kernel dispatch) is
~49 µs vs ~3 µs for the project's fused C kernel — a **16× overhead** for the
same computation.

For a full-model ORT path, the projection matmuls (which dominate decode time)
would also run through ORT's generic MatMul kernel without Arm-specific tuning.
ORT does use OpenMP and basic NEON for MatMul, but lacks the row-sweep GEMV
optimization (§15), INT8/INT4 weight quantization (§16, §26), or fused scan
that this project provides.

## 5. Cross-toolchain comparison

| Tool | GDN op exists? | Recurrence expressible? | Arch-specific kernels? | Runs on CPU? |
|---|---|---|---|---|
| CIX NOE | ✅ (lowerable) | ❌ `Scan`/`Loop` rejected | N/A (NPU only) | ❌ |
| Rockchip RKNN | ✅ (lowerable) | ⚠️ `Scan` accepted, untested on NPU | N/A (NPU only) | ❌ |
| KleidiAI | ❌ | ❌ no scan primitive | ✅ (109 matmul kernels) | ✅ (matmul only) |
| **ONNX Runtime** | **❌** | **✅ via generic `Loop`** | **❌ generic MatMul** | **✅** |
| llama.cpp / ggml | ✅ `GATED_DELTA_NET` | ✅ first-class op | ❌ generic `vec_*` | ✅ |
| **This project** | **✅** | **✅** | **✅ SVE2/NEON/INT8/INT4** | **✅** |

ORT occupies a unique middle ground: more capable than NPU toolchains (can
actually run the recurrence) but less optimized than llama.cpp (no dedicated op,
generic Loop dispatch overhead). The project's contribution fills the gap that
both ORT and llama.cpp leave: architecture-specific micro-kernels for GDN.

## 6. Sources

- ORT 1.28.0 installed via pip on rk3588-t4 (aarch64 manylinux wheel)
- Probe script: [`scripts/ort_gdn_probe.py`](../scripts/ort_gdn_probe.py)
- ONNX opset 27 spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md
- FINDINGS §1-§2a: prior NOE/RKNN ONNX probe work
- FINDINGS §ob-mrd.9: llama.cpp GDN op audit (companion)
