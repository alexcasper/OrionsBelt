# ob-mrd.9 — ggml op-enum and CPU-backend audit for Gated DeltaNet

**Bead:** `ob-mrd.9` · **Date:** 2026-08-06
**Type:** READ-ONLY research (source grep of upstream `ggml-org/llama.cpp` at HEAD).
**Companion doc:** [`ob-8qt.6-llama-cpp-gdn-support.md`](ob-8qt.6-llama-cpp-gdn-support.md)
answers the higher-level "is GDN supported?" question with PR/issue/user evidence.
This doc answers ob-mrd.9's specific ask: *grep the ggml op enum and CPU backend for
a delta-rule/gated-scan primitive*, and check for related Mamba2/linear-attention
infrastructure.

---

## VERDICT (one line)

**`GGML_OP_GATED_DELTA_NET` is a first-class op in the ggml tensor library, with a
dedicated CPU-backend implementation (~200 lines).** It is not decomposed into
generic matmul/add/mul primitives — the delta-rule recurrence is a single fused op.
However, the implementation uses generic vectorized helpers (`ggml_vec_dot_f32`,
`ggml_vec_mad_f32`), not architecture-specific micro-kernels (no NEON/SVE/i8mm
tuning for the GDN op itself). This confirms upstream support and sharpens the
project's contribution claim: the project's hand-written SVE2/i8mm kernels would
be an optimization beyond what llama.cpp ships today.

---

## 1. The ggml op enum — GDN and the recurrence family

Source: `ggml/include/ggml.h`, enum `ggml_op` at upstream HEAD (2026-08-06).

The op enum contains **five distinct recurrence/gated-attention primitives**,
all with first-class status (not user-defined custom ops):

| Op | Purpose | GDN relevance |
|---|---|---|
| `GGML_OP_GATED_DELTA_NET` | **Gated DeltaNet** scan — the exact architecture | **Direct hit** |
| `GGML_OP_SSM_SCAN` | Mamba2's SSM state-space scan | Architecturally analogous (sequential state update) |
| `GGML_OP_SSM_CONV` | Mamba2's causal conv1d | Analogous to GDN's causal conv |
| `GGML_OP_GATED_LINEAR_ATTN` | Gated linear attention (GLA) | Another gated-recurrent attention variant |
| `GGML_OP_RWKV_WKV7` | RWKV v7 time-mix | Yet another recurrent attention family |

All five are dispatched in the CPU backend (`ggml/src/ggml-cpu/ggml-cpu.c`):

```c
case GGML_OP_SSM_CONV:        // line 2011
case GGML_OP_SSM_SCAN:        // line 2015
case GGML_OP_GATED_LINEAR_ATTN: // line 2047
case GGML_OP_RWKV_WKV7:       // line 2051
case GGML_OP_GATED_DELTA_NET: // line 2059
```

The `GATED_DELTA_NET` op was added as part of the Qwen3.5 / Qwen3-Next support
wave (PRs #16095, #19139 — see the companion doc). Its presence in the enum means
llama.cpp recognizes GDN as a named architecture, not a graph of generic ops.

---

## 2. CPU-backend implementation — the delta-rule recurrence

Source: `ggml/src/ggml-cpu/ops.cpp`, lines ~10734–10945.

### 2a. Op signature (from `ggml.h`)

```c
struct ggml_tensor * ggml_gated_delta_net(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,     // [S_k, H_k, n_tokens, n_seqs]
    struct ggml_tensor  * k,     // [S_k, H_k, n_tokens, n_seqs]
    struct ggml_tensor  * v,     // [S_v, H_v, n_tokens, n_seqs]
    struct ggml_tensor  * g,     // [1|S_v, H_v, n_tokens, n_seqs] — gate/decay
    struct ggml_tensor  * beta,  // [1, H_v, n_tokens, n_seqs] — delta-rule beta
    struct ggml_tensor  * state, // [S_v, S_v, H_v, n_seqs] — initial recurrent state s0
    int64_t               K);    // number of state snapshots to retain
```

The `K` parameter supports chunkwise processing: `K=1` keeps only the final state;
`K>1` retains intermediate state snapshots for re-computation. This is the same
chunkwise-recurrence design the project targets (PLAN.md §3.1).

### 2b. The per-token recurrence (the exact delta rule)

The inner loop (`ggml_compute_forward_gated_delta_net_one_chunk`, simplified):

```c
for (int64_t t = 0; t < n_tokens; t++) {
    // 1. Apply gate decay: S *= exp(g)
    if (kda) {  // key-dependent decay
        for (i = 0; i < S_v; i++) delta[i] = expf(g_d[i]);
        for (j = 0; j < S_v; j++)
            ggml_vec_mul_f32(S_v, &s_out[j*S_v], &s_out[j*S_v], delta);
    } else {
        ggml_vec_scale_f32(S_v * S_v, s_out, expf(g_d[0]));
    }

    // 2. Delta rule: delta[j] = (v[j] - S[:,j]·k) * beta
    for (j = 0; j < S_v; j++) {
        float sum = 0;
        ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j*S_v], 0, k_d, 0, 1);
        delta[j] = (v_d[j] - sum) * beta_val;
    }

    // 3. State update: S[:,j] += k * delta[j]  (outer product)
    for (j = 0; j < S_v; j++)
        ggml_vec_mad_f32(S_v, &s_out[j*S_v], k_d, delta[j]);

    // 4. Attention output: attn[j] = (S[:,j]·q) * scale
    for (j = 0; j < S_v; j++) {
        float sum = 0;
        ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j*S_v], 0, q_d, 0, 1);
        attn_data[j] = sum * scale;  // scale = 1/sqrt(S_v)
    }
}
```

This is **exactly the delta rule** the project's `gdn_sve.c` implements
(FINDINGS.md §4): gate decay → error correction → state update → query projection.

### 2c. Vectorized primitives (the ARM-relevant question)

The recurrence body uses four generic ggml vector helpers:

| Helper | Operation | ARM dispatch |
|---|---|---|
| `ggml_vec_dot_f32` | dot product (S_v wide) | NEON via `vdotq`/`vdotq_lane` on ARMv8.2+ |
| `ggml_vec_mul_f32` | elementwise multiply | NEON `vmulq` |
| `ggml_vec_mad_f32` | multiply-accumulate | NEON `vfmaq` (fused) |
| `ggml_vec_scale_f32` | scalar broadcast multiply | NEON `vmulq_n` |

These helpers auto-dispatch to NEON on ARM targets, but they are **generic
floating-point** implementations — there is **no i8mm, no SVE2, no predicated-tail
handling** specific to the GDN op. The helpers are shared across all ops that need
dot products (attention, MLP, etc.), so they are tuned for the common case (large
GEMM tiles), not for GDN's small per-head dimensions (S_v = 128, 16 heads).

**This is the gap the project fills.** The project's hand-written kernels
(`gdn_sve.c`) use SVE2 predication to handle the S_v = 128 tail without scalar
fallback, and the i8mm path (ob-8qt.2) would use INT8 dot-product for the delta-rule
matmuls. Neither exists in upstream llama.cpp.

### 2d. Threading model

The op is parallelised across the `(H × n_seqs)` head×sequence dimension:

```c
int nth_scaled = nth * 4;  // 4x chunks per thread
int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
```

Each thread processes a contiguous range of (head, sequence) pairs, with a
per-thread work buffer of `S_v` (or `S_v²` for K>1) floats. This is coarse-grained
parallelism — the sequential token loop within each head is **not** parallelised,
which is structurally correct (the recurrence is inherently sequential in `t`).

---

## 3. Related recurrence ops — Mamba2 infrastructure

The `SSM_SCAN` op (Mamba2's SSM scan) implements the structurally analogous
state-space recurrence:

```
x[t] = A·x[t-1] + B·u[t]     // state update
y[t] = C·x[t]                // output projection
```

vs. GDN's delta rule:

```
S[t] = diag(exp(g[t]))·S[t-1] + k[t]·δ[t]ᵀ   // state update
y[t] = S[t]·q[t]                              // output projection
```

Both are sequential, stateful, and use the same "per-token inner loop" structure.
Mamba2 support (SSM_SCAN + SSM_CONV) landed in llama.cpp before GDN (Mamba2 PRs
#15046, #16095), and GDN's op reuses the same CPU-backend infrastructure pattern
(per-thread state buffers, chunked parallelism, f32-only implementation).

**Implication for the project's claim:** the "no existing CPU path handles this
architecture" framing needs to be narrowed. The accurate statement is: **generic
CPU inference engines (llama.cpp) already run GDN end-to-end via first-class ops,
but the implementations use generic floating-point code with no Arm-specific
micro-kernel optimization.** The project's contribution is the optimised
micro-kernels, not the existence of a CPU path.

---

## 4. Cross-reference table: what each tool provides for GDN

| Tool | GDN op exists? | Dedicated CPU impl? | Arch-specific (NEON/SVE/i8mm)? | Recurrence works? |
|---|---|---|---|---|
| NOE Compiler | ✅ (lowerable ops) | NPU only | N/A | ❌ `Scan`/runtime-`Loop` rejected |
| KleidiAI | ❌ | ❌ | ✅ (109 matmul kernels) | ❌ no scan primitive |
| **llama.cpp / ggml** | **✅ `GATED_DELTA_NET`** | **✅ ops.cpp ~200 LOC** | **❌ generic `vec_*` only** | **✅ token-to-token** |
| This project (planned) | ✅ | ✅ `gdn_sve.c` | ✅ SVE2/i8mm | ✅ |

---

## 5. Implications for the write-up

1. **The "no tooling support" claim must be qualified.** The companion doc
   (§4) already established that llama.cpp is the "exception that proves the rule."
   This op-enum analysis strengthens that finding: it's not just that llama.cpp
   *runs* GDN — it has a *named, first-class op* for it. The write-up should say
   "ahead-of-time accelerator toolchains (NOE) and fixed kernel libraries
   (KleidiAI) lack GDN support; general-purpose CPU inference engines (llama.cpp)
   support it but with generic, unoptimised code."

2. **The project's contribution is architecture-specific optimization, not
   enabling.** The project does not make GDN runnable on Arm — llama.cpp already
   does that. The project makes it *fast* on Arm via hand-written SVE2/i8mm
   micro-kernels that llama.cpp lacks. This is a more defensible and more honest
   framing than "we made it work."

3. **llama.cpp as the baseline to beat.** A llama.cpp Q4_K_M run on any Arm
   device is a legitimate "unoptimised off-the-shelf" baseline. The project's
   kernels should be benchmarked against it to quantify the optimization gain.

4. **Upstreaming path.** The `GGML_OP_GATED_DELTA_NET` op already exists in the
   enum and has a CPU dispatch path. The project's SVE2/i8mm kernels could be
   upstreamed as an optimised backend for this op — a concrete contribution path
   that is more compelling than "we wrote standalone benchmarks."

---

## 6. Sources

All source code fetched from upstream `ggml-org/llama.cpp` at HEAD (2026-08-06):

- **Op enum:** `ggml/include/ggml.h`, enum `ggml_op`
- **Op signature:** `ggml/include/ggml.h`, function `ggml_gated_delta_net()`
- **CPU dispatch:** `ggml/src/ggml-cpu/ggml-cpu.c`, lines 2059, 2258, 2984
- **CPU implementation:** `ggml/src/ggml-cpu/ops.cpp`, lines 10734–10945
  - `ggml_compute_forward_gated_delta_net_one_chunk()` — per-token recurrence
  - `ggml_compute_forward_gated_delta_net_f32()` — thread-level chunking
  - `ggml_compute_forward_gated_delta_net()` — dtype dispatch
- **Work-buffer allocation:** `ggml/src/ggml-cpu/ggml-cpu.c`, line 2984
- **SSM_SCAN implementation:** `ggml/src/ggml-cpu/ops.cpp`, lines 9627–9853
- **SSM_CONV dispatch:** `ggml/src/ggml-cpu/ggml-cpu.c`, line 2011
- **Op declarations:** `ggml/src/ggml-cpu/ops.h`, lines 95–107

PR/issue sources: see companion doc [`ob-8qt.6-llama-cpp-gdn-support.md`](ob-8qt.6-llama-cpp-gdn-support.md) §7.
