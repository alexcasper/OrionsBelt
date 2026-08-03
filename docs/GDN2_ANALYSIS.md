# GDN-2 paper analysis: decoupled gating, RULER claims, edge-scale implications

**Bead:** `ob-9lm.1` · **Paper:** GatedDeltaNet-2, Hatamizadeh, Choi, Kautz — [arXiv 2605.22791](https://arxiv.org/abs/2605.22791)
**Read:** 2026-08-02 from pi5-r5 · **Config:** 1.3B params, 100B FineWeb-Edu tokens

This analysis fills gaps not covered by [ADR 0001](./adr/0001-gdn2-decoupled-gating-hypothesis.md)
or [FINDINGS.md §1](./FINDINGS.md). It documents the GDN-2 paper's key claims and assesses
their relevance to this project's edge-scale comparison plan (PLAN.md §4/E8).

---

## 1. The core innovation: decoupled erase and write gates

### What GDN (Qwen3.5) does

From the ob-37v audit of `modeling_qwen3_5.py`, the GDN update is:

```
S_t = S_{t-1} * exp(g_t)                          # gated decay (scalar per value head)
delta = (v_t - S_{t-1} @ k_t) * beta_t            # prediction error × write gate
S_t += k_t ⊗ delta                                # delta-rule correction
```

Here `g_t` (from `A_log`, `dt_bias`, input `a`) controls **both** erasure (how much old
state to forget) and write strength (via `beta_t = sigmoid(b_t)`). The decay `exp(g_t)`
applies uniformly across the key dimension — a single scalar per value head that scales
the entire state matrix.

### What GDN-2 changes

GDN-2 **separates the erase and write roles** into two independent channel-wise gates:

- **`b_t`** — channel-wise **erase gate**: controls how much old content to forget,
  applied per-channel (per key dimension) rather than as a single scalar.
- **`w_t`** — channel-wise **write gate**: controls how much new content to commit,
  also per-channel.

The update becomes (conceptually):

```
S_t = b_t ⊙ S_{t-1} + w_t ⊗ (v_t - S_{t-1} @ k_t)   # decoupled erase + write
```

Where `b_t` and `w_t` are vectors of length `d_k` (the key dimension), not scalars.

### The reduction hierarchy

The paper states:
- **GDN-2 → KDA** when `b_t = w_t` (both gates collapse to the same scalar) — Kimi Delta Attention
- **KDA → GDN** when the decay also collapses — standard Gated DeltaNet

So GDN-2 **strictly generalizes** both prior architectures. This is important for the
project: it means a GDN-2 implementation can reproduce GDN behavior as a special case,
making A/B comparison clean (set `b_t = w_t = scalar` → GDN baseline).

### Why this matters for memory access patterns

In GDN, the decay `exp(g_t)` is a scalar multiply — one FLOP per state element. In GDN-2,
`b_t` and `w_t` are per-channel vectors — still element-wise, but with `d_k` independent
gates instead of 1. The **memory traffic is identical**: both read and write the full
state matrix. The additional cost is `O(d_k)` extra parameter traffic for the gate vectors,
which is negligible against the `O(d_k × d_v)` state matrix.

**Implication for this project:** GDN-2 does NOT change the bandwidth-bound analysis
(METRICS.md §9). The 0.25 FLOP/byte arithmetic intensity holds for GDN-2 as well, because
the state access pattern is unchanged. The decode-throughput ceiling is the same.

---

## 2. RULER multi-key retrieval claims

The paper's headline result: **"its advantage is most pronounced on long-context RULER
needle-in-haystack benchmarks, where it improves the evaluated multi-key retrieval setting."**

### What this means concretely

RULER multi-key retrieval tests whether a model can retrieve multiple specific key-value
pairs from a long context — exactly the evaluation we built the prompt corpus for (ob-del,
`bench/prompts/ruler_*.txt`). GDN-2 claims to outperform GDN on this specific task.

### Why decoupled gates might help retrieval

The hypothesis (ADR 0001): a single scalar gate forces the model to forget old information
uniformly when writing new information. If two keys map to overlapping state regions,
writing the second partially erases the first. Separate erase/write gates allow the model
to selectively protect important stored associations while still incorporating new
information — preserving retrieval accuracy under interference.

### Scale of the improvement

The paper does **not quantify** the RULER improvement in the abstract ("improves the
evaluated multi-key retrieval setting" is qualitative). This aligns with PLAN.md §2.3's
note that "GDN-2's RULER gains are real but not quantified in the abstract, so they must
be described qualitatively until the full paper is read."

**Implication for this project:** the GDN-2 comparison (E8, option a: microbenchmark
GDN-2 vs GDN gating on-device) can test this claim using our committed RULER prompt corpus.
A negative result ("GDN-2's multi-key advantage does not hold at edge scale") is publishable
per PLAN.md §4/E8: "Negative results are publishable here."

---

## 3. Edge-scale implications

### Does GDN-2's advantage transfer to edge silicon?

The paper trains at **1.3B parameters on 100B tokens** — a datacenter-scale training
configuration. The question for this project is whether the architectural advantage
(better multi-key retrieval from decoupled gating) is:

1. **An intrinsic property of the architecture** that holds at any scale → testable on edge
2. **A training-scale effect** that requires 1.3B+ params and 100B+ tokens to emerge → not
   testable on edge without retraining

The paper's claim that GDN-2 "remains strong in both recurrent and hybrid settings"
suggests the advantage is architectural, not training-scale-dependent. But this is about
inference-time evaluation, not training.

### What we can and cannot test on edge

| Testable on edge (our device fleet) | NOT testable on edge |
|---|---|
| **Decode throughput** — same state access pattern, same bandwidth ceiling | **Training quality** — requires GPU + 100B tokens |
| **Memory footprint** — identical O(1) state, same formula | **RULER absolute scores** — requires a trained GDN-2 checkpoint |
| **Multi-key retrieval at inference** — if a trained checkpoint exists | **Training convergence** — requires training infrastructure |
| **GDN vs GDN-2 kernel comparison** — same hardware, same workloads | **Scaling laws** — requires multiple model sizes |

**Implication:** The project's E8 plan (option a: microbenchmark GDN-2 vs GDN gating) is
the right edge-scale test. It measures the **inference-time** properties (throughput,
memory, retrieval accuracy with a pre-trained checkpoint) that are architecturally
determined, not the training-scale properties.

---

## 4. Implications for the project's GDN-2 comparison plan

### Option (a): Microbenchmark GDN-2 vs GDN gating (recommended, PLAN.md §4/E8)

This is now well-supported by the paper analysis:
- The GDN-2 update is a **strict generalization** of GDN — a GDN-2 implementation can
  reproduce GDN as a special case, making A/B comparison clean.
- The **memory traffic is identical** — the bandwidth-bound analysis holds for both.
- Our **RULER prompt corpus** (ob-del) tests exactly the multi-key retrieval task where
  GDN-2 claims improvement.
- The comparison can use the **same Backend ABC and harness** (ob-ljh) — just with a
  different gating mechanism in the model's forward pass.

### What we need for the comparison

1. **A GDN-2 model or layer implementation** — the paper says "Code is available" (NVLabs
   repo). ob-y3f (P3) tracks cloning and smoke-testing this on x86.
2. **A trained GDN-2 checkpoint at a comparable scale** — the paper uses 1.3B, which is
   between our 0.8B and 4B Qwen3.5 checkpoints.
3. **Our RULER prompt corpus** — already committed (ob-del).

### The honest framing for the submission

If we complete the comparison: "We tested whether GDN-2's decoupled-gating retrieval
advantage holds at edge scale, using the same device fleet and prompt corpus as our GDN
benchmarks."

If we don't complete it (descope under PLAN.md §7): "GDN-2's decoupled gating is a strict
generalization of GDN, with identical memory access patterns and bandwidth characteristics.
The multi-key retrieval advantage claimed at 1.3B scale is architecturally motivated but
not independently verified at edge scale in this submission."

Either framing is honest and adds value to the submission.

---

## 5. Technical details for implementation

### The chunkwise WY algorithm with channel-wise decay

The paper describes "a chunkwise WY algorithm with channel-wise decay absorbed into
asymmetric erase factors." This is the prefill-path analog of GDN's `chunk_gated_delta_rule`
(ob-37v, `modeling_qwen3_5.py` line ~260), but with:

- **Asymmetric erase factors**: instead of a single cumulative decay `g.cumsum()`, the
  decay is applied per-channel, producing a matrix of decay factors rather than a vector.
- **Gate-aware backward pass**: the separate gates require a modified gradient computation
  that the paper derives.

**Implication for kernel work:** GDN-2's chunkwise scan has the same shape as GDN's
(sequential across chunks, dense within chunks), so our SVE/NEON kernels (ob-8qt.3) and
the NPU op-coverage analysis (ob-t3b.1) apply with minor modifications for the per-channel
gates.

### State precision (confirmed)

The paper doesn't specify state precision, but our quantization policy (ob-qpa,
QUANTIZATION_POLICY.md) already establishes the principle: recurrent state stays FP32
because quantization errors compound across the sequence. This applies equally to GDN-2's
state, since the access pattern is identical.

---

## 6. Summary table for the write-up

| Property | GDN (Qwen3.5) | GDN-2 |
|---|---|---|
| Erase mechanism | Scalar decay `exp(g_t)` per value head | Channel-wise `b_t` per key dim |
| Write mechanism | Scalar gate `beta_t` per value head | Channel-wise `w_t` per key dim |
| State shape | `(n_v_heads, d_k, d_v)` | Same — identical |
| Memory traffic | `d_k × d_v × n_v_heads` per layer | Same — identical |
| Bandwidth-bound? | Yes, ~0.25 FLOP/byte | Yes, same intensity |
| Generalizes? | Special case of GDN-2 (decay collapses) | Generalizes GDN + KDA |
| RULER multi-key | Baseline | Claimed improvement (qualitative) |
| Trained at | Shipping in Qwen3.5 (4B/0.8B) | 1.3B on 100B tokens (research) |
| Edge-relevant? | Yes — our primary architecture | Yes — same bandwidth, testable retrieval |
