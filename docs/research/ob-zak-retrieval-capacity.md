# ob-zak — GDN-1 vs GDN-2 Associative Recall Capacity (Portable, State-Level)

**Bead:** `ob-zak` · **Date:** 2026-08-09
**Device:** rk3588-t4 (RK3588, Cortex-A76 big cluster, governor=performance)
**Code:** `bench/gdn2_retrieval.py` · **Tests:** `tests/test_gdn2_retrieval.py` (28 tests)
**Data:** `results/raw/rk3588-t4_gdn2_retrieval.csv`
**Manifest:** `results/manifests/rk3588-t4_gdn2_retrieval.json`

---

## TL;DR

**At matched gate configurations, GDN-1 and GDN-2 have bit-identical associative
recall capacity (max |S₁ − S₂| = 0.00).** Both achieve 100% top-1 retrieval up to
N = K = 128 associations, then degrade identically above the rank ceiling
(98.1% at N=160, 94.8% at N=192, 77.0% at N=256). Under uniform exponential
decay, both degrade identically (100% → 62.5% → 34.4%).

**Conclusion:** GDN-2's hypothesised multi-key retrieval advantage (ADR-0001) is
**not testable without an adapted model**. With matched gates the two
architectures are mathematically equivalent — the retrieval benefit can only
arise from *learned, input-dependent per-channel gating*, which requires model
fine-tuning that is infrastructure-blocked on edge devices.

This closes the "no RULER quality eval" gap in the ob-7b5 research note with a
precise, reproducible explanation rather than an untested claim.

---

## 1. What was tested

A **recurrence-state-level associative recall** test — the standard capacity
benchmark for linear-attention / RNN state memory:

1. Generate *N* random unit-norm key vectors in ℝᴷ and one-hot value vectors in
   ℝⱽ.
2. Write all *N* associations into the recurrent state via the delta-rule
   recurrence (one step per pair).
3. Query each key against the final state: `o_i = scale · k_i @ S`; correctness
   is `argmax(o_i) == i`.
4. Score top-1 accuracy.

**Shapes:** K = V = 128 (Qwen3.5-4B verified: `linear_key_head_dim=128`,
`linear_value_head_dim=128`), single head. Pure NumPy (float64 accumulation),
no model weights.

This tests the recurrent state's capacity directly — the mechanism that matters
at decode, where GDN's O(1) memory is the architectural selling point.

### Why not full RULER?

The bead as originally scoped envisions a full-model RULER multi-key evaluation:
feed the long-context corpus (ob-del) through an adapted GDN-2 checkpoint
(ob-68l) and score generation-level retrieval. That requires torch +
transformers + the swapped checkpoint, none of which exist on RK3588 edge nodes
(`torch=null` in the manifest; ob-68l's swap ran on x86/CUDA). This portable
study delivers the part of the answer achievable on shipping Arm hardware.

## 2. The mathematical equivalence

With no decay and unit gates:

| | GDN-1 (α=1, β=1) | GDN-2 (g=0, b=1, w=1) |
|---|---|---|
| Update | `S ← (I − kkᵀ)S + kvᵀ` | `S ← S + k(v − kᵀS)ᵀ` |

These are the **same expression** rearranged. The test confirms it numerically:
**max |S₁ − S₂| = 0.00** (bit-identical, not just within float epsilon).

GDN-2's distinct gates only matter when they differ from uniform:
- **b < 1** (partial erase): weakens the `(b⊙k)ᵀS` erase term → less old-value
  removal, more interference with prior associations.
- **w < 1** (partial write): weakens the write → weaker new-value encoding.
- **per-channel g** (non-uniform decay): selective per-dimension forgetting.

All three require **learned, input-dependent values** to produce a retrieval
benefit. Setting them by hand to a uniform constant collapses GDN-2 back to
GDN-1.

## 3. Results

### Capacity sweep (accuracy vs number of keys, K=V=128)

| N keys | GDN-1 | GDN-2 |
|---:|---:|---:|
| 8 | 100% | 100% |
| 32 | 100% | 100% |
| 64 | 100% | 100% |
| 96 | 100% | 100% |
| **128** | **100%** | **100%** |
| 160 | 98.1% | 98.1% |
| 192 | 94.8% | 94.8% |
| 256 | 77.0% | 77.0% |

The capacity ceiling is at **N = K = 128** — the rank limit of the state matrix
S ∈ ℝ¹²⁸ˣ¹²⁸. Below it, the delta rule resolves associations exactly (random
unit-norm keys in ℝ¹²⁸ are near-orthogonal for N ≤ 128). Above it, interference
grows monotonically. Both models are identical at every point.

### Decay sweep (N=64 keys, uniform exponential decay)

| Decay | GDN-1 | GDN-2 |
|---:|---:|---:|
| 0.000 | 100% | 100% |
| 0.005 | 100% | 100% |
| 0.010 | 100% | 100% |
| 0.050 | 62.5% | 62.5% |
| 0.100 | 34.4% | 34.4% |

Under *uniform* decay GDN-1's scalar α and GDN-2's per-channel g (all equal)
are the same operation. Retrieval of early-written associations degrades
identically. The difference would only appear with **non-uniform** learned
gates — e.g., a model that selectively preserves high-salience channels while
forgetting others.

## 4. What this means for the submission

1. **The capacity ceiling is K** — for Qwen3.5-4B's K=128, each GDN head's
   state holds up to 128 clean associations. This is the architectural limit
   that makes GDN a *compressed* summary, not an unlimited memory. At 3:1
   hybrid ratio with 24 GDN layers × 16 key heads, that is ~49K clean
   associations per forward pass — far below the unbounded KV cache, but
   sufficient for the compressed-summary role.

2. **GDN-2's quality claim remains the paper's, unverified by us at the
   model level** — but we now have a precise, reproducible explanation of *why*:
   the architectural difference (decoupled gating) is inert without learned
   weights, and the state-level capacity is identical. This is more defensible
   than an unstated gap.

3. **Consistent with ob-7b5** — the operator-level cost note reported GDN-2 is
   "nearly free at decode on big cores (1.2–1.5×), but costs 2.2–2.7× at
   prefill." Combined with this finding: GDN-2's added cost buys potential
   retrieval quality *only* in a learned-gating regime we could not test.

## 5. Limitations

- **State-level, not model-level:** this tests the recurrent state's raw
  capacity, not end-to-end generation quality. A full RULER eval would capture
  attention-layer interactions, normalization effects, and learned gate
  distributions.
- **Random keys, not learned embeddings:** real keys are projections of learned
  token embeddings and are correlated. The near-orthogonality assumption
  overstates capacity slightly; the *relative* GDN-1/GDN-2 comparison is
  unaffected.
- **No adapted checkpoint:** the GDN-2 layer swap (ob-68l) was a single-layer,
  30-step MSE distillation on Qwen3.5-0.8B — insufficient for a meaningful
  retrieval-quality comparison even if torch were available on-device.
