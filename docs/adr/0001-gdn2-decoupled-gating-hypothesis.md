# ADR 0001: Record the testable hypothesis for GDN-2 decoupled erase/write gating on edge silicon

- **Status:** Proposed
- **Date:** 2026-08-02
- **Bead:** `ob-8m7`
- **Deciders:** GDN-2 research track (E8)

## Context

**Source access.** The full paper was reachable. `arxiv.org/abs/2605.22791` gave the abstract
and metadata; `arxiv.org/html/2605.22791` (arXiv's HTML rendering) gave the full text, including
equations and result tables, which was fetched directly and parsed (not just summarized) to
confirm the numbers and equations quoted below are the paper's own, not a paraphrase. This
supersedes the abstract-only caveat in [`docs/CLAIM_VERIFICATION.md`](../CLAIM_VERIFICATION.md)
§1.5 — that file predicted numeric RULER scores would require the full paper, and the full paper
does contain them (Table 3). Everything below marked "full paper" was pulled from that HTML
source; nothing here is invented.

**Paper:** [Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention](https://arxiv.org/abs/2605.22791),
Ali Hatamizadeh, Yejin Choi, Jan Kautz, arXiv:2605.22791 [cs.AI]. Reference code:
[`github.com/NVlabs/GatedDeltaNet-2`](https://github.com/NVlabs/GatedDeltaNet-2).

### The mechanism

The paper builds the argument in three steps (its Eq. 5–10), each adding one degree of freedom
to the recurrent state update `S_t` (a per-head `d_k × d_v` fast-weight matrix):

1. **DeltaNet** (no gating): `S_t = (I − β_t k_t k_tᵀ) S_{t-1} + β_t k_t v_tᵀ`, a scalar delta rule.
2. **Gated DeltaNet** (Qwen3.5's mechanism) adds a scalar decay `α_t`:
   `S_t = α_t (I − β_t k_t k_tᵀ) S_{t-1} + β_t k_t v_tᵀ`.
   Both `α_t` and `β_t` are **single scalars per head per step** — one number controls how much
   of the *entire* old state is decayed, and the same tied scalar `β_t` controls both how much of
   the old association is erased (via the key-side projector `k_t k_tᵀ`) and how strongly the new
   value is written.
3. **KDA** promotes decay to channel-wise: `D_t = Diag(α_t)`, `α_t ∈ (0,1]^{d_k}`, giving
   `S_t = (I − β_t k_t k_tᵀ) D_t S_{t-1} + β_t k_t v_tᵀ`. Decay is now per-channel, but the erase/write
   strength `β_t` is *still one tied scalar*.
4. **Gated DeltaNet-2** removes that last tie. It defines a channel-wise erase gate
   `b_t ∈ [0,1]^{d_k}` (key-side) and a channel-wise write gate `w_t ∈ [0,1]^{d_v}` (value-side),
   each produced by an independent linear projection + sigmoid (`b_t = σ(W_b x_t)`,
   `w_t = σ(W_w x_t)`), and writes the update as (paper's boxed Eq. 10):

   ```
   S_t = (I − k_t (b_t ⊙ k_t)ᵀ) D_t S_{t-1} + k_t (w_t ⊙ v_t)ᵀ
   ```

   Concretely: the erase factor's *left* factor stays `k_t` (preserving which association is
   targeted), but its *right* factor becomes `b_t ⊙ k_t` — channel-selective erasure of the read
   direction — while the write term becomes `k_t (w_t ⊙ v_t)ᵀ`, channel-selective value insertion.
   Gated DeltaNet-2 recovers KDA exactly when `b_t = β_t·1` and `w_t = β_t·1`, and recovers Gated
   DeltaNet by further tying the decay — so GDN is a genuine special case, not just an analogy.

   Critically, **the state itself does not grow**: the paper explicitly matches per-layer state
   size across all compared architectures at `H·d_k·d_v = 16·128·128 = 262,144` floats/layer,
   identical for Mamba-2, Gated DeltaNet, KDA, and Gated DeltaNet-2. What changes is only the
   *gate* tensors: 2 scalars/head/step (GDN) → `d_k + d_v = 256` floats/head/step (GDN-2), plus
   the two independent projection weight matrices (`W_b`, `W_w`) that produce them.

### Why decoupling should plausibly help multi-key retrieval

The paper's own intuition (§3.1): a tied scalar forces "how much old content is erased from the
read direction" and "how much new value is written" to move together. In a single-scalar model,
remembering a *new* key-value pair at high fidelity (high `β_t`) necessarily means erasing more
of whatever the state held on the *matching* key subspace — there is no way to write strongly
without also erasing strongly, because one number does both jobs. In a multi-key retrieval
setting (RULER MK-NIAH: several key–value pairs must be held simultaneously and later
distinguished), this tie is exactly where interference arises — writing key N can bleed into
erasing information relevant to keys 1..N-1 sharing overlapping channels. Decoupling `b_t` and
`w_t` lets the model erase precisely the key channels that need revision while writing value
channels at a different, independently-learned strength — so a new key-value write need not cost
old keys stored on channels the erase gate has learned to leave alone.

**This is now more than intuition — the full paper measures it directly (full paper, Table 3,
MK-NIAH-1, recurrent-only, matched state size):**

| Model | 1K | 2K | 4K |
|---|---:|---:|---:|
| Mamba-2 | 29.0 | 21.2 | 21.4 |
| Gated DeltaNet | 58.0 | 37.0 | 27.8 |
| KDA | 54.0 | 44.2 | 28.0 |
| **Gated DeltaNet-2** | **72.6** | **51.4** | **37.8** |

and in the hybrid (SWA) setting, MK-NIAH-1: GDN-2 93.0 / 84.6 / 48.0 at 1K/2K/4K vs. GDN
91.0 / 78.4 / 44.8 — a smaller but still consistent edge once local attention absorbs
short-range recall. All at 1.3B params / 100B FineWeb-Edu tokens, the only scale the paper
evaluates. The paper's own ablation (Table 5) further shows the erase-side channel structure
(`b_t`) does most of the work — scalarizing `w_t` alone recovers most of the gain, scalarizing
`b_t` alone loses much more — consistent with the erase-side being where multi-key interference
actually lives.

**What the paper does *not* measure:** any decode-time / single-token autoregressive inference
latency or memory-bandwidth figure, on any hardware. Its only efficiency number (Fig. 2) is
**training** throughput on an H100 GPU across sequence lengths 2K–16K, in Kt/s (thousand
tokens/sec) under a fixed token budget — a compute-bound, batched, chunk-parallel regime, not
the memory-bandwidth-bound single-token decode regime this project cares about. In that regime
GDN-2 "preserves the near-flat scaling profile... dropping only mildly from 38.0 to 36.1 Kt/s,"
which the authors attribute to "the added channel-wise erase and write gates" relative to KDA —
i.e. a small, roughly constant training-time cost. This number is not directly transferable to
edge decode cost (see below) but it is the only quantitative signal the paper gives about the
mechanism's added cost, and it points toward "modest," not "free."

### Connecting to our own verified finding: decode is already memory-bandwidth-bound

Separately verified for Qwen3.5's GDN kernels (`docs/CLAIM_VERIFICATION.md` §3, transformers
docs): optimizing the DeltaNet kernel path speeds up prefill 1.38–1.49× and leaves decode flat,
"because the single-token DeltaNet recurrence is memory-bandwidth-bound." On the Orion O6's
100GB/s LPDDR5 — a fraction of GB10 or H100 bandwidth — this binds harder, not softer. Single-
token decode for a GDN layer must, every step, stream: (a) the recurrent state `S_{t-1}`
(`H·d_k·d_v` floats/layer — unchanged between GDN and GDN-2, per the paper's matched-state-size
design), and (b) the layer's projection weights (`W_q, W_k, W_v`, plus gate projections) — for
batch=1 decode, weight traffic dominates total bytes moved, since there is no batching to
amortize a weight read across multiple tokens. This is the load-bearing fact for the cost
analysis below.

## Decision

We record the following **testable, falsifiable hypothesis** as the object of E8's experiment,
rather than assuming the paper's result at 1.3B on FineWeb-Edu transfers to our setting:

> **At edge-appropriate model sizes (≤4B parameters), swapping Gated DeltaNet-2's decoupled
> channel-wise erase/write gating in for Qwen3.5's Gated DeltaNet gating improves RULER
> multi-key retrieval (MK-NIAH-1) accuracy by a margin larger than run-to-run variance
> (measured via repeated seeds/samples per our harness's percentile-reporting convention),
> at no worse than a 15% decode tokens/sec regression on our target aarch64 hardware
> relative to unmodified Gated DeltaNet at matched precision.**

The 15% figure is a placeholder threshold chosen to be a meaningful-but-not-devastating cost —
it should be revisited once `t-harness-mem`/benchmark infrastructure exists and can supply an
actual run-to-run variance figure to calibrate "larger than noise" and a real decode-throughput
baseline to calibrate the cost ceiling; it is not derived from any measurement yet. What must
not move is the shape of the hypothesis: quality gain and decode cost are reported and judged
together, never quality alone. A result that improves retrieval but blows through the decode
budget is a *documented tradeoff*, not a validated hypothesis.

**Both halves can fail independently**, and either failure is informative:
- Retrieval gain does not replicate at ≤4B / our training or fine-tuning budget → the 1.3B/100B-token
  result does not transfer to edge scale or our (much smaller) adaptation budget.
- Retrieval gain replicates but decode cost exceeds the threshold → the mechanism is a genuine
  quality/bandwidth tradeoff that is a poor fit for a 100GB/s board specifically, even though it
  is a good fit for GPU training/serving where bandwidth is far less constrained.

### Cost analysis: will GDN-2 be faster or slower on a bandwidth-bound edge board?

**Honest read: plausibly slower on decode, and the paper gives us no evidence either way for
that regime, because it never measures it.** The reasoning:

1. **What does not change.** The recurrent state `S_t` is `d_k × d_v` per head, identical in
   size to GDN's — this is a controlled variable in the paper's own experimental design, not
   something GDN-2 inflates. So the single largest piece of per-step data (the state read-modify-
   write) does not get more expensive by architecture alone.
2. **What does change.** GDN-2 needs two extra gate vectors per head per step (`b_t`, `w_t`,
   sized `d_k` and `d_v` — 256 floats/head vs. 2 scalars/head for GDN's `α_t, β_t`), each produced
   by its own linear projection (`W_b`, `W_w`) rather than reusing KDA's existing channel-wise
   decay projection. For batch=1 decode, every one of those projection weight matrices must be
   streamed from DRAM every token — there is no reuse across tokens to amortize the read. This is
   pure *additional* weight traffic layered on top of an already bandwidth-saturated per-token
   budget; it is the kind of cost that shows up as decode latency almost one-for-one on a
   bandwidth-bound system, because there is no spare compute time to hide it behind.
3. **The only quantitative cost hint we have (training, not decode) points to "modest but
   nonzero."** GDN-2 costs about 5% (38.0 → 36.1 Kt/s) relative to the near-equivalent KDA
   baseline in a compute-bound, chunk-parallel training regime on an H100. That regime hides
   memory latency behind massive batch/sequence parallelism that single-token edge decode does
   not have. It would be unjustified to claim the edge decode cost is "about 5%" by analogy — the
   bottleneck resource is different (compute+overlappable-bandwidth vs. hard bandwidth wall) — but
   it is reasonable to read this as a *floor*: if the mechanism costs something even where
   bandwidth is relatively abundant and parallelism is high, it is unlikely to cost *nothing* where
   bandwidth is the scarcest resource on the whole system and there is no parallelism to spend.
4. **Bottom line for this project.** If GDN-2 does improve multi-key retrieval at our scale, it
   may still be the wrong choice for the 100GB/s edge target specifically — a real, publishable,
   negative-leaning finding — even while being an unambiguous win for GPU training/serving where
   the paper's own numbers apply. This is exactly the tension the hypothesis above is designed to
   surface rather than paper over.

### Alternatives considered

| Option | Description | Why / why not |
|---|---|---|
| **(a) Benchmark-only microbenchmark** | Implement/port Gated DeltaNet-2's recurrent kernel (from the NVLabs reference) standalone, and measure decode tokens/sec and RULER-style retrieval accuracy in isolation, against an equivalently-sized GDN kernel, on our target hardware. No trained checkpoint modification. | Cheap, safe, fully within our compute budget, directly measures the bandwidth question this ADR cares about. Weaker signal on retrieval quality, since it requires either a small from-scratch training run or reuse of the paper's public checkpoints/config rather than testing inside an actual Qwen3.5 model. **Default choice**, per PLAN.md §4 E8 and the descope ladder (T-2, Aug 12, cuts option (b) if behind schedule). |
| **(b) Layer-swap into a Qwen3.5-architecture checkpoint** | Replace some or all GDN layers in a small Qwen3.5 dense checkpoint (≤4B) with Gated DeltaNet-2 layers, fine-tune/adapt, and evaluate RULER multi-key retrieval end-to-end in the real hybrid architecture (3:1 GDN:full-attention). | Higher-reward: tests the actual hypothesis in the actual target architecture, including interaction with Qwen3.5's periodic full-attention layers (whose presence changes the retrieval picture, per the paper's own hybrid-vs-recurrent-only split in Table 3). Higher-risk: needs training/fine-tuning compute we have not budgeted, needs care reconciling shapes (Qwen3.5 uses 16 key heads / 32 value heads vs. the paper's matched `H=16` for both), and needs a schedule position ahead of plan (R5 in `PLAN.md`, escalation only if ahead at Aug 10). Reversal cost is nontrivial: a partially fine-tuned checkpoint is throwaway compute if abandoned, whereas option (a) is a standalone kernel that costs nothing to discard. |

## Consequences

**Accepted costs.** We commit to reporting a *paired* result (quality delta **and** decode cost
delta) rather than quality alone, which is more work than a single benchmark but is the only
version of this experiment that is not misleading. We also accept that our 15% cost threshold is
provisional and must be recalibrated once real variance/baseline numbers exist — shipping a
number now without a citation would violate this project's evidence discipline.

**Follow-on work.** The E8 epic (PLAN.md §4) already scaffolds the next steps: clone and
smoke-test the NVLabs reference implementation, then execute option (a) by default. Any bead
needed to implement the microbenchmark, adapt the reference kernel to our target hardware, or
(if schedule allows) attempt option (b) should be filed under E8 rather than invented here — this
ADR's job is the hypothesis and cost reasoning, not the task breakdown.

**Reversal cost.** Low for the default path: option (a) is a standalone kernel comparison that
can be dropped or redone without touching any trained checkpoint. Reversal cost is high for
option (b) if attempted — partially-adapted/fine-tuned weights are not cheaply salvageable if the
hypothesis fails or the schedule runs out, which is why the descope ladder (`PLAN.md` §7) cuts
(b) first, at T-2 (Aug 12), keeping (a) as the floor that always ships.

**Negative results are an acceptable, and explicitly desired, outcome here.** If the RULER gain
does not replicate at ≤4B, or replicates but exceeds the decode-cost threshold on our hardware,
that is a real, reportable finding — "the decoupled-gating retrieval advantage measured at 1.3B
on GPU training infrastructure does not transfer to a 100GB/s edge decode target, and here is the
bandwidth profile showing why" is exactly the kind of honest, evidence-backed claim this project
is built to make (PLAN.md §9, "Negative and partial results get written up honestly"), and it
costs nothing under a judge's scrutiny in the way an overstated claim would.
