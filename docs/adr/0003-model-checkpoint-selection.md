# ADR 0003: Target Qwen3.5-4B as primary, Qwen3.5-0.8B as fast-iteration fallback

- **Status:** Accepted
- **Date:** 2026-08-02
- **Bead:** `ob-eae`
- **Deciders:** maintainer + agent
- **Supersedes nothing.** Refines the survey in [`docs/MODEL_SURVEY.md`](../MODEL_SURVEY.md) with
  figures read directly from each checkpoint's `config.json`.

## Context

Bead `ob-7fv` surveyed the family and recommended 4B primary / 0.8B fallback. Before committing,
both `config.json` files were fetched from HuggingFace and read directly rather than relying on the
survey's summary or on documentation defaults.

| | **Qwen3.5-4B** | **Qwen3.5-0.8B** |
|---|---|---|
| `model_type` | `qwen3_5` | `qwen3_5` |
| layers | **32 = 24 GDN + 8 full attention** | **24 = 18 GDN + 6 full attention** |
| `hidden_size` | 2560 | 1024 |
| native context | **262,144** | **262,144** |
| full-attn heads / KV heads / head_dim | 16 / 4 / 256 | 8 / 2 / 256 |
| linear key / value heads | 16 / **32** | 16 / **16** |
| linear key / value head dim | 128 / 128 | 128 / 128 |
| conv kernel width | 4 | 4 |
| vision tower | yes | yes |

The 3:1 ratio holds exactly in both, read from the explicit `layer_types` array rather than
inferred. Both reach the full 262K native context, which matters: the smaller model is not a
context-limited toy.

### The memory decomposition, computed from config

This is the project's central claim made quantitative *before* any measurement. GDN state is
`n_value_heads × d_k × d_v` per layer and is **independent of context length**; the KV cache exists
only in the full-attention layers and grows linearly.

**Qwen3.5-4B** — GDN state: 32 × 128 × 128 = 524,288 floats/layer = 2.0 MB fp32, **48 MB across 24
layers, flat at every context length**. KV cache across its 8 full-attention layers (fp16, GQA with
4 KV heads):

| Context | KV cache | GDN state | Ratio |
|---|---:|---:|---:|
| 4K | 128 MB | 48 MB | 2.7× |
| 32K | 1.0 GB | 48 MB | 21× |
| 128K | 4.0 GB | 48 MB | 85× |
| 262K | **8.0 GB** | **48 MB** | **170×** |

**The counterfactual is the headline.** Had all 32 layers been full attention, the KV cache at 262K
would be 32/8 × 8.0 GB ≈ **32.8 GB**. The hybrid stack instead needs 8.0 GB of cache plus 48 MB of
state — a **~75% reduction, about 24.8 GB saved**, on a board with 64 GB total. That is the
difference between 262K context fitting alongside weights and activations, and not fitting.

Weights are comfortable either way: 4B at fp16 ≈ 8 GB, INT8 ≈ 4 GB, INT4 ≈ 2 GB, all far inside
the ~10B-parameter NPU ceiling.

## Decision

**Primary: `Qwen/Qwen3.5-4B`. Fallback for fast iteration: `Qwen/Qwen3.5-0.8B`.**

Reasons, in order of weight:

1. **4B gives the canonical 24 GDN + 8 full-attention split** the plan, kernels, and boundary-
   crossing analysis are all written against (`PLAN.md` §3.1 — 16 engine crossings per token
   assumes 8 attention layers). Using it keeps every downstream number directly comparable.
2. **It maximises the memory story without straining the board.** At 262K the hybrid saves ~24.8 GB
   versus an all-attention equivalent — a large, honest, verifiable claim.
3. **It sits well inside the NPU's stated ~10B ceiling**, unlike the 9B variant which at ~9.65B
   total leaves almost no headroom and makes it ambiguous whether the bundled vision tower counts.
4. **0.8B is genuinely useful as a fallback, not a consolation.** Same 3:1 architecture, same 262K
   context, ~4× smaller state (18 MB fp32 total), so quantization and harness iteration are fast,
   and it is the realistic choice if the Edge AI hedge lands on a phone with limited RAM.

Both are Apache-2.0 (verified from the LICENSE files during `ob-7fv`), so redistribution terms
pose no problem and weights can be fetched at setup time rather than vendored.

## Alternatives considered

| Option | Why not | What would change our mind |
|---|---|---|
| **Qwen3.5-9B primary** | ~9.65B total against a ~10B NPU ceiling leaves no headroom, and it is unclear whether the ceiling counts the bundled vision + MTP weights. fp16 weights ≈ 19 GB would also crowd a 262K run. | If the ceiling turns out to exclude vision/MTP weight and NPU offload proves valuable, 9B becomes the more impressive demo. |
| **Qwen3.5-2B primary** | Nothing wrong with it, but its layer count would differ from the 24+8 the analysis is built on, adding translation work for no gain. | If 4B proves too slow to iterate on and 0.8B too small to be credible. |
| **0.8B as primary** | Understates what the hardware can do; a 64 GB board running a 0.8B model is not a compelling Physical AI story. | If the board never arrives and the hedge is a phone, this becomes primary by necessity. |
| **MoE (35B-A3B)** | Total weights far exceed the NPU ceiling and the 64 GB budget at any useful precision. | Not for this submission. |

## Consequences

**Accepted costs.** Both checkpoints bundle a vision tower we do not need (no text-only weights
exist at any size in this family), so we carry 5–15% dead weight. A text-only *code path* exists
via `Qwen3_5ForCausalLM` + `Qwen3_5TextConfig`, so this costs download size and memory, not compute.

**Precision note carried forward.** The 4B has **32 linear value heads, not 16** — so its per-layer
GDN state is 524,288 floats, double the 262,144 figure quoted earlier in this project (which came
from the GDN-2 paper's 1.3B configuration and from the 0.8B). Docs and the `bd` memory have been
corrected. Anything sized against the old number needs rechecking.

**One assumption still open.** The state-size arithmetic above assumes the layout is
`n_value_heads × d_k × d_v`. That is the standard DeltaNet form and is consistent with the GDN-2
paper's matched-state note, but it has **not** yet been confirmed against Qwen3.5's modeling code.
Bead `ob-37v` owns that confirmation, and every memory figure here depends on it.

**Follow-on work.** `ob-37v` (confirm state layout from modeling code), `ob-aqv` (x86 reference on
the 4B), `ob-qpa` (quantization policy — note the FP16 carve-out for recurrent state and the fp32
decay accumulator established in `docs/FINDINGS.md` §4), `ob-ixt` (weight fetch scripts).

**Reversal cost.** Low. Both checkpoints share `model_type` and the same architecture family, so
switching between them is a config change plus a re-run, not a redesign. Switching *away* from the
Qwen3.5 family would invalidate the layer-split analysis and the op-coverage audit's shape
assumptions — that would be expensive, and there is no reason to.
