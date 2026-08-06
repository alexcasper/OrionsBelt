# Qwen3.5 checkpoint survey (0.8B–4B, plus 9B upper reference)

Bead `ob-7fv` (child of `ob-xh3`, blocks `ob-eae`). Hardware-independent — every figure below is
derived from HuggingFace Hub API responses, raw `config.json` / `LICENSE` files fetched directly
from the `Qwen` org, and the [transformers `Qwen3.5` docs](https://huggingface.co/docs/transformers/model_doc/qwen3_5).
This file does not contradict [`PLAN.md`](./archive/PLAN.md) or [`CLAIM_VERIFICATION.md`](./CLAIM_VERIFICATION.md);
it adds per-checkpoint detail underneath their verified facts. Surveyed 2026-08-02.

**Scope note:** the family also ships 27B / 35B‑A3B / 122B‑A10B(-FP8) MoE and dense checkpoints
above our target range. Those are out of scope here and are not discussed.

---

## 1. Checkpoint identity and naming convention

Every size in this survey ships as exactly two dense text+vision checkpoints, both Apache-2.0,
plus quantized/format spin-offs (GGUF, GPTQ-Int4, FP8) not needed for our purposes:

| Size | Instruct/chat-ready | Base (pretrain-only) |
|---|---|---|
| 0.8B | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) | [`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) |
| 2B | [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) | [`Qwen/Qwen3.5-2B-Base`](https://huggingface.co/Qwen/Qwen3.5-2B-Base) |
| 4B | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) | [`Qwen/Qwen3.5-4B-Base`](https://huggingface.co/Qwen/Qwen3.5-4B-Base) |
| 9B (upper reference) | [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) | [`Qwen/Qwen3.5-9B-Base`](https://huggingface.co/Qwen/Qwen3.5-9B-Base) |

There is **no separate `-Instruct` suffix** the way older Qwen releases had one — the un-suffixed
repo *is* the post-trained, chat-ready checkpoint. Verified directly from repo file listings via
the HF API (`/api/models/<repo>`), not just the model card prose: the un-suffixed `Qwen3.5-4B`
repo ships a `chat_template.jinja`; `Qwen3.5-4B-Base` does not. The `-Base` model card additionally
states its "intended use cases are fine-tuning, in-context learning experiments, and other research
or development purposes, not direct interaction." Same pattern confirmed for all four sizes.

**Recommendation in this survey is against the instruct/chat-ready (un-suffixed) checkpoints** —
they're directly runnable and representative of what an actual deployment would serve.

---

## 2. Architecture: layer split, verified directly from each `config.json`

Fetched raw `config.json` (`text_config.layer_types`, not a summarized/secondary description) for
all four sizes. This is the ground truth `t-arch-audit` will also read.

| Size | `num_hidden_layers` | `hidden_size` | GDN (linear_attention) | Full attention | Split pattern |
|---|---:|---:|---:|---:|---|
| 0.8B | 24 | 1024 | **18** | **6** | 6 × (3 GDN → 1 full) |
| 2B | 24 | 2048 | **18** | **6** | 6 × (3 GDN → 1 full) |
| 4B | 32 | 2560 | **24** | **8** | 8 × (3 GDN → 1 full) |
| 9B | 32 | 4096 | **24** | **8** | 8 × (3 GDN → 1 full) |

All four confirm the 3:1 ratio exactly (`linear_attention` count / `full_attention` count = 3.0).
0.8B and 2B share the smaller 24-layer / 18-GDN-6-full shape; 4B and 9B share the 32-layer /
24-GDN-8-full shape that PLAN.md and CLAIM_VERIFICATION.md already cite as the dense default.
`layer_types[:8]` is identical across all four
(`linear, linear, linear, full, linear, linear, linear, full, ...`), confirming the pattern starts
the same way at every scale.

### Linear-attention (Gated DeltaNet) shapes — **not uniform across sizes**

This corrects an implicit assumption worth flagging: CLAIM_VERIFICATION.md's verified shapes
(`linear_num_key_heads=16, linear_num_value_heads=32`) hold for **4B and 9B**, but **not** for
0.8B/2B, which use `linear_num_value_heads=16` (symmetric key/value head count). Direct from
`config.json`:

| Size | `linear_conv_kernel_dim` | `linear_key_head_dim` | `linear_value_head_dim` | `linear_num_key_heads` | `linear_num_value_heads` |
|---|---:|---:|---:|---:|---:|
| 0.8B | 4 | 128 | 128 | 16 | **16** |
| 2B | 4 | 128 | 128 | 16 | **16** |
| 4B | 4 | 128 | 128 | 16 | **32** |
| 9B | 4 | 128 | 128 | 16 | **32** |

`linear_conv_kernel_dim`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_num_key_heads`
are constant across the whole small-dense range; only `linear_num_value_heads` doubles once you
cross into the 32-layer/2560+ hidden-size tier.

### Full-attention (Gated Attention) shapes, for KV-cache accounting

Not explicitly requested but needed to reason about "room for long-context state," so recorded
here (also direct from `config.json`):

| Size | `num_attention_heads` (Q) | `num_key_value_heads` (KV, GQA) | `head_dim` |
|---|---:|---:|---:|
| 0.8B | 8 | 2 | 256 |
| 2B | 8 | 2 | 256 |
| 4B | 16 | 4 | 256 |
| 9B | 16 | 4 | 256 |

### Context length

All four: `max_position_embeddings = 262144` in the raw config, and **no `rope_scaling` block is
set by default** (confirms CLAIM_VERIFICATION.md: 262K is native; ~1M needs YaRN applied
explicitly via `rope_scaling`, it is not baked into the shipped config).

---

## 3. License — confirmed Apache-2.0, no custom terms, at every size

Fetched the actual `LICENSE` file (not a summary) from all eight repos (base + instruct-ready ×
four sizes): every one opens with `Apache License / Version 2.0, January 2004`. No custom Qwen
license, no field-of-use or scale-gated restriction (unlike Llama's community license). This
satisfies R10 in PLAN.md's risk register (checkpoint license restricting redistribution) — **not
a risk for any candidate in this range.** Also satisfies the competition's MIT-or-Apache-2.0
requirement independent of our own repo license.

---

## 4. Multimodality — natively multimodal at every size, no text-only checkpoint published

Confirmed by listing the full Qwen3.5 collection (21 items) via the HF collection page: **every**
dense/MoE checkpoint in the family is tagged `Image-Text-to-Text`. There is **no separate
text-only weights release** at any size — 0.8B through the largest MoE all ship a `vision_config`
in `config.json` and a `model.visual.*` tensor prefix in the safetensors weights.

There **is** a text-only *code path*: the transformers docs are explicit — "Use
`Qwen3_5ForCausalLM` for text-only generation with `Qwen3_5TextConfig`; use
`Qwen3_5ForConditionalGeneration` with the full `Qwen3_5Config` and a processor" for multimodal
input. So you can run text-only inference through the causal-LM head without ever touching the
vision tower's forward pass. But — verified directly from each checkpoint's `model.safetensors`
header (not the model card, the actual tensor shape metadata) — **the downloaded weight file
always bundles the vision tower and an MTP (multi-token-prediction) head alongside the text
backbone**; there is no smaller "text-only" file to fetch instead. Breakdown, summed from
safetensors header shapes:

| Size | Total params | `language_model` (text) | `visual.*` (vision tower) | `mtp.*` (speculative-decode head) |
|---|---:|---:|---:|---:|
| 0.8B | 873,438,784 | 752,393,024 (86.1%) | 100,592,896 (11.5%) | 20,452,864 (2.3%) |
| 2B | 2,274,069,824 | 1,881,825,088 (82.8%) | 331,416,576 (14.6%) | 60,828,160 (2.7%) |
| 4B | 4,659,865,088 | 4,205,751,296 (90.3%) | 333,514,240 (7.2%) | 120,599,552 (2.6%) |
| 9B | 9,653,104,368 | 8,953,803,264 (92.8%) | 456,010,480 (4.7%) | 243,290,624 (2.5%) |

Practical read: the vision tower is a fixed ~100–460M-parameter cost that shrinks as a *share* of
the checkpoint as size grows (11.5% at 0.8B down to 4.7% at 9B), but in absolute terms it's small
everywhere in this range. If we ever want to shed it for a leaner Arm deployment, that requires
either re-saving a trimmed `state_dict` ourselves or accepting the dead weight; there is no vendor
"text-only" artifact to substitute. The `mtp.*` block (multi-token-prediction / speculative
decoding head) is a separate, similarly small, droppable-if-unused component.

**Reported "parameter count" (0.8B/2B/4B/9B) is the whole checkpoint including vision+MTP, not the
text backbone alone** — worth remembering when comparing against the "within ten billion
parameters" NPU ceiling, since the number vendors advertise already contains ~5–15% of non-text-generation weight.

---

## 5. Memory footprint

Bytes/param: FP16/BF16 = 2, INT8 = 1, INT4 = 0.5. Two views: **whole checkpoint** (what you
actually download) and **text backbone only** (`language_model.*`, what matters for a pure-text
GDN-scaling demo if vision+MTP tensors are stripped).

| Size | Precision | Whole checkpoint | Text backbone only |
|---|---|---:|---:|
| 0.8B | FP16 | 1.75 GB | 1.50 GB |
| 0.8B | INT8 | 0.87 GB | 0.75 GB |
| 0.8B | INT4 | 0.44 GB | 0.38 GB |
| 2B | FP16 | 4.55 GB | 3.76 GB |
| 2B | INT8 | 2.27 GB | 1.88 GB |
| 2B | INT4 | 1.14 GB | 0.94 GB |
| 4B | FP16 | 9.32 GB | 8.41 GB |
| 4B | INT8 | 4.66 GB | 4.21 GB |
| 4B | INT4 | 2.33 GB | 2.10 GB |
| 9B | FP16 | 19.31 GB | 17.91 GB |
| 9B | INT8 | 9.65 GB | 8.95 GB |
| 9B | INT4 | 4.83 GB | 4.48 GB |

**Every candidate fits the 64GB board at every precision, including full FP16 for 9B**, with tens
of GB free even before subtracting KV cache / recurrent state. Weight footprint is **not the
discriminating constraint** in this size range — see §6 for what actually differs.

### Full-attention KV cache at native 262,144-token context (computed, not vendor-published)

Standard formula `2 (K,V) × num_key_value_heads × head_dim × bytes/elem × num_full_attention_layers
× context_len`, using the config values from §2. This is an independent calculation from the
verified shapes, flagged as **derived**, not a primary-source figure on its own:

| Size | Full-attn layers | KV cache @ 262,144 tok, FP16 | @ INT8 (if KV quantized) |
|---|---:|---:|---:|
| 0.8B / 2B | 6 | ≈ 3.00 GB | ≈ 1.50 GB |
| 4B / 9B | 8 | ≈ 8.59 GB | ≈ 4.30 GB |

The GDN (linear-attention) layers contribute **no context-dependent term at all** — their
recurrent state is a fixed-size matrix per layer, independent of sequence length (this is the
whole thesis; exact per-layer state-tensor byte count needs the modeling code, tracked separately
under `t-arch-audit`, not computed here to avoid overclaiming a number we haven't verified against
source).

**Combined worst case (weights FP16 + full-context KV cache FP16), on a 64GB board:**

| Size | Weights (whole ckpt) | + KV cache @ 262K | Total | Headroom left in 64GB |
|---|---:|---:|---:|---:|
| 0.8B | 1.75 GB | 3.00 GB | 4.75 GB | **~59 GB** |
| 4B | 9.32 GB | 8.59 GB | 17.91 GB | **~46 GB** |
| 9B | 19.31 GB | 8.59 GB | 27.90 GB | **~36 GB** |

All three leave enormous headroom — the 64GB board is not remotely the bottleneck for any
candidate here, at any precision, at full native context. This matters for the recommendation
below: since fit is never in question, the choice should be driven by iteration speed and
narrative clarity, not by squeezing under a memory ceiling.

---

## 6. Full comparison table

| Field | 0.8B | 2B | 4B | 9B (upper ref.) |
|---|---|---|---|---|
| Instruct/chat repo | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) | [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) | [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) |
| Base repo | [`-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) | [`-2B-Base`](https://huggingface.co/Qwen/Qwen3.5-2B-Base) | [`-4B-Base`](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | [`-9B-Base`](https://huggingface.co/Qwen/Qwen3.5-9B-Base) |
| Total params (incl. vision+MTP) | 873.4M | 2.274B | 4.660B | 9.653B |
| Text-backbone params | 752.4M | 1.882B | 4.206B | 8.954B |
| `hidden_size` | 1024 | 2048 | 2560 | 4096 |
| `num_hidden_layers` | 24 | 24 | 32 | 32 |
| GDN : full-attn split | 18 : 6 | 18 : 6 | **24 : 8** | **24 : 8** |
| `linear_num_value_heads` | 16 | 16 | 32 | 32 |
| Native context | 262,144 | 262,144 | 262,144 | 262,144 |
| License | Apache-2.0 | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| FP16 weights (whole ckpt) | 1.75 GB | 4.55 GB | 9.32 GB | 19.31 GB |
| INT4 weights (whole ckpt) | 0.44 GB | 1.14 GB | 2.33 GB | 4.83 GB |
| Fits 64GB w/ room for 262K KV | Yes, trivially | Yes, trivially | Yes | Yes, but closest to the ceiling |
| Multimodal | Yes (no text-only ckpt) | Yes (no text-only ckpt) | Yes (no text-only ckpt) | Yes (no text-only ckpt) |
| Distance below "≤10B" NPU ceiling | 9.13B headroom | 7.73B headroom | 5.34B headroom | **0.35B headroom (tight)** |

---

## 7. Recommendation

### Primary: `Qwen/Qwen3.5-4B`

### Fallback (fast iteration): `Qwen/Qwen3.5-0.8B`

**Reasoning:**

- **The 64GB/precision fit is a non-issue at every size in this range** (§5) — so the choice
  should not be driven by "which one barely fits," it should be driven by what best demonstrates
  the O(1)-recurrent-state-vs-linear-KV-cache thesis, fast, within a 12-day sprint.
- **4B keeps the 32-layer / 24-GDN-8-full-attention split** that PLAN.md and
  CLAIM_VERIFICATION.md already treat as the canonical dense-checkpoint shape throughout the
  write-up narrative (§3 architecture background, the "24 GDN + 8 full attention" figure quoted
  repeatedly). Picking a 24-layer/18-GDN-6-full model (0.8B or 2B) as *primary* would mean
  every already-written narrative number needs re-deriving for a different split — avoidable churn.
- **4B is comfortably clear of the "within ten billion parameters" NPU ceiling** (5.34B of
  headroom), unlike 9B, which sits at 9.65B total parameters — only 0.35B under the vendor's
  stated ceiling before you've quantized anything, and it's ambiguous whether the ceiling is meant
  to count the vision tower + MTP head that come bundled in the checkpoint (§4). 4B removes that
  ambiguity entirely.
- **4B is ~2× smaller than 9B in weights and in full-context KV cache**, which matters directly
  because decode on the DeltaNet recurrence is *memory-bandwidth-bound* (confirmed finding in
  CLAIM_VERIFICATION.md §3) — on the O6's 100GB/s LPDDR5, a smaller model means faster per-token
  reads, faster benchmark repeats, and more sweep points (4K/32K/128K/262K) completed inside the
  schedule. Since the KV-cache-vs-recurrent-state story is architectural and present identically
  at every size, a 9B model buys no additional demonstration value for roughly double the
  iteration cost — exactly the "big model that cannot reach long context in the time available"
  failure mode the task asked to avoid. 4B can.
- **0.8B as fallback** shares the same Apache-2.0 license and the same 262,144 native context, so
  every finding (kernel gaps, quantization policy, dispatcher logic) developed against it transfers
  directly to 4B. Its only architectural difference is the 24/18/6 vs 32/24/8 split and the halved
  `linear_num_value_heads` (16 vs 32, §2) — both of which are worth exercising anyway, since they
  confirm the mapping/quantization code handles both shapes, not just one. At <2GB FP16 and <500MB
  INT4, it is the fastest possible loop for harness development, correctness-oracle wiring, and
  quantization-policy iteration before running the same pipeline against 4B.
- **9B is recorded here as the credible upper reference**, useful if schedule allows a "does the
  advantage hold at larger scale too" data point late in the sprint, but is not the primary choice
  given the above.
- 2B is not separately recommended: it shares 0.8B's 24-layer/18-6 split and shape family, so it
  adds no fallback value 0.8B doesn't already provide, at 2.6× the memory cost.

### What could not be verified from a primary source

- The exact byte/parameter size of the GDN recurrent state tensor per layer (needed to quantify,
  not just qualitatively assert, the "O(1) vs linearly-growing KV cache" memory story). This
  requires reading `Qwen3NextGatedDeltaNet` modeling code directly, already tracked as a distinct
  step (`t-arch-audit`) — deliberately not estimated here to avoid publishing an unverified number.
- Whether the "within ten billion parameters" NPU ceiling is meant to count vision-tower/MTP
  weights that ship bundled in every checkpoint, or the text backbone alone. Not stated in any
  Radxa/CIX source found. Flagged as ambiguous in §4/§7, not resolved.
- Real on-device tokens/sec, TTFT, or thermal behavior for any of these checkpoints on the Orion
  O6 — no hardware access at time of writing (per PLAN.md §2.1, board not yet in hand). Everything
  in this survey is config/weights-file-derived, not measured.
- Whether `causal_conv1d` / `fla` (or the `Atlas-Inference/gdn` Hub kernel) have any aarch64 build
  — explicitly out of scope for this survey and already tracked separately in
  CLAIM_VERIFICATION.md §4.

---

## Sources

- [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), [config.json](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json), [LICENSE](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/LICENSE)
- [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
- [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), [config.json](https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/config.json)
- [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base)
- [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [config.json](https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json), [LICENSE](https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/LICENSE)
- [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base)
- [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), [config.json](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json), [LICENSE](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/LICENSE)
- [Qwen/Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base)
- [Qwen3.5 — transformers docs](https://huggingface.co/docs/transformers/model_doc/qwen3_5) (`Qwen3_5TextConfig`, `Qwen3_5Config`, `Qwen3_5ForCausalLM` vs `Qwen3_5ForConditionalGeneration`)
- HuggingFace Hub API, `GET /api/models/<repo>` (`safetensors` dtype/parameter totals, `siblings` file listings) — queried directly for all eight repos above
- Raw `model.safetensors(.index).json` headers, fetched via HTTP range request — used to compute the `language_model` / `visual` / `mtp` parameter split in §4
- [`PLAN.md`](./archive/PLAN.md) and [`CLAIM_VERIFICATION.md`](./CLAIM_VERIFICATION.md) (ground truth this survey does not contradict)
