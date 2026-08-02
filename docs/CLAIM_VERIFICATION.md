# Claim verification against primary sources

Bead `ob-ofk` (`t-verify-claims`). `brief.md` mixes primary sources with weak secondary ones
(X posts, blog aggregators, a YouTube video). Judges will check our numbers, so every figure
we intend to quote is traced here to a primary source, corrected, or dropped.

Verified 2026-08-02.

---

## 1. Corrections — brief.md is wrong

### 1.1 The third track is **Edge AI**, not "Mobile AI" ⚠️

`brief.md` repeatedly refers to a "Mobile AI" track. The [official rules](https://arm-ai-optimization-challenge.devpost.com/rules)
list the three category prizes as:

| Prize | Amount |
|---|---|
| Best in Track — **Physical AI** | $1,000 |
| Best in Track — **Cloud AI** | $1,000 |
| Best in Track — **Edge AI** | $1,000 |

There is no Mobile AI track. This matters directly: it changes which track the hedge plan
targets, and "Edge AI" is a *better* fit for an Orion O6 inference box than "Mobile AI" ever
was — the O6 is an edge device, not a phone.

### 1.2 The "60 layers" figure is wrong for the models we care about

`brief.md` claims "3:1 linear:full attention across 60 layers", sourced from an X post. Primary
sources disagree:

| Checkpoint | Layers | Source |
|---|---|---|
| Qwen3.5 dense (default config) | `num_hidden_layers=32` | transformers `Qwen3_5TextConfig` |
| Qwen3.5-4B | 32, as `8 × (3× DeltaNet → FFN → 1× Attention → FFN)` | secondary, consistent with above |
| Qwen3.5-35B-A3B (MoE) | `num_hidden_layers=40` | transformers `Qwen3_5MoeTextConfig` |

For a 32-layer model at 3:1 that is **24 Gated DeltaNet layers and 8 full-attention layers** —
the concrete numbers our profiling and mapping work needs. Do not quote 60.

### 1.3 The 397B-A17B upper bound is unconfirmed

`brief.md` states the family runs "from 0.8B to 397B-A17B". The 0.8B floor is confirmed. The
397B-A17B ceiling appears only in weak sources; the transformers MoE docs document
`Qwen3.5-35B-A3B`, and at least one secondary source describes the family as topping out
around 122B. **Drop this figure** — it is irrelevant to us anyway, since we target the small
dense end.

### 1.4 Minor overstatements

| brief.md | Primary source |
|---|---|
| "over 100GB/s" memory bandwidth | "100GB/s" (128-bit LPDDR5 @ 5500MT/s) |
| License must be "visible in the GitHub About section" | Rules require a public repo under MIT or Apache-2.0; the About-section specificity is not in the rules text |

The About-section detail is worth doing regardless — it costs nothing and makes compliance
obvious to a judge — but it is our practice, not a stated requirement.

### 1.5 GDN-2's RULER gains are real but not quantified in the abstract

The paper confirms the retrieval advantage qualitatively: "Its advantage is most pronounced on
long-context RULER needle-in-a-haystack benchmarks, where it improves the evaluated multi-key
retrieval setting." No numeric RULER scores appear in the abstract. `brief.md`'s plan to
"directly cite the RULER retrieval gains" therefore needs the **full paper**, not the abstract.
Until then, describe the claim qualitatively.

---

## 2. Confirmed claims

### 2.1 Competition (source: [Devpost rules](https://arm-ai-optimization-challenge.devpost.com/rules))

| Claim | Status |
|---|---|
| Deadline 2026-08-14, 16:00 PT | ✅ Confirmed — 4:00pm **PDT** |
| $8,000 total: $3,000 overall / $2,000 runner-up / 3 × $1,000 track | ✅ Confirmed |
| Judging 40 technological / 25 WOW / 20 impact / 15 developer experience | ✅ Confirmed, sums to 100 |
| MIT or Apache-2.0, public open-source repo | ✅ Confirmed |
| Write-up: overview, functionality/output, step-by-step setup and validation | ✅ Confirmed |
| Video optional, under 3 minutes, shows project on intended device | ✅ Confirmed |

### 2.2 Orion O6 hardware (source: [radxa.com](https://radxa.com/products/orion/o6/), [Radxa docs](https://docs.radxa.com/en/orion/o6/app-development/artificial-intelligence))

| Claim | Status |
|---|---|
| Up to 45 TOPS combined (NPU + CPU + GPU) | ✅ Confirmed |
| NPU 28.8 TOPS | ✅ Confirmed |
| INT4 / INT8 / INT16 / FP16 / BF16 / TF32 | ✅ Confirmed |
| 12-core CPU: 4× A720 big @2.8GHz, 4× A720 medium @2.4GHz, 4× A520 little @1.8GHz, 12MB shared L3 | ✅ Confirmed — note the **three distinct frequency tiers**, which matters for affinity work |
| Immortalis G720 MC10 GPU | ✅ Confirmed |
| Up to 64GB, 128-bit LPDDR5 @ 5500MT/s | ✅ Confirmed |
| ~30 tokens/sec on Qwen2-1.5B | ✅ Confirmed — vendor figure, unstated precision and context length, so treat as a rough sanity target rather than a benchmark |
| Models "within ten billion parameters" | ✅ Confirmed as vendor wording |
| CIX Early Bird Program required for latest P1 software | ✅ Confirmed |

The NOE Compiler's Python 3.8 pin could **not** be confirmed — the cited Radxa NPU SDK page
now 404s. Treat as unverified and re-check during `t-py38-noe`.

### 2.3 Qwen3.5 architecture (source: [transformers docs](https://huggingface.co/docs/transformers/model_doc/qwen3_5))

| Claim | Status |
|---|---|
| 3:1 hybrid: three Gated DeltaNet layers per one Gated Attention layer | ✅ **Confirmed by primary source** — the load-bearing claim holds |
| Small dense variants exist at 0.8B, 2B, 4B, 9B | ✅ Confirmed — PLAN.md's 0.8B–4B target range is viable |
| 262K native context | ✅ Confirmed: "Native context is 262,144 tokens." ~1M requires **YaRN rope scaling** via `rope_scaling`; plain loading gives the native window only |

Concretely useful details found:

- `layer_types` on the text config is a per-layer list of `"linear_attention"` / `"full_attention"`.
  **This is exactly the introspection hook `t-arch-audit` needs** — the hybrid layout is readable
  from config, no modeling-code archaeology required.
- Linear-attention layer shapes: `linear_conv_kernel_dim=4`, `linear_key_head_dim=128`,
  `linear_value_head_dim=128`, `linear_num_key_heads=16`, `linear_num_value_heads=32`.
- The DeltaNet path is `Qwen3NextGatedDeltaNet` — Qwen3.5's text backbone reuses Qwen3-Next's
  linear-attention decoder, so Qwen3-Next tooling is likely to transfer.
- Dense and MoE checkpoints share the same GDN core (`Qwen3_5MoeGatedDeltaNet` ≡
  `Qwen3_5GatedDeltaNet`) but have very different shapes; weights are not interchangeable.

---

## 3. Finding that changes the technical thesis 🔴

The transformers docs report measured kernel numbers that we should absorb *before* setting
expectations. On an NVIDIA GB10, swapping the slow PyTorch GDN fallback for an optimized kernel:

| Checkpoint | TTFT (prefill) | Decode |
|---|---|---|
| Qwen3.6-27B dense | 1.66 s → 1.11 s (**1.49× faster**) | 4.11 → 4.14 tok/s (flat) |
| Qwen3.6-35B-A3B MoE | 0.73 s → 0.53 s (**1.38× faster**) | 16.3 → 16.7 tok/s (flat) |

With the upstream explanation:

> "Decode is roughly flat because the single-token DeltaNet recurrence is memory-bandwidth-bound;
> the win is on the chunked-prefill core and grows with prompt length."

### What this means for us

**Optimizing GDN kernels buys prefill/TTFT, not decode throughput.** The single-token recurrence
is bandwidth-bound, and no amount of kernel cleverness changes that — on the O6, with 100GB/s of
LPDDR5, it will be *more* bandwidth-bound than on a GB10, not less.

So the two honest claims to build the submission around are:

1. **Prefill / TTFT throughput** — where kernel work genuinely pays, and where the win *grows
   with context length*. This is the right target for the GPU scan kernel (`t-gpu-scan`) and the
   NPU offload.
2. **Decode memory footprint** — O(1) recurrent state versus a KV cache growing linearly with
   context. This is an architectural property, not a kernel optimization, and it is what makes
   long context feasible on a 64GB edge board at all.

What we should **not** promise is a large decode tokens/sec win from GDN kernel optimization.
Physics is against it, and claiming it would be the kind of overstatement that collapses under a
judge's scrutiny. Better to predict the flat decode result, measure it, and explain *why* — that
reads as competence rather than a miss.

This also sharpens `t-harness-mem` from "nice measurement" to **the** measurement: the memory
decomposition is the load-bearing evidence, and prefill scaling is the performance headline.

### A second finding: the gap our project fills

> "The DeltaNet path needs the optional `causal_conv1d` and `fla` packages for its fast kernels —
> without them, the model silently falls back to slower and more memory hungry PyTorch ops."

Upstream already documents an architecture-specific hole: on NVIDIA GB10 (SM121) neither package
ships a build, so the fast path silently degrades. **Arm/Vulkan is the same class of hole, wider.**
That is the concrete, verifiable statement of what this project contributes — not "we optimized a
model" but "the fast GDN kernels do not exist for this silicon, and here is what we built and
measured instead." Upstream's own workaround is a Hub kernel (`Atlas-Inference/gdn`), which is a
useful precedent for how a per-architecture GDN kernel gets packaged.

---

## 4. Remaining unverified

| Item | Next step |
|---|---|
| NOE Compiler Python 3.8 pin | Radxa NPU SDK page 404s — re-check during `t-py38-noe` |
| GDN-2 numeric RULER scores | Read the full paper (`t-gdn2-read`) |
| Per-checkpoint layer counts and `layer_types` for the exact model chosen | Read `config.json` directly during `t-arch-audit` |
| Whether `causal_conv1d` / `fla` have any aarch64 build | Investigate during `t-x86-ref` / `t-gpu-scan` — determines our baseline |

## Sources

- [Arm Create: AI Optimization Challenge — Rules](https://arm-ai-optimization-challenge.devpost.com/rules)
- [Radxa Orion O6 product page](https://radxa.com/products/orion/o6/)
- [Radxa Orion O6 — Artificial Intelligence docs](https://docs.radxa.com/en/orion/o6/app-development/artificial-intelligence)
- [Qwen3.5 — transformers docs](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Qwen3.5 MoE — transformers docs](https://huggingface.co/docs/transformers/model_doc/qwen3_5_moe)
- [Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention (arXiv 2605.22791)](https://arxiv.org/abs/2605.22791)
- [Atlas-Inference/gdn Hub kernel](https://huggingface.co/kernels/Atlas-Inference/gdn)
