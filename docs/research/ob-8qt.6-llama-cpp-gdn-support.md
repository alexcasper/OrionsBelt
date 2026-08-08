# ob-8qt.6 — Does llama.cpp support Qwen3.5 GDN at all?

**Bead:** `ob-8qt.6` · **Task:** "Does llama.cpp support Qwen3.5 GDN at all?" · **Date:** 2026-08-04
**Type:** READ-ONLY research (web + repo context). No code, hardware, git, or dolt ops.

---

## VERDICT (one line)

**SUPPORTED — upstream llama.cpp runs Qwen3.5 GDN (Gated DeltaNet) end to end,
including the chunk-to-chunk recurrence that NOE could not compile.** It is the
**exception that proves the rule**, not the third instance of the CIX/NOE/KleidiAI
tooling gap. The quantized (GGUF) route **exists**.

Confidence: **high** for upstream llama.cpp (multiple independent primary sources).
The CIX board's *bundled* `cix-llama-cpp` build (v1.3.1) is a separate, **UNCONFIRMED**
sub-question — see §5.

---

## Why this was a real question

Going in, the prior evidence pointed toward "no":

- **CIX AI Model Hub** ships 38 LLMs, **zero** linear-attention models
  ([FINDINGS.md §2](../FINDINGS.md)). The absent thing is the architecture class.
- **NOE Compiler** can lower all of GDN's arithmetic but **rejects the recurrence**
  (`Scan` rejected; runtime-trip-count `Loop` rejected) — [FINDINGS.md §1](../FINDINGS.md).
- **KleidiAI** has **no recurrence / prefix-scan primitive** of any kind
  ([FINDINGS.md §3.3](../FINDINGS.md)).

So a third "no" from llama.cpp would have been a consistent result. It is not what
the evidence shows.

---

## 1. CONFIRMED evidence (primary sources, verified directly)

### 1a. The llama.cpp maintainers publish an official GGUF of a Qwen3.5 GDN model

`ggml-org/Qwen3.5-35B-A3B-GGUF` is published by the **official `ggml-org` account**
(the org behind llama.cpp / ggml). The model card records:

- **Architecture:** `qwen35moe` (Qwen3.5 Mixture-of-Experts)
- **Quantization:** `Q8_0` (36.9 GB), converted from `Qwen/Qwen3.5-35B-A3B-Base`
- **Recommended run command:** `llama-server -hf ggml-org/Qwen3.5-35B-A3B-GGUF`

That the maintainers themselves convert, publish, and document a run command for a
Qwen3.5 model is, by itself, definitive proof the architecture is supported upstream.

Source: <https://huggingface.co/ggml-org/Qwen3.5-35B-A3B-GGUF>

### 1b. Dense (non-MoE) Qwen3.5 GGUFs that the project's sizes depend on

The project targets **Qwen3.5-4B** and **Qwen3.5-0.8B** (dense). GGUFs exist for both:

- <https://huggingface.co/unsloth/Qwen3.5-4B-GGUF>
- <https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF>
- <https://huggingface.co/lmstudio-community/Qwen3.5-0.8B-GGUF>

These conversion targets only work because `convert_hf_to_gguf.py` recognises the
architecture. (The communityGGUF provenance is reported via search; the lmstudio-
community card explicitly targets "local apps" i.e. llama.cpp-based runtimes.)

### 1c. `convert_hf_to_gguf.py` explicitly handles `qwen3next` and `qwen3.5`

PR **#19139** ("llama: Add option to merge gate and exp weights") modifies
`convert_hf_to_gguf.py`, and its diff is "reduced to just deepseek2 (GLM Flash) and
**qwen3next and qwen3.5**." This confirms both the Qwen3-Next and Qwen3.5
architectures are registered in the converter — the GDN hybrid lineage.

Source: <https://github.com/ggml-org/llama.cpp/pull/19139>

### 1d. The recurrence runs token-to-token (the exact thing NOE could not compile)

This is the load-bearing question for this project, because NOE failed specifically
on the sequential recurrence, not on the arithmetic. llama.cpp **does** maintain and
update the GDN recurrent state:

- The codebase has a dedicated recurrent-state module
  (`llama-memory-recurrent.h`) and public model-class predicates
  `llama_model_is_recurrent(model)` and `llama_model_is_hybrid(model)`.
- The ik_llama.cpp fork issue #1762 (tracking the same upstream code path) confirms
  the recurrent state is computed every token — its bug is that for `qwen3next`,
  "`pos_min` always equals the full sequence length" in checkpoint validity checks,
  and the fix gates on `llama_model_is_recurrent(model) || llama_model_is_hybrid(model)`.
- The **limitation is multi-sequence batching**, not the recurrence: when a second
  sequence is decoded concurrently, the server logs
  `qwen3next mixed-sequence batch contains repeated seq_id values; falling back to
  single-token chunking` (decode drops ~21 t/s → ~0.59 t/s). This is an architectural
  property of delta-rule recurrent layers (state is sequence-dependent), not a
  missing operator.

Sources:
- <https://github.com/ikawrakow/ik_llama.cpp/issues/1762>
- <https://github.com/ggml-org/llama.cpp/issues/23817> (see 1e)

### 1e. Issue #23817 — "stateful API for recurrent/SSM models" (closed, May 2026)

A feature request asking to expose recurrent state in serialization. It **confirms
the models are already supported** and names Gated DeltaNet and Mamba explicitly:

> "recurrent state buffers already exist internally, they are not currently reusable
> in the same way as KV cache."

The gap is **state save/restore (session checkpointing)**, not inference. The model
runs; you just can't yet snapshot and resume its recurrent state cleanly. This is the
opposite problem from NOE (which can't *run* the recurrence at all).

Source: <https://github.com/ggml-org/llama.cpp/issues/23817>

### 1f. Real users running it, with measured throughput

The `AesSedai/Qwen3.5-35B-A3B-GGUF` discussion shows users running the GDN hybrid on
consumer GPUs after compiling the relevant PRs, reporting **prefill ~800 tok/s** (up
from 700) on a 4070 12GB after PR #19139, with a follow-up fix in PR #20416 for
`--n-cpu-moe` offloading with fused gate+up tensors.

- `AaryanK/Qwen3.5-9B-GGUF` states the architecture is **"Hybrid Gated DeltaNet +
  MoE"** and instructs users to "use the absolute latest version of `llama.cpp` to
  support these new operators," with a working `llama-cli` invocation.

Sources:
- <https://huggingface.co/AesSedai/Qwen3.5-35B-A3B-GGUF/discussions/6>
- <https://huggingface.co/AaryanK/Qwen3.5-9B-GGUF>
- <https://github.com/ggml-org/llama.cpp/pull/20416>

### 1g. Architecture pattern independently corroborated

A public gist analysing Qwen3.5 confirms the **3 GDN : 1 full-attention** interleaving
("every 4th layer is full attention"), giving 24 linear + 8 full-attention for a
32-layer model. **This exactly matches the project's own verified layer layout**
(`src/orionsbelt/model/gdn_layer_info.py`: "32 layers: 24 GDN + 8 full-attention,
pattern 8×(3 GDN → 1 full)"). Same model, same architecture.

Source: <https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15>

---

## 2. The GDN lineage in llama.cpp (how support arrived)

`Qwen3-Next` was the first Qwen model to ship the GDN hybrid (released **September
2025**). llama.cpp tracked it in issue **#15940** ("a hybrid model with a custom SSM")
and landed support in **PR #16095** (announced "Qwen3-Next support in llama.cpp almost
ready!"). **Qwen3.5 reuses the same GDN hybrid**, so its support builds on that base —
which is why PR #19139 names both `qwen3next` and `qwen3.5` together (§1c).

Sources (reported via search; URLs are primary):
- Tracking issue: <https://github.com/ggml-org/llama.cpp/issues/15940>
- Foundational PR: <https://github.com/ggml-org/llama.cpp/pull/16095>
- Unsloth "Run Locally" guide dates Qwen3-Next to Sept 2025

---

## 3. How llama.cpp handles the recurrence NOE couldn't compile

This is the analytical core, and it explains *why* llama.cpp is the exception.

NOE is a **static compiler** that must lower a graph to fixed-function NPU IR. It
rejects `Scan` outright and rejects any `Loop` whose trip count is a **runtime**
input (FINDINGS.md §1) — it can only unroll loops with compile-time-constant trip
counts, which cannot express variable-length decode.

llama.cpp is a **CPU-first interpreter**. The GDN recurrence is just a sequential C
loop over the time axis with a caller-owned state pointer — **structurally identical
to what the project's own `gdn_sve.c` does** (`state[]`/`hist[]` are explicit
caller-owned buffers; FINDINGS.md §4). There is no compiler IR that has to "express"
a dynamic trip count, and no fixed micro-kernel that has to be "shipped" for the scan
(KleidiAI's gap, FINDINGS.md §3.3). The recurrence is simply code in the model's
forward pass.

So the tooling gap is **specific to accelerator compilers (NOE) and fixed kernel
libraries (KleidiAI)**, not to CPU inference engines. A CPU-first engine sidesteps
the entire problem class — which is exactly the design premise of docs/archive/PLAN.md §3.1
("CPU hosts the GDN recurrence"). llama.cpp is existence proof that the premise is
sound at the *engine* level, even though it fails at the *accelerator-compiler*
level.

One nuance worth recording: on backends that lack a native GDN kernel (e.g. the
search-reported CANN/Ascend case where "the fused Gated Delta Net tensor is assigned
to device CPU (usually due to missing support)"), llama.cpp **falls back to CPU for
the GDN ops and the model still runs**. That is the same CPU-hosts-recurrence mapping
the project intends — except in llama.cpp it is a working fallback, whereas in NOE
the recurrence had no home at all.

---

## 4. Cross-reference: the "third instance of the gap" hypothesis

| Tool | GDN arithmetic | GDN recurrence | Verdict |
|---|---|---|---|
| CIX Model Hub (38 LLMs) | — | — | **gap** — zero linear-attention models shipped |
| NOE Compiler | ✅ all ops | ❌ `Scan`/runtime-`Loop` rejected | **gap** |
| KleidiAI kernels | ✅ 109 matmuls | ❌ no scan/prefix-product primitive | **gap** |
| **llama.cpp** | ✅ | ✅ runs token-to-token | **EXCEPTION** |

llama.cpp is **not** the third instance of the gap — it is the exception that proves
the rule. The rule that emerges is sharper than "the ecosystem lags": **the gap is
confined to ahead-of-time accelerator toolchains; general-purpose CPU inference
engines already run GDN.** This is a more useful and more honest framing than either
"nothing supports it" or "everything supports it."

---

## 5. UNCONFIRMED — the CIX board's bundled build

FINDINGS.md §2 notes the SDK "does ship a vendor `cix-llama-cpp` build (1.3.1) for the
board ... though its GDN support is unverified." This research resolves the
*upstream* question (yes) but **not the vendor-build question**:

- GDN support in llama.cpp dates to **PR #16095** (Qwen3-Next, ~late 2025) with
  Qwen3.5-specific handling in **PR #19139**. These are high-numbered PRs.
- The vendor tag **`1.3.1`** is, on its face, an old/low version number. By version
  numbering alone it is **INFERRED** to predate the Qwen3-Next/Qwen3.5 GDN work —
  i.e. the board's shipped binary very likely **cannot** run Qwen3.5-GDN without a
  rebuild against a current llama.cpp HEAD.

**What would confirm it:** on the board, `cix-llama-cpp --version` (or `llama-cli
--version`) to read the build commit/number, then check whether that commit is
descended from PR #16095. A build number below ~16095 cannot run GDN. (Board-gated;
tracked under `ob-8xc`.)

This is the single most important follow-up: **upstream support does not imply the
board's binary has it.** Rebuilding `cix-llama-cpp` from a current llama.cpp HEAD for
aarch64 is straightforward and is the most likely way to get a known-good GGUF
inference path on the O6.

---

## 6. Implications

### For `ob-8qt.5` (plan) — the quantized route exists; the plan stands

The plan's premise that a GGUF/llama.cpp quantized inference route is available is
**vindicated**, not overturned. Q4_K_M GGUFs of Qwen3.5-4B/0.8B are real and runnable.
Two consequences:

1. **llama.cpp is a defensible baseline**, not a dead end. A Q4_K_M run via
   `llama-cli`/`llama-server` on Cortex-A720 (or the fleet's A76/A57 devices) is a
   legitimate "off-the-shelf quantized inference" comparison number for the write-up.
2. **It does not obsolete the project's custom kernels.** llama.cpp's generic GDN
   recurrence is plain C; it will not have the SVE2/NEON-optimised, predicated-tail
   micro-kernels the project is writing (`gdn_sve.c`, FINDINGS.md §4), nor the
   INT4/i8mm KleidiAI delta-rule matmuls (ADR [0006](../adr/0006-quantization-policy.md)).
   The contribution (three named kernels upstreamable to KleidiAI that exist nowhere
   today for non-SME Armv9.2) is unchanged — llama.cpp is the *baseline to beat*,
   and the CPU-hosted-recurrence design (docs/archive/PLAN.md §3.1) is now backed by an
   independent engine-level existence proof.

### For `ob-mrd.2` (disk budget) — NOT moot; the checkpoint is a live target

Because the quantized route **exists**, the disk budget for a quantized checkpoint is
**relevant, not moot.** A Q4_K_M of Qwen3.5-4B is ~2.3–2.5 GB (vs ~8 GB FP16);
Q8_0 is ~4.3 GB; the 0.8B equivalents are proportionally smaller. The budget should
plan for at least one Q4_K_M and one Q8_0 of the primary checkpoint (4B), plus the
0.8B secondary. (Exact byte sizes should be read from a chosen GGUF file listing
rather than estimated here.)

### For `ob-8xc` (board) — add one task

Verify/refresh the board's `cix-llama-cpp`: read its build commit, and if it predates
PR #16095, rebuild from a current llama.cpp HEAD for aarch64. Only then does the
upstream SUPPORTED verdict apply on the O6 itself.

---

## 7. Sources (primary)

**Verified directly (WebFetch of the live page):**
- Official maintainer GGUF, arch `qwen35moe`: <https://huggingface.co/ggml-org/Qwen3.5-35B-A3B-GGUF>
- PR #19139 (names `qwen3next` + `qwen3.5` in `convert_hf_to_gguf.py`): <https://github.com/ggml-org/llama.cpp/pull/19139>
- Issue #23817 (recurrent state buffers exist; GDN/Mamba supported; checkpointing is the gap): <https://github.com/ggml-org/llama.cpp/issues/23817>
- ik_llama.cpp #1762 (recurrent state computed per-token; `llama_model_is_recurrent/_is_hybrid`; multi-seq limitation): <https://github.com/ikawrakow/ik_llama.cpp/issues/1762>
- AaryanK/Qwen3.5-9B-GGUF ("Hybrid Gated DeltaNet + MoE", `llama-cli` instructions): <https://huggingface.co/AaryanK/Qwen3.5-9B-GGUF>
- AesSedai/Qwen3.5-35B-A3B-GGUF discussion (runs; ~800 tok/s prefill; PR #19139/#20416): <https://huggingface.co/AesSedai/Qwen3.5-35B-A3B-GGUF/discussions/6>
- Qwen3.5 arch gist (3:1 GDN:attention, 24+8 — matches project `gdn_layer_info.py`): <https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15>

**Reported via search (URLs primary, content not separately fetched):**
- Foundational Qwen3-Next PR #16095: <https://github.com/ggml-org/llama.cpp/pull/16095>
- Qwen3-Next tracking issue #15940: <https://github.com/ggml-org/llama.cpp/issues/15940>
- PR #20416 (`--n-cpu-moe` fix for fused gate+up): <https://github.com/ggml-org/llama.cpp/pull/20416>
- Dense 4B GGUF: <https://huggingface.co/unsloth/Qwen3.5-4B-GGUF>
- Dense 0.8B GGUF: <https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF>, <https://huggingface.co/lmstudio-community/Qwen3.5-0.8B-GGUF>
- "Qwen3.5 Support Merged in llama.cpp" (r/LocalLLaMA): <https://www.reddit.com/r/LocalLLaMA/comments/1qzppr7/qwen35_support_merged_in_llamacpp/>
- "Qwen3-Next support in llama.cpp almost ready!" (r/LocalLLaMA): <https://www.reddit.com/r/LocalLLaMA/comments/1p5by1a/qwen3next_support_in_llamacpp_almost_ready/>

**Project-internal cross-references:**
- [FINDINGS.md §1](../FINDINGS.md) (NOE recurrence rejection), [§2](../FINDINGS.md) (CIX hub; vendor `cix-llama-cpp` 1.3.1), [§3.3](../FINDINGS.md) (KleidiAI), [§4](../FINDINGS.md) (`gdn_sve.c`)
- [ADR 0006 — quantization policy](../adr/0006-quantization-policy.md)
- [ADR 0002 — portable hedge target](../adr/0002-portable-hedge-target.md) (already cites llama.cpp for Vulkan/Metal/Termux)
