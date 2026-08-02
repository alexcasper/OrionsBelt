# OrionsBelt — Implementation Plan

**Project:** Deploy and optimize a Qwen3.5-family model (Gated DeltaNet hybrid architecture) on Arm silicon, primarily the Radxa Orion O6 (CIX P1 SoC), with a Gated DeltaNet-2 research comparison as the differentiator.

**Competition:** [Arm Create: AI Optimization Challenge](https://arm-ai-optimization-challenge.devpost.com/)
**Submission deadline:** 2026-08-14, 16:00 PT
**Plan authored:** 2026-08-02 (**T-12 days**)
**Source brief:** [`brief.md`](./brief.md)
**Issue tracking:** [beads](https://beads.gascity.com/) (`bd`), prefix `ob-` — see [`docs/BEADS.md`](./docs/BEADS.md)

---

## 1. Executive summary

The technical thesis is strong and unusually well-matched to the competition: **Gated DeltaNet (GDN) linear-attention layers decode in O(1) memory per token**, where full attention grows a KV cache linearly with context. That advantage is *most visible precisely where memory is scarce* — edge silicon. Demonstrating it concretely on an Arm SoC, with honest before/after numbers, hits the rubric's "Arm-specific optimization" (40 pts) and "WOW factor" (25 pts) at the same time.

The **schedule, not the engineering, is the primary risk.** As of today we hold neither the Orion O6 board nor CIX Early Bird SDK access, and the deadline is 12 days out. Board procurement plus NPU toolchain enrollment are both externally gated — no amount of effort compresses them. This plan therefore runs **two tracks in parallel from day one**:

| Track | Target | Devpost track | Gated on |
|---|---|---|---|
| **Primary** | Radxa Orion O6 (CIX P1: 12-core CPU + Immortalis G720 + 28.8 TOPS NPU) | Physical AI | Board arrival + CIX Early Bird approval |
| **Hedge** | Generic aarch64 (Android phone via Termux, or Apple silicon / AWS Graviton) | Edge AI | Nothing — starts immediately |

> **Corrected 2026-08-02:** the three tracks are Physical AI, Cloud AI, and **Edge AI**. `brief.md`'s
> "Mobile AI" track does not exist — see [`docs/CLAIM_VERIFICATION.md`](./docs/CLAIM_VERIFICATION.md) §1.1.
> Edge AI is a better fit than Mobile AI ever was: the O6 is an edge device, not a phone, so *both*
> tracks remain open to an O6 submission and the hedge no longer requires a phone specifically.

Everything genuinely portable — benchmark harness, correctness oracle, model selection, GDN architecture audit, GDN-2 research, write-up scaffolding — is built **hardware-independent** and runs on either target. A **hard go/no-go on 2026-08-09** picks the track based on what hardware actually exists by then. The hedge is not a consolation prize: the GDN memory-scaling story is fully demonstrable on any aarch64 device with Arm i8mm/SVE and a Vulkan or OpenCL GPU, and Edge AI is a legitimate prize category.

**Explicit non-goal:** we are not pretending to a result we cannot measure. If the NPU never becomes accessible, we ship a CPU+GPU hybrid with rigorous numbers and say so plainly. A reproducible, honestly-reported partial result scores better than an unverifiable claim.

---

## 2. Constraints and ground truth

### 2.1 Hard constraints

| Constraint | Value | Source |
|---|---|---|
| Submission close | 2026-08-14 **16:00 PDT** | Devpost rules (verified) |
| License | OSI-approved (**MIT or Apache-2.0**), public repo | Devpost rules — hard requirement. Making it visible in the About section is our practice, not a stated rule |
| Repository | Public, open source | Devpost rules |
| Write-up | Project overview, functionality/output, setup instructions | Devpost rules |
| Newness | Built or significantly updated during submission period | Devpost rules |
| Benchmarking | Arm Performix for standardized reporting | Challenge brief |
| Demo video | Optional, <3 min, YouTube/Vimeo/Youku | Devpost rules |
| NOE Compiler | **Python 3.10** (verified). Runs on an **x86 host**, direct download, no approval gate | Radxa env-setup docs |
| NPU model ceiling | models "within ten billion parameters" | Radxa docs (verified) |

### 2.2 Judging rubric → where each point lands

The rubric is the spec. Every workstream below is justified by a line here.

| Criterion | Points | Our answer |
|---|---:|---|
| Technological implementation / Arm-specific optimization | 40 | Heterogeneous NPU+GPU+CPU partitioning of GDN layers; INT4/INT8 NPU quantization; Vulkan compute-shader chunkwise scan; big.LITTLE affinity; i8mm/SVE GEMM paths |
| WOW factor | 25 | Dynamic heterogeneous dispatcher + GDN-2 research comparison — novel architecture optimized *before* broad tooling support exists |
| Potential impact | 20 | Reusable reference implementation, documented op-coverage findings, migration template for GDN-class models on Arm |
| Developer experience | 15 | Clean-clone reproduction rehearsal, scripted setup, CI, run manifests |

### 2.3 Claim verification — **done**, see [`docs/CLAIM_VERIFICATION.md`](./docs/CLAIM_VERIFICATION.md)

Bead `ob-ofk` verified `brief.md`'s load-bearing figures against primary sources on 2026-08-02.
Summary:

**Confirmed:** the 3:1 Gated DeltaNet-to-full-attention ratio (primary source, and it is the
claim the whole project rests on); small dense variants at 0.8B / 2B / 4B / 9B, so the planned
0.8B–4B target range is viable; 262K native context (~1M only via YaRN rope scaling); 45 TOPS
combined and 28.8 TOPS NPU with INT4–TF32; the 12-core three-tier CPU; ~30 tok/s on Qwen2-1.5B
as a vendor figure; and every competition fact (deadline, $8,000 prize split, 40/25/20/15 rubric,
MIT-or-Apache-2.0).

**Corrected:** the third track is **Edge AI**, not "Mobile AI"; the "60 layers" figure is wrong
for the models we care about (dense default is 32 layers → **24 GDN + 8 full attention** at 3:1;
MoE 35B-A3B is 40); the 397B-A17B upper bound is unconfirmed and dropped; memory bandwidth is
100GB/s, not "over" it; and GDN-2's RULER gains are real but not quantified in the abstract, so
they must be described qualitatively until the full paper is read.

**Corrected again (2026-08-02, after the CIX wiki pointer):** the NOE Compiler needs **Python 3.10,
not 3.8**; the SDK is a **direct download with no approval gate**; and it **runs on an x86 host**.
This dissolves most of risk R2 and makes the NOE op-coverage audit for GDN operators actionable
*without the board* — see `docs/CLAIM_VERIFICATION.md` §2.2a.

### 2.4 Verified finding that reshapes the thesis

Upstream publishes measured GDN kernel numbers: optimizing the DeltaNet path speeds up **prefill
by 1.38–1.49×** and leaves **decode flat**, because "the single-token DeltaNet recurrence is
memory-bandwidth-bound." On the O6's 100GB/s LPDDR5 this will bind *harder* than on the GB10 those
numbers came from.

So the two claims worth building the submission around are:

1. **Prefill / TTFT throughput** — where kernel work genuinely pays, and where the advantage
   *grows with context length*. The right target for `t-gpu-scan` and the NPU offload.
2. **Decode memory footprint** — O(1) recurrent state versus a linearly-growing KV cache. An
   architectural property rather than a kernel win, and the thing that makes long context feasible
   on a 64GB board at all.

What we must **not** promise is a large decode tokens/sec win from GDN kernel optimization. Better
to predict the flat decode result, measure it, and explain why — that reads as competence, whereas
claiming it would collapse under scrutiny. This promotes `t-harness-mem` from a nice measurement to
*the* measurement.

There is also a verified statement of what this project contributes: the fast GDN kernels
(`causal_conv1d`, `fla`) have no build for some architectures, and upstream documents the model
*silently* falling back to slower, more memory-hungry PyTorch ops when they are missing. Arm/Vulkan
is the same hole, wider. That is the contribution — not "we optimized a model" but "the fast kernels
do not exist for this silicon, and here is what we built and measured instead."

### 2.5 Reconciling the two briefs

`brief.md` contains two overlapping agent briefs: a Qwen3.5/GDN-specific one and an earlier generic Orion O6 one. The generic brief's "Phase 1 — Environment and toolchain setup" is a strict subset of this plan's toolchain workstream (E2); its Phase 2 optimization-target selection is subsumed by the GDN-specific framing, which is the stronger differentiator. Where they conflict, **the GDN-specific brief wins**.

---

## 3. Architecture background (also the README's job)

Judges may not know GDN. The repo must explain it, because an unexplained differentiator is not a differentiator.

| Feature | Gated DeltaNet (Qwen3.5) | Gated DeltaNet-2 |
|---|---|---|
| Gating | Single scalar gate ties erase + write | Separate channel-wise erase (`b_t`) and write (`w_t`) gates |
| Relation | Combines Mamba2 decay + delta rule | Generalizes GDN and KDA as special cases |
| Integration | Shipping in Qwen3.5 — 3:1 hybrid, so a 32-layer dense checkpoint is **24 GDN + 8 full attention** | Not yet in a released Qwen checkpoint |
| Claimed strength | Long-context efficiency; 262K native context (~1M via YaRN) | Long-context RULER multi-key retrieval (qualitative in the abstract; no published numbers yet) |
| Reference | `Qwen3NextGatedDeltaNet`; layout readable from `layer_types` in the HF config | `github.com/NVlabs/GatedDeltaNet-2` — Hatamizadeh, Choi, Kautz; [arXiv 2605.22791](https://arxiv.org/abs/2605.22791) |

Verified layer shapes for the dense config: `linear_conv_kernel_dim=4`, `linear_key_head_dim=128`,
`linear_value_head_dim=128`, `linear_num_key_heads=16`, `linear_num_value_heads=32`. The hybrid
layout does not require reading modeling code — `layer_types` is a per-layer list of
`"linear_attention"` / `"full_attention"`, which is the introspection hook `t-arch-audit` needs.

**The kernel gap, stated precisely.** Upstream documents that the GDN fast path needs the optional
`causal_conv1d` and `fla` packages, and that without them the model *silently* falls back to slower,
more memory-hungry PyTorch ops. It already happens on NVIDIA GB10 (SM121), where neither package
ships a build. **Arm/Vulkan is the same hole, wider** — and that is the verifiable one-line statement
of what this project contributes.

**Why this is hard on an NPU.** GDN layers are not standard attention. The chunkwise WY-style recurrent update — delta rule + gated decay + causal Conv1D — is a *sequential scan* over chunk states. NPU accelerators are tuned for dense matmuls, and the CIX NOE Compiler may have no kernel for a gated recurrent scan at all. The interesting engineering question, and the core of our contribution, is therefore:

> Which layers go on which engine? Do GDN scan layers run on GPU/CPU while the periodic full-attention layers and MoE FFN blocks (dense matmul, NPU-friendly) go to the NPU — and what does the handoff cost?

Answering that with measurements, and documenting the op-coverage gaps we find, is reusable value for anyone porting a GDN-class model to Arm.

---

## 4. Workstreams (epics)

Ten epics, `ob-` prefixed in beads. Full dependency graph in §6.

### E0 — Program spine
Go/no-go decisions, risk register, descope ladder. Deliberately a first-class epic: with 12 days and external gates, *deciding on time* is the deliverable most likely to save the submission.

### E1 — Repo foundation and submission hygiene
Apache-2.0 license visible in About (day one — hard requirement), directory skeleton, Python tooling (uv/ruff/pytest), CI, results schema contract, run-manifest provenance capture. The results schema is an early dependency because the harness, the plots, and the final table all bind to it; changing it late is expensive.

### E2 — Hardware and toolchain acquisition (externally gated)
Board procurement, CIX Early Bird enrollment, Debian 12 flash and bring-up, Python 3.8 + NOE Compiler, NPU runtime smoke test, Vulkan/OpenCL validation, perf tooling. **Both procurement beads are P0 with no dependencies — they start immediately and block a large fraction of the graph.**

### E3 — Portable Arm hedge track
Select and bring up a generic aarch64 target; build llama.cpp with i8mm/SVE; run baselines. Keeps a viable **Edge AI** submission alive independent of the O6.

### E4 — Model selection and reference baseline
Survey Qwen3.5 checkpoints in the 0.8B–4B range (license, GDN layer config, NPU ceiling fit), pick primary + fallback, audit the GDN layer structure from modeling code, stand up an x86/CUDA reference as the **correctness oracle**, define the quantization policy (which layers must stay FP16 — recurrent state and gates are the obvious candidates).

### E5 — Benchmark harness and methodology
Precise metric definitions, context sweep (4K / 32K / 128K / 262K), warmup + repeats + percentiles, memory instrumentation that *separates KV cache from GDN recurrent state* (this separation is the whole point — it's what makes the scaling advantage visible), energy sampling, Arm Performix integration, long-context prompt corpus, plotting, CI smoke tests.

### E6 — GDN operator analysis and heterogeneous mapping
Per-layer latency profile (GDN linear vs full-attention vs MoE/FFN), characterize the recurrent-scan bottleneck, audit NOE Compiler op coverage for GDN ops, then an **ADR fixing the layer→engine assignment**, and the partitioned execution runtime.

### E7 — Optimization implementation
NPU subgraph export + INT8→INT4 quantization with accuracy regression against the oracle; Vulkan/OpenCL compute shader for the chunkwise gated delta-rule scan (numerically validated); big.LITTLE affinity and i8mm/SVE CPU paths; the **dynamic heterogeneous dispatcher**; sustained-load thermal characterization; and the ablation matrix that produces the headline table.

### E8 — GDN-2 stretch research
Read the paper and record the hypothesis as an ADR; clone and smoke-test the NVLabs reference; then decide between **(a)** microbenchmarking GDN-2 vs GDN gating on-device, or **(b)** a small-scale GDN-2 layer swap into a Qwen3.5-architecture checkpoint evaluated on RULER multi-key retrieval. Option (a) is cheap and safe; (b) is higher-reward and much riskier at this timeline. Default to (a), escalate to (b) only if ahead of schedule. **Negative results are publishable here** — "we tested the decoupled-gating retrieval hypothesis at edge scale and it did not hold" is a real contribution, and we will report it as such.

### E9 — Results, write-up, demo, submission
Master comparison table, clean-clone reproduction rehearsal, Devpost write-up mapped section-by-section to the rubric, final README, <3 min demo video, license/compliance pass, submit.

---

## 5. Milestones

| Milestone | Window | Exit criteria |
|---|---|---|
| **M0 — Foundations** | Aug 2–3 | Board ordered; CIX enrollment submitted; Apache-2.0 live in About; repo skeleton + CI; results schema frozen; model survey done; hedge target chosen |
| **M1 — Portable core** | Aug 4–6 | x86 reference oracle running; benchmark harness passing CI; GDN arch audit written; GDN-2 paper ADR; hedge target producing baseline numbers |
| **M2 — Hardware bring-up** | Aug 7–9 | O6 booted and profiled *if it arrived*; NPU toolchain smoke-tested *if approved*; per-layer profile done. **Aug 9: hard track go/no-go (Physical AI vs Edge AI)** |
| **M3 — Optimization** | Aug 10–12 | Mapping ADR fixed; optimizations landed and validated; ablation matrix complete; GDN-2 experiment done; results table drafted |
| **M4 — Submission prep** | Aug 13 | Clean-clone repro rehearsal passed; write-up + README final; video recorded |
| **M5 — Submit** | Aug 14 (target: Aug 13 EOD) | Submitted on Devpost with ≥16h slack before 16:00 PT |

---

## 6. Dependency structure

### Critical path

```
procure O6 ──┐
             ├─→ flash Debian ─→ system baseline ─┐
CIX enroll ──┘                                    ├─→ per-layer profile ─→ mapping ADR ─→ partition runtime ─┐
                                                  │                                                          ├─→ hetero dispatcher ─→ ablations ─→ results table ─→ write-up ─→ SUBMIT
model survey ─→ model select ─→ arch audit ───────┤                                                          │
                            └─→ x86 reference ─→ correctness oracle ─────────────────────────────────────────┘
results schema ─→ metrics spec ─→ harness core ───┘
```

Two independent chains feed the same junction. The **hardware chain is externally gated and cannot be compressed**; the **portable chain is fully under our control** and must therefore be finished early, so that the moment hardware appears we are profiling within hours rather than still writing a harness.

### Structural rules used in beads

- Epics own tasks via `parent_key` (hierarchy), never via blocking edges.
- Blocking edges (`blocks`) connect tasks only. Direction: `from_key` is the **dependent**, `to_key` is the **prerequisite** — matching `bd dep add <issue> <depends-on>`.
- Decision beads (`type: decision`) sit at every fork so the choice is recorded as an artifact, not lost in chat.
- Beads gated on external parties (procurement, enrollment) carry the `external-gate` label so `bd ready` never implies we can just do them.

---

## 7. Risk register

| # | Risk | Likelihood | Impact | Mitigation / trigger |
|---|---|---|---|---|
| R1 | **O6 board does not arrive before deadline** | High | Critical | Hedge track (E3) from day one; track go/no-go Aug 9; Edge AI submission on generic aarch64 |
| R2 | **CIX Early Bird access not granted in time** | High | High | NPU work is isolated behind one ADR; CPU+GPU hybrid is a complete, honest submission on its own |
| R3 | NOE Compiler has no kernel for GDN recurrent scan | High | Medium | This is a *finding, not a failure* — document op-coverage gaps as a contribution; route scan to GPU/CPU by design |
| R4 | Long contexts (262K) exceed 64GB or take too long to benchmark | Medium | Medium | Sweep is incremental (4K→32K→128K→262K); each point is independently publishable; drop the top point at T-1 |
| R5 | GDN-2 layer swap needs training compute we don't have | High | Low | Default to benchmark-only option (a); (b) requires being ahead of schedule |
| R6 | Quantization destroys GDN accuracy (recurrent state is precision-sensitive) | Medium | Medium | Correctness oracle gates every quantization step; per-layer policy keeps state/gates in FP16 |
| R7 | Thermal throttling makes numbers irreproducible | Medium | Medium | Sustained-load characterization + run manifests capturing clocks/thermals; report percentiles, not bests |
| R8 | Brief's cited figures are wrong (weak secondary sources) | Medium | High | P0 verification bead against primary sources before any number enters the write-up |
| R9 | Time runs out mid-optimization | Medium | High | Descope ladder (§8), pre-agreed rather than improvised under pressure |
| R10 | Qwen3.5 checkpoint license restricts redistribution | Low | Medium | License audit is part of model survey; scripts download rather than vendor weights |

### Descope ladder (pre-agreed)

Deciding this now, calmly, is worth more than deciding it at 2am on Aug 13.

| Trigger | Cut | Keep |
|---|---|---|
| **T-4 (Aug 10)** — no NPU access | NPU offload, INT4 path | CPU+GPU hybrid: Vulkan scan, i8mm/SVE, big.LITTLE — still genuinely Arm-specific |
| **T-3 (Aug 11)** — no O6 board | Physical AI framing, all O6-specific work | Edge AI track on aarch64; GDN memory-scaling story intact |
| **T-2 (Aug 12)** | GDN-2 layer swap (option b) | GDN-2 microbenchmark comparison (option a) |
| **T-1 (Aug 13)** | Demo video (optional per rules), 262K context point | Write-up, README, repro rehearsal, results table — all mandatory |

Anything cut gets filed as a follow-up bead rather than deleted, so the repo honestly shows intended scope versus delivered scope.

---

## 8. Success criteria

**Minimum viable submission (must have):**
- Public repo, Apache-2.0 visible in About
- One Qwen3.5-family GDN model running on an Arm target
- Benchmark harness producing reproducible CSV: tokens/s, TTFT, peak memory across ≥3 context lengths
- Measured before/after showing at least one Arm-specific optimization working
- Write-up with overview, functionality/output, setup instructions
- Clean-clone reproduction verified by following our own README verbatim

**Target submission (should have):**
- Heterogeneous CPU+GPU (+NPU if accessible) partitioned execution with per-engine attribution
- Full context sweep to ≥128K with the KV-cache-vs-recurrent-state memory separation plotted
- Ablation matrix: full-attention-only vs hybrid GDN vs optimized hybrid
- Arm Performix standardized report
- GDN-2 comparison with an honest verdict

**Stretch (nice to have):**
- Dynamic thermal/load-aware dispatcher
- GDN-2 layer swap with RULER retrieval evaluation
- Demo video on physical hardware
- 262K context data point

---

## 9. Working agreements

- **Beads is the task tracker.** No markdown TODO lists, no ad-hoc task files. `bd ready` is the source of truth for what to work on. See [`docs/BEADS.md`](./docs/BEADS.md).
- **Every benchmark run emits a manifest** — device, kernel, SDK versions, governor, clocks, thermal state, git SHA. A number without a manifest is not a result.
- **The correctness oracle gates every optimization.** Speed that changes outputs is not speed.
- **Report percentiles and repeat counts, never a single best run.**
- **Negative and partial results get written up honestly.** "We tried X, it didn't help, here's the profile showing why" is worth real points under Potential Impact and costs nothing under scrutiny.
- **Decisions become ADRs** in `docs/adr/`, linked from their decision bead.

---

## 10. Repository layout (target)

```
OrionsBelt/
├── LICENSE                  # Apache-2.0 (day one, visible in About)
├── README.md                # GDN background for judges + results + setup
├── PLAN.md                  # this file
├── brief.md                 # original research brief
├── AGENTS.md / CLAUDE.md    # agent instructions (beads pointer)
├── docs/
│   ├── BEADS.md             # how to use beads on this repo
│   ├── SETUP_O6.md          # Orion O6 bring-up, NPU SDK, Python 3.8
│   ├── SETUP_PORTABLE.md    # generic aarch64 hedge target
│   ├── METHODOLOGY.md       # metric definitions, statistical approach
│   ├── FINDINGS.md          # NOE op-coverage gaps, profiling results
│   └── adr/                 # architecture decision records
├── src/orionsbelt/
│   ├── model/               # loading, GDN layer introspection
│   ├── engines/             # npu/ gpu/ cpu/ backends
│   ├── partition/           # layer→engine assignment + dispatcher
│   └── quant/               # quantization policies
├── bench/
│   ├── harness.py           # runner: context sweep, warmup, repeats
│   ├── metrics.py           # tokens/s, TTFT, memory accounting
│   ├── manifest.py          # provenance capture
│   └── plots.py             # scaling curves, comparison tables
├── scripts/                 # setup + repro entry points
├── results/                 # committed CSVs + generated plots
└── tests/                   # unit + correctness-oracle tests
```

---

## 11. Open questions

1. **Board sourcing** — is there a faster path than retail (Radxa direct, a loaner, a colleague's board, remote SSH access to someone else's O6)? This single question dominates the schedule.
2. **CIX Early Bird lead time** — unknown approval latency. Worth submitting the application immediately and asking directly about turnaround.
3. **Hedge target choice** — Android/Termux gives the most device-like "Edge AI" framing; Graviton is easiest to automate but is server silicon; Apple silicon is convenient but not Arm-vendor-aligned for the challenge's framing. Tracked as a decision bead.
4. **GDN-2 depth** — benchmark-only vs layer-swap. Gated on schedule position at Aug 10.
