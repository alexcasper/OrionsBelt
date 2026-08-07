# OrionsBelt

**Building and contributing optimized CPU kernels for Gated DeltaNet (linear attention) toward Arm's [KleidiAI](https://gitlab.arm.com/kleidi/kleidiai), demonstrated on a Qwen3.5 GDN hybrid model across the aarch64 edge device fleet.**

Submission for the [Arm Create: AI Optimization Challenge](https://arm-ai-optimization-challenge.devpost.com/) (deadline 2026-08-14, 16:00 PDT). **Committed to the Edge AI track** (ADR [0007](./docs/adr/0007-commit-to-edge-ai-track.md)) — the Orion O6 board never arrived, so the project re-centered on what's provable without it: NEON/SVE2 CPU kernels for GDN's chunkwise recurrence (scan, decay, causal conv1d, and the delta-rule matmul), verified correct and benchmarked across five real Arm CPUs, with the headline deliverable being an actual upstream contribution to KleidiAI rather than a heterogeneous NPU/GPU/CPU demo. NPU and GPU exploration continues as secondary findings (see [`docs/FINDINGS.md`](./docs/FINDINGS.md) §1, §7, §8) — see [`docs/archive/`](./docs/archive/) for the original O6/NPU-primary plan and why it changed.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

Licensed under **Apache-2.0** — see [`LICENSE`](./LICENSE).

---

## Headline results

Three GDN CPU kernels (gated cumulative decay, gated delta-rule scan, causal depthwise Conv1D), benchmarked on RK3588 Cortex-A76 (big cluster, 8-thread) at verified Qwen3.5-4B shapes:

| Kernel | GiB/s | % of 34 GB/s spec | Spread |
|---|---:|---:|---:|
| Cumulative decay | 21.1 | 62% | 3.5% |
| Causal Conv1D | 18.7 | 55% | 4.8% |
| Gated delta-rule scan | 10.6 | 31% | 5.4% |

> Cumulative decay achieves **62% of theoretical DRAM bandwidth** — near the memory ceiling.
> Scan runs at 31% because its sequential recurrence is **instruction-overhead-bound**, not
> bandwidth-bound. fp16 state gives **1.77×** on decay; scan is compute-bound and shows no
> bandwidth benefit. (Manifest git_sha `f015982`, dirty=false, governor=performance, 30 repeats.
> Full table with cross-device validation: [`comparison_table.md`](./results/figures/comparison_table.md).)

**Memory advantage at long context** — GDN's O(1) recurrent state vs attention's O(n) KV cache:

| Context | GDN state | KV cache (8 attn layers) | Savings vs all-attention |
|---:|---:|---:|---:|
| 32K | 51 MiB | 1.0 GiB | 2.95 GiB |
| 128K | 51 MiB | 4.0 GiB | 11.95 GiB |
| 262K | 51 MiB | 8.0 GiB | **23.95 GiB** |

> At 262K, the KV cache alone (8.0 GiB) **exceeds the fp16 weight footprint** (7.83 GiB).
> The recurrent state never grows. ([`memory_comparison.md`](./results/figures/memory_comparison.md))

---

## Table of contents

- [What Gated DeltaNet is, and why it matters on edge silicon](#what-gated-deltanet-is-and-why-it-matters-on-edge-silicon)
- [The hybrid stack](#the-hybrid-stack)
- [What we are actually claiming, and what we are not](#what-we-are-actually-claiming-and-what-we-are-not)
- [The gap this project fills](#the-gap-this-project-fills)
- [Target hardware](#target-hardware)
- [Status](#status)
- [Repository layout](#repository-layout)
- [Reproducing](#reproducing)
- [License](#license)
- [Issue tracking](#issue-tracking)

---

## What Gated DeltaNet is, and why it matters on edge silicon

Standard transformer self-attention has a well-known scaling problem at inference time: to generate token *N+1*, the model attends back over every previous token, and it does so by keeping a **KV cache** — a stored key/value vector for every token seen so far. That cache grows linearly with context length. At 4K tokens it's small; at 262K tokens it reaches 8.0 GiB for the 4B checkpoint — the same order as the weights themselves, and **larger than them once the weights are quantized** (~4.1x at INT4, though still below 7.83 GiB FP16 weights). The unbounded growth is the point rather than any single crossover: the cache has no ceiling, while a recurrent state does. On a memory-constrained edge board, a linearly growing cache is the thing that runs you out of RAM, or forces you to shrink your usable context window to fit.

**Gated DeltaNet (GDN)** is a linear-attention mechanism that avoids this. Instead of an ever-growing cache, each GDN layer carries a **fixed-size recurrent state** — think of it as a compressed summary of everything seen so far, updated in place at every step. Decoding token *N+1* only ever touches that fixed-size state, never the full history. That makes GDN's per-token decode memory **O(1) in context length**, as opposed to full attention's O(context length).

Conceptually, a GDN layer combines three pieces:

1. **The delta rule.** At each step, the layer doesn't just add new information to its state — it first partially *erases* the piece of the state that's least consistent with the new key/value pair, then writes the new information in. This is what lets a fixed-size state hold useful long-range information instead of just averaging everything together.
2. **Gated decay.** A learned, input-dependent gate controls how much of the old state to keep versus forget at each step (the same family of idea as Mamba-2's selective decay). This is what lets the model decide, per token, how aggressively to overwrite its memory.
3. **A causal Conv1D.** A short causal convolution over the input sequence gives the layer a small amount of local, position-aware context before the recurrent update, cheaply capturing short-range patterns that a pure recurrence handles less naturally.

Naively, the delta-rule update is a strictly sequential per-token recurrence — bad for throughput on parallel hardware. The practical trick that makes GDN fast to *train and prefill* is the **chunkwise formulation**: the sequence is split into chunks; the delta-rule update *within* a chunk is reformulated as a matmul-friendly (WY-style) parallel computation, while the recurrence *across* chunks stays sequential, carrying only the compact chunk-boundary state forward. This gives you matmul-level parallelism during prefill while still preserving an exact, fixed-size recurrent state at decode time.

**Why this is exactly the architecture that benefits most from memory-constrained hardware:** on a board with a fixed, shared memory pool (weights + cache + everything else competing for the same LPDDR5), a KV cache that grows with context directly trades away your context budget, your batch size, and your headroom for anything else running on the device. A recurrent state that never grows removes that trade entirely. The advantage is architectural, not a peephole optimization, and it is most visible precisely where memory is scarce — which is the edge, not the datacenter.

## The hybrid stack

Qwen3.5 does not replace attention with GDN everywhere. It is a **hybrid**, mixing GDN and full-attention layers at a fixed ratio: **three Gated DeltaNet layers for every one full-attention layer (3:1)**. This is confirmed against the primary source (transformers' `Qwen3_5TextConfig`; see [`docs/CLAIM_VERIFICATION.md`](./docs/CLAIM_VERIFICATION.md) §2.3).

Concretely, for the default 32-layer dense configuration, 3:1 works out to **24 Gated DeltaNet layers and 8 full-attention layers** — not "60 layers," a figure that appears in early secondary sourcing and was traced and rejected during verification (see [`docs/CLAIM_VERIFICATION.md`](./docs/CLAIM_VERIFICATION.md) §1.2). The periodic full-attention layers preserve some of standard attention's exact, unbounded-range retrieval ability; the GDN layers carry the bulk of the sequence-mixing work at O(1) decode memory.

The layer layout does not require reading modeling code to discover: the HF config exposes a per-layer `layer_types` list (`"linear_attention"` / `"full_attention"`), which is the exact introspection hook this project's architecture audit uses to confirm the layout for whatever checkpoint is ultimately selected.

Verified layer shapes for the dense config: `linear_conv_kernel_dim=4`, `linear_key_head_dim=128`, `linear_value_head_dim=128`, `linear_num_key_heads=16`, `linear_num_value_heads=32`.

## What we are actually claiming, and what we are not

This is the part we want to be most scrupulous about, because an unexplained or overstated differentiator is worse than none.

Upstream (the transformers project) publishes measured before/after numbers for swapping the slow, generic PyTorch fallback for optimized GDN kernels, on an NVIDIA GB10:

| Checkpoint | TTFT (prefill) | Decode |
|---|---|---|
| Qwen3.6-27B dense | 1.66 s → 1.11 s (**1.49× faster**) | 4.11 → 4.14 tok/s (flat) |
| Qwen3.6-35B-A3B MoE | 0.73 s → 0.53 s (**1.38× faster**) | 16.3 → 16.7 tok/s (flat) |

With upstream's own explanation: decode is roughly flat because the single-token DeltaNet recurrence is **memory-bandwidth-bound** — at one token per step, there simply isn't enough arithmetic per byte moved for a faster kernel to move the needle. The prefill win, by contrast, comes from the chunkwise formulation described above, and it **grows with context length** because more chunks means more matmul-parallel work to accelerate.

The Orion O6's LPDDR5 delivers 100GB/s — less bandwidth than the GB10 those numbers were measured on. So if anything, decode should be **more** bandwidth-bound on our target, not less.

We therefore build this submission around exactly two claims, and no others:

1. **Prefill / time-to-first-token (TTFT) throughput.** This is where kernel optimization work genuinely pays off, and where the advantage should grow with context length. This is the right target for GPU/NPU kernel work on the O6.
2. **Decode memory footprint.** O(1) recurrent state versus a linearly-growing KV cache is an architectural property, not a kernel win, and it is what makes long context feasible on a fixed-memory edge board at all.

**We explicitly do not claim a large decode tokens/second win from GDN kernel optimization.** The physics is against it on any hardware, and it would be worse on ours. Predicting the flat-decode result, measuring it, and explaining why it's flat is the honest and — we think — more credible framing than promising a speedup that would not survive scrutiny. This is also why the harness's memory-decomposition work (separating weights, full-attention KV cache, and GDN recurrent state) is treated as the load-bearing measurement of the project, not a secondary chart.

## The gap this project fills

It would be easy to read the above and conclude the contribution here is "we optimized a model." That's not quite it, and we want to be precise about what actually is novel.

The GDN fast path depends on two optional packages, `causal_conv1d` and `fla`. Without them, the model does not error — it **silently falls back** to slower, more memory-hungry generic PyTorch ops. Upstream documents this happening already, today, on NVIDIA's own SM121 (GB10) architecture, because neither package ships a prebuilt kernel for it yet.

**Arm/Vulkan is the same hole, only wider** — there is no reason to expect either package has an aarch64/Vulkan build, and as far as we can determine, none currently exists. So the actual contribution of this project is not "a faster model." It is: *the fast GDN kernels do not exist for this silicon; here is what we built and measured in their absence, and what we found trying to fill that gap.* Op-coverage gaps, kernel ports, and honestly-reported partial wins are the deliverable, not a polished end-to-end speedup.

## Target hardware

**Edge AI track (committed, [ADR 0007](./docs/adr/0007-commit-to-edge-ai-track.md)):** portable aarch64 device fleet.

| Device | Cores | ISA | Spec BW (GiB/s) |
|---|---|---|---|
| Raspberry Pi 5 | 4× Cortex-A76 @ 2.4 GHz | Armv8.2-A + dotprod | 17.0 |
| RK3588 (big cluster) | 4× Cortex-A76 @ 2.3 GHz | Armv8.2-A + dotprod | 34.0 |
| RK3588 (little cluster) | 4× Cortex-A55 @ 1.8 GHz | Armv8.2-A | 34.0 |
| Jetson Nano (j1/j2) | 4× Cortex-A57 @ 1.48 GHz | Armv8.0-A (NEON only) | 25.6 |

**Stretch target (if hardware arrives):** Radxa Orion O6, built on the **CIX P1** SoC.

| Component | Spec |
|---|---|
| CPU | 12-core Armv9.2, three tiers: 4× Cortex-A720 "big" @ 2.8GHz, 4× Cortex-A720 "medium" @ 2.4GHz, 4× Cortex-A520 "little" @ 1.8GHz, 12MB shared L3 |
| GPU | Immortalis-G720 MC10 |
| NPU | 28.8 TOPS, supporting INT4 / INT8 / INT16 / FP16 / BF16 / TF32 |
| Combined | Up to 45 TOPS (NPU + CPU + GPU) |
| Memory | Up to 64GB, 128-bit LPDDR5 @ 5500MT/s (**100GB/s** peak bandwidth) |

All figures above are verified against primary sources (Radxa product page and docs) in [`docs/CLAIM_VERIFICATION.md`](./docs/CLAIM_VERIFICATION.md) §2.2. Note in particular that memory bandwidth is **100GB/s**, not "over" 100GB/s as an early secondary source claimed. The O6 is a stretch target pending board availability; all current results are from the portable aarch64 fleet above.

## Status

**This is an in-progress research repository as of 2026-08-07.** The project has committed to the **Edge AI track** ([ADR 0007](./docs/adr/0007-commit-to-edge-ai-track.md)) after the Orion O6 board did not arrive by its last-useful-arrival date. All work continues on the portable aarch64 device fleet.

**Device-fleet microbenchmarks are complete across five Arm devices.** Three GDN CPU kernels (gated cumulative decay, gated delta-rule scan, causal depthwise Conv1D) have been measured at verified Qwen3.5-4B and 0.8B shapes on the full fleet: Jetson Nano (Cortex-A57, NEON), Raspberry Pi 5 (Cortex-A76), and RK3588 (Cortex-A76 big + Cortex-A55 little clusters). The optimization stack (OpenMP parallelization + NEON unrolling + fp16 state) delivers 2.6–5.1× on A76 silicon and 2.6–3.1× on A57. The key cross-device finding — that these kernels are **instruction-overhead-bound, not DRAM-bandwidth-bound** at seq=64 working-set sizes — is documented in the [fleet bandwidth-scaling analysis](./results/figures/fleet_bandwidth_scaling.md).

**End-to-end model decode** runs the full Qwen3.5 forward pass in C with optimized NEON GEMV kernels. With the full optimization stack (row-sweep GEMV + INT8 weight-only quantization), the 0.8B model achieves **2.45 tok/s on the Jetson Nano (Cortex-A57)** and **10.6 tok/s on RK3588 (Cortex-A76)** — practical for real-time edge deployment. The 4B model achieves 0.51 tok/s (A57 INT8) and 1.84 tok/s (A76 INT8). INT8 weight quantization adds 1.1–1.8× on top of the GEMV optimization, for a cumulative 26–30× over the naive baseline. Bottleneck analysis confirms the model is **matmul-bound** (FFN 54–72%), not recurrence-bound — GDN's novel kernels account for <1% of total time. See the [e2e fleet comparison](./results/figures/e2e_fleet_comparison.md) and [FINDINGS.md §16](./docs/FINDINGS.md).

**Context-length scaling proves GDN's core value proposition on silicon (§17).** Sweeping context length from 1 to 4096 tokens with real grouped-query attention: **pure-GDN throughput is flat to within 0.3%** while the hybrid model degrades 1.55× (4B) to 2.14× (0.8B) — entirely from the full-attention layers whose KV cache reads grow linearly. INT8 KV cache quantization (§20) cuts KV memory 4× and delivers 1.7–2.6× full-attention speedup at long context, but full-attention's cost still scales O(n). Sustained-load tests confirm 0.3% throughput decay over 94s — burst numbers are steady-state sustainable (§18). Cross-validated on both A57 and A76. See [FINDINGS.md §17–20](./docs/FINDINGS.md).

**Operator analysis findings** ([`docs/FINDINGS.md`](./docs/FINDINGS.md), 20+ sections):
- CIX NOE and Rockchip RKNN toolchains both reject GDN's runtime-length recurrence — the limitation generalises beyond one vendor (§1, §7)
- KleidiAI packed GEMM wins 1.7–3.6× on matmul but packing cost dominates at decode; dual-path strategy recommended (§8)
- big.LITTLE affinity: pinning to A76 big cores is 2–3× faster than default scheduler placement (§9)
- GDN-2 vs GDN-1: decoupled gating costs 2.2–2.7× at prefill (bandwidth-bound), 1.2–1.5× at decode on big cores but 2.2–2.4× on A55 little cores (compute-bound); clean-tree re-run, single-thread (§10)

| Item | Status |
|---|---|
| Implementation plan (`docs/archive/PLAN.md`) | Done (archived — superseded by Edge AI pivot, ADR 0007) |
| Claim verification against primary sources (`docs/CLAIM_VERIFICATION.md`) | Done |
| Repository skeleton, Apache-2.0 license | Done |
| Results schema (`docs/RESULTS_SCHEMA.md`) | Done |
| Benchmark harness (`bench/`) + device microbenchmark (`bench_gdn.c`) | Producing data |
| CI: lint + unit tests (1747 tests, 99% coverage) | Done — `.github/workflows/ci.yaml` |
| Device-fleet microbenchmarks (5 devices) | Done — [fleet analysis](./results/figures/fleet_bandwidth_scaling.md) |
| Ablation matrix (6 configs, synthetic) | Done — [comparison table](./results/figures/ablation_comparison.md) |
| Memory decomposition (analytical) | Done — [figures](./results/figures/) |
| Architecture decision records (`docs/adr/`) | 8 ADRs recorded |
| CPU GDN kernels (NEON/SVE/scalar) | Verified, benchmarked across fleet |
| Mixed-precision state kernels (bf16/fp16) | Implemented, benchmarked on Jetson |
| NPU operator-coverage audit (CIX NOE + RKNN) | Done — [FINDINGS.md](./docs/FINDINGS.md) §1, §7 |
| KleidiAI matmul evaluation | Done — [FINDINGS.md](./docs/FINDINGS.md) §8 |
| big.LITTLE affinity policy | Done — [FINDINGS.md](./docs/FINDINGS.md) §9 |
| GDN-2 vs GDN-1 microbenchmark | Done — [FINDINGS.md](./docs/FINDINGS.md) §10, clean-tree re-run |
| E2E model decode (tokens/sec, TTFT) | FP32 + INT8 measured across fleet — 0.8B: 2.45 tok/s (A57 INT8), 10.6 tok/s (A76 INT8). [e2e comparison](./results/figures/e2e_fleet_comparison.md) |
| Context-length scaling (GDN O(1) vs full-attn O(n)) | Done — pure-GDN flat to 0.3% across ctx=1–4096, cross-validated A57+A76. [§17](./docs/FINDINGS.md) |
| INT8 KV cache quantization | Done — 1.7–2.6× full-attn speedup at long context, 4× KV memory reduction. [§20](./docs/FINDINGS.md) |
| Sustained-load thermal characterization | Done — 0.3% throughput decay over 94s on RK3588 (§18) |
| Track decision: Edge AI | Done — [ADR 0007](./docs/adr/0007-commit-to-edge-ai-track.md) |
| Model survey / selection (`docs/MODEL_SURVEY.md`) | Done |
| Orion O6 board bring-up | **Pending** — board not yet in hand |
| CIX Early Bird SDK / NPU toolchain access | **Pending** — not yet approved |
| Per-layer engine mapping (NPU/GPU/CPU) | Hypothesis only — pending measurements |
| Full inference results (tokens/sec, TTFT, memory) | Done — C decode loop (FP32+INT8), ctx-length scaling (§17–20), cross-device (A57+A76), sustained-load thermal stability (§18). [e2e comparison](./results/figures/e2e_fleet_comparison.md) |

> **Results so far:** 120 CSVs from the device fleet, 85 provenance manifests, 76 generated figures/tables, 35 FINDINGS sections.
>
> ```
> results/
>   raw/         <- 120 per-run CSVs across 5 devices
>   manifests/   <- 85 provenance manifests (git SHA, governor, thermals)
>   figures/     <- fleet analysis, comparison table, kernel/memory plots> ```
>
> See [`results/README.md`](./results/README.md) for the layout, [`docs/FINDINGS.md`](./docs/FINDINGS.md) for findings, and [`results/figures/fleet_bandwidth_scaling.md`](./results/figures/fleet_bandwidth_scaling.md) for the headline cross-device analysis.

## Repository layout

Highlights:

- [`docs/FINDINGS.md`](./docs/FINDINGS.md) — the living results and findings doc; current source of truth for what's built and measured.
- [`docs/CLAIM_VERIFICATION.md`](./docs/CLAIM_VERIFICATION.md) — every quantitative claim in this README traced to a primary source, corrected, or dropped. Ground truth for numbers.
- [`docs/DEVPOST_SUBMISSION.md`](./docs/DEVPOST_SUBMISSION.md) — the Devpost write-up, mapped section-by-section to the judging rubric.
- [`docs/BEADS.md`](./docs/BEADS.md) — how issue tracking works on this project; `bd ready` and `bd show <epic>` are the current plan, not a static document.
- [`docs/adr/`](./docs/adr/) — architecture decision records for irreversible forks (track selection, hedge target, layer→engine mapping, GDN-2 scope).
- [`docs/archive/`](./docs/archive/) — the original implementation plan, source brief, and O6-arrival risk register, superseded by ADR 0007's pivot to the CPU-fleet/KleidiAI-contribution focus. Kept for history, not current direction.
- [`bench/`](./bench/README.md) — measurement harness: context sweep, metrics, provenance manifests, plotting.
- [`results/`](./results/README.md) — committed CSVs and figures, once they exist.
- [`src/orionsbelt/`](./src/orionsbelt/) — model loading/introspection, NPU/GPU/CPU engine backends, layer partitioning, quantization policy.
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — contribution workflow.

## Reproducing

### Portable Edge AI track (any aarch64 device)

Full step-by-step guide: [`docs/SETUP_PORTABLE.md`](./docs/SETUP_PORTABLE.md).

Quick start:

```bash
# 1. Build the static benchmark binaries (native gcc or cross-compiler)
./scripts/build_device_bench.sh

# 2. Set CPU governor to performance for valid numbers
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do \
  echo performance | sudo tee "$c" >/dev/null; done

# 3. Run the benchmark (pick the binary matching your core's ISA)
#    e.g. bench_gdn_rk3588_a76, bench_gdn_pi5_a76, bench_gdn_jetson_a57
./dist/bench_gdn_rk3588_a76 --repeats 30 --csv > results/raw/my-device.csv

# 4. Capture provenance manifest
python3 bench/manifest.py > results/manifests/my-device.json

# 5. Generate comparison tables and figures
python3 -m bench.plots results/raw/ --text-only --output-dir results/figures

# 6. Regenerate memory scaling figures (the O(1) vs O(n) thesis)
python3 scripts/generate_memory_plots.py
```

No GPU, NPU, or proprietary SDK is required. Runs on Pi 5, Jetson Nano,
RK3588, Graviton, or any 64-bit Arm board. See
[`scripts/README.md`](./scripts/README.md) for all script entry points.

### Stretch target: Orion O6 + NPU (if hardware arrives)

- `docs/SETUP_O6.md` (Orion O6 bring-up, NPU SDK, CIX NOE Compiler) — **pending hardware**

## License

Apache-2.0. See [`LICENSE`](./LICENSE).

## Issue tracking

All work is tracked in [beads](https://github.com/gastownhall/beads) (`bd`), prefix `ob-`. See [`docs/BEADS.md`](./docs/BEADS.md) for setup and conventions.
