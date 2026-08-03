# OrionsBelt

**Optimizing a Qwen3.5 Gated DeltaNet hybrid model for Arm edge silicon (Radxa Orion O6 / CIX P1).**

Submission for the [Arm Create: AI Optimization Challenge](https://arm-ai-optimization-challenge.devpost.com/) (deadline 2026-08-14, 16:00 PDT). This repository is the technical home of that submission: an in-progress, hardware-independent measurement harness and optimization effort aimed at the Physical AI and Edge AI tracks.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

Licensed under **Apache-2.0** — see [`LICENSE`](./LICENSE).

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

Standard transformer self-attention has a well-known scaling problem at inference time: to generate token *N+1*, the model attends back over every previous token, and it does so by keeping a **KV cache** — a stored key/value vector for every token seen so far. That cache grows linearly with context length. At 4K tokens it's small; at 262K tokens it reaches 8.0 GiB for the 4B checkpoint — the same order as the weights themselves, and **larger than them once the weights are quantized** (~3.1x at INT4, though still below 10.4 GiB FP16 weights). The unbounded growth is the point rather than any single crossover: the cache has no ceiling, while a recurrent state does. On a memory-constrained edge board, a linearly growing cache is the thing that runs you out of RAM, or forces you to shrink your usable context window to fit.

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

Primary target: **Radxa Orion O6**, built on the **CIX P1** SoC.

| Component | Spec |
|---|---|
| CPU | 12-core Armv9.2, three tiers: 4× Cortex-A720 "big" @ 2.8GHz, 4× Cortex-A720 "medium" @ 2.4GHz, 4× Cortex-A520 "little" @ 1.8GHz, 12MB shared L3 |
| GPU | Immortalis-G720 MC10 |
| NPU | 28.8 TOPS, supporting INT4 / INT8 / INT16 / FP16 / BF16 / TF32 |
| Combined | Up to 45 TOPS (NPU + CPU + GPU) |
| Memory | Up to 64GB, 128-bit LPDDR5 @ 5500MT/s (**100GB/s** peak bandwidth) |

All figures above are verified against primary sources (Radxa product page and docs) in [`docs/CLAIM_VERIFICATION.md`](./docs/CLAIM_VERIFICATION.md) §2.2. Note in particular that memory bandwidth is **100GB/s**, not "over" 100GB/s as an early secondary source claimed.

A portable aarch64 target (the **Edge AI** hedge track) is being brought up in parallel and does not depend on O6 hardware access; see [Status](#status).

## Status

**This is an in-progress research repository as of 2026-08-03.** Neither the Orion O6 board nor CIX Early Bird SDK access is in hand yet — both are externally gated (procurement, vendor approval) and cannot be compressed by effort alone. A hard go/no-go between the Physical AI (O6) and Edge AI (portable aarch64) framing is scheduled for 2026-08-09; the portable hedge track runs from day one regardless of that decision, so hardware-independent work is not blocked on it.

**Device microbenchmarks are running on the Jetson Nano (Cortex-A57, NEON).** Three GDN CPU kernels (gated cumulative decay, gated delta-rule scan, causal depthwise Conv1D) plus mixed-precision bf16/fp16 variants have been measured at verified Qwen3.5-4B and 0.8B shapes, in both prefill and decode phases. Results are committed with provenance manifests; see [`docs/FINDINGS.md`](./docs/FINDINGS.md) for analysis and [`results/raw/`](./results/raw/) for CSVs. The full inference harness (end-to-end tokens/sec, TTFT, memory decomposition) is still pending hardware access.

| Item | Status |
|---|---|
| Implementation plan (`PLAN.md`) | Done |
| Claim verification against primary sources (`docs/CLAIM_VERIFICATION.md`) | Done |
| Repository skeleton, Apache-2.0 license | Done |
| Results schema (`docs/RESULTS_SCHEMA.md`) | Done |
| Benchmark harness (`bench/`) + device microbenchmark (`bench_gdn.c`) | Producing data |
| Model survey / selection (`docs/MODEL_SURVEY.md`) | In progress |
| Architecture decision records (`docs/adr/`) | 5 ADRs recorded |
| CPU GDN kernels (NEON/SVE/scalar) | Verified, benchmarked on A57 |
| Mixed-precision state kernels (bf16/fp16) | Implemented, benchmarked on A57 |
| Orion O6 board bring-up | **Pending** — board not yet in hand |
| CIX Early Bird SDK / NPU toolchain access | **Pending** — not yet approved |
| Portable aarch64 hedge target (Pi 5 / RK3588) | Pending |
| Per-layer engine mapping (NPU/GPU/CPU) | Hypothesis only — pending measurements |
| Full inference results (tokens/sec, TTFT, memory) | **Not started — needs hardware** |

> **Results so far:** device-fleet microbenchmark data from the Jetson Nano.
>
> ```
> results/
>   raw/         <- per-run CSVs — jetson-j1.csv (28 rows), jetson-j1_sustained.csv
>   manifests/   <- provenance manifests — jetson-j1.json, jetson-j1_sustained.json
>   figures/     <- generated plots — pending plots.py run on a Python 3.10+ host
> ```
>
> See [`results/README.md`](./results/README.md) for the layout and [`docs/FINDINGS.md`](./docs/FINDINGS.md) for findings.

## Repository layout

Full target layout and rationale are in [`PLAN.md`](./PLAN.md) §10. Highlights:

- [`PLAN.md`](./PLAN.md) — the implementation plan: workstreams, milestones, risk register, descope ladder.
- [`docs/CLAIM_VERIFICATION.md`](./docs/CLAIM_VERIFICATION.md) — every quantitative claim in this README traced to a primary source, corrected, or dropped. Ground truth for numbers.
- [`docs/BEADS.md`](./docs/BEADS.md) — how issue tracking works on this project.
- [`docs/adr/`](./docs/adr/) — architecture decision records for irreversible forks (track selection, hedge target, layer→engine mapping, GDN-2 scope).
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
./dist/bench_gdn_jetson_a57 --repeats 30 --csv > results/raw/my-device.csv

# 4. Capture provenance manifest
python3 bench/manifest.py > results/manifests/my-device.json

# 5. Generate comparison tables and figures
python3 -m bench.plots results/raw/ --text-only --output-dir results/figures
```

No GPU, NPU, or proprietary SDK is required. Runs on Pi 5, Jetson Nano,
RK3588, Graviton, or any 64-bit Arm board. See
[`scripts/README.md`](./scripts/README.md) for all script entry points.

### Physical AI track (Orion O6 + NPU)

- `docs/SETUP_O6.md` (Orion O6 bring-up, NPU SDK, CIX NOE Compiler) — **pending hardware**

## License

Apache-2.0. See [`LICENSE`](./LICENSE).

## Issue tracking

All work is tracked in [beads](https://github.com/gastownhall/beads) (`bd`), prefix `ob-`. See [`docs/BEADS.md`](./docs/BEADS.md) for setup and conventions.
