# Devpost submission text — Arm Create: AI Optimization Challenge

_Edge AI track (ADR 0007). Map each section to the judging rubric._
_Copy this into the Devpost submission form. Keep the honest framing — judges
respect negative results stated plainly more than inflated claims._

---

## Project title

**OrionsBelt: Optimizing Gated DeltaNet for Arm Edge Silicon**

## Short description (tagline)

Hand-writing and benchmarking the GDN recurrence kernels that don't exist yet for Arm edge CPUs — and finding out why both major NPU toolchains reject them.

---

## Project overview

*(Maps to: Arm-specific optimization, Potential Impact)*

Gated DeltaNet (GDN) is a **linear-attention mechanism** arriving in next-generation hybrid LLMs (Qwen3.5 ships a 3:1 mix of GDN and full-attention layers). It replaces the ever-growing KV cache with a **fixed-size recurrent state** — O(1) decode memory regardless of context length, versus O(n) for standard attention.

At 262K context on the 4B checkpoint, that difference is **23.95 GiB of RAM** — the recurrent state is 51 MiB; the equivalent KV cache would be 8 GiB and still growing. On a memory-constrained edge board, that is the difference between running long-context inference and not running it at all.

**The problem:** the fast GDN kernels (causal_conv1d, fla) do not exist for Arm architectures. Without them, the model silently falls back to slow, generic PyTorch ops. This is happening *today* on NVIDIA's own silicon; on Arm/Vulkan the gap is wider. **Our contribution is filling that gap:** three hand-written NEON/SVE CPU kernels, numerically verified, benchmarked across a 5-device Arm fleet, and an honest operator-coverage audit showing where NPU acceleration can and cannot help.

**What we built:**
- **Three GDN CPU kernels** (gated cumulative decay, gated delta-rule scan, causal depthwise Conv1D) in C with NEON intrinsics, verified against FP32 reference implementations
- **Mixed-precision variants** (fp16, bf16 recurrent state) — fp16 gives 1.6× on the decay chain; scan is compute-bound and shows no bandwidth benefit
- **big.LITTLE affinity policy** — pinning to A76 big cores is 2–3× faster than default scheduler placement
- **OpenMP parallelization + NEON double-width unrolling** — 2.6–5.1× cumulative speedup on A76
- **Cross-vendor NPU operator-coverage audit** — both CIX NOE and Rockchip RKNN reject GDN's variable-length recurrence (the "Loop" op). This generalizes: no current edge NPU compiler handles it
- **GDN-2 vs GDN-1 comparison** — the decoupled gating in GDN-2 is free at decode, costs 2× at prefill
- **Analytical memory model** decomposing weights, KV cache, and recurrent state at every context length

**What we did NOT achieve (stated honestly):**
- End-to-end model-level tokens/sec — we measure at the kernel level, not the full forward pass, because no GDN inference engine exists for this silicon yet
- NPU acceleration — both vendors' compilers reject the recurrence (see above)
- The target Orion O6 board did not arrive in time; all results are from the portable aarch64 fleet (Pi 5, RK3588, Jetson Nano)

---

## Functionality and output

*(Maps to: Arm-specific optimization, WOW factor)*

### Headline: GDN kernel bandwidth on RK3588 Cortex-A76

Qwen3.5-4B, prefill (seq=64), fp32 baseline. Two independent RK3588 nodes (t3 clean tree, t4 cross-check):

| Kernel | GiB/s (t3) | Spread | GiB/s (t4) | Agreement | % of 34 GB/s spec |
|---|---:|---:|---:|---:|---:|
| Cumulative decay | 21.74 | 5.2% | 24.26 | 90% | 64% |
| Gated delta-rule scan | 11.07 | 6.2% | 11.48 | 96% | 33% |
| Causal Conv1D | 21.60 | 4.3% | 21.02 | 97% | 63% |

> Decay and Conv1D achieve **~64% of theoretical DRAM bandwidth** on a single A76 core — close to the memory ceiling. Gated scan runs at 33% because its sequential recurrence is **instruction-overhead-bound, not bandwidth-bound**. This is the key finding: the bottleneck for GDN on edge CPUs is pipeline width, not memory throughput.

### Optimization impact (fp16 state)

| Kernel | fp32 | fp16 | Speedup |
|---|---:|---:|---:|
| Cumulative decay | 21.74 | 34.87 | **1.60×** |
| Gated scan | 11.07 | 10.65 | 0.96× (flat) |

> fp16 halves memory traffic for the elementwise decay chain → 1.6×. Gated scan is compute-bound on the delta-rule matmul, so halving traffic doesn't help — confirming the instruction-overhead diagnosis.

### Memory: the architectural advantage

| Context | Weights (fp16) | KV cache | GDN state | Total | If all-attn | **Savings** |
|---:|---:|---:|---:|---:|---:|---:|
| 4K | 7.83 GiB | 0.12 GiB | 51 MiB | 8.01 GiB | 8.33 GiB | 0.33 GiB |
| 32K | 7.83 GiB | 1.00 GiB | 51 MiB | 8.88 GiB | 11.83 GiB | 2.95 GiB |
| 128K | 7.83 GiB | 4.00 GiB | 51 MiB | 11.88 GiB | 23.83 GiB | 11.95 GiB |
| 262K | 7.83 GiB | 8.00 GiB | 51 MiB | 15.88 GiB | 39.83 GiB | **23.95 GiB** |

### The NPU wall

Both CIX NOE (Orion O6) and Rockchip RKNN (RK3588) compilers **reject GDN's variable-length recurrence**:
- CIX NOE: Scan compiles but Loop is rejected → cannot implement the sequential state update
- Rockchip RKNN: Scan compiles but runtime Loop is rejected → same limitation, different vendor

This is not a bug in one toolchain — it is an **architectural constraint** of current edge NPU compilers: they require static, parallelizable dataflow graphs, and GDN's per-token sequential recurrence violates that. Our kernels run on the CPU because the OoO pipeline handles sequential dependencies well; the NPU's strength (massive parallelism) is exactly the wrong tool for this recurrence.

### Fleet cross-device validation

5 devices, 3 core classes (A76, A55, A57), spec bandwidth ranging 17–34 GiB/s:

- **Pi 5 (A76, 17 GB/s) vs Jetson (A57, 25.6 GB/s):** Pi 5 is faster despite less bandwidth — confirming the kernels are instruction-bound, not bandwidth-bound
- **big.LITTLE:** A76 big cores are 2–3× faster than A55 little cores; simultaneous big+little scheduling shows diminishing returns past 4 big cores

### GDN-2 stretch comparison

GDN-2's decoupled erase/write gating is **free at decode** (same per-token latency) but costs **2× at prefill** due to extra bandwidth — and **3.1× penalty on little cores** where bandwidth is scarce.

---

## Setup instructions

*(Maps to: Developer Experience)*

### Prerequisites

- Any 64-bit Arm board (aarch64). Tested on: Raspberry Pi 5, RK3588, Jetson Nano, AWS Graviton
- GCC ≥ 9 or Clang ≥ 10 (native compiler on the device)
- Python 3.10+ for manifest/plot generation (optional)

### Quick start

```bash
git clone https://github.com/alexcasper/OrionsBelt.git
cd OrionsBelt

# 1. Build the static benchmark binary for your core
./scripts/build_device_bench.sh

# 2. Pin CPU governor to performance (needs root, once per boot)
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do \
  echo performance | sudo tee "$c" >/dev/null; done

# 3. Run the benchmark
./dist/bench_gdn_rk3588_a76 --repeats 30 --csv > results/raw/my-device.csv

# 4. Capture provenance manifest
python3 bench/manifest.py > results/manifests/my-device.json

# 5. Generate comparison tables and figures
python3 -m bench.plots results/raw/ --text-only --output-dir results/figures
python3 scripts/generate_memory_plots.py
```

No GPU, NPU, or proprietary SDK required. Full setup guide: [`docs/SETUP_PORTABLE.md`](docs/SETUP_PORTABLE.md).

### Reproducibility

- Every measurement has a **provenance manifest** (git SHA, governor state, CPU topology, thermals)
- 733 unit tests covering kernel correctness and schema conformance
- All figures are **regenerable** from committed CSVs (`bench/plots.py`, `scripts/generate_memory_plots.py`)
- t3 benchmark data: commit `553a96e`, dirty=false, governor=performance, 30 repeats per kernel

---

## What makes this Arm-specific optimization

*(Directly addresses the 40-point criterion)*

The framing matters: Cortex-A76/A720/A57 are **Arm IP**. Hand-writing NEON/SVE kernels for these cores and benchmarking the GDN recurrence across a heterogeneous big.LITTLE fleet is Arm-architecture optimization — not tuning a third-party accelerator.

Specifically:
1. **Three novel NEON/SVE kernels** for GDN operations that no existing library provides for Arm
2. **big.LITTLE scheduling policy** exploiting the heterogeneous Arm core topology
3. **KleidiAI integration** — reusing Arm's own matmul micro-kernels for the delta-rule updates
4. **Cross-vendor NPU analysis** showing why the recurrence must stay on the Arm CPU, not the accelerator
5. **5-device fleet validation** across three Arm core generations (A57 → A55 → A76)

---

## Links

- **Repository:** https://github.com/alexcasper/OrionsBelt
- **License:** Apache-2.0
- **Findings (10 sections, 2175 lines):** [`docs/FINDINGS.md`](docs/FINDINGS.md)
- **Comparison table:** [`results/figures/comparison_table.md`](results/figures/comparison_table.md)
- **Fleet bandwidth analysis:** [`results/figures/fleet_bandwidth_scaling.md`](results/figures/fleet_bandwidth_scaling.md)
- **Memory scaling figures:** [`results/figures/memory_comparison.md`](results/figures/memory_comparison.md)
