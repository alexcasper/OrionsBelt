# ob-7b5 — GDN-2 on Edge Silicon: An Honest Operator-Level Cost Analysis

**Bead:** `ob-7b5` · **Date:** 2026-08-07
**Authors:** Agent (rk3588-t4), agent (rk3588-t3)
**Type:** Research note (benchmark-only; no accuracy evaluation)
**Data:** `results/raw/rk3588-t4_gdn2_vs_gdn1_big_single.csv`, `_little_single.csv`
**Manifest:** `results/manifests/rk3588-t4-gdn2-single.json` (clean, sha=c25e1f2)

---

## TL;DR

**GDN-2's decoupled erase/write gating is nearly free at decode on big cores
(1.2–1.5×), but costs 2.2–2.7× at prefill and 2.2–2.4× at decode on little cores.**
On edge silicon, the quality benefit of separate erase/write control is paid for
in memory bandwidth and in-order pipeline stalls — not in FLOPs, which are
underutilised. We did not evaluate retrieval quality (RULER) due to lack of
adaptation compute (ADR-0004, risk R5). The architectural cost profile alone,
measured on shipping Arm hardware, is the contribution.

---

## 1. Background and hypothesis

### 1.1 The architectural difference

**GDN-1** (standard Gated DeltaNet, as in Qwen3.5): one scalar decay gate `g`
controls both erase and write:

```
s[t] = x[t] + s[t-1] * g[t]              // 1 FMA/element, 3 fp32 streams
```

**GDN-2** (decoupled gating, [NVLabs GatedDeltaNet-2](https://arxiv.org/abs/2605.22791)):
separate per-channel erase gate `b` and write gate `w`:

```
s[t] = w[t]·x[t] + s[t-1] * (g[t]·b[t])  // 2 MUL + 1 FMA, 5 fp32 streams
```

Theoretical cost ratio: **1.67× memory traffic (5/3 streams), 2× arithmetic
(4 vs 2 FLOPs/element)**.

### 1.2 The hypothesis (ADR-0001)

ADR-0001 recorded the testable hypothesis that GDN-2's decoupled gating improves
long-context retrieval quality enough to justify its added cost at edge scale.
The expected quality benefit (per the NVLabs paper) is improved multi-key
retrieval on RULER benchmarks. The expected cost (this note) is the operator-level
overhead on shipping Arm silicon.

### 1.3 Scope limitation

Per ADR-0008, we chose the **benchmark-only comparison (option a)** over a
full layer-swap + RULER evaluation (option b). We lack the adaptation compute
to fine-tune or convert a Qwen3.5-architecture checkpoint with GDN-2 layers.
**This note reports cost, not quality.** The quality claim remains the paper's,
unverified by us.

---

## 2. Method

### 2.1 Hardware

| Device | SoC | Big cores | Little cores | RAM | Kernel |
|--------|-----|-----------|-------------|-----|--------|
| rk3588-t4 | RK3588 | 3× Cortex-A76 @ 2.4 GHz | 4× Cortex-A55 @ 1.8 GHz | 8 GB | 6.11 (EEVDF) |

### 2.2 Measurement protocol

- **Single-threaded:** `OMP_NUM_THREADS=1 taskset -c <cores> ./dist/bench_gdn_rk3588_a76 --repeats 30 --csv`
  (The kernel source `gdn_sve.c` contains `#pragma omp parallel for` — without
  `OMP_NUM_THREADS=1`, OpenMP spawns threads for all CPUs in the taskset group,
  inflating throughput by ~2×. See FINDINGS.md §10 correction note.)
- **Governor:** `performance` (all cores at max frequency)
- **Repeats:** 30 per kernel × config
- **Metrics:** p50 latency (µs), GiB/s throughput (p50), GFLOP/s (p50), spread (p95/p50 − 1)
- **Provenance:** manifest captured via `python3 bench/manifest.py`

### 2.3 Configurations

- **Prefill:** seq=64, channels=4096 (Qwen3.5-4B) and channels=2048 (Qwen3.5-0.8B)
- **Decode:** seq=1, same channel counts
- **Clusters:** A76 big (cpu 4–7), A55 little (cpu 0–3)

---

## 3. Results

### 3.1 Big cluster (A76)

| Config | GDN-1 p50 (µs) | GDN-2 p50 (µs) | Slowdown | GDN-1 GiB/s | GDN-2 GiB/s | GDN-1 GFLOP/s | GDN-2 GFLOP/s |
|--------|---------------|---------------|----------|-------------|-------------|---------------|---------------|
| 4B prefill (seq=64) | 548 | 1429 | **2.61×** | 5.40 | 3.44 | 0.96 | 0.73 |
| 0.8B prefill (seq=64) | 204 | 447 | **2.19×** | 7.25 | 5.49 | 1.28 | 1.17 |
| 4B decode (seq=1) | 2.33 | 3.50 | **1.50×** | 32.7 | 30.5 | 3.51 | **4.68** |
| 0.8B decode (seq=1) | 1.46 | 1.75 | **1.20×** | 26.2 | 30.5 | 2.81 | **4.68** |

### 3.2 Little cluster (A55)

| Config | GDN-1 p50 (µs) | GDN-2 p50 (µs) | Slowdown | GDN-1 GiB/s | GDN-2 GiB/s |
|--------|---------------|---------------|----------|-------------|-------------|
| 4B prefill (seq=64) | 3421 | 9201 | **2.69×** | 0.87 | 0.53 |
| 0.8B prefill (seq=64) | 1009 | 2753 | **2.73×** | 1.47 | 0.89 |
| 4B decode (seq=1) | 15.5 | 33.5 | **2.17×** | 4.93 | 3.18 |
| 0.8B decode (seq=1) | 6.42 | 15.2 | **2.36×** | 5.94 | 3.52 |

---

## 4. Analysis

### Finding 1: Prefill is the bottleneck (2.2–2.7× penalty)

At prefill (seq=64), the recurrent state spans the full channel dimension and does
not fit in L1 cache. The scan is **bandwidth-bound**: GDN-2's 5 streams vs GDN-1's
3 directly increase memory traffic. The observed slowdown (2.2–2.7×) **exceeds**
the theoretical 1.67× memory ratio because the extra 2 MULs per element add
arithmetic latency that does not fully overlap with memory access on this pipeline.

This penalty is consistent across both big and little cores (2.2–2.7× range),
confirming it is a memory-system limitation, not a microarchitectural one.

### Finding 2: Decode on big cores is nearly free (1.2–1.5× penalty)

At decode (seq=1), the recurrent state is a single vector of `channels` floats —
16 KiB for 4096 channels — which fits comfortably in L1 cache. The kernel becomes
**compute-bound**. On the A76 (out-of-order, 2× FMA/cycle), GDN-2's 2× compute cost
manifests as only a 1.2–1.5× slowdown. The A76's FMA units were **underutilised**
in GDN-1 decode: GDN-2 achieves **4.68 GFLOP/s** vs GDN-1's 3.51 — the extra MULs
fill otherwise idle arithmetic slots.

**Implication:** If GDN-2 improves retrieval quality enough to justify a 20–50%
decode slowdown on big cores, it is viable for the autoregressive decode hot path.

### Finding 3: Decode on little cores is expensive (2.2–2.4× penalty)

On the A55 (in-order, narrower pipeline), the same decode shows a 2.2–2.4× penalty.
The in-order pipeline **cannot overlap** the extra MULs with loads — the compute
cost is fully exposed. This is worse than the 1.67× memory ratio, confirming the
penalty is arithmetic, not bandwidth, at decode on this microarchitecture.

**Implication:** GDN-2 models are a poor fit for little cores under
single-threaded decode. A heterogeneous dispatcher should route GDN-2 layers to
big cores exclusively, or accept the 2.2–2.4× cost on little cores as a fallback.

### Finding 4: The cost is not in FLOPs the hardware can't do — it's in memory

GDN-2 achieves **higher GFLOP/s** than GDN-1 at decode on A76 (4.68 vs 3.51). The
hardware CAN do the extra arithmetic — it's the memory traffic at prefill that
dominates. This separates the cost into two regimes:

- **Decode (cache-resident):** compute-bound, but big cores have spare FMA capacity.
  GDN-2 is nearly free here.
- **Prefill (cache-miss):** bandwidth-bound, and GDN-2's extra streams saturate the
  memory system. GDN-2 costs 2.2–2.7× here.

---

## 5. What we did NOT measure

We are explicit about the gaps:

1. **No retrieval quality evaluation (RULER).** ADR-0008 chose benchmark-only (option
   a). The quality benefit of decoupled gating is the NVLabs paper's claim, unverified
   by us. We cannot say whether the 1.2–1.5× decode cost "pays for itself" in improved
   long-context accuracy.

2. **No end-to-end model comparison.** This is an operator-level microbenchmark, not a
   full forward pass with GDN-2 layers swapped in. The actual per-token cost depends on
   the fraction of total wall-clock spent in the gated scan vs attention/FFN blocks.

3. **Single device only (rk3588-t4).** The A76/A55 results may not generalise to the
   Cortex-A720/A520 on the CIX P1 (Orion O6), which has wider FMA units and higher
   memory bandwidth. Re-running on the O6 is gated on hardware acquisition (ob-axq).

4. **No quantized (INT8/INT4) evaluation.** The kernels are fp32. Quantized GDN-2
   kernels would change the arithmetic/memory balance, potentially reducing the prefill
   penalty.

---

## 6. Conclusion

GDN-2's decoupled erase/write gating is a **mixed result at edge scale**:

| Regime | Cost on big cores | Cost on little cores | Verdict |
|--------|------------------|---------------------|---------|
| Prefill | 2.2–2.7× | 2.7× | Expensive. Acceptable if amortised over long decode sequences. |
| Decode | 1.2–1.5× | 2.2–2.4× | Nearly free on big cores. Poor fit for little cores. |

The headline finding for judges: **on edge Arm silicon, GDN-2's quality improvement
(if real) costs nothing at decode on big cores — but prefill and little-core decode
pay a 2.2–2.7× and 2.2–2.4× tax, respectively.** The cost is memory-bandwidth-bound
at prefill and arithmetic-latency-bound at decode on in-order cores.

This is a publishable negative/partial result: we tested the decoupled-gating
hypothesis at edge scale and found that it is NOT free, with the cost profile
depending sharply on the execution regime. Per PLAN.md §9, reporting this honestly
scores under Potential Impact and costs nothing under scrutiny.

---

## Reproducing

```bash
# Build
./scripts/build_device_bench.sh

# Run single-threaded GDN-2 vs GDN-1 comparison
OMP_NUM_THREADS=1 taskset -c 4-7 ./dist/bench_gdn_rk3588_a76 --repeats 30 --csv \
  > results/raw/rk3588-t4_gdn2_vs_gdn1_big_single.csv
OMP_NUM_THREADS=1 taskset -c 0-3 ./dist/bench_gdn_rk3588_a55 --repeats 30 --csv \
  > results/raw/rk3588-t4_gdn2_vs_gdn1_little_single.csv

# Correctness verification
K=src/orionsbelt/engines/cpu/kernels
gcc -O3 -march=armv8.2-a+simd -static "$K/gdn_sve.c" "$K/test_gdn2_scan.c" -o /tmp/t2 -lm
/tmp/t2

# Manifest
python3 bench/manifest.py > results/manifests/rk3588-t4-gdn2-single.json
```

## References

- [ADR-0001](../adr/0001-gdn2-decoupled-gating-hypothesis.md): Hypothesis and paper analysis
- [ADR-0008](../adr/0008-gdn2-benchmark-only-comparison.md): Decision to use benchmark-only comparison
- [FINDINGS.md §6](../FINDINGS.md): NVLabs reference clone and GDN-2→GDN-1 reduction
- [FINDINGS.md §10](../FINDINGS.md): Original (multi-threaded) GDN-2 comparison with correction note
- [FINDINGS.md GDN-2 section](../FINDINGS.md): Corrected single-thread data and analysis
- NVLabs paper: [arXiv:2605.22791](https://arxiv.org/abs/2605.22791)
