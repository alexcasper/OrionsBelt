# Benchmark methodology

**Bead:** `ob-aoo` (`t-methodology`) · **Status:** Active 2026-08-02 · **Parent:** `ob-mrd` (E5)
**Companion documents:** [`METRICS.md`](./METRICS.md) (per-metric definitions), [`RESULTS_SCHEMA.md`](./RESULTS_SCHEMA.md) (data contract), [`DEVICE_RUNBOOK.md`](./DEVICE_RUNBOOK.md) (execution procedure), [`CONTRIBUTING.md`](../CONTRIBUTING.md) (reproducibility conventions)

This document states *how* we measure — the experimental methodology a reviewer or judge uses to
decide whether our results are trustworthy. It is deliberately distinct from
[`METRICS.md`](./METRICS.md), which defines the *semantics* of each metric (clock conventions,
start/stop events, numerators, denominators). Two independent people reading both documents should
be able to reproduce directly comparable numbers.

---

## 1. What we measure, and why

The project rests on two independent claims, each of which requires a different kind of measurement.
Every metric, sweep, and protocol decision below exists to support one or both of these claims, and
nothing else is measured.

### Claim A: Prefill throughput is where kernel optimization pays

Upstream measured that optimizing the Gated DeltaNet (GDN) kernel path speeds up prefill by
1.38–1.49×, and that the advantage grows with context length. The project's first goal is to
demonstrate this on Arm silicon. Prefill throughput (`prefill_tokens_per_sec`) and time-to-first-
token (`ttft_seconds`) are the metrics that carry this claim.

### Claim B: Decode is memory-bandwidth-bound, not compute-bound

The single-token GDN recurrence moves state at ~0.25 FLOP/byte — far below the roofline ridge point
of any modern compute engine. No kernel rewrite can change which side of the roofline the operation
sits on, because the operation's shape (one FMA per state element, one state-sized read, one
state-sized write) is fixed by the recurrence's definition, not by its implementation
(`METRICS.md` §9). The project's second goal is to show this empirically and explain it honestly —
predicting and reporting a flat decode-throughput result is itself a finding, not a null result to
hide.

Decode throughput (`decode_tokens_per_sec`) carries the throughput half of this claim. The
three-way memory decomposition (`peak_memory_bytes` with `metric_component` ∈ {`weights`,
`kv_cache`, `recurrent_state`}) carries the memory half — KV cache grows linearly with context
length while recurrent state stays O(1), and that contrast is the architectural advantage the
project exists to demonstrate.

Energy efficiency (`energy_joules_per_token`) is a secondary metric relevant to the "edge" framing
but is not load-bearing for either headline claim.

---

## 2. Metric inventory

All five metrics are defined exhaustively in [`METRICS.md`](./METRICS.md) §2–§6. This section is a
quick-reference summary; defer to METRICS.md for any ambiguity.

| Metric | Unit | Phase | What it measures |
|---||---|---|
| `prefill_tokens_per_sec` | tokens/sec | prefill | Prompt-processing throughput — where GDN kernel optimization is expected to pay |
| `ttft_seconds` | seconds | prefill | Time-to-first-token from request submission, including tokenization |
| `decode_tokens_per_sec` | tokens/sec | decode | Steady-state autoregressive generation throughput — expected flat across optimizations |
| `peak_memory_bytes` | bytes | both | Three components reported separately: `weights` (flat), `kv_cache` (linear), `recurrent_state` (O(1)) |
| `energy_joules_per_token` | joules/token | both | Gross energy per token from power sampling, no idle-baseline subtraction |

**Prefill and decode are never averaged into one "tokens/sec" number.** They are different values of
the `phase` column, reported as separate figures. This is the single most important reporting rule
in the project (`CONTRIBUTING.md`, `RESULTS_SCHEMA.md` §1).

---

## 3. Experimental design

### 3.1 Context-length sweep

Every measurement is collected across a fixed set of context lengths:

| Sweep point | Role | Headline repeats |
|---|---|---:|
| **4,096** (4K) | Short-context baseline | 30 |
| **32,768** (32K) | Mid-range, representative of practical use | 30 |
| **131,072** (128K) | Long context — where GDN's memory advantage is most visible | 10 |
| **262,144** (262K) | Maximum native context — the stress point | 10 |

The sweep is incremental: each point is independently publishable, and the top point may be dropped
under schedule pressure (PLAN.md §7, ADR 0004) without invalidating the rest. Shorter points are
collected first so the result table is never empty if time runs out.

### 3.2 Batch size

All measurements assume **single-request, non-batched inference** (batch size 1). This matches the
project's scope (PLAN.md §2.4, §3.1): the recurrent state and KV cache figures are per-request.
Batched serving would require reintroducing a batch-size factor into every byte/FLOP formula and is
out of scope.

### 3.3 Generation length

Decode throughput is measured over a fixed, pre-declared number of generated tokens per run,
independent of context length: **N = 257** (1 prefill-produced token + 256 decode-phase tokens).
This keeps the denominator fixed so repeats and sweep points differ only in rate, not in work done.
EOS does not stop generation early during benchmarking — the decode loop runs to exactly N tokens.
See `METRICS.md` §4 for the full rationale.

### 3.4 The prefill/decode boundary

Token 1 (the first generated token) belongs to **prefill**, not decode. In every architecture this
project measures, producing token 1 requires only the logits already computed for the prompt's last
position by the prefill forward pass — there is no additional single-token forward step. Token 1 is
therefore counted toward `prefill_tokens_per_sec` and `ttft_seconds`, and excluded from
`decode_tokens_per_sec`'s numerator (which counts only tokens 2..N). This boundary decision is
architectural, not conventional (`METRICS.md` §1).

---

## 4. Statistical protocol

Full specification in `METRICS.md` §7. Summary:

### 4.1 Warmup

**3 warmup repeats are run and discarded before measurement begins.** Each warmup runs the full
prefill+decode cycle at the sweep point's actual context length — not a short synthetic warmup — so
that allocator arenas are sized, JIT/compile costs are paid, mmap'd weight pages are faulted in, and
the board has ramped toward steady-state temperature before the first measured repeat.

### 4.2 Repeat counts

| Tier | Context lengths | N (measured repeats) |
|---|---|---:|
| Exploratory / dev-loop | any | 10 |
| Headline (write-up table) | 4K, 32K | 30 |
| Headline (write-up table) | 128K, 262K | 10 (minimum) |

**N < 5 is never reported** for any metric, under any schedule pressure. Below 5, a percentile is
indistinguishable from the observed min/max.

### 4.3 Percentiles

**p50 and p95 are mandatory.** The nearest-rank method is used (not interpolation). p99 is
explicitly not mandated because at N = 10–30 it would be nearly indistinguishable from the max
observed sample, which is exactly the "best-of-N" reporting the project prohibits.

### 4.4 Dispersion

Alongside p50 (the headline number) and p95 (the tail indicator), we report:
- **Spread:** `p95 − p50` (absolute variability)
- **Normalized spread:** `(p95 − p50) / p50` (comparable across metrics and context lengths)

The normalized form makes "how noisy was 4K decode" comparable to "how noisy was 262K prefill"
despite very different absolute magnitudes. This matters because thermal variance on passively-cooled
edge hardware is a named risk (PLAN.md §7, R7).

### 4.5 Minimum reportable difference

A claimed improvement between two configurations A and B is reportable as real — not noise — only if
**both** hold:

1. `|p50(A) − p50(B)| > 2 × max(spread(A), spread(B))` — the point estimates differ by more than
   twice the noise.
2. The p50-to-p95 ranges do not overlap — if A is claimed faster, then `p95(A) < p50(B)` (A's noisy
   worst case is still better than B's typical case).

This is deliberately simple and non-parametric rather than a t-test, because the normality
assumption is not justified at N = 10–30 on hardware with known non-Gaussian noise sources (thermal
throttling produces one-sided degradation, not symmetric noise). A difference that fails this test is
reported as "not distinguishable from run-to-run variance at this repeat count" — which is itself a
valid, honest finding.

---

## 5. Thermal and frequency control

On passively-cooled edge hardware (every device in this project's fleet: Orion O6, Raspberry Pi 5,
RK3588, Jetson Nano), thermal state alone can move throughput enough to invalidate a comparison.
The following controls are mandatory for every benchmark run.

### 5.1 CPU frequency governor

The cpufreq governor is set to `performance` on all cores before measurement begins, locking
frequency at its maximum rated value. This eliminates DVFS (dynamic voltage and frequency scaling)
as a noise source: without it, the scheduler may downclock mid-run in response to transient thermal
or power events, making two runs of the same code look different.

The governor setting is recorded in the run manifest. If `performance` cannot be set (e.g.
permissions), `schedutil` or `ondemand` is used and noted — but runs under different governor
policies are different experiments, not the same experiment under two conditions, and cannot be
directly compared.

### 5.2 Thermal monitoring

Core temperature is read from sysfs thermal zones immediately **before** and **after** each run:
```
cat /sys/class/thermal/thermal_zone*/temp
```

**Thermal throttle invalidation rule:** if the post-run clock frequency (or core temperature
inferred from thermal zones) drops more than **10%** from the start-of-first-measured-repeat value,
the run is invalidated for headline reporting (`METRICS.md` §8). The fix is a cooldown pause between
repeats or between sweep points, then a re-run — not discarding the low outlier while keeping the
rest, which would bias the reported percentiles.

### 5.3 big.LITTLE affinity

On asymmetric SoCs (RK3588: 4× A76 + 4× A55), the benchmark binary is pinned to one cluster using
`taskset`. An unpinned run is close to meaningless because the scheduler will migrate the workload
mid-measurement. The pinning is verified by checking each core's `cpuinfo_max_freq` before trusting
the cluster assignment, which is board-dependent (`DEVICE_RUNBOOK.md` §1).

On homogeneous devices (Pi 5: 4× A76; Jetson Nano: 4× A57), pinning is not required but the binary
variant must match the ISA: the Jetson runs the `armv8.0` / NEON-only variant, not the `armv8.2` or
`armv9.2` build, which would illegal-instruction.

### 5.4 Background load

The device is idled before each run. Any other significant CPU/GPU/NPU consumer contaminates timing
(and, for energy measurements, contaminates the power integral directly). If the system cannot be
quieted, the run is deferred rather than reported with a caveat.

---

## 6. Provenance and reproducibility

### 6.1 Every run emits a manifest

A number without a manifest is not a result (PLAN.md §9, `bench/README.md` rule 1). The manifest
captures:

| Field | Source |
|---|---|
| Device identifier | `device` column (enum: `o6`, `generic_aarch64`, `x86_reference`) |
| OS / kernel | `uname -a` |
| CPU model, core count, ISA features | `/proc/cpuinfo` |
| Memory | `free -m` |
| CPU governor | `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` |
| Max / min CPU frequency | `cpuinfo_max_freq`, `cpuinfo_min_freq` |
| Thermal zone readings (pre and post) | `/sys/class/thermal/thermal_zone*/temp` |
| Git SHA of harness/model/optimization code | `git rev-parse HEAD` |
| SDK / driver versions | as available per platform |

The manifest is generated by `bench/manifest.py` (stdlib-only, degrades gracefully on minimal
images) and stored as `results/manifests/<run_id>.json`, paired with its CSV at
`results/raw/<run_id>.csv`. The CSV's `manifest_ref` column points to it.

### 6.2 Clean-clone reproduction

The project's Developer Experience score (15 pts on the rubric) depends on a judge being able to
clone the repo and reproduce the result pipeline. The clean-clone rehearsal (bead `ob-kdi`) follows
the README verbatim — no steps that exist only in an agent's memory, no paths that depend on a
specific developer's home directory. If a step fails on a clean clone, the README is wrong and must
be fixed before the rehearsal passes.

### 6.3 Figure reproducibility

All figures under `results/figures/` are generated from committed CSV data in `results/raw/` by
`bench/plots.py` — never hand-assembled. If a figure cannot be regenerated from what is committed,
it does not belong in the write-up (`CONTRIBUTING.md`).

---

## 7. Correctness tolerances

### 7.1 The correctness oracle

Every optimization — quantization, kernel swap, engine reassignment — is gated by the x86/CUDA
reference, which serves as the correctness oracle. The oracle produces golden logits (and, where
applicable, perplexity) from the unmodified model. Speed that changes outputs is not speed.

The gating protocol:

1. Run the oracle on the same input prompts at the same context lengths.
2. Run the optimized variant on the same inputs.
3. Compare outputs against the tolerances in §7.2 below.
4. If the comparison fails, the optimization does not ship — regardless of the throughput numbers.

### 7.2 Numerical tolerances

| Comparison | Tolerance | Rationale |
|---|---|---|
| Logits (oracle vs FP16 reference) | `max_abs ≤ 1e-3`, `max_rel ≤ 1e-2` | FP16 accumulation differences are expected at this scale; both quantities cross zero, so relative error near zero is meaningless and absolute error is the binding constraint |
| Logits (FP16 vs INT8/INT4 quantized) | `max_abs ≤ 5e-2` per-token, KL divergence ≤ 0.1 over full distribution | Quantization introduces bounded error; per-token tolerance is looser because we care about downstream generation quality, not bit-exactness |
| Perplexity (oracle vs optimized) | relative Δ ≤ 5% | The downstream NLP-quality signal; a 5% perplexity shift is within the range that a human reader would not perceive as degradation in generated text |
| Kernel correctness (SVE/NEON vs scalar reference) | bit-identical for scan; `max_abs ≤ 1 ULP` for conv (FMA contraction) | The CPU kernels are verified against a precision-matched reference at the kernel level (`FINDINGS.md` §4). The one-ULP conv difference is from `svmla`/`vfmaq` FMA contraction, not a bug |

### 7.3 Recurrent state is precision-sensitive

GDN's recurrent state compounds quantization error multiplicatively through the decay-gated
recurrence (`S_t = a_t ⊙ S_{t-1} + write_term`). Unlike a KV cache, where each token's error is
local, a state error at token *t* propagates to all subsequent tokens. This is why the quantization
policy (`docs/QUANTIZATION_POLICY.md`) mandates **FP32 for recurrent state** — the error compounding
makes it the one tensor where aggressive quantization is unsafe without explicit validation.

---

## 8. What invalidates a measurement

Any of the following, if present, means the affected rows must be flagged (in the `notes` column and
the manifest) and must not be reported as clean headline numbers without that caveat. Full
specification in `METRICS.md` §8; this is the operational checklist:

1. **Thermal throttle mid-run** — post-run clock >10% below start-of-measured-repeat clock.
2. **Background load** — another significant compute consumer active during the run.
3. **Insufficient warmup** — `repeat_index=0` is a clear outlier relative to the rest, indicating
   allocator/JIT/page-fault warmup was not fully absorbed by the 3 discarded warmup repeats.
4. **Unrecorded governor state** — comparing two runs collected under different (or undocumented)
   governor policies is comparing two different experiments.
5. **Non-monotonic clock** — any implementation using `time.time()` instead of a monotonic clock for
   duration measurement is invalid, since an NTP adjustment could silently corrupt the duration.
6. **Config/runtime mismatch for memory** — a `peak_memory_bytes` row computed from the *intended*
   quantization value rather than introspected from the *actual* runtime tensors is a correctness
   bug, not a data point.
7. **QEMU timings** — QEMU emulates instruction by instruction; timings under QEMU are
   performance-meaningless and must never appear in a result table (`FINDINGS.md` §5). QEMU's
   legitimate role is correctness verification only.

---

## 9. Honest reporting

This project's integrity depends on reporting what happened, not what we hoped would happen. The
specific commitments:

- **Negative results are published.** "We tried X, it didn't help, here is the profile showing why"
  is a valid entry in the write-up, not something to omit. A reviewer who can see *why* something
  didn't work trusts the rest of the write-up more, not less.
- **A flat decode-throughput number is a finding.** The bandwidth-bound explanation (§1, Claim B)
  transforms "no improvement" from a null result into a predicted, explained, and measured result.
  Reporting it honestly with the roofline analysis reads as competence; dressing it up as a win
  reads as exactly what judges are trained to catch.
- **No best-of-N.** Every reported figure carries N and the distribution (p50/p95), never a
  cherry-picked minimum or maximum.
- **Partial results are labeled as partial.** If a device or context length could not be measured,
  the table says so — an empty cell with a note, not a silently omitted row that makes the table
  look more complete than it is.

---

## 10. Known limitations

Stated plainly so a reader can calibrate trust:

1. **Activation memory is not measured.** The three `peak_memory_bytes` components (weights, KV
   cache, recurrent state) do not include transient activation tensors (attention scores, MLP hidden
   states). If activation memory turns out to be large enough to matter, it will be surfaced as an
   explicit finding, not silently folded into an existing component (`METRICS.md` §5.0).

2. **Energy instrumentation is platform-dependent.** The project does not mandate a specific power
   sensor; `energy_joules_per_token` is reported gross (no idle-baseline subtraction) using whatever
   the platform exposes — on-board rail monitor, external bench meter, or OS-level energy counter,
   in descending order of preference. The manifest records which was used and at what sample rate.

3. **Cross-device comparison requires matching methodology.** Results from different devices are only
   comparable if both were collected under the same governor policy, thermal controls, and sweep
   protocol. The manifest exists precisely so this can be verified.

4. **The bandwidth-bound thesis is tested, not assumed.** The device fleet spans ~17 GB/s (Pi 5) →
   ~25.6 GB/s (Jetson Nano) → ~34 GB/s (RK3588) of spec memory bandwidth. If achieved throughput
   tracks bandwidth roughly linearly and ignores core generation, that is evidence for the thesis.
   If the Pi 5 (newest cores, lowest bandwidth) beats the Jetson Nano (oldest cores, more bandwidth)
   comfortably, the thesis is wrong or incomplete — and that outcome is published, not suppressed
   (`DEVICE_RUNBOOK.md`, "What we are actually testing").

5. **The O6's NPU is externally gated.** If board procurement or CIX SDK access does not resolve
   before the deadline, the project ships a CPU+GPU hybrid with rigorous numbers and says so plainly
   (PLAN.md §1). A reproducible, honestly-reported partial result scores better than an unverifiable
   claim.
