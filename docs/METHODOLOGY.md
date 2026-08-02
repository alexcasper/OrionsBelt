# Benchmark methodology

**Bead:** `ob-aoo` · **Status:** Active 2026-08-02 · **Parent:** `ob-mrd` (E5)

This document describes **how we run a valid experiment and report it honestly** —
the experimental protocol that surrounds the metric definitions. It is the companion
to [`METRICS.md`](./METRICS.md), which fixes the *semantics* of each timer
(exactly which wall-clock instant starts and stops each clock, what counts in each
numerator/denominator). This document fixes everything *around* the timers: how the
device is prepared, how thermal state is controlled, how many repeats we run, which
percentiles are mandatory, what tolerance a "correct" output must satisfy, and what
conditions invalidate a measurement.

If `METRICS.md` is the contract between two harness implementations, this document is
the contract between two experimenters — the procedure that makes two people running
on two different boards at two different times produce results that are directly
comparable, not just individually well-defined.

---

## 1. Scope

| This document answers | Where to look instead if… |
|---|---|
| How is the device prepared? (§2) | …you need device-specific copy-paste: [`DEVICE_RUNBOOK.md`](./DEVICE_RUNBOOK.md) |
| What are the warmup/repeat/percentile rules? (§3) | …you need per-timer start/stop semantics: [`METRICS.md`](./METRICS.md) |
| How is thermal state controlled and reported? (§4) | |
| What are the correctness tolerances? (§5) | …you need the schema column reference: [`RESULTS_SCHEMA.md`](./RESULTS_SCHEMA.md) |
| What invalidates a measurement? (§6) | |
| How are results reported and compared? (§7) | |
| How do I run a valid experiment, step by step? (§8) | |

---

## 2. Device preparation

Every factor below must be recorded in the run manifest (`bench/manifest.py`). A number
collected without these controls is not a result — it is a measurement of unknown
provenance that cannot be compared to any other number in the study.

### 2.1 CPU frequency governor

**Policy: `performance` governor on all cores, always, for every run.**

The `powersave` or `schedutil` governor lets the OS dynamically scale clock frequency
in response to load, thermal headroom, and scheduler heuristics. That variability
contaminates timing: two runs of the same code at the same context length can differ by
2× purely because the governor decided to boost one and not the other. The
`performance` governor pins all cores to their maximum rated frequency, removing this
source of variance.

Set it before the first measured repeat and record the setting in the manifest:

```bash
# Requires root; password varies by device
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance | sudo tee "$c" >/dev/null
done
```

If a device does not permit governor changes (locked-down firmware, container without
privileges), record `unknown` or `locked` in the manifest and flag every row from that
run with a `notes` entry. Numbers from a non-`performance` governor are still
reportable — but only with that caveat, never as a clean headline figure.

### 2.2 CPU pinning on asymmetric (big.LITTLE) SoCs

**Policy: pin to one cluster using `taskset`; never run unpinned on an asymmetric SoC.**

An unpinned process on a big.LITTLE SoC will migrate between clusters mid-measurement
as the scheduler load-balances. Since the big and little clusters have different
clock speeds, pipeline widths, and cache hierarchies, a single run can sample a
bimodal distribution of execution times that looks like noise but is actually the
scheduler's doing. This was confirmed empirically on the RK3588 (see
[`FINDINGS.md`](./FINDINGS.md) §6): the A76 big cluster is 4–5× faster than the A55
little cluster on the GDN scan kernel, despite a clock ratio of only 1.28×. An
unpinned run blends these two distributions and produces numbers that describe
neither cluster.

```bash
# Confirm the cluster mapping first — it is board-dependent
for c in /sys/devices/system/cpu/cpu[0-9]*; do
  freq=$(cat "$c/cpufreq/cpuinfo_max_freq" 2>/dev/null)
  echo "$(basename $c): max_freq=${freq}"
done
# Higher max_freq = the big cluster

# Pin to the cluster you are measuring
taskset -c 4-7 <command>   # RK3588 A76 bigs
taskset -c 0-3 <command>   # RK3588 A55 littles
```

On a homogeneous SoC (Pi 5: 4× A76; Jetson Nano: 4× A57), pinning is still recommended
to prevent migration within the cluster, though the contamination is far smaller.

### 2.3 Background load

**Policy: the system must be otherwise idle.**

Any other significant CPU, GPU, or NPU consumer active during a run contaminates both
timing and energy measurements. Before each run:

```bash
# Check for unexpected load
uptime        # load average should be near 0
top -bn1 | head -20   # verify no competing process
```

If the idle check fails, wait for the load to clear or re-run later. Do not report
numbers collected under competing load, even with a caveat — the confound is too
severe to correct for.

### 2.4 Software environment

Record in the manifest:
- **Kernel version** (`uname -r`)
- **Compiler and version** (`gcc --version`)
- **Python version** (`python3 --version`) — for harness runs
- **Git SHA** of the harness/model/optimization code (`git rev-parse --short HEAD`)
- **ISA features** (`grep Features /proc/cpuinfo`) — determines which kernel dispatch
  path is taken (NEON, SVE, SVE2, i8mm, dotprod)

The manifest (`bench/manifest.py`) captures all of these automatically on devices with
Python 3. On minimal devices without Python, capture the equivalent by hand (see
`DEVICE_RUNBOOK.md` §2).

---

## 3. Statistical protocol

This section restates and operationalises the statistical rules from
`METRICS.md` §7. Refer there for the full justification; this section is the
experimenter-facing summary.

### 3.1 Warmup

**3 full repeats, run and discarded before the first measured repeat.** Each warmup
repeat runs the complete pipeline (prefill + decode at the sweep point's actual
context length) — not a short synthetic warmup. This ensures:

- Allocator arenas are sized to steady state (KV-cache and recurrent-state buffers
  are already at their maximum before the first measured repeat).
- Any first-call JIT/compile cost has already been paid.
- Lazily-mapped weight-file pages have already faulted in.
- The board has ramped from cold idle toward its steady operating temperature.

If a specific engine is observed to still be warming up after 3 repeats (check: is
`repeat_index=0` an outlier?), raise the warmup count for that engine and note it.

### 3.2 Repeat counts

| Tier | Context lengths | Repeats (N) | Use case |
|---|---|---:|---|
| Exploratory / dev-loop | any | 10 | Iterating, sanity checks, CI smoke |
| Headline (4K, 32K) | 4096, 32768 | 30 | Final comparison table, README |
| Headline (128K, 262K) | 131072, 262144 | 10 (minimum) | Final table; wall-clock cost too high for 30 |

**Never report N < 5** for any metric, under any schedule pressure. Below 5, a
"percentile" is indistinguishable from the observed min/max.

### 3.3 Percentiles

**p50 (median) and p95 are mandatory.** p99 is explicitly not mandated at this
project's repeat counts (10–30): at N=10–30, a reported p99 is effectively the max
observed sample, which is exactly the single-best/worst-run reporting that
`PLAN.md` §9 prohibits.

If a future run uses N≥100 specifically to resolve tail behaviour (e.g. investigating
a thermal-throttle hypothesis), p99 becomes meaningful and may be added.

### 3.4 Dispersion

Alongside p50 and p95, report:
- **Spread:** `p95 − p50` (absolute, in the metric's unit)
- **Normalized spread:** `(p95 − p50) / p50` (dimensionless)

The normalized form makes "how noisy was 4K decode" comparable to "how noisy was 262K
prefill" despite very different absolute magnitudes.

### 3.5 Minimum reportable difference

A claimed improvement between two configurations A and B is only reportable as a real
difference — not noise — if **both** of the following hold:

1. `|p50(A) − p50(B)| > 2 × max(spread(A), spread(B))`
2. The p50-to-p95 ranges of A and B do not overlap: if A is claimed faster, then
   `p95(A) < p50(B)`.

This is a deliberately simple, non-parametric rule. A t-test's normality assumption
is not well justified at N=10–30 on hardware with known non-Gaussian noise sources
(thermal throttling produces a one-sided, not symmetric, degradation). A difference
that fails this test should be reported as "not distinguishable from run-to-run
variance at this repeat count" — itself a valid, honest finding.

---

## 4. Thermal management

Thermal state is the single largest confound on passively-cooled edge hardware
(`PLAN.md` §7, risk R7). A device under sustained load will heat up, and many SoCs
will silently reduce clock frequency to protect themselves — a phenomenon that looks
like "our optimization made things slower" but is actually physics.

### 4.1 What to record

For every run, the manifest must capture:
- **Temperature before:** read from `thermal_zone` sensors immediately before the
  first measured repeat (after warmup, since warmup is expected to include some
  thermal ramp).
- **Temperature after:** read immediately after the last repeat.
- **Clock frequency at start and end:** from `cpuinfo_cur_freq` (or equivalent), so
  a frequency drop is detectable even if temperature sensors are coarse.

```bash
# Temperature sensors
for z in /sys/class/thermal/thermal_zone*; do
  echo "$(cat $z/type): $(cat $z/temp)"
done

# Per-core current frequency
for c in /sys/devices/system/cpu/cpu*/cpufreq; do
  echo "$(basename $c): $(cat $c/cpuinfo_cur_freq 2>/dev/null)"
done
```

### 4.2 Throttle detection

A run is flagged as thermally compromised if **the end-of-run clock frequency is more
than 10% below the start-of-run frequency** on any pinned core. This is the bright
line from `METRICS.md` §8:

- **>10% frequency drop:** the run is invalidated for headline reporting. Re-run with
  a cooldown pause between context-length sweep points or between repeats.
- **≤10% frequency drop:** the run is valid. Minor thermal ramp is expected and is
  part of what the warmup repeats and percentile reporting absorb.

### 4.3 Cooldown protocol

Between context-length sweep points (especially when stepping up to a larger context
that sustains higher power draw):

1. **Wait** until temperature returns to within 2°C of the pre-run baseline.
2. **Verify** clock frequency is back to rated maximum.
3. **Re-warmup** (3 repeats) if the cooldown exceeded 30 seconds — the caches will
   have cooled enough that the first measured repeat would otherwise see cold-cache
   latency.

On actively-cooled devices (fan, heatsink), thermal throttling is unlikely but still
must be verified. The protocol is the same.

### 4.4 What the RK3588 data showed

The first real-silicon run (RK3588, `FINDINGS.md` §6) showed **flat thermals**
(38°C before → 38°C after, no frequency drop) on both clusters. This is because the
GDN kernel microbenchmark runs for only seconds per kernel — not long enough to build
meaningful heat. The full model-level harness (prefill of 32K–262K tokens) will run
for much longer and may hit thermal limits on passively-cooled boards. **The throttle
detection protocol in §4.2 must be applied to every model-level run, not just the
kernel microbenchmark.**

---

## 5. Correctness tolerances

**Policy: speed that changes outputs is not speed** (`PLAN.md` §9). Every optimization —
a kernel rewrite, a quantization step, a precision reduction — must be validated
against a trusted reference output before its performance numbers are reportable.

### 5.1 Reference oracle

The x86/CUDA reference inference (`ob-aqv`) produces the golden outputs every on-device
optimization is validated against. The correctness oracle harness (`ob-3uh`) automates
the comparison. Until those are operational, any optimization claim must include a manual
cross-check against a second independent implementation of the same computation.

### 5.2 Tolerance levels

Correctness is checked at three levels, each with a different tolerance appropriate to
what it guards:

| Level | What is compared | Tolerance | Rationale |
|---|---|---|---|
| **Logit-level** | Per-position output logits vs reference | `max_abs_diff ≤ 2 × rtol × \|ref\| + atol` where `rtol=1e-3`, `atol=1e-4` for FP16; `rtol=1e-2`, `atol=1e-3` for INT8 | FP16 accumulation introduces rounding differences that are numerically real but do not change downstream decisions. INT8 quantization is expected to shift logits by O(1%) relative. |
| **Perplexity** | Model perplexity on a fixed evaluation set vs reference | `|ppl_opt − ppl_ref| / ppl_ref ≤ 0.05` (5% relative) | Perplexity is a smoother metric than per-position logits; a 5% relative change is the threshold below which downstream generation quality is empirically indistinguishable. |
| **Output-level** | Generated token sequence (greedy decode) vs reference | **Token-level agreement ≥ 95%** on the first 256 generated tokens for short-context prompts (≤4K); **≥ 80%** for long-context prompts (>32K) | At long context, minor numerical drift in the recurrent state can compound over the prefill scan, causing tokenization-level divergence that does not reflect a quality regression. The lower bar at 32K+ reflects this, not lower standards. |

These tolerances are defaults for the initial implementation. If a specific
optimization (e.g. INT4 weight quantization) requires looser tolerances, the new
tolerance must be justified in an ADR and documented in the results `notes`.

### 5.3 Long-context drift

GDN's recurrent state is updated sequentially — each chunk's state depends on all
previous chunks. Numerical error introduced per chunk (from quantization, reduced
precision, or a non-identical kernel implementation) accumulates over the scan, so
drift grows with context length. This is a structural property of the architecture,
not a bug, and the tolerance schedule in §5.2 accounts for it by relaxing the
output-level bar at >32K context.

If long-context drift exceeds the tolerance, the optimization is rejected — it is not
"speed with a caveat." The oracle gates every quantization step (`PLAN.md` §7, risk
R6), and a failure means the quantization policy must keep the affected layer(s) at
higher precision (e.g. FP16 gates and recurrent state, INT8 weights only).

### 5.4 What this does not cover

- **Correctness of the reference oracle itself.** The x86/CUDA reference runs the
  upstream model implementation unchanged — it is trusted by definition, not validated
  against a third source. If the upstream implementation has a bug, that bug is the
  baseline, and our optimization matching it is correct by this protocol's standard.
- **Semantic correctness of generated text.** The oracle checks that our optimized
  model produces the same *tokens* as the reference, not that those tokens are *good*.
  Quality evaluation (BLEU, human rating, RULER retrieval) is a separate concern owned
  by E8/E9.

---

## 6. What invalidates a measurement

Any of the following means the affected row(s) must be flagged (`notes` column) and
must not be reported as a clean headline number:

| Condition | Detection | Action |
|---|---|---|
| **Thermal throttle mid-run** | End-of-run clock >10% below start-of-run clock | Re-run with cooldown; do not report |
| **Background load** | Load average or `top` shows competing process | Wait, re-run; do not report |
| **Non-performance governor** | Manifest records governor ≠ `performance` | Flag in `notes`; only report with caveat |
| **Unpinned run on big.LITTLE** | No `taskset` in the run command / manifest | Invalidate entirely; re-run pinned |
| **Insufficient repeats** | `repeat_count < 5` for any reported percentile | Re-run with more repeats |
| **First-repeat outlier** | `repeat_index=0` value is a clear outlier relative to the distribution | Raise warmup count, re-run |
| **Non-monotonic clock** | Implementation uses `time.time()` instead of `time.perf_counter()` | Fix the harness; invalidate all affected rows |
| **Config/runtime mismatch** | Memory `weights` computed from intended config, not introspected runtime tensors | Fix the harness; invalidate affected rows |
| **Correctness failure** | Oracle comparison exceeds tolerance (§5) | Speed claim invalidated; do not report performance of an incorrect implementation |

---

## 7. Result reporting conventions

### 7.1 Every result has a manifest

A CSV without its companion manifest is not a result (`PLAN.md` §9, `RESULTS_SCHEMA.md`
§2). The manifest records device, kernel, SDK versions, governor, clocks, thermal
state, and git SHA — the provenance that lets a reviewer or a teammate reproduce or
trust the number. The `manifest_ref` column in every CSV row links to it.

### 7.2 Report percentiles, never a single best run

Every reported figure carries p50 (the headline), p95 (the tail/variance indicator),
and the repeat count N. A best-of-N number is a statement about noise, not about the
change being measured.

### 7.3 Prefill and decode are never averaged

These are reported separately, always. Averaging them into one "tokens/sec" would
erase the central distinction this project exists to demonstrate: GDN kernel
optimization speeds up prefill but leaves decode flat (`PLAN.md` §2.4).

### 7.4 Memory is attributed three ways

Weights, KV cache, and recurrent state are separate components, not one RSS number.
The three-way split *is* the claim: one component grows with context (KV cache),
another stays flat (recurrent state). Collapsing them into a single number turns a
demonstration into an assertion.

### 7.5 Negative results are published

"We tried X, it didn't help, here's the profile showing why" is worth real points
under the Devpost rubric's Potential Impact criterion (20 pts) and costs nothing under
scrutiny. `CONTRIBUTING.md` and `PLAN.md` §9 both mandate honest reporting of partial
and negative results.

### 7.6 Cross-device comparison

When comparing results across devices (the device-fleet study, ADR 0005), normalize
achieved throughput by each device's spec memory bandwidth to check the
bandwidth-bound hypothesis. A device achieving 5 GiB/s with 34 GB/s spec bandwidth is
at ~15% utilization — if another device at 17 GB/s spec achieves ~3 GiB/s (~18%
utilization), the hypothesis holds; if the lower-bandwidth device achieves
disproportionately more, it does not.

The RK3588 data (`FINDINGS.md` §6) showed these kernels are **latency-bound at the
kernel level** (6–16% of spec bandwidth), not bandwidth-bound — the serial dependency
chain in the scan/decay recurrence gates throughput on FMA pipeline latency, not data
movement. Cross-device comparison should therefore report both achieved GiB/s and
GFLOP/s so the latency-bound vs bandwidth-bound distinction is visible.

---

## 8. Running a valid experiment — step by step

This is the generalized procedure for a model-level benchmark run (the kernel
microbenchmark follows `DEVICE_RUNBOOK.md`).

### Before the run

1. **Confirm hardware identity:** `uname -m; grep -m1 Features /proc/cpuinfo; nproc`
2. **Set governor to performance** (§2.1) and verify it took effect.
3. **Confirm cluster mapping** and decide which cluster to pin to (§2.2).
4. **Check background load** is negligible (§2.3).
5. **Read and record thermals** (§4.1) — temperature and clock frequency before warmup.
6. **Confirm git SHA** matches the code you intend to benchmark.
7. **Capture manifest** (`python3 bench/manifest.py > results/manifests/<run_id>.json`).

### During the run

8. **Run warmup** (3 full repeats, discarded — §3.1).
9. **Run measured repeats** at each context-length sweep point, pinned to the target
   cluster (§2.2), with the repeat count appropriate to the tier (§3.2).
10. **Flush CSV** after each context-length sweep point so partial data is preserved
    if the run is interrupted.
11. **Between sweep points:** cooldown if temperature has risen significantly (§4.3).

### After the run

12. **Read and record thermals** — temperature and clock frequency after the last
    repeat (§4.1).
13. **Check for throttle** — if end-of-run clock >10% below start-of-run clock, the
    run is invalidated (§4.2). Re-run with cooldown.
14. **Validate every CSV row** against `bench/schema.py` — any schema violation is a
    harness bug, not a result.
15. **Run the correctness oracle** against the optimization being measured (§5). If
    it fails, the performance numbers are not reportable.
16. **Commit** the CSV, manifest, and any notes to `results/raw/` and
    `results/manifests/`.

### Checklist (copy into `notes` or a pre-flight script)

```
[ ] Governor = performance (verified, not assumed)
[ ] Pinned to target cluster (taskset -c <cores>)
[ ] Background load negligible (uptime, top)
[ ] Thermals recorded (before)
[ ] Clock frequency recorded (before)
[ ] Git SHA recorded
[ ] Manifest captured
[ ] Warmup: 3 full repeats (discarded)
[ ] Repeats: N ≥ tier minimum (§3.2)
[ ] Thermals recorded (after)
[ ] Clock frequency recorded (after)
[ ] Throttle check: clock drop ≤ 10%
[ ] Schema validation passed
[ ] Correctness oracle passed (if optimization)
```

---

## 9. Relationship to other documents

| Document | Scope | Relationship to this document |
|---|---|---|
| [`METRICS.md`](./METRICS.md) | Per-metric timer semantics (start/stop events, numerators, denominators) | **Complement.** METRICS.md defines *what* each metric measures; this document defines *how* it is collected. |
| [`RESULTS_SCHEMA.md`](./RESULTS_SCHEMA.md) | Column names, types, enum values for the CSV format | **Contract.** Every row this methodology produces must conform to the schema. |
| [`DEVICE_RUNBOOK.md`](./DEVICE_RUNBOOK.md) | Copy-paste procedure for the kernel microbenchmark on specific devices | **Specialization.** The runbook is this methodology applied to the kernel microbenchmark on the device fleet. |
| [`FINDINGS.md`](./FINDINGS.md) | Empirical results and their interpretation | **Output.** Findings are what this methodology produces when applied correctly. |
| [`RISK_REGISTER.md`](./RISK_REGISTER.md) §R7 | Thermal throttling as a named project risk | **Source.** R7 is the risk this document's §4 mitigates. |
| [`CONTRIBUTING.md`](./CONTRIBUTING.md) | Repo-wide conventions including honest reporting | **Alignment.** This document operationalizes CONTRIBUTING.md's reproducibility conventions into a specific experimental protocol. |
