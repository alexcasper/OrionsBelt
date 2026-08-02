# Methodology

**Bead:** `ob-aoo` · **Detailed specs:** [`METRICS.md`](./METRICS.md),
[`RESULTS_SCHEMA.md`](./RESULTS_SCHEMA.md) · **Harness:** [`bench/harness.py`](../bench/harness.py)

Transparent methodology is what separates a credible benchmark from a marketing
number. This document states — in submission-facing terms — how we measure, how
many repeats we run, which percentiles we report, how we control thermal state,
and what validates that our optimizations have not changed the model's outputs.
The full operational contract lives in `METRICS.md`; this is the readable summary.

---

## 1. What we measure

Every benchmark run produces a **tidy/long CSV** (`RESULTS_SCHEMA.md`) where each
row is exactly one measurement: one metric, one phase, one repeat. We never average
before writing data.

| Metric | Phase | What it captures |
|---|---|---|
| `prefill_tokens_per_sec` | prefill | Prompt-processing throughput — where GDN kernel optimization shows the 1.38–1.49× win ([METRICS.md §2](./METRICS.md)) |
| `decode_tokens_per_sec` | decode | Per-token autoregressive throughput — predicted to stay flat because the single-token recurrence is memory-bandwidth-bound ([METRICS.md §4](./METRICS.md)) |
| `ttft_seconds` | prefill | Time to first token — the user-facing latency, including tokenization ([METRICS.md §3](./METRICS.md)) |
| `peak_memory_bytes` | both | Three-way memory attribution: weights (flat), KV cache (grows linearly), GDN recurrent state (O(1)) — this split *is* the project's central claim ([METRICS.md §5](./METRICS.md)) |

**The phase boundary is architectural, not conventional.** Token 1 belongs to
prefill (it requires no autoregressive step), and `decode_tokens_per_sec` measures
only tokens 2..N. This prevents blending one prefill-shaped event into a metric
whose entire point is isolating the bandwidth-bound steady-state recurrence.

---

## 2. Statistical protocol

### Repeats and warmup

| Context lengths | Warmup (discarded) | Measured repeats | Rationale |
|---|---:|---:|---|
| Any (exploratory/CI) | 3 | 10 | Fast iteration, sanity checks |
| 4K, 32K (headline) | 3 | 30 | Numbers that enter the final comparison table |
| 128K, 262K (headline) | 3 | 10 (minimum) | Wall-clock cost per repeat is large; 30 is impractical |

Warmup repeats run the **full** prefill + decode cycle (not a synthetic shortcut)
so that allocator arenas, JIT compile costs, page faults, and thermal ramp are all
absorbed before the first measured repeat ([METRICS.md §7](./METRICS.md)).

**We never report N < 5** — below 5, a "percentile" is indistinguishable from the
observed min/max, defeating the purpose of percentile reporting.

### Percentiles

**p50 (median) and p95 are mandatory.** We use the nearest-rank method: for N
sorted values, the p-th percentile is the value at rank `ceil(p/100 × N)`, clamped
to [1, N]. At N = 10–30 this resolves p50 and p95 from actual data rather than
interpolating between samples.

We explicitly **do not mandate p99**: at N = 10–30 it would be nearly
indistinguishable from the max observed sample — exactly the single-best/worst-run
reporting our working agreements prohibit.

### Dispersion and minimum reportable difference

Alongside p50 and p95, we report:
- **Spread:** `p95 − p50`
- **Normalized spread:** `(p95 − p50) / p50` — makes noise comparable across very
  different magnitudes (4K decode vs. 262K prefill)

A claimed improvement between configurations A and B is reportable as real — not
noise — only if **both** hold:
1. `|p50(A) − p50(B)| > 2 × max(spread(A), spread(B))`
2. The p50-to-p95 ranges do not overlap (if A is faster, `p95(A) < p50(B)`)

This is deliberately non-parametric: a t-test's normality assumption is not
justified at N = 10–30 on hardware with known non-Gaussian noise (thermal
throttling produces one-sided degradation, not symmetric spread).

---

## 3. Thermal and governor control

On passively-cooled edge hardware, thermal state alone can move throughput enough
to invalidate a comparison (PLAN.md R7). We control for this:

| Control | Procedure | Recorded in |
|---|---|---|
| **CPU governor** | Set to `performance` before every run (disables DVFS downclocking) | Manifest `host.cpu_topology[].governor` |
| **Thermal snapshot** | Read `/sys/class/thermal/thermal_zone*/temp` before and after each run | Manifest `thermal_zones[].temp_millicelsius` |
| **Throttle detection** | A clock drop > 10% between run start and end invalidates the run for headline reporting | Compared against manifest clock readings |
| **Background load** | Verify the system is otherwise idle immediately before each run | Documented in `notes` if violated |

If p95 is far above p50, we suspect thermal throttling first — not a real tail.

---

## 4. Correctness tolerances

**Speed that changes outputs is not speed** (PLAN.md §9). Every optimization
passes the correctness oracle before its numbers enter a results table.

| Check | Tolerance | Gate |
|---|---|---|
| Token-level agreement | Generated text matches FP16 reference on ≥ 95% of test prompts | Per context length (4K, 32K, 128K) |
| Logit KL divergence | KL(output ‖ FP16 reference) < 0.1 | On the test corpus |
| Long-context retrieval | Multi-key retrieval accuracy at 128K+ | Explicitly tested, not just perplexity |
| Kernel numerical validation | Bit-identical or < 1 fp32 ULP vs. matched reference | FINDINGS.md §4 (verified for our SVE/NEON kernels) |

If INT4 quantization fails the oracle, we fall back to INT8 (still a 2× decode
speedup). If INT8 also fails, we investigate per-layer sensitivity
([QUANTIZATION_POLICY.md §6](./QUANTIZATION_POLICY.md)).

---

## 5. Provenance and reproducibility

**A number without a manifest is not a result** (PLAN.md §9). Every CSV has a
companion manifest at `results/manifests/<run_id>.json` recording:

- Device model, core count, CPU topology (frequencies, governor, capacity)
- ISA features (dotprod, i8mm, SVE/SVE2, bf16)
- Kernel/git SHA of the harness and model code
- Thermal state, memory available, Python/software versions
- The exact sweep parameters (context lengths, repeats, decode length)

The manifest is generated by [`bench/manifest.py`](../bench/manifest.py), which is
stdlib-only and degrades gracefully on any platform — it never crashes a benchmark
run, and the worst case is a manifest with more null fields, not a missing one.

**We never benchmark under QEMU.** QEMU emulates instruction-by-instruction; its
timings are meaningless as measurements. QEMU's legitimate role in this project is
*correctness verification* (`scripts/verify_cpu_kernels.sh`), never speed.

---

## 6. What we are honest about

- **Decode throughput stays flat under GDN kernel optimization.** This is the
  predicted result, not a failure — the single-token recurrence is
  memory-bandwidth-bound at ~0.25 FLOP/byte ([METRICS.md §9](./METRICS.md)).
  Predicting it, measuring it, and explaining why reads as competence.

- **The recurrent state is FP32 and we are not narrowing it for throughput.** At
  INT4 weights, BF16 state saves ~2–3% of decode traffic — not worth the precision
  risk on the model's memory mechanism
  ([QUANTIZATION_POLICY.md §3](./QUANTIZATION_POLICY.md)).

- **Negative and partial results are written up honestly.** "We tried X, it didn't
  help, here's the profile showing why" is worth real points under Potential Impact
  and costs nothing under scrutiny.

---

## 7. Instrumentation summary

| Component | Module | Status |
|---|---|---|
| Benchmark runner (sweep, warmup, repeats, percentiles) | `bench/harness.py` | ✅ Landed (ob-ljh) |
| Schema validation (frozen tidy/long CSV) | `bench/schema.py` | ✅ Frozen (ob-q9i) |
| Metric definitions (timing protocol, memory formulas) | `docs/METRICS.md` | ✅ Frozen (ob-ar3) |
| Provenance capture | `bench/manifest.py` | ✅ Landed (ob-u37) |
| Statistical helpers (percentile, summarize) | `bench/metrics.py` | ✅ Landed (ob-ljh) |
| Correctness oracle | `ob-3uh` | ⏳ Pending x86 reference (ob-aqv) |
| Energy/token sampling | `ob-agf` | ⏳ Future |
| Plot and table generation | `ob-9y8` | ⏳ Pending |
