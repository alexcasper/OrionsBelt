# Methodology

**Bead:** `ob-aoo` · **Status:** Complete 2026-08-02 · **Parent:** `ob-mrd` (E5)

This document is the **submission-facing methodology** — the narrative a judge reads to
understand how every number in this project was produced and why it is trustworthy. It
synthesises the operational specifications into a coherent whole. For the exhaustive
per-metric timing contracts, see [`docs/METRICS.md`](./METRICS.md); for the data format,
see [`docs/RESULTS_SCHEMA.md`](./RESULTS_SCHEMA.md); for the device run procedure, see
[`docs/DEVICE_RUNBOOK.md`](./DEVICE_RUNBOOK.md).

> **Transparent methodology is what separates a credible benchmark from a marketing number.**
> Every claim here is backed by an executable spec, a committed manifest, or a committed
> source document. Nothing is asserted without a traceable path to the raw measurement.

---

## 1. What we measure and why

Gated DeltaNet (GDN) is a hybrid linear-attention architecture with **O(1) recurrent
state** per token — unlike full attention, whose KV cache grows linearly with context
length. Qwen3.5 uses a 3:1 ratio of GDN to full-attention layers (24 GDN + 8 attention
for the 4B checkpoint), verified from the modeling code
([`GDN_LAYER_AUDIT.md`](./GDN_LAYER_AUDIT.md), `ob-37v`).

This benchmark exists to answer three questions:

1. **Does GDN's flat recurrent state produce a real memory advantage at long context?**
   We decompose peak memory into three components — weights (flat), KV cache (linear),
   recurrent state (O(1)) — and show the crossover.
2. **Where does kernel optimization help, and where doesn't it?** Prefill (chunkwise
   matmuls) is compute-bound and optimisable; decode (single-token recurrence) is
   bandwidth-bound at ~0.25 FLOP/byte and is not moved by kernel work
   (METRICS.md §9).
3. **What is the practical decode-throughput ceiling on Arm silicon?** We derive it from
   measured bandwidth and compare against the device's spec bandwidth.

---

## 2. Experimental design

### 2.1 Context sweep

Every measurement is run at four canonical context lengths, covering three orders of
magnitude:

| Sweep point | Purpose |
|---:|---|
| 4,096 | Short-context baseline (typical chat) |
| 32,768 | Medium context (document-level) |
| 131,072 | Long context (novel-length) |
| 262,144 | Maximum native context for Qwen3.5 |

This sweep is the backbone of the scaling story — it is where KV cache growth becomes
visible against the flat recurrent state.

### 2.2 Model checkpoints

**Primary:** `Qwen3.5-4B` (24 GDN + 8 attention layers, 2560 hidden size, 262K native
context). **Fallback:** `Qwen3.5-0.8B` (18 GDN + 6 attention, same architecture ratio).
Selection rationale: [ADR 0003](./adr/0003-model-checkpoint-selection.md).

### 2.3 Engines and quantization

The harness is **backend-agnostic** (`bench/harness.py`, `ob-ljh`) — the `BenchmarkBackend`
ABC allows different execution engines (CPU NEON/SVE, NPU, GPU Vulkan) to be benchmarked
under identical timing control. Per-layer-class engine assignment (GDN layers on one
engine, full-attention layers on another) is captured in the schema's separate
`engine_gdn` and `engine_full_attention` columns.

Quantization follows the per-tensor policy in [ADR 0006](./adr/0006-quantization-policy.md):
INT4 weight-only for large projections (99.87% of params), FP16 carve-outs for
precision-sensitive tensors (gates, norms, conv), and an unchangeable fp32 floor on the
recurrent state.

---

## 3. Timing methodology

Every timing decision is specified exhaustively in [METRICS.md](./METRICS.md). The
load-bearing principles:

### 3.1 The prefill/decode boundary

**Token 1 belongs to prefill, not decode.** Token 1 is produced by the prefill forward
pass — no autoregressive step is involved. Decode throughput measures only tokens 2..N,
which are the steps that exercise the single-token GDN recurrent update and KV-cache
append. This is architectural, not a convention of convenience.

### 3.2 Measurement excludes model load

Every timer starts after the model is loaded, weights are resident, and warmup repeats
are complete. Model loading, checkpoint deserialisation, and JIT compilation are
one-time process costs, not per-request costs.

### 3.3 Monotonic clock

All durations use `time.perf_counter()` / `clock_gettime(CLOCK_MONOTONIC)`, never
wall-clock time-of-day, which can jump under NTP adjustment.

### 3.4 Batch size = 1

All formulas assume single-request, non-batched inference, matching the project's
benchmark scope. A batched serving test would require reintroducing batch-size factors
into every per-token byte/FLOP formula.

### 3.5 Fixed generation length

Decode is measured over a fixed `N = 257` generated tokens (1 prefill + 256 decode).
EOS does not stop generation — the decode loop runs to exactly N tokens so throughput
is a controlled, comparable quantity.

---

## 4. Metrics

Five metrics, each with a frozen `metric_name`, canonical `unit`, and restricted `phase`
([RESULTS_SCHEMA.md](./RESULTS_SCHEMA.md) §4):

| Metric | Unit | Phase | What it tells you |
|---|---|---|---|
| `prefill_tokens_per_sec` | tok/s | prefill | Prompt-processing throughput — where GDN kernel optimization shows the 1.38–1.49× win |
| `decode_tokens_per_sec` | tok/s | decode | Steady-state generation — bandwidth-bound, expected flat across optimizations |
| `ttft_seconds` | s | prefill | User-facing latency from prompt submission to first token (includes tokenization) |
| `peak_memory_bytes` | bytes | both | Three-component split: weights / kv_cache / recurrent_state |
| `energy_joules_per_token` | J/tok | both | Energy cost per token from on-board power sampling |

**Prefill and decode are never averaged into a single throughput number.** The `phase`
column is required and explicit on every row — the project's central finding (prefill
is optimisable, decode is bandwidth-bound) would be invisible without this separation.

---

## 5. Memory attribution: the three-component decomposition

Peak memory is decomposed into three components, each computed from model introspection
and known tensor shapes — **never from process RSS** (METRICS.md §5.0 explains why RSS
cannot be split).

| Component | Formula | Scaling with context |
|---|---|---|
| **Weights** | Σ `numel × bytes_per_element` over all parameter tensors | Flat (constant) |
| **KV cache** | `num_attn_layers × 2 × seq_len × n_kv_heads × head_dim × dtype_bytes` | **Linear** with context length |
| **Recurrent state** | `num_gdn_layers × n_v_heads × d_k × d_v × 4` (fp32) | **O(1)** — flat at every context length |

This decomposition is the project's central claim made quantitative. At 262K context on
the 4B model, the hybrid stack needs 8.0 GB of KV cache + 48 MB of recurrent state —
versus ~32.8 GB had all layers been full attention (ADR 0003). The recurrent state is
**48 MB flat, regardless of context length** — that is the advantage of GDN over
attention, demonstrated as a measurement, not an assertion.

All formulas are implemented in [`bench/metrics.py`](../bench/metrics.py) and unit-tested
against known-good values ([`tests/test_metrics.py`](../tests/test_metrics.py)).

---

## 6. Statistical protocol

Full specification: [METRICS.md](./METRICS.md) §7.

### 6.1 Warmup

Three full repeats (prefill + decode at the sweep point's actual context length) are
discarded before measurement. This absorbs allocator warm-up, JIT compilation,
lazy page faults, and cold-to-steady thermal ramp.

### 6.2 Repeats and percentiles

| Tier | Context lengths | Repeats (N) |
|---|---|---:|
| Exploratory / dev-loop | any | 10 |
| Headline (write-up table) | 4K, 32K | 30 |
| Headline (write-up table) | 128K, 262K | 10 (minimum) |

**Never N < 5.** Below 5, a percentile is indistinguishable from min/max.

**p50 and p95 are mandatory.** p99 is explicitly not required — at N=10–30 it would
collapse to "the max observed sample," which is the single-best-run reporting we prohibit.

**Dispersion** is reported as both absolute spread (`p95 − p50`) and normalised spread
(`(p95 − p50) / p50`) for cross-context comparison.

### 6.3 Minimum reportable difference

A claimed improvement between configurations A and B is only reportable as real (not
noise) if **both**:

1. `|p50(A) − p50(B)| > 2 × max(spread(A), spread(B))`, and
2. the p50-to-p95 ranges do not overlap.

This non-parametric rule replaces a t-test, whose normality assumption is not justified
at N=10–30 on hardware with one-sided thermal-throttling noise.

---

## 7. Thermal and frequency control

### 7.1 CPU governor

All benchmark runs set the CPU governor to `performance` on every core, locking
frequency at maximum. The governor state is recorded in the run manifest. A run
under a different governor is a different experiment and is not comparable.

### 7.2 Core pinning on big.LITTLE

Asymmetric Arm SoCs (RK3588: 4× A55 + 4× A76) require explicit core pinning
(`taskset`). An unpinned run on a big.LITTLE SoC is close to meaningless because the
scheduler will migrate the process mid-measurement between clusters with different
clock speeds and cache hierarchies. The pinning cluster (big vs little) is recorded
in the run manifest.

### 7.3 Thermal monitoring

Thermal zone temperatures are read before and after every run and committed in the
manifest. A core-clock drop greater than 10% from start to end of the measured
repeats invalidates the run for headline reporting (METRICS.md §8).

---

## 8. Provenance: every number has a manifest

**A number without a manifest is not a result** (PLAN.md §9).

Every benchmark run produces two artifacts:

1. **CSV** (`results/raw/<run_id>.csv`) — the measurements in tidy/long format
   ([RESULTS_SCHEMA.md](./RESULTS_SCHEMA.md)).
2. **Manifest** (`results/manifests/<run_id>.json`) — the provenance record, captured by
   [`bench/manifest.py`](../bench/manifest.py). Includes:
   - Device: hostname, CPU model, core count, topology (per-core governor, min/max freq)
   - ISA features: `asimddp`, `bf16`, `i8mm`, `sve`, `sve2`, `sme`
   - Thermal zones: all `thermal_zone*` readings
   - Memory: total and available
   - Software: Python version, installed packages (torch, transformers, onnxruntime)
   - Git: commit SHA and dirty flag
   - Timestamp: ISO 8601 UTC

The CSV's `manifest_ref` column links each row to its manifest. The `git_sha` column
links each row to the exact code commit that produced it.

---

## 9. Correctness validation

**The correctness oracle gates every optimization.** Speed that changes outputs is
not speed (PLAN.md §9).

The x86/CUDA reference inference (`ob-aqv`) serves as the correctness oracle — its
outputs are the ground truth against which all optimised paths are compared. The
validation protocol for quantization (ADR 0006) requires:

- **Token-level cosine similarity** > 0.995 between quantised and fp16 reference at
  sampled positions across 4K, 32K, and 128K context.
- **Perplexity** within the METRICS.md §7 minimum reportable difference.
- **Long-sequence drift test:** output-token distribution compared to reference at
  tokens 1, 64, 128, 256 — catching accumulation errors that short tests miss.

Any tensor failing these thresholds is dropped to the next precision tier (INT4 → INT8
→ FP16) before the result is reported.

---

## 10. What invalidates a measurement

Any of the following flags the affected rows as invalid for headline reporting
(METRICS.md §8):

- **Thermal throttle mid-run** (>10% clock drop start-to-end)
- **Background load** during the run (non-idle system)
- **First-run page faults / allocator warm-up** not absorbed by the 3 discarded warmups
- **Inconsistent governor state** across compared runs
- **Non-monotonic timing source** (wall-clock instead of monotonic clock)
- **Config/runtime mismatch** for memory metrics (intended vs actual runtime dtype)
- **QEMU timings** — emulation is for correctness verification only, never performance
  ([FINDINGS.md](./FINDINGS.md) §5)

---

## 11. Reproducibility

### 11.1 Device benchmark (static microbenchmark)

```bash
# On the x86 host:
./scripts/build_device_bench.sh          # produces dist/bench_gdn_*

# Copy to device and run:
scp dist/bench_gdn_<variant> <device>:/tmp/
ssh <device>
# Set governor to performance, pin to a cluster, run:
/tmp/bench_gdn_<variant> --repeats 30 --csv > results.csv
python3 /tmp/manifest.py > manifest.json
```

Full procedure: [`DEVICE_RUNBOOK.md`](./DEVICE_RUNBOOK.md).

### 11.2 Harness benchmark (full model)

```bash
python3 -m bench.harness --backend mock --contexts 4096,32768 --repeats 30 \
    --device generic_aarch64 --engine-gdn cpu
```

The `MockBackend` produces deterministic synthetic data for CI testing; real backends
(transformers, llama.cpp) plug into the same harness.

### 11.3 Clean-clone verification

Bead `ob-kdi` (unblocked by this document) will verify that a clean clone of the repo
followed by the README produces a passing test suite and a smoke-run benchmark result.
