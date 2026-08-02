# Metric definitions

**Bead:** `ob-ar3` (`t-metrics-spec`) · **Status:** Frozen 2026-08-02, alongside
[`docs/RESULTS_SCHEMA.md`](./RESULTS_SCHEMA.md) · **Parent:** `ob-mrd` (E5)

This document is the operational contract for every `metric_name` in the frozen results
schema (`docs/RESULTS_SCHEMA.md` §4, `bench/schema.py::MetricName`). The schema fixes the
*vocabulary* — column names, enum values, units. This document fixes the *semantics* —
exactly which wall-clock instant starts and stops each timer, what is and is not counted
in each numerator/denominator, how memory is attributed, and what invalidates a run. Two
independent people following this document with two different harness implementations
should get numbers that are directly comparable. If a harness's behavior and this
document disagree, the harness is wrong.

This bead closes two open questions flagged during `ob-q35` and left deliberately
qualitative by `CONTRIBUTING.md`: the minimum repeat count `N`, and which percentiles are
mandatory. §7 answers both with concrete defaults.

**Vocabulary discipline.** Every definition below uses exactly the `metric_name`, `phase`,
`metric_component`, and `unit` values frozen in `docs/RESULTS_SCHEMA.md` — nothing here
invents a new metric or renames an existing one. Where this document needs to talk about
something the schema doesn't have a column for (tokenization time, idle power baseline,
activation memory), that is called out explicitly as **out of schema** so it is never
confused with a load-bearing figure.

---

## 1. Global conventions

These apply to every metric below and are not repeated per-section.

**Clock.** All durations are measured with a monotonic clock (e.g. `time.perf_counter()` /
`clock_gettime(CLOCK_MONOTONIC)`), never wall-clock time-of-day. A wall-clock read can
jump backwards or forwards under NTP adjustment; a monotonic clock cannot. The
`timestamp` column in the results schema (ISO 8601, for provenance/ordering) is a
separate concern from the clock used to compute `value` for a `seconds`-unit row —
the former is "when," the latter is "how long."

**The measurement boundary excludes model load.** Every metric's timer starts *after* the
model is loaded, weights are resident, and the process has completed the warmup repeats
in §7. Model loading, checkpoint deserialization, and any one-time JIT/compile step are
never part of any reported metric — they are one-time process-lifetime costs, not
per-request costs, and mixing them into a per-token or per-request number would make
runs with different process lifetimes incomparable. If load time itself is worth
reporting, it belongs in the manifest, not in `results/raw/`.

**`t_submit` is the shared origin.** For a given (prefill, decode) request, define
`t_submit` = the wall-clock instant the harness hands the prompt string to the
model-serving entry point, with the model already loaded and warm. Every phase timer
below is defined relative to `t_submit` or to another named event, never to an
independent zero.

**Batch size is 1.** Every formula and every timer below assumes single-request,
non-batched inference, matching this project's benchmark scope (PLAN.md §2.4, §3.1: the
recurrent state and KV cache figures the project reports are per-request). If batched
serving is ever benchmarked, every per-token byte/FLOP formula in §5 and §9 needs a
batch-size factor reintroduced — flag this explicitly rather than silently assuming it
away.

**The first generated token belongs to prefill, not decode.** This is the single most
consequential boundary decision in this document, so it is stated once here and then
referenced everywhere it matters (§3, §4). In every architecture this project measures,
producing token 1 requires only the logits already computed for the prompt's last
position by the prefill forward pass — there is no additional autoregressive,
single-token forward step involved. Token 1 is therefore counted toward
`prefill_tokens_per_sec`'s denominator-side work and `ttft_seconds`, and it is
**excluded** from `decode_tokens_per_sec`'s numerator. `decode_tokens_per_sec` measures
only tokens 2..N, i.e. only steps that actually exercise the single-token GDN recurrent
update and (for the periodic full-attention layers) a KV-cache append — which is the
thing this project's central claim is about. Counting token 1 as a "decode" token would
blend one prefill-shaped event into a metric whose entire point is isolating the
bandwidth-bound steady-state recurrence.

---

## 2. `prefill_tokens_per_sec`

*Schema: `unit=tokens_per_sec`, `phase=prefill` only, `metric_component` empty.*

**Start event:** the instant the prefill forward pass is invoked over the full,
already-tokenized prompt (`t_prefill_start`). This is *after* tokenization completes —
tokenization is a CPU string/BPE operation, not a model-execution cost, and folding it
into a "tokens/sec" throughput figure would make the number partly about the tokenizer
implementation rather than about the GDN/attention kernels this metric exists to
characterize.

**Stop event:** the instant the forward pass over the full prompt completes and the
last prompt position's logits are materialized (`t_prefill_logits`) — *before* sampling
token 1. Sampling is common to both this metric and `ttft_seconds` but belongs, by
construction, only to the latter (see §3).

**Numerator:** the number of prompt tokens fed to the model — i.e. `len(input_ids)`
after tokenization, not characters, not words, not the requested `context_length`
rounded to a sweep point. For a run at the `context_length=131072` sweep point, if the
corpus prompt actually tokenizes to 131,050 tokens, the numerator is 131,050.

**Denominator:** `t_prefill_logits - t_prefill_start`, in seconds.

**Value:** `prompt_token_count / (t_prefill_logits - t_prefill_start)`.

**Batching within prefill.** If the prefill implementation internally chunks the prompt
(e.g. chunkwise GDN scan processing 64-token chunks), that is invisible to this metric —
the timer spans the *entire* prefill call, chunked or not. Chunking is an implementation
detail of how the 1.38–1.49× kernel win is realized; it does not change what event
starts or stops the outer timer.

---

## 3. `ttft_seconds`

*Schema: `unit=seconds`, `phase=prefill` only, `metric_component` empty.*

**Start event:** `t_submit` (§1) — the prompt string handed to the harness, model already
loaded and warm. **This includes tokenization.** Rationale: TTFT is explicitly documented
in `docs/RESULTS_SCHEMA.md` §4 as "the user-facing face of prefill throughput," and a
user submitting a prompt experiences tokenization as part of their wait. Excluding it
would make TTFT a purely internal model-execution number, which is what
`prefill_tokens_per_sec` already is — the two metrics would become redundant instead of
answering different questions ("how fast is the kernel" vs. "how long does the user
wait"). **Consequence to watch:** at 128K/262K context, tokenization of a long prompt is
not free. If it is ever large enough to be a meaningful fraction of TTFT, that fact
belongs in `notes` (schema §3) so a reader isn't confused about why TTFT and
`1/prefill_tokens_per_sec × prompt_token_count` don't match exactly.

**Stop event:** `t_first_token` — the instant token 1's id is produced, i.e. immediately
after sampling (argmax or whatever sampler is configured) is applied to the prefill
pass's last-position logits. **Excludes detokenization** (converting the token id back
to display text) and any output streaming/transport — those are downstream of "the model
produced a token" and are not part of this project's benchmark scope (there is no
network hop in this harness; PLAN.md's targets are all local inference).

**Value:** `t_first_token - t_submit`.

**Relationship to `prefill_tokens_per_sec`:** `ttft_seconds` ⊇ tokenization time +
prefill-forward time (the same interval `prefill_tokens_per_sec` measures) + sampling
time for one token. Sampling time is O(vocab_size), a single softmax/argmax over one
position, and is expected to be negligible relative to the forward pass at any context
length worth benchmarking — but it is not assumed to be exactly zero, hence the separate
stop event rather than reusing `t_prefill_logits`.

---

## 4. `decode_tokens_per_sec`

*Schema: `unit=tokens_per_sec`, `phase=decode` only, `metric_component` empty.*

**Start event:** `t_decode_start := t_first_token` (§3) — decode begins the instant
token 1 exists and the recurrent state / KV cache populated by prefill is carried
forward into the autoregressive loop.

**Stop event:** the instant token `N`'s id is produced, where `N` is the fixed generation
length for the run (see the generation-length convention below).

**Numerator:** `N - 1` — the count of tokens produced by actual single-token decode
steps. **Token 1 is not counted** (§1). This is the direct answer to "is the first token
counted in decode throughput": no, and the reason is architectural, not a convention
chosen for convenience — token 1 was never produced by a decode-phase forward pass in
the first place.

**Denominator:** `t_N - t_decode_start`, the total wall-clock span of all `N-1` decode
steps, summed as one interval — **not** the mean of `N-1` individually-timed per-step
rates. Report `(N-1) / (t_N - t_decode_start)` as a single ratio-of-sums. Averaging
per-step instantaneous rates instead (mean of `1/Δt_i`) is a different and biased
quantity (harmonic-mean-shaped, sensitive to a single slow step in a way the
ratio-of-sums is not) and must not be used.

**Generation-length convention.** Decode throughput must be measured over a fixed,
pre-declared number of generated tokens per run, independent of context length, so that
`repeat_index` rows within a run — and rows across different context-length sweep points
— differ only in *rate*, not in how much work each repeat happened to do. **Default:
`N = 257`** (1 prefill-produced token + 256 decode-phase tokens), chosen because it is
long enough to amortize any residual per-call dispatch jitter (16 engine-boundary
crossings per token per PLAN.md §3.1, if a heterogeneous mapping is in effect) into a
stable steady-state rate, while remaining cheap enough to run at every sweep point
including 262K, where prefill alone already dominates wall time. **EOS must not stop
generation early:** the decode loop runs to exactly `N` tokens regardless of any sampled
end-of-sequence token, ignoring/discarding EOS for benchmarking purposes only. Without
this, `N-1` would vary repeat-to-repeat depending on where the model happened to sample
EOS, breaking the fixed-denominator comparability the schema's `repeat_count` column is
built to support. This is a deliberate divergence from a realistic interactive session,
made solely so throughput is a controlled, comparable quantity — flag it in `notes` only
if a specific run's harness could not honor it (e.g. a vendor runtime that hard-stops on
EOS), since that would itself invalidate cross-run comparison for that row.

---

## 5. `peak_memory_bytes` × `metric_component`

*Schema: `unit=bytes`, `phase ∈ {prefill, decode}`, `metric_component` **required**: one
of `weights`, `kv_cache`, `recurrent_state`.*

### 5.0 The pitfall this section exists to avoid

**Process RSS (or any OS-reported resident-set/allocator high-water-mark figure) cannot
be split into these three components.** RSS is one number for the whole process's
resident pages; weights, KV cache, recurrent state, the framework's tensor allocator
overhead, and interpreter/runtime baseline are all mixed into it with no OS-visible tag
saying which byte belongs to which logical buffer. A KV cache growing by 10MB and a
framework's allocator arena growing by 10MB for an unrelated reason are indistinguishable
in an RSS reading. **Consequently, none of the three `peak_memory_bytes` rows in this
schema may be derived from RSS.** They must instead be derived from **model
introspection plus known tensor shapes** — walking the loaded model's actual parameter
and buffer tensors (or, where that is impractical for a given engine/runtime, computing
the byte count analytically from the checkpoint's config and the run's actual
quantization/precision) — the same way `bench/README.md` rule 2 and
`docs/RESULTS_SCHEMA.md` §3 already require the three-way split to exist as first-class
values, not as an incidental byproduct of a single OS reading.

RSS still has a legitimate, narrower role: as a **cross-check**, recorded in the
manifest or `notes`, not in `results/raw/` as one of the three components. If RSS
diverges persistently from `weights + kv_cache + recurrent_state + <documented
overhead>`, that gap is real and must be investigated (activation memory, allocator
fragmentation, mmap'd page cache for the weight file, a second framework's runtime) —
never silently folded into one of the three buckets to make the arithmetic close.

**Scope limitation, stated honestly:** none of the three components here is "activation
memory" (the transient intermediate tensors produced during a forward pass — attention
scores, MLP hidden states, etc.). The schema's three enum values do not include an
activation component, and this document does not invent one. `weights` below is
restricted strictly to parameter tensors; it must never silently absorb activation
memory to make a total look complete. If activation memory turns out to be large enough
to matter for the write-up, that is worth surfacing as an explicit finding (a fourth,
currently-unaccounted memory consumer) rather than quietly merged into an existing
component — `docs/RESULTS_SCHEMA.md` §6 permits adding a new optional column or enum
value additively for exactly this kind of case, without invalidating existing rows.

### 5.1 Sampling instant: peak within a phase is the phase-end value

Both `kv_cache` and the two variable-length quantities relevant to it grow
**monotonically** within a phase (the KV cache never shrinks mid-generation; the number
of tokens processed only increases). Therefore the *peak* value for a phase is
analytically equal to its value at the **end** of that phase — no dense time-series
polling of memory is required.

- **`phase=prefill`** rows are sampled at `t_prefill_logits` (§2) — the instant the full
  prompt's KV cache / recurrent state has been populated and before token 1 is sampled.
- **`phase=decode`** rows are sampled at `t_N` (§4) — the instant the last requested
  decode token has been produced, i.e. the point of maximum accumulated KV cache growth
  and (trivially, since it doesn't grow) the same recurrent-state footprint as at every
  other decode step.

(At very short context — the 4K sweep point — a `prefill`-phase KV-cache sample differs
from a sample taken one token later, at the start of decode, by exactly one token's worth
of cache: a ~1/4096 relative difference. This is noted for completeness; it is far below
run-to-run noise and not a correction anyone needs to apply.)

### 5.2 `metric_component = weights`

**Definition:** Σ over every parameter tensor actually resident in memory for this run,
of `numel(tensor) × bytes_per_element(tensor's actual runtime dtype)`. "Actual runtime
dtype" means: if the `quantization` column says `int8_w8a8`, the weights counted must be
the ones actually loaded as int8 — read this from the live model's tensors via
introspection, never assumed from the `quantization` string, since a config/runtime
mismatch would otherwise silently corrupt the row.

**Expected behavior (sanity check, not a measurement):** flat across `context_length`,
because weights don't depend on how much has been processed. If two rows for the same
`model_checkpoint` + `quantization` at different `context_length` values disagree on
`weights`, that indicates a harness bug (e.g. accidentally counting something
context-dependent), not a real hardware effect — treat it as a correctness failure to
fix, not a result to report.

### 5.3 `metric_component = kv_cache`

**Definition:** total bytes held by the full-attention layers' key/value cache at the
sampling instant (§5.1). Computed as:

```
kv_cache_bytes = num_full_attention_layers × 2 (K and V)
               × batch (= 1, §1)
               × seq_len_at_sample_instant
               × n_kv_heads × head_dim
               × bytes_per_element(cache dtype)
```

`num_full_attention_layers` is read from the checkpoint's `layer_types` (the count of
`"full_attention"` entries) — 8 for the verified 32-layer dense config
(`docs/CLAIM_VERIFICATION.md` §2.3), but must be read per-checkpoint, never hardcoded,
since MoE variants differ (40 layers total, per PLAN.md §2.2/§1.2). `n_kv_heads` and
`head_dim` are the **full-attention** layers' own attention config fields (e.g.
`num_key_value_heads`, `head_dim` in the checkpoint's `config.json`) — these are a
different set of shapes from the GDN linear-layer dimensions
(`linear_key_head_dim`/`linear_num_key_heads`/etc.) documented in
`docs/CLAIM_VERIFICATION.md` §2.3, and must not be conflated with them. `seq_len_at_
sample_instant` is the number of tokens whose K/V have been cached at that instant
(the full prompt length at prefill's end; prompt length + generated-tokens-so-far at
decode's end).

**Expected behavior:** grows linearly with `context_length` — this is the component that
scales, and is one half of the project's central three-way-split claim.

### 5.4 `metric_component = recurrent_state`

**Definition:** total bytes held by the GDN layers' recurrent state at the sampling
instant. Computed as:

```
recurrent_state_bytes = num_gdn_layers × H × d_k × d_v
                       × batch (= 1, §1)
                       × bytes_per_element(state dtype)
```

`num_gdn_layers` is the count of `"linear_attention"` entries in `layer_types` — 24 for
the verified 32-layer dense config. `H`, `d_k`, `d_v` are read from the specific
checkpoint's own `linear_num_key_heads`/`linear_key_head_dim`/`linear_value_head_dim`
config fields, **not** from the family-default values — `docs/CLAIM_VERIFICATION.md`
§2.3 already documents that `linear_num_value_heads` is 16 at 0.8B/2B and 32 at 4B/9B, so
a hardcoded `H=16` would silently misreport this figure for the larger checkpoints.

Wherever practical, cross-check this analytic figure against the *actual* allocated
state-tensor shape read from the running model/kernel (introspection), not just the
config-derived formula — the formula assumes the implementation is correct; introspecting
the live tensor catches the case where it isn't (e.g. a bug that lets state grow with
context, which would silently falsify the project's central claim if only ever computed
analytically and never checked against reality).

**Expected behavior:** flat (O(1)) with respect to `context_length`, by construction —
this is the other half of the three-way-split claim, and the one that must survive
contact with a real, introspected measurement rather than only a formula.

### 5.5 Units and precision (all three components)

Reported in the schema's `bytes` unit as an exact, non-rounded integer count (stored in
the `float`-typed `value` column, per `bench/schema.py`, without loss — byte counts here
are always exactly representable in a float64). No rounding is applied at collection
time; rounding to human-friendly units (KiB/MiB/GiB) is a presentation-layer concern for
`bench/plots.py` and the results table, never for `results/raw/`.

---

## 6. `energy_joules_per_token`

*Schema: `unit=joules_per_token`, `phase ∈ {prefill, decode}`, `metric_component` empty.*

**Window:** identical to the window used for the corresponding phase's throughput
metric — `[t_prefill_start, t_prefill_logits]` for `phase=prefill`,
`[t_decode_start, t_N]` for `phase=decode`. Reusing the exact same start/stop instants
(rather than an independently-timed power-sampling window) is deliberate: it guarantees
`energy_joules_per_token` and `{prefill,decode}_tokens_per_sec` divide the same amount of
work by directly comparable time spans, so a reader can sanity-check one against the
other.

**Numerator:** total energy in Joules, obtained by trapezoidal integration of
instantaneous power samples `P(t)` (Watts) collected over the window above:
`E = ∫ P(t) dt ≈ Σᵢ (P(tᵢ) + P(tᵢ₊₁))/2 × (tᵢ₊₁ − tᵢ)`.

**Denominator:** the same token count as the corresponding `*_tokens_per_sec` metric for
that phase — prompt-token count for `prefill`, `N-1` for `decode` (§4; token 1 is
excluded from decode energy accounting for the same architectural reason it is excluded
from decode throughput).

**Instrumentation, ranked, and what must go in the manifest.** Neither `PLAN.md` nor
`docs/CLAIM_VERIFICATION.md` specifies which power-sampling capability the target
hardware actually exposes (a separate bead, `ob-agf`, owns integrating whatever is
available) — this document therefore does not invent a specific sensor. In descending
order of preference:

1. **On-board rail/SoC power monitor** (e.g. an INA-family sensor exposed via sysfs or a
   vendor power tool) sampled at a fixed rate — the tightest attribution to the actual
   compute event.
2. **External bench power meter** at the board's input supply — coarser (includes
   whatever fixed peripheral/idle draw the board has), but still valid for **relative**
   comparison across runs on the same board, since that fixed overhead is constant.
3. **OS-level energy-counter API** (an Arm-SoC equivalent of x86 RAPL), if one exists for
   this platform — currently unverified; do not assume it is present.

Whichever is used, the manifest (per PLAN.md §9 — "every run emits a manifest") must
record: which of the three, the sample rate (Hz), and the instrument's stated
accuracy/resolution, since that resolution is the practical precision floor for this
metric — reporting more significant figures than the instrument resolves would be false
precision.

**Baseline convention:** report **gross** energy — the raw integral of `P(t)` over the
window, with **no idle-baseline subtraction**. This is a deliberate choice, stated here
so it is not re-litigated per run: idle-baseline subtraction requires a second,
independently-timed measurement (idle power immediately before the run) that is itself
subject to thermal/background drift, and the schema has no column for a baseline value
to subtract. If idle/background power is large enough relative to workload power to
distort the comparison meaningfully, record that qualitative observation in `notes`
rather than subtracting a number that isn't itself part of the frozen schema.

**Units and precision:** Joules per token, float, no rounding at collection time;
present to the instrument's actual resolution (typically 3 significant figures) in
downstream tables.

---

## 7. Statistical protocol

This section answers the two questions `ob-q35` flagged as unresolved by `PLAN.md`:
minimum repeat count, and which percentiles are mandatory.

**Warmup: 3 repeats, discarded, never written to `results/raw/`.** Each warmup repeat
runs the **full** phase (prefill *and* decode, at the sweep point's actual
`context_length`) — not a short synthetic warmup — so that:
- allocator arenas (framework caching allocator, KV-cache/recurrent-state buffers) are
  already sized to steady state before the first *measured* repeat, rather than resizing
  mid-measurement and contaminating one repeat as an outlier;
- any first-call JIT/kernel-compile cost (e.g. a Vulkan shader's first pipeline
  creation) has already paid its one-time cost;
- lazily-mapped weight-file pages (if weights are `mmap`'d) have already faulted in,
  so `repeat_index=0` isn't silently slower purely from cold page faults;
- the board has ramped from cold idle temperature toward its steady operating
  temperature under load, so the first *measured* repeat isn't an artifact of a cold
  start rather than a real number.

Three is chosen as the smallest count that plausibly absorbs all four effects above
without materially extending sweep time at the expensive end of the context sweep
(128K/262K); it is not derived from a formal convergence test, and if a specific engine
is observed to still be warming up after 3 repeats (check: is `repeat_index=0`'s value
an outlier relative to the rest of the run?), raise the warmup count for that
engine/config and note it.

**Measured repeats — two tiers, both reporting `repeat_count` per PLAN.md §9:**

| Tier | Context lengths | `N` (repeat_count) | When to use |
|---|---|---:|---|
| **Exploratory / dev-loop** | any | **10** | Iterating on a change, sanity-checking a mapping decision, CI smoke runs |
| **Headline (write-up table)** | 4K, 32K | **30** | Any number that lands in the final comparison table or the README |
| **Headline (write-up table)** | 128K, 262K | **10** (minimum) | Same, but wall-clock cost per repeat is large enough that 30 is not practical within the schedule; PLAN.md's own descope ladder (§7) already accepts dropping the 262K point entirely under schedule pressure before it would accept an unrepeatable number — reducing repeats at these two points, with the reduction itself recorded in `notes`, is the smaller compromise |

**Never report `N < 5`, for any metric, under any schedule pressure.** Below 5, a
"percentile" is indistinguishable from "the observed min/max," which defeats the purpose
of the whole percentile-reporting convention (PLAN.md §9); at that point the honest
thing to report is exactly what it is — an unrepeated single measurement or a tiny
sample — not a percentile that implies more statistical grounding than 5 points provide.

**Percentiles: p50 and p95 are mandatory. p99 is explicitly not mandated, and here is
why:** with the nearest-rank method, resolving the 95th percentile from data (rather than
extrapolating/interpolating between the two most extreme observed samples) needs on the
order of `1/(1-0.95) = 20` samples for there to typically be at least one observation
beyond it; p99 needs on the order of 100. At this project's repeat counts (10–30), a
reported "p99" would in practice be nearly indistinguishable from "the max observed
sample," which is exactly the single-best/single-worst-run reporting PLAN.md §9
prohibits — just at the other tail. Reporting p50/p95 at N=10–30 is honest about what the
sample size can support; reporting p99 at the same N would not be. If a future run ever
uses N≥100 specifically to resolve tail behavior (e.g. investigating a specific
thermal-throttle hypothesis), p99 becomes meaningful and may be added — but it is not a
default expectation.

**Dispersion:** alongside p50 (the headline number) and p95 (the tail/variance
indicator), report the **spread** `p95 - p50` and, for cross-metric/cross-context
comparison, the **normalized spread** `(p95 - p50) / p50`. The normalized form is what
makes "how noisy was 4K decode" comparable to "how noisy was 262K prefill" despite very
different absolute magnitudes — directly relevant given PLAN.md §7 (R7) names thermal
variance on passively-cooled edge hardware as a specific, named risk, not a generic
disclaimer.

**Minimum reportable difference.** Given run-to-run variance, a claimed improvement
between two configurations A and B (e.g. "baseline" vs. "optimized kernel") is only
reportable as a real difference — not noise — if **both** of the following hold:

1. `|p50(A) - p50(B)| > 2 × max(spread(A), spread(B))`, where `spread(X) = p95(X) - p50(X)`
   as defined above; and
2. the p50-to-p95 ranges of A and B do not overlap — i.e. if A is claimed faster, then
   `p95(A) < p50(B)` (A's noisy top case is still better than B's typical case).

This is a deliberately simple, non-parametric rule rather than a t-test, because a
t-test's normality assumption is not well justified at N=10–30 on hardware with known
non-Gaussian noise sources (thermal throttling produces a one-sided, not symmetric,
degradation). Both conditions together guard against the two failure modes separately:
condition 1 rejects "the point estimates differ, but so does the noise" false positives;
condition 2 rejects "the point estimates differ by a lot, but only because of one
outlier tail" false positives. A difference that fails this test should be reported as
"not distinguishable from run-to-run variance at this repeat count," which is itself a
valid, honest finding (per `CONTRIBUTING.md`'s honest-reporting section) — not omitted.

---

## 8. What invalidates a measurement

Any of the following, if present, means the affected row(s) must be flagged (`notes`
column; the underlying cause belongs in the manifest per PLAN.md §9) and must not be
reported as a clean headline number without that caveat:

- **Thermal throttle mid-run.** Compare the manifest-recorded core/GPU clock at the end
  of the run to its value at the start of the first *measured* repeat (not the warmup
  repeats, which are expected to include some thermal ramp, §7). A drop greater than
  **10%** invalidates the run for headline reporting — the fix is a cooldown pause
  between repeats or between context-length sweep points, then a re-run, not discarding
  only the low outlier while keeping the rest.
- **Background load.** Any other significant CPU/GPU/NPU consumer active during the run
  (another benchmark, a build, background OS services under unusually high load)
  contaminates timing and, for `energy_joules_per_token`, contaminates the power
  integral directly. Verify the system is otherwise idle immediately before each run;
  if the idle check fails, re-run rather than report.
- **First-run page faults / allocator warm-up.** This is exactly what the 3 discarded
  warmup repeats (§7) exist to absorb. A measured repeat (`repeat_index ≥ 0` in the
  written data) that is a clear outlier relative to the rest of its run's distribution,
  especially the first one, is a signal the warmup count was insufficient for that
  engine/config, not a real result — raise the warmup count and re-run rather than
  silently keep the outlier in the reported percentile.
- **Inconsistent or unrecorded governor state.** Every run must have an explicit,
  manifest-recorded CPU/GPU frequency-governor setting. Comparing two runs collected
  under different (or undocumented) governor policies is comparing two different
  experiments, not the same experiment under two conditions.
- **Non-monotonic timing source.** Any implementation using a wall-clock
  (`time.time()`-style) read instead of a monotonic clock (§1) for a duration
  measurement is invalid regardless of the resulting numbers, since a clock adjustment
  during the window could silently corrupt the duration.
- **Config/runtime mismatch for `peak_memory_bytes`.** A `weights` (or any component) row
  computed from the *intended* `quantization`/config value rather than introspected from
  the *actual* runtime tensors (§5.2) is invalid — this is a correctness bug in the
  harness, not a data point to keep with a caveat.

---

## 9. Arithmetic intensity: why GDN decode is bandwidth-bound

This is the quantitative justification for the project's central claim (PLAN.md §2.4):
that optimizing GDN kernels can move prefill (1.38–1.49× measured upstream) but cannot
move decode, because the single-token recurrence has nowhere near enough arithmetic per
byte moved for any amount of compute cleverness to matter. All figures below are a
back-of-envelope roofline-style model, not a live measurement — the assumptions are
stated so the calculation can be checked and does not need to be taken on faith.

**Assumptions:**
- Recurrent state per GDN layer: `n_value_heads × d_k × d_v` elements. **For the selected primary
  checkpoint Qwen3.5-4B that is `32 × 128 × 128 = 524,288`** (ADR 0003, read from `config.json`).
  The fallback 0.8B has 16 value heads → 262,144. (Superseded figure
  for the checkpoint configuration this project targets — PLAN.md's verified facts,
  corroborated independently in `docs/FINDINGS.md` §3 / ADR 0001, `H·d_k·d_v =
  16·128·128 = 262,144`, which matches the 0.8B and the GDN-2 paper's 1.3B config, **not** the 4B.)
- State stored in fp32 (4 bytes/element) → `524,288 × 4 = 2,097,152` bytes = 2 MiB per
  layer, matching the verified "~24MB across 24 GDN layers" figure (`24 × 1 MiB ≈
  24 MiB`).
- One multiply-accumulate (MAC) per state element per token for the dominant
  gated-decay update `S_t = a_t ⊙ S_{t-1} + (write term)`; a MAC counts as 2 FLOPs
  (1 multiply + 1 add), the standard roofline-model convention.
- Batch = 1 (§1): no cross-token or cross-request reuse of a byte once it's fetched.

**Per layer, per token:**

| Quantity | Value |
|---|---:|
| State elements | 524,288 |
| MACs (gated-decay update) | ≈ 524,288 |
| FLOPs (2 × MACs) | ≈ 1,048,576 |
| Bytes read (old state `S_{t-1}`) | 1,048,576 (1 MiB) |
| Bytes written (new state `S_t`) | 1,048,576 (1 MiB) |
| **Total bytes moved** | **4,194,304 (4 MiB)** |
| **Arithmetic intensity** | **1,048,576 / 4,194,304 ≈ 0.25 FLOP/byte** |

(Key/value/gate vectors touched in the same step are `O(d_k + d_v) ≈ 256` floats ≈ 1 KiB
— roughly three orders of magnitude smaller than the 2 MiB of state traffic, and
negligible to this calculation. Accounting for the delta-rule write term as a second,
comparably-sized elementwise update at most doubles the FLOP count to ≈1.05M — moving
the intensity to ≈0.5 FLOP/byte — without changing the conclusion below by more than a
factor of 2, which is irrelevant against the gaps involved.)

**Scaled to the full model (24 GDN layers), per token:**

| Quantity | Value |
|---|---:|
| MACs | 24 × 524,288 = 12,582,912 |
| Bytes moved (read + write) | 24 × 4,194,304 = 100,663,296 (96 MiB) |
| Arithmetic intensity | unchanged, ≈ 0.25 FLOP/byte (both terms scale identically with layer count) |

**0.25 FLOP/byte is far below where compute becomes the bottleneck on any general-purpose
CPU, GPU, or NPU built in the last decade** — modern engines with SIMD/FMA units sustain
compute-to-bandwidth ratios that put their roofline "ridge point" (the arithmetic
intensity above which an engine is compute-bound rather than bandwidth-bound) at several
to a few tens of FLOPs/byte, one to two **orders of magnitude** above 0.25. (This
document does not assert a specific peak-FLOPs figure for Cortex-A720, Immortalis G720,
or the 28.8 TOPS NPU, since none of the source documents pins down the per-cycle FMA
throughput needed to compute an exact ridge point for this SoC — the qualitative gap is
large enough that the conclusion does not depend on that number.) A workload at
0.25 FLOP/byte is bandwidth-bound on essentially any of these engines; **no kernel
rewrite changes which side of the roofline the operation sits on**, because the
operation's shape — one FMA per state element, one state-sized read, one state-sized
write — is fixed by the recurrence's definition, not by how well it's implemented.

**What this bounds, concretely.** At the O6's verified 100GB/s LPDDR5 (PLAN.md's
verified facts; `docs/CLAIM_VERIFICATION.md` §2.2), GDN state traffic alone imposes a
per-token time floor:

```
50,331,648 bytes / 100×10⁹ bytes/s ≈ 5.03×10⁻⁴ s = 0.503 ms/token
→ a ceiling of ≈ 993 decode tokens/sec from state traffic alone
```

This is an **upper bound**, not a prediction of achieved decode throughput — it ignores
weight-streaming traffic, which for batch=1 decode is typically the larger term (every
projection weight matrix must be re-read from DRAM every token, since there is no
batching to amortize a single weight read across multiple tokens' worth of compute). As
a sanity check using the confirmed vendor figure (`docs/CLAIM_VERIFICATION.md` §2.2,
"~30 tokens/sec on Qwen2-1.5B"): a weight-bandwidth-bound estimate for a 1.5B-parameter
model at fp16 (2 bytes/param) gives

```
(1.5×10⁹ params × 2 bytes) / 100×10⁹ bytes/s = 0.03 s/token → ≈ 33.3 tokens/sec
```

— the same order of magnitude as the vendor's cited ~30 tokens/sec, corroborating that
decode on this class of hardware is *already* weight-bandwidth-bound before any GDN
recurrent-state traffic is added on top. That has a forward-looking implication worth
recording: quantizing weights (int8, int4) shrinks the weight-traffic term, but the
24–48 MiB/token of GDN state traffic computed above is comparatively fixed by
`H × d_k × d_v` and the state's own precision — so as weight quantization gets more
aggressive, the *relative* share of decode-time bandwidth spent on GDN state traffic
grows, even though its absolute size does not change. This is exactly why
`docs/FINDINGS.md` §4 flags a bf16/fp16 state variant (halving state traffic) as a still-
open, worthwhile follow-up: it is the one lever in this picture that isn't already being
pulled by weight quantization work.

---

## Appendix: two robustness notes on the arithmetic-intensity argument

Added 2026-08-02 after recomputing against the selected checkpoint (ADR 0003).

### The intensity ratio is invariant across checkpoints

The absolute figures above are 4B-specific, but the *conclusion* does not depend on them. Both the
MAC count and the byte count scale linearly with state size, so the ratio is identical:

| Checkpoint | State/layer | MACs/layer/token | Bytes/layer/token | Intensity |
|---|---:|---:|---:|---:|
| Qwen3.5-4B (32 v-heads, 24 GDN layers) | 524,288 | 524,288 | 4,194,304 | **0.25 FLOP/byte** |
| Qwen3.5-0.8B (16 v-heads, 18 GDN layers) | 262,144 | 262,144 | 2,097,152 | **0.25 FLOP/byte** |

That invariance is worth stating in the write-up: GDN decode is bandwidth-bound as a *structural
property of the rank-1 state update*, not an artefact of one model's dimensions. Each state element
is touched exactly once per token, so intensity is fixed at ~0.25 FLOP/byte regardless of scale.

### State traffic is real but secondary — weights dominate decode

Comparing the two decode traffic sources at 100 GB/s, this corrects an over-claim made earlier in
the project:

| Source | Traffic per token | Share |
|---|---:|---:|
| Weights, fp16 (4B) | 7.5 GiB | ~99% |
| Weights, INT4 (4B) | 1.9 GiB | ~95% |
| GDN state, fp32 (24 layers) | 96 MiB | ~5% at INT4 |
| GDN state, bf16 (24 layers) | 48 MiB | ~2.4% at INT4 |

Cross-check against a published figure: Qwen2-1.5B at fp16 is ~2.8 GiB of weights, giving ≈33
tok/s at 100 GB/s — closely matching the vendor's cited ~30 tok/s for that model on this board.
The weight-bandwidth model is therefore sound.

**Consequence, stated plainly: narrowing the recurrent state to bf16 buys roughly 2–3% of decode
traffic, not a step change.** Earlier project notes described the bf16 state variant as "the one
lever that moves decode" — that was wrong. The dominant decode lever is **weight quantization**
(INT4 weights are a ~4× traffic reduction and take decode from ~12.5 to ~50 tok/s by this model),
which is where KleidiAI's INT4/i8mm GEMV micro-kernels genuinely earn their place.

The bf16 state variant remains worth doing — it also halves the resident state footprint, which
matters for fitting long context alongside weights — but it should be prioritised as a memory
optimisation, not as a decode-throughput one. Bead `ob-8qt.4` has been re-scoped accordingly.
