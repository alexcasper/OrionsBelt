# Results schema contract

**Bead:** `ob-q9i` (`t-results-schema`) · **Status:** Frozen 2026-08-02 · **Executable form:** [`bench/schema.py`](../bench/schema.py)

This is the one schema every benchmark CSV in `results/raw/` must satisfy. It exists
because the harness (`bench/harness.py`), the plotting code (`bench/plots.py`), and the
final comparison table (`ob-9y8`, `t-results-table`) all bind to these column names, types,
and enum values. **Freezing it now, before any run is collected, is the entire point of
this bead** — see docs/archive/PLAN.md §2.4 and §5 (M0 exit criterion: "results schema frozen") and the
"changing this schema" section at the bottom of this file.

If you are about to add a metric, rename a column, or change an enum value: stop and read
that section first.

---

## 1. Format: tidy/long, not wide

Each row is exactly **one measurement**: one metric, for one phase, for one repeat, of one
run. Rows are never averaged or pivoted before landing in `results/raw/`.

**Why tidy and not wide:** the load-bearing claim of this project (docs/archive/PLAN.md §2.4) is that
prefill and decode behave completely differently under GDN kernel optimization — prefill
speeds up 1.38–1.49×, decode stays flat because the single-token recurrence is
memory-bandwidth-bound — and that peak memory splits three ways (model weights /
full-attention KV cache / GDN recurrent state) with different scaling behavior against
context length. A wide format (one column per metric, one row per run) forces a premature
decision about which things are comparable columns, and it is exactly the shape that lets
someone accidentally average `decode_tokens_per_sec` and `prefill_tokens_per_sec` into one
"tokens/sec" number — the one mistake this project must not make. Tidy format keeps
`phase` and `metric_component` as explicit *values in a column*, not baked into column
names, so:

- prefill and decode are never collapsed — they are different values of `phase`, and any
  aggregation across them has to be a deliberate `groupby`, not an accidental default;
- the three-way memory split is one metric (`peak_memory_bytes`) crossed with one
  dimension (`metric_component` ∈ `weights | kv_cache | recurrent_state`), not three
  separate ad-hoc column names that plotting code has to know about in advance;
- adding a new metric or a new per-layer-class breakdown later is a new **row shape**
  (more values in an existing column, or one new optional column), never a schema
  migration of existing columns — see §6.

Downstream code (`bench/plots.py`, the results table) is expected to `groupby` on
`phase`, `metric_name`, and `metric_component` and compute percentiles across
`repeat_index` — never to read a single "best" row (docs/archive/PLAN.md §9).

## 2. File layout and naming

- One CSV per benchmark run, at `results/raw/<run_id>.csv`. A "run" is one invocation of
  the harness for one fixed (device, engine assignment, model checkpoint, quantization
  config, context length) tuple; it contains every phase, every metric, and every repeat
  collected during that invocation — so one file typically has dozens of rows, not one.
- `run_id` convention (documentary, not enforced by the validator, since the manifest is
  the authority on provenance): `<device>_<yyyymmddTHHMMSSZ>_<short_git_sha>`, e.g.
  `o6_20260810T143000Z_a1b2c3d`. This keeps filenames sortable by time and greppable by
  device without opening the file.
- Every CSV has exactly one companion manifest at `results/manifests/<run_id>.json`
  (owned by `ob-u37` / `t-manifest`). `manifest_ref` in the CSV points to it. Per
  docs/archive/PLAN.md §9: **a CSV without its manifest is not a result.**
- `results/figures/` and `results/performix/` are downstream artifacts generated from
  `results/raw/`, never hand-edited (see `results/README.md`).

## 3. Column reference

All columns are required unless marked **optional**. "Additive" below means the column
was added after the initial freeze without breaking existing data — see §6.

| Column | Type | Units / allowed values | Why it exists |
|---|---|---|---|
| `run_id` | string | free-form; see naming convention above | Groups every row from one harness invocation; joins a CSV to its manifest and lets multiple metrics/phases/repeats share identity without repeating a manifest per row. |
| `timestamp` | string | ISO 8601 UTC, e.g. `2026-08-10T14:30:00Z` | Wall-clock time the *specific measurement in this row* was recorded. Distinct from the run's start time (that's in the manifest) because a single run can span minutes across context lengths and phases. |
| `git_sha` | string | 7–40 lowercase hex chars | Commit of the harness/model/optimization code that produced this row. Any number without a traceable commit cannot be reproduced or trusted (docs/archive/PLAN.md §9). |
| `manifest_ref` | string | path relative to repo root, e.g. `results/manifests/o6_20260810T143000Z_a1b2c3d.json` | Points at the provenance record (device, kernel, SDK versions, governor, clocks, thermal state). The row's numbers are meaningless without it. |
| `device` | enum: `o6`, `generic_aarch64`, `x86_reference` | — | Which physical/hedge target produced this row. `x86_reference` is reserved for the correctness-oracle target (`t-x86-ref`), never for a performance claim. |
| `engine_gdn` | enum: `npu`, `gpu_vulkan`, `gpu_opencl`, `cpu`, `cuda_reference` | — | Which execution engine ran the **Gated DeltaNet (linear-attention) layers** for this row. Separated from `engine_full_attention` because per-layer-class engine assignment (docs/archive/PLAN.md §3, §6) is the central design question of E6 — GDN's sequential recurrent scan and full attention's dense matmuls are expected to land on *different* engines, and a single "engine" column could not express that. |
| `engine_full_attention` | enum: same as `engine_gdn` | — | Which execution engine ran the **full-attention layers** (8 of 32 in the dense checkpoint, docs/archive/PLAN.md §3) for this row. Kept independent from `engine_gdn` for the same reason. |
| `model_checkpoint` | string | HF repo id + revision, e.g. `Qwen/Qwen3.5-4B@a1b2c3d` | Identifies which checkpoint and exact revision was measured; the family spans 0.8B–4B candidates (docs/archive/PLAN.md §2.3) and revisions can change layer shapes. |
| `quantization` | string | free-form short code, e.g. `fp16`, `int8_w8a8`, `int4_npu_fp16_gate` | Identifies the quantization policy in force. Left free-form (not a closed enum) because the per-layer quantization policy (which layers must stay FP16 — recurrent state and gates are the obvious candidates, docs/archive/PLAN.md §4/E4) is still being decided by a separate bead; the schema must not block that work by pre-freezing its vocabulary. |
| `context_length` | integer | tokens, > 0 | Canonical sweep points are `4096`, `32768`, `131072`, `262144` (docs/archive/PLAN.md §5/E5), but the column accepts any positive integer so intermediate or exploratory points are not schema violations. |
| `phase` | enum: `prefill`, `decode` | — | **The single most important column in this schema.** Optimizing GDN kernels speeds up prefill 1.38–1.49× and leaves decode flat (docs/archive/PLAN.md §2.4) because decode is a memory-bandwidth-bound single-token recurrence. Averaging prefill and decode into one throughput number would erase the exact distinction the project exists to demonstrate — so `phase` is a required, explicit value on every throughput/latency row, never implicit in a column name. |
| `metric_name` | enum, see §4 | — | Which measurement this row reports. |
| `metric_component` | enum: `weights`, `kv_cache`, `recurrent_state` | **required if** `metric_name == peak_memory_bytes`; **must be empty otherwise** | Expresses the three-way memory attribution (docs/archive/PLAN.md §2.4, `bench/README.md` rule 2): model weights are flat baseline cost, `kv_cache` grows linearly with context (full-attention layers only), `recurrent_state` stays O(1) per token (GDN layers only). This is the load-bearing measurement — without this column as a first-class dimension, the central claim can only be asserted, not shown. |
| `value` | float | metric-dependent, see §4; always ≥ 0 | The measured number. |
| `unit` | enum: `tokens_per_sec`, `seconds`, `bytes`, `joules_per_token` | — | Explicit unit per row (rather than assumed from `metric_name` alone) so a CSV is self-describing even in isolation; the validator additionally checks it matches the metric's canonical unit. |
| `repeat_index` | integer | ≥ 0, zero-based | Which repeat within the run this row is. docs/archive/PLAN.md §9: "report percentiles and repeat counts, never a single best run" — a schema with no repeat dimension could not honor that rule. |
| `repeat_count` | integer | ≥ 1, `repeat_index < repeat_count` | Total repeats planned for this (run, metric, phase, component) combination. Makes each row self-describing for percentile computation without cross-referencing the manifest. |
| `layer_class` **(optional)** | enum: `gdn`, `full_attention`, `ffn`, `all` | default `all` | Reserved for future per-layer-class breakdowns (e.g. per-layer-class latency profiling in E6) without a schema migration. Rows produced by this bead's initial harness use `all`; a later bead may emit `gdn`/`full_attention`/`ffn` rows for finer-grained metrics. Additive — see §6. |
| `notes` **(optional)** | string | free text, default empty | Escape hatch for anomalies observed at collection time (e.g. "thermal throttling observed mid-run", "governor forced to performance"). Never load-bearing for a claim by itself — a real anomaly belongs in the manifest too — but useful for a human skimming `results/raw/`. |

## 4. Metric vocabulary

Every `metric_name` has exactly one canonical `unit` and a restricted set of valid
`phase` values. The validator in `bench/schema.py` enforces both.

| `metric_name` | Canonical `unit` | Valid `phase` | `metric_component` | Meaning |
|---|---|---|---|---|
| `prefill_tokens_per_sec` | `tokens_per_sec` | `prefill` only | must be empty | Prompt-processing throughput. Where GDN kernel optimization is expected to show the 1.38–1.49× win (docs/archive/PLAN.md §2.4), and where the win should *grow with context length*. |
| `decode_tokens_per_sec` | `tokens_per_sec` | `decode` only | must be empty | Per-token autoregressive generation throughput. Expected to stay flat across kernel optimizations — the single-token DeltaNet recurrence is memory-bandwidth-bound. Predicting and reporting the flat result is itself a finding, not a null result to hide. |
| `ttft_seconds` | `seconds` | `prefill` only | must be empty | Time to first token — wall-clock latency from request submission to the first generated token. The user-facing face of prefill throughput. |
| `peak_memory_bytes` | `bytes` | `prefill` or `decode` | **required**, one of `weights`/`kv_cache`/`recurrent_state` | Peak resident memory for the given component, measured within the given phase. `weights` is expected flat across context length; `kv_cache` is expected to grow linearly with context (full-attention layers only); `recurrent_state` is expected to stay O(1) per token (GDN layers only) — this contrast is the demonstration of the architecture's central advantage. |
| `energy_joules_per_token` | `joules_per_token` | `prefill` or `decode` | must be empty | Energy cost per token generated (decode) or per token processed (prefill), from on-board power sampling. Secondary to the throughput/memory story but relevant to "edge" framing. |

Rows with a `metric_name` not in this table, a `unit` that does not match the table, a
`phase` outside the metric's valid set, or a `metric_component` that violates the
required/must-be-empty rule are all schema violations and `validate_row()` raises on
them.

## 5. Worked example

Four rows from one (hypothetical) run: O6 board, Qwen3.5-4B, INT8 weights with the GDN
gate kept FP16, 32K context, GDN layers on the GPU Vulkan scan kernel, full-attention
layers on the NPU, first of five repeats.

| run_id | timestamp | git_sha | device | engine_gdn | engine_full_attention | context_length | phase | metric_name | metric_component | value | unit | repeat_index | repeat_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `o6_20260810T143000Z_a1b2c3d` | 2026-08-10T14:30:05Z | a1b2c3d4 | `o6` | `gpu_vulkan` | `npu` | 32768 | `prefill` | `prefill_tokens_per_sec` | *(empty)* | 812.4 | `tokens_per_sec` | 0 | 5 |
| `o6_20260810T143000Z_a1b2c3d` | 2026-08-10T14:30:05Z | a1b2c3d4 | `o6` | `gpu_vulkan` | `npu` | 32768 | `prefill` | `ttft_seconds` | *(empty)* | 0.183 | `seconds` | 0 | 5 |
| `o6_20260810T143000Z_a1b2c3d` | 2026-08-10T14:31:40Z | a1b2c3d4 | `o6` | `gpu_vulkan` | `npu` | 32768 | `decode` | `decode_tokens_per_sec` | *(empty)* | 14.2 | `tokens_per_sec` | 0 | 5 |
| `o6_20260810T143000Z_a1b2c3d` | 2026-08-10T14:31:40Z | a1b2c3d4 | `o6` | `gpu_vulkan` | `npu` | 32768 | `decode` | `peak_memory_bytes` | `recurrent_state` | 41943040 | `bytes` | 0 | 5 |

(`manifest_ref`, `model_checkpoint`, `quantization`, `layer_class`, `notes` omitted from
the table above for width; every real row has them — `manifest_ref` would be
`results/manifests/o6_20260810T143000Z_a1b2c3d.json`, `model_checkpoint` would be
`Qwen/Qwen3.5-4B@a1b2c3d`, `quantization` would be `int8_w8a8_gdn_gate_fp16`,
`layer_class` would be `all`, `notes` empty.)

A full context sweep at 4K/32K/128K/262K, each with several metrics and 5 repeats, will
produce on the order of a few hundred rows per (device, engine assignment, quantization)
combination — this is expected and is why the format is tidy, not wide.

## 6. Changing this schema

This schema is **frozen** as of 2026-08-02. It is depended on by the harness, the
plotting code, and the final comparison table — changing it after data collection starts
invalidates already-collected CSVs (docs/archive/PLAN.md §2.4, §5).

**Allowed without invalidating existing data:**
- Adding a new **optional** column (must have a documented default so old rows remain
  valid when read back).
- Adding a new enum *value* to an existing column (e.g. a new `device`, a new `engine`,
  a new `metric_name` with its own unit/phase rules) — existing rows are unaffected.
- Adding a new canonical `context_length` sweep point — the column already accepts any
  positive integer.

**Not allowed without a new schema version and an explicit migration:**
- Renaming or removing any column in §3.
- Renaming or removing any enum value already in use by committed data.
- Changing a metric's canonical `unit` or valid `phase` set in §4.
- Changing the file layout in §2 (one CSV per run, tidy rows) to a wide format.

If a change in the second category is truly necessary, it must go through an ADR in
`docs/adr/` explaining what existing data becomes invalid and why the change is worth
that cost, per docs/archive/PLAN.md §9 ("Decisions become ADRs"). `docs/RESULTS_SCHEMA.md` and
`bench/schema.py` are changed together, in the same commit — they are two views of one
contract, and letting them drift apart defeats the purpose of freezing this at all.
