# Benchmark harness

Measurement apparatus for the project. Built deliberately **hardware-independent**
(docs/archive/PLAN.md section 1) so it is finished before any board arrives — the moment hardware
exists we should be profiling within hours, not still writing a runner.

| File | Purpose | Owning bead |
|---|---|---|
| `harness.py` | Runner CLI: context sweep (4K/32K/128K/262K), warmup, repeats, percentiles | `t-harness-core` |
| `metrics.py` | Decode tok/s, prefill throughput, TTFT, memory accounting, energy/token | `t-metrics-spec`, `t-harness-mem` |
| `manifest.py` | Provenance capture — device, kernel, SDK versions, governor, clocks, thermals, git SHA | `t-manifest` |
| `plots.py` | Scaling curves, stacked memory decomposition, comparison matrix | `t-plots` |

## Two rules that are not negotiable

1. **Every run emits a manifest.** Per docs/archive/PLAN.md section 9, a number without provenance is
   not a result. On passively-cooled edge hardware, thermal state alone can move
   throughput enough to invalidate a comparison.
2. **Memory must be attributed three ways** — weights, full-attention KV cache, and GDN
   recurrent state — separately. This split *is* the project's central claim: one
   component grows with context while the other stays flat. Without it, the argument is
   asserted rather than demonstrated.

Report percentiles and repeat counts, never a single best run.
