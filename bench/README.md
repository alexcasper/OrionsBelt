# Benchmark harness

Measurement apparatus for the project. Built deliberately **hardware-independent**
(PLAN.md section 1) so it is finished before any board arrives — the moment hardware
exists we should be profiling within hours, not still writing a runner.

| File | Purpose | Owning bead |
|---|---|---|
| `harness.py` | Runner CLI: context sweep (4K/32K/128K/262K), warmup, repeats, percentiles, Backend ABC | `ob-ljh` |
| `metrics.py` | Nearest-rank percentiles (p50/p95), spread, summarize | `ob-ljh` |
| `memory.py` | Three-way memory decomposition (weights, KV cache, recurrent state) + cross-check | `ob-vfp` |
| `manifest.py` | Provenance capture — device, kernel, SDK versions, governor, clocks, thermals, git SHA | `ob-u37` |
| `plots.py` | Memory decomposition chart, throughput curves, device fleet comparison | `ob-9y8` |
| `comparison_table.py` | Markdown comparison table from harness CSVs (ablation grid) | `ob-8qt.5` |
| `prompts/` | Committed prompt corpus: needle-in-haystack + RULER multi-key (4K–262K) | `ob-del` |

## Two rules that are not negotiable

1. **Every run emits a manifest.** Per PLAN.md section 9, a number without provenance is
   not a result. On passively-cooled edge hardware, thermal state alone can move
   throughput enough to invalidate a comparison.
2. **Memory must be attributed three ways** — weights, full-attention KV cache, and GDN
   recurrent state — separately. This split *is* the project's central claim: one
   component grows with context while the other stays flat. Without it, the argument is
   asserted rather than demonstrated.

Report percentiles and repeat counts, never a single best run.
