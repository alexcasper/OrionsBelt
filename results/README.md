# Benchmark results

Committed CSV output plus figures generated from it. Every figure in the README and the
write-up must be reproducible from data in this directory — no hand-assembled numbers.

## Layout

| Directory | Contents | Status |
|---|---|---|
| `raw/` | Device benchmark CSVs (device-bench schema) + ablation sweep CSVs (harness schema) | ✅ Pi5 + 6 ablation configs |
| `raw/ablation/` | Ablation matrix: 6 engine configs × context lengths (synthetic backend, pipeline proof) | ✅ `ob-8qt.5` |
| `manifests/` | One run manifest per CSV — device, governor, clocks, thermals, git SHA | ✅ Pi5 manifest |
| `figures/` | Plots from `bench/plots.py`: memory decomposition, throughput curves, device comparison | ✅ Memory decomposition + ablation table |
| `performix/` | Arm Performix standardized reports | ⏳ Pending hardware |

## Schema note

Device benchmark CSVs (`raw/pi5-r5.csv`) use the device-bench schema
(model, kernel, dispatch_path, repeats, p50_us, gib_per_s_p50). Harness sweep
CSVs (`raw/ablation/*.csv`) use the frozen tidy/long schema
(`docs/RESULTS_SCHEMA.md` / `bench/schema.py`). Both are validated — the device
bench by its own format, the harness CSVs by `schema.validate_rows()`.

## Reproducibility

Every result is traceable to a manifest-backed run (PLAN.md §9). To regenerate
figures: `python3 bench/plots.py --memory`. To regenerate the ablation table:
`python3 scripts/run_ablation.py --context 4096`.
