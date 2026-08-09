# RULER Multi-Key Retrieval: GDN-1 vs GDN-2 (ob-zak)

**Date:** 2026-08-09
**Device:** RK3588 (t3), Cortex-A76 CPU, aarch64
**Commit:** `1743c3e` (bench/t3)
**Bead:** ob-zak

## Objective

Test whether decoupled erase/write gating (GDN-2) improves multi-key
retrieval at edge-appropriate scale, compared to GDN-1's single input
gate. This is the direct test of the paper's retrieval hypothesis.

## Method

### Evaluation: Log-likelihood scoring

For each prompt with N key-value pairs, one key is queried. Each
candidate answer (correct + N−1 distractors) is scored by computing
`log P(answer | prompt)` via teacher-forced forward pass. Accuracy is
defined as the fraction of prompts where the correct answer has the
highest total log-probability among all candidates.

This avoids autoregressive generation (infeasible on CPU) while still
measuring whether the model has retrieved the correct key-value
association.

### Prompts

- 10 prompts, 5 keys each, 256-token context (~290 actual tokens)
- Deterministic seeds (100–109), reproducible via `bench/corpus.py`
- Random baseline: 20% (1 in 5 candidates)

### Models

1. **GDN-1 baseline:** Unmodified Qwen3.5-0.8B
2. **GDN-2 (30-step adapted):** Layer 0 swapped to GDN-2, 30-step
   isolated MSE distillation (see ob-68l findings for details)

## Results

| Metric | GDN-1 | GDN-2 (adapted) |
|--------|-------|-----------------|
| Accuracy | 30% (3/10) | 10% (1/10) |
| Random baseline | 20% | 20% |
| Avg correct log-prob | −14.6 | −71.2 |
| Avg per-token log-prob | −2.0 | −9.9 |
| Forward pass time | ~32s | ~35s |

### Per-prompt comparison

| Seed | Query Key | GDN-1 hit | GDN-2 hit | GDN-1 lp | GDN-2 lp |
|------|-----------|-----------|-----------|----------|----------|
| 100 | config_beta_0001 | ✗ | ✗ | −15.0 | −79.5 |
| 101 | config_beta_0001 | ✗ | ✗ | −12.6 | −88.2 |
| 102 | config_delta_0003 | ✗ | ✗ | −15.4 | −54.6 |
| 103 | config_beta_0001 | ✓ | ✓ | −10.1 | −59.8 |
| 104 | config_delta_0003 | ✗ | ✗ | −18.3 | −74.7 |
| 105 | config_gamma_0002 | ✗ | ✗ | −16.2 | −71.6 |
| 106 | config_gamma_0002 | ✓ | ✗ | −13.9 | −75.2 |
| 107 | config_beta_0001 | ✗ | ✗ | −13.1 | −72.5 |
| 108 | config_beta_0001 | ✓ | ✗ | −12.5 | −65.2 |
| 109 | config_delta_0003 | ✗ | ✗ | −19.1 | −70.9 |

## Analysis

### Key findings

1. **GDN-2 at 30-step adaptation is insufficient for retrieval.** Accuracy
   drops from 30% (GDN-1) to 10% (GDN-2), which is **below the 20% random
   baseline**. The model is too degraded to test the architectural
   hypothesis.

2. **Log-prob degradation is severe and consistent.** GDN-2's average
   correct-answer log-prob (−71.2) is 5× worse than GDN-1 (−14.6). Every
   prompt shows degradation; no prompt improves under GDN-2.

3. **The one GDN-2 hit was the easiest prompt.** Seed 103
   (`emerald-engine-856`) was correctly retrieved by both models — it
   appears to be the prompt where the correct answer had the highest
   initial signal.

### Why GDN-2 fails here

The 30-step isolated MSE distillation (from ob-68l) reduced the layer-0
output MSE by 94% but left CE loss at 10.3 (vs 3.3 baseline). This means
the GDN-2 layer is still producing significantly different outputs than
GDN-1, and the downstream layers amplify this mismatch. For retrieval
tasks that require precise information routing through all 24 layers,
this degradation is fatal.

### Statistical caveats

- **Small sample:** 10 prompts × 5 candidates is a pilot-scale
  evaluation. The 30% vs 10% difference is suggestive but not
  statistically significant (Fisher's exact p ≈ 0.26).
- **Log-prob comparison is robust:** The 5× log-prob degradation is
  consistent across all 10 prompts and is the stronger signal.
- **Only 1 layer swapped:** Swapping layer 0 tests the mechanism but
  doesn't represent a full GDN-2 model. Multiple-layer swaps with
  extensive adaptation would be needed for a fair test.

### What this means for the GDN-2 hypothesis

This experiment **does not support or refute** the hypothesis that
decoupled erase/write gating improves multi-key retrieval. The GDN-2
model is too under-adapted to serve as a fair comparison. The
hypothesis remains untested pending:

1. **Full model fine-tuning** (not just 30 isolated steps on 1 layer)
2. ~~**Smart gate initialization** from GDN-1's β values~~ — **done**
   (ob-t3b.9, commit `8faec1e`): smart init improved CE recovery from
   17.1% to 19.9% (+2.8 pp) but both strategies converge to the same
   isolated MSE. The improvement is well below the threshold that would
   change the RULER outcome — CE loss remains ~10 vs 2.9 baseline, so
   the model is still too degraded for a fair retrieval comparison.
3. **GPU acceleration** to make full-model backprop feasible (436s/step
   on this CPU)

## Hardware notes

- Forward pass: ~32s (GDN-1), ~35s (GDN-2) for 290 tokens
- Adaptation: 58.6s for 30 isolated steps
- Total evaluation time: ~29 min per model variant
- All on Cortex-A76 CPU, bf16 precision

## Files

| File | Description |
|------|-------------|
| `bench/gdn2_ruler.py` | RULER evaluation script |
| `results/raw/ruler_gdn1_t3.csv` | GDN-1 retrieval results |
| `results/raw/ruler_gdn2_t3.csv` | GDN-2 retrieval results |
| `results/manifests/ruler_gdn1_t3.json` | GDN-1 manifest |
| `results/manifests/ruler_gdn2_t3.json` | GDN-2 manifest |

## Conclusion

GDN-2 with 30-step isolated adaptation achieves 10% retrieval accuracy
(below the 20% random baseline) vs GDN-1's 30%. The 5× log-prob
degradation is consistent across all prompts. This negative result
reflects insufficient adaptation rather than a flaw in the GDN-2
architecture: the partially trained model cannot be used to test the
retrieval hypothesis. Smart gate initialization (ob-t3b.9) was completed
after this evaluation — it improved CE recovery from 17.1% to 19.9%,
insufficient to change the RULER outcome. Full end-to-end fine-tuning
(436 s/step on this CPU) remains the only path to a meaningful
architectural comparison.
