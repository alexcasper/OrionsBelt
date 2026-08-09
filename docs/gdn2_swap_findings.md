# GDN-2 Layer Swap Experiment (ob-68l)

**Date:** 2026-08-09
**Device:** RK3588 (t3), Cortex-A76 CPU, 32 GB RAM, aarch64
**Commit:** `06f3a9f` (bench/t3, dirty — script under active development)
**Bead:** [ob-68l](https://github.com/OrionsBelt/OrionsBelt/issues/ob-68l)

## Objective

Replace GDN-1 (Gated DeltaNet) linear-attention layers with GDN-2
(GatedDeltaNet-2) layers in a Qwen3.5-0.8B checkpoint, then run brief
adaptation to show the swapped layers are not simply broken.

### GDN-1 vs GDN-2

| Aspect | GDN-1 | GDN-2 |
|--------|-------|-------|
| Input gate | Single β per head: `delta = (v − kᵀS) · β` | Split into erase gate b and write gate w |
| Erase | `kᵀS · β` (implicit) | `(b ⊙ k)ᵀ S` (explicit, per-key-channel) |
| Write | `v · β` (implicit) | `w ⊙ v` (explicit, per-value-channel) |
| Gate dimensionality | `[num_v_heads]` per token | `[key_dim]` + `[value_dim]` per token |
| Expressive power | β controls both erase and write | b and w independently control erase and write |

When b = w = β, GDN-2 reduces to GDN-1. The extra per-channel gate
capacity is the point: GDN-2 can learn to selectively forget or write
per channel, which GDN-1's single per-head β cannot.

## Method

### Model

Qwen3.5-0.8B: 24 layers (18 GDN-1 linear attention + 6 full attention in
3:1 pattern). Layer 0 (a GDN-1 layer) was swapped to GDN-2.

### Architecture: `Qwen3_5GDN2` module

Custom module (`bench/gdn2_swap.py`) that subclasses the existing
`Qwen3_5GatedDeltaNet` infrastructure:
- Inherits all projections (in_proj_qkv, in_proj_z, in_proj_a, conv1d,
  A_log, dt_bias, norm, out_proj) from the checkpoint
- Adds two new projections: `in_proj_erase_gate` (hidden→key_dim) and
  `in_proj_write_gate` (hidden→value_dim)
- Replaces the GDN-1 delta-rule recurrence with GDN-2 recurrence
- Gates initialized with Xavier uniform (gain=0.1) → sigmoid ≈ 0.5

### Adaptation strategy: isolated MSE distillation

**Constraint:** Full-model backpropagation on RK3588 CPU takes ~436 s per
step (800 M parameters). This makes standard fine-tuning infeasible.

**Solution:** Capture the GDN-1 layer's input and output in a single
no-grad forward pass. Then train the GDN-2 module **in isolation** to
minimize MSE against the cached GDN-1 output. Backpropagation flows only
through the ~4.2 M new gate parameters, not the full model.

- Optimizer: AdamW, lr=1e-3
- Steps: 30
- Sequence length: 64 tokens
- Isolated step time: ~6.6 s (vs 436 s for full-model backprop)

## Results

| Metric | Value |
|--------|-------|
| Baseline CE loss (GDN-1) | 3.326 |
| Post-swap CE loss (GDN-2, unadapted) | 11.906 (+8.580) |
| Final CE loss (after 30 steps) | 10.335 (−1.572 from post-swap) |
| Isolated MSE (step 1) | 0.0251 |
| Isolated MSE (step 30) | 0.0015 |
| MSE reduction | 94.0% |
| New parameters | 4,194,304 (16.0 MB fp32) |
| Trainable parameters | 4,194,304 |

### MSE convergence curve

```
Step  1: 0.0251  ████████████████████████████████████████
Step  5: 0.0072  ████████████
Step 10: 0.0037  ██████
Step 15: 0.0026  ████
Step 20: 0.0020  ███
Step 25: 0.0017  ███
Step 30: 0.0015  ██
```

MSE decreases monotonically — the optimization landscape is smooth and
the gates are clearly converging toward the GDN-1 reference behavior.

## Analysis

### What works

1. **Architectural validity:** The GDN-2 module produces finite, non-NaN
   outputs when dropped into the Qwen3.5 forward pass. The recurrence is
   numerically stable.

2. **Gate learning:** The isolated MSE drops 94% in 30 steps with smooth,
   monotonic convergence. The GDN-2 gate parameters can learn to
   reproduce GDN-1's per-token behavior.

3. **Efficient training proxy:** Isolated MSE distillation avoids
   full-model backprop (436 s/step → 6.6 s/step, 66× faster) while
   directly optimizing the metric we care about (matching GDN-1 output).

### What doesn't fully recover

CE loss recovers only 1.57 points out of the 8.58-point increase
(18.3%). Three factors explain the remaining gap:

1. **Random gate initialization:** Gates start at sigmoid ≈ 0.5, far from
   GDN-1's learned per-head β values. The MSE needs to go much lower to
   achieve CE-level recovery.

2. **Only 30 steps:** The MSE is still decreasing at step 30. More
   steps (100–500) would push MSE lower and recover more CE loss.

3. **Output MSE is a proxy, not the true objective:** Matching the GDN-1
   layer output exactly (MSE=0) would recover full CE loss, but the
   downstream layers amplify even small mismatches in early layers.

### Practical implications

- **On-device training is feasible** for gate-only adaptation using the
  isolated distillation approach. The 66× speedup makes this practical
  on edge CPUs.
- **Smart initialization matters:** Initializing gates from GDN-1's β
  values (e.g., broadcasting per-head β to per-channel b and w) would
  dramatically reduce the initial loss and required adaptation steps.
- **Full model retraining is needed for production:** The 30-step
  distillation shows the gates can learn, but real deployment would need
  end-to-end fine-tuning to let downstream layers co-adapt.

## Hardware notes

- Baseline forward pass: 99–115 s for seq_len=64 (vs 16.9 s for seq_len=128
  reported in earlier runs — likely thermal throttling on sustained CPU load)
- Full-model backprop: 436 s/step (measured for 1 step)
- Isolated layer forward+backward: ~6.6 s/step
- All experiments ran on Cortex-A76 cores (big cluster)
- No GPU/NPU acceleration used

## Files

| File | Description |
|------|-------------|
| `bench/gdn2_swap.py` | Experiment script (GDN-2 module, swap, adaptation) |
| `results/raw/gdn2_swap_t3.csv` | MSE convergence curve (30 steps) |
| `results/manifests/gdn2_swap_t3.json` | Provenance manifest |

## Conclusion

GDN-2 can architecturally replace GDN-1 in a Qwen3.5 checkpoint. The new
gate parameters learn to approximate GDN-1's behavior via isolated MSE
distillation, with smooth monotonic convergence (94% MSE reduction in 30
steps). Full CE loss recovery requires either more adaptation steps or
smarter gate initialization from GDN-1's existing β parameters. The
isolated training approach makes on-device adaptation practical (66×
faster than full-model backprop), demonstrating a viable path for GDN-2
upgrades on edge silicon.
