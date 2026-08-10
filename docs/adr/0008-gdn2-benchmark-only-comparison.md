# ADR 0008: GDN-2 comparison via microbenchmark only (option a)

- **Status:** Accepted — option (b) subsequently attempted (see Update below)
- **Date:** 2026-08-06
- **Bead:** `ob-9ke`
- **Deciders:** Claude (agent, t3 node)

## Context

ADR 0001 proposed testing GDN-2's decoupled erase/write gating hypothesis. Two paths:
(a) microbenchmark GDN-2 vs GDN-1 gating on target hardware; (b) swap GDN-2 layers
into a Qwen3.5 checkpoint and evaluate on RULER. ADR 0004's T-2 trigger (Aug 12)
pre-committed to cutting option (b) if schedule is tight. ADR 0007 committed to the
Edge AI track. The gdn2_gated_scan kernel is already implemented and benchmarked
alongside gdn_gated_scan in Phase 1 data.

## Decision

**Option (a): benchmark-only comparison.** The microbenchmark data already exists
in the Phase 1 CSVs — both `gdn_gated_scan` and `gdn2_gated_scan` were measured at
identical shapes and conditions on the RK3588. No additional runs needed; the
analysis is a writing task.

## Alternatives considered

| Option | Why not |
|---|---|
| Option (b): layer swap + RULER | Needs adaptation compute we don't have (R5). ADR 0004 T-2 trigger would cut this anyway. |
| Defer entirely | The data is already captured; writing it up is nearly free and strengthens the submission's Potential Impact section. |

## Consequences

**Accepted costs.** No retrieval-quality evaluation (RULER). The comparison is
architectural/throughput only, not accuracy.

**Follow-on work.** ob-82b (write the microbenchmark comparison), ob-7b5 (research note).

**Reversal cost.** Low — if schedule opens before T-2 (Aug 12), option (b) can be
attempted. The microbenchmark remains valid regardless.

---

## Update (2026-08-09): Option (b) attempted

Schedule opened earlier than expected and the GDN-2 layer swap was completed
(PR #204, FINDINGS §40). Layer 0 of Qwen3.5-0.8B was swapped to GDN-2 with
isolated MSE distillation, and a 10-prompt RULER multi-key retrieval evaluation
was run. Result: CE recovery plateaus at ~20% and RULER accuracy is 20% (vs
GDN-1's 30%, at the 20% random baseline) — the model is too under-adapted for
a fair architectural comparison. This is an honest negative result; the
microbenchmark comparison from option (a) remains the stronger evidence.
The original decision's caution was vindicated: the swap confirmed that
isolated distillation is insufficient without full fine-tuning.
