# ADR 0008: GDN-2 comparison via microbenchmark only (option a)

- **Status:** Accepted
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
