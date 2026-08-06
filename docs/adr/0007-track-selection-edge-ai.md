# ADR 0007: Commit to Edge AI track on RK3588

- **Status:** Accepted
- **Date:** 2026-08-06
- **Bead:** `ob-imb`
- **Deciders:** t4 agent (device maintainer), ratified 3 days ahead of the Aug 9 hard date because the facts are settled.

## Context

The PLAN.md two-track strategy (§1) ran Physical AI (O6) and Edge AI (hedge) in parallel from
day one, with a hard go/no-go on Aug 9. Today is Aug 6 (T-8 to the Aug 14 deadline). The facts:

1. **The O6 board has not arrived.** `ob-axq` (source the board) remains `OPEN`, unclaimed, last
   updated Aug 2 — four days ago with no movement. No teammate has reported receiving it.

2. **Even if the O6 arrived today, the schedule is prohibitive.** The remaining work for a Physical
   AI submission on the O6 is: board boot + SSH (`ob-iae`, est. 0.5–1 day), CIX SDK on-device
   setup (1 day), per-layer latency profile (`ob-c9k`, 1–2 days), NPU subgraph export + quantization
   (`ob-onz`, 2+ days), engine-boundary measurement (`ob-t3b.3`, 1 day), optimization + ablation
   (`ob-rqd`, 2+ days), then write-up + demo (2 days). That is 9–11 days of work with 8 days
   remaining, and every step depends on the one before it.

3. **The Edge AI hedge is not a consolation prize — it has real results.** On the RK3588 we have:
   - Device-fleet bandwidth-scaling benchmarks (`ob-8ms.3`): Pi 5, RK3588 (t3+t4), Jetson Nano
   - Three hand-written SVE/i8mm GDN kernels, numerically verified (`ob-8qt.1`)
   - Mixed-precision recurrent-state kernel measurements showing fp16 wins and software bf16
     penalties on A76 (`ob-8qt.4`)
   - Cross-vendor NPU operator-coverage probe: CIX NOE rejects Scan/Loop; Rockchip RKNN accepts
     Scan but rejects runtime Loop — both vendors reject variable-length recurrence (`ob-t3b.5`)
   - GDN layer structure audit from actual modeling code (`ob-37v`): 3:1 GDN:full-attention ratio,
     48 MB flat recurrent state vs 8 GB KV cache at 262K context

4. **Edge AI is a legitimate prize category.** The three Devpost tracks are Physical AI, Cloud AI,
   and Edge AI (verified, `docs/CLAIM_VERIFICATION.md` §1.1). The RK3588 is an edge device with a
   heterogeneous CPU+GPU+NPU architecture — a strong Edge AI story. And the O6, if it ever arrives,
   is also an edge device, so the framing does not change.

## Decision

**Commit to the Edge AI track.** Primary device: RK3588 (what we have). Stop spending time on
O6-specific work. If the O6 arrives before the deadline, it becomes additional data, not a
last-minute track switch.

## Alternatives considered

| Option | Why not |
|---|---|
| Wait until Aug 9 to decide | The facts will not change: the board has shown no sign of arriving in four days. Waiting costs 3 days of write-up time for zero information gain. |
| Physical AI on O6 (if it arrives this week) | 9–11 days of serial-dependent work with 8 days remaining. Even in the best case, the submission would be incomplete. The descope ladder (ADR 0004) anticipated this as the T-3 trigger. |
| Cloud AI | Wrong framing: the project is about edge inference on Arm silicon, not cloud deployment. No cloud hardware, no cloud story. |
| Dual submission (both tracks) | Devpost allows one track per submission. Splitting effort across two submissions halves the quality of both. |

## Consequences

**Accepted costs.**

- All O6-specific beads (`ob-axq`, `ob-iae`, `ob-huw`, `ob-mrd.1`, `ob-onz`, `ob-88p`, `ob-8xc`)
  are de-prioritized for the remainder of the sprint. They remain open for post-submission work.
- The NPU operator-coverage findings (CIX NOE audit `ob-t3b.1`, RKNN cross-vendor `ob-t3b.5`)
  remain in the write-up as architecture-level analysis but are not backed by on-device NPU
  execution benchmarks.
- The headline numbers come from CPU kernel microbenchmarks and the analytical memory model, not
  from end-to-end model inference on the target NPU.

**Follow-on work.**

- `ob-fnq` (README final pass) — reframe for Edge AI.
- `ob-f7k` (Devpost write-up) — map to the Edge AI rubric.
- `ob-ami` (master comparison table) — assemble from committed CSVs.
- `ob-kdi` (clean-clone rehearsal) — verify reproduction path.
- `ob-9e2` (compliance pass) — verify all checklist items.

**Reversal cost.**

Low. If the O6 arrives and boots successfully before Aug 11 (T-3), we could add a section with
on-device numbers. The write-up framing does not change — Edge AI covers the O6. The trigger for
reconsideration would be: O6 booted with SSH access AND CIX SDK working on-device, both confirmed
by Aug 10.
