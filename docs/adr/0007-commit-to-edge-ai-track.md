# ADR 0007: Commit to the Edge AI track — the Orion O6 has not arrived

- **Status:** Accepted
- **Date:** 2026-08-06
- **Bead:** `ob-imb`
- **Deciders:** Claude (agent, t3 node) — formalising the outcome ADR 0004 identified as
  expected; the maintainer retains override authority if the board arrives before the
  last-useful-arrival cutoff (2026-08-08).

## Context

The single most consequential scheduling decision in this project: which Devpost track to
target. `ob-imb` set a hard date of 2026-08-09 so the write-up has five clear days of stable
framing before the 2026-08-14 16:00 PT deadline. Today is 2026-08-06.

**The board has not arrived.** `bd show ob-axq` confirms the Orion O6 sourcing bead is still
OPEN — no retail purchase, no loaner, no remote SSH access has materialised in the four days
since ADR 0004 ratified the descope ladder. `ob-iae` (flash/boot) remains transitively blocked,
and the entire hardware bring-up → profiling → optimisation → ablation chain depends on it.

**ADR 0004 computed the last-useful-arrival date as 2026-08-08** (two days from now). A board
arriving after that date cannot yield a genuine, honestly-measured on-device result in time for
the submission; at best it produces a single opportunistic number bolted onto an already-complete
Edge AI entry. That window has not closed yet, but it closes in 48 hours, and no signal suggests
imminent resolution.

**The Edge AI track is not a fallback — it is a proven, measured path.** Since ADR 0005
established the device fleet strategy, the following results have been captured on real Arm
silicon, all with committed manifests:

| Result | Device | Finding | Source |
|---|---|---|---|
| Phase 1 kernel benchmarks | RK3588 (A76 + A55) | 28-row CSVs, p50/p95, 8 kernel variants × 4 model configs | `results/raw/rk3588-t3_big.csv`, `_little.csv` |
| NPU operator coverage — CIX NOE | x86 host (NOE compiler) | Scan rejected; Loop trap; all arithmetic ops supported | `docs/FINDINGS.md` §1 |
| NPU operator coverage — Rockchip RKNN | RK3588 NPU | Recurrence fails independently; Scan → CPU-only; Loop → rejected | `docs/FINDINGS.md` §7 |
| KleidiAI matmul evaluation | RK3588 A76 | 1.7–3.6× over hand-NEON on matmul; packing cost dominates at decode | `docs/FINDINGS.md` §8 |
| big.LITTLE affinity policy | RK3588 (A76 + A55) | Pinning to big cores: 2–3× over default scheduler; split workload <10% interference | `docs/FINDINGS.md` §9 |
| Three CPU kernels | Verified under QEMU + real A76 silicon | gated scan, cumulative decay, causal conv1d — NEON dispatch, correctness confirmed | `docs/FINDINGS.md` §4 |

These are not speculative plans. They are committed, reproducible results with provenance
artifacts. The Edge AI submission is already credible; it does not need the O6 to become so.

**ADR 0004's own honest assessment**: "this is a credible Edge AI entry." The minimum viable
submission bar (PLAN.md §8) is cleared without the board: public repo, one GDN model on an Arm
target, reproducible CSV, measured before/after optimisation, write-up, clean-clone repro.

## Decision

**Commit to the Edge AI track effective immediately.** All submission framing, write-up, and
results tables target Edge AI on the available device fleet (RK3588, Raspberry Pi 5, Jetson
Nano). O6-specific work is deprioritised from the critical path.

Specifically:

1. **README and Devpost write-up** frame the project as an Edge AI entry. The fleet
   bandwidth-scaling study (ADR 0005) is the headline result. The NPU operator-coverage finding
   (both CIX NOE and Rockchip RKNN) is the core technical contribution — it generalises beyond
   one vendor and is measurable without the target board.

2. **O6 arrival after 2026-08-08 is treated as additive bonus**, not a framing pivot. If a board
   materialises (especially remote SSH to someone else's unit), take one opportunistic measurement
   and append it as a datapoint. Do not restructure the submission around it.

3. **The descope ladder (ADR 0004) T-4 and T-3 triggers are pre-committed**: T-4 (Aug 10) cuts
   on-device NPU execution; T-3 (Aug 11) cuts Physical AI framing. This ADR makes that outcome
   explicit three days early so the write-up can proceed with stable framing for the full
   five-day window `ob-imb` requires.

4. **Remaining agent effort focuses on portable, device-available work**: the fleet
   bandwidth-scaling comparison across all devices, the GDN-2 microbenchmark (ob-82b), and
   submission materials (README final pass, comparison table, reproduction rehearsal).

## Alternatives considered

| Option | Why not | What would change our mind |
|---|---|---|
| Wait until the Aug 9 hard date before deciding | Costs three days of write-up runway. ADR 0004 already identified this as the expected outcome ("this should already be the expected, pre-committed outcome by this date, not a live debate"). The board status has not changed since ADR 0004 was written. | If ob-axq closes (board sourced) before Aug 8, the decision reverses — but we would know that immediately. |
| Keep both tracks open (ambiguous framing) | Worst option. A write-up that hedges between Physical AI and Edge AI satisfies neither rubric. Devpost judges assess track fit; an ambiguous entry scores lower on both. | Nothing — dual-track framing is strictly worse than committing. |
| Commit to Cloud AI instead | The fleet is edge hardware (SBCs with Arm CPUs). Cloud AI would require x86/CUDA infrastructure (ob-aqv) which is also not set up. Edge AI is the natural fit for what we have. | If a cloud GPU instance materialised and the fleet proved unable to run any model end-to-end — neither condition holds. |

## Consequences

**Accepted costs.**
- Physical AI track is forfeited. The "dynamic heterogeneous NPU+GPU+CPU dispatcher on real
  edge silicon" WOW-factor framing (PLAN.md §2.2) is replaced with a narrower but still
  Arm-specific "CPU+GPU heterogeneous path on a measured device fleet" story.
- NPU on-device execution and INT4 quantisation validation will not happen unless a board
  arrives before Aug 8. The NPU offload design remains as a "designed but not executed"
  contribution.
- No available device has SVE/i8mm. The SVE2 kernel path (ob-8qt.1) cannot be measured on
  this fleet. The NEON path — which all three devices use — is the measured contribution.

**Follow-on work.**
- File follow-up beads per ADR 0004's T-4 and T-3 rows when their triggers fire (Aug 10/11),
  referencing this ADR: "NPU on-device execution deferred" and "O6-specific work deferred."
- `ob-fnq` (README final pass), `ob-ami` (master comparison table), and `ob-f7k` (Devpost
  write-up) now have a stable framing to work against.
- `ob-kdi` (clean-clone reproduction rehearsal) should verify the Edge AI path end-to-end.

## T-4 Trigger Firing (2026-08-10)

Confirmed by t3 agent on 2026-08-10 (T-4, 4 days to deadline).

- `ob-axq` (Source a Radxa Orion O6) is still **OPEN** — no retail purchase, no loaner, no remote
  access has materialised.
- The board has not arrived by the last-useful-arrival cutoff (2026-08-08).
- **T-4 trigger fired on schedule.** Per ADR 0004 §T-4, on-device NPU execution and INT4
  quantization validation (`ob-onz`, `ob-huw`) are cut from the critical path. They remain as
  "designed but not executed" contributions (ADR 0004 §T-4 Kept (c)).
- Follow-up bead **`ob-9t0.10`** (filed+closed 2026-08-10T04:32Z) is the formal record,
  referencing `ob-onz`, `ob-huw`, ADR 0004, and this ADR.
- **T-3 trigger (Aug 11)** is pre-committed by this ADR: Physical AI framing is already dropped;
  the submission is locked to Edge AI. No reversal is expected.
- The Edge AI submission remains complete and submission-ready (15/15 readiness checks, 2253
  tests, ruff clean, all numbers provenance-backed).

**Reversal cost.**
- **If the board arrives before 2026-08-08**: reversal is cheap. The Edge AI submission is
  intact and additive; bolt on O6 measurements as bonus datapoints. The write-up gains a
  "designed for edge, validated on target silicon" framing that is strictly stronger.
- **If the board arrives after 2026-08-08**: do not reverse. Take one opportunistic measurement
  if cheap, but do not restructure the submission. The Edge AI framing is locked.
- **After 2026-08-14 16:00 PT**: reversal cost is infinite. Devpost submissions close at a
  fixed time with no grace period.
