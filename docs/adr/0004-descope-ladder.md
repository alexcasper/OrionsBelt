# ADR 0004: Ratify the descope ladder with dated trigger points

- **Status:** Accepted
- **Date:** 2026-08-02
- **Bead:** `ob-atp`
- **Deciders:** Claude (agent) — ratifying the ladder as an operable artifact; the maintainer
  retains override authority at every T-date below, and two of the four decisions (T-4, T-3)
  are explicitly maintainer-owned, not agent-mechanical (see "Who decides" per row).

## Context

Today is 2026-08-02, T-12. Deadline is 2026-08-14 16:00 PDT. PLAN.md §7 pre-agreed a descope
ladder (T-4 Aug 10 / T-3 Aug 11 / T-2 Aug 12 / T-1 Aug 13) precisely so this wasn't improvised
under pressure. `docs/RISK_REGISTER.md` §3 restated it with bead references. This ADR does not
re-derive the ladder from nothing — it ratifies it, but **corrects it against facts that did
not exist when it was written**, because ratifying stale assumptions as if they were current
would defeat the entire purpose of the exercise.

**What has changed since PLAN.md §7 was drafted, same-day:**

- The CIX NOE Compiler / NPU SDK is **in hand and working on an x86 host**
  (`cixbuilder-6.1.3753.3`). Risk R2 (CIX access) was downgraded High/High → Low/Low
  (`docs/RISK_REGISTER.md` R2). The original T-4 trigger — "no NPU access → drop NPU offload" —
  is **largely obsolete for the compiler path**. We already have NPU access in the only sense
  that matters for the analysis/design half of the work.
- The NOE operator-coverage audit is **done**, without a board: every GDN arithmetic operator
  is natively supported; `Scan` is rejected; `Loop` is accepted only via compile-time-constant
  unrolling, which is a trap, not a win (`docs/FINDINGS.md` §1). This is the project's core
  technical finding and it required zero hardware.
- Three CPU kernels (gated cumulative decay, gated chunkwise scan, causal depthwise conv1D) are
  **implemented and numerically verified** under QEMU across SVE1/SVE2/NEON/scalar
  (`docs/FINDINGS.md` §4). Correctness is settled. Performance is not — QEMU tells us nothing
  about Cortex-A720 cycle counts or memory bandwidth.
- KleidiAI has no recurrence primitive at all, and its one depthwise-conv family is SME2-only,
  which Cortex-A720 lacks (`docs/FINDINGS.md` §3). Our three kernels fill a gap that is real,
  not rhetorical.
- **The board is still not in hand.** `bd show ob-axq` confirms the sourcing bead is still
  `OPEN`, unclaimed. `bd blocked` currently lists 51 blocked issues; `ob-axq` gates `ob-iae`
  (flash/boot) and `ob-imb` (the Aug 9 track decision) directly, and the entire hardware
  bring-up → profiling → mapping-ADR → optimization → ablation chain transitively. It is now
  the **single remaining external gate** of consequence — R2 no longer blocks anything the
  compiler-only path needs.
- **Zero benchmark numbers exist anywhere** — not on the O6 (no board), and not yet on the
  hedge target either (`ob-ng6`, hedge bring-up, is still open and unstarted per `bd ready`).
  Everything achieved so far is design, audit, and QEMU-verified correctness. This matters for
  the "board-free minimum viable submission" question below: the hedge track is not yet proven
  either, it is simply *unblocked*, which is a materially weaker claim.

**The shape of the risk inverted.** PLAN.md §7 was written under the assumption that NPU
access was the risky, gated unknown and that a board was the more likely near-term outcome —
hence the ladder's original ordering (T-4 checks NPU access; T-3 checks the board, one day
later, as if it were the second domino). Reality is the opposite: the toolchain arrived first
and cheaply, the board did not arrive at all, and it remains gated on a human action (shipping,
a loaner, or remote SSH access to someone else's unit) that no amount of agent effort moves
(`docs/RISK_REGISTER.md` §4). The ladder below is ratified with that inversion made explicit,
not smoothed over.

## Decision

Ratify the descope ladder as four dated, operable decision points. Each restates PLAN.md §7 /
RISK_REGISTER.md §3 with the trigger reinterpreted against present facts where the situation
changed underneath it (T-4 only), and with the additional "last useful arrival" analysis below
folded in as the load-bearing addition this ADR makes.

### T-4 — 2026-08-10

| | |
|---|---|
| **Trigger (dated, checkable)** | The O6 board is not physically in hand by this date. (Reinterpreted: the original trigger — "no NPU *access*" — is moot, since NPU/NOE access has existed since 2026-08-02. The fact that still gates anything is board possession, checkable via `bd show ob-axq` — closed with a claim date, or still open.) |
| **Cut** | On-device NPU execution and INT4 quantization *validation* — i.e., running the exported/quantized subgraphs and measuring them (`ob-onz` downstream of real hardware). |
| **Kept** | (a) The NOE op-coverage audit and its finding — already complete, needs no board, stands regardless. (b) The CPU+GPU hybrid path: SVE1/SVE2/NEON kernels (verified) and the Vulkan/OpenCL scan design, both of which need only the hedge target or the O6's CPU/GPU, not its NPU. (c) A documented, unexecuted NPU offload design (subgraph boundaries, quantization policy) as a "designed but not run" contribution. |
| **Decider** | Maintainer — this is a scope call with prize-framing consequences (whether any NPU story survives into the write-up at all), not a mechanical check. |
| **Submission still claims** | "Every GDN operator the NPU toolchain needs is supported except the sequential scan, which is architecturally inexpressible on this NPU (`Loop` unrolling trap included) — verified via the compiler frontend, without silicon. We designed but did not execute an NPU offload path for the periodic full-attention/FFN layers because the board never arrived; here is that design. The optimization we *did* measure is a CPU+GPU heterogeneous path — Arm SVE/NEON kernels plus a Vulkan/OpenCL scan — which is itself a legitimate, fully Arm-specific answer to the 40-point rubric line." |
| **Follow-up bead** | File "NPU on-device execution and INT4 accuracy regression — designed, not executed (board never arrived by T-4)," referencing `ob-onz`, `ob-huw`, and this ADR. |

### T-3 — 2026-08-11

| | |
|---|---|
| **Trigger** | The O6 board is not physically in hand **and booted** by this date. |
| **Cut** | Physical AI framing in its entirety: all O6-specific claims in the About section, write-up, and results table; anything downstream of `ob-iae` (flash/boot). |
| **Kept** | The Edge AI track on the aarch64 hedge target (`ob-8ms`/E3, per ADR 0002): the GDN memory-scaling story (O(1) recurrent state vs. linearly growing KV cache), built hardware-independent from day one and therefore intact regardless of this trigger. |
| **Decider** | Maintainer — but per `ob-imb` (the Aug 9 hard track go/no-go), this should already be the *expected*, pre-committed outcome by this date, not a live debate. If it is a surprise on Aug 11, the daily go/no-go ritual in `docs/RISK_REGISTER.md` §2 failed earlier in the week. |
| **Submission still claims** | "A Gated DeltaNet model running on an Arm aarch64 edge device (hedge target per ADR 0002), with a measured, reproducible benchmark harness (tokens/s, TTFT, peak memory across ≥3 context lengths), at least one honestly measured before/after Arm-specific optimization (the verified SVE/NEON kernels), the O(1)-state-vs-KV-cache memory story, the NOE op-coverage finding, and the KleidiAI gap analysis — all achieved without ever holding the target Physical AI hardware, and said so plainly." |
| **Follow-up bead** | File "O6-specific work designed/started but not completed — partial profiling data if any exists," referencing `ob-axq`, `ob-iae`, and this ADR. |

### T-2 — 2026-08-12

| | |
|---|---|
| **Trigger** | Schedule position at this date shows insufficient slack to attempt the GDN-2 layer-swap experiment (option b) safely — checkable mechanically via `bd blocked`/`bd ready` state on `ob-9ke`. |
| **Cut** | GDN-2 layer swap into a Qwen3.5-architecture checkpoint evaluated on RULER (`ob-68l`, `ob-zak`). |
| **Kept** | GDN-2 microbenchmark comparison (option a, `ob-82b`, `ob-7b5`) — cheap, safe, and per PLAN.md §9's own working agreement, a negative result here ("the decoupled-gating retrieval hypothesis did not hold at edge scale") is a real Potential Impact contribution, not a failure to hide. |
| **Decider** | Agent-executable — the closest of the four to a mechanical schedule check — but the maintainer is notified same-day regardless. |
| **Submission still claims** | "GDN-2's decoupled erase/write gating hypothesis, documented as ADR 0001 with real RULER numbers from the paper (Table 3, hybrid setting +2.0/+6.2/+3.2 over GDN), plus our own microbenchmark-level comparison and an honest verdict — not a full retrieval evaluation on a swapped checkpoint, which needs training compute we decided in advance we didn't have." |
| **Follow-up bead** | File "GDN-2 layer-swap design, deferred not abandoned," referencing `ob-68l`'s existing description and this ADR. |

### T-1 — 2026-08-13

| | |
|---|---|
| **Trigger** | Time running out ahead of the final push — the last scheduled descope pass before the M4 submission-prep window closes. |
| **Cut** | Demo video (optional per Devpost rules) and the 262K context data point (top of the sweep, R4). |
| **Kept** | Write-up, README, clean-clone reproduction rehearsal, results table — all mandatory per PLAN.md §8's minimum-viable-submission bar. |
| **Decider** | Maintainer for the final call, but per the ritual in `docs/RISK_REGISTER.md` §2, this should be nearly automatic by this date. |
| **Submission still claims** | "Everything PLAN.md §8 lists as minimum-viable: public Apache-2.0 repo, one GDN model running on an Arm target, a reproducible benchmark CSV across ≥3 context lengths up to at least 128K, a measured before/after Arm-specific optimization, a write-up, and a clean-clone reproduction we verified ourselves. The demo video and the 262K point are explicitly out of scope for this cycle, not silently missing." |
| **Follow-up bead** | File "Demo video and 262K context point — intentionally out of scope for this submission," referencing this ADR. |

**Anything cut under any of the four rows above is filed as a follow-up bead, never deleted.**
The repo must show intended scope (what PLAN.md and the beads promised) alongside delivered
scope (what actually shipped) so a judge or future maintainer sees an honest gap, not a quietly
shrunk plan. This is restated because it is the single rule that makes the rest of this ADR
trustworthy — a descope ladder that hides its own cuts is worse than no ladder at all.

### The ladder's new shape

PLAN.md §7 ordered the ladder NPU-first (T-4) then board-first (T-3), one day apart, because it
assumed both were live external risks of comparable and adjacent likelihood. That assumption is
now falsified in one direction and unchanged in the other: NPU/NOE access resolved itself
immediately and for free; the board has not moved at all in the same window and remains gated on
a purely human action. The two triggers are no longer peers — T-4 is now a check on a much
smaller remaining risk (on-device NPU *execution*, not NPU access), and T-3 is the check that
was always going to matter. This ADR keeps both dates because other beads (`ob-imb`,
`docs/RISK_REGISTER.md` §2) already reference them and moving dates now would desynchronize
artifacts across concurrently-working agents — but it records plainly that **T-4's practical
weight is now much lower than PLAN.md §7 implied**, and the real decision the project is riding
on is T-3, exactly as the maintainer-owned §4 escalation in `docs/RISK_REGISTER.md` already says.

### Minimum viable submission if the board never arrives

Concretely, if `ob-axq` never closes before the deadline, the submission consists of:

1. **The NOE op-coverage audit** (`docs/FINDINGS.md` §1) — a real, citable, hardware-independent
   finding: every GDN operator is supported, the scan is not, and the "Loop looks like it works"
   trap is documented with verbatim compiler output.
2. **Three CPU kernels, numerically verified** on a hedge aarch64 target (SVE1/SVE2/NEON/scalar
   under QEMU today; real-silicon timing once `ob-ng6` lands) — gated cumulative decay, gated
   chunkwise scan, causal depthwise conv1D — with an honest statement that correctness is proven
   and performance is not yet measured on real hardware.
3. **The KleidiAI gap analysis** (`docs/FINDINGS.md` §3) — a specific, checkable claim that Arm's
   own optimized-kernel library covers the dense matmuls GDN needs and nothing else, with its one
   depthwise-conv family targeting an extension (SME2) this class of core does not implement.
4. **Whatever the hedge target (ADR 0002) actually measures** — tokens/s, TTFT, and the KV-cache
   vs. GDN-state memory split across a context sweep, run on an Android phone or the Graviton
   fallback.

**Honest assessment: this is a credible Edge AI entry, but a conditional one.** It fully clears
PLAN.md §8's *minimum viable* bar — public repo, one GDN model on an Arm target, a reproducible
CSV, at least one measured before/after optimization, a write-up, and a verified clean-clone
repro — none of which need the O6. It plausibly reaches the *target* tier too, since ADR 0002's
phone option keeps a real Vulkan GPU leg alongside the CPU kernels, so "heterogeneous CPU+GPU
partitioned execution" survives even without the board; only the "+NPU if accessible" clause of
the target tier is lost, and that clause was always conditional. What it **cannot** do is
compete for Physical AI, and it forfeits the specific WOW-factor framing PLAN.md §2.2 leans on —
"a dynamic heterogeneous dispatcher tested on real edge silicon offering an NPU" — leaving a
CPU+GPU story that is still genuinely Arm-specific but narrower in ambition.

The condition this rests on: **the hedge track has to actually produce numbers.** As of today,
`ob-ng6` (hedge bring-up) is unstarted, so "board-free minimum viable submission" is currently a
credible *plan*, not yet a credible *result* — nothing has been measured on any piece of
hardware yet, hedge included. That is the nearer-term risk this ADR flags for the record,
separate from the board question, though outside this ADR's scope to descope against (ADR 0002
owns the hedge-target bring-up risk and its own fallback).

### Last useful arrival date

This is the number PLAN.md §7 never computed, because it treated "board arrives" as binary
against the T-3 abandonment date rather than asking a sharper question: **if the board arrives
on day X, is there still enough runway left to turn it into a measured result before submission,
or does it merely arrive too late to matter?** Bring-up, profiling, optimization, and write-up
integration are not a one-day job, and the existing T-3/T-4 dates were set as *abandonment*
triggers, not *usefulness* cutoffs — those are different questions and conflating them is exactly
the kind of false reassurance this ADR should not produce.

Working backward from PLAN.md's own internal submission target (§5, M5: "submit today [Aug 13],
leaving Aug 14 as pure slack" — not the hard Aug 14 16:00 PDT deadline, which this ADR treats as
an emergency-only reserve, consistent with the plan's own stated intent to bank ≥16h of margin):

| Stage | Work | Realistic duration |
|---|---|---|
| 1 | Flash Debian 12, first boot, SSH (`ob-iae`) | 1 day — bring-up on new silicon has already surprised us twice on the compiler side alone (the `cixparse` unconditional `tensorflow` import, the GCC 13 `-mcpu=cortex-a720` rejection); budget for at least one equivalent surprise on the board |
| 2 | NPU runtime smoke test (`ob-huw`) + Vulkan/OpenCL GPU stack validation (`ob-88p`), parallelizable | 1 day |
| 3 | Re-run the **already-verified** SVE/NEON kernels and the (already-built, portable) harness on real silicon, producing the first honestly-measured on-device number | 1 day |
| — | **Bare minimum-viable on-device addendum reached: ~3 days** | |
| 4 | Per-layer latency profile (`ob-c9k`/`ob-41j`) + boundary-crossing cost (`ob-t3b.3`) | 1 day |
| 5 | Mapping ADR (`ob-o4g`) finalized against real data | 0.5 day |
| 6 | NPU export/INT8→INT4 quantization + accuracy regression (`ob-onz`/`ob-27y`), Vulkan scan kernel validated on real GPU (`ob-q44`/`ob-gzk`), dispatcher wiring | 2 days |
| 7 | Sustained-load thermal characterization + ablation matrix + results table | 1 day |
| — | **Full "target"-tier heterogeneous submission: ~7.5–8.5 days of hardware work** | |
| 8 | Write-up/README integration and a clean-clone repro rehearsal specific to the real hardware (M4) | 1 day |

**Bare-minimum path (3 days hardware + ~1 day write-up patch ≈ 4 days):** counting back from
Aug 13 EOD, the board must be in hand no later than **2026-08-09**. This is one day *before*
the existing T-4 date (Aug 10) — meaning if T-4 fires as written (no board by Aug 10), a
bare-minimum on-device addendum was already foreclosed the day before, not the day of. Padding
by one further day for the near-certainty of at least one bring-up surprise (per the compiler
precedent above) moves the genuinely defensible cutoff to **2026-08-08**.

**Full "target"-tier path (~8.5 days hardware + 1 day write-up ≈ 9.5 days):** counting back from
Aug 13 EOD lands at **2026-08-04** — effectively already gone by the time this ADR is written.

**Ratified cutoff: 2026-08-08.** A board arriving on or before this date can still be turned
into a genuine, honestly-measured on-device result worth adding to the submission. A board
arriving after 2026-08-08 and on or before 2026-08-10 (T-4) can, at best, yield a single
opportunistic on-device number bolted onto an already-complete Edge AI submission, at real risk
of not making it in cleanly — treat that window as a bonus, not a plan. A board arriving after
2026-08-10 should not be built into any submission plan at all; if it happens, take whatever
measurement is cheap and honest, but do not let it change the write-up's framing or the Edge AI
commitment already locked in by T-3.

This does **not** mean the maintainer should stop pursuing the board after 2026-08-08 — sourcing
it is a zero-cost-to-the-plan, maintainer-only action (`docs/RISK_REGISTER.md` §4), and the
remote-SSH-to-someone-else's-board path in particular could still land inside the useful window
if it resolves quickly. It means the *plan* stops being built around the assumption that it will.

## Alternatives considered

| Option | Why not |
|---|---|
| Leave T-4 exactly as PLAN.md §7 phrased it ("no NPU access → drop NPU offload") without reinterpretation | Would ratify a trigger condition that already resolved itself on 2026-08-02 (R2 downgrade). Reratifying a stale trigger unmodified would either fire on a condition that's already false (falsely implying NPU work is entirely safe) or silently mean something different from what it says. What would change our mind: if R2's downgrade were itself found to be wrong (e.g., the SDK license terms are discovered to require an approval we missed) — nothing in this session's verification suggests that. |
| Move the T-3/T-4 calendar dates earlier to match the computed "last useful arrival" cutoff (2026-08-08) | Rejected on file-ownership and coordination grounds, not technical ones: `ob-imb` (Aug 9 go/no-go), `docs/RISK_REGISTER.md` §2's daily ritual, and other concurrently-working agents already reference Aug 10/11 as fixed dates. Changing them here would desynchronize artifacts this task is explicitly not authorized to edit. Instead this ADR records the *effective* usefulness cutoff as earlier than the *abandonment* date, which is the honest correction without touching other files. What would change our mind: if a future revision of PLAN.md or the risk register itself moves these dates, in which case this ADR should be revisited to match. |
| Stop pursuing the board entirely once the last-useful-arrival cutoff passes, to avoid wasted maintainer attention | Rejected — pursuing the board (retail, loaner, remote SSH) is a maintainer-side action with essentially zero cost to the agent-executed plan, and the upside of an unexpectedly fast resolution (especially remote SSH access to someone else's unit, PLAN.md §11 open question 1) is large enough that giving up early is pure downside. What would change our mind: evidence that continuing to chase the board actively costs maintainer time needed elsewhere in the final week — not observed. |
| Treat "board never arrives" as equivalent to a failed submission and downgrade overall confidence accordingly | Rejected as inaccurate, not merely as unwelcome. `docs/RISK_REGISTER.md` §5's own first review already concluded the MVP bar is achievable on Edge AI alone; this ADR's "minimum viable submission" analysis above reaches the same conclusion independently. The honest position is "Physical AI is at serious risk; a complete, legitimate submission is not," and collapsing those two into one verdict would misstate the actual risk. |

## Consequences

**Accepted costs.**
- If T-4/T-3 fire as expected (the likelier-than-not outcome given today's state), the
  submission's WOW-factor ceiling drops from "dynamic heterogeneous NPU+GPU+CPU dispatcher
  measured on real edge silicon" to "CPU+GPU heterogeneous path, NPU offload designed but
  unexecuted, all measured on a hedge target instead of the named competition hardware." That is
  a real reduction in ambition, not a cosmetic one, and this ADR does not pretend otherwise.
- The Physical AI track is very likely forfeited outright, per the honest read already on record
  in `docs/RISK_REGISTER.md` §5. This ADR does not change that assessment; it operationalizes
  what happens given that assessment.
- Every cut filed as a follow-up bead is a permanent, visible admission of reduced scope in the
  public repo. That is the intended behavior (honesty over polish), but it is a cost worth naming
  plainly: a judge who reads the beads sees exactly what was promised and not delivered.

**Follow-on work.** Four follow-up beads as specified per row above (T-4: NPU on-device
execution deferred; T-3: O6-specific work deferred; T-2: GDN-2 layer-swap deferred; T-1: demo
video/262K point deferred), each filed only if and when its trigger actually fires — not
pre-filed speculatively, since filing them now for triggers that may not fire would clutter
`bd ready`/`bd blocked` with beads whose relevance is conditional. No `bd` write commands were
run under this task's constraints; recommend the maintainer or a future agent session file these
at the point each trigger is confirmed to have fired, referencing this ADR and the relevant
existing beads named per row above.

**Reversal cost.**
- **T-4 (NPU on-device work cut) is cheap to reverse if the board arrives late.** The design work
  (subgraph boundaries, quantization policy, the mapping hypothesis in PLAN.md §3.1) is already
  hardware-independent and complete; reversal is *resuming execution*, not redesigning anything.
- **T-3 (Physical AI framing cut) is not something to "reverse" in the usual sense if the board
  arrives after 2026-08-08 but the Edge AI submission is already locked.** Per the last-useful-
  arrival analysis, a late board should be treated as an *additive* bonus measurement layered on
  top of an already-complete Edge AI entry, not a wholesale pivot back to Physical AI framing
  under time pressure. Attempting a full reversal this late is the expensive, high-risk path;
  taking one opportunistic measurement is the cheap one.
- **T-2 (GDN-2 layer swap cut) is cheap to reverse** if the project is genuinely ahead of
  schedule at the Aug 12 checkpoint — this is the plan's own pre-agreed escalation clause, not a
  one-way door.
- **T-1 (video/262K point cut) is nearly free to reverse before the 2026-08-14 16:00 PDT hard
  deadline** — both are additive, not structural, and can be added back in any slack that
  materializes. After the hard deadline, reversal cost is not "expensive," it is **infinite** —
  Devpost submissions close at a fixed time with no grace period assumed. This is the one row
  in the ladder where "reversal" stops being a cost question and becomes a hard wall.
