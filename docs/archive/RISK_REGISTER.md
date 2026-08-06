# Risk Register and Go/No-Go Ritual

Operable companion to [`PLAN.md`](./PLAN.md) §7 (risk register / descope ladder) and §5
(milestones). PLAN.md states the risks in prose and lists them once; this document tracks
them as a living artifact with concrete triggers, owners, and a daily ritual someone can
actually run between now (2026-08-02, T-12) and submission (2026-08-14, 16:00 PDT).

This file does not replace PLAN.md §7 — it operationalizes it. Numbering (R1–R10) matches
PLAN.md exactly. If the two ever disagree, this file is the current status and PLAN.md is the
original design intent; reconcile by editing this file, not by treating PLAN.md as stale.

**Owner legend:**
- **maintainer** — requires a human with purchasing power, an email address, or physical
  possession of hardware. No amount of agent effort substitutes.
- **agent** — can be advanced by automation/agent work alone, though it may be *blocked on*
  a maintainer-owned risk.

---

## 1. Register

| # | Risk | Likelihood | Impact | Owner | Trigger condition (dated, observable) | Mitigation | Status (2026-08-02) |
|---|---|---|---|---|---|---|---|
| R1 | O6 board does not arrive before deadline | High | Critical | **maintainer** | No tracking number / confirmed ship date by **EOD 2026-08-05** (T-9), OR board not physically in hand by **2026-08-11 (T-3)** | Hedge track (E3) built from day one, hardware-independent; hard track go/no-go 2026-08-09 (`ob-imb`); Edge AI submission on generic aarch64 if trigger fires | Open — no board, no tracking number, no loaner/remote-access lead confirmed as of today  **Usefulness cutoff 2026-08-08** (ADR 0004): a board arriving after this cannot change the submission even though we keep pursuing it until T-4. This is 2 days EARLIER than the T-4 abandonment trigger — do not conflate them. |
| R2 | ~~CIX Early Bird access not granted in time~~ **LARGELY DISSOLVED 2026-08-02** | Low | Low | agent | N/A — superseded. The NOE Compiler / NPU SDK is a **direct download with no approval gate**, requires **Python 3.10** (not 3.8), and runs on an **x86 host**, per [Radxa env-setup docs](https://docs.radxa.com/en/orion/o6/app-development/artificial-intelligence/env-setup) and the public [cix-manifest wiki](https://github.com/cixtech/cix-manifest/wiki). See `CLAIM_VERIFICATION.md` §2.2a | No mitigation needed for the compile path. Residual risk is only that *running* a compiled artifact still needs the board (folded into R1) | **Downgraded** — was High/High/maintainer-owned. NPU compiler work is now actionable today with no hardware: see bead `ob-t3b.1` |
| R3 | NOE Compiler has no kernel for GDN recurrent scan | High | Medium | agent | Op-coverage audit (`ob-8xc`) returns "no GDN scan op" for the layer types found in `t-arch-audit` — checkable as soon as NPU access exists | Treated as a finding, not a failure: document the op-coverage gap as a contribution; route the scan to GPU/CPU by design regardless of NPU outcome | Not yet assessable — blocked behind R2 (need NPU/NOE access first) |
| R4 | Long contexts (262K) exceed 64GB or take too long to benchmark | Medium | Medium | agent | Any single context-length run in the sweep (`ob-del`) exceeds device RAM or a single run exceeds ~30 min wall-clock during harness dry-run | Incremental sweep 4K→32K→128K→262K; each point is independently publishable; drop the top point at T-1 (2026-08-13) per descope ladder | Not yet assessable — harness (`ob-ljh`, `ob-ar3`) not built |
| R5 | GDN-2 layer swap needs training compute we don't have | High | Low | agent | By **2026-08-10 (T-2 for this decision)**, `ob-9ke` (benchmark-only vs layer-swap decision) is not yet unblocked with schedule slack to spare | Default to benchmark-only option (a); option (b) only attempted if genuinely ahead of schedule at that checkpoint | On track — plan already defaults to option (a); no action needed yet |
| R6 | Quantization destroys GDN accuracy (recurrent state is precision-sensitive) | Medium | Medium | agent | Accuracy regression (`ob-27y`) against the oracle exceeds the agreed tolerance for any quantized layer | Correctness oracle (`ob-3uh`) gates every quantization step; per-layer policy (`ob-qpa`) keeps recurrent state/gates in FP16 | Not yet assessable — oracle and quant policy not yet built |
| R7 | Thermal throttling makes numbers irreproducible | Medium | Medium | agent | Sustained-load run (`ob-dgn`) shows >10% throughput drop from first-minute to steady-state at matched conditions | Sustained-load characterization + run manifests capturing clocks/thermals (`ob-u37`); report percentiles, not best-of-N | Not yet assessable — no hardware to characterize yet |
| R8 | Brief's cited figures are wrong (weak secondary sources) | Medium | High | agent | N/A — trigger already fired and was resolved | P0 verification bead (`ob-ofk`) against primary sources before any number enters the write-up | **Closed 2026-08-02** — see [`CLAIM_VERIFICATION.md`](./CLAIM_VERIFICATION.md); corrections folded into PLAN.md |
| R9 | Time runs out mid-optimization | Medium | High | maintainer + agent | Any T-4/T-3/T-2/T-1 trigger below fires (see §3) | Pre-agreed descope ladder (§3) executed on the date, not improvised | Open — ladder defined, not yet needed |
| R10 | Qwen3.5 checkpoint license restricts redistribution | Low | Medium | agent | License audit during model survey (`ob-eae`/`ob-7fv`) finds a non-redistributable license on the selected checkpoint | License audit is part of model survey; fetch scripts (`ob-ixt`) download weights rather than vendoring them | Not yet assessable — model selection decision not yet made |

**Read R1 carefully: it is maintainer-owned.** Every other risk in this table can, in principle,
be advanced or resolved by agent work alone (subject to being unblocked). R1 cannot — see §4.

> **Update 2026-08-02 (same day this register was written):** R2 has been **downgraded from
> High/High to Low/Low**. It was written on `brief.md`'s claim that the NPU toolchain sits behind
> CIX Early Bird enrollment and pins Python 3.8. Both are wrong: the SDK downloads directly with no
> approval, needs Python 3.10, and — most importantly — **runs on an x86 host**, so GDN operator
> compilation needs no board at all. This is the single largest de-risking event so far: the NOE
> op-coverage audit (risk R3, and the project's core technical contribution) moved from
> double-gated to actionable today. Only R1 (board access) remains a true external gate, and it now
> blocks only *on-device execution and measurement*, not compiler work.

---

## 2. Daily go/no-go checklist

Run this once per day, ideally at the start of the working session. It should take under five
minutes when nothing has changed, and exists precisely so that a changed situation doesn't sit
unnoticed for 24+ hours in a 12-day window. Each day names the milestone window it falls in
(PLAN.md §5) and the one question that actually matters that day.

Commands to run every day, in order:

```bash
bd ready       # what's actually unblocked right now
bd blocked     # what's stuck, and on what
bd stats       # aggregate view: open/in-progress/blocked/closed counts
```

| Date | Milestone window | Key question of the day | Extra check |
|---|---|---|---|
| **Aug 2** (today) | M0 Foundations | Are both external-gate beads (`ob-axq`, `ob-aop`) filed today? | `bd show ob-axq ob-aop` — confirm not sitting in `open` unclaimed |
| **Aug 3** | M0 Foundations | Is the repo skeleton + Apache-2.0 + results schema actually done, independent of hardware? | `bd blocked` should show zero E1 items blocked on hardware |
| **Aug 4** | M1 Portable core | Does the x86 reference oracle run yet? | Check `ob-aqv`, `ob-eae` status |
| **Aug 5** | M1 Portable core | **R1 trigger check:** tracking number / confirmed ship date for O6 in hand? | If no by EOD today, R1 escalates — see §4 |
| **Aug 6** | M1 Portable core | Is the hedge target (E3) producing baseline numbers yet? | `bd show ob-8ms` epic status; this is the fallback's own proof of life |
| **Aug 7** | M1→M2 boundary | **R2 trigger check:** any response from CIX Early Bird? | If none by EOD today, R2 escalates — see §4 |
| **Aug 8** | M2 Hardware bring-up | Is there anything left to do before the Aug 9 go/no-go besides waiting? | Pre-stage the `ob-imb` decision bead with whatever facts exist |
| **Aug 9** | **M2 — hard go/no-go** | Physical AI or Edge AI? Decide and record as an ADR (`ob-imb`). No half-measures past this point. | `bd show ob-imb`; confirm ADR filed in `docs/adr/` before end of day |
| **Aug 10** | M3 Optimization / **T-4** | Is there NPU access? If not, execute the T-4 cut now (see §3) | `bd blocked` — confirm NPU-dependent beads are formally descoped, not just idle |
| **Aug 11** | M3 Optimization / **T-3** | Is the O6 physically in hand and booted? If not, execute the T-3 cut now | Confirm `ob-imb` ADR matches reality; file follow-up beads for anything cut |
| **Aug 12** | M3 Optimization / **T-2** | Is the ablation matrix and results table on track for Aug 13 draft? Execute T-2 cut if needed | `bd ready` should show write-up/repro beads becoming unblocked |
| **Aug 13** | **M4 Submission prep / T-1** | Is the clean-clone repro rehearsal passing? Execute T-1 cut (demo video, 262K point) if time is short | Target: submit *today*, leaving the Aug 14 window as pure slack |
| **Aug 14** | M5 Submit (target already met) | Is the Devpost submission actually live, with ≥16h slack before 16:00 PT? | `bd show ob-j7f` — confirm closed |

---

## 3. Descope ladder — dated decision points

Restated from PLAN.md §7 with calendar dates and named deciders. **Anything cut is filed as a
follow-up bead, never deleted** — the repo should honestly show intended scope (what PLAN.md
promised) versus delivered scope (what shipped), and a judge or future maintainer should be able
to see both.

### T-4 — 2026-08-10

- **Trigger:** No CIX NPU access granted (R2 unresolved as of this date).
- **Cut:** NPU offload work, INT4 quantization path (`ob-onz`, and anything downstream of it in
  the NPU chain).
- **Keep:** CPU+GPU hybrid — Vulkan compute-shader scan (`ob-q44`), i8mm/SVE CPU paths
  (`ob-dqu`), big.LITTLE affinity. This remains genuinely Arm-specific and satisfies the
  "Arm-specific optimization" rubric line without the NPU.
- **Decider:** maintainer (this is a scope call with prize-track consequences, not a mechanical
  one).
- **Follow-up:** file a bead documenting the NPU path as "designed but not executed due to access
  gate," referencing `ob-aop`.

### T-3 — 2026-08-11

- **Trigger:** No O6 board physically in hand (R1 unresolved as of this date).
- **Cut:** Physical AI framing, all O6-specific work (anything downstream of `ob-iae` flash/boot).
- **Keep:** Edge AI track on the aarch64 hedge target (E3); the GDN memory-scaling story stays
  fully intact because it was built hardware-independent from day one.
- **Decider:** maintainer, though by this date the Aug 9 `ob-imb` decision should already have
  made this the *expected* path, not a surprise.
- **Follow-up:** file a bead recording the O6-specific work that was designed/started but not
  completed, with whatever partial profiling data exists.

### T-2 — 2026-08-12

- **Trigger:** Schedule position — insufficient slack to attempt the GDN-2 layer-swap experiment
  safely.
- **Cut:** GDN-2 layer swap (option b, `ob-68l`, `ob-zak`).
- **Keep:** GDN-2 microbenchmark comparison (option a, `ob-82b`, `ob-7b5`) — cheap, safe, and a
  negative result here is still a publishable contribution per PLAN.md's own working agreements.
- **Decider:** agent-executable (this is closest to a mechanical schedule check — `bd blocked` /
  `bd ready` state on `ob-9ke` tells the story), but maintainer should be notified same-day.
- **Follow-up:** file a bead capturing the layer-swap design (already in `ob-68l`'s description)
  as deferred, not abandoned.

### T-1 — 2026-08-13

- **Trigger:** Time running out ahead of the final push to submission (final descope pass).
- **Cut:** Demo video (optional per Devpost rules), the 262K context data point (top of the
  sweep).
- **Keep:** Write-up, README, clean-clone repro rehearsal, results table — all mandatory per
  PLAN.md §8 minimum-viable-submission criteria.
- **Decider:** maintainer for the final call, but this should be nearly automatic by this date —
  if it's a surprise on Aug 13, the ritual in §2 failed earlier in the week.
- **Follow-up:** file a bead noting the video/262K point as intentionally out of scope for this
  submission, so a future pass can add them.

---

## 4. Escalation — what only the maintainer can do

**R1 and R2 cannot be resolved by any amount of agent work.** This is not a hedge or a
disclaimer — it is the literal, load-bearing fact this whole register exists to make visible.
Agent work can prepare everything around these two beads (harness, oracle, hedge target,
write-up scaffolding) so that the moment either gate opens, downstream work starts within hours.
But the gates themselves — a board shipping, a program approving an application — require a
human with purchasing authority, an email inbox, or a phone.

**What the maintainer needs to be asked for, and by when:**

1. **Today (2026-08-02):** Confirm `ob-axq` (source an O6) and `ob-aop` (CIX Early Bird
   application) are both actually submitted/ordered today, not just filed as beads. A bead in
   `bd ready` that nobody has acted on is not progress.
2. **By EOD 2026-08-05 (T-9):** A tracking number or confirmed ship date for the board, or an
   honest statement that neither exists yet. If neither exists by this date, R1 should be
   treated as trending toward its worst case and the Aug 9 go/no-go should be planned around
   Edge AI as the default outcome, not a fallback.
3. **By EOD 2026-08-07:** Any response — approval, rejection, or estimated turnaround — from the
   CIX Early Bird Program. Silence past this point should be actively chased (a direct follow-up
   email/call), not just monitored.
4. **2026-08-09:** A firm decision on Physical AI vs Edge AI (`ob-imb`), made with whatever facts
   exist on that date. This cannot slip — PLAN.md fixed it as a hard date specifically so the
   write-up has five clear days of stable framing.

**The single highest-leverage unblock available:** remote SSH access to *someone else's* O6 —
a colleague's board, a Radxa-provided loaner, or any third party willing to grant temporary
access — would satisfy nearly all hardware-dependent work (`ob-iae` through the profiling and
optimization chain) without waiting on shipping at all. This is explicitly called out in
PLAN.md §11's open questions and deserves the maintainer's first phone call or email today, not
day nine. If this materializes, it collapses R1 to near-zero without touching R2, and the CIX
enrollment (R2) becomes the sole remaining hardware-side blocker.

---

## 5. First review — 2026-08-02

Conducted today, alongside the register's creation, per this bead's definition of done.

**State observed (via `bd stats`, `bd ready`, `bd blocked`):**

- 80 total issues; **4 closed**, 8 in progress, 68 open, **56 blocked**.
- 12 issues ready to work, none of which touch hardware except the two gate beads themselves
  (`ob-axq`, `ob-aop`) and epic containers.
- Both external gates (`ob-axq` source-a-board, `ob-aop` CIX enrollment) are still open and, as
  far as this review can tell from bead state, not yet visibly acted on (no claim, no notes on
  either bead indicating an order was placed or an application submitted).
- No board. No SDK access. No tracking number. No CIX response.
- The critical-path decision bead `ob-imb` (Physical AI vs Edge AI, hard date Aug 9) is itself
  blocked on exactly these two gates plus the hedge target coming up — i.e., the single most
  consequential decision in the whole plan is currently sitting on the two risks that no agent
  can move.
- The four closed beads are all genuinely portable, hardware-independent work: license/About
  section, claim verification, repo skeleton, ADR template. This confirms the "portable chain"
  described in PLAN.md §6 is the only chain currently making progress, exactly as designed.

**Honest assessment of deadline feasibility:**

Twelve days out, with zero hardware-dependent progress possible and both gating beads still
sitting unclaimed at T-12, **the Physical AI track is at serious risk.** The plan's own
structure anticipated this — the two-track design and the Aug 9 go/no-go exist precisely because
this exact situation (external gates unresolved this deep into the window) was foreseen as
plausible, not just possible. But foreseeing a risk is not the same as it not mattering: if
today's state (no tracking number, no CIX response) is unchanged by 2026-08-05–07, the realistic
default outcome by 2026-08-09 is **Edge AI**, not Physical AI, and the plan should be run on that
assumption rather than hoping the gates resolve. This is not a reason to stop pursuing the O6
board or CIX access — the upside if either resolves quickly (especially the remote-SSH path in
§4) is large, and the portable/hedge track was explicitly designed to lose nothing by running in
parallel. But an honest read today says: **treat Edge AI as the likely path, and treat any
Physical AI outcome as a pleasant surprise the hedge work will still support if the Aug 9
go/no-go says otherwise.**

The good news, equally honestly stated: none of this blocks a *complete, legitimate submission*.
The GDN memory-scaling thesis (§1 of PLAN.md), the O(1)-state-vs-linear-KV-cache story, and the
"fast kernels don't exist for this silicon" contribution are all fully demonstrable on the
aarch64 hedge target with no dependency on either external gate. The minimum-viable-submission
bar in PLAN.md §8 is achievable on Edge AI alone. What is genuinely at risk is the *stretch*
value tied to Physical AI specifically — the NPU heterogeneous-dispatcher story and the
"optimized before broad tooling exists" WOW-factor framing that assumes real O6 hardware.

**Bottom line:** the deadline is very likely met for *a* submission. Whether that submission is
Physical AI (higher WOW-factor ceiling) or Edge AI (safe, fully controllable) depends entirely on
two maintainer-only actions landing in the next 3-5 days. Nothing agent-side is blocking either
outcome; everything portable is either done or in progress.
