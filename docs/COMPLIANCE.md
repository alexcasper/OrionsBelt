# Devpost submission compliance checklist (ob-9e2)

**Status:** Active checklist · **Bead:** `ob-9e2` · **Deadline:** 2026-08-14 16:00 PT

Each item is individually capable of disqualifying an otherwise strong entry.
This checklist is the mechanical verification that nothing is missed.

---

## Hard requirements (Devpost rules)

| # | Requirement | Status | Evidence / action needed |
|---|---|---|---|
| 1 | **Public, open-source repo** | ✅ Done | `github.com/alexcasper/OrionsBelt` |
| 2 | **OSI-approved license** (MIT or Apache-2.0) | ✅ Done | `LICENSE` — Apache-2.0 |
| 3 | **License visible in GitHub About section** | ⚠️ Maintainer action | License file exists; maintainer must select "Apache-2.0" in repo Settings → General → "Choose a license" so it appears in the About sidebar. Cannot be done via API push. |
| 4 | **Project overview** (write-up section 1) | ✅ Present | `README.md` §"What Gated DeltaNet is, and why it matters on edge silicon" + §"The hybrid stack" |
| 5 | **Functionality / output** (write-up section 2) | ✅ Present | `README.md` §"What we are actually claiming" + §"Status" + `docs/METHODOLOGY.md` |
| 6 | **Setup instructions** (write-up section 3) | ✅ Present | `README.md` §"Reproducing" + `docs/DEVICE_RUNBOOK.md` + `scripts/fetch_weights.py` |
| 7 | **Built or significantly updated during submission period** | ✅ Done | All commits Aug 2, 2026 — within the submission window |
| 8 | **Demo video < 3 min** (optional) | ⏳ Future | Optional per rules; cut under T-1 descope (PLAN.md §7) |
| 9 | **NOTICE for third-party licenses** | ✅ Done | `NOTICE` — covers numpy, pytest, ruff, pandas, matplotlib, Qwen weights |
| 10 | **Weight license compliance** | ✅ Done | `docs/WEIGHTS_LICENSE.md` — both checkpoints Apache-2.0, not vendored |

---

## Track selection

| # | Item | Status | Notes |
|---|---|---|---|
| 11 | **Track declared** | ✅ Edge AI (primary hedge) | ADR 0002; O6 board may not arrive (PLAN.md §7, R1). Physical AI remains open if board arrives by Aug 8 (ADR 0004). |
| 12 | **Track-appropriate hardware** | ✅ Edge AI | Device fleet (ADR 0005): Pi5, RK3588, Jetson Nano. All Arm aarch64. |

---

## Technical deliverables

| # | Item | Status | Evidence |
|---|---|---|---|
| 13 | **Benchmark harness producing reproducible CSV** | ✅ Done | `bench/harness.py` (ob-ljh), frozen schema (ob-q9i) |
| 14 | **≥3 context lengths measured** | ✅ Done | Sweep covers 4K/32K/128K/262K (ob-ljh, ob-del) |
| 15 | **Measured before/after optimization** | ⏳ Pending | Needs x86 reference (ob-aqv) + optimized kernels (ob-8qt). Baseline kernels exist (ob-8qt.3). |
| 16 | **Memory decomposition (3-way split)** | ✅ Done | `bench/memory.py` (ob-vfp), `results/figures/memory_decomposition_qwen3.5-4b.png` (ob-9y8) |
| 17 | **Correctness oracle gates optimizations** | ⏳ Pending | Tolerances defined (docs/METHODOLOGY.md §4). Needs x86 reference (ob-aqv). |
| 18 | **Clean-clone reproduction rehearsal** | ⏳ Pending | ob-kdi — now unblocked by ob-aoo |
| 19 | **Arm Performix standardized report** | ⏳ Pending | ob-zzj — needs real hardware results |

---

## Remaining risks to compliance

1. **License in About section (#3):** requires maintainer with GitHub web UI access.
   This is a 30-second action but cannot be automated. **Must verify before submit.**

2. **Correctness oracle (#17):** the tolerances and methodology are defined, but
   the x86 reference model (ob-aqv) must run before any "before/after" claim is
   reportable. If the reference never lands, report baseline-only numbers honestly.

3. **Arm Performix (#19):** requires either the O6 or a device that Performix
   supports. If unavailable, report our own standardized CSVs with the methodology
   document (docs/METHODOLOGY.md) as the transparency layer.

---

## Pre-submission gate (ob-j7f)

Before clicking "Submit" on Devpost, verify:
- [ ] All ✅ items above are still ✅
- [ ] ⚠️ item #3 (license in About) is resolved
- [ ] README is current with latest results
- [ ] NOTICE and WEIGHTS_LICENSE.md are accurate
- [ ] At least one context sweep CSV + manifest is committed
- [ ] The repo clones cleanly from scratch (ob-kdi)
