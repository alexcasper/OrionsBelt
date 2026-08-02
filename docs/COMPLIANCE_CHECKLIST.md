# Submission compliance checklist

Bead: `ob-9e2`. Mechanical verification that the repository meets every hard
submission requirement for the [Arm Create: AI Optimization Challenge](https://arm-ai-optimization-challenge.devpost.com/rules)
(deadline 2026-08-14 16:00 PT). Each item can independently disqualify an entry.

Last checked: 2026-08-02.

---

## 1. Track selection

| Requirement | Status | Evidence |
|---|---|---|
| Choose one: Physical AI, Cloud AI, or Mobile AI | **READY (conditional)** | Primary target: Physical AI (Orion O6). Fallback: Edge AI (portable aarch64 hedge). Final go/no-go decision scheduled 2026-08-09 per PLAN.md §5. |

**Action needed:** Confirm final track at the Aug 9 decision point. The hedge
track is ready as a fallback if the O6 board does not arrive.

## 2. Public repository with OSI license

| Requirement | Status | Evidence |
|---|---|---|
| Public repository | **PASS** | https://github.com/alexcasper/OrionsBelt — public |
| OSI license (MIT or Apache 2.0) | **PASS** | Apache-2.0 ([LICENSE](../LICENSE)) |
| License visible in GitHub "About" section | **PASS** | GitHub detects license as "Apache License 2.0" from LICENSE file |
| Repository description set | **PASS** | Set 2026-08-02: "Optimizing Qwen3.5 Gated DeltaNet hybrid model for Arm edge silicon (Orion O6 / CIX P1)" |
| Repository topics set | **PASS** | arm, ai-optimization, gated-deltanet, qwen, edge-ai, physical-ai, npu, cix, orion-o6 |

## 3. Write-up sections

The challenge requires three write-up sections. Current status:

| Section | Status | Location | Gap |
|---|---|---|---|
| **Project overview** (purpose, what it does) | **PASS** | [README.md](../README.md) §"What Gated DeltaNet is", §"The hybrid stack", §"The gap this project fills" | None |
| **Functionality/output** (what it produces) | **PARTIAL** | [README.md](../README.md) §"Status", [docs/FINDINGS.md](FINDINGS.md) | No measured benchmark numbers yet. Placeholder results table in README. Must populate before submission. |
| **Setup instructions** (reproducibility) | **PARTIAL** | [README.md](../README.md) §"Reproducing", [CONTRIBUTING.md](../CONTRIBUTING.md), [scripts/README.md](../scripts/README.md) | `docs/SETUP_O6.md` and `docs/SETUP_PORTABLE.md` are pending. Full clean-clone reproduction path must be verified by rehearsal. |

**Action needed:**
- Populate the results table with measured numbers (blocked on hardware/correctness oracle).
- Write `docs/SETUP_O6.md` (Orion O6 bring-up) or `docs/SETUP_PORTABLE.md` (hedge target).
- Run a clean-clone reproduction rehearsal before submission.

## 4. Demo video (optional)

| Requirement | Status | Evidence |
|---|---|---|
| Under 3 minutes, shows device functioning | **NOT STARTED** | Optional but "helps judges substantially" per challenge rules |
| Hosted on YouTube/Vimeo/Youku | **NOT STARTED** | — |

**Action needed:** Script and record after benchmark numbers are available.
Link from README and Devpost submission. Lower priority than hard requirements.

## 5. Benchmarking with Arm Performix

| Requirement | Status | Evidence |
|---|---|---|
| Use Arm Performix for standardized results | **NOT STARTED** | Referenced in PLAN.md and README. No integration yet. |

**Action needed:** Integrate Arm Performix for reporting standardized results.
This is a hard requirement — results without Performix may not count.
Tracked via the benchmark harness epic (ob-mrd).

## 6. Deadline

| Requirement | Status | Evidence |
|---|---|---|
| Submit before 2026-08-14 16:00 PT | **ON TRACK** | Current date: 2026-08-02. 12 days remaining. |

**Risk:** Hardware acquisition (Orion O6) and CIX SDK access are externally
gated. If not resolved by Aug 9, submit with Edge AI hedge track results.
See [docs/RISK_REGISTER.md](RISK_REGISTER.md) for the daily go/no-go checklist.

## 7. Newness

| Requirement | Status | Evidence |
|---|---|---|
| Newly built or significantly updated during submission period | **PASS** | All code, docs, and artifacts created 2026-08-01/02. No prior submissions of this project exist. |

## 8. Third-party licenses and NOTICE

| Requirement | Status | Evidence |
|---|---|---|
| NOTICE file for dependencies and weights | **PASS** | [NOTICE](../NOTICE) — lists all Python dependencies, model checkpoints, and Arm tooling |
| Model weights license recorded | **PASS** | [docs/MODEL_LICENSES.md](MODEL_LICENSES.md) — Apache-2.0 for both Qwen3.5-4B and 0.8B, attribution requirements documented |
| Weights not vendored in repo | **PASS** | `models/` is in `.gitignore`; `scripts/fetch_weights.py` downloads at setup time |
| Quantization modifications documented | **PASS** | [ADR 0006](adr/0006-quantization-policy.md) records what is changed from the original checkpoint (Apache-2.0 §4 "state changes" obligation) |

## 9. Devpost submission

| Requirement | Status | Evidence |
|---|---|---|
| Submit on Devpost before deadline | **NOT DONE** | Tracked as ob-j7f. Must be done after all hard requirements pass. |

---

## Summary

| Category | Hard requirements | Status |
|---|---|---|
| Track selection | 1 | Conditional (Aug 9 decision) |
| Repository + license | 5 | **5/5 PASS** |
| Write-up sections | 3 | 1 pass, 2 partial |
| Demo video | 2 | Optional (not started) |
| Benchmarking | 1 | Not started |
| Deadline | 1 | On track |
| Newness | 1 | **PASS** |
| Licenses + NOTICE | 4 | **4/4 PASS** |
| Devpost submission | 1 | Not done |

**Blocking items for submission (must resolve):**
1. Write `SETUP_O6.md` or `SETUP_PORTABLE.md` and run reproduction rehearsal
2. Populate results table with measured numbers (blocked on hardware/oracle)
3. Integrate Arm Performix reporting
4. Confirm track selection (Aug 9 decision point)
5. Submit on Devpost (ob-j7f)

**Non-blocking but high-value:**
- Demo video
- Clean-clone reproduction rehearsal
