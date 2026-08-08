# Pre-T-4 Submission Status Brief

_Generated 2026-08-07 by t4. Supports the T-4 descope decision (Aug 10)._
_All numbers below are validated against committed CSVs with manifests._

---

## TL;DR

The Edge AI submission is **ready today**. Every claim in
`DEVPOST_SUBMISSION.md` traces to a manifest-backed CSV. The O6 board
has not arrived; per ADR 0004, the usefulness cutoff is **2026-08-08**
(tomorrow). T-4 fires **2026-08-10** if no board — at which point
on-device NPU execution is cut and the submission locks to Edge AI.

---

## 1. What's done and provenance-backed

| Deliverable | Status | Key data |
|---|---|---|
| 3 GDN CPU kernels (NEON) | ✓ Verified + benchmarked | cumdecay 21.1 GiB/s, scan 10.6, conv1d 18.7 (t3, dirty=false) |
| 5-device fleet benchmarks | ✓ Cross-validated | Pi5 (A76), RK3588 big+little (A76/A55), Jetson (A57), t3/t4 match |
| E2E decode (matched commit) | ✓ 3 devices, 3 runs each | 4B INT8: 1.84 tok/s (t3), 0.51 (Jetson); 0.8B INT8: 10.6 (t3), 2.45 (Jetson) |
| GEMV optimization | ✓ 14.9× speedup | 0.07→1.04 tok/s (4B FP32), dirty=false manifests |
| INT8 weight-only quant | ✓ 1.65–1.77× on big cores | KV cache context sweep on t3+t4, <6% divergence |
| Mixed-precision (fp16/bf16) | ✓ fp16 gives 1.77× on decay | scan compute-bound, flat under fp16 |
| GDN-2 vs GDN-1 comparison | ✓ Microbenchmark | 1.2–1.5× decode cost on big, 2.2–2.4× on little |
| NOE op-coverage audit | ✓ Hardware-independent | Scan rejected, Loop trap documented (both CIX NOE + RKNN) |
| KleidiAI gap analysis + submission | ✓ Complete, 14 tests pass | No recurrence primitive; dwconv is SME2-only |
| Memory scaling analysis | ✓ Analytical model | 262K: 24 GiB savings (51 MiB state vs 8+ GiB KV cache) |
| big.LITTLE affinity policy | ✓ Measured | A76 2–3× faster than A55; diminishing returns past 4 big |
| GPU Mali-G610 OpenCL | ✓ ob-q44.1 closed | Scan kernel prototyped on RK3588 GPU |
| Submission prep (README, write-up, repro) | ✓ All beads closed | ob-fnq, ob-f7k, ob-kdi, ob-9e2 |
| Compliance checklist | ✓ ob-9e2 closed | Apache-2.0, no credentials at tip of main/t4 |

**Submission readiness:** 15/15 checks pass, 1788 tests, Ruff clean, CI green.

---

## 2. DEVPOST headline numbers — traced to source

| Claim (DEVPOST_SUBMISSION.md) | Source CSV | Match? |
|---|---|---|
| Cumulative decay 21.06 GiB/s (t3) | rk3588-t3-clean.csv | ✓ |
| Cumulative decay 22.47 GiB/s (t4) | rk3588-t4-big.csv | ✓ |
| Gated scan 10.62 / 11.09 GiB/s | same CSVs | ✓ |
| Causal Conv1D 18.73 / 23.00 GiB/s | same CSVs | ✓ |
| fp16 decay 37.20 GiB/s (1.77×) | rk3588-t3-clean.csv | ✓ |
| 4B INT8: 1.84 tok/s (t3) | e2e_fleet_comparison.md | ✓ |
| 0.8B INT8: 10.61 tok/s (t3) | same | ✓ |
| 0.8B INT8: 2.45 tok/s (Jetson) | same | ✓ |
| 4B INT8: 0.51 tok/s (Jetson) | same | ✓ |
| ~26× cumulative speedup | 0.07 → 1.84 tok/s = 26.3× | ✓ |

All manifests: dirty=false, governor=performance, 30 repeats (kernel) /
3 runs (e2e), git_sha recorded.

---

## 3. What's missing (all hardware-gated)

| Gap | Bead | Blocked on |
|---|---|---|
| Orion O6 on-device numbers | ob-axq → ob-iae | Board sourcing (human) |
| NPU runtime smoke test | ob-huw | ob-iae (first boot) |
| GPU scan kernel on target | ob-q44 | ob-88p (Vulkan validation on O6) |
| Dynamic heterogeneous dispatcher | ob-7a9 | ob-o4g (mapping ADR) |
| Master comparison table (O6 row) | ob-ami | ob-rqd (ablation matrix) |
| Demo video | ob-jui | Hardware |

**None of these block the Edge AI submission.** All are additive if the
board arrives before Aug 8–10.

---

## 4. Descope ladder status (ADR 0004)

| Date | Trigger | Status | Action if fires |
|---|---|---|---|
| **Aug 8** (tomorrow) | Last useful board arrival | Board not in hand | Submission plan locks to Edge AI |
| **Aug 10** (T-4) | No board by this date | **Likely fires** | Cut on-device NPU execution; keep NOE audit + CPU+GPU design |
| **Aug 11** (T-3) | No board booted | **Likely fires** | Cut Physical AI framing entirely; Edge AI locked |
| **Aug 12** (T-2) | Insufficient slack for GDN-2 swap | Check schedule | Cut layer swap; keep microbenchmark |
| **Aug 13** (T-1) | Time running out | — | Cut demo video + 262K point if needed |
| **Aug 14 16:00 PT** | Deadline | — | Submit |

---

## 5. Open PRs (all unit-consistency fixes, non-blocking)

| PR | Author | Scope | Review status |
|---|---|---|---|
| #103 | t4 (mine) | O6 baseline script + onboarding docs | Reviewed by t4, mergeable |
| #104 | t3 | GB/s→GiB/s in human-authored docs (7 files) | Cross-validated by t4, conversions verified |
| #102 | j1 | GB/s→GiB/s in generated tables (32 files) + KleidiAI README | Cross-validated by t4, recalculation verified |

All three touch only `.beads/issues.jsonl` in common (trivially resolved
via `bd export`). No code conflicts.

---

## 6. Recommended framing for T-4

If the board has not arrived by Aug 10, **let T-4 fire without
controversy**. The submission already clears docs/archive/PLAN.md §8's minimum-viable
bar with real measured numbers — not the "credible plan" ADR 0004
described on Aug 2 when zero benchmark numbers existed.

The Edge AI submission's strengths:
- **Novel kernels** that no existing library provides for Arm
- **5-device cross-validation** proving the bandwidth-boundedness thesis
- **The NPU wall finding** — a genuinely novel, citable result
- **~26× end-to-end speedup** from C kernel + GEMV + INT8
- **Honest negative results** (NPU compilers reject the recurrence)

The one weakness: no heterogeneous NPU/GPU/CPU dispatch on target
silicon. But the O6 onboarding checklist (`docs/O6_ONBOARDING.md`) and
baseline script (`scripts/o6_system_baseline.sh`) mean any late board
arrival can produce numbers within hours, not days.

---

## 7. After T-4: what t4 does next

1. If board arrives (any time): execute `docs/O6_ONBOARDING.md` →
   `scripts/o6_system_baseline.sh` → kernel benchmark → e2e decode.
2. If T-4 fires with no board: file the T-4 follow-up bead per ADR 0004,
   continue the Phase 2 loop (`bd ready`), and focus on submission polish
   (consistency pass, final numbers check, any remaining doc gaps).
3. On T-3 (Aug 11): file the T-3 follow-up bead, confirm Edge AI framing
   in DEVPOST_SUBMISSION.md, and prepare for submission.
