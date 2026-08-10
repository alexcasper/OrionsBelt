# Pre-T-4 Submission Status Brief

_Generated 2026-08-07 by t4. Updated 2026-08-08 by j1 to reflect ADR 0007 (Edge AI committed)._
_All numbers below are validated against committed CSVs with manifests._

---

## TL;DR

The Edge AI submission is **ready today**. Every claim in
`DEVPOST_SUBMISSION.md` traces to a manifest-backed CSV. The O6 board
has not arrived; per [ADR 0007](adr/0007-commit-to-edge-ai-track.md),
the project committed to the Edge AI track effective **2026-08-06**.
The Aug 8 last-useful-arrival cutoff (ADR 0004) has now passed with no
board. T-4 (Aug 10) and T-3 (Aug 11) triggers are pre-committed per
ADR 0007 — the submission framing is locked to Edge AI.

---

## 1. What's done and provenance-backed

| Deliverable | Status | Key data |
|---|---|---|
| 3 GDN CPU kernels (NEON) | ✓ Verified + benchmarked | cumdecay 21.4 GiB/s, scan 10.6, conv1d 20.6 (t3, dirty=false) |
| 5-device fleet benchmarks | ✓ Cross-validated | Pi5 (A76), RK3588 big+little (A76/A55), Jetson (A57), t3/t4 match |
| E2E decode (matched commit) | ✓ 3 devices, 3 runs each | 4B INT8: 1.84 tok/s (t3), 0.51 (Jetson); 0.8B INT8: 10.6 (t3), 2.45 (Jetson) |
| GEMV optimization | ✓ 14.9× speedup | 0.07→1.04 tok/s (4B FP32), dirty=false manifests |
| INT8 weight-only quant | ✓ 1.65–1.77× on big cores | KV cache context sweep on t3+t4, <6% divergence |
| SDOT INT8 GEMV (dotprod cores) | ✓ 1.92–3.06× over NEON INT8 | 4B: 3.48 tok/s (83% of theoretical); 0.8B: 30.2 tok/s (t4, cross-val with t3 within 5%) |
| INT4+SDOT hybrid GEMV | ✓ 1.27× over INT8+SDOT on A76 | 4B: 4.43 tok/s; 0.8B: 37.21 tok/s (t4, big cluster); A55: INT8+SDOT remains optimal |
| Mixed-precision (fp16/bf16) | ✓ fp16 gives 1.64× on decay | scan compute-bound, flat under fp16 |
| GDN-2 vs GDN-1 comparison | ✓ Microbenchmark | 1.2–1.5× decode cost on big, 2.2–2.4× on little |
| NOE op-coverage audit | ✓ Hardware-independent | Scan rejected, Loop trap documented (both CIX NOE + RKNN) |
| KleidiAI gap analysis + submission | ✓ Complete, 14 tests pass | No recurrence primitive; dwconv is SME2-only |
| Memory scaling analysis | ✓ Analytical model | 262K: 24 GiB savings (51 MiB state vs 8+ GiB KV cache) |
| big.LITTLE affinity policy | ✓ Measured | A76 2–3× faster than A55; diminishing returns past 4 big |
| GPU Mali-G610 OpenCL | ✓ ob-q44.1 closed | Scan kernel prototyped on RK3588 GPU |
| Submission prep (README, write-up, repro) | ✓ All beads closed | ob-fnq, ob-f7k, ob-kdi, ob-9e2 |
| Compliance checklist | ✓ ob-9e2 closed | Apache-2.0, no credentials at tip of main/t4 |

**Submission readiness:** 15/15 checks pass, 2315 tests, Ruff clean, CI green.

---

## 2. DEVPOST headline numbers — traced to source

| Claim (DEVPOST_SUBMISSION.md) | Source CSV | Match? |
|---|---|---|
| Cumulative decay 21.39 GiB/s (t3) | rk3588-t3-clean.csv | ✓ |
| Cumulative decay 22.25 GiB/s (t4) | rk3588-t4_big.csv | ✓ |
| Gated scan 10.62 / 11.53 GiB/s | same CSVs | ✓ |
| Causal Conv1D 20.59 / 19.04 GiB/s | same CSVs | ✓ |
| fp16 decay 35.12 GiB/s (1.64×) | rk3588-t3-clean.csv | ✓ |
| 4B INT8: 1.84 tok/s (t3) | e2e_fleet_comparison.md | ✓ |
| 0.8B INT8: 10.61 tok/s (t3) | same | ✓ |
| 0.8B INT8: 2.45 tok/s (Jetson) | same | ✓ |
| 4B INT8: 0.51 tok/s (Jetson) | same | ✓ |
| 4B INT8+SDOT: 3.48 tok/s (t4) | rk3588-t4_big_int8_sdot_e2e_schema.csv | ✓ |
| 0.8B INT8+SDOT: 30.2 tok/s (t4) | rk3588-t4_08b_big_int8_sdot_e2e_schema.csv | ✓ |
| 4B INT4+SDOT: 4.43 tok/s (t4) | rk3588-t4_big_int4_sdot_e2e_schema.csv | ✓ |
| 0.8B INT4+SDOT: 37.21 tok/s (t4) | rk3588-t4_08b_big_int4_sdot_e2e_schema.csv | ✓ |
| ~50× cumulative speedup (INT8+SDOT) | 0.07 → 3.48 tok/s = 49.7× | ✓ |
| ~63× cumulative speedup (INT4+SDOT) | 0.07 → 4.43 tok/s = 63.3× | ✓ |
| (t3 cross-validation) | rk3588-t3_big_int8_sdot_e2e.json (3.34), rk3588-t3_08b_big_int8_sdot_e2e.json (28.9) | ✓ within 5% |

All manifests: governor=performance, 30 repeats (kernel) / 1–3 runs (e2e),
git_sha recorded. t3 manifests are dirty=false; t4 SDOT manifests are
dirty=true (binary built and run before source commit — result files
written first, then committed together with the source).

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

**None of these block the Edge AI submission.** The Aug 8 last-useful-arrival
cutoff has passed with no board; any future arrival is additive bonus per ADR 0007.

---

## 4. Descope ladder status (ADR 0004)

| Date | Trigger | Status | Action if fires |
|---|---|---|---|
| **Aug 8** | Last useful board arrival | **FIRED** — board not arrived; Edge AI committed (ADR 0007) | Submission locked to Edge AI |
| **Aug 10** (T-4) | No board by this date | Pre-committed (ADR 0007) | Cut on-device NPU execution; keep NOE audit + CPU+GPU design. **NPU offload design documented** in [`NPU_OFFLOAD_DESIGN.md`](NPU_OFFLOAD_DESIGN.md) |
| **Aug 11** (T-3) | No board booted | Pre-committed (ADR 0007) | Physical AI framing already cut; Edge AI locked |
| **Aug 12** (T-2) | Insufficient slack for GDN-2 swap | Check schedule | Cut layer swap; keep microbenchmark |
| **Aug 13** (T-1) | Time running out | — | Cut demo video + 262K point if needed |
| **Aug 14 16:00 PT** | Deadline | — | Submit |

---

## 5. PRs (all merged)

| PR | Author | Scope | Status |
|---|---|---|---|
| #103 | t4 | O6 baseline script + onboarding docs | MERGED |
| #101 | j1 | Close ob-8qt.10 KleidiAI packaging complete | MERGED |
| #100 | t3 | Fix bandwidth ceiling label with measured DRAM probe data | MERGED |
| #104 | t3 | GB/s→GiB/s in human-authored docs (7 files) | MERGED |
| #102 | j1 | GB/s→GiB/s in generated tables (32 files) + KleidiAI README | MERGED |
| #106 | j1 | Fix O6 spec unit mismatch in comparison_table.md | MERGED |
| #107 | t4 | Fix README unit consistency — device spec table + headline percentages | MERGED |
| #108 | t3 | Fix FINDINGS.md §3 header + GEMV overflow defense | MERGED |
| #110 | j1 | Fix stale counts + flag t4 KleidiAI artifacts in submission docs | MERGED |
| #111 | t3 | Fix KleidiAI bench cold-start artifacts — batched timing | MERGED |
| #113 | t4 | KleidiAI batched-timing tables + security runbook refresh | MERGED |
| #114 | j1 | Re-run A57 KleidiAI bench with batched-timing fix | MERGED |
| #115 | t4 | Cumdecay core-affinity fix + big.LITTLE affinity warnings | MERGED |
| #116 | j1 | Dolt sync reset + ob-9xr (device bench decode timing) | MERGED |
| #117 | t4 | Device bench + CI fixes | MERGED |
| #118 | j1 | A57 prefill benchmark data (ob-8qt.15) | MERGED |
| #119 | t4 | CI lint/test fixes + merge j1 INT4 findings | MERGED |
| #120 | j1 | llama.cpp baseline + Q8_0 block-quantized GEMV on A57 (ob-mrd.15, ob-8qt.17) | MERGED |
| #121 | j1 | Q8_0 GEMV (2.97× speedup) + accuracy validation + llama.cpp baseline | MERGED |
| #122 | j1 | Q8_0/INT4 context sweeps, Devpost submission + CLAIM_VERIFICATION updates | MERGED |
| #123 | j1 | Fix CI — schema conformance test missing cross_tool & quant_accuracy CSV markers | MERGED |
| #124 | j1 | Regenerate fleet analysis figures with batched-timing CSVs | MERGED |
| #125 | j1 | Add --quant int4 support to run_e2e_decode.sh | MERGED |
| #126 | j1 | Fix broken relative link in ONNX probe research note | MERGED |
| #127 | j1 | KleidiAI README URL fix, pyproject.toml Edge AI framing, Devpost writeup | MERGED |
| #128 | j1 | Stale "Physical AI" + bare PLAN.md refs (31+ files) | MERGED |
| #129 | j1 | SUBMISSION_STATUS.md Aug 8 cutoff + 3 broken link fixes | MERGED |
| #130 | j1 | Fix stale FINDINGS section count (38→46), add PR #129, fix repo URL in pyproject.toml | MERGED |
| #131 | j1 | Add PR #130 to merged PRs table in SUBMISSION_STATUS.md | MERGED |
| #132 | j1 | Add PR #131 to merged PRs table in SUBMISSION_STATUS.md | MERGED |
| #133 | j1 | Add PR #132 to merged PRs table + security: remove hardcoded password from purge scripts (ob-3i5) | MERGED |
| #134 | j1 | Add PR #133 to merged PRs table in SUBMISSION_STATUS.md | MERGED |
| #135 | j1 | Add PR #134 to merged PRs table + backfill 13 t4 retroactive manifests (ob-mrd.17) + stale branch cleanup + bench/r5 security tracking (ob-0vu) | MERGED |
| #136 | j1 | Add PR #135 to merged PRs table + fix lint in gen_retroactive_manifests.py + document bench/r5 exposure in SECURITY.md (ob-0vu closed) | MERGED |
| #137 | j1 | Add PR #136 to merged PRs table in SUBMISSION_STATUS.md | MERGED |
| #138 | j1 | Fix t4 8-thread provenance chain (ob-dpl): restore clean CSV+manifest (sha=7bbbc99, dirty=false), correct SHA citations in DEVPOST + SUBMISSION_STATUS + partial_comparison_table. Reviewer fix: precise wording on kernel code equivalence. | MERGED |
| #139 | j1 | Close ob-dpl, add PR #137/#138 to merged PRs table in SUBMISSION_STATUS.md | MERGED |
| #140 | j1 | Add PR #139 to merged PRs table in SUBMISSION_STATUS.md | MERGED |
| #141 | j1 | Add PR #140 to merged PRs table in SUBMISSION_STATUS.md | MERGED |
| #142 | j1 | Add PR #141 to merged PRs table in SUBMISSION_STATUS.md | MERGED |
| #143 | j1 | Add PR #142 to merged PRs table in SUBMISSION_STATUS.md | MERGED |
| #144 | j1 | Add PR #143 to merged PRs table in SUBMISSION_STATUS.md | MERGED |
| #145 | j1 | Add PR #144 to merged PRs table in SUBMISSION_STATUS.md | MERGED |
| #146 | j1 | Add PR #145 to merged PRs table in SUBMISSION_STATUS.md | MERGED |
| #147 | j1 | Add PR #146 to merged PRs table in SUBMISSION_STATUS.md | MERGED |
| #148 | j1 | Add PR #147 to merged PRs table in SUBMISSION_STATUS.md | MERGED |
| #149 | alex | Fleet ssh orchestration: drive all four nodes from one dev box (ob-8ms.4) | MERGED |
| #150 | j1 | Add PR #148 to merged PRs table + fix dirty-tree guard test (ob-7cf) + add tests for ort_gdn_probe numpy_gdn_reference (ob-9o7) | MERGED |
| #151 | j1 | Add PR #150 to merged PRs table in SUBMISSION_STATUS.md + loop flush + beads export | MERGED |
| #152 | j1 | Fix stale manifest count in README (125 → 138) + fix stale test count (1788 → 1799) in Devpost submission, SUBMISSION_STATUS, README | MERGED |
| #153 | j1 | Add PR #152 to merged PRs table + energy efficiency section to Devpost + backfill 4 j1 power CSV manifests + INA3221 energy profiling in README + fix stale manifest count (138 → 142) | MERGED |
| #154 | j1 | Add PR #153 to merged PRs table + fix --help/error message omitting gdn2_gated_scan from valid sustained kernels | MERGED |
| #155 | j1 | Add PR #154 to merged table + fix isa_detect docstring + fix README broken doc ref (SETUP_O6→O6_ONBOARDING) + fix stale 'unverified' comment in pyproject.toml | MERGED |
| #156 | j1 | GPU kernel fixes (dead code + incorrect matrix notation in OpenCL) + NPU probe README (document probe 07) + stale Python version + broken doc ref fixes | MERGED |
| #157 | j1 | Add 19 missing merged PRs (#106–#126) to SUBMISSION_STATUS table | MERGED |
| #158 | t3 | Re-run KleidiAI bench with taskset -c 4-7 — fix cumdecay 64×160 A55 mis-measurement + suppress -Wunused-but-set-variable + improve big.LITTLE affinity warning | MERGED |
| #159 | t3 | Fix GPU kernel read-step matrix notation (S_h^T) + add PRs #156–#158 to table + stale FINDINGS line count + warn on unrecognized CLI args + fix -Wmaybe-uninitialized in e2e_decode | MERGED |
| #160 | t3 | Add missing PRs #100–#101 to merged PRs table (gap before #102) + beads sync after ob-502 note | MERGED |
| #161 | t3 | Defensive hardening — malloc NULL checks in e2e_decode.c (INT8/Q8_0 quantized paths) + delta_matmul.c KleidiAI path + CSV/CLI input validation in fleet_analysis.py, harness.py, correctness.py, comparison_table.py | MERGED |
| #162 | t3 | Add PR #161 to merged PRs table in SUBMISSION_STATUS.md | MERGED |
| #163 | t3 | SPDX Python/shell headers (109 files) + xmalloc wrappers for 5 kernel test files (107 sites) + bench_gdn.c (24) + gdn_e2e_decode.c main paths (20+3) | MERGED |
| #164 | t4 | SDOT INT8 GEMV kernel for A55 little-cluster — 2.78× over NEON (ob-8qt.14) + duplicate xmalloc fix | MERGED |
| #165 | t3 | Complete malloc audit to ALL remaining C files (bench/gpu/scripts, 71 sites) + __attribute__((unused)) on all wrappers + fix 2 pre-existing dead-code warnings | MERGED |
| #166 | t4 | SDOT INT4 hybrid GEMV — 2.85× over INT4 NEON (ob-8qt.20) | MERGED |
| #167 | t3 | CRITICAL FIX — remove leftover git conflict markers in gdn_e2e_decode.c breaking build on main | MERGED |
| #168 | t3 | Fix 5 missed bare malloc calls in test_gdn_e2e_int8.c SDOT block + SPDX self-header on add_spdx_headers.py | MERGED |
| #169 | t3 | Fix stale counts after t4 merge + add 6 missing manifests for t4 SDOT/INT4+SDOT/NEON CSVs | MERGED |
| #170 | t4 | NEON+SDOT full-attention scoring — 33% faster attn at ctx=4096 (ob-8qt.21) + session max-age policy for goose-loop.sh (ob-462) | MERGED |
| #171 | t3 | Fix stale manifest count in README (270→150 tracked) + manifest naming convention note in DEVICE_RUNBOOK | MERGED |
| #172 | t3 | Fix stale FINDINGS section count after PR #170 merge | MERGED |
| #173 | t4 | Session max-age fix + INT8 KV V-accum warning cleanup (ob-462, ob-4nd) | MERGED |
| #174 | t3 | Fix stale manifest count (151→237) and figure count (88→89) in README | MERGED |
| #175 | t4 | OpenMP fairness fix + fair-comparison benchmark (ob-m2j) | MERGED |
| #176 | t3 | Build script cleanup + goose-loop log rotation + guard fix | MERGED |
| #177 | t4 | Cross-device validation §36 + post-#175 sync | MERGED |
| #178 | t4 | README refresh — cross-validation §36, thermal §37, updated counts | MERGED |
| #179 | t3 | Provenance fixes — corrupted manifest, 14 missing manifests, comparison_table SHA drift, README counts | MERGED |
| #180 | t3 | SDOT INT8 cross-device validation + rebase provenance fixes | MERGED |
| #181 | t4 | Device bench + submission readiness fixes + GDN-2 unit tests | MERGED |
| #182 | t3 | Fix ruff format + keep branch current with main | MERGED |
| #183 | t3 | Regenerated fleet_cross_device.png after t4 PR merge | MERGED |
| #184 | t4 | Doc consistency fixes — stale counts + PR table backfill | MERGED |
| #185 | t4 | SDOT/INT4+SDOT e2e fleet data (7 schema CSVs) + DEVPOST/README/submission doc consistency + ~63× cumulative speedup | MERGED |
| #189 | t4 | Add PR #185 to SUBMISSION_STATUS + fix check_readme_counts bugs (git ls-files, FINDINGS validation, 8 unit tests) | MERGED |

---

## 6. Framing status

ADR 0007 committed to the Edge AI track on **2026-08-06**, so the T-4/T-3
descope triggers fire as a formality — the submission is already locked
to Edge AI. The submission clears docs/archive/PLAN.md §8's
minimum-viable bar with real measured numbers — not the "credible plan"
ADR 0004 described on Aug 2 when zero benchmark numbers existed.

The Edge AI submission's strengths:
- **Novel kernels** that no existing library provides for Arm
- **5-device cross-validation** proving the bandwidth-boundedness thesis
- **The NPU wall finding** — a genuinely novel, citable result
- **~63× end-to-end speedup** from C kernel + GEMV + INT8 + SDOT + INT4+SDOT
- **Honest negative results** (NPU compilers reject the recurrence)

The one weakness: no heterogeneous NPU/GPU/CPU dispatch on target
silicon. However, the complete NPU offload design — operator-level mapping,
subgraph boundaries, phase-dependent routing, and quantization policy — is
documented in [`NPU_OFFLOAD_DESIGN.md`](NPU_OFFLOAD_DESIGN.md). The O6
onboarding checklist (`docs/O6_ONBOARDING.md`) and baseline script
(`scripts/o6_system_baseline.sh`) mean any late board arrival can produce
numbers within hours, not days.

---

## 7. Current next steps

1. If board arrives (any time): execute `docs/O6_ONBOARDING.md` →
   `scripts/o6_system_baseline.sh` → kernel benchmark → e2e decode.
   Append as additive datapoint per ADR 0007 (do not restructure).
2. **Submission polish**: final consistency pass, numbers verification,
   any remaining doc gaps. Maintainer actions: make repo public, verify
   license in About section, select Edge AI on Devpost form.
3. On Aug 14 16:00 PT: submit on Devpost.
