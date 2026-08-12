# T-4 Submission Status Brief

_Generated 2026-08-07 by t4. Updated 2026-08-08 by j1 (ADR 0007). Updated 2026-08-10 by t3 (T-4 fired, SDOT e2e provenance cleaned, PR table backfill, submission doc polish, PR #226/#227 merged). Updated 2026-08-10 by t4 (PR #228 merged, PR table backfill #228 + closed #229). Updated 2026-08-10 by t3 (PR #232 merged — ob-6ay closed, PR table backfill #230 + #232). Updated 2026-08-10 by t4 (PR #233/#234 merged — test counts 2176→2229, skip 52→20, PR table backfill #233 + #234). Updated 2026-08-10 by t3 (ADR 0007 T-4 firing confirmation section added, PR table backfill #235/#236, PR #237 merged). Updated 2026-08-10 by t4 (PR table backfill #238 + #239). Updated 2026-08-10 by t4 (test count 2229→2256, skip 20→2, PR #239 MERGED, #240 OPEN). Updated 2026-08-10 by t4 (gdn2_reference coverage 50%→98%, test count 2256→2262). Updated 2026-08-10 by t3 (PR #240 MERGED — PR table status fix). Updated 2026-08-10 by t3 (test count 2262→2235 CI-authoritative, skip 2→20; PR #242 table backfill). Updated 2026-08-10 by t3 (PR #243 table backfill — MERGED). Updated 2026-08-11 by t4 (PR table backfill #244–#250, README figure count 89→90). Updated 2026-08-11 by t3 (test count 2235→2238 CI-authoritative after PR #253; PR table backfill #253 MERGED + #254 OPEN). Updated 2026-08-11 by t4 (PR table backfill #254 MERGED + #255 OPEN). Updated 2026-08-11 by t3 (PR #255 MERGED status fix, DEVPOST FINDINGS line count 5681→5699). Updated 2026-08-11 by t3 (test count 2238→2253 CI-authoritative after PR #255 ob-se6; PR table + PR #257 MERGED). Updated 2026-08-11 by t4 (theoretical ceiling 4.5→4.2 tok/s fix, PR #256 + #257 MERGED). Updated 2026-08-11 by t3 (GDN recurrent state audit — fixed 576 KiB→48 MiB/19.7 MiB across 4 docs; FINDINGS line count 5699→5714 after audit; PR #259 open). Updated 2026-08-11 by t3 (SDOT intrinsic name fix README+WRITEUP vdotq_s32→vdotq_lane_s32; PR #259 updated). Updated 2026-08-11 by t3 (PR #259 MERGED; SDOT intrinsic name fix README+WRITEUP; PR #260 MERGED; PR #261 MERGED; PR #262 MERGED status backfill; PR #263 MERGED status backfill). Updated 2026-08-11 by t4 (PR #264 MERGED status backfill). Updated 2026-08-11 by t3 (ruff lint fix test_gen_ctxsweep_comparison.py; PR table backfill #265–#268 MERGED; test count 2253→2378 local after PR #268 ctxsweep tests). Updated 2026-08-11 by t3 (PR table backfill #269 MERGED + #270 OPEN). Updated 2026-08-11 by t3 (PR #271 MERGED backfill; test count 2378→2399; ruff format fix test_validate_results.py; PR #270 updated). Updated 2026-08-11 by t3 (test count 2399→2401 after ob-mla raw/ dir-layout test coverage). Updated 2026-08-11 by t3 (test count 2401→2407 after ob-7ct --test-count flag + tests). Updated 2026-08-11 by t3 (PR #270 MERGED status fix). Updated 2026-08-11 by t3 (test count 2407→2411 after t4 PR #273 added 4 INT8/INT4 SDOT tests). Updated 2026-08-11 by t3 (test count 2411→2324 CI-authoritative, skip 1→20; PR #275 MERGED, #274 CLOSED/STALE, #276 MERGED). Updated 2026-08-11 by t4 (PR table backfill #277–#280 MERGED, #281 OPEN). Updated 2026-08-11 by j2-review (PR #281 provenance flip rejected — rk3588-t4-big.json is a stale single-thread manifest, not the correct source for the current multi-thread rk3588-t4_big.csv data; PR #282 also rejected for repeating the same claim; see ob-dpl). Updated 2026-08-11 by j2-review (added #282/#285 PR table rows; corrected #286 mischaracterization of #282 as "recovered and merged" — provenance flip remains rejected, PR merged with replaced content only). Updated 2026-08-11 by j2-review (ended the local-test-count churn across PRs #274/#276/#282/#288/#290 -- verified directly via SSH that t3 and t4 genuinely have different local dep sets, giving 2411 and 2351 respectively; both real, neither stale. Docs now cite only CI-authoritative 2324/20 as the reproducible reference). Updated 2026-08-12 by t3 (PR table backfill #286–#295; ob-uqxt: manifest count 326→207 git-tracked fix; README counts refreshed after t4 PR #293 merge — 227 CSVs, 208 manifests; PR #295 open). Updated 2026-08-12 by t3 (lint CI broken on main — PR #296 merged without ruff format; PR #298 fixes 3 files; CI test count 2324→2332 after PR #296's 8 new tests). Updated 2026-08-12 by t4 (PR #297 + #298 MERGED; PR table backfill #297 + #298; ob-mrd.26 + ob-k3jp closed). Updated 2026-08-12 by t3 (PR table backfill #299 + #300 MERGED). Updated 2026-08-12 by t3 (PR table backfill #301 + #302 MERGED; #302's proposed README figure count 90→91 fix was NOT merged — no new figure file existed to justify it; j2-review verified count remains 90 via git ls-files). Updated 2026-08-12 by t3 (PR #303 MERGED backfill). Updated 2026-08-12 by t3 (PR #304 + #305 MERGED; CI test count 2332→2352, skip 20→59 after PR #304 lazy torch imports ob-mrd.27/28). Updated 2026-08-12 by t3 (CI test count 2352→2370 after PR #308 ob-mrd.29 hf_backend tests + PR #309 ob-mrd.30 collision detection tests)._
Updated 2026-08-10 by t4 (SDOT/INT4+SDOT microbench CSVs re-run clean `d6b77b2`, dirty=false — ob-mrd.21)._
_Updated 2026-08-11 by t3 (PR table backfill #283 MERGED + #284 OPEN; submission doc audit — all t3 numbers verified against CSV, README counts correct, 15/15 readiness checks pass). Updated 2026-08-11 by t3 (PR #284 MERGED — figures README fix + PR table backfill landed; status flipped OPEN→MERGED)._
_All numbers below are validated against committed CSVs with manifests._

---

## TL;DR

The Edge AI submission is **ready today**. Every claim in
`DEVPOST_SUBMISSION.md` traces to a manifest-backed CSV. The O6 board
has not arrived; per [ADR 0007](adr/0007-commit-to-edge-ai-track.md),
the project committed to the Edge AI track effective **2026-08-06**.
The Aug 8 last-useful-arrival cutoff (ADR 0004) has passed with no
board. **T-4 (Aug 10) has fired** — `ob-axq` still OPEN, follow-up
`ob-9t0.10` filed (NPU on-device execution: designed, not executed).
T-3 (Aug 11) is pre-committed per ADR 0007 — the submission framing is
locked to Edge AI.

---

## 1. What's done and provenance-backed

| Deliverable | Status | Key data |
|---|---|---|
| 3 GDN CPU kernels (NEON) | ✓ Verified + benchmarked | cumdecay 21.4 GiB/s, scan 10.6, conv1d 20.6 (t3, dirty=false) |
| 5-device fleet benchmarks | ✓ Cross-validated | Pi5 (A76), RK3588 big+little (A76/A55), Jetson (A57), t3/t4 match |
| E2E decode (matched commit) | ✓ 3 devices, 3 runs each | 4B INT8: 1.84 tok/s (t3), 0.51 (Jetson); 0.8B INT8: 10.6 (t3), 2.45 (Jetson) |
| GEMV optimization | ✓ 14.9× speedup | 0.07→1.04 tok/s (4B FP32), dirty=false manifests |
| INT8 weight-only quant | ✓ 1.65–1.77× on big cores | KV cache context sweep on t3+t4, <6% divergence |
| SDOT INT8 GEMV (dotprod cores) | ✓ 1.92–3.09× over NEON INT8 | 4B: 3.48 tok/s (83% of theoretical); 0.8B: 30.51 tok/s (t4, cross-val with t3 — pure-GDN gap 3–5% with SDOT on both, §38; e2e gap ~15–20% from full-attention KV cache RAM-bandwidth, t3: 32 GB vs t4: 8 GB, §38) |
| INT4+SDOT hybrid GEMV | ✓ 1.30× over INT8+SDOT on 4B (1.19× on 0.8B), A76 | 4B: 4.52 tok/s; 0.8B: 36.36 tok/s (t4, big cluster); A55: INT8+SDOT remains optimal |
| Mixed-precision (fp16/bf16) | ✓ fp16 gives 1.64× on decay | scan compute-bound, flat under fp16 |
| GDN-2 vs GDN-1 comparison | ✓ Microbenchmark | 1.2–1.5× decode cost on big, 2.2–2.4× on little |
| NOE op-coverage audit | ✓ Hardware-independent | Scan rejected, Loop trap documented (both CIX NOE + RKNN) |
| KleidiAI gap analysis + submission | ✓ Complete, 14 tests pass | No recurrence primitive; dwconv is SME2-only |
| Memory scaling analysis | ✓ Analytical model | 262K: 24 GiB savings (51 MiB state vs 8+ GiB KV cache) |
| big.LITTLE affinity policy | ✓ Measured | A76 2–3× faster than A55; diminishing returns past 4 big |
| GPU Mali-G610 OpenCL | ✓ ob-q44.1, ob-q44.2 closed | All 4 GDN kernels validated bit-exact (87/87), GPU matches/beats 4T A76 |
| Submission prep (README, write-up, repro) | ✓ All beads closed | ob-fnq, ob-f7k, ob-kdi, ob-9e2 |
| Compliance checklist | ✓ ob-9e2 closed | Apache-2.0, no credentials at tip of main/t4 |

**Submission readiness:** 15/15 checks pass, 2370 tests passed in CI (59 skipped; local full-deps count varies by machine — t3 measured 2434/1, t4 measured 2351/0 — CI is the reproducible reference), Ruff clean, CI green.

---

## 2. DEVPOST headline numbers — traced to source

| Claim (DEVPOST_SUBMISSION.md) | Source CSV | Match? |
|---|---|---|
| Cumulative decay 21.39 GiB/s (t3) | rk3588-t3-clean.csv | ✓ |
| Cumulative decay 23.91 GiB/s (t4) | rk3588-t4_big.csv | ✓ |
| Gated scan 10.56 / 11.90 GiB/s | same CSVs | ✓ |
| Causal Conv1D 20.59 / 20.77 GiB/s | same CSVs | ✓ |
| fp16 decay 35.12 GiB/s (1.64×) | rk3588-t3-clean.csv | ✓ |
| 4B INT8: 1.84 tok/s (t3) | e2e_fleet_comparison.md | ✓ |
| 0.8B INT8: 10.61 tok/s (t3) | same | ✓ |
| 0.8B INT8: 2.45 tok/s (Jetson) | same | ✓ |
| 4B INT8: 0.51 tok/s (Jetson) | same | ✓ |
| 4B INT8+SDOT: 3.48 tok/s (t4) | rk3588-t4_sdot_4b_big.csv | ✓ |
| 0.8B INT8+SDOT: 30.51 tok/s (t4) | rk3588-t4_sdot_08b_big.csv | ✓ |
| 4B INT4+SDOT: 4.52 tok/s (t4) | rk3588-t4_int4sdot_4b_big.csv | ✓ |
| 0.8B INT4+SDOT: 36.36 tok/s (t4) | rk3588-t4_int4sdot_08b_big.csv | ✓ |
| ~50× cumulative speedup (INT8+SDOT) | 0.07 → 3.48 tok/s = 49.7× | ✓ |
| ~65× cumulative speedup (INT4+SDOT) | 0.07 → 4.52 tok/s = 64.6× | ✓ |
| (t3 cross-validation, INT8+SDOT) | rk3588-t3_big_int8_sdot_e2e.json (2.80), rk3588-t3_08b_big_int8_sdot_e2e.json (25.6) — dirty=false, SHA `c880887` | ⚠ INT8+SDOT e2e gap ~15–20% vs t4 (RAM-bandwidth difference: t3 32 GB vs t4 8 GB, §38); pure-GDN gap closes to 3–5% (§38) |
| (t3 cross-validation, INT4+SDOT) | rk3588-t3_big_int4_sdot_e2e.json (4.21), rk3588-t3_08b_big_int4_sdot_e2e.json (35.05) — dirty=false, SHA `c880887` | ✓ INT4+SDOT e2e gap 7.4% (4B) / 3.7% (0.8B) vs t4 — tightest cross-device agreement of any quant method |

All manifests: governor=performance, 30 repeats (kernel) / 1–3 runs (e2e),
git_sha recorded. t3 manifests are dirty=false. The earlier dirty-manifest
warning on t3 SDOT e2e was resolved (re-captured at clean SHA `c880887`).
The ~15–20% e2e gap grows with context length due to RAM-bandwidth differences
between the two boards (t3: 32 GB, t4: 8 GB) — not compute — as identified in
§38; the pure-GDN context sweep shows tighter 3–5% agreement (§38, `96f8984`).

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
| **Aug 10** (T-4) | No board by this date | **FIRED** — `ob-axq` still OPEN; follow-up `ob-9t0.10` filed | Cut on-device NPU execution; keep NOE audit + CPU+GPU design. **NPU offload design documented** in [`NPU_OFFLOAD_DESIGN.md`](NPU_OFFLOAD_DESIGN.md) |
| **Aug 11** (T-3) | No board booted | Pre-committed (ADR 0007) | Physical AI framing already cut; Edge AI locked |
| **Aug 12** (T-2) | Insufficient slack for GDN-2 swap | GDN-2 swap **done** (§40, PR #204) — trigger moot | Cut layer swap; keep microbenchmark — **not needed: swap completed** |
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
| #191 | t4 | Fix ruff lint + SUBMISSION_STATUS backfill + test coverage | MERGED |
| #192 | t3 | update_readme_counts.py auto-fixer + 30 tests | MERGED |
| #193 | t3 | Lint fix, SPDX header, retroactive manifests, fleet analysis | MERGED |
| #194 | t4 | Ruff noqa fix + test count sync + provenance drift fix | MERGED |
| #195 | t4 | Beads export — portable work exhausted, awaiting O6 hardware | MERGED |
| #196 | t4 | INT4+SDOT submission docs + optimization visualization | MERGED |
| #197 | t4 | Device bench + portable improvements (tests, NPU design) | MERGED |
| #199 | t3 | Symlink CLAUDE.md → AGENTS.md (fix bd doctor divergence) | MERGED |
| #200 | t4 | Coverage improvements across scripts/ and bench/ | MERGED |
| #201 | t3 | Beads sync after holding session (portable work exhausted) | MERGED |
| #202 | t4 | Script/bench test coverage + main merge | MERGED |
| #203 | t4 | Boundary-crossing cost micro-benchmark (ob-t3b.6) | MERGED |
| #204 | t3 | GDN-2 layer swap + RULER retrieval eval + lint/manifest/conformance fixes | MERGED |
| #205 | t4 | DEVPOST + SUBMISSION_STATUS stale count sync, boundary-crossing in NPU wall | MERGED |
| #206 | t3 | GDN-2 smart gate init + adaptation depth sweep (ob-t3b.9) | MERGED |
| #208–209 | t4 | Beads export sync (post-#205, disk audit update) | MERGED |
| #210 | t3 | FINDINGS §40 (GDN-2 swap/RULER) + section count update | MERGED |
| #211 | t3 | Sync stale FINDINGS-count references (53→54) | MERGED |
| #212 | t3 | CLAIM_VERIFICATION §2.6 — boundary-crossing cost claims | MERGED |
| #213 | t4 | RK3588 cross-check analysis + SDOT e2e spread data | MERGED |
| #214 | t3 | INT4+INT8 SDOT e2e benchmark — fleet gap fill (big+little, 4B+0.8B) | MERGED |
| #215 | t3 | Add missing t4 SDOT ctxsweep manifests + fix masked provenance gap | MERGED |
| #216 | t3 | Fix stale GDN-2 numbers + PROV map drift in partial_comparison_table.py | MERGED |
| #217–219 | t3 | Regenerate fleet_cross_device.png (loop flushes) | MERGED |
| #218 | t3 | Re-run rk3588-t3-clean at clean HEAD — fix dirty-manifest provenance | MERGED |
| #220 | t3 | Fix dirty provenance on little-cluster SDOT e2e (4 variants) | MERGED |
| #221 | t3 | RULER manifest schema fix + --max-time arg + clean provenance re-runs | MERGED |
| #222 | t3 | Fix stale pre-SDOT ctx-length scaling numbers + figure flush (ob-9ea) | MERGED |
| #223 | t3 | Beads: ob-9t0.11 — flag bench/t4 stale branch for rebasing | MERGED |
| #224 | t3 | Deterministic PNG output (constrained_layout) + docs provenance corrections | MERGED |
| #225 | t3 | SUBMISSION_STATUS T-4 update — trigger fired, PR table backfill, provenance cleaned | MERGED |
| #226 | t3 | Submission doc polish — FINDINGS line count + WRITEUP table t3/t4 fix | MERGED |
| #227 | t3 | Mark PR #226 as MERGED in SUBMISSION_STATUS table | MERGED |
| #228 | t4 | Fix stale test count to CI-authoritative 2176 passed (2257 was non-reproducible partial-deps artifact) + backfill PR table #191–#227 + SDOT provenance citations in §2 + close ob-9t0.11 (t4 rebased onto corrected main) | MERGED |
| #230 | t4 | Backfill PR #228 (merged) + #229 (closed) into SUBMISSION_STATUS table | MERGED |
| #232 | t3 | Pin matplotlib rcParams for cross-device PNG determinism (ob-6ay) + isolate readiness check from committed PNG + regression test | MERGED |
| #233 | t3 | Fix CI red — move importorskip before matplotlib import in rcParams test (ob-6ay follow-up) | MERGED |
| #234 | t4 | Add matplotlib to [dev] extras so CI runs fleet plotting tests (ob-mrd.20) — 53 more tests now execute in CI (2176→2229 passed) | MERGED |
| #236 | t4 | Update test counts 2176→2229 + skip 52→20 + backfill PR table #233/#234 + close ob-ns4 | MERGED |
| #237 | t3 | ADR 0007 T-4 firing confirmation — board absent, trigger fired on schedule | MERGED |
| #238 | t3 | PR #237 → MERGED in SUBMISSION_STATUS table (was merged as 961a28b) | MERGED |
| #239 | t4 | Merge sync + SUBMISSION_STATUS #237 status fix + PR table backfill #238/#239 | MERGED |
| #240 | t4 | Fix stale test counts + gdn2_reference coverage 50%→98% (2256→2262 passed, skip 20→2; README, DEVPOST, SUBMISSION_STATUS, ADR 0007) | MERGED |
| #241 | t3 | PR #240 status fix (OPEN→MERGED) + ADR 0007 stale test count (2256→2262) | MERGED |
| #242 | t3 | Backfill PR #241 (MERGED) in SUBMISSION_STATUS table | MERGED |
| #243 | t3 | Fix stale test count 2262→2235 CI-authoritative (skip 2→20) + PR #242 table backfill | MERGED |
| #244 | t3 | PR #243 table backfill (MERGED) | MERGED |
| #245 | t4 | beads export sync (portable work exhausted, all 31 open items HW-gated) | CLOSED |
| #246 | t4 | SDOT/INT4+SDOT clean re-run + KleidiAI provenance fix (ob-mrd.21) | MERGED |
| #247 | t3 | Fix submission doc arithmetic errors (scan citation 10.62→10.56 ob-9t0.13 + MSE 94%→84% ob-9t0.14) | MERGED |
| #248 | t4 | beads sync (security audit refresh ob-3i5 — bench/r5 still has credential) | MERGED |
| #249 | t4 | Security audit refresh + beads sync + goose-loop manifest cleanup fix + README count fix (89→90) + SUBMISSION_STATUS PR table backfill #244–#250 | MERGED |
| #250 | t3 | beads cleanup — remove 3 accidental junk issues (ob-kor, ob-z5p, ob-mtf) | MERGED |
| #251 | t3 | Provenance disclosures for dirty-tree CSVs in FINDINGS.md (ob-7fs) | MERGED |
| #252 | t4 | DEVPOST t4 dirty-tree provenance disclosure + INT4+SDOT ratio accuracy fix in SUBMISSION + WRITEUP (0.8B: 1.19× not 1.30×) + merged origin/main (t3 PR #251) | MERGED |
| #253 | t4 | FINDINGS count auto-repair in update_readme_counts.py — closes tooling gap where validate detects drift but update couldn't fix it | MERGED |
| #254 | t3 | Provenance disclosures for jetson-j1-clean.csv dirty tree + stale SHA citations in FINDINGS.md (ob-9n4) | MERGED |
| #255 | t4 | Row-level CSV validation for 5 previously-unvalidated CSV types (ob-se6) — validator now checks every data row, not just headers | MERGED |
| #256 | t4 | Fix theoretical ceiling 4.5→4.2 tok/s (83% claim now arithmetically correct) + test count/PR table maintenance | MERGED |
| #257 | t3 | Fix stale test count 2238→2253 CI-authoritative (15 ob-se6 validator tests) + PR #255 status OPEN→MERGED + DEVPOST FINDINGS line count 5681→5699 | MERGED |
| #258 | t4 | PR table backfill #256 MERGED + changelog fix | MERGED |
| #259 | t3 | Fix GDN recurrent state size errors across submission docs (576 KiB→48 MiB for 4B, 576 KiB→19.7 MiB for 0.8B; conv state audit fix 131K→32K elements) | MERGED |
| #260 | t3 | Fix SDOT intrinsic name in README+WRITEUP (vdotq_s32→vdotq_lane_s32) + cumulative speedup phrasing | MERGED |
| #261 | t4 | ob-3i5 security audit refresh + beads export | MERGED |
| #262 | t3 | PR #260 MERGED status fix + PR #261 backfill in SUBMISSION_STATUS | MERGED |
| #263 | t4 | PR table backfill #261 MERGED + #262 MERGED in SUBMISSION_STATUS | MERGED |
| #264 | t4 | PR table backfill #263 MERGED in SUBMISSION_STATUS | MERGED |
| #265 | t4 | PR table backfill #264 MERGED in SUBMISSION_STATUS | MERGED |
| #266 | t4 | FINDINGS.md provenance SHA fix for rk3588-t4 fresh bench (8227e98→79d1b47) | MERGED |
| #267 | t3 | Regenerate fleet_bandwidth_scaling.md with updated t4 data (11.53→11.94 GiB/s) | MERGED |
| #268 | t4 | Master context-sweep comparison table + 38 tests (ob-ami partial completion) | MERGED |
| #269 | t4 | Lint fix test_gen_ctxsweep_comparison.py + stale t4_big.csv number fix (cumdecay 22.25→21.46, scan 11.53→11.94, conv1d 19.04→19.35) | MERGED |
| #270 | t3 | Ruff lint fix + PR table backfill #265–#268 + stale test count 2253→2378 + stale t4 number fix in DEVPOST_WRITEUP + comparison_table + raw/ CSV count fix + generator root-cause fix + ruff format fixes | MERGED |
| #271 | t4 | Fix stale t4_big 4B numbers in 3 more files + validate_results.py test coverage (ob-8qt.25, 21 new tests) | MERGED |
| #272 | t3 | --test-count flag for update_readme_counts.py + raw/ dir-layout test coverage + ruff format fixes + PR table backfill | MERGED |
| #273 | t4 | INT8/INT4 SDOT speedup test coverage (ob-8qt.25) + 4 cross-document data discrepancy fixes (ob-9t0.15) + bug bead ob-mrd.22 for Pi 5 stale manifest | MERGED |
| #274 | t4 | Fix stale test count + README figure count 91→90 + PR table backfill | CLOSED — superseded by #276 (2351 was t4-local, not CI-authoritative) |
| #275 | t3 | Fix stale test count 2407→2411 local after t4 PR #273 added 4 INT8/INT4 SDOT tests | MERGED |
| #276 | t3 | Fix test count to CI-authoritative 2324/20 (supersedes #274 and #275 local-count convention) | MERGED |
| #277 | t4 | PR table backfill #274 CLOSED, #276 MERGED | MERGED |
| #278 | t3 | Regenerate stale plots and tables (full regen via bench/plots.py) | MERGED |
| #279 | t3 | Fix numerical consistency across 5 submission docs | MERGED |
| #280 | t3 | Back KleidiAI claim with §8 speedup data + packing caveat | MERGED |
| #281 | t4 | Close ob-ami (ctxsweep comparison table) + re-close ob-gzk (lost Dolt close) + PR table backfill #277-#280. Proposed dirty=true→false provenance flip for rk3588-t4_big.csv was NOT merged — rk3588-t4-big.json is a stale (2026-08-07), single-threaded manifest; the current multi-thread CSV data (updated 2026-08-11) is correctly dirty=true per rk3588-t4.json, updated in the same commit. See ob-dpl (2026-08-08) for the prior instance of this exact confusion. | MERGED (partial — provenance flip rejected) |
| #282 | t4 | Attempted to reintroduce the #281 provenance flip (dirty=true→false) plus a "regression test" asserting rk3588-t4-big.json takes precedence. Same root cause as #281: that manifest is genuinely stale/single-thread for this CSV (verified again via its own `effective_threads: 1` field), so the flip and the test were both rejected. Merged content was limited to a corrected #281 table entry (this row) and a note pointing at ob-dpl for the prior occurrence. | MERGED (partial — provenance flip + test rejected again) |
| #283 | t3 | T-3 descope filing ob-9t0.16 (O6 work designed not completed). Beads export sync only — no code changes. | MERGED |
| #284 | t3 | Fix figures README regression from loop flush (0→35 figures restored via bench/plots.py) | MERGED |
| #285 | t3 | PR #284 MERGED status flip + PR table backfill #283/#284 | MERGED |
| #286 | t4 | Fix stale PR table (#282 MERGED not rejected, add #285) | MERGED |
| #287 | t4 | DEVPOST/FINDINGS factual fixes — provenance, SDOT table, cross-device gap | MERGED |
| #288 | t4 | Post-merge doc accuracy sweep — stale counts + §38 attribution fixes | MERGED |
| #289 | t3 | Fix stale FINDINGS line count 5715→5718 | CLOSED — superseded (FINDINGS grew to 5743 after GPU benchmark refresh) |
| #290 | t4 | Fix stale local test count 2411→2351, skip 1→2 | MERGED (partial — see #290 correction: local counts vary by machine, docs cite CI-authoritative 2324/20 instead) |
| #291 | t3 | Beads export sync | MERGED |
| #292 | t3 | Fix stale data claims + refresh GPU benchmark (ob-5kw, ob-mrd.23, ob-q44.2) | MERGED |
| #293 | t4 | t4 device bench + portable work (clean re-runs: dirty→false for kernel/e2e/delta-matmul CSVs) | MERGED |
| #294 | t3 | Submission-facing doc consistency fixes (ob-q44.3, ob-4ke, ob-rp2, CLAIM_VERIFICATION §2.7) | MERGED |
| #295 | t3 | Fix manifest count 326→208 git-tracked + refresh README counts + update stale t4 data after #293 (ob-uqxt, ob-edvc) | MERGED |
| #296 | t4 | Fix manifest dirty-check bug (bench/manifest.py excluded results/.beads/ output files) + clean 102 false-positive manifests (75 t4 + 27 non-t4, ob-mrd.24) | MERGED — doc updates to DEVPOST_SUBMISSION/WRITEUP/FINDINGS superseded by #295's more complete fix (sha+numbers+dirty); CLAIM_VERIFICATION.md update kept |
| #297 | t4 | Thread manifest_dir through run_ablation so tests don't pollute repo (ob-mrd.26) + fix stale dirty=true ref in FINDINGS.md | MERGED |
| #298 | t3 | Fix ruff format on 3 files from PR #296 (lint CI broken on main) + CI test count 2324→2332 + fix run_ablation CLI manifest leak — add --manifest-dir (ob-k3jp) | MERGED |
| #299 | t3 | Fix last 2 run_ablation manifest leaks in test_run_ablation.py (ob-k3jp) + regenerate stale fleet_bandwidth_scaling.md (manifests now show dirty=false) | MERGED |
| #300 | t4 | PR table backfill #297 + #298 | MERGED |
| #301 | t4 | Fix stale dirty=true ref in comparison_table.md — ctxsweep manifests now confirmed clean (ob-mrd.24 false positive) | MERGED |
| #302 | t3 | PR table backfill #299–#300 | MERGED (partial — proposed figure count 90→91 fix rejected, no new figure file to justify it) |
| #303 | t3 | PR table backfill #301 + #302 (both MERGED) | MERGED |
| #304 | t4 | Lazy torch imports in gdn2_ruler.py + gdn2_swap.py (ob-mrd.27/28) — 15 ruler tests from skip→pass | MERGED |
| #305 | t3 | PR table backfill #303 (MERGED) | MERGED |
| #306 | t4 | Lazy torch imports in gdn2_swap.py + hf_backend memory check tests (ob-mrd.28 + ob-mrd.29) | MERGED |
| #307 | t3 | CI test count 2332→2352 + ruff format fix + PR table backfill #304/#305 | MERGED |
| #308 | t3 | Deterministic ablation timing — fix non-deterministic ablation_comparison.md (ob-mrd.30) | MERGED |
| #309 | t4 | Corpus collision detection tests for multikey (ob-mrd.31; renamed from ob-mrd.30 to resolve a disconnected-Dolt-sync ID collision) | MERGED |
| #310 | t3 | CI test count 2352→2370 across README/DEVPOST/SUBMISSION_STATUS + PR table backfill #306–#309 (ob-mrd.32; renamed from ob-mrd.31 to resolve a disconnected-Dolt-sync ID collision) | MERGED |
| #311 | t4 | Add 6 missing ablation manifests (ob-mrd.33; renamed from ob-mrd.31 to resolve a disconnected-Dolt-sync ID collision) | MERGED |

> **PRs #190, #198, #207, #229, #235, #245, #289:** closed (not merged). #282's proposed provenance content was rejected but its PR itself shows MERGED on GitHub (content replaced during review — see row above). No gaps in numbering.

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
- **GPU OpenCL kernels validated bit-exact on Mali-G610** (87/87 tests) — scan 1.99×, cumdecay 1.03×, DWConv1D 1.30× vs 4-thread A76 (§13)
- **~65× end-to-end speedup** from C kernel + GEMV + INT8 + SDOT + INT4+SDOT
- **Honest negative results** (NPU compilers reject the recurrence)

The one weakness: no heterogeneous NPU/GPU/CPU *dispatcher* on target
silicon. The individual GPU kernels ARE validated (bit-exact, above),
and the complete NPU offload design — operator-level mapping,
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
