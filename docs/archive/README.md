# Archive

Superseded planning documents, kept for history rather than deleted — git history preserves
everything either way, but these are moved here so the working docs tree reflects the current
target and nobody mistakes a stale plan for a live one.

| File | Why it's archived |
|---|---|
| [`brief.md`](./brief.md) | The original externally-sourced Devpost research brief (two overlapping agent briefs: a generic Orion O6/NPU one and a Qwen3.5/GDN-specific one). Superseded by the project's own verified framing — see [`README.md`](../../README.md) and [`docs/CLAIM_VERIFICATION.md`](../CLAIM_VERIFICATION.md). |
| [`PLAN.md`](./PLAN.md) | The original implementation plan (authored 2026-08-02), built around a primary Orion O6/NPU track with a "hedge" CPU-fleet track. [ADR 0007](../adr/0007-commit-to-edge-ai-track.md) committed to the CPU-fleet track on 2026-08-06, and the project has since re-centered on CPU linear-attention kernels (contributed toward Arm's KleidiAI) as the headline contribution rather than NPU/O6 heterogeneous mapping. Current status lives in [`README.md`](../../README.md)'s Status section, [`docs/FINDINGS.md`](../FINDINGS.md), and `bd ready`/`bd show <epic>` — not in a static plan document. |
| [`RISK_REGISTER.md`](./RISK_REGISTER.md) | An "operable companion to PLAN.md §7" — a daily go/no-go ritual and descope ladder scheduled against the Orion O6 arrival deadline (T-4 through T-1 before submission). That decision already resolved (ADR 0007), so the ritual is moot. |

Other docs (`docs/MODEL_SURVEY.md`, `docs/METHODOLOGY.md`, `docs/METRICS.md`,
`docs/DEVICE_RUNBOOK.md`, `docs/BACKEND_GUIDE.md`, `docs/COMPLIANCE_CHECKLIST.md`,
`docs/adr/*`) are **not** archived — they're still accurate and load-bearing for the current
CPU-kernel-focused work. Architecture decision records in particular are never revised or moved
here even when a later ADR supersedes one; that's what the ADR sequence itself records.

Section-number citations elsewhere in the repo (e.g. "PLAN.md §3.1") were left as-is rather than
mass-edited — the sections didn't move, only the file's location did, and the citation is still
findable via search. Only actual markdown links (`[text](path)`) were repointed at the new paths.
