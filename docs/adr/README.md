# Architecture Decision Records

Every bead of type `decision` must produce an ADR here before it closes, linked from the
bead's notes. With a 12-day schedule and several irreversible forks, the point is that a
choice and its reasoning survive as an artifact instead of evaporating into chat history.

Naming: `NNNN-short-slug.md`, numbered sequentially from `0001`.

Decisions already scheduled (see PLAN.md):

| Bead | Decision |
|---|---|
| `t-descope-ladder` | Descope ladder and trigger dates |
| `t-track-decision` | **Physical AI vs Mobile AI track — hard date 2026-08-09** |
| `t-hedge-select` | Portable aarch64 hedge target |
| `t-model-select` | Primary and fallback Qwen3.5 checkpoint |
| `t-mapping-adr` | Layer-to-engine assignment across NPU/GPU/CPU |
| `t-gdn2-decision` | GDN-2 benchmark-only vs layer-swap |

## Numbering

`NNNN-short-slug.md`, sequential from `0001`. **Claim your number before writing** so
parallel agents don't collide. Assigned so far:

| Number | Subject | Bead |
|---|---|---|
| 0001 | GDN-2 decoupled erase/write gating hypothesis | `ob-8m7` |
| 0002 | Portable aarch64 hedge target | `ob-zh4` |
| 0003 | Primary and fallback Qwen3.5 checkpoint | `ob-eae` |
| 0004 | Descope ladder and trigger dates | `ob-atp` |
| 0005 | Device fleet as Edge AI hedge target; bandwidth-scaling study | `ob-zh4` follow-on |
| 0006 | Quantization policy: INT4 weights, FP32 recurrent state and gates | `ob-qpa` |
| 0007 | Commit to Edge AI track — O6 has not arrived | `ob-imb` |
| 0008 | GDN-2 comparison via microbenchmark only (option a) | `ob-9ke` |

Note: 0006 was written twice in parallel — on `bench/r5` and `bench/t3` — which is
exactly the collision this table exists to prevent. Both derivations reached the same
decision independently, which is corroborating rather than contradictory; r5's text was
kept and t3's survives in that branch's history.

Similarly, 0007 was written by both the t3 and t4 agents on the same day. Both reached
the same decision independently. The t3 version (`0007-commit-to-edge-ai-track.md`) is
kept here as the more detailed canonical text; the t4 version survives in git history.

The template is [`template.md`](./template.md).
