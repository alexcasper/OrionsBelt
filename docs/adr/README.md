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

The template is `template.md`, added by bead `t-adr-scaffold`.
