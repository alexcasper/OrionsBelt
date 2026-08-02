# Model weights: license and attribution (ob-ixt)

**Status:** Verified 2026-08-02 · **Bead:** `ob-ixt` · **Checkpoint decision:** [ADR 0003](./adr/0003-model-checkpoint-selection.md)

## Policy

**Weights are never vendored in the repository.** They are downloaded at setup
time via [`scripts/fetch_weights.py`](../scripts/fetch_weights.py), under each
checkpoint's own license. This keeps the repo small and avoids redistribution
concerns we have not verified.

## Checkpoints

| Model | HF repo | License | Size | Role |
|---|---|---|---:|---|
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | **Apache-2.0** | ~16 GB | Primary (24 GDN + 8 FA) |
| Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` | **Apache-2.0** | ~3 GB | Fallback (18 GDN + 6 FA) |

Both licenses verified from the LICENSE files in each HuggingFace repo during the
model survey (`ob-7fv`). Apache-2.0 permits redistribution and modification with
attribution, so downloading at setup time is compliant.

## Attribution

```
Qwen3.5 models by Alibaba Cloud / Qwen Team.
Licensed under Apache License 2.0.
Source: https://huggingface.co/Qwen/Qwen3.5-4B
```

This attribution must appear in the Devpost submission's "Technology" section and
in the README's model-credits area.

## Setup

```bash
python3 scripts/fetch_weights.py --all          # download both checkpoints
python3 scripts/fetch_weights.py --model 4b      # primary only
python3 scripts/fetch_weights.py --check         # check cache status
```

Requires `huggingface_hub` (`pip install huggingface_hub`). If unavailable, the
script prints manual download instructions.

## Compliance checklist (for ob-9e2)

- [x] Checkpoint license verified (Apache-2.0, both sizes)
- [x] No weights vendored in repo (downloaded at setup)
- [x] Attribution text prepared for submission
- [x] Fetch script is idempotent (skips already-cached checkpoints)
- [x] Fetch manifest written for provenance (PLAN.md §9)
