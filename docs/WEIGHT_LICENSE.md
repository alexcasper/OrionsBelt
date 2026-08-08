# Model weight license and compliance

**Bead:** `ob-ixt` · **Status:** Verified 2026-08-02 · **Parent:** `ob-xh3` (E4)

This document records the licensing and redistribution terms for every model
checkpoint this project downloads, so the compliance pass (bead `ob-9e2`) has
the information it needs without re-deriving it.

---

## 1. Checkpoints used

| Model | HuggingFace repo | Role | Approx. size | License |
|---|---|---|---|---|
| Qwen3.5-4B | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) | Primary | ~8 GB | Apache-2.0 |
| Qwen3.5-0.8B | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Fallback | ~1.6 GB | Apache-2.0 |

Selected by ADR 0003 (bead `ob-eae`). Both are dense text+vision checkpoints in
the Qwen3.5 family with the Gated DeltaNet hybrid architecture (3:1 GDN:full-
attention ratio, verified from `layer_types` in each `config.json`).

## 2. License verification

**Method:** The actual `LICENSE` file (not a summary) was fetched directly from
all eight Qwen3.5 repos (base + instruct-ready × four sizes: 0.8B, 2B, 4B, 9B)
during the model survey (bead `ob-7fv`, see `docs/MODEL_SURVEY.md` §3).

**Result:** Every file opens with `Apache License / Version 2.0, January 2004`.
No custom Qwen license, no field-of-use restriction, no scale-gated clause
(unlike Llama's community license). This satisfies risk R10 in docs/archive/PLAN.md's risk
register (checkpoint license restricting redistribution) — **not a risk for any
candidate in this range.**

**Re-verify directly.** The two checkpoints this project actually uses serve their
license files at stable raw URLs, so a reviewer can confirm the above without
trusting this document:

- <https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/LICENSE>
- <https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/LICENSE>

## 3. Redistribution terms

Under Apache-2.0:

- **Redistribution is permitted**, including of the model weights themselves,
  provided the license and copyright notice are included.
- **Modification is permitted**, including quantization and architectural
  modifications, with a notice of changes made.
- **No additional restrictions** may be imposed beyond the Apache-2.0 terms.

This means our project may redistribute the downloaded weights alongside our
code if we choose to, though we elect not to (§4 below).

## 4. Weights are not vendored in this repository

Weights are downloaded at setup time by [`scripts/fetch_weights.py`](../scripts/fetch_weights.py),
not committed to git. The `weights/` directory is in `.gitignore`. This is a
deliberate choice for three reasons:

1. **Repo size** — 8 GB of safetensors files would make the repo uncloneable
   for judges and contributors.
2. **License cleanliness** — even though Apache-2.0 permits redistribution,
   the compliance pass is simpler if the repo contains only our own code.
3. **Reproducibility** — the fetch script pins the repo ID, so a clean clone
   downloads the exact same checkpoint with verifiable checksums (the script
   writes a `.fetch_manifest.json` with SHA256 hashes of every file).

## 5. Attribution

The Qwen3.5 models were developed by the Qwen Team at Alibaba Cloud (Tongyi
Lab). The project's README and Devpost submission include attribution to the
model source, as required by Apache-2.0 §4(d) (a NOTICE file, if provided with
the original distribution, must be included in derivative distributions).

The `NOTICE` file from each checkpoint is downloaded alongside the weights by
`fetch_weights.py`.

## 6. Compliance checklist

For the final compliance pass (bead `ob-9e2`):

- [x] License identified: Apache-2.0 for all checkpoints
- [x] License verified from primary source (fetched `LICENSE` file, not summary)
- [x] No custom terms or restrictions beyond Apache-2.0
- [x] Weights not vendored (downloaded at setup)
- [x] Attribution included in README and submission
- [x] NOTICE files downloaded alongside weights
- [ ] Checksums recorded in manifests (written by `fetch_weights.py` on download)
