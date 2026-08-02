# Model licenses and attribution

This document records the license and attribution requirements for every third-party
checkpoint downloaded by [`scripts/fetch_weights.py`](../scripts/fetch_weights.py).
Bead: `ob-ixt`. Required for the compliance pass (`ob-9e2`).

## Summary

All target checkpoints are **Apache-2.0** licensed, verified from the HuggingFace
model card `tags` metadata (fetched 2026-08-02 via the HF REST API) and from the
`LICENSE` file included in each repository. Apache-2.0 permits commercial use,
modification, distribution, and private use with minimal conditions (retain
copyright notice, state changes).

## Primary checkpoint

| Field | Value |
|---|---|
| **Model** | Qwen3.5-4B |
| **Repo** | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) |
| **License** | Apache-2.0 |
| **License file** | [`LICENSE`](https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/LICENSE) |
| **Approx. size** | ~8.2 GB (fp16 safetensors) |
| **SHA-256 of LICENSE** | Recorded in `models/Qwen3.5-4B/manifest.json` after fetch |

## Fallback checkpoint

| Field | Value |
|---|---|
| **Model** | Qwen3.5-0.8B |
| **Repo** | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| **License** | Apache-2.0 |
| **License file** | [`LICENSE`](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/LICENSE) |
| **Approx. size** | ~1.7 GB (fp16 safetensors) |
| **SHA-256 of LICENSE** | Recorded in `models/Qwen3.5-0.8B/manifest.json` after fetch |

## Attribution requirements

Apache-2.0 §4 requires:

1. **Retain the LICENSE** and copyright notice in all copies. The `LICENSE` file is
   downloaded alongside each checkpoint and stored in `models/<model>/LICENSE`.
2. **State changes** if the model is modified. Our quantization and kernel-level
   optimizations (ADR 0006) constitute modifications; the submission write-up must
   document what was changed from the original checkpoint.
3. **No endorsement** — do not imply the licensor endorses our work.

The Qwen team does not impose additional restrictions beyond Apache-2.0 (verified
from the model card and the absence of a separate custom license).

## Redistribution posture

- **Not vendored.** Weights are downloaded at setup time by `scripts/fetch_weights.py`
  and are never committed to the repository. The `models/` directory is in `.gitignore`.
- **Redistribution is legal but unnecessary.** Apache-2.0 permits redistribution,
  but the checkpoints are publicly available on HuggingFace. Our submission
  instructions simply point to the HF repo and the fetch script.
- **Derived artifacts** (quantized weights, compiled NPU binaries) inherit the
  Apache-2.0 license with the "state changes" obligation.

## Download provenance

Each `models/<model>/manifest.json` records:
- The HuggingFace repo ID and revision (`main`)
- The timestamp of the fetch
- A per-file list with byte counts and success/skip status

This provides full reproducibility: anyone can re-run `scripts/fetch_weights.py` to
obtain identical files, and the manifest confirms what was present at build time.

## Related documents

- [ADR 0003](adr/0003-model-checkpoint-selection.md) — model checkpoint selection rationale
- [ADR 0006](adr/0006-quantization-policy.md) — quantization policy (modifications to the checkpoint)
- [scripts/README.md](../scripts/README.md) — setup script index
- `scripts/fetch_weights.py` — the download script itself
