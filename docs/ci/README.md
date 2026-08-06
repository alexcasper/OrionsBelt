# Reference Capture CI Workflow

This directory holds a GitHub Actions workflow definition that **cannot be
pushed from `bench/j1`** because the device's OAuth token lacks the `workflow`
scope (GitHub rejects workflow-file pushes without it).

## To deploy

Someone with a GitHub token that has the `workflow` scope (e.g. a developer
with `repo` + `workflow` scopes) should run:

```bash
cp docs/ci/reference_capture.yaml .github/workflows/reference_capture.yaml
git add .github/workflows/reference_capture.yaml
git commit -m "ci: add reference capture workflow (ob-aqv)"
git push
```

## What it does

Manually triggered (`workflow_dispatch`) workflow that:

1. Installs CPU-only PyTorch + transformers on an `ubuntu-latest` runner
2. Runs `scripts/capture_reference.py --model 0.8b --device cpu`
3. Verifies the output JSON files have the expected schema
4. Uploads `results/reference/` as a 90-day artifact

The 0.8B model in fp16 needs ~1.6 GB weights + ~3 GB overhead — well within
the 7 GB Actions runner. The 4B model needs ~8 GB and requires a larger runner
or a CUDA host.

This produces the golden reference logits for the correctness oracle
(bead **ob-3uh**) and advances **ob-aqv** (stand up x86 reference inference).

## Bead

- **ob-aqv** — Stand up x86/CUDA reference inference as the oracle source
- **ob-3uh** — Correctness oracle harness with tolerances (depends on ob-aqv)
