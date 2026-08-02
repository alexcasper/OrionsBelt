# CI workflow — staged, needs a human to activate

[`github-workflow-ci.yml`](./github-workflow-ci.yml) is a complete, locally-verified GitHub Actions
workflow. It is **not active**, because it lives here instead of `.github/workflows/`.

## Why it is here

The agent session that wrote it pushes over an OAuth app without the `workflow` scope, so GitHub
refuses the push outright:

```
! [remote rejected] refusing to allow an OAuth App to create or update
  workflow `.github/workflows/ci.yml` without `workflow` scope
```

That restriction exists so an app cannot silently introduce CI that executes code in your
repository. Rather than work around it, the file is parked here for you to review and move
deliberately.

## Activating it

```bash
mkdir -p .github/workflows
git mv ci/github-workflow-ci.yml .github/workflows/ci.yml
git commit -m "Enable CI workflow"
git push
```

Do that from a session with `workflow` scope (a normal local clone with your own credentials is
fine). Read it first — it installs apt packages (`gcc-aarch64-linux-gnu`, `qemu-user`) on the
runner.

## What it does

Four jobs, on push and pull_request, with a concurrency group so superseded pushes cancel, pinned
action versions, and no `continue-on-error`:

| Job | What it checks |
|---|---|
| `lint` | `ruff check .` and `ruff format --check .` |
| `test` | `pytest` on Python 3.10 (the `requires-python` floor) and 3.13 |
| `kernels` | Cross-compiles the Armv9.2 GDN kernels and verifies them numerically under QEMU via `scripts/verify_cpu_kernels.sh` — all seven portability-matrix targets (SVE1 @128/256/512, Neoverse-V1/V2, SVE2/Armv9-A, no-SVE scalar) |
| `onnx-probes` | Regenerates the NPU operator probes and asserts each still passes `onnx.checker` and executes under onnxruntime with finite outputs |

The `kernels` job is the valuable one: it gives continuous proof that the SVE, NEON, and scalar
dispatch paths still agree, which is the project's core correctness claim and would otherwise rot
silently.

**It deliberately does not run `cixparse`/`cixbuild`.** The CIX NOE SDK is a licensed ~461MB
download that cannot be redistributed or fetched in CI. The operator-coverage results in
[`docs/FINDINGS.md`](../docs/FINDINGS.md) §1 were produced locally and their logs are committed
under `artifacts/npu_op_probe/audit/` as the durable evidence.

## Verified locally before staging

Every command the workflow runs was executed on this machine: YAML parses, `ruff check` clean,
`ruff format --check` clean across 13 files, `pytest` 4 passed, `scripts/verify_cpu_kernels.sh`
passing all seven targets, and all six ONNX probes checking and executing.

One gotcha worth keeping if you edit the `onnx-probes` step: probe `02_decay_cumprod` computes
`Log(a)`, so it must be fed **positive** inputs. A naive standard-normal draw produces NaNs. That
is a property of the probe graph, not a bug.
