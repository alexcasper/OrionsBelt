# Contributing to OrionsBelt

OrionsBelt optimizes a Qwen3.5-family Gated DeltaNet (GDN) hybrid model for Arm silicon,
primarily the Radxa Orion O6 (CIX P1 SoC). This document is for anyone — human or agent —
picking up work on the project. It covers how the repo is set up, how work is tracked, and,
most importantly, the conventions that make our benchmark numbers trustworthy. If you only
read one section, read "The reproducibility conventions" below: it's the difference between
a number a judge can trust and a number they can't.

For the full technical plan, rubric mapping, and verified claims, start with
[`PLAN.md`](./PLAN.md) and [`docs/CLAIM_VERIFICATION.md`](./docs/CLAIM_VERIFICATION.md).

## Getting set up

Clone the repo as usual. Python dependencies and tooling are declared in
[`pyproject.toml`](./pyproject.toml) at the repo root — treat that file, not this one, as the
authoritative list of packages and extras; it is being actively maintained alongside the rest
of the project, so if something below looks stale, `pyproject.toml` wins.

This project deliberately runs **two separate Python environments**, and they must not be
merged:

1. **The NPU toolchain environment.** The CIX NOE Compiler used to target the Orion O6's NPU
   is reported to require Python 3.8. As recorded in `PLAN.md` §2.1 and
   `docs/CLAIM_VERIFICATION.md` §4, **this pin is unverified** — the Radxa page that documented
   it now 404s — so treat it as "design for 3.8 until proven otherwise," not as confirmed fact.
   Anything that has to run inside this environment (NPU export/quantization code, and any
   shared library code it imports) must stay Python-3.8-compatible; see "Code style" below.
2. **The harness and dev-tooling environment.** The benchmark harness, plotting, linting, and
   general development tooling target a modern Python and can use whatever language features
   and dependencies are convenient there.

Keeping these separate is intentional, not an oversight: the moment shared code imports
something the 3.8 toolchain can't parse, the NPU path silently breaks, possibly for weeks
before anyone notices on real hardware. If you're setting up a fresh environment, install the
NPU-facing extras and the dev/bench extras into two different virtualenvs (or conda envs), and
don't let either one satisfy the other's dependency graph.

## Issue tracking is beads, not markdown TODOs

All work is tracked with `bd` (beads), a dependency-aware issue tracker. Do not create
markdown TODO lists or ad-hoc task files — `bd ready` is the source of truth for what's
available to work on. Full conventions (issue prefix, types, priorities, labels, dependency
direction) live in [`docs/BEADS.md`](./docs/BEADS.md); this section is just the loop you'll run
day to day:

```bash
bd ready                     # what's unblocked right now?
bd show <id>                 # read the full issue before touching it
bd update <id> --claim       # atomically claim it
# ... do the work ...
bd close <id> --reason "..." # close with a reason referencing what landed
```

One label worth knowing before you start: `external-gate`. It marks work blocked on a third
party — board procurement, CIX Early Bird enrollment — that no amount of engineering effort
resolves. `bd ready` may still list these; seeing one there is not permission to treat it as
actionable. Prefer `portable`-labelled work when hardware or toolchain access isn't in hand.

## The reproducibility conventions

This is the part that decides whether our numbers survive a judge's scrutiny, so it gets its
own section rather than a bullet buried in a list. Every convention below exists because of a
specific way edge-device benchmarks go wrong.

**Every benchmark run emits a manifest.** Device, kernel, SDK/driver versions, cpufreq
governor, observed clocks, thermal state, and git SHA — all captured alongside the result, not
reconstructed afterward from memory. On a passively-cooled edge board like the Orion O6, there
is no fan fighting the CPU/GPU/NPU for headroom: thermal state by itself can move throughput
enough to make two runs of the *same* code look like a real optimization, or hide a real
regression. A number without a manifest is not a result — it cannot be told apart from a lucky
(or unlucky) thermal moment.

**Report percentiles and repeat counts, never a single best run.** A best-of-N run is a
statement about noise, not about the change you made. Every reported figure should carry how
many repeats went into it and the distribution (e.g. p50/p95), not a cherry-picked minimum.

**Prefill and decode are reported separately and never averaged into one "tokens/sec."**
Upstream's measured GDN kernel numbers show why this matters concretely: optimizing the
DeltaNet path speeds up prefill by 1.38-1.49x, while decode stays flat, because the
single-token recurrence is memory-bandwidth-bound rather than compute-bound — and the O6's
100GB/s LPDDR5 binds harder than the hardware those upstream numbers came from. A single
blended tokens/sec figure would average a real win against a real (and expected, physics-based)
non-win and hide the actual result. Report them as two numbers, always.

**Memory is attributed three ways: model weights, full-attention KV cache, and GDN recurrent
state.** These are tracked separately because they behave differently — KV cache grows
linearly with context length, while GDN recurrent state is O(1) per token regardless of
context. That asymmetry *is* the project's central claim (see `PLAN.md` §2.4 and
`bench/README.md`). Collapsing the three into one "peak memory" number would erase the exact
evidence the project exists to produce.

**Every optimization is gated by the correctness oracle.** Quantization, kernel swaps, engine
reassignment — none of it counts as a result until it's checked against golden logits/
perplexity from the x86 reference within justified tolerances (see
[`tests/README.md`](./tests/README.md)). Speed that changes outputs is not speed.

**Results are committed as CSV under `results/raw/`, conforming to the frozen schema in
[`docs/RESULTS_SCHEMA.md`](./docs/RESULTS_SCHEMA.md)** (that document is being written
concurrently — treat it, not this file, as the schema's source of truth once it lands). Each
CSV is paired with a manifest under `results/manifests/`; a CSV without its manifest is not a
result, per `results/README.md`. Figures under `results/figures/` are always generated from
this committed data by `bench/plots.py` — never hand-assembled in a slide deck or spreadsheet.
If a figure can't be regenerated from what's in `results/raw/`, it doesn't belong in the
write-up.

**Model weights are never committed to the repo.** They are downloaded at setup time by
scripts, not vendored into git history. This keeps the repo small, keeps license compliance
clean (some Qwen3.5 checkpoints may carry redistribution restrictions), and matches how a judge
or a future contributor will actually reproduce a clean-clone run.

## Decisions become ADRs

Every `decision`-type bead must produce an Architecture Decision Record in `docs/adr/` before
it closes, using [`docs/adr/template.md`](./docs/adr/template.md), and linked from the bead's
notes. ADRs are numbered sequentially (`NNNN-short-slug.md` from `0001`); **claim your number
in [`docs/adr/README.md`](./docs/adr/README.md) before writing** so two agents working in
parallel don't collide on the same number.

## Honest reporting

Negative and partial results get written up plainly: "we tried X, it didn't help, here is the
profile showing why" is a complete, valid entry in the write-up, not something to omit or soften.
This kind of finding scores under the rubric's Potential Impact criterion and costs nothing
under scrutiny — a reviewer who can see *why* something didn't work trusts the rest of the
write-up more, not less. The failure mode we most want to avoid is the opposite one:
overstating a result that doesn't hold up. A flat decode-throughput number, reported honestly
with the memory-bandwidth explanation, reads as competence. The same number dressed up as a win
reads as the thing judges are specifically trained to catch.

## Code style

- Lint and format with `ruff`; configuration (line length, target Python version, rule set)
  lives in `pyproject.toml` — check there rather than assuming a default.
- Anything that must run inside the NOE Compiler / NPU toolchain environment needs to stay
  compatible with Python 3.8. Concretely: no `match` statements, no `X | Y` union syntax — use
  `typing.Optional`, `typing.Union`, `typing.List`, `typing.Dict`, etc. When in doubt about
  whether a piece of code is toolchain-facing, write it 3.8-compatible anyway; it's cheap
  insurance against the pin turning out to be real.
- Tests live in `tests/`, split between fast unit tests (metrics, schema conformance, manifest
  capture, partitioning logic) and the correctness oracle (golden-output comparison against the
  x86 reference). See `tests/README.md` for the distinction — don't conflate the two.

## Commit and PR conventions

Reference the relevant bead ID in commit messages, e.g. `Freeze results schema contract
(ob-q9i)`. This keeps the connection between a change and the issue-tracker record of *why* it
happened, which matters both for review and for anyone reconstructing project history later.

If your change adds, updates, or closes any beads, keep the committed JSONL export in sync
before you finish:

```bash
bd export -o .beads/issues.jsonl
```

`.beads/issues.jsonl` is a passive export for fresh clones and human/reviewer readability — the
live Dolt database is the actual source of truth during normal work — but an out-of-sync export
means the next clone starts from a stale issue graph, so don't skip this step when it applies.
