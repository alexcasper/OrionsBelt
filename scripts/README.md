# Setup and reproduction scripts

Entry points that make the repo reproducible by someone who has never seen it. Judges
score Developer Experience at 15 points, and bead `t-repro-rehearsal` verifies these by
following the documented path verbatim on a clean system.

Expected contents as beads land:

- Orion O6 bring-up (Debian 12 flash, first boot) — `t-o6-flash`
- Python 3.8 environment plus NOE Compiler install — `t-py38-noe`
  (kept separate from the harness environment: the NOE Compiler pins Python 3.8)
- Portable aarch64 hedge-target setup — `t-hedge-bringup`
- Weight acquisition, downloaded at setup rather than vendored — `t-weights-fetch`
- Benchmark sweep drivers — `t-harness-core`

Scripts should be non-interactive and idempotent.
