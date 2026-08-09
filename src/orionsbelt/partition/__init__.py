# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Layer-to-engine assignment and the heterogeneous runtime.

Owned by beads:

- ``t-mapping-adr``       — the ADR fixing which layer classes run on which
  engine, with profiling measurements behind it.
- ``t-partition-runtime`` — per-engine subgraph invocation, tensor residency,
  minimal-copy handoff, and correct recurrent-state threading across engine
  boundaries (the subtle correctness hazard).
- ``t-dispatcher``        — dynamic routing by inference phase, engine
  occupancy, and thermal headroom, degrading gracefully to the static mapping
  when an engine is unavailable.
"""
