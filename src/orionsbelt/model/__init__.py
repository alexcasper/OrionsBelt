"""Checkpoint loading and Gated DeltaNet layer introspection.

Owned by beads:

- ``t-arch-audit`` — document the GDN layer structure (causal Conv1D, decay
  gating, delta-rule update, chunk size, recurrent state shape) and the real
  linear-to-full attention layer ratio, read from the modeling code rather
  than from any secondary source.
- ``t-x86-ref``   — reference inference path used as the correctness oracle.
- ``t-weights-fetch`` — download-at-setup weight acquisition.
"""
