"""Per-accelerator execution backends for the CIX P1 SoC.

One subpackage per engine, because the interesting question in this project is
which layers belong on which engine:

- ``npu`` — ~28.8 TOPS, INT4/INT8/FP16. Tuned for dense matmuls, so the
  natural home for full-attention layers and FFN/MoE blocks.
- ``gpu`` — Immortalis G720 MC10 via Vulkan/OpenCL. Most likely home for the
  chunkwise gated delta-rule scan.
- ``cpu`` — Armv9.2 big.LITTLE (4x A720 big / 4x A720 medium / 4x A520 little)
  with i8mm and SVE paths.

Engines must expose a common enough surface that ``partition`` can route
between them and thread recurrent state across engine boundaries.
"""
