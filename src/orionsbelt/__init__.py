"""OrionsBelt — optimizing a Qwen3.5 Gated DeltaNet hybrid model for Arm silicon.

Package layout (see PLAN.md section 10):

- ``model``     — checkpoint loading and Gated DeltaNet layer introspection
- ``engines``   — per-accelerator backends (``npu``, ``gpu``, ``cpu``)
- ``partition`` — layer-to-engine assignment and the runtime dispatcher
- ``quant``     — quantization policies and precision carve-outs
"""

__all__ = ["model", "engines", "partition", "quant"]
