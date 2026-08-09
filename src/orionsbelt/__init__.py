# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""OrionsBelt — optimizing a Qwen3.5 Gated DeltaNet hybrid model for Arm silicon.

Package layout (see docs/archive/PLAN.md section 10):

- ``model``     — checkpoint loading and Gated DeltaNet layer introspection
- ``engines``   — per-accelerator backends (``npu``, ``gpu``, ``cpu``)
- ``partition`` — layer-to-engine assignment and the runtime dispatcher
- ``quant``     — quantization policies and precision carve-outs
"""

__all__ = ["model", "engines", "partition", "quant"]
