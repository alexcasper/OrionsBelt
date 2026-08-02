"""GPU backend for the Immortalis G720 via Vulkan/OpenCL compute.

Owned by beads ``t-gpu-scan`` (chunkwise gated delta-rule scan compute shader)
and ``t-gpu-validate`` (numerical validation against the reference, including
the long-context cases where recurrent-state error compounds).
"""
