"""NPU backend via the CIX NOE Compiler.

Owned by beads ``t-npu-opcov`` (op-coverage audit for GDN operators),
``t-npu-export`` (subgraph export and INT8/INT4 quantization), and
``t-npu-accuracy`` (regression against the correctness oracle).

Gated on CIX Early Bird enrollment. Per PLAN.md risk R3, discovering that the
NOE Compiler has no kernel for a gated recurrent scan is a documented finding,
not a failure — see docs/FINDINGS.md.
"""
