"""CPU backend for Armv9.2 big.LITTLE.

Owned by bead ``t-cpu-biglittle``: cluster-aware affinity (latency-critical
prefill pinned to A720 big cores, housekeeping on A520 little cores, no
cross-cluster migration mid-inference) plus i8mm/SVE GEMM paths verified
active at runtime rather than merely compiled in.

Unlike the NPU path, none of this is gated on external approval.
"""
