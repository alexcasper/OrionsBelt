# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Quantization policy for Qwen3.5 on Arm silicon.

Derived from the GDN architecture audit (``docs/GDN_ARCHITECTURE_AUDIT.md``,
bead ``ob-37v``) and the decode-bandwidth analysis (``docs/METRICS.md``
appendix, bead ``ob-qpa``).

The central principle: **GDN recurrent state is fed back through every decode
step, so quantization error accumulates multiplicatively over the sequence
rather than staying local the way KV-cache error does.** State and gating
tensors therefore require FP16+ carve-outs even when everything else is INT4.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Precision(str, Enum):
    """Precision tiers used in the policy.

    Ordered from lowest (most compressed) to highest (most accurate).
    """

    INT4 = "int4"
    INT8 = "int8"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"


class QuantScheme(str, Enum):
    """Quantization scheme for a tensor group."""

    WEIGHT_ONLY_INT4 = "int4_w4a16"  # INT4 weights, FP16 activations
    WEIGHT_ONLY_INT8 = "int8_w8a16"  # INT8 weights, FP16 activations
    W8A8 = "int8_w8a8"  # INT8 weights + activations
    FP16 = "fp16"  # no quantization, half precision
    FP32 = "fp32"  # no quantization, full precision


@dataclass(frozen=True)
class TensorGroupPolicy:
    """Policy for one group of tensors in the model.

    ``reason`` explains *why* this precision was chosen, so a future
    contributor can understand the constraint rather than blindly reapply it.
    """

    tensor_group: str
    scheme: QuantScheme
    precision_runtime: Precision  # precision during forward pass
    reason: str


# ---------------------------------------------------------------------------
# Per-layer-class policies
# ---------------------------------------------------------------------------

#: GDN (linear attention) layer tensor policies.
#:
#: The recurrent state, decay gate, and beta gate must stay high-precision
#: because their errors compound through the recurrence loop.
GDN_POLICIES: list[TensorGroupPolicy] = [
    TensorGroupPolicy(
        tensor_group="in_proj_qkv.weight",
        scheme=QuantScheme.WEIGHT_ONLY_INT4,
        precision_runtime=Precision.FP16,
        reason=(
            "Dense matmul projecting hidden→QKV. Standard GEMV — ideal for "
            "INT4 weight quantization via KleidiAI i8mm micro-kernels. "
            "Activation stays FP16 because the output feeds the conv and "
            "delta-rule, where precision matters."
        ),
    ),
    TensorGroupPolicy(
        tensor_group="in_proj_z.weight",
        scheme=QuantScheme.WEIGHT_ONLY_INT4,
        precision_runtime=Precision.FP16,
        reason=(
            "Gate projection (SiLU gate). Weight-only INT4 is safe; the gate "
            "value z is applied after RMSNorm and is not fed back through "
            "the recurrence."
        ),
    ),
    TensorGroupPolicy(
        tensor_group="in_proj_b.weight",
        scheme=QuantScheme.WEIGHT_ONLY_INT8,
        precision_runtime=Precision.FP16,
        reason=(
            "Beta (write-gate) projection. INT8 not INT4: beta controls the "
            "magnitude of every state write via the delta rule, so a 4-bit "
            "weight error here directly perturbs the accumulated state. INT8 "
            "is a conservative middle ground."
        ),
    ),
    TensorGroupPolicy(
        tensor_group="in_proj_a.weight",
        scheme=QuantScheme.WEIGHT_ONLY_INT8,
        precision_runtime=Precision.FP16,
        reason=(
            "Decay-gate input projection. The output `a` enters "
            "g = -exp(A_log) * softplus(a + dt_bias). The exp() amplifies "
            "errors, so `a` needs more precision than a generic projection. "
            "INT8, not INT4."
        ),
    ),
    TensorGroupPolicy(
        tensor_group="out_proj.weight",
        scheme=QuantScheme.WEIGHT_ONLY_INT4,
        precision_runtime=Precision.FP16,
        reason=(
            "Output projection. Dense matmul, no recurrence interaction. Safe for INT4 weight-only."
        ),
    ),
    TensorGroupPolicy(
        tensor_group="conv1d.weight",
        scheme=QuantScheme.WEIGHT_ONLY_INT8,
        precision_runtime=Precision.FP16,
        reason=(
            "Depthwise Conv1D (kernel=4, groups=8192). INT8 weights are safe "
            "since the depthwise multiply-accumulate is a short, stable "
            "computation. Output stays FP16 as it feeds Q/K/V."
        ),
    ),
    TensorGroupPolicy(
        tensor_group="A_log",
        scheme=QuantScheme.FP16,
        precision_runtime=Precision.FP16,
        reason=(
            "Learnable decay-rate parameter used inside exp(A_log). Must stay "
            "FP16+ — any quantization error here is exponentially amplified "
            "by the exp() and then applied to the recurrent state every step."
        ),
    ),
    TensorGroupPolicy(
        tensor_group="dt_bias",
        scheme=QuantScheme.FP16,
        precision_runtime=Precision.FP16,
        reason=(
            "Bias added to decay-gate input before softplus. Same exp() "
            "amplification risk as A_log."
        ),
    ),
    TensorGroupPolicy(
        tensor_group="recurrent_state",
        scheme=QuantScheme.FP32,
        precision_runtime=Precision.FP32,
        reason=(
            "THE critical carve-out. The recurrent state S_t = S_{t-1} * "
            "exp(g_t) + k⊗delta is fed back through every token. Quantization "
            "error accumulates multiplicatively over the sequence — at 262K "
            "tokens, even a per-step ε=1e-4 compounds into systematic drift. "
            "FP32 is mandatory; BF16 is a stretch option (ob-8qt.4) only if "
            "accuracy regression testing passes."
        ),
    ),
    TensorGroupPolicy(
        tensor_group="norm.weight",
        scheme=QuantScheme.FP16,
        precision_runtime=Precision.FP32,
        reason=(
            "RMSNormGated weight. Stored FP16, computed FP32 (standard "
            "practice — all normalizations are numerically sensitive)."
        ),
    ),
]

#: Full-attention layer tensor policies.
#:
#: Standard attention with KV cache — errors are local (each token's KV
#: entry is independent), so activation quantization (W8A8) is viable.
FULL_ATTENTION_POLICIES: list[TensorGroupPolicy] = [
    TensorGroupPolicy(
        tensor_group="q_proj.weight",
        scheme=QuantScheme.WEIGHT_ONLY_INT4,
        precision_runtime=Precision.FP16,
        reason=(
            "Standard GQA projection. INT4 weight-only via KleidiAI. "
            "Note: q_proj outputs 2× (query + gate) — the gate portion "
            "controls output scaling but is not recurrent."
        ),
    ),
    TensorGroupPolicy(
        tensor_group="k_proj.weight",
        scheme=QuantScheme.WEIGHT_ONLY_INT4,
        precision_runtime=Precision.FP16,
        reason="Standard KV projection, no recurrence. Safe for INT4.",
    ),
    TensorGroupPolicy(
        tensor_group="v_proj.weight",
        scheme=QuantScheme.WEIGHT_ONLY_INT4,
        precision_runtime=Precision.FP16,
        reason="Standard KV projection, no recurrence. Safe for INT4.",
    ),
    TensorGroupPolicy(
        tensor_group="o_proj.weight",
        scheme=QuantScheme.WEIGHT_ONLY_INT4,
        precision_runtime=Precision.FP16,
        reason="Output projection. Safe for INT4.",
    ),
    TensorGroupPolicy(
        tensor_group="kv_cache",
        scheme=QuantScheme.FP16,
        precision_runtime=Precision.FP16,
        reason=(
            "KV cache entries. Unlike GDN state, each entry is independent — "
            "quantization error is local. FP16 is sufficient; INT8 KV cache "
            "is a possible future optimization if accuracy allows."
        ),
    ),
    TensorGroupPolicy(
        tensor_group="q_norm/k_norm.weight",
        scheme=QuantScheme.FP16,
        precision_runtime=Precision.FP32,
        reason="RMSNorm per-head. Computed FP32, stored FP16.",
    ),
]

#: FFN/MLP layer tensor policies.
#:
#: Dense SwiGLU MLP — the largest weight block and the biggest INT4 win.
MLP_POLICIES: list[TensorGroupPolicy] = [
    TensorGroupPolicy(
        tensor_group="gate_proj.weight",
        scheme=QuantScheme.WEIGHT_ONLY_INT4,
        precision_runtime=Precision.FP16,
        reason=(
            "SwiGLU gate projection. Largest single weight tensor in the "
            "model. INT4 is the dominant decode-throughput lever (METRICS.md "
            "appendix: ~4× traffic reduction)."
        ),
    ),
    TensorGroupPolicy(
        tensor_group="up_proj.weight",
        scheme=QuantScheme.WEIGHT_ONLY_INT4,
        precision_runtime=Precision.FP16,
        reason="SwiGLU up projection. Same as gate_proj — INT4 weight-only.",
    ),
    TensorGroupPolicy(
        tensor_group="down_proj.weight",
        scheme=QuantScheme.WEIGHT_ONLY_INT4,
        precision_runtime=Precision.FP16,
        reason="SwiGLU down projection. INT4 weight-only.",
    ),
]

#: Embedding / LM head policies.
EMBEDDING_POLICIES: list[TensorGroupPolicy] = [
    TensorGroupPolicy(
        tensor_group="embed_tokens.weight",
        scheme=QuantScheme.WEIGHT_ONLY_INT8,
        precision_runtime=Precision.FP16,
        reason=(
            "Token embedding lookup. INT8 is a conservative choice; embedding "
            "quantization can affect rare-token representations. INT4 is "
            "risky for vocabulary diversity."
        ),
    ),
    TensorGroupPolicy(
        tensor_group="lm_head.weight",
        scheme=QuantScheme.WEIGHT_ONLY_INT8,
        precision_runtime=Precision.FP16,
        reason=(
            "LM head (final classifier over 248K vocab). INT8 to preserve "
            "logit resolution across the large vocabulary."
        ),
    ),
]

#: All policies combined, keyed by tensor-group name.
ALL_POLICIES: dict[str, TensorGroupPolicy] = {
    p.tensor_group: p
    for policies in (GDN_POLICIES, FULL_ATTENTION_POLICIES, MLP_POLICIES, EMBEDDING_POLICIES)
    for p in policies
}


def policy_for(tensor_name: str) -> TensorGroupPolicy:
    """Look up the quantization policy for a tensor by name.

    Matches by suffix — e.g. ``layers.5.linear_attn.in_proj_qkv.weight``
    matches the ``in_proj_qkv.weight`` policy.

    Raises ``KeyError`` if no policy matches, so callers cannot silently
    apply a wrong precision.
    """
    for key, policy in ALL_POLICIES.items():
        if tensor_name.endswith(key) or key in tensor_name:
            return policy
    raise KeyError(f"No quantization policy for tensor: {tensor_name}")


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------


def estimate_weight_footprint_mib(
    total_params: int,
    int4_fraction: float = 0.85,
    int8_fraction: float = 0.10,
) -> dict[str, float]:
    """Estimate model weight footprint under the policy.

    Default fractions assume ~85% of params are MLP/attention projections
    (INT4), ~10% are embedding/sensitive projections (INT8), and ~5% are
    FP16 (norms, A_log, dt_bias — negligible by count).

    Returns a dict with breakdown by precision tier.
    """
    fp16_fraction = 1.0 - int4_fraction - int8_fraction
    int4_bytes = total_params * int4_fraction * 0.5  # 4 bits = 0.5 bytes
    int8_bytes = total_params * int8_fraction * 1.0  # 8 bits = 1 byte
    fp16_bytes = total_params * fp16_fraction * 2.0  # 16 bits = 2 bytes
    total_mib = (int4_bytes + int8_bytes + fp16_bytes) / (1024 * 1024)
    return {
        "int4_mib": int4_bytes / (1024 * 1024),
        "int8_mib": int8_bytes / (1024 * 1024),
        "fp16_mib": fp16_bytes / (1024 * 1024),
        "total_mib": total_mib,
        "bits_per_param_avg": (int4_bytes + int8_bytes + fp16_bytes) * 8 / total_params,
    }
