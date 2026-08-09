# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Three-way memory instrumentation: weights / KV cache / recurrent state.

The central measurement of this project (docs/archive/PLAN.md §2.4, bead ``ob-vfp``):
attribute memory to the three components whose scaling behavior against
context length is the whole point — weights (flat), KV cache (linear),
recurrent state (O(1)). Without this split the GDN advantage is asserted,
not demonstrated.

Uses ``gdn_layer_info`` for verified shapes and ``quant.policy`` for
precision tiers. Both static (predicted) and runtime (RSS-based) paths
are provided.
"""

from __future__ import annotations

from dataclasses import dataclass

from orionsbelt.model.gdn_layer_info import LAYER_INFO, CheckpointLayerInfo

# Bytes per element by dtype
_DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}


@dataclass(frozen=True)
class MemoryBreakdown:
    """Three-way memory split at a given context length.

    All values in bytes. ``total`` is the sum of the three components
    (does not include framework overhead, activations, or temporary
    buffers — those are reported separately if measured at runtime).
    """

    weights_bytes: int
    kv_cache_bytes: int
    recurrent_state_bytes: int
    conv_state_bytes: int  # GDN conv state (decode only, O(1))
    context_length: int

    @property
    def total_bytes(self) -> int:
        return (
            self.weights_bytes
            + self.kv_cache_bytes
            + self.recurrent_state_bytes
            + self.conv_state_bytes
        )

    @property
    def total_mib(self) -> float:
        return self.total_bytes / (1024 * 1024)

    def to_dict(self) -> dict[str, int]:
        return {
            "weights": self.weights_bytes,
            "kv_cache": self.kv_cache_bytes,
            "recurrent_state": self.recurrent_state_bytes,
            "conv_state": self.conv_state_bytes,
            "total": self.total_bytes,
            "context_length": self.context_length,
        }


def estimate_weights(
    info: CheckpointLayerInfo,
    weight_dtype_bytes: float = 2.0,  # FP16 default
) -> int:
    """Estimate total weight memory for a checkpoint.

    Counts all Linear/Conv/Embedding parameters using the config-derived
    dimensions. This is the *theoretical* minimum — actual frameworks add
    padding, optimizer state (training only), and metadata overhead.
    """
    h = info.hidden_size
    # GDN layers
    gdn_per_layer = (
        h * (info.key_dim * 2 + info.value_dim)  # in_proj_qkv
        + h * info.value_dim  # in_proj_z
        + h * info.linear_num_value_heads  # in_proj_b
        + h * info.linear_num_value_heads  # in_proj_a
        + info.value_dim * h  # out_proj
        + info.conv_dim * info.linear_conv_kernel_dim  # conv1d (depthwise)
        + info.linear_value_head_dim  # norm weight
    )
    # Full attention layers
    fa_per_layer = (
        h * (info.num_attention_heads * info.full_attn_head_dim * 2)  # q_proj (2x for gate)
        + h * (info.num_key_value_heads * info.full_attn_head_dim)  # k_proj
        + h * (info.num_key_value_heads * info.full_attn_head_dim)  # v_proj
        + (info.num_attention_heads * info.full_attn_head_dim) * h  # o_proj
        + info.full_attn_head_dim * 2  # q_norm + k_norm
    )
    # MLP/FFN layers (every layer has MLP)
    intermediate = info.intermediate_size
    mlp_per_layer = (
        h * intermediate  # gate_proj
        + h * intermediate  # up_proj
        + intermediate * h  # down_proj
    )
    # Embedding + LM head
    vocab = info.vocab_size
    embed = vocab * h
    # When embeddings are tied, lm_head reuses the embed_tokens weight tensor
    # — do not double-count it.
    lm_head = 0 if info.tie_word_embeddings else vocab * h

    total_params = (
        gdn_per_layer * info.num_gdn_layers
        + fa_per_layer * info.num_full_attention_layers
        + mlp_per_layer * info.num_hidden_layers
        + embed
        + lm_head
    )
    return int(total_params * weight_dtype_bytes)


def predict_breakdown(
    checkpoint: str = "4B",
    context_length: int = 4096,
    weight_dtype: str = "fp16",
    state_dtype: str = "fp32",
    kv_cache_dtype: str = "fp16",
) -> MemoryBreakdown:
    """Predict the three-way memory split at a given context length.

    This is the *theoretical* decomposition used to design experiments and
    validate that runtime measurements are in the right ballpark. The actual
    runtime measurement (via ``measure_rss`` or framework introspection)
    is the authority for published numbers.
    """
    info = LAYER_INFO[checkpoint]
    w_bytes = _DTYPE_BYTES[weight_dtype]
    s_bytes = _DTYPE_BYTES[state_dtype]
    kv_bytes = _DTYPE_BYTES[kv_cache_dtype]

    weights = estimate_weights(info, w_bytes)

    kv_cache = (
        info.num_key_value_heads
        * info.full_attn_head_dim
        * 2  # K + V
        * context_length
        * info.num_full_attention_layers
        * kv_bytes
    )
    recurrent_state = info.recurrent_state_elements_per_layer * info.num_gdn_layers * s_bytes
    conv_state = info.conv_state_elements_per_layer * info.num_gdn_layers * s_bytes

    return MemoryBreakdown(
        weights_bytes=weights,
        kv_cache_bytes=kv_cache,
        recurrent_state_bytes=recurrent_state,
        conv_state_bytes=conv_state,
        context_length=context_length,
    )


def sweep_context(
    checkpoint: str = "4B",
    context_lengths: list[int] | None = None,
    weight_dtype: str = "fp16",
    state_dtype: str = "fp32",
    kv_cache_dtype: str = "fp16",
) -> list[MemoryBreakdown]:
    """Predict memory breakdown across a range of context lengths.

    Shows the central claim: weights flat, KV cache linear, state flat.
    """
    if context_lengths is None:
        context_lengths = [4096, 32768, 131072, 262144]
    return [
        predict_breakdown(checkpoint, ctx, weight_dtype, state_dtype, kv_cache_dtype)
        for ctx in context_lengths
    ]


def format_breakdown_table(breakdowns: list[MemoryBreakdown]) -> str:
    """Format a sweep as a human-readable table for docs/reports."""
    lines = []
    lines.append(
        f"{'ctx':>8}  {'weights':>10}  {'kv_cache':>10}  {'state':>10}  {'conv':>8}  {'total':>10}"
    )
    lines.append("-" * 68)
    for b in breakdowns:
        lines.append(
            f"{b.context_length:>8}  "
            f"{b.weights_bytes / 1e6:>8.1f}MB  "
            f"{b.kv_cache_bytes / 1e6:>8.1f}MB  "
            f"{b.recurrent_state_bytes / 1e6:>8.1f}MB  "
            f"{b.conv_state_bytes / 1e6:>6.1f}MB  "
            f"{b.total_bytes / 1e6:>8.1f}MB"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runtime measurement (best-effort, platform-dependent)
# ---------------------------------------------------------------------------


def _rss_bytes() -> int | None:
    """Get current process RSS from /proc/self/status (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024  # kB → bytes
    except Exception:
        return None
    return None


def measure_delta(fn, *args, **kwargs) -> int | None:
    """Measure RSS delta around a function call.

    Returns the difference in RSS (bytes) after vs before calling ``fn``,
    or None if /proc is unavailable. Useful for measuring how much memory
    a specific allocation (e.g. loading KV cache, initializing state) consumes.
    """
    before = _rss_bytes()
    if before is None:
        return None
    fn(*args, **kwargs)
    after = _rss_bytes()
    if after is None:
        return None
    return after - before
