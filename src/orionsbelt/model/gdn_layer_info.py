"""Programmatic GDN layer-info for Qwen3.5 checkpoints.

Derived from the modeling-code audit (``docs/GDN_ARCHITECTURE_AUDIT.md``,
bead ``ob-37v``).  All figures verified against ``config.json`` fetched
directly from HuggingFace and the transformers ``modeling_qwen3_5.py``
source.

Usage::

    from orionsbelt.model.gdn_layer_info import LAYER_INFO

    info = LAYER_INFO["4B"]
    print(info.recurrent_state_elements_per_layer)  # 524_288
    print(info.gdn_layer_count)                      # 24
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CheckpointLayerInfo:
    """Static layer-structure facts for one Qwen3.5 checkpoint size.

    Every field is derived from ``config.json`` or the modeling source code,
    not from a secondary summary.  See
    :doc:`GDN_ARCHITECTURE_AUDIT </docs/GDN_ARCHITECTURE_AUDIT>` for
    line-by-line provenance.
    """

    # --- identity ---
    name: str
    hf_repo: str

    # --- global dims ---
    hidden_size: int
    num_hidden_layers: int
    num_gdn_layers: int
    num_full_attention_layers: int
    layer_pattern: str  # human-readable, e.g. "8x(3 GDN -> 1 full)"

    # --- GDN (linear attention) dims ---
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int

    # --- full-attention dims ---
    num_attention_heads: int
    num_key_value_heads: int
    full_attn_head_dim: int
    partial_rotary_factor: float

    # --- derived (computed in __post_init__) ---
    key_dim: int = field(init=False)
    value_dim: int = field(init=False)
    conv_dim: int = field(init=False)
    kv_head_ratio: int = field(init=False)  # num_v_heads // num_k_heads

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_dim", self.linear_key_head_dim * self.linear_num_key_heads)
        object.__setattr__(
            self, "value_dim", self.linear_value_head_dim * self.linear_num_value_heads
        )
        object.__setattr__(self, "conv_dim", self.key_dim * 2 + self.value_dim)
        object.__setattr__(
            self, "kv_head_ratio", self.linear_num_value_heads // self.linear_num_key_heads
        )

    # --- recurrent state ---
    @property
    def recurrent_state_shape(self) -> tuple[int, ...]:
        """Per-layer recurrent state tensor shape (batch excluded).

        ``(num_v_heads, head_k_dim, head_v_dim)`` — the outer-product state
        ``k (x) v`` threaded across chunks/tokens.
        """
        return (self.linear_num_value_heads, self.linear_key_head_dim, self.linear_value_head_dim)

    @property
    def recurrent_state_elements_per_layer(self) -> int:
        v, k, vd = self.recurrent_state_shape
        return v * k * vd

    @property
    def recurrent_state_bytes_per_layer(self, dtype_size: int = 4) -> int:
        return self.recurrent_state_elements_per_layer * dtype_size

    def recurrent_state_total_mib(self, dtype_size: int = 4) -> float:
        """Total recurrent state across all GDN layers, in MiB."""
        total = self.recurrent_state_elements_per_layer * self.num_gdn_layers * dtype_size
        return total / (1024 * 1024)

    # --- conv state (decode only) ---
    @property
    def conv_state_elements_per_layer(self) -> int:
        return self.conv_dim * self.linear_conv_kernel_dim

    def conv_state_total_mib(self, dtype_size: int = 4) -> float:
        total = self.conv_state_elements_per_layer * self.num_gdn_layers * dtype_size
        return total / (1024 * 1024)

    # --- KV cache (full-attention layers) ---
    def kv_cache_bytes_per_token(self, dtype_size: int = 2) -> int:
        """Bytes of KV cache per token, per layer, for one full-attention layer."""
        floats = self.num_key_value_heads * self.full_attn_head_dim * 2  # K + V
        return floats * dtype_size

    def kv_cache_mib_at_context(self, context_length: int, dtype_size: int = 2) -> float:
        """Total KV cache (all FA layers) at a given context length, in MiB."""
        floats = (
            self.num_key_value_heads
            * self.full_attn_head_dim
            * 2
            * context_length
            * self.num_full_attention_layers
        )
        return floats * dtype_size / (1024 * 1024)

    # --- layer placement ---
    @property
    def layer_types(self) -> list[str]:
        """Reconstruct the ``layer_types`` list from the pattern."""
        interval = 4  # full attention every 4th layer
        result: list[str] = []
        for i in range(self.num_hidden_layers):
            if (i + 1) % interval == 0:
                result.append("full_attention")
            else:
                result.append("linear_attention")
        return result

    @property
    def full_attention_layer_indices(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    @property
    def gdn_layer_indices(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]


# ---------------------------------------------------------------------------
# Verified checkpoints
# ---------------------------------------------------------------------------

_4B = CheckpointLayerInfo(
    name="4B",
    hf_repo="Qwen/Qwen3.5-4B",
    hidden_size=2560,
    num_hidden_layers=32,
    num_gdn_layers=24,
    num_full_attention_layers=8,
    layer_pattern="8x(3 GDN -> 1 full)",
    linear_conv_kernel_dim=4,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    num_attention_heads=16,
    num_key_value_heads=4,
    full_attn_head_dim=256,
    partial_rotary_factor=0.25,
)

_0_8B = CheckpointLayerInfo(
    name="0.8B",
    hf_repo="Qwen/Qwen3.5-0.8B",
    hidden_size=1024,
    num_hidden_layers=24,
    num_gdn_layers=18,
    num_full_attention_layers=6,
    layer_pattern="6x(3 GDN -> 1 full)",
    linear_conv_kernel_dim=4,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_num_key_heads=16,
    linear_num_value_heads=16,
    num_attention_heads=8,
    num_key_value_heads=2,
    full_attn_head_dim=256,
    partial_rotary_factor=0.25,
)

#: Lookup by name.  Primary checkpoint is "4B"; fallback is "0.8B" (ADR 0003).
LAYER_INFO: dict[str, CheckpointLayerInfo] = {
    "4B": _4B,
    "0.8B": _0_8B,
}
