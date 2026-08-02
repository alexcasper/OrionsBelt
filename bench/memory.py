"""Three-way memory attribution: weights, KV cache, GDN recurrent state.

This module computes the three memory components analytically from model
configuration, following ``docs/METRICS.md`` §5. It does NOT use process RSS
(RSS cannot be split into these components — see METRICS.md §5.0).

All figures are derived from the checkpoint's ``config.json`` and the run's
quantization/precision settings, cross-checked against the modeling-code audit
in ``docs/GDN_LAYER_AUDIT.md``.

Usage::

    from bench.memory import MemoryModel, compute_memory
    model = MemoryModel.from_config(config_dict)
    snap = compute_memory(model, context_length=32768, generated_tokens=0)
    print(snap.weights, snap.kv_cache, snap.recurrent_state)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Bytes per element for common dtypes
DTYPE_BYTES: dict[str, int] = {
    "fp32": 4,
    "float32": 4,
    "fp16": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 1,  # packed 2 per byte, but allocator rounds up; use 1 for ceiling
    "uint8": 1,
}


@dataclass
class MemoryModel:
    """Structural parameters of a model relevant to memory attribution.

    All values are read from the checkpoint's ``config.json`` (text_config section
    for Qwen3.5). See ``docs/GDN_LAYER_AUDIT.md`` for the verified ground truth.
    """

    hidden_size: int
    num_gdn_layers: int
    num_full_attn_layers: int
    # GDN linear-attention dimensions
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    # Full-attention dimensions
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    # State precision (config: mamba_ssm_dtype)
    state_dtype: str = "float32"
    # Quantization (default fp16 weights)
    weight_dtype: str = "float16"
    cache_dtype: str = "float16"
    # Parameter count (for weight estimation; None = estimate from architecture)
    num_parameters: int | None = None

    @property
    def num_layers(self) -> int:
        return self.num_gdn_layers + self.num_full_attn_layers

    @property
    def key_dim(self) -> int:
        """Total key dimension = num_key_heads × key_head_dim."""
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def value_dim(self) -> int:
        """Total value dimension = num_value_heads × value_head_dim."""
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def conv_dim(self) -> int:
        """Conv1D input channels = key_dim * 2 + value_dim (Q,K concatenated with V)."""
        return self.key_dim * 2 + self.value_dim

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MemoryModel:
        """Build a MemoryModel from a parsed config.json dict.

        Handles both top-level config and nested ``text_config`` (Qwen3.5 multimodal).
        """
        tc = config.get("text_config", config)

        layer_types = tc.get("layer_types", [])
        num_gdn = sum(1 for lt in layer_types if "linear" in lt)
        num_full = sum(1 for lt in layer_types if "full" in lt)

        # Fallback to computed ratio if layer_types missing
        if not layer_types:
            total = tc.get("num_hidden_layers", 32)
            interval = tc.get("full_attention_interval", 4)
            num_full = total // interval
            num_gdn = total - num_full

        return cls(
            hidden_size=tc["hidden_size"],
            num_gdn_layers=num_gdn,
            num_full_attn_layers=num_full,
            linear_num_key_heads=tc.get("linear_num_key_heads", 16),
            linear_num_value_heads=tc.get("linear_num_value_heads", 32),
            linear_key_head_dim=tc.get("linear_key_head_dim", 128),
            linear_value_head_dim=tc.get("linear_value_head_dim", 128),
            linear_conv_kernel_dim=tc.get("linear_conv_kernel_dim", 4),
            num_attention_heads=tc.get("num_attention_heads", 16),
            num_key_value_heads=tc.get("num_key_value_heads", 4),
            head_dim=tc.get("head_dim", 256),
            state_dtype=tc.get("mamba_ssm_dtype", "float32"),
        )

    @classmethod
    def qwen35_4b(cls) -> MemoryModel:
        """Pre-configured for Qwen3.5-4B (primary checkpoint)."""
        return cls(
            hidden_size=2560,
            num_gdn_layers=24,
            num_full_attn_layers=8,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=256,
            state_dtype="float32",
            num_parameters=4_000_000_000,
        )

    @classmethod
    def qwen35_0_8b(cls) -> MemoryModel:
        """Pre-configured for Qwen3.5-0.8B (fallback checkpoint)."""
        return cls(
            hidden_size=1024,
            num_gdn_layers=18,
            num_full_attn_layers=6,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=256,
            state_dtype="float32",
            num_parameters=800_000_000,
        )


@dataclass
class MemoryBreakdown:
    """Three-way memory attribution result.

    All values are exact integer byte counts, as required by METRICS.md §5.5.
    """

    weights: int
    kv_cache: int
    recurrent_state: int
    # Supplementary (included in recurrent_state but broken out for analysis)
    conv_state: int = 0
    # Metadata
    context_length: int = 0
    generated_tokens: int = 0
    phase: str = ""
    model_label: str = ""

    @property
    def total(self) -> int:
        return self.weights + self.kv_cache + self.recurrent_state

    @property
    def total_mib(self) -> float:
        return self.total / (1024 * 1024)

    @property
    def weights_mib(self) -> float:
        return self.weights / (1024 * 1024)

    @property
    def kv_cache_mib(self) -> float:
        return self.kv_cache / (1024 * 1024)

    @property
    def recurrent_state_mib(self) -> float:
        return self.recurrent_state / (1024 * 1024)


def _dtype_size(dtype: str) -> int:
    """Bytes per element for a dtype string."""
    return DTYPE_BYTES.get(dtype, DTYPE_BYTES.get(dtype.lower(), 4))


def compute_weights(model: MemoryModel, weight_dtype: str | None = None) -> int:
    """Compute total weight bytes from parameter count and dtype.

    If ``num_parameters`` is None, estimates from architecture:
    in_proj_qkv + in_proj_z + in_proj_b + in_proj_a + out_proj + conv1d + embeddings,
    per GDN layer, plus full-attention layers and embeddings.
    """
    dtype = weight_dtype or model.weight_dtype
    bpe = _dtype_size(dtype)

    if model.num_parameters is not None:
        return model.num_parameters * bpe

    # Architectural estimate (rough, for when exact param count is unknown)
    hs = model.hidden_size
    # Per GDN layer: in_proj_qkv (hs → 2*key_dim+value_dim) + in_proj_z (hs → value_dim)
    # + in_proj_b (hs → n_v_heads) + in_proj_a (hs → n_v_heads) + out_proj (value_dim → hs)
    # + conv1d (conv_dim × kernel) + A_log (n_v_heads) + dt_bias (n_v_heads) + norm (v_head_dim)
    gdn_per_layer = (
        hs * (2 * model.key_dim + model.value_dim)  # in_proj_qkv
        + hs * model.value_dim                      # in_proj_z
        + hs * model.linear_num_value_heads          # in_proj_b
        + hs * model.linear_num_value_heads          # in_proj_a
        + model.value_dim * hs                       # out_proj
        + model.conv_dim * model.linear_conv_kernel_dim  # conv1d
        + model.linear_num_value_heads * 2           # A_log + dt_bias
        + model.linear_value_head_dim               # norm
    )
    # Per full-attn layer: q_proj + k_proj + v_proj + o_proj (with GQA)
    attn_per_layer = (
        hs * (model.num_attention_heads * model.head_dim)   # q_proj
        + hs * (model.num_key_value_heads * model.head_dim)  # k_proj
        + hs * (model.num_key_value_heads * model.head_dim)  # v_proj
        + (model.num_attention_heads * model.head_dim) * hs  # o_proj
    )
    # FFN per layer (SiLU, intermediate_size typically ~3-4× hidden)
    # Estimate intermediate as 3.6× hidden (Qwen3.5-4B: 9216/2560 ≈ 3.6)
    inter = int(hs * 3.6)
    ffn_per_layer = hs * inter * 3  # gate + up + down
    # Embeddings (tied)
    vocab_size = 248320  # Qwen3.5 vocab
    embedding = vocab_size * hs

    total_params = (
        model.num_gdn_layers * gdn_per_layer
        + model.num_full_attn_layers * attn_per_layer
        + model.num_layers * ffn_per_layer
        + embedding  # tied embeddings, counted once
    )
    return total_params * bpe


def compute_kv_cache(
    model: MemoryModel,
    total_tokens: int,
    cache_dtype: str | None = None,
) -> int:
    """Compute KV cache bytes for full-attention layers at a given sequence length.

    Formula (METRICS.md §5.3)::

        kv_cache = num_full_attn_layers × 2 (K and V)
                  × batch (= 1)
                  × seq_len
                  × n_kv_heads × head_dim
                  × bytes_per_element
    """
    dtype = cache_dtype or model.cache_dtype
    bpe = _dtype_size(dtype)
    return (
        model.num_full_attn_layers
        * 2  # K and V
        * total_tokens
        * model.num_key_value_heads
        * model.head_dim
        * bpe
    )


def compute_recurrent_state(
    model: MemoryModel,
    state_dtype: str | None = None,
) -> int:
    """Compute GDN recurrent state bytes (conv_state included).

    Formula (METRICS.md §5.4, GDN_LAYER_AUDIT.md §3)::

        recurrent_state = num_gdn_layers × n_v_heads × d_k × d_v × bpe
                        + num_gdn_layers × conv_dim × kernel_size × bpe

    The conv_state is O(1) (kernel_size is fixed at 4), so the total is flat
    regardless of context length — this is the project's central memory claim.
    """
    dtype = state_dtype or model.state_dtype
    bpe = _dtype_size(dtype)

    # Recurrent state: (batch, n_v_heads, d_k, d_v) per layer
    state_per_layer = (
        model.linear_num_value_heads
        * model.linear_key_head_dim
        * model.linear_value_head_dim
        * bpe
    )

    # Conv1D state: (batch, conv_dim, kernel_size) per layer
    conv_per_layer = model.conv_dim * model.linear_conv_kernel_dim * bpe

    return model.num_gdn_layers * (state_per_layer + conv_per_layer)


def compute_memory(
    model: MemoryModel,
    context_length: int,
    generated_tokens: int = 0,
    phase: str = "prefill",
    weight_dtype: str | None = None,
    cache_dtype: str | None = None,
    state_dtype: str | None = None,
) -> MemoryBreakdown:
    """Compute the full three-way memory breakdown for a point in generation.

    This is the main entry point. At prefill end, ``generated_tokens=0``;
    at decode end, ``generated_tokens=N-1``.
    """
    total_tokens = context_length + generated_tokens
    return MemoryBreakdown(
        weights=compute_weights(model, weight_dtype),
        kv_cache=compute_kv_cache(model, total_tokens, cache_dtype),
        recurrent_state=compute_recurrent_state(model, state_dtype),
        conv_state=compute_recurrent_state(model, state_dtype)
        - model.num_gdn_layers
        * model.linear_num_value_heads
        * model.linear_key_head_dim
        * model.linear_value_head_dim
        * _dtype_size(state_dtype or model.state_dtype),
        context_length=context_length,
        generated_tokens=generated_tokens,
        phase=phase,
    )


def memory_scaling_table(
    model: MemoryModel,
    context_lengths: tuple[int, ...] = (4096, 32768, 131072, 262144),
    generated_tokens: int = 0,
) -> list[MemoryBreakdown]:
    """Generate a memory scaling table across canonical context lengths.

    Returns a list of MemoryBreakdown, one per context length, showing
    how kv_cache grows while weights and recurrent_state stay flat.
    """
    return [
        compute_memory(model, ctx, generated_tokens, "prefill")
        for ctx in context_lengths
    ]


def format_scaling_table(breakdowns: list[MemoryBreakdown]) -> str:
    """Format a memory scaling table as a markdown string."""
    lines = [
        "| Context | Weights (MiB) | KV cache (MiB) | Recurrent state (MiB) | Total (MiB) | KV:State |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for b in breakdowns:
        ratio = b.kv_cache / b.recurrent_state if b.recurrent_state > 0 else 0
        lines.append(
            f"| {b.context_length:,} | {b.weights_mib:,.1f} | {b.kv_cache_mib:,.1f} "
            f"| {b.recurrent_state_mib:,.1f} | {b.total_mib:,.1f} | {ratio:.1f}× |"
        )
    return "\n".join(lines)
