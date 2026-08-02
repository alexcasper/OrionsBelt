#!/usr/bin/env python3
"""Memory instrumentation: three-component attribution from model config.

Bead ``ob-vfp`` (``t-harness-mem``). This is **the single most important
measurement in the project** — the three-way split that makes the GDN scaling
advantage visible: weights (flat), KV cache (grows linearly with context),
recurrent state (O(1), flat).

All formulas follow ``docs/METRICS.md`` section 5 and the ground-truth layer
shapes audited in ``docs/GDN_LAYER_AUDIT.md`` (bead ob-37v). Every byte count
is derived from model introspection / known tensor shapes, never from process
RSS (METRICS.md section 5.0 — RSS cannot be split into these components).

Stdlib-only: no torch, no transformers, no numpy. The config is passed as a
plain dict (read from the checkpoint's ``config.json`` ``text_config``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """The subset of checkpoint config fields needed for memory accounting.

    All values are read from ``config.json`` ``text_config`` — see
    ``docs/GDN_LAYER_AUDIT.md`` for the source mapping.
    """

    hidden_size: int
    num_hidden_layers: int
    # Full-attention config
    num_attention_heads: int
    num_key_value_heads: int
    full_attn_head_dim: int
    # GDN linear-attention config
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    # MLP config
    intermediate_size: int
    # Vocabulary
    vocab_size: int
    tie_word_embeddings: bool = True
    # Precision
    weight_dtype_bytes: int = 2  # bf16/fp16
    cache_dtype_bytes: int = 2  # KV cache in fp16/bf16
    state_dtype_bytes: int = 4  # recurrent state in fp32 (mamba_ssm_dtype)

    @classmethod
    def from_hf_config(cls, config: dict) -> ModelConfig:
        """Build from a HuggingFace config.json dict (top-level or text_config)."""
        tc = config.get("text_config", config)
        return cls(
            hidden_size=tc["hidden_size"],
            num_hidden_layers=tc["num_hidden_layers"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            full_attn_head_dim=tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"]),
            linear_num_value_heads=tc["linear_num_value_heads"],
            linear_num_key_heads=tc["linear_num_key_heads"],
            linear_key_head_dim=tc["linear_key_head_dim"],
            linear_value_head_dim=tc["linear_value_head_dim"],
            linear_conv_kernel_dim=tc.get("linear_conv_kernel_dim", 4),
            intermediate_size=tc["intermediate_size"],
            vocab_size=tc["vocab_size"],
            tie_word_embeddings=tc.get("tie_word_embeddings", True),
        )

    @property
    def layer_types(self) -> list[str]:
        """Reconstruct layer_types from full_attention_interval (=4, so 3:1)."""
        interval = 4
        return [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(self.num_hidden_layers)
        ]

    @property
    def num_gdn_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "linear_attention")

    @property
    def num_full_attention_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "full_attention")

    @property
    def key_dim(self) -> int:
        return self.linear_key_head_dim * self.linear_num_key_heads

    @property
    def value_dim(self) -> int:
        return self.linear_value_head_dim * self.linear_num_value_heads

    @property
    def conv_dim(self) -> int:
        return self.key_dim * 2 + self.value_dim


# ---------------------------------------------------------------------------
# Weight accounting — analytical from config dimensions
# ---------------------------------------------------------------------------


def _gdn_layer_params(cfg: ModelConfig) -> int:
    """Parameter count for one GDN (linear attention) layer's projections."""
    h = cfg.hidden_size
    p = (
        # in_proj_qkv: hidden -> key_dim*2 + value_dim
        h * (cfg.key_dim * 2 + cfg.value_dim)
        # in_proj_z: hidden -> value_dim
        + h * cfg.value_dim
        # in_proj_b: hidden -> num_v_heads
        + h * cfg.linear_num_value_heads
        # in_proj_a: hidden -> num_v_heads
        + h * cfg.linear_num_value_heads
        # out_proj: value_dim -> hidden
        + cfg.value_dim * h
        # conv1d: depthwise, conv_dim channels × kernel_size (no bias)
        + cfg.conv_dim * cfg.linear_conv_kernel_dim
        # A_log + dt_bias: num_v_heads each
        + cfg.linear_num_value_heads  # A_log
        + cfg.linear_num_value_heads  # dt_bias
        # norm weight: head_v_dim per head → total value_dim
        + cfg.value_dim
        # layernorms: input_layernorm + post_attention_layernorm
        + h * 2
    )
    return p


def _attention_layer_params(cfg: ModelConfig) -> int:
    """Parameter count for one full-attention layer's projections."""
    h = cfg.hidden_size
    hd = cfg.full_attn_head_dim
    nah = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    p = (
        # q_proj: hidden -> num_heads × head_dim × 2 (doubled for output gate)
        h * nah * hd * 2
        # k_proj: hidden -> n_kv_heads × head_dim
        + h * nkv * hd
        # v_proj: hidden -> n_kv_heads × head_dim
        + h * nkv * hd
        # o_proj: num_heads × head_dim -> hidden
        + nah * hd * h
        # q_norm + k_norm: head_dim each
        + hd * 2
        # layernorms
        + h * 2
    )
    return p


def _mlp_params(cfg: ModelConfig) -> int:
    """Parameter count for one SwiGLU MLP block."""
    h = cfg.hidden_size
    i = cfg.intermediate_size
    return h * i + h * i + i * h  # gate_proj + up_proj + down_proj


def weight_bytes(cfg: ModelConfig) -> int:
    """Total weight parameter bytes at the configured runtime dtype.

    Analytical from config dimensions. Cross-check against the checkpoint's
    safetensors index when available (METRICS.md section 5.2 — introspect, don't
    assume). This formula is the fallback when introspection isn't available.
    """
    gdn = _gdn_layer_params(cfg) * cfg.num_gdn_layers
    attn = _attention_layer_params(cfg) * cfg.num_full_attention_layers
    mlp = _mlp_params(cfg) * cfg.num_hidden_layers

    # Embedding (tied or untied)
    embed = cfg.vocab_size * cfg.hidden_size
    if not cfg.tie_word_embeddings:
        embed *= 2  # separate lm_head

    # Final norm
    final_norm = cfg.hidden_size

    total_params = gdn + attn + mlp + embed + final_norm
    return total_params * cfg.weight_dtype_bytes


# ---------------------------------------------------------------------------
# KV cache — grows linearly with context length (full-attention layers only)
# ---------------------------------------------------------------------------


def kv_cache_bytes(cfg: ModelConfig, seq_len: int, batch: int = 1) -> int:
    """Full-attention KV cache bytes at ``seq_len`` tokens.

    Formula from METRICS.md section 5.3:
    ``num_full_attention_layers × 2 × batch × seq_len × n_kv_heads × head_dim × dtype_bytes``
    """
    return (
        cfg.num_full_attention_layers
        * 2  # K and V
        * batch
        * seq_len
        * cfg.num_key_value_heads
        * cfg.full_attn_head_dim
        * cfg.cache_dtype_bytes
    )


# ---------------------------------------------------------------------------
# Recurrent state — O(1), flat regardless of context length (GDN layers only)
# ---------------------------------------------------------------------------


def recurrent_state_bytes(cfg: ModelConfig, batch: int = 1) -> int:
    """GDN recurrent state bytes — flat at every context length.

    Formula from METRICS.md section 5.4 and GDN_LAYER_AUDIT.md section 2:
    ``num_gdn_layers × n_v_heads × d_k × d_v × batch × state_dtype_bytes``

    The state is fp32 (``mamba_ssm_dtype = 'float32'``) regardless of model dtype.
    """
    return (
        cfg.num_gdn_layers
        * cfg.linear_num_value_heads
        * cfg.linear_key_head_dim
        * cfg.linear_value_head_dim
        * batch
        * cfg.state_dtype_bytes
    )


# ---------------------------------------------------------------------------
# Combined three-way breakdown
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryBreakdown:
    """The three-component memory attribution — the project's central claim."""

    weights: int
    kv_cache: int
    recurrent_state: int

    @property
    def total(self) -> int:
        return self.weights + self.kv_cache + self.recurrent_state


def memory_breakdown(
    cfg: ModelConfig, seq_len: int, batch: int = 1
) -> MemoryBreakdown:
    """Compute the full three-way memory breakdown at ``seq_len`` tokens.

    - ``weights``: flat (does not depend on seq_len)
    - ``kv_cache``: grows linearly with seq_len
    - ``recurrent_state``: O(1), flat at every context length
    """
    return MemoryBreakdown(
        weights=weight_bytes(cfg),
        kv_cache=kv_cache_bytes(cfg, seq_len, batch),
        recurrent_state=recurrent_state_bytes(cfg, batch),
    )


# ---------------------------------------------------------------------------
# Context sweep — the headline numbers at each canonical context length
# ---------------------------------------------------------------------------

CANONICAL_CONTEXT_LENGTHS = (4096, 32768, 131072, 262144)


def context_sweep(
    cfg: ModelConfig,
    context_lengths: tuple[int, ...] = CANONICAL_CONTEXT_LENGTHS,
    batch: int = 1,
) -> list[dict]:
    """Memory breakdown at each context length — the scaling plot data.

    Each entry has: context_length, weights_gib, kv_cache_gib,
    recurrent_state_mib, total_gib.
    """
    results = []
    w = weight_bytes(cfg)
    rs = recurrent_state_bytes(cfg, batch)
    for ctx in context_lengths:
        kv = kv_cache_bytes(cfg, ctx, batch)
        results.append({
            "context_length": ctx,
            "weights_gib": w / (1024**3),
            "kv_cache_gib": kv / (1024**3),
            "recurrent_state_mib": rs / (1024**2),
            "total_gib": (w + kv + rs) / (1024**3),
        })
    return results


__all__ = [
    "ModelConfig",
    "MemoryBreakdown",
    "weight_bytes",
    "kv_cache_bytes",
    "recurrent_state_bytes",
    "memory_breakdown",
    "context_sweep",
    "CANONICAL_CONTEXT_LENGTHS",
]
