# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Three-component memory accounting for GDN hybrid models — the single source
of truth for the predicted ``peak_memory_bytes`` columns (ob-7m6).

Computes the three-way memory decomposition — **weights**, **KV cache**,
**recurrent state** — analytically from a checkpoint's config dimensions. This
is the load-bearing measurement of the project (docs/archive/PLAN.md §2.4): one component
grows linearly with context while another stays O(1) flat, and the contrast
*is* the architectural advantage GDN provides on memory-constrained edge
silicon.

Per ``docs/METRICS.md`` §5.0: **none of these three components may be derived
from process RSS.** They are computed analytically from the checkpoint's config
and the run's quantization/precision, cross-checked against introspected tensor
shapes where a running model is available (``cross_check``). The harness routes
``backend.memory_bytes()`` for the three ``peak_memory_bytes`` component rows
through this module, so the numbers in every emitted CSV originate here — not
from a synthetic/mock backend. Actual RSS-style measurements (when taken) are a
separate cross-check concern and must never be conflated with these predicted
columns (METRICS.md §5.0: RSS cannot be split into these components).

**Provenance (ob-7m6):** the ``ModelConfig.from_hf_config`` classmethod, the
``layer_types`` derivation from ``full_attention_interval``, and the analytical
``weight_bytes`` formula (``_gdn_layer_params`` / ``_attention_layer_params`` /
``_mlp_params``) are ported from ``origin/bench/t4:bench/metrics.py``, the most
rigorous config-driven accounting of the four competing implementations. All
formulas follow ``docs/METRICS.md`` §5 and the ground-truth layer shapes audited
in ``docs/GDN_LAYER_AUDIT.md`` (bead ob-37v).

Stdlib-only: no torch, no transformers, no numpy. The config is passed as a
plain dict (read from the checkpoint's ``config.json``, optionally nested under
``text_config``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Dtype mapping (config string → bytes per element)
# ---------------------------------------------------------------------------

_DTYPE_BYTES = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
}


# ---------------------------------------------------------------------------
# ModelConfig — the subset of checkpoint config fields needed for accounting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Checkpoint dimensions needed for analytic memory attribution.

    All values are read from ``config.json`` ``text_config`` — see
    ``docs/GDN_LAYER_AUDIT.md`` for the source mapping. ``from_hf_config`` is
    the config-driven constructor ported from ``origin/bench/t4``; the layer
    counts (``num_gdn_layers`` / ``num_full_attention_layers``) are always
    *derived* from ``layer_types`` (itself derived from
    ``full_attention_interval``), never hardcoded or assumed round.
    """

    # Identity
    name: str = ""
    # Core transformer dimensions
    hidden_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    # Full-attention config (METRICS.md §5.3)
    num_key_value_heads: int = 0
    full_attn_head_dim: int = 0
    # GDN linear-attention config (METRICS.md §5.4)
    linear_num_value_heads: int = 0
    linear_num_key_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_conv_kernel_dim: int = 4
    # MLP config
    intermediate_size: int = 0
    # Vocabulary
    vocab_size: int = 0
    tie_word_embeddings: bool = True
    # Layer structure: derive layer_types from full_attention_interval (t4).
    # GDN_LAYER_AUDIT.md §1: every 4th layer is full attention (3:1 ratio).
    full_attention_interval: int = 4
    # Precision — kept as separate fields per METRICS.md §5.2/§5.3/§5.4 so a
    # quantization policy can change one without affecting the others.
    weight_dtype_bytes: int = 2  # bf16/fp16 weights
    cache_dtype_bytes: int = 2  # KV cache in fp16/bf16
    state_dtype_bytes: int = 4  # recurrent state in fp32 (mamba_ssm_dtype)

    @classmethod
    def from_hf_config(cls, config: dict, name: str = "") -> ModelConfig:
        """Build from a HuggingFace ``config.json`` dict (top-level or text_config).

        Ported from ``origin/bench/t4:bench/metrics.py``. Reads the real
        checkpoint config, resolves the layer structure, and maps
        ``mamba_ssm_dtype`` to ``state_dtype_bytes``. Stdlib-only.

        **Layer-structure resolution.** Real Qwen3.5 configs carry an explicit
        per-layer ``layer_types`` array (GDN_LAYER_AUDIT.md §1: ground truth).
        When present, it is translated to the equivalent
        ``num_hidden_layers`` + ``full_attention_interval`` so the
        interval-derived ``layer_types`` property reproduces it exactly. When
        absent, both are read from the config (``full_attention_interval``
        defaults to 4). Either way the count is *derived* via the t4
        comprehension, never hardcoded.
        """
        tc = config.get("text_config", config)

        explicit_layers = tc.get("layer_types")
        if explicit_layers:
            # Translate the explicit ground-truth list into the equivalent
            # interval. Qwen3.5 places full attention every Nth layer
            # (GDN_LAYER_AUDIT.md §1), so the first "full_attention" index + 1
            # recovers the interval; the property then reproduces the list.
            num_hidden_layers = len(explicit_layers)
            fa_indices = [i for i, t in enumerate(explicit_layers) if t == "full_attention"]
            full_attention_interval = fa_indices[0] + 1 if fa_indices else num_hidden_layers + 1
        else:
            num_hidden_layers = tc.get("num_hidden_layers", 0)
            full_attention_interval = tc.get("full_attention_interval", 4)

        state_dtype = tc.get("mamba_ssm_dtype", "float32")
        state_dtype_bytes = _DTYPE_BYTES.get(state_dtype, 4)

        # head_dim defaults to hidden_size // num_attention_heads when not set
        # (t4); guard the division for partial configs that omit both.
        head_dim = tc.get("head_dim")
        if not head_dim:
            h = tc.get("hidden_size", 0)
            nah = tc.get("num_attention_heads", 0)
            head_dim = (h // nah) if nah else 0

        return cls(
            name=name or tc.get("model_type", config.get("model_type", "unknown")),
            hidden_size=tc.get("hidden_size", 0),
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=tc.get("num_attention_heads", 0),
            num_key_value_heads=tc.get("num_key_value_heads", 0),
            full_attn_head_dim=head_dim,
            linear_num_value_heads=tc.get("linear_num_value_heads", 0),
            linear_num_key_heads=tc.get("linear_num_key_heads", 0),
            linear_key_head_dim=tc.get("linear_key_head_dim", 0),
            linear_value_head_dim=tc.get("linear_value_head_dim", 0),
            linear_conv_kernel_dim=tc.get("linear_conv_kernel_dim", 4),
            intermediate_size=tc.get("intermediate_size", 0),
            vocab_size=tc.get("vocab_size", 0),
            tie_word_embeddings=tc.get("tie_word_embeddings", True),
            full_attention_interval=full_attention_interval,
            state_dtype_bytes=state_dtype_bytes,
        )

    @property
    def layer_types(self) -> list[str]:
        """Reconstruct ``layer_types`` from ``full_attention_interval`` (t4).

        Every ``interval``-th layer (1-indexed) is full attention; the rest are
        Gated DeltaNet (linear attention). For the verified Qwen3.5 configs
        ``interval = 4`` → 3:1 GDN:full-attn ratio (GDN_LAYER_AUDIT.md §1).
        """
        interval = self.full_attention_interval
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

    # Convenience: aggregate GDN projection widths (GDN_LAYER_AUDIT.md §2/§8).

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
# Weight accounting — analytical from config dimensions (ported from t4)
# ---------------------------------------------------------------------------


def _gdn_layer_params(cfg: ModelConfig) -> int:
    """Parameter count for one GDN (linear attention) layer's projections.

    Shapes from GDN_LAYER_AUDIT.md §8 (the projections table), verified against
    the transformers ``Qwen3_5GatedDeltaNet`` modeling code (ob-37v).
    """
    h = cfg.hidden_size
    return (
        # in_proj_qkv: hidden -> key_dim*2 + value_dim
        h * (cfg.key_dim * 2 + cfg.value_dim)
        # in_proj_z: hidden -> value_dim (gate for RMSNorm)
        + h * cfg.value_dim
        # in_proj_b: hidden -> num_v_heads (write gate beta)
        + h * cfg.linear_num_value_heads
        # in_proj_a: hidden -> num_v_heads (decay gate alpha)
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


def _attention_layer_params(cfg: ModelConfig) -> int:
    """Parameter count for one full-attention layer's projections."""
    h = cfg.hidden_size
    hd = cfg.full_attn_head_dim
    nah = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    return (
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


def _mlp_params(cfg: ModelConfig) -> int:
    """Parameter count for one SwiGLU MLP block (gate_proj + up_proj + down_proj)."""
    h = cfg.hidden_size
    i = cfg.intermediate_size
    return h * i + h * i + i * h


def weights_bytes(cfg: ModelConfig) -> int:
    """Total weight parameter bytes at the configured runtime dtype.

    Analytical from config dimensions (ported from t4). Flat across context
    length — weights don't depend on how much has been processed
    (METRICS.md §5.2). Cross-check against the checkpoint's safetensors index
    when available (METRICS.md §5.2 — introspect, don't assume); this formula
    is the fallback when introspection isn't available, and is always
    preferable to a round assumed ``num_params`` figure.
    """
    gdn = _gdn_layer_params(cfg) * cfg.num_gdn_layers
    attn = _attention_layer_params(cfg) * cfg.num_full_attention_layers
    mlp = _mlp_params(cfg) * cfg.num_hidden_layers

    # Embedding (tied or untied lm_head)
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

    Formula (METRICS.md §5.3)::

        num_full_attention_layers × 2 (K+V) × batch × seq_len
        × n_kv_heads × head_dim × cache_dtype_bytes

    Grows linearly with ``seq_len`` — this is the component that scales.
    ``num_full_attention_layers`` is read per-checkpoint from ``layer_types``,
    never hardcoded (METRICS.md §5.3).
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

    Formula (METRICS.md §5.4, GDN_LAYER_AUDIT.md §3)::

        num_gdn_layers × H × d_k × d_v × batch × state_dtype_bytes

    where ``H`` = ``linear_num_value_heads`` (NOT key heads —
    GDN_LAYER_AUDIT.md §3 confirms the state shape is
    ``(batch, n_v_heads, d_k, d_v)``). The state is fp32
    (``mamba_ssm_dtype = 'float32'``) regardless of model dtype.
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
    """The three-component memory attribution at a given context length."""

    weights: int
    kv_cache: int
    recurrent_state: int

    @property
    def total(self) -> int:
        return self.weights + self.kv_cache + self.recurrent_state


def memory_breakdown(cfg: ModelConfig, seq_len: int, batch: int = 1) -> MemoryBreakdown:
    """Compute the full three-way memory breakdown at ``seq_len`` tokens.

    - ``weights``: flat (does not depend on seq_len)
    - ``kv_cache``: grows linearly with seq_len
    - ``recurrent_state``: O(1), flat at every context length
    """
    return MemoryBreakdown(
        weights=weights_bytes(cfg),
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
    """Memory breakdown at each context length — the scaling-plot data.

    Each entry has: ``context_length``, ``weights_gib``, ``kv_cache_gib``,
    ``recurrent_state_mib``, ``total_gib``.
    """
    results = []
    w = weights_bytes(cfg)
    rs = recurrent_state_bytes(cfg, batch)
    for ctx in context_lengths:
        kv = kv_cache_bytes(cfg, ctx, batch)
        results.append(
            {
                "context_length": ctx,
                "weights_gib": w / (1024**3),
                "kv_cache_gib": kv / (1024**3),
                "recurrent_state_mib": rs / (1024**2),
                "total_gib": (w + kv + rs) / (1024**3),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Decomposition table (with all-attention counterfactual) — ADR 0003
# ---------------------------------------------------------------------------


def decomposition(
    cfg: ModelConfig,
    context_lengths: list[int],
    *,
    include_counterfactual: bool = True,
) -> list[dict]:
    """Compute the three-way memory decomposition across context lengths.

    Returns a list of dicts, one per context length, with keys:
    ``context_length``, ``weights``, ``kv_cache``, ``recurrent_state``,
    ``hybrid_total``, and (when ``include_counterfactual``) the
    ``all_attention_kv_equivalent`` / ``all_attention_total`` / ``saved_bytes``
    counterfactual — the KV cache if all layers were full attention.

    This is the table ADR 0003 uses to quantify the GDN memory advantage:
    at 262K context the 4B saves ~24.8 GiB versus an all-attention equivalent.
    """
    w = weights_bytes(cfg)
    rs = recurrent_state_bytes(cfg)
    total_layers = cfg.num_gdn_layers + cfg.num_full_attention_layers
    fa_ratio = (
        total_layers / cfg.num_full_attention_layers if cfg.num_full_attention_layers > 0 else 1.0
    )

    rows = []
    for ctx in context_lengths:
        kv = kv_cache_bytes(cfg, ctx)
        row = {
            "context_length": ctx,
            "weights": w,
            "kv_cache": kv,
            "recurrent_state": rs,
            "hybrid_total": w + kv + rs,
        }
        if include_counterfactual:
            row["all_attention_kv_equivalent"] = int(kv * fa_ratio)
            row["all_attention_total"] = w + int(kv * fa_ratio) + rs
            row["saved_bytes"] = row["all_attention_total"] - row["hybrid_total"]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Cross-check hook (for live model introspection, METRICS.md §5.4)
# ---------------------------------------------------------------------------


def cross_check(
    cfg: ModelConfig,
    *,
    introspected_weights: int | None = None,
    introspected_state_shape: tuple[int, ...] | None = None,
    introspected_state_dtype_bytes: int | None = None,
) -> list[str]:
    """Compare analytic formulas against introspected live tensors.

    Returns a list of discrepancy messages (empty if all checks pass).
    This is the hook METRICS.md §5.4 requires: "cross-check this analytic
    figure against the actual allocated state-tensor shape read from the
    running model/kernel (introspection), not just the config-derived formula."
    """
    discrepancies: list[str] = []

    if introspected_weights is not None:
        expected_w = weights_bytes(cfg)
        if introspected_weights != expected_w:
            discrepancies.append(
                f"weights mismatch: analytic={expected_w}, introspected={introspected_weights}"
            )

    if introspected_state_shape is not None:
        expected_shape = (
            1,  # batch
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
        )
        # Allow batch dimension to differ (introspection might see unbatched)
        shape = introspected_state_shape
        if len(shape) == 3:
            shape = (1,) + shape
        if shape != expected_shape:
            discrepancies.append(
                f"recurrent state shape mismatch: analytic={expected_shape}, introspected={shape}"
            )

    if (
        introspected_state_dtype_bytes is not None
        and introspected_state_dtype_bytes != cfg.state_dtype_bytes
    ):
        discrepancies.append(
            f"state dtype mismatch: analytic={cfg.state_dtype_bytes}, "
            f"introspected={introspected_state_dtype_bytes}"
        )

    return discrepancies


# ---------------------------------------------------------------------------
# Human-readable table
# ---------------------------------------------------------------------------


def _fmt_bytes(n: int) -> str:
    """Format bytes as KiB/MiB/GiB."""
    f = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(f) < 1024:
            return f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} PiB"


def print_decomposition(cfg: ModelConfig, context_lengths: list[int]) -> None:
    """Print a human-readable memory decomposition table to stdout."""
    rows = decomposition(cfg, context_lengths)

    print(f"\nMemory decomposition: {cfg.name}")
    print(f"  {cfg.num_gdn_layers} GDN + {cfg.num_full_attention_layers} FA layers")
    print(
        f"  State: {cfg.linear_num_value_heads}×{cfg.linear_key_head_dim}×"
        f"{cfg.linear_value_head_dim} = "
        f"{_fmt_bytes(recurrent_state_bytes(cfg))} total, fp{cfg.state_dtype_bytes * 8}"
    )
    print()
    print(
        f"  {'Context':>8}  {'Weights':>10}  {'KV cache':>10}  {'GDN state':>10}  "
        f"{'Hybrid total':>12}  {'All-attn equiv':>14}  {'Saved':>10}"
    )
    print(f"  {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 12}  {'-' * 14}  {'-' * 10}")

    for row in rows:
        ctx = row["context_length"]
        print(
            f"  {ctx:>8,}  {_fmt_bytes(row['weights']):>10}  "
            f"{_fmt_bytes(row['kv_cache']):>10}  "
            f"{_fmt_bytes(row['recurrent_state']):>10}  "
            f"{_fmt_bytes(row['hybrid_total']):>12}  "
            f"{_fmt_bytes(row['all_attention_kv_equivalent']):>14}  "
            f"{_fmt_bytes(row['saved_bytes']):>10}"
        )
    print()


__all__ = [
    "ModelConfig",
    "MemoryBreakdown",
    "weights_bytes",
    "kv_cache_bytes",
    "recurrent_state_bytes",
    "memory_breakdown",
    "context_sweep",
    "CANONICAL_CONTEXT_LENGTHS",
    "decomposition",
    "cross_check",
    "print_decomposition",
]


if __name__ == "__main__":  # pragma: no cover
    # Ensure repo root is importable
    _ROOT = str(Path(__file__).resolve().parent.parent)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    from bench.harness import QWEN35_08B, QWEN35_4B  # noqa: E402

    model = QWEN35_4B if "--4b" in sys.argv or len(sys.argv) < 2 else QWEN35_08B
    ctx_lengths = [4096, 32768, 131072, 262144]
    print_decomposition(model, ctx_lengths)
