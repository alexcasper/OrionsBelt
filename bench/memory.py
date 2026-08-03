"""Standalone memory accounting for GDN hybrid models (ob-vfp).

Computes the three-way memory decomposition — **weights**, **KV cache**,
**recurrent state** — from a ``ModelConfig`` and a sequence length. This is the
load-bearing measurement of the project: one component grows linearly with
context while the other stays O(1) flat, and the contrast *is* the architectural
advantage GDN provides on memory-constrained edge silicon.

Per ``docs/METRICS.md`` section 5.0: **none of these three components may be
derived from process RSS.** They are computed analytically from the checkpoint's
config and the run's quantization/precision, cross-checked against introspected
tensor shapes where a running model is available (``cross_check``).

Stdlib-only, same constraint as ``bench/schema.py``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bench.harness import ModelConfig


# ---------------------------------------------------------------------------
# Analytic formulas (METRICS.md section 5, confirmed by ob-37v)
# ---------------------------------------------------------------------------


def weights_bytes(config: ModelConfig) -> int:
    """Total parameter tensor bytes.

    Flat across context length — weights don't depend on how much has been
    processed (METRICS.md section 5.2). Computed from ``num_params`` and the
    weight dtype, which should be read from the live model's tensors when
    available rather than assumed from the config string.
    """
    return config.num_params * config.weight_dtype_bytes


def kv_cache_bytes(config: ModelConfig, seq_len: int) -> int:
    """Full-attention layers' key/value cache bytes at ``seq_len`` tokens.

    Grows linearly with ``seq_len`` — this is the component that scales
    (METRICS.md section 5.3). Only the full-attention layers (8 of 32 in the
    4B) maintain a KV cache; GDN layers do not.

    Formula: ``num_full_attention_layers × 2 (K+V) × seq_len × n_kv_heads
    × head_dim × cache_dtype_bytes``
    """
    return (
        config.num_full_attention_layers
        * 2  # K and V
        * seq_len
        * config.fa_n_kv_heads
        * config.fa_head_dim
        * config.cache_dtype_bytes
    )


def recurrent_state_bytes(config: ModelConfig) -> int:
    """GDN layers' recurrent state bytes.

    O(1) per token — constant regardless of context length (METRICS.md
    section 5.4). The state shape ``(batch, num_v_heads, head_k_dim,
    head_v_dim)`` was confirmed from the modeling code (ob-37v, FINDINGS.md
    section 6). State is kept in FP32 per the quantization policy carve-out
    (QUANTIZATION_POLICY.md section 1).
    """
    return (
        config.num_gdn_layers
        * config.linear_num_value_heads
        * config.linear_key_head_dim
        * config.linear_value_head_dim
        * config.state_dtype_bytes
    )


# ---------------------------------------------------------------------------
# Full decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryBreakdown:
    """One row of the three-way memory decomposition at a given context length."""

    context_length: int
    weights: int
    kv_cache: int
    recurrent_state: int

    @property
    def total(self) -> int:
        return self.weights + self.kv_cache + self.recurrent_state


def decomposition(
    config: ModelConfig,
    context_lengths: list[int],
    *,
    include_counterfactual: bool = True,
) -> list[dict]:
    """Compute the three-way memory decomposition across context lengths.

    Returns a list of dicts, one per context length, with keys:
    ``context_length``, ``weights``, ``kv_cache``, ``recurrent_state``,
    ``hybrid_total``, and optionally ``all_attention_kv_equivalent`` (the
    counterfactual KV cache if all layers were full attention).

    This is the table ADR 0003 uses to quantify the GDN memory advantage:
    at 262K context the 4B saves ~24.8 GiB versus an all-attention equivalent.
    """
    w = weights_bytes(config)
    rs = recurrent_state_bytes(config)
    total_layers = config.num_gdn_layers + config.num_full_attention_layers
    fa_ratio = (
        total_layers / config.num_full_attention_layers
        if config.num_full_attention_layers > 0
        else 1.0
    )

    rows = []
    for ctx in context_lengths:
        kv = kv_cache_bytes(config, ctx)
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
# Cross-check hook (for live model introspection, METRICS.md section 5.4)
# ---------------------------------------------------------------------------


def cross_check(
    config: ModelConfig,
    *,
    introspected_weights: int | None = None,
    introspected_state_shape: tuple[int, ...] | None = None,
    introspected_state_dtype_bytes: int | None = None,
) -> list[str]:
    """Compare analytic formulas against introspected live tensors.

    Returns a list of discrepancy messages (empty if all checks pass).
    This is the hook METRICS.md section 5.4 requires: "cross-check this
    analytic figure against the actual allocated state-tensor shape read
    from the running model/kernel (introspection), not just the
    config-derived formula."

    When the x86 reference (ob-aqv) lands, call this from the harness with
    the live model's actual tensor shapes to catch implementation bugs that
    would silently falsify the project's central claim (e.g. a bug that lets
    state grow with context).
    """
    discrepancies: list[str] = []

    if introspected_weights is not None:
        expected_w = weights_bytes(config)
        if introspected_weights != expected_w:
            discrepancies.append(
                f"weights mismatch: analytic={expected_w}, introspected={introspected_weights}"
            )

    if introspected_state_shape is not None:
        expected_shape = (
            1,  # batch
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        )
        # Allow batch dimension to differ (introspection might see unbatched)
        if len(introspected_state_shape) == 3:
            introspected_state_shape = (1,) + introspected_state_shape
        if introspected_state_shape != expected_shape:
            discrepancies.append(
                f"recurrent state shape mismatch: analytic={expected_shape}, "
                f"introspected={introspected_state_shape}"
            )

    if (
        introspected_state_dtype_bytes is not None
        and introspected_state_dtype_bytes != config.state_dtype_bytes
    ):
        discrepancies.append(
            f"state dtype mismatch: analytic={config.state_dtype_bytes}, "
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


def print_decomposition(config: ModelConfig, context_lengths: list[int]) -> None:
    """Print a human-readable memory decomposition table to stdout."""
    rows = decomposition(config, context_lengths)
    name = config.name

    print(f"\nMemory decomposition: {name}")
    print(f"  {config.num_gdn_layers} GDN + {config.num_full_attention_layers} FA layers")
    print(
        f"  State: {config.linear_num_value_heads}×{config.linear_key_head_dim}×"
        f"{config.linear_value_head_dim} = "
        f"{_fmt_bytes(recurrent_state_bytes(config))} total, fp{config.state_dtype_bytes * 8}"
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
    "MemoryBreakdown",
    "weights_bytes",
    "kv_cache_bytes",
    "recurrent_state_bytes",
    "decomposition",
    "cross_check",
    "print_decomposition",
]


if __name__ == "__main__":
    # Ensure repo root is importable
    _ROOT = str(Path(__file__).resolve().parent.parent)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    from bench.harness import QWEN35_08B, QWEN35_4B  # noqa: E402

    model = QWEN35_4B if "--4b" in sys.argv or len(sys.argv) < 2 else QWEN35_08B
    ctx_lengths = [4096, 32768, 131072, 262144]
    print_decomposition(model, ctx_lengths)
