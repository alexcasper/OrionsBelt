#!/usr/bin/env python3
"""Generate memory scaling figures from the analytical model.

The central thesis of OrionsBelt is that GDN's recurrent state is O(1) in
context length, while full-attention's KV cache is O(n).  This script turns
that claim into a regenerable figure + table using the verified shapes in
``src/orionsbelt/engines/memory.py``.

Usage::

    python3 scripts/generate_memory_plots.py
    python3 scripts/generate_memory_plots.py --output-dir results/figures
    python3 scripts/generate_memory_plots.py --text-only  # skip PNG

Outputs (in ``--output-dir``, default ``results/figures``):

    - ``memory_scaling_4b.png``  — stacked bar: weights + KV cache + state
    - ``memory_scaling_0.8b.png`` — same for the 0.8B checkpoint
    - ``memory_decomposition_qwen3.5-4b.png`` — area chart with hypothetical line
    - ``memory_comparison.md`` — markdown table

The figures are regenerated from committed code + verified model config —
no harness runs, no hand-assembled numbers.  If the shapes in
``gdn_layer_info`` are updated, re-running this script picks up the change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable when run as a standalone script
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from orionsbelt.engines.memory import (  # noqa: E402
    predict_breakdown,
    sweep_context,
)
from orionsbelt.model.gdn_layer_info import LAYER_INFO  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTEXT_LENGTHS = [4096, 32768, 131072, 262144]
CONTEXT_LABELS = {4096: "4K", 32768: "32K", 131072: "128K", 262144: "262K"}

# Palette (consistent with original plots.py)
C_WEIGHTS = "#2c3e50"
C_KV_CACHE = "#e74c3c"
C_RECURRENT = "#27ae60"
C_HYPOTHETICAL = "#f39c12"


def _gib(b: int) -> float:
    """Bytes → GiB (1024^3)."""
    return b / (1024**3)


def _mib(b: int) -> float:
    """Bytes → MiB (1024^2)."""
    return b / (1024**2)


def _hypothetical_all_attention_kv(checkpoint: str, context_length: int) -> int:
    """KV cache if ALL layers were full attention (vs the actual 3:1 hybrid).

    The hybrid has ``num_full_attention_layers`` attention layers.
    If all ``num_hidden_layers`` were attention, the KV cache would be
    ``(num_hidden / num_full_attn)`` times larger.
    """
    info = LAYER_INFO[checkpoint]
    actual = predict_breakdown(checkpoint, context_length, kv_cache_dtype="fp16")
    ratio = info.num_hidden_layers / info.num_full_attention_layers
    return int(actual.kv_cache_bytes * ratio)


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def plot_stacked_bar(
    checkpoint: str,
    output_path: str,
    plt,
) -> bool:
    """Stacked bar: weights + KV cache + recurrent state across context lengths.

    Shows the central claim: KV cache grows linearly while state stays flat.
    Includes a hypothetical 'all-attention' line for contrast.
    """
    import matplotlib.ticker as mticker
    import numpy as np

    breakdowns = sweep_context(checkpoint, CONTEXT_LENGTHS)

    weights_gib = [_gib(b.weights_bytes) for b in breakdowns]
    kv_gib = [_gib(b.kv_cache_bytes) for b in breakdowns]
    state_gib = [_gib(b.recurrent_state_bytes + b.conv_state_bytes) for b in breakdowns]
    hyp_kv_gib = [_gib(_hypothetical_all_attention_kv(checkpoint, ctx)) for ctx in CONTEXT_LENGTHS]

    model_label = checkpoint.replace("Qwen3.5-", "")
    info = LAYER_INFO[checkpoint]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CONTEXT_LENGTHS))
    width = 0.55

    # Stacked bars
    ax.bar(x, weights_gib, width, label="Weights (fp16, flat)", color=C_WEIGHTS)
    ax.bar(
        x,
        kv_gib,
        width,
        bottom=weights_gib,
        label=f"KV cache ({info.num_full_attention_layers} attn layers)",
        color=C_KV_CACHE,
    )
    bottom2 = [w + k for w, k in zip(weights_gib, kv_gib, strict=True)]
    ax.bar(
        x,
        state_gib,
        width,
        bottom=bottom2,
        label=f"GDN recurrent state ({_mib(breakdowns[0].recurrent_state_bytes):.0f} MiB, flat)",
        color=C_RECURRENT,
    )

    # Hypothetical all-attention line
    hyp_total = [w + h for w, h in zip(weights_gib, hyp_kv_gib, strict=True)]
    ax.plot(
        x,
        hyp_total,
        "--o",
        color=C_HYPOTHETICAL,
        markersize=5,
        label=f"If all {info.num_hidden_layers} layers were attention",
    )

    # Annotate the KV cache values
    for i, kv in enumerate(kv_gib):
        ax.annotate(
            f"{kv:.1f}",
            (i, weights_gib[i] + kv + 0.3),
            ha="center",
            fontsize=9,
            color=C_KV_CACHE,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CONTEXT_LABELS[c] for c in CONTEXT_LENGTHS])
    ax.set_ylabel("Memory (GiB)")
    ax.set_title(
        f"Peak memory decomposition — Qwen3.5-{model_label}\n"
        f"GDN state is flat; KV cache grows linearly"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def plot_decomposition_area(
    checkpoint: str,
    output_path: str,
    plt,
) -> bool:
    """Stacked area: weights / KV cache / recurrent state vs context length.

    Includes the hypothetical all-attention total as a dashed line.
    """

    # Fine-grained context sweep for smooth area chart
    fine_ctxs = [2048 * (2**i) for i in range(1, 9)]  # 4K to 512K
    breakdowns = sweep_context(checkpoint, fine_ctxs)

    ctxs = [b.context_length for b in breakdowns]
    weights = [_gib(b.weights_bytes) for b in breakdowns]
    kv = [_gib(b.kv_cache_bytes) for b in breakdowns]
    state = [_gib(b.recurrent_state_bytes + b.conv_state_bytes) for b in breakdowns]
    hyp_kv = [_gib(_hypothetical_all_attention_kv(checkpoint, c)) for c in ctxs]

    model_label = checkpoint.replace("Qwen3.5-", "")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(ctxs, 0, weights, alpha=0.6, color=C_WEIGHTS, label="Weights (fp16)")
    ax.fill_between(
        ctxs,
        weights,
        [w + k for w, k in zip(weights, kv, strict=True)],
        alpha=0.6,
        color=C_KV_CACHE,
        label="KV cache (8 attn layers)",
    )
    ax.fill_between(
        ctxs,
        [w + k for w, k in zip(weights, kv, strict=True)],
        [w + k + s for w, k, s in zip(weights, kv, state, strict=True)],
        alpha=0.8,
        color=C_RECURRENT,
        label="GDN recurrent state",
    )

    # Hypothetical all-attention line
    hyp_total = [w + h for w, h in zip(weights, hyp_kv, strict=True)]
    ax.plot(
        ctxs,
        hyp_total,
        "--",
        color=C_HYPOTHETICAL,
        linewidth=2,
        label="If all 32 layers were attention",
    )

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Memory (GiB)")
    ax.set_title(
        f"Memory decomposition — Qwen3.5-{model_label}\nRecurrent state is O(1); KV cache is O(n)"
    )
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------


def generate_comparison_table() -> str:
    """Markdown table comparing memory at each context length for both checkpoints."""
    lines = []
    lines.append("# Memory Scaling: GDN O(1) State vs Attention O(n) KV Cache\n")
    lines.append(
        "_Generated by `scripts/generate_memory_plots.py` from the analytical "
        "model in `src/orionsbelt/engines/memory.py`. "
        "Shapes verified against primary sources (see "
        "[`docs/CLAIM_VERIFICATION.md`](../../docs/CLAIM_VERIFICATION.md))._\n"
    )

    for checkpoint in ["4B", "0.8B"]:
        info = LAYER_INFO[checkpoint]
        lines.append(f"## Qwen3.5-{checkpoint}\n")
        lines.append(
            f"_{info.num_gdn_layers} GDN + {info.num_full_attention_layers} full-attention "
            f"layers (3:1 hybrid), hidden_size={info.hidden_size}_\n"
        )
        lines.append(
            "| Context | Weights (fp16) | KV cache (8 attn) | GDN state | "
            "Total (hybrid) | If all-attn | Savings |"
        )
        lines.append(
            "|---------|----------------|-------------------|-----------|"
            "---------------|-------------|---------|"
        )

        breakdowns = sweep_context(checkpoint, CONTEXT_LENGTHS)
        for b in breakdowns:
            hyp_kv = _hypothetical_all_attention_kv(checkpoint, b.context_length)
            hyp_total = b.weights_bytes + hyp_kv
            savings = hyp_total - b.total_bytes
            ctx_label = CONTEXT_LABELS[b.context_length]
            lines.append(
                f"| {ctx_label} "
                f"| {_gib(b.weights_bytes):.2f} GiB "
                f"| {_gib(b.kv_cache_bytes):.2f} GiB "
                f"| {_mib(b.recurrent_state_bytes + b.conv_state_bytes):.1f} MiB "
                f"| {_gib(b.total_bytes):.2f} GiB "
                f"| {_gib(hyp_total):.2f} GiB "
                f"| {_gib(savings):.2f} GiB |"
            )
        lines.append("")

    # Key insight
    b4b_262k = predict_breakdown("4B", 262144)
    hyp_262k = _hypothetical_all_attention_kv("4B", 262144)
    savings_262k = _gib(b4b_262k.weights_bytes + hyp_262k - b4b_262k.total_bytes)
    total_state_262k = b4b_262k.recurrent_state_bytes + b4b_262k.conv_state_bytes
    lines.append("## Key insight\n")
    lines.append(
        f"At **262K context** on the 4B checkpoint, the GDN hybrid saves "
        f"**{savings_262k:.1f} GiB** versus a hypothetical all-attention model. "
        f"The recurrent state is **{_mib(total_state_262k):.0f} MiB** "
        f"regardless of context length, while the KV cache alone reaches "
        f"**{_gib(b4b_262k.kv_cache_bytes):.1f} GiB** — "
        f"{_gib(b4b_262k.kv_cache_bytes) / _gib(b4b_262k.weights_bytes) * 100:.0f}% "
        f"of the weight footprint.\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate memory scaling figures from the analytical model."
    )
    parser.add_argument(
        "--output-dir",
        default="results/figures",
        help="Output directory (default: results/figures)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Skip PNG generation, only write markdown table",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always generate the comparison table
    table_path = output_dir / "memory_comparison.md"
    table_path.write_text(generate_comparison_table(), encoding="utf-8")
    print(f"  ✓ {table_path}")

    if args.text_only:
        print("\nText-only mode — skipping PNG generation.")
        return 0

    # Try to import matplotlib
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\nWARNING: matplotlib not available — generating table only.\n"
            "Install with: pip install matplotlib",
            file=sys.stderr,
        )
        return 0

    # Generate figures
    for checkpoint in ["4B", "0.8B"]:
        model_tag = checkpoint.lower().replace(".", "")
        bar_path = output_dir / f"memory_scaling_{model_tag}.png"
        if plot_stacked_bar(checkpoint, str(bar_path), plt):
            print(f"  ✓ {bar_path}")

    # Decomposition area chart (4B only — the headline figure)
    decomp_path = output_dir / "memory_decomposition_qwen3.5-4b.png"
    if plot_decomposition_area("4B", str(decomp_path), plt):
        print(f"  ✓ {decomp_path}")

    print(f"\nDone. {3 + 1} files written to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
