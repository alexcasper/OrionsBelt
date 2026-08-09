#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Generate a bar chart showing the cumulative decode optimization stack on RK3588 A76.

Data sourced from results/figures/comparison_table.md §7 and FINDINGS.md §33-34.
Every number traces to a manifest-backed CSV.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Data from comparison_table.md §7 (rk3588-t4, A76 big cluster) ──────────────

STAGES = [
    "Naive\nFP32 C",
    "Row-sweep\nGEMV\n(OpenMP+NEON)",
    "+ INT8\nweight-only",
    "+ SDOT\nINT8 GEMV",
    "+ INT4+SDOT\nhybrid",
]

# 4B model — t4 A76
TPUT_4B = [0.07, 1.04, 1.83, 3.48, 4.43]
COMMITS_4B = ["a756662", "3d914b6", "861bdf2", "be4d3ca", "3bff376"]

# 0.8B model — t4 A76
TPUT_08B = [0.68, 8.32, 10.03, 30.17, 37.21]

COLORS = ["#9e9e9e", "#64b5f6", "#42a5f5", "#1e88e5", "#0d47a1"]


def generate_chart(output_path: str | Path) -> str:
    """Render the optimization-stack bar chart and save to *output_path*.

    Returns the output path as a string.
    """
    output = str(output_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- 4B panel ---
    bars1 = ax1.bar(range(len(STAGES)), TPUT_4B, color=COLORS, edgecolor="white", linewidth=0.8)
    ax1.set_xticks(range(len(STAGES)))
    ax1.set_xticklabels(STAGES, fontsize=8.5)
    ax1.set_ylabel("Decode throughput (tok/s)", fontsize=11)
    ax1.set_title("Qwen3.5-4B (A76 big cluster)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 5.2)
    for bar, val in zip(bars1, TPUT_4B, strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax1.annotate(
        "",
        xy=(4, 4.43),
        xytext=(0, 0.07),
        arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.5, ls="--"),
    )
    ax1.text(
        2.5, 4.8, "~63× cumulative", ha="center", fontsize=11, color="#c62828", fontweight="bold"
    )
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # --- 0.8B panel ---
    bars2 = ax2.bar(range(len(STAGES)), TPUT_08B, color=COLORS, edgecolor="white", linewidth=0.8)
    ax2.set_xticks(range(len(STAGES)))
    ax2.set_xticklabels(STAGES, fontsize=8.5)
    ax2.set_ylabel("Decode throughput (tok/s)", fontsize=11)
    ax2.set_title("Qwen3.5-0.8B (A76 big cluster)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 42)
    for bar, val in zip(bars2, TPUT_08B, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax2.annotate(
        "",
        xy=(4, 37.21),
        xytext=(0, 0.68),
        arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.5, ls="--"),
    )
    ax2.text(
        2.5, 39, "~55× cumulative", ha="center", fontsize=11, color="#c62828", fontweight="bold"
    )

    fig.suptitle(
        "Decode Optimization Stack — RK3588 Cortex-A76 (rk3588-t4)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")
    return output


def main() -> None:
    output = os.path.join("results", "figures", "optimization_stack.png")
    generate_chart(output)


if __name__ == "__main__":
    main()
