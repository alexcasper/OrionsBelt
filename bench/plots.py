"""Plot and table generation from results (ob-9y8).

Generates reproducible figures from committed data:

1. **Memory decomposition** — stacked bar chart of weights / KV cache / recurrent
   state across context lengths, showing the GDN architectural advantage.
2. **Throughput curves** — prefill/decode tokens/sec vs context length from
   harness CSVs (RESULTS_SCHEMA.md tidy/long format).
3. **Device comparison** — achieved GiB/s across the device fleet from device
   bench CSVs.

Every figure is generated from data, never hand-assembled. Uses matplotlib with
a consistent visual language.

Usage::

    python3 bench/plots.py --memory          # memory decomposition chart
    python3 bench/plots.py --throughput CSV  # throughput curves from a harness CSV
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt  # noqa: E402

if TYPE_CHECKING:
    from bench.harness import ModelConfig

# Ensure repo root is importable
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

FIGURES_DIR = Path(_ROOT) / "results" / "figures"

# Consistent visual language
_COLORS = {
    "weights": "#2196F3",  # blue
    "kv_cache": "#FF9800",  # orange
    "recurrent_state": "#4CAF50",  # green
    "prefill": "#2196F3",
    "decode": "#FF5722",
}
_FIGSIZE = (10, 6)
_DPI = 150


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


# ---------------------------------------------------------------------------
# 1. Memory decomposition (from bench/memory.py)
# ---------------------------------------------------------------------------


def plot_memory_decomposition(
    config: ModelConfig,
    context_lengths: list[int],
    output_path: Path | None = None,
) -> Path:
    """Stacked bar chart: weights + KV cache + recurrent state vs context length.

    This is the project's central claim made visual: weights are flat, KV cache
    grows linearly, recurrent state stays O(1).
    """
    from bench.memory import decomposition

    rows = decomposition(config, context_lengths)

    ctx_labels = [f"{c // 1024}K" for c in context_lengths]
    weights = [r["weights"] / (1024**3) for r in rows]  # GiB
    kv_cache = [r["kv_cache"] / (1024**3) for r in rows]
    rec_state = [r["recurrent_state"] / (1024**3) for r in rows]
    counterfactual = [r["all_attention_kv_equivalent"] / (1024**3) for r in rows]

    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)

    x = range(len(context_lengths))
    w = 0.6

    # Stacked bars: weights + kv_cache + recurrent_state
    ax.bar(x, weights, w, label="Weights", color=_COLORS["weights"])
    ax.bar(
        x, kv_cache, w, bottom=weights, label="KV cache (8 FA layers)", color=_COLORS["kv_cache"]
    )
    bottom_rs = [wt + kv for wt, kv in zip(weights, kv_cache, strict=True)]
    ax.bar(
        x,
        rec_state,
        w,
        bottom=bottom_rs,
        label="GDN state (O(1))",
        color=_COLORS["recurrent_state"],
    )

    # Counterfactual line: all-attention KV cache equivalent
    ax.plot(
        x,
        [wt + cf for wt, cf in zip(weights, counterfactual, strict=True)],
        "r--o",
        markersize=5,
        label="All-attention equivalent",
        alpha=0.7,
    )

    ax.set_xlabel("Context Length")
    ax.set_ylabel("Memory (GiB)")
    ax.set_title(
        f"Memory Decomposition: {config.name}\n"
        f"({config.num_gdn_layers} GDN + {config.num_full_attention_layers} FA layers)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ctx_labels)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    if output_path is None:
        output_path = FIGURES_DIR / f"memory_decomposition_{config.name.split('/')[-1].lower()}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 2. Throughput curves (from harness CSV, RESULTS_SCHEMA.md format)
# ---------------------------------------------------------------------------


def plot_throughput_curves(
    csv_path: str,
    output_path: Path | None = None,
) -> Path:
    """Throughput vs context length from a harness CSV (tidy/long schema).

    Plots p50 prefill_tokens_per_sec and decode_tokens_per_sec across context
    lengths, computed from per-repeat rows in the CSV.
    """
    # Read CSV and group by (context_length, metric_name)
    groups: dict[tuple[int, str], list[float]] = defaultdict(list)
    run_id = ""

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row.get("run_id", run_id)
            metric = row.get("metric_name", "")
            if metric in ("prefill_tokens_per_sec", "decode_tokens_per_sec"):
                ctx = int(row["context_length"])
                groups[(ctx, metric)].append(float(row["value"]))

    if not groups:
        raise ValueError(f"no throughput rows found in {csv_path}")

    # Compute p50 per group
    from bench.metrics import percentile

    contexts = sorted({ctx for ctx, _ in groups})
    prefill_p50 = []
    decode_p50 = []
    for ctx in contexts:
        prefill_vals = groups.get((ctx, "prefill_tokens_per_sec"), [])
        decode_vals = groups.get((ctx, "decode_tokens_per_sec"), [])
        prefill_p50.append(percentile(prefill_vals, 50) if prefill_vals else 0)
        decode_p50.append(percentile(decode_vals, 50) if decode_vals else 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=_FIGSIZE, dpi=_DPI, sharex=True)

    ctx_labels = [f"{c // 1024}K" if c >= 1024 else str(c) for c in contexts]
    x = range(len(contexts))

    ax1.bar(x, prefill_p50, 0.5, color=_COLORS["prefill"], alpha=0.8)
    ax1.set_ylabel("Prefill (tok/s)")
    ax1.set_title(f"Throughput vs Context Length: {run_id}")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, decode_p50, 0.5, color=_COLORS["decode"], alpha=0.8)
    ax2.set_ylabel("Decode (tok/s)")
    ax2.set_xlabel("Context Length")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ctx_labels)
    ax2.grid(axis="y", alpha=0.3)

    if output_path is None:
        output_path = FIGURES_DIR / f"throughput_{run_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 3. Device comparison (from device bench CSVs)
# ---------------------------------------------------------------------------


def plot_device_comparison(
    csv_dir: str = "results/raw",
    output_path: Path | None = None,
) -> Path:
    """Compare achieved GiB/s across the device fleet from device bench CSVs.

    Each CSV in csv_dir matching the device-bench format (columns: model, kernel,
    dispatch_path, ..., gib_per_s_p50) is read and its p50 GiB/s plotted.
    """
    csv_dir_path = Path(csv_dir)
    if not csv_dir_path.exists():
        raise FileNotFoundError(f"no results directory: {csv_dir}")

    # Collect data: {(device, model, kernel) -> [(gib_per_s, kernel_name)]}
    devices: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for csv_file in sorted(csv_dir_path.glob("*.csv")):
        device_name = csv_file.stem  # e.g., "pi5-r5"
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gib = float(row.get("gib_per_s_p50", 0))
                kernel = row.get("kernel", "?")
                model = row.get("model", "?")
                devices[device_name].append((f"{model}\n{kernel}", gib))

    if not devices:
        raise ValueError(f"no device bench CSVs found in {csv_dir}")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=_DPI)

    bar_width = 0.8 / max(len(devices), 1)
    all_kernels: list[str] = []
    for entries in devices.values():
        for label, _ in entries:
            if label not in all_kernels:
                all_kernels.append(label)

    x = range(len(all_kernels))
    colors = plt.cm.Set2.colors

    for i, (device, entries) in enumerate(sorted(devices.items())):
        vals = []
        for label in all_kernels:
            gib = next((g for lb, g in entries if lb == label), 0)
            vals.append(gib)
        offset = (i - len(devices) / 2 + 0.5) * bar_width
        ax.bar(
            [xi + offset for xi in x],
            vals,
            bar_width * 0.9,
            label=device,
            color=colors[i % len(colors)],
            alpha=0.85,
        )

    ax.set_ylabel("Achieved GiB/s (p50)")
    ax.set_title("Device Fleet Comparison: GDN Kernel Throughput")
    ax.set_xticks(x)
    ax.set_xticklabels(all_kernels, fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    if output_path is None:
        output_path = FIGURES_DIR / "device_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate plots from benchmark data (ob-9y8)")
    parser.add_argument("--memory", action="store_true", help="Generate memory decomposition plot")
    parser.add_argument(
        "--throughput", metavar="CSV", help="Generate throughput curves from a harness CSV"
    )
    parser.add_argument(
        "--device-comparison", action="store_true", help="Generate device fleet comparison"
    )
    parser.add_argument("--model", default="4b", choices=["4b", "0.8b"])
    parser.add_argument(
        "--output", "-o", default="", help="Output path (default: results/figures/)"
    )
    args = parser.parse_args(argv)

    from bench.harness import QWEN35_08B, QWEN35_4B

    if args.memory:
        model = QWEN35_4B if args.model == "4b" else QWEN35_08B
        ctx = [4096, 32768, 131072, 262144]
        out = Path(args.output) if args.output else None
        path = plot_memory_decomposition(model, ctx, out)
        print(f"  Memory decomposition: {path}")

    if args.throughput:
        out = Path(args.output) if args.output else None
        path = plot_throughput_curves(args.throughput, out)
        print(f"  Throughput curves: {path}")

    if args.device_comparison:
        out = Path(args.output) if args.output else None
        path = plot_device_comparison(output_path=out)
        print(f"  Device comparison: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
