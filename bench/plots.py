#!/usr/bin/env python3
"""Plot and table generation from committed results and model config.

Bead ``ob-9y8`` (``t-results-table``). Generates reproducible figures from committed
data so every chart in the write-up traces back to a CSV or a config formula, never to
a hand-assembled spreadsheet.

Two data sources:

1. **Schema-conformant CSVs** — ``results/raw/*.csv`` in the frozen tidy/long format
   (``bench/schema.py``). These come from ``bench/harness.py`` runs.
2. **Static microbenchmark CSVs** — the device-fleet kernel benchmark
   (``results/raw/rk3588-*.csv``) with columns ``model,kernel,dispatch_path,...``.
   These predate the harness and use a simpler schema.

Plus **analytical figures** from ``bench/metrics.py`` (memory decomposition, decode
traffic model) that require no measurement data — they are computed from config
dimensions and are the project's central claim made quantitative.

CLI::

    python3 -m bench.plots --all                    # generate everything
    python3 -m bench.plots --memory-scaling         # memory decomposition chart
    python3 -m bench.plots --kernel-bandwidth       # device microbenchmark chart
    python3 -m bench.plots --decode-traffic         # decode bandwidth model
    python3 -m bench.plots --table                  # markdown comparison table

Output goes to ``results/figures/`` (PNG, 150 DPI).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display on the benchmark devices
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _REPO_ROOT / "results"
_FIGURES_DIR = _RESULTS_DIR / "figures"
_RAW_DIR = _RESULTS_DIR / "raw"

# ---------------------------------------------------------------------------
# Visual language — consistent across all figures
# ---------------------------------------------------------------------------

# Colour palette (Okabe-Ito-inspired, colourblind-safe)
C_WEIGHTS = "#0072B2"     # blue
C_KV_CACHE = "#D55E00"    # vermilion
C_RECURRENT = "#009E73"   # green
C_HYPOTHETICAL = "#999999"  # grey (for "what if all-attention")
C_BIG = "#0072B2"          # blue (big cluster)
C_LITTLE = "#D55E00"       # vermilion (little cluster)

_FONT_SIZE = 11
_TITLE_SIZE = 14
_LABEL_SIZE = 12

plt.rcParams.update({
    "font.size": _FONT_SIZE,
    "axes.titlesize": _TITLE_SIZE,
    "axes.labelsize": _LABEL_SIZE,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONTEXT_LABELS = {
    4096: "4K",
    32768: "32K",
    131072: "128K",
    262144: "262K",
}


def _gib(b: float) -> float:
    """Bytes → GiB."""
    return b / (1024**3)


def _mib(b: float) -> float:
    """Bytes → MiB."""
    return b / (1024**2)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> Path:
    _ensure_dir(_FIGURES_DIR)
    out = _FIGURES_DIR / name
    fig.savefig(out)
    plt.close(fig)
    try:
        display = out.relative_to(_REPO_ROOT)
    except ValueError:
        display = out
    print(f"  wrote {display}")
    return out


# ---------------------------------------------------------------------------
# 1. Memory scaling chart (from bench/metrics.py — analytical, no measurement)
# ---------------------------------------------------------------------------

def plot_memory_scaling() -> Path:
    """Stacked bar chart: weights + KV cache + recurrent state across context lengths.

    Shows the central claim: KV cache grows linearly while recurrent state stays flat.
    Includes a hypothetical "all-attention" line for contrast.
    """
    # Import here so the module loads even if metrics.py is not importable
    # (e.g. in an env without the bench package installed).
    sys.path.insert(0, str(_REPO_ROOT / "bench"))
    from metrics import (
        ModelConfig,
        kv_cache_bytes,
        memory_breakdown,
    )

    # Qwen3.5-4B config (ground truth from GDN_LAYER_AUDIT.md)
    cfg = ModelConfig(
        hidden_size=2560,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
        full_attn_head_dim=256,
        linear_num_value_heads=32,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        intermediate_size=6912,
        vocab_size=152064,
        tie_word_embeddings=False,
        weight_dtype_bytes=2,
        cache_dtype_bytes=2,
        state_dtype_bytes=4,
    )

    contexts = [4096, 32768, 131072, 262144]
    labels = [_CONTEXT_LABELS[c] for c in contexts]

    weights_gib = []
    kv_gib = []
    recurrent_mib = []
    hypothetical_kv_gib = []

    n_all_attn = 32  # if all 32 layers were full attention (vs actual 8)

    for ctx in contexts:
        bd = memory_breakdown(cfg, ctx)
        weights_gib.append(_gib(bd.weights))
        kv_gib.append(_gib(bd.kv_cache))
        recurrent_mib.append(_mib(bd.recurrent_state))
        # Hypothetical: KV cache if ALL layers were full attention
        hyp_kv = kv_cache_bytes(cfg, ctx) * (n_all_attn / cfg.num_full_attention_layers)
        hypothetical_kv_gib.append(_gib(hyp_kv))

    recurrent_gib = [v / 1024 for v in recurrent_mib]  # convert MiB to GiB for stacking

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(contexts))
    width = 0.55

    # Stacked bars
    ax.bar(x, weights_gib, width, label="Weights (fp16, flat)", color=C_WEIGHTS)
    ax.bar(x, kv_gib, width, bottom=weights_gib,
                label="KV cache (8 attn layers)", color=C_KV_CACHE)
    bottom2 = [w + k for w, k in zip(weights_gib, kv_gib, strict=True)]
    ax.bar(x, recurrent_gib, width, bottom=bottom2,
           label="GDN recurrent state (48 MiB, flat)", color=C_RECURRENT)

    # Hypothetical all-attention line
    hyp_total = [w + h for w, h in zip(weights_gib, hypothetical_kv_gib, strict=True)]
    ax.plot(x, hyp_total, "--o", color=C_HYPOTHETICAL, markersize=5,
            label="If all 32 layers were attention")

    # Annotate the KV cache values
    for i, (kv, _hyp) in enumerate(zip(kv_gib, hypothetical_kv_gib, strict=True)):
        ax.annotate(f"{kv:.1f}", (i, weights_gib[i] + kv + 0.3),
                    ha="center", fontsize=9, color=C_KV_CACHE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Memory (GiB)")
    ax.set_title("Peak memory decomposition — Qwen3.5-4B\n"
                 "GDN state is flat; KV cache grows linearly")
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    fig.tight_layout()
    return _save(fig, "memory_scaling_4b.png")


# ---------------------------------------------------------------------------
# 2. Kernel bandwidth chart (from static microbenchmark CSVs)
# ---------------------------------------------------------------------------

def _read_microbenchmark_csv(path: Path) -> list[dict]:
    """Read a static microbenchmark CSV into a list of dicts."""
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["gib_per_s_p50"] = float(row["gib_per_s_p50"])
            row["gflop_per_s_p50"] = float(row["gflop_per_s_p50"])
            row["spread_pct"] = float(row["spread_pct"])
            row["p50_us"] = float(row["p50_us"])
            rows.append(row)
    return rows


def plot_kernel_bandwidth() -> list[Path]:
    """Grouped bar chart of achieved GiB/s per kernel, per cluster.

    Reads all ``results/raw/rk3588-*.csv`` files.
    """
    # Only device-microbenchmark CSVs (rk3588-{host}_{big|little}[_singlethread].csv).
    # Exclude model-level and per-layer profiling CSVs which have different schemas.
    csvs = sorted(c for c in _RAW_DIR.glob("rk3588-*.csv")
                  if "_big" in c.stem or "_little" in c.stem)
    if not csvs:
        print("  (no rk3588 microbenchmark CSVs found, skipping)")
        return []

    paths = []

    for model_name in ["Qwen3.5-4B", "Qwen3.5-0.8B"]:
        # Collect data for this model across all CSV files
        all_rows: dict[str, list[tuple[str, float]]] = {}
        for csv_path in csvs:
            # Derive cluster label from filename: *_big.csv or *_little.csv
            stem = csv_path.stem
            if "_big" in stem:
                cluster = "A76 (big)"
            elif "_little" in stem:
                cluster = "A55 (little)"
            else:
                cluster = stem

            for row in _read_microbenchmark_csv(csv_path):
                if row["model"] != model_name:
                    continue
                key = (cluster, row["kernel"])
                if key not in all_rows:
                    all_rows[key] = []
                all_rows[key].append(row["gib_per_s_p50"])

        if not all_rows:
            continue

        # Unique kernels and clusters
        kernels = sorted({k[1] for k in all_rows})
        clusters = sorted({k[0] for k in all_rows})

        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(kernels))
        n_clusters = len(clusters)
        bar_width = 0.35

        for i, cluster in enumerate(clusters):
            vals = []
            for kern in kernels:
                key = (cluster, kern)
                vals.append(all_rows.get(key, [0.0])[0])
            offset = (i - n_clusters / 2 + 0.5) * bar_width
            color = C_BIG if "big" in cluster.lower() else C_LITTLE
            bars = ax.bar(x + offset, vals, bar_width, label=cluster, color=color)
            for bar, val in zip(bars, vals, strict=True):
                if val > 0:
                    ax.annotate(f"{val:.1f}", (bar.get_x() + bar.get_width() / 2, val),
                                ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        kernel_labels = [k.replace("gdn_", "").replace("_", "\n") for k in kernels]
        ax.set_xticklabels(kernel_labels)
        ax.set_ylabel("Achieved bandwidth (GiB/s)")
        ax.set_title(f"Kernel microbenchmark — {model_name}\n"
                     f"rk3588-t4, governor=performance, NEON dispatch")
        ax.legend(fontsize=9)

        fig.tight_layout()
        name = f"kernel_bandwidth_{model_name.replace('.', '').replace('-', '_').lower()}.png"
        paths.append(_save(fig, name))

    return paths


# ---------------------------------------------------------------------------
# 3. Decode traffic breakdown (from the bandwidth model in METRICS.md §9)
# ---------------------------------------------------------------------------

def plot_decode_traffic() -> Path:
    """Stacked bar chart: weight traffic vs GDN state traffic per token at each
    quantization level, showing that weights dominate decode bandwidth."""
    # Total params for 4B (from bench/metrics.py computation)
    total_params = 3_782_483_968
    state_bytes_per_token = 24 * 32 * 128 * 128 * 4 * 2  # read + write, fp32

    quant_configs = [
        ("fp16", 2),
        ("INT8", 1),
        ("INT4", 0.5),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(quant_configs))
    width = 0.5

    weight_gib = []
    state_gib = []
    tok_per_s = []

    for _label, bytes_per_param in quant_configs:
        wt = total_params * bytes_per_param
        weight_gib.append(_gib(wt))
        state_gib.append(_gib(state_bytes_per_token))
        total_traffic = wt + state_bytes_per_token
        tok_per_s.append(100e9 / total_traffic)

    ax.bar(x, weight_gib, width, label="Weight traffic", color=C_WEIGHTS)
    ax.bar(x, state_gib, width, bottom=weight_gib,
                    label="GDN state traffic (fp32, constant)", color=C_RECURRENT)

    # Annotate tok/s ceiling
    for i, (wg, sg, tps) in enumerate(zip(weight_gib, state_gib, tok_per_s, strict=True)):
        total = wg + sg
        ax.annotate(f"≈{tps:.0f} tok/s\n@100 GB/s", (i, total + 0.15),
                    ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in quant_configs])
    ax.set_ylabel("Traffic per decode token (GiB)")
    ax.set_title("Decode bandwidth breakdown — Qwen3.5-4B\n"
                 "Weight quantization is the dominant lever")
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    return _save(fig, "decode_traffic_breakdown.png")


# ---------------------------------------------------------------------------
# 4. Markdown comparison table
# ---------------------------------------------------------------------------

def generate_table() -> Path:
    """Generate a markdown comparison table from committed results.

    Summarizes the static microbenchmark results and the analytical memory model.
    """
    _ensure_dir(_FIGURES_DIR)
    out = _FIGURES_DIR / "comparison_table.md"

    lines = [
        "# Results comparison table",
        "",
        "Auto-generated by `bench/plots.py --table`. All figures trace to committed "
        "CSVs or config formulas.",
        "",
        "## Static kernel microbenchmark (rk3588-t4)",
        "",
    ]

    # Kernel benchmark table
    csvs = sorted(c for c in _RAW_DIR.glob("rk3588-*.csv")
                  if "_big" in c.stem or "_little" in c.stem)
    if csvs:
        lines.append("| Device | Model | Kernel | Cluster | GiB/s (p50) | GFLOP/s | Spread % |")
        lines.append("|---|---|---|---|---:|---:|---:|")
        for csv_path in csvs:
            stem = csv_path.stem
            cluster = "A76 (big)" if "_big" in stem else "A55 (little)" if "_little" in stem else stem
            device = stem.split("_")[0]
            for row in _read_microbenchmark_csv(csv_path):
                lines.append(
                    f"| {device} | {row['model']} | `{row['kernel']}` | {cluster} "
                    f"| {row['gib_per_s_p50']:.2f} | {row['gflop_per_s_p50']:.2f} "
                    f"| {row['spread_pct']:.1f} |"
                )

    # Memory model table
    lines.extend([
        "",
        "## Memory decomposition — Qwen3.5-4B (analytical, from config)",
        "",
        "| Context | Weights (GiB) | KV cache (GiB) | Recurrent state (MiB) | "
        "Total (GiB) | If all-attn (GiB) | Savings |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])

    sys.path.insert(0, str(_REPO_ROOT / "bench"))
    from metrics import ModelConfig, kv_cache_bytes, memory_breakdown

    cfg = ModelConfig(
        hidden_size=2560, num_hidden_layers=32,
        num_attention_heads=16, num_key_value_heads=4, full_attn_head_dim=256,
        linear_num_value_heads=32, linear_num_key_heads=16,
        linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4, intermediate_size=6912,
        vocab_size=152064, tie_word_embeddings=False,
        weight_dtype_bytes=2, cache_dtype_bytes=2, state_dtype_bytes=4,
    )

    n_all_attn = 32
    for ctx in [4096, 32768, 131072, 262144]:
        bd = memory_breakdown(cfg, ctx)
        total = bd.total
        hyp_kv = kv_cache_bytes(cfg, ctx) * (n_all_attn / cfg.num_full_attention_layers)
        hyp_total = bd.weights + hyp_kv + bd.recurrent_state
        savings = hyp_total - total
        label = _CONTEXT_LABELS[ctx]
        lines.append(
            f"| {label} | {_gib(bd.weights):.2f} | {_gib(bd.kv_cache):.2f} "
            f"| {_mib(bd.recurrent_state):.0f} | {_gib(total):.2f} "
            f"| {_gib(hyp_total):.2f} | {_gib(savings):.2f} GiB |"
        )

    # Decode traffic table
    lines.extend([
        "",
        "## Decode bandwidth model — Qwen3.5-4B at 100 GB/s",
        "",
        "| Quant | Weight traffic/token | State traffic/token | Total | Ceiling tok/s |",
        "|---|---:|---:|---:|---:|",
    ])

    total_params = 3_782_483_968
    state_bytes = 24 * 32 * 128 * 128 * 4 * 2
    for label, bpp in [("fp16", 2), ("INT8", 1), ("INT4 (W4A16)", 0.5)]:
        wt = total_params * bpp
        total_traffic = wt + state_bytes
        tps = 100e9 / total_traffic
        lines.append(
            f"| {label} | {_gib(wt):.2f} GiB | {_mib(state_bytes):.0f} MiB "
            f"| {_gib(total_traffic):.2f} GiB | ≈{tps:.0f} |"
        )

    out.write_text("\n".join(lines) + "\n")
    try:
        display = out.relative_to(_REPO_ROOT)
    except ValueError:
        display = out
    print(f"  wrote {display}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate reproducible plots and tables from committed results.",
    )
    parser.add_argument("--all", action="store_true", help="Generate everything")
    parser.add_argument("--memory-scaling", action="store_true",
                        help="Memory decomposition chart (analytical)")
    parser.add_argument("--kernel-bandwidth", action="store_true",
                        help="Device kernel microbenchmark charts")
    parser.add_argument("--decode-traffic", action="store_true",
                        help="Decode bandwidth breakdown chart")
    parser.add_argument("--table", action="store_true",
                        help="Markdown comparison table")
    args = parser.parse_args(argv)

    if not any(vars(args).values()):
        parser.print_help()
        return 1

    if args.all or args.memory_scaling:
        print("[memory-scaling]")
        plot_memory_scaling()

    if args.all or args.kernel_bandwidth:
        print("[kernel-bandwidth]")
        plot_kernel_bandwidth()

    if args.all or args.decode_traffic:
        print("[decode-traffic]")
        plot_decode_traffic()

    if args.all or args.table:
        print("[table]")
        generate_table()

    return 0


if __name__ == "__main__":
    sys.exit(main())
