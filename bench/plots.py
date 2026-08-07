"""Plot and table generation from committed benchmark results.

Bead ``ob-9y8`` (t-plots).  Every figure in the write-up and README must be
regenerable from data committed under ``results/raw/`` — no hand-assembled
numbers, no spreadsheet screenshots.

Two CSV formats are supported, auto-detected by header:

1. **Microbenchmark CSV** (from ``bench_gdn.c``, the device-fleet bandwidth
   study, bead ``ob-8ms``).  Columns: ``model, kernel, dispatch_path, seq,
   channels, repeats, p50_us, p95_us, spread_pct, gib_per_s_p50,
   gflop_per_s_p50``.

2. **Schema-conformant CSV** (from ``bench/harness.py``, the full inference
   harness).  Columns defined in ``bench/schema.py::COLUMNS``.

Usage::

    # Generate all figures + tables from everything in results/raw/
    python -m bench.plots --output-dir results/figures

    # Process a specific file
    python -m bench.plots results/raw/jetson-j1.csv --output-dir results/figures

    # Text-only mode (no matplotlib dependency)
    python -m bench.plots --text-only --output-dir results/figures

The script degrades gracefully: if matplotlib is unavailable, it skips PNG
generation and still writes markdown/text tables to ``results/figures/``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

# Canonical device spec bandwidth (GiB/s).  Vendor datasheets quote GB/s
# (decimal); the bench binary measures GiB/s (÷2^30).  We convert so the
# achieved-vs-spec ratio is unit-consistent.  See ADR 0005 for GB/s originals.
DEVICE_SPEC_BANDWIDTH: dict[str, float] = {
    # device identifier (lowercase prefix match) → spec GiB/s
    "pi5": 15.8,  # 17.0 GB/s ÷ 1.0737
    "pi": 15.8,
    "rk3588": 31.7,  # 34.0 GB/s ÷ 1.0737
    "jetson": 23.8,  # 25.6 GB/s ÷ 1.0737
    "jetson-j1": 23.8,
    "o6": 93.1,  # 100 GB/s ÷ 1.0737
}

# Human-readable kernel names for plot labels.
KERNEL_LABELS: dict[str, str] = {
    "gdn_cumdecay": "Gated Cumulative Decay",
    "gdn_gated_scan": "Gated Delta-Rule Scan",
    "gdn_causal_dwconv1d": "Causal Depthwise Conv1D",
}


@dataclass
class MicrobenchRow:
    """One row from a microbenchmark CSV (bench_gdn.c output)."""

    model: str
    kernel: str
    dispatch_path: str
    seq: int
    channels: int
    repeats: int
    p50_us: float
    p95_us: float
    spread_pct: float
    gib_per_s: float
    gflop_per_s: float


@dataclass
class SchemaRow:
    """One aggregated row from a schema-conformant CSV (already percentile-summarized).

    The harness writes per-repeat rows; we aggregate to p50/p95 here.
    """

    run_id: str
    device: str
    engine_gdn: str
    engine_full_attention: str
    model_checkpoint: str
    quantization: str
    context_length: int
    phase: str
    metric_name: str
    metric_component: str
    unit: str
    p50: float
    p95: float
    repeat_count: int


# ---------------------------------------------------------------------------
# CSV detection and parsing
# ---------------------------------------------------------------------------

MICROBENCH_COLUMNS = [
    "model",
    "kernel",
    "dispatch_path",
    "seq",
    "channels",
    "repeats",
    "p50_us",
    "p95_us",
    "spread_pct",
    "gib_per_s_p50",
    "gflop_per_s_p50",
]


def detect_format(header: list[str]) -> str | None:
    """Return 'microbench', 'schema', or None based on CSV header.

    Returns None rather than raising for unrecognised headers: results/raw/ also
    holds power-monitor CSVs, and generate_all() walks the whole directory, so a
    raise here took the entire plot run down over a file it never needed.
    """
    if "kernel" in header and "dispatch_path" in header:
        return "microbench"
    if "metric_name" in header and "phase" in header:
        return "schema"
    return None  # unrecognised (e.g. power monitoring CSVs)


def parse_microbench(path: str) -> list[MicrobenchRow]:
    """Parse a microbenchmark CSV into a list of MicrobenchRow."""
    rows: list[MicrobenchRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(
                MicrobenchRow(
                    model=raw["model"],
                    kernel=raw["kernel"],
                    dispatch_path=raw["dispatch_path"],
                    seq=int(raw["seq"]),
                    channels=int(raw["channels"]),
                    repeats=int(raw["repeats"]),
                    p50_us=float(raw["p50_us"]),
                    p95_us=float(raw["p95_us"]),
                    spread_pct=float(raw["spread_pct"]),
                    gib_per_s=float(raw["gib_per_s_p50"]),
                    gflop_per_s=float(raw["gflop_per_s_p50"]),
                )
            )
    return rows


def parse_schema(path: str) -> list[SchemaRow]:
    """Parse a schema-conformant CSV and aggregate repeats to p50/p95.

    The harness writes per-repeat rows (one row per repeat_index).  We group by
    all dimensions except repeat_index/value and compute p50 and p95 using the
    nearest-rank method.
    """
    raw_rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    # Group rows by all dimensions except repeat_index
    groups: dict[tuple, list[float]] = defaultdict(list)
    meta: dict[tuple, dict] = {}

    for raw in raw_rows:
        key = (
            raw["run_id"],
            raw["device"],
            raw["engine_gdn"],
            raw["engine_full_attention"],
            raw["model_checkpoint"],
            raw["quantization"],
            int(raw["context_length"]),
            raw["phase"],
            raw["metric_name"],
            raw.get("metric_component", "") or "",
            raw["unit"],
        )
        val = float(raw["value"])
        groups[key].append(val)
        meta[key] = raw

    result: list[SchemaRow] = []
    for key, values in sorted(groups.items()):
        values_sorted = sorted(values)
        n = len(values_sorted)
        # Nearest-rank percentile
        p50_idx = max(0, min(n - 1, int(round(0.50 * (n - 1)))))
        p95_idx = max(0, min(n - 1, int(round(0.95 * (n - 1)))))
        result.append(
            SchemaRow(
                run_id=key[0],
                device=key[1],
                engine_gdn=key[2],
                engine_full_attention=key[3],
                model_checkpoint=key[4],
                quantization=key[5],
                context_length=key[6],
                phase=key[7],
                metric_name=key[8],
                metric_component=key[9],
                unit=key[10],
                p50=values_sorted[p50_idx],
                p95=values_sorted[p95_idx],
                repeat_count=n,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Table generation (always available — no matplotlib required)
# ---------------------------------------------------------------------------


def microbench_to_markdown(rows: list[MicrobenchRow]) -> str:
    """Generate a markdown comparison table from microbenchmark rows."""
    if not rows:
        return "_(no data)_\n"

    lines = [
        "| Model | Kernel | Dispatch | Channels | p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        kernel_label = KERNEL_LABELS.get(r.kernel, r.kernel)
        lines.append(
            f"| {r.model} | {kernel_label} | {r.dispatch_path} | {r.channels:,} "
            f"| {r.p50_us:,.1f} | {r.p95_us:,.1f} | {r.spread_pct:.1f}% "
            f"| {r.gib_per_s:.2f} | {r.gflop_per_s:.2f} |"
        )
    return "\n".join(lines) + "\n"


def microbench_bandwidth_table(rows: list[MicrobenchRow], device_name: str = "") -> str:
    """Generate a bandwidth comparison table with spec bandwidth if known."""
    spec_bw = _lookup_spec_bandwidth(device_name)

    if not rows:
        return "_(no data)_\n"

    lines = [
        "## Achieved vs Spec Bandwidth\n",
    ]
    if spec_bw:
        lines.append(f"**Device spec bandwidth:** {spec_bw:.1f} GiB/s\n")
    else:
        lines.append("**Device spec bandwidth:** _(unknown — add to DEVICE_SPEC_BANDWIDTH)_\n")

    lines.append("")
    lines.append("| Kernel | Achieved (GiB/s) | % of Spec | p50 (µs) | Spread |")
    lines.append("|---|---:|---:|---:|---:|")

    for r in rows:
        pct = (r.gib_per_s / spec_bw * 100.0) if spec_bw else 0.0
        pct_str = f"{pct:.1f}%" if spec_bw else "—"
        kernel_label = KERNEL_LABELS.get(r.kernel, r.kernel)
        lines.append(
            f"| {kernel_label} | {r.gib_per_s:.2f} | {pct_str} "
            f"| {r.p50_us:,.1f} | {r.spread_pct:.1f}% |"
        )

    return "\n".join(lines) + "\n"


def schema_throughput_table(rows: list[SchemaRow]) -> str:
    """Generate a throughput vs context-length table from schema rows."""
    tput = [r for r in rows if r.metric_name.endswith("tokens_per_sec")]
    if not tput:
        return "_(no throughput data)_\n"

    # Group by (model, device, metric) then sort by context_length
    groups: dict[tuple, list[SchemaRow]] = defaultdict(list)
    for r in tput:
        key = (r.model_checkpoint, r.device, r.metric_name)
        groups[key].append(r)

    lines = ["## Throughput vs Context Length\n"]
    for (model, device, metric), group in sorted(groups.items()):
        group.sort(key=lambda r: r.context_length)
        phase = "Prefill" if "prefill" in metric else "Decode"
        lines.append(f"\n**{model} — {device} — {phase}**\n")
        lines.append("| Context | p50 (tok/s) | p95 (tok/s) | Spread | Repeats |")
        lines.append("|---:|---:|---:|---:|---:|")
        for r in group:
            spread = r.p95 - r.p50
            lines.append(
                f"| {r.context_length:,} | {r.p50:.1f} | {r.p95:.1f} "
                f"| {spread:.1f} | {r.repeat_count} |"
            )

    return "\n".join(lines) + "\n"


def schema_memory_table(rows: list[SchemaRow]) -> str:
    """Generate a three-way memory decomposition table from schema rows."""
    mem = [r for r in rows if r.metric_name == "peak_memory_bytes"]
    if not mem:
        return "_(no memory data)_\n"

    # Group by (model, device, context_length) and show the three components
    groups: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in mem:
        key = (r.model_checkpoint, r.device, r.context_length)
        comp = r.metric_component or "unknown"
        groups[key][comp] = r.p50

    lines = ["## Memory Decomposition (p50)\n"]
    lines.append(
        "| Model | Device | Context | Weights (MiB) | KV Cache (MiB) | Recurrent State (MiB) | Total (MiB) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for (model, device, ctx), comps in sorted(groups.items()):
        w = comps.get("weights", 0) / 1048576.0
        k = comps.get("kv_cache", 0) / 1048576.0
        s = comps.get("recurrent_state", 0) / 1048576.0
        total = w + k + s
        lines.append(
            f"| {model} | {device} | {ctx:,} | {w:,.1f} | {k:,.1f} | {s:,.1f} | {total:,.1f} |"
        )

    return "\n".join(lines) + "\n"


def _lookup_spec_bandwidth(device_name: str) -> float | None:
    """Look up spec bandwidth for a device name (prefix match, case-insensitive)."""
    name = device_name.lower().strip()
    for prefix, bw in DEVICE_SPEC_BANDWIDTH.items():
        if name.startswith(prefix) or prefix.startswith(name):
            return bw
    return None


# ---------------------------------------------------------------------------
# Figure generation (requires matplotlib)
# ---------------------------------------------------------------------------

# Consistent visual language across all figures.
FIG_DPI = 150
FIG_W = 8
FIG_H = 5

# Colour palette — distinct, colour-blind-friendly (Okabe-Ito subset).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]


def _try_import_matplotlib():
    """Import matplotlib with Agg backend. Return None if unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def plot_bandwidth_bars(
    rows: list[MicrobenchRow],
    device_name: str,
    output_path: str,
    plt=None,
) -> str | None:
    """Bar chart: achieved GiB/s per kernel, with spec bandwidth reference line.

    Returns the output path if the figure was written, None otherwise.
    """
    if plt is None:
        return None

    spec_bw = _lookup_spec_bandwidth(device_name)

    # Group by kernel
    kernels: OrderedDict = OrderedDict()
    for r in rows:
        label = KERNEL_LABELS.get(r.kernel, r.kernel)
        if label not in kernels:
            kernels[label] = []
        kernels[label].append(r)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    x_positions = []
    x_labels = []
    bar_width = 0.35

    for i, k_rows in enumerate(kernels.values()):
        for j, r in enumerate(k_rows):
            model_short = r.model.replace("Qwen3.5-", "")
            color = PALETTE[j % len(PALETTE)]
            x = i + j * bar_width
            ax.bar(x, r.gib_per_s, bar_width, label=model_short, color=color, alpha=0.85)
            x_positions.append(x)
            x_labels.append(model_short)

    ax.set_xticks([i + bar_width * (len(kernels) - 1) / 2 for i in range(len(kernels))])
    ax.set_xticklabels(list(kernels.keys()), fontsize=9)

    if spec_bw:
        ax.axhline(spec_bw, color="#CC0000", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(
            len(kernels) - 0.5,
            spec_bw * 1.02,
            f"Spec: {spec_bw:.1f} GiB/s",
            fontsize=8,
            color="#CC0000",
            ha="right",
        )

    # Deduplicate legend (convert to lists for matplotlib compatibility)
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles, strict=False))
    ax.legend(list(by_label.values()), list(by_label.keys()), fontsize=8, loc="upper right")

    ax.set_ylabel("Achieved Bandwidth (GiB/s)")
    ax.set_title(
        f"GDN Kernel Bandwidth — {device_name}\n({rows[0].dispatch_path} dispatch)" if rows else ""
    )
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI)
    plt.close(fig)
    return output_path


def plot_throughput_curve(
    rows: list[SchemaRow],
    output_path: str,
    plt=None,
) -> str | None:
    """Line plot: throughput (prefill & decode) vs context length.

    Returns the output path if the figure was written, None otherwise.
    """
    if plt is None or not rows:
        return None

    tput = [r for r in rows if r.metric_name.endswith("tokens_per_sec")]
    if not tput:
        return None

    # Group by (model, device, metric)
    groups: dict[tuple, list[SchemaRow]] = defaultdict(list)
    for r in tput:
        key = (r.model_checkpoint, r.device, r.metric_name)
        groups[key].append(r)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for color_idx, ((model, device, metric), group) in enumerate(sorted(groups.items())):
        group.sort(key=lambda r: r.context_length)
        ctxs = [r.context_length for r in group]
        p50s = [r.p50 for r in group]
        p95s = [r.p95 for r in group]

        model_short = model.replace("Qwen/Qwen3.5-", "")
        phase = "prefill" if "prefill" in metric else "decode"
        label = f"{model_short} {device} ({phase})"
        color = PALETTE[color_idx % len(PALETTE)]

        ax.plot(ctxs, p50s, marker="o", color=color, label=label, linewidth=2)
        ax.fill_between(ctxs, p50s, p95s, alpha=0.15, color=color)

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Throughput vs Context Length")
    ax.set_xscale("log")
    ax.set_xticks([4096, 32768, 131072, 262144])
    ax.set_xticklabels(["4K", "32K", "128K", "256K"])
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI)
    plt.close(fig)
    return output_path


def plot_memory_decomposition(
    rows: list[SchemaRow],
    output_path: str,
    plt=None,
) -> str | None:
    """Stacked area plot: weights / KV cache / recurrent state vs context length.

    Returns the output path if the figure was written, None otherwise.
    """
    if plt is None or not rows:
        return None

    mem = [r for r in rows if r.metric_name == "peak_memory_bytes"]
    if not mem:
        return None

    # Group by (model, device) and plot all context lengths
    groups: dict[tuple, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in mem:
        key = (r.model_checkpoint, r.device)
        comp = r.metric_component or "unknown"
        groups[key][r.context_length][comp] = r.p50 / 1048576.0  # MiB

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for color_idx, ((model, device), ctx_map) in enumerate(sorted(groups.items())):
        ctxs = sorted(ctx_map.keys())
        weights = [ctx_map[c].get("weights", 0) for c in ctxs]
        kv = [ctx_map[c].get("kv_cache", 0) for c in ctxs]
        state = [ctx_map[c].get("recurrent_state", 0) for c in ctxs]

        model_short = model.replace("Qwen/Qwen3.5-", "")
        base_label = f"{model_short} {device}"

        ax.fill_between(
            ctxs,
            0,
            weights,
            alpha=0.6,
            color=PALETTE[0],
            label=f"{base_label} — Weights" if color_idx == 0 else "",
        )
        ax.fill_between(
            ctxs,
            weights,
            [w + k for w, k in zip(weights, kv, strict=False)],
            alpha=0.6,
            color=PALETTE[1],
            label=f"{base_label} — KV Cache" if color_idx == 0 else "",
        )
        ax.fill_between(
            ctxs,
            [w + k for w, k in zip(weights, kv, strict=False)],
            [w + k + s for w, k, s in zip(weights, kv, state, strict=False)],
            alpha=0.6,
            color=PALETTE[2],
            label=f"{base_label} — Recurrent State" if color_idx == 0 else "",
        )

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Memory (MiB)")
    ax.set_title("Memory Decomposition: Weights + KV Cache + Recurrent State")
    ax.set_xscale("log")
    ax.set_xticks([4096, 32768, 131072, 262144])
    ax.set_xticklabels(["4K", "32K", "128K", "256K"])
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


@dataclass
class PlotResult:
    """Summary of what was generated."""

    figures: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def generate_all(
    csv_paths: list[str],
    output_dir: str,
    text_only: bool = False,
) -> PlotResult:
    """Process all CSV files and generate figures + tables.

    Auto-detects each file's format and routes to the appropriate generators.
    """
    os.makedirs(output_dir, exist_ok=True)
    result = PlotResult()

    plt = None if text_only else _try_import_matplotlib()
    if plt is None and not text_only:
        result.warnings.append(
            "matplotlib not available — generating text tables only. "
            "Install with: pip install matplotlib"
        )

    all_microbench: list[MicrobenchRow] = []
    all_schema: list[SchemaRow] = []
    microbench_by_device: dict[str, list[MicrobenchRow]] = defaultdict(list)

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            result.warnings.append(f"File not found: {csv_path}")
            continue

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

        fmt = detect_format(header)

        if fmt == "microbench":
            rows = parse_microbench(csv_path)
            all_microbench.extend(rows)
            # Derive device name from filename
            device = Path(csv_path).stem
            microbench_by_device[device].extend(rows)
        elif fmt == "schema":
            rows = parse_schema(csv_path)
            all_schema.extend(rows)
        else:
            # Skip unrecognised formats (e.g. power monitoring CSVs)
            continue

    # --- Microbenchmark figures and tables ---
    for device, rows in sorted(microbench_by_device.items()):
        # Bandwidth bar chart
        if plt is not None:
            fig_path = os.path.join(output_dir, f"{device}_bandwidth.png")
            if plot_bandwidth_bars(rows, device, fig_path, plt):
                result.figures.append(fig_path)

        # Markdown tables
        table_path = os.path.join(output_dir, f"{device}_table.md")
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(f"# {device} — Microbenchmark Results\n\n")
            f.write("_Source: committed CSVs in results/raw/_\n\n")
            f.write(microbench_to_markdown(rows))
            f.write("\n")
            f.write(microbench_bandwidth_table(rows, device))
        result.tables.append(table_path)

    # Cross-device bandwidth comparison (if we have multiple devices)
    if len(microbench_by_device) > 1 and plt is not None:
        cross_path = os.path.join(output_dir, "cross_device_bandwidth.png")
        _plot_cross_device(microbench_by_device, cross_path, plt)
        result.figures.append(cross_path)

    # --- Schema-conformant figures and tables ---
    if all_schema:
        if plt is not None:
            tput_fig = os.path.join(output_dir, "throughput_vs_context.png")
            if plot_throughput_curve(all_schema, tput_fig, plt):
                result.figures.append(tput_fig)

            mem_fig = os.path.join(output_dir, "memory_decomposition.png")
            if plot_memory_decomposition(all_schema, mem_fig, plt):
                result.figures.append(mem_fig)

        # Schema tables
        schema_table_path = os.path.join(output_dir, "harness_tables.md")
        with open(schema_table_path, "w", encoding="utf-8") as f:
            f.write("# Harness Results — Schema-Conformant\n\n")
            f.write("_Source: committed CSVs from bench/harness.py_\n\n")
            f.write(schema_throughput_table(all_schema))
            f.write("\n")
            f.write(schema_memory_table(all_schema))
        result.tables.append(schema_table_path)

    # --- Master summary ---
    summary_path = os.path.join(output_dir, "README.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Generated Figures and Tables\n\n")
        f.write("_Auto-generated by `bench/plots.py`. Do not hand-edit._\n\n")
        f.write(f"Generated: {len(result.figures)} figures, {len(result.tables)} tables\n\n")

        if result.warnings:
            f.write("## Warnings\n\n")
            for w in result.warnings:
                f.write(f"- {w}\n")
            f.write("\n")

        if result.figures:
            f.write("## Figures\n\n")
            for fig in sorted(result.figures):
                name = os.path.basename(fig)
                f.write(f"- [{name}]({name})\n")
            f.write("\n")

        if result.tables:
            f.write("## Tables\n\n")
            for tbl in sorted(result.tables):
                name = os.path.basename(tbl)
                f.write(f"- [{name}]({name})\n")
            f.write("\n")

    return result


def _plot_cross_device(
    by_device: dict[str, list[MicrobenchRow]],
    output_path: str,
    plt,
) -> None:
    """Bar chart comparing achieved bandwidth across devices for the scan kernel."""
    # Focus on gated_scan — the core GDN recurrence
    scan_by_device: dict[str, dict[str, float]] = defaultdict(dict)
    for device, rows in by_device.items():
        for r in rows:
            if r.kernel == "gdn_gated_scan":
                model_short = r.model.replace("Qwen3.5-", "")
                scan_by_device[device][model_short] = r.gib_per_s

    devices = sorted(scan_by_device.keys())
    if not devices:
        return

    # Collect all model variants
    all_models = sorted({m for d in devices for m in scan_by_device[d]})
    n_models = len(all_models)
    n_devices = len(devices)

    fig, ax = plt.subplots(figsize=(max(FIG_W, n_devices * 1.5), FIG_H))

    bar_width = 0.8 / max(n_models, 1)
    for j, model in enumerate(all_models):
        x_positions = []
        heights = []
        for i, device in enumerate(devices):
            x_positions.append(i + j * bar_width)
            heights.append(scan_by_device[device].get(model, 0))
        ax.bar(
            x_positions,
            heights,
            bar_width,
            label=model,
            color=PALETTE[j % len(PALETTE)],
            alpha=0.85,
        )

    ax.set_xticks([i + bar_width * (n_models - 1) / 2 for i in range(n_devices)])
    ax.set_xticklabels(devices, fontsize=9)

    # Spec bandwidth reference lines
    for i, device in enumerate(devices):
        spec = _lookup_spec_bandwidth(device)
        if spec:
            ax.plot(
                [i - 0.4, i + n_models * bar_width - 0.4 + 0.4],
                [spec, spec],
                color="#CC0000",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
            )

    ax.set_ylabel("Achieved Bandwidth (GiB/s)")
    ax.set_title("Cross-Device Comparison: Gated Delta-Rule Scan")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate plots and tables from committed benchmark CSVs."
    )
    parser.add_argument(
        "csv_paths",
        nargs="*",
        default=["results/raw/"],
        help="CSV files or directories to process (default: results/raw/)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/figures",
        help="Output directory for figures and tables (default: results/figures)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Skip figure generation (matplotlib not required)",
    )
    args = parser.parse_args(argv)

    # Expand directories to individual CSV files
    csv_files: list[str] = []
    for path in args.csv_paths:
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                if fname.endswith(".csv"):
                    csv_files.append(os.path.join(path, fname))
        elif path.endswith(".csv"):
            csv_files.append(path)

    if not csv_files:
        print("No CSV files found.", file=sys.stderr)
        return 1

    print(f"Processing {len(csv_files)} CSV file(s)...")
    result = generate_all(csv_files, args.output_dir, text_only=args.text_only)

    for fig in result.figures:
        print(f"  ✓ figure: {fig}")
    for tbl in result.tables:
        print(f"  ✓ table:  {tbl}")
    for w in result.warnings:
        print(f"  ⚠ {w}", file=sys.stderr)

    print(f"\nDone: {len(result.figures)} figures, {len(result.tables)} tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
