"""Plot and table generation from committed result CSVs.

Reads ``results/raw/`` and writes ``results/figures/``. Every figure in the README
and write-up must be reproducible from data in this directory — no hand-assembled
numbers.

Two CSV formats are supported:

1. **RESULTS_SCHEMA format** (tidy/long, from ``bench/harness.py``):
   Columns: ``run_id, timestamp, git_sha, manifest_ref, device, engine_gdn,
   engine_full_attention, model_checkpoint, quantization, context_length, phase,
   metric_name, metric_component, value, unit, repeat_index, repeat_count,
   layer_class, notes``.

2. **Device-bench format** (from the static GDN kernel microbenchmark):
   Columns: ``model, kernel, dispatch_path, seq, channels, repeats, p50_us,
   p95_us, spread_pct, gib_per_s_p50, gflop_per_s_p50``.

matplotlib is imported lazily inside the plot functions so this module loads even
without the ``bench`` extras installed. The markdown table generator is stdlib-only
and always available.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _REPO_ROOT / "results"
_RAW_DIR = _RESULTS_DIR / "raw"
_FIGURES_DIR = _RESULTS_DIR / "figures"

# Column sets for format detection
_SCHEMA_COLUMNS = frozenset({
    "run_id", "timestamp", "git_sha", "manifest_ref", "device",
    "engine_gdn", "engine_full_attention", "model_checkpoint",
    "quantization", "context_length", "phase", "metric_name",
    "metric_component", "value", "unit", "repeat_index", "repeat_count",
})

_DEVICE_BENCH_COLUMNS = frozenset({
    "model", "kernel", "dispatch_path", "seq", "channels",
    "repeats", "p50_us", "p95_us", "spread_pct",
    "gib_per_s_p50", "gflop_per_s_p50",
})


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SchemaRow:
    """One row from a RESULTS_SCHEMA-format CSV."""

    run_id: str
    device: str
    model_checkpoint: str
    quantization: str
    context_length: int
    phase: str
    metric_name: str
    metric_component: str
    value: float
    repeat_index: int
    repeat_count: int

    @property
    def group_key(self) -> tuple[str, str, str, str, int, str, str, str]:
        """Key that uniquely identifies a measurement group for aggregation."""
        return (
            self.device,
            self.model_checkpoint,
            self.quantization,
            "",
            self.context_length,
            self.phase,
            self.metric_name,
            self.metric_component,
        )


@dataclass
class DeviceBenchRow:
    """One row from a device-bench-format CSV."""

    model: str
    kernel: str
    dispatch_path: str
    seq: int
    channels: int
    repeats: int
    p50_us: float
    p95_us: float
    spread_pct: float
    gib_per_s_p50: float
    gflop_per_s_p50: float
    source_file: str = ""

    @property
    def device_label(self) -> str:
        """Extract device name from source file (e.g. rk3588-t3_big → rk3588-t3 big)."""
        stem = self.source_file
        if stem.endswith("_big.csv"):
            return stem.replace("_big.csv", "").replace("_", "-") + " (big)"
        if stem.endswith("_little.csv"):
            return stem.replace("_little.csv", "").replace("_", "-") + " (little)"
        return stem.replace(".csv", "").replace("_", " ")


@dataclass
class AggregatedMetric:
    """Percentile summary of a measurement group across repeats."""

    device: str
    model_checkpoint: str
    quantization: str
    context_length: int
    phase: str
    metric_name: str
    metric_component: str
    p50: float
    p95: float
    spread: float
    normalized_spread: float
    n_repeats: int


# ---------------------------------------------------------------------------
# CSV loading and format detection
# ---------------------------------------------------------------------------


def _detect_format(header: list[str]) -> str:
    """Detect CSV format from header columns."""
    cols = frozenset(header)
    if _SCHEMA_COLUMNS.issubset(cols):
        return "schema"
    if _DEVICE_BENCH_COLUMNS.issubset(cols):
        return "device_bench"
    return "unknown"


def load_csv(path: Path) -> tuple[list[SchemaRow], list[DeviceBenchRow]]:
    """Load a CSV file, auto-detecting its format.

    Returns a tuple of (schema_rows, device_bench_rows). Only one list will be
    populated depending on the detected format.
    """
    schema_rows: list[SchemaRow] = []
    device_rows: list[DeviceBenchRow] = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return [], []
        fmt = _detect_format(list(reader.fieldnames))

        if fmt == "schema":
            for raw in reader:
                metric_component = raw.get("metric_component", "") or ""
                schema_rows.append(
                    SchemaRow(
                        run_id=raw["run_id"],
                        device=raw["device"],
                        model_checkpoint=raw["model_checkpoint"],
                        quantization=raw["quantization"],
                        context_length=int(raw["context_length"]),
                        phase=raw["phase"],
                        metric_name=raw["metric_name"],
                        metric_component=metric_component,
                        value=float(raw["value"]),
                        repeat_index=int(raw["repeat_index"]),
                        repeat_count=int(raw["repeat_count"]),
                    )
                )
        elif fmt == "device_bench":
            for raw in reader:
                device_rows.append(
                    DeviceBenchRow(
                        model=raw["model"],
                        kernel=raw["kernel"],
                        dispatch_path=raw["dispatch_path"],
                        seq=int(raw["seq"]),
                        channels=int(raw["channels"]),
                        repeats=int(raw["repeats"]),
                        p50_us=float(raw["p50_us"]),
                        p95_us=float(raw["p95_us"]),
                        spread_pct=float(raw["spread_pct"]),
                        gib_per_s_p50=float(raw["gib_per_s_p50"]),
                        gflop_per_s_p50=float(raw["gflop_per_s_p50"]),
                        source_file=path.stem,
                    )
                )

    return schema_rows, device_rows


def load_all_csvs(raw_dir: Path | None = None) -> tuple[list[SchemaRow], list[DeviceBenchRow]]:
    """Load all CSVs from the results/raw/ directory."""
    d = raw_dir or _RAW_DIR
    all_schema: list[SchemaRow] = []
    all_device: list[DeviceBenchRow] = []

    if not d.exists():
        return [], []

    for path in sorted(d.glob("*.csv")):
        schema, device = load_csv(path)
        all_schema.extend(schema)
        all_device.extend(device)

    return all_schema, all_device


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile. ``pct`` in [0, 100]."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    # Nearest-rank method: rank = ceil(pct/100 * n), 1-indexed
    import math
    rank = max(1, math.ceil(pct / 100.0 * n))
    return sorted_values[min(rank, n) - 1]


def aggregate_schema_rows(rows: list[SchemaRow]) -> list[AggregatedMetric]:
    """Group schema rows by measurement group and compute percentile summaries."""
    groups: dict[tuple, list[float]] = defaultdict(list)

    for row in rows:
        groups[row.group_key].append(row.value)

    results: list[AggregatedMetric] = []
    for key, values in groups.items():
        sv = sorted(values)
        p50 = _percentile(sv, 50)
        p95 = _percentile(sv, 95)
        spread = p95 - p50
        norm_spread = (spread / p50) if p50 > 0 else 0.0
        results.append(
            AggregatedMetric(
                device=key[0],
                model_checkpoint=key[1],
                quantization=key[2],
                context_length=key[4],
                phase=key[5],
                metric_name=key[6],
                metric_component=key[7],
                p50=p50,
                p95=p95,
                spread=spread,
                normalized_spread=norm_spread,
                n_repeats=len(values),
            )
        )

    results.sort(key=lambda m: (m.device, m.model_checkpoint, m.context_length, m.phase, m.metric_name))
    return results


# ---------------------------------------------------------------------------
# Plot generation (matplotlib, lazily imported)
# ---------------------------------------------------------------------------


def _lazy_mpl() -> Any:
    """Import matplotlib lazily, raising a helpful error if not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot generation. "
            "Install with: pip install -e '.[bench]'"
        ) from exc


# Consistent visual language
_COLOR_MAP = {
    "prefill": "#2563eb",   # blue
    "decode": "#dc2626",    # red
    "weights": "#6b7280",   # gray
    "kv_cache": "#f59e0b",  # amber
    "recurrent_state": "#10b981",  # green
}


def generate_throughput_plot(
    agg: list[AggregatedMetric],
    output_path: Path,
    title: str = "Throughput vs context length",
) -> bool:
    """Generate prefill/decode throughput vs context-length curves.

    Returns True if the plot was generated, False if insufficient data.
    """
    throughput_metrics = {"prefill_tokens_per_sec", "decode_tokens_per_sec"}
    relevant = [m for m in agg if m.metric_name in throughput_metrics]
    if not relevant:
        return False

    plt = _lazy_mpl()

    # Group by (device, model, quant, phase) for separate lines
    series: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
    for m in relevant:
        label_key = (m.device, m.model_checkpoint, m.phase)
        series[label_key].append((m.context_length, m.p50))

    fig, ax = plt.subplots(figsize=(8, 5))

    for (device, model, phase), points in sorted(series.items()):
        points.sort()
        ctxs = [p[0] for p in points]
        vals = [p[1] for p in points]
        color = _COLOR_MAP.get(phase, "#333333")
        short_model = model.split("@")[0].split("/")[-1] if model else ""
        label = f"{short_model} {phase} ({device})"
        marker = "o" if phase == "prefill" else "s"
        ax.plot(ctxs, vals, color=color, marker=marker, linewidth=2, markersize=6, label=label)

    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Tokens / sec (p50)")
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def generate_memory_plot(
    agg: list[AggregatedMetric],
    output_path: Path,
    title: str = "Memory decomposition vs context length",
) -> bool:
    """Generate stacked memory decomposition (weights + kv_cache + recurrent_state).

    Returns True if the plot was generated, False if insufficient data.
    """
    relevant = [m for m in agg if m.metric_name == "peak_memory_bytes"]
    if not relevant:
        return False

    plt = _lazy_mpl()

    # Group by (device, model, quant, phase) — use prefill phase for the main chart
    series: dict[tuple, dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: {"weights": [], "kv_cache": [], "recurrent_state": []}
    )

    for m in relevant:
        if m.metric_component not in ("weights", "kv_cache", "recurrent_state"):
            continue
        label_key = (m.device, m.model_checkpoint, m.phase)
        series[label_key][m.metric_component].append((m.context_length, m.p50))

    if not series:
        return False

    # Pick the first group for the primary chart
    (label_key, components), *_ = list(sorted(series.items()))
    device, model, phase = label_key

    fig, ax = plt.subplots(figsize=(8, 5))

    ctxs_set: set[int] = set()
    for comp_points in components.values():
        ctxs_set.update(p[0] for p in comp_points)

    ctxs = sorted(ctxs_set)
    mib = 1024 * 1024

    bottoms = [0.0] * len(ctxs)
    comp_order = ["weights", "kv_cache", "recurrent_state"]
    for comp_name in comp_order:
        comp_data = dict(components[comp_name])
        vals = [comp_data.get(ctx, 0) / mib for ctx in ctxs]
        color = _COLOR_MAP.get(comp_name, "#333333")
        ax.bar(
            [str(c) for c in ctxs],
            vals,
            bottom=bottoms,
            color=color,
            label=comp_name,
            width=0.5,
        )
        bottoms = [b + v for b, v in zip(bottoms, vals, strict=False)]

    short_model = model.split("@")[0].split("/")[-1] if model else ""
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Peak memory (MiB)")
    ax.set_title(f"{title} — {short_model} {phase} ({device})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def generate_kernel_comparison(
    device_rows: list[DeviceBenchRow],
    output_path: Path,
    title: str = "GDN kernel throughput by device",
) -> bool:
    """Generate a grouped bar chart comparing kernel throughput across devices.

    Returns True if the plot was generated, False if insufficient data.
    """
    if not device_rows:
        return False

    plt = _lazy_mpl()

    # Group by (device_label, model, kernel)
    devices = sorted(set(r.device_label for r in device_rows))
    kernels = sorted(set(r.kernel for r in device_rows))
    models = sorted(set(r.model for r in device_rows))

    # For each (model, kernel), bar per device
    fig, axes = plt.subplots(
        len(models), 1, figsize=(max(8, len(devices) * 2), 4 * len(models)),
        squeeze=False,
    )

    colors = plt.cm.tab10(range(len(devices)))

    for ax_idx, model in enumerate(models):
        ax = axes[ax_idx][0]
        model_rows = [r for r in device_rows if r.model == model]
        x_positions: list[float] = []
        x_labels: list[str] = []
        bar_width = 0.8 / max(len(devices), 1)

        for k_idx, kernel in enumerate(kernels):
            kernel_rows = [r for r in model_rows if r.kernel == kernel]
            if not kernel_rows:
                continue
            x_pos = k_idx
            x_positions.append(x_pos)
            x_labels.append(kernel)

            for d_idx, device in enumerate(devices):
                d_rows = [r for r in kernel_rows if r.device_label == device]
                if not d_rows:
                    continue
                gib = d_rows[0].gib_per_s_p50
                ax.bar(
                    x_pos + d_idx * bar_width - 0.4 + bar_width / 2,
                    gib,
                    width=bar_width,
                    color=colors[d_idx],
                    label=device if k_idx == 0 else "",
                )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("GiB/s (p50)")
        short_model = model.split("/")[-1] if model else model
        ax.set_title(short_model, fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Markdown table generation (stdlib-only, always available)
# ---------------------------------------------------------------------------


def _human_bytes(b: float) -> str:
    """Format bytes as a human-readable string."""
    if b >= 1024 ** 3:
        return f"{b / 1024 ** 3:.1f} GiB"
    if b >= 1024 ** 2:
        return f"{b / 1024 ** 2:.1f} MiB"
    if b >= 1024:
        return f"{b / 1024:.1f} KiB"
    return f"{b:.0f} B"


def _human_throughput(val: float, metric: str) -> str:
    """Format throughput value with appropriate unit."""
    if "tokens_per_sec" in metric:
        if val >= 1000:
            return f"{val:.0f} tok/s"
        return f"{val:.1f} tok/s"
    if metric == "ttft_seconds":
        if val < 1:
            return f"{val * 1000:.0f} ms"
        return f"{val:.2f} s"
    return f"{val:.2f}"


def generate_schema_table(
    agg: list[AggregatedMetric],
    output_path: Path,
) -> bool:
    """Generate a markdown summary table from schema-format aggregated metrics.

    Returns True if a table was generated, False if no data.
    """
    if not agg:
        return False

    lines: list[str] = []
    lines.append("# Benchmark results summary\n")
    lines.append("Auto-generated by `bench/plots.py` from committed CSVs.\n")

    # Group by device
    by_device: dict[str, list[AggregatedMetric]] = defaultdict(list)
    for m in agg:
        by_device[m.device].append(m)

    for device in sorted(by_device):
        device_metrics = by_device[device]
        lines.append(f"\n## {device}\n")

        # Throughput table
        throughput = [m for m in device_metrics if "tokens_per_sec" in m.metric_name]
        if throughput:
            lines.append("\n### Throughput\n")
            lines.append(
                "| Model | Context | Phase | p50 | p95 | Spread | N |"
            )
            lines.append(
                "|---|---:|---|---:|---:|---:|---:|"
            )
            for m in sorted(throughput, key=lambda x: (x.model_checkpoint, x.context_length, x.phase)):
                short_model = m.model_checkpoint.split("@")[0].split("/")[-1]
                lines.append(
                    f"| {short_model} | {m.context_length:,} | {m.phase} | "
                    f"{_human_throughput(m.p50, m.metric_name)} | "
                    f"{_human_throughput(m.p95, m.metric_name)} | "
                    f"{m.normalized_spread:.1%} | {m.n_repeats} |"
                )

        # TTFT table
        ttft = [m for m in device_metrics if m.metric_name == "ttft_seconds"]
        if ttft:
            lines.append("\n### Time to first token\n")
            lines.append("| Model | Context | p50 | p95 | Spread | N |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for m in sorted(ttft, key=lambda x: (x.model_checkpoint, x.context_length)):
                short_model = m.model_checkpoint.split("@")[0].split("/")[-1]
                lines.append(
                    f"| {short_model} | {m.context_length:,} | "
                    f"{_human_throughput(m.p50, m.metric_name)} | "
                    f"{_human_throughput(m.p95, m.metric_name)} | "
                    f"{m.normalized_spread:.1%} | {m.n_repeats} |"
                )

        # Memory table
        memory = [m for m in device_metrics if m.metric_name == "peak_memory_bytes"]
        if memory:
            lines.append("\n### Peak memory by component\n")
            lines.append("| Model | Context | Phase | Component | p50 | Spread | N |")
            lines.append("|---|---:|---|---|---:|---:|---:|")
            for m in sorted(memory, key=lambda x: (x.model_checkpoint, x.context_length, x.phase, x.metric_component)):
                short_model = m.model_checkpoint.split("@")[0].split("/")[-1]
                lines.append(
                    f"| {short_model} | {m.context_length:,} | {m.phase} | "
                    f"{m.metric_component} | {_human_bytes(m.p50)} | "
                    f"{m.normalized_spread:.1%} | {m.n_repeats} |"
                )

        # Energy table
        energy = [m for m in device_metrics if m.metric_name == "energy_joules_per_token"]
        if energy:
            lines.append("\n### Energy per token\n")
            lines.append("| Model | Context | Phase | p50 (J/tok) | Spread | N |")
            lines.append("|---|---:|---|---:|---:|---:|")
            for m in sorted(energy, key=lambda x: (x.model_checkpoint, x.context_length, x.phase)):
                short_model = m.model_checkpoint.split("@")[0].split("/")[-1]
                lines.append(
                    f"| {short_model} | {m.context_length:,} | {m.phase} | "
                    f"{m.p50:.4f} | {m.normalized_spread:.1%} | {m.n_repeats} |"
                )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def generate_device_bench_table(
    device_rows: list[DeviceBenchRow],
    output_path: Path,
) -> bool:
    """Generate a markdown summary table from device-bench-format rows.

    Returns True if a table was generated, False if no data.
    """
    if not device_rows:
        return False

    lines: list[str] = []
    lines.append("# GDN kernel microbenchmark results\n")
    lines.append("Auto-generated by `bench/plots.py` from committed device-bench CSVs.\n")

    # Group by device
    by_device: dict[str, list[DeviceBenchRow]] = defaultdict(list)
    for r in device_rows:
        by_device[r.device_label].append(r)

    for device in sorted(by_device):
        rows = by_device[device]
        lines.append(f"\n## {device}\n")
        lines.append(
            "| Model | Kernel | Dispatch | Seq | Channels | "
            "p50 (µs) | p95 (µs) | Spread | GiB/s | GFLOP/s | N |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(rows, key=lambda x: (x.model, x.kernel)):
            short_model = r.model.split("/")[-1]
            lines.append(
                f"| {short_model} | {r.kernel} | {r.dispatch_path} | "
                f"{r.seq} | {r.channels:,} | "
                f"{r.p50_us:,.1f} | {r.p95_us:,.1f} | {r.spread_pct:.1f}% | "
                f"{r.gib_per_s_p50:.2f} | {r.gflop_per_s_p50:.2f} | {r.repeats} |"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _rel_or_abs(path: Path) -> str:
    """Return path relative to repo root, or absolute if outside it."""
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for plot and table generation."""
    parser = argparse.ArgumentParser(
        prog="bench/plots.py",
        description="Generate plots and tables from committed result CSVs.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_RAW_DIR,
        help="Directory containing raw CSVs (default: results/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_FIGURES_DIR,
        help="Directory for output figures and tables (default: results/figures)",
    )
    parser.add_argument(
        "--format",
        choices=["all", "plots", "tables"],
        default="all",
        help="What to generate (default: all)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (tables only, no matplotlib needed)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List loaded data summary without generating output",
    )

    args = parser.parse_args(argv)

    schema_rows, device_rows = load_all_csvs(args.raw_dir)
    agg = aggregate_schema_rows(schema_rows)

    if args.list:
        print(f"Schema-format rows: {len(schema_rows)}")
        print(f"Device-bench rows:  {len(device_rows)}")
        print(f"Aggregated groups:  {len(agg)}")
        print("\nSchema CSVs found:")
        csvs = sorted(args.raw_dir.glob("*.csv")) if args.raw_dir.exists() else []
        for c in csvs:
            s, d = load_csv(c)
            fmt = "schema" if s else "device_bench" if d else "unknown"
            print(f"  {c.name}: {fmt} ({len(s)} schema, {len(d)} device)")
        if not csvs:
            print("  (none)")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []
    skipped: list[str] = []

    # --- Tables (always generated, stdlib-only) ---
    if args.format in ("all", "tables"):
        schema_table_path = args.output_dir / "results_table.md"
        if generate_schema_table(agg, schema_table_path):
            generated.append(_rel_or_abs(schema_table_path))
        else:
            skipped.append("schema table (no schema-format data)")

        device_table_path = args.output_dir / "kernel_table.md"
        if generate_device_bench_table(device_rows, device_table_path):
            generated.append(_rel_or_abs(device_table_path))
        else:
            skipped.append("device bench table (no device-bench data)")

    # --- Plots (matplotlib required) ---
    if args.format in ("all", "plots") and not args.no_plots:
        try:
            throughput_path = args.output_dir / "throughput_vs_context.png"
            if generate_throughput_plot(agg, throughput_path):
                generated.append(_rel_or_abs(throughput_path))
            else:
                skipped.append("throughput plot (no throughput data)")

            memory_path = args.output_dir / "memory_decomposition.png"
            if generate_memory_plot(agg, memory_path):
                generated.append(_rel_or_abs(memory_path))
            else:
                skipped.append("memory plot (no memory data)")

            kernel_path = args.output_dir / "kernel_comparison.png"
            if generate_kernel_comparison(device_rows, kernel_path):
                generated.append(_rel_or_abs(kernel_path))
            else:
                skipped.append("kernel comparison plot (no device-bench data)")

        except ImportError as e:
            print(f"WARNING: {e}", file=sys.stderr)
            print("Use --no-plots or --format tables for text-only output.", file=sys.stderr)

    # --- Summary ---
    if generated:
        print(f"Generated {len(generated)} file(s):")
        for f in generated:
            print(f"  ✓ {f}")
    else:
        print("No output generated.")

    if skipped:
        print(f"\nSkipped {len(skipped)} (insufficient data):")
        for s in skipped:
            print(f"  — {s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
