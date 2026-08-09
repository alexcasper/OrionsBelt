"""Comparison table generator from harness CSVs (ob-8qt.5).

Reads one or more CSVs in the frozen tidy/long schema (RESULTS_SCHEMA.md) and
produces a markdown comparison table showing configurations as rows and metrics
as columns, with p50 values computed from per-repeat rows.

This is the table structure ``ob-ami`` (master comparison table) needs: it shows
the ablation grid (full-attention-only vs hybrid GDN vs optimized) across context
lengths, with every cell traceable to a manifest-backed run.

Usage::

    python3 -m bench.comparison_table results/raw/*.csv
    python3 -m bench.comparison_table --csv results/raw/run1.csv results/raw/run2.csv

.. warning::

    ``results/figures/comparison_table.md`` is a **hand-curated** file with
    analytical commentary. This script's ``--output`` guard refuses to
    overwrite it. Write auto-generated output to a different filename.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

# Sentinel: this filename in a figures/ directory is hand-curated (234 lines,
# 8 analytical sections). The auto-generated table must never overwrite it.
_CURATED_TABLE_FILENAME = "comparison_table.md"

# Ensure repo root is importable
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.metrics import percentile  # noqa: E402


def load_and_summarize(csv_paths: Sequence[str]) -> list[dict]:
    """Load CSVs, compute p50 per group, return sorted list of summary dicts.

    Groups by (engine_gdn, engine_full_attention, quantization, context_length,
    phase, metric_name, metric_component).
    """
    groups: dict[tuple, list[float]] = defaultdict(list)

    # Required columns from the frozen tidy/long schema (RESULTS_SCHEMA §1–3).
    # CSVs that don't conform (raw kernel microbenchmarks, power logs, etc.)
    # are silently skipped so the tool works on a mixed results/raw/ directory.
    required_cols = {"context_length", "phase", "metric_name", "value"}

    for csv_path in csv_paths:
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or not required_cols.issubset(reader.fieldnames):
                    continue  # non-schema CSV — skip
                for row in reader:
                    # Skip rows from non-sweep CSVs (standard benchmark CSVs use a
                    # different schema without context_length/engine_gdn/value).
                    if "context_length" not in row or "value" not in row:
                        continue
                    try:
                        ctx = int(row["context_length"])
                        val = float(row["value"])
                    except (ValueError, KeyError):
                        continue
                    key = (
                        row.get("engine_gdn", "?"),
                        row.get("engine_full_attention", "?"),
                        row.get("quantization", "?"),
                        ctx,
                        row.get("phase", "?"),
                        row.get("metric_name", "?"),
                        row.get("metric_component") or "",
                    )
                    groups[key].append(val)
        except OSError:
            continue  # unreadable file — skip

    results = []
    for key, values in sorted(groups.items()):
        (eng_gdn, eng_fa, quant, ctx, phase, metric, comp) = key
        results.append(
            {
                "engine_gdn": eng_gdn,
                "engine_full_attention": eng_fa,
                "quantization": quant,
                "context_length": ctx,
                "phase": phase,
                "metric_name": metric,
                "metric_component": comp,
                "p50": percentile(values, 50),
                "p95": percentile(values, 95),
                "n": len(values),
            }
        )
    return results


def _fmt_value(metric: str, value: float) -> str:
    if metric.endswith("_per_sec"):
        return f"{value:.1f}"
    if metric == "ttft_seconds":
        return f"{value * 1000:.1f}ms"
    if metric == "peak_memory_bytes":
        for unit in ("B", "KiB", "MiB", "GiB"):
            if abs(value) < 1024:
                return f"{value:.1f} {unit}"
            value /= 1024
        return f"{value:.1f} TiB"
    return f"{value:.4g}"


def generate_markdown_table(summaries: list[dict]) -> str:
    """Generate a markdown comparison table from summary dicts."""
    # Pivot: rows = (engine config, context_length), columns = (phase, metric, comp)
    # Collect all unique column labels
    col_keys = sorted({(s["phase"], s["metric_name"], s["metric_component"]) for s in summaries})
    row_keys = sorted(
        {
            (s["engine_gdn"], s["engine_full_attention"], s["quantization"], s["context_length"])
            for s in summaries
        }
    )

    # Build lookup: (row_key, col_key) -> summary
    lookup: dict[tuple, dict] = {}
    for s in summaries:
        rk = (s["engine_gdn"], s["engine_full_attention"], s["quantization"], s["context_length"])
        ck = (s["phase"], s["metric_name"], s["metric_component"])
        lookup[(rk, ck)] = s

    # Column headers
    col_labels = []
    for phase, metric, comp in col_keys:
        label = f"{phase}/{metric}"
        if comp:
            label += f"[{comp}]"
        col_labels.append(label)

    lines = []
    # Header
    header = "| Config | Quant | Ctx | " + " | ".join(col_labels) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (3 + len(col_labels)))

    # Rows
    for eng_gdn, eng_fa, quant, ctx in row_keys:
        config = f"{eng_gdn}/{eng_fa}"
        ctx_label = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
        cells = [config, quant, ctx_label]
        for ck in col_keys:
            s = lookup.get(((eng_gdn, eng_fa, quant, ctx), ck))
            if s:
                cells.append(_fmt_value(s["metric_name"], s["p50"]))
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def generate_comparison(csv_paths: Sequence[str]) -> str:
    """Full pipeline: load CSVs, summarize, generate markdown table."""
    summaries = load_and_summarize(csv_paths)
    if not summaries:
        return "No data found in the provided CSVs."
    return generate_markdown_table(summaries)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate comparison table from harness CSVs")
    parser.add_argument("--csv", nargs="+", help="CSV files to read")
    parser.add_argument("--output", "-o", default="", help="Output file (default: stdout)")
    args = parser.parse_args(argv)

    csv_paths = args.csv or [str(p) for p in sorted(Path("results/raw").glob("*.csv"))]
    if not csv_paths:
        print("No CSV files found.", file=sys.stderr)
        return 1

    table = generate_comparison(csv_paths)

    if args.output:
        out = Path(args.output)
        if out.name == _CURATED_TABLE_FILENAME and "figures" in out.parts:
            print(
                "ERROR: refusing to overwrite results/figures/comparison_table.md\n"
                "       This file is hand-curated with 8 analytical sections and is\n"
                "       referenced from docs/FINDINGS.md. The auto-generated table is a\n"
                "       different artifact. Use a different filename, e.g.:\n"
                "         --output results/figures/comparison_table_auto.md",
                file=sys.stderr,
            )
            return 1
        out.write_text(table + "\n", encoding="utf-8")
        print(f"  Table written to {args.output}")
    else:
        print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
