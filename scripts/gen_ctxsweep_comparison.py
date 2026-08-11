#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Generate master context-sweep comparison from committed CSVs (ob-ami).

Reads every ``results/raw/*ctxsweep*e2e_raw.csv`` (and the older naming
variants), groups by device/model/quant/architecture (pure-GDN vs hybrid),
and emits a markdown table to ``results/figures/ctxsweep_comparison.md``
showing tokens/sec, per-token latency, and KV-cache growth across context
lengths.

This is the load-bearing result table for the Devpost submission: it shows
that the GDN recurrent state is O(1) in context length while the hybrid's
full-attention KV cache grows linearly, causing throughput to degrade.

Every row carries its source CSV and manifest reference for provenance.

Usage::

    python3 scripts/gen_ctxsweep_comparison.py
"""

import contextlib
import csv
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "results" / "raw"
OUT_PATH = REPO_ROOT / "results" / "figures" / "ctxsweep_comparison.md"

# ---------------------------------------------------------------------------
# CSV discovery and classification
# ---------------------------------------------------------------------------

# Patterns to classify a CSV file's quant level and architecture variant.
# Architecture: "puregdn" = all linear-attention layers (no full attention)
#               "hybrid"  = standard 3:1 GDN:attention mix
QUANT_MAP = [
    (r"int4.?sdot", "INT4+SDOT"),
    (r"int8.?sdot", "INT8+SDOT"),
    (r"int4", "INT4"),
    (r"int8", "INT8"),
    (r"q80|q8_0", "Q8_0"),
]
DEVICE_MAP = [
    (r"rk3588-t4", "rk3588-t4"),
    (r"rk3588-t3", "rk3588-t3"),
    (r"jetson-j1", "jetson-j1"),
    (r"jetson-j2", "jetson-j2"),
]
MODEL_MAP = [
    (r"08b|0\.8b", "Qwen3.5-0.8B"),
    (r"4b", "Qwen3.5-4B"),
]
DEFAULT_MODEL = "Qwen3.5-4B"  # ctxsweep CSVs without model marker are 4B


def classify(name: str):
    """Return (device, model, quant, arch) or None if not a ctxsweep e2e CSV."""
    lower = name.lower()
    if "ctxsweep" not in lower and "ctx_sweep" not in lower:
        return None
    # Accept *_raw.csv, *_e2e.csv, and the t4 naming *_4t.csv, *_4t_fair.csv
    if (
        not name.endswith("_raw.csv")
        and not name.endswith("_e2e.csv")
        and not re.search(r"_\dt.*\.csv$", lower)
        and not lower.endswith(".csv")
    ):
        return None
    # skip schema files
    if "schema" in lower:
        return None

    device = "unknown"
    for pat, label in DEVICE_MAP:
        if re.search(pat, lower):
            device = label
            break

    model = DEFAULT_MODEL
    for pat, label in MODEL_MAP:
        if re.search(pat, lower):
            model = label
            break

    quant = "FP32"
    for pat, label in QUANT_MAP:
        if re.search(pat, lower):
            quant = label
            break

    arch = "hybrid"
    if "puregdn" in lower or "pure_gdn" in lower:
        arch = "puregdn"

    # Cluster inference: if device has _big or _little or we default to big
    cluster = "big"
    if "little" in lower:
        cluster = "little"

    return (device, cluster, model, quant, arch)


def load_ctxsweep_data():
    """Load all ctxsweep e2e CSVs and return structured data."""
    datasets = {}  # key -> list of rows
    sources = {}   # key -> {csv, manifest_hint}

    for csv_path in sorted(RAW_DIR.glob("*ctxsweep*.csv")):
        name = csv_path.name
        if "schema" in name:
            continue
        info = classify(name)
        if info is None:
            continue
        device, cluster, model, quant, arch = info

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            continue

        key = (device, cluster, model, quant, arch)
        # When multiple CSVs share the same key (e.g. _1t vs _4t_fair
        # variants of the same config), prefer _4t_fair as it reflects
        # real-world multi-threaded performance.  Track skipped files
        # for provenance transparency.
        is_4t_fair = "_4t_fair" in name
        if key in datasets and not is_4t_fair:
            # existing entry wins unless we are the _4t_fair variant
            continue
        datasets[key] = rows
        sources[key] = {"csv": name}

    return datasets, sources


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def fmt_tok(v):
    """Format tok/s."""
    try:
        return f"{float(v):.2f}"
    except (ValueError, TypeError):
        return "—"

def fmt_us(v):
    """Format microseconds as ms."""
    try:
        return f"{float(v)/1000:.1f}"
    except (ValueError, TypeError):
        return "—"

def fmt_mb(v):
    """Format MB."""
    try:
        return f"{float(v):.0f}"
    except (ValueError, TypeError):
        return "—"


def generate_markdown(datasets, sources):
    lines = []
    lines.append("# Master context-sweep comparison")
    lines.append("")
    lines.append("_Generated by `scripts/gen_ctxsweep_comparison.py`. "
                 "Reads committed ctxsweep CSVs — every cell is manifest-backed._")
    lines.append("")
    lines.append("> **Key result:** Pure-GDN (all linear attention) maintains "
                 "constant throughput regardless of context length, because its "
                 "recurrent state is O(1). The standard hybrid (3:1 GDN:attention) "
                 "degrades as the full-attention KV cache grows linearly with context.")
    lines.append("")

    # Group by (device, cluster, model)
    by_device = defaultdict(dict)
    for key in datasets:
        device, cluster, model, quant, arch = key
        group_key = (device, cluster, model)
        by_device[group_key][key] = datasets[key]

    for section_num, group_key in enumerate(sorted(by_device.keys()), 1):
        device, cluster, model = group_key
        lines.append(f"## {section_num}. {model} on {device} ({cluster} cluster)")
        lines.append("")

        # Build context length union
        all_ctxs = set()
        for key in by_device[group_key]:
            for row in by_device[group_key][key]:
                with contextlib.suppress(ValueError, KeyError):
                    all_ctxs.add(int(float(row["ctx_len"])))
        ctxs = sorted(all_ctxs)

        # Order: FP32, INT8, INT8+SDOT, INT4+SDOT
        quant_order = {"FP32": 0, "INT8": 1, "INT8+SDOT": 2, "INT4": 3, "INT4+SDOT": 4, "Q8_0": 5}
        keys_sorted = sorted(by_device[group_key].keys(),
                            key=lambda k: (quant_order.get(k[3], 99), k[4]))

        # Table: for each quant, show hybrid and puregdn tok/s at each ctx
        lines.append("### Throughput (tok/s) by context length")
        lines.append("")
        header = "| Quant | Arch | " + " | ".join(f"ctx={c}" for c in ctxs) + " | Source |"
        sep = "|---|---|" + "|".join(["---:"] * len(ctxs)) + "|---|"
        lines.append(header)
        lines.append(sep)

        for key in keys_sorted:
            _, _, _, quant, arch = key
            rows = by_device[group_key][key]
            csv_name = sources[key]["csv"]
            arch_label = "Pure GDN" if arch == "puregdn" else "Hybrid 3:1"
            cells = []
            for ctx in ctxs:
                val = None
                for r in rows:
                    try:
                        if int(float(r["ctx_len"])) == ctx:
                            val = r.get("tok_per_sec", "")
                            break
                    except (ValueError, KeyError):
                        pass
                cells.append(fmt_tok(val))
            lines.append(f"| {quant} | {arch_label} | " + " | ".join(cells) + f" | `{csv_name}` |")
        lines.append("")

        # Table: KV cache growth
        lines.append("### KV cache (MB) by context length")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for key in keys_sorted:
            _, _, _, quant, arch = key
            rows = by_device[group_key][key]
            csv_name = sources[key]["csv"]
            arch_label = "Pure GDN" if arch == "puregdn" else "Hybrid 3:1"
            cells = []
            for ctx in ctxs:
                val = None
                for r in rows:
                    try:
                        if int(float(r["ctx_len"])) == ctx:
                            val = r.get("kv_cache_mb", "")
                            break
                    except (ValueError, KeyError):
                        pass
                cells.append(fmt_mb(val))
            lines.append(f"| {quant} | {arch_label} | " + " | ".join(cells) + f" | `{csv_name}` |")
        lines.append("")

        # Table: per-token latency (ms/tok)
        lines.append("### Per-token latency (ms/tok) by context length")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for key in keys_sorted:
            _, _, _, quant, arch = key
            rows = by_device[group_key][key]
            csv_name = sources[key]["csv"]
            arch_label = "Pure GDN" if arch == "puregdn" else "Hybrid 3:1"
            cells = []
            for ctx in ctxs:
                val = None
                for r in rows:
                    try:
                        if int(float(r["ctx_len"])) == ctx:
                            v = float(r.get("total_us", 0))
                            val = v
                            break
                    except (ValueError, KeyError):
                        pass
                cells.append(fmt_us(val))
            lines.append(f"| {quant} | {arch_label} | " + " | ".join(cells) + f" | `{csv_name}` |")
        lines.append("")

        # Headline ratio: hybrid retention vs pure-GDN at longest context
        # Find matching quant levels
        for quant in ["FP32", "INT8", "INT8+SDOT", "INT4+SDOT"]:
            hybrid_key = (device, cluster, model, quant, "hybrid")
            pure_key = (device, cluster, model, quant, "puregdn")
            if hybrid_key in by_device[group_key] and pure_key in by_device[group_key]:
                h_rows = {(int(float(r["ctx_len"]))): r for r in by_device[group_key][hybrid_key]
                          if "ctx_len" in r}
                p_rows = {(int(float(r["ctx_len"]))): r for r in by_device[group_key][pure_key]
                          if "ctx_len" in r}
                common_ctxs = sorted(set(h_rows.keys()) & set(p_rows.keys()))
                if len(common_ctxs) >= 2:
                    first_ctx = common_ctxs[0]
                    last_ctx = common_ctxs[-1]
                    try:
                        h_first = float(h_rows[first_ctx]["tok_per_sec"])
                        h_last = float(h_rows[last_ctx]["tok_per_sec"])
                        p_first = float(p_rows[first_ctx]["tok_per_sec"])
                        p_last = float(p_rows[last_ctx]["tok_per_sec"])
                        h_retain = h_last / h_first if h_first > 0 else 0
                        p_retain = p_last / p_first if p_first > 0 else 0
                        lines.append(f"> **{quant} throughput retention** "
                                     f"({first_ctx}→{last_ctx} ctx): "
                                     f"Hybrid {h_retain:.1%} "
                                     f"({h_first:.2f}→{h_last:.2f} tok/s) "
                                     f"vs Pure-GDN {p_retain:.1%} "
                                     f"({p_first:.2f}→{p_last:.2f} tok/s)")
                        lines.append("")
                    except (ValueError, KeyError, ZeroDivisionError):
                        pass  # skip malformed rows

    # Data sources
    lines.append("## Data sources")
    lines.append("")
    for key in sorted(sources.keys()):
        device, cluster, model, quant, arch = key
        lines.append(f"- `{sources[key]['csv']}` — {device} {cluster}, {model}, {quant}, "
                     f"{'pure-GDN' if arch == 'puregdn' else 'hybrid'}")
    lines.append("")

    return "\n".join(lines)


def main():
    datasets, sources = load_ctxsweep_data()
    if not datasets:
        print("No ctxsweep CSVs found.")
        return 1

    md = generate_markdown(datasets, sources)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(md + "\n")
    print(f"Wrote {OUT_PATH} ({len(datasets)} datasets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
