#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Run the ablation matrix: sweep engine configs across context lengths (ob-8qt.5).

Executes the harness across the ablation grid — different engine assignments
(CPU-only, CPU+GPU, CPU+NPU) and quantization levels — at each context length,
then generates the master comparison table from the collected CSVs.

With the synthetic backend this proves the pipeline end-to-end. When real
backends land, the same script produces actual numbers — the grid structure
and table format are unchanged.

Usage::

    python3 scripts/run_ablation.py                    # full grid (synthetic)
    python3 scripts/run_ablation.py --context 4096,32768  # subset of context lengths
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.comparison_table import generate_comparison  # noqa: E402
from bench.harness import (  # noqa: E402
    QWEN35_4B,
    SweepConfig,
    SyntheticBackend,
    run_sweep,
)
from bench.manifest import capture  # noqa: E402
from bench.manifest import write as write_manifest  # noqa: E402
from bench.schema import write_csv  # noqa: E402

# The ablation grid: each entry is one configuration to benchmark.
# engine_gdn / engine_full_attention define the layer-to-engine assignment.
ABLATION_GRID = [
    {
        "name": "cpu-only",
        "engine_gdn": "cpu",
        "engine_full_attention": "cpu",
        "quantization": "fp16",
    },
    {
        "name": "cpu-only-int4",
        "engine_gdn": "cpu",
        "engine_full_attention": "cpu",
        "quantization": "int4_w4a16",
    },
    {
        "name": "cpu-gpu",
        "engine_gdn": "cpu",
        "engine_full_attention": "gpu_vulkan",
        "quantization": "fp16",
    },
    {
        "name": "cpu-gpu-int4",
        "engine_gdn": "cpu",
        "engine_full_attention": "gpu_vulkan",
        "quantization": "int4_w4a16",
    },
    {
        "name": "cpu-npu",
        "engine_gdn": "cpu",
        "engine_full_attention": "npu",
        "quantization": "fp16",
    },
    {
        "name": "cpu-npu-int4",
        "engine_gdn": "cpu",
        "engine_full_attention": "npu",
        "quantization": "int4_w4a16",
    },
]


def _write_manifest_for_rows(
    rows: list,
    config: SweepConfig,
    manifest_dir: str = "results/manifests",
) -> str | None:
    """Capture and write a provenance manifest for an ablation sweep (ob-20t).

    ``run_sweep`` embeds a ``manifest_ref`` in every row but does not write
    the manifest file itself — that was the harness CLI's job. This function
    fills that gap so ablation CSVs are always accompanied by their manifest.
    Returns the manifest path or None if there are no rows.
    """
    if not rows:
        return None
    run_id = rows[0].run_id
    manifest = capture(
        run_id=run_id,
        backend="SyntheticBackend",
        model_checkpoint=config.model_checkpoint,
        quantization=config.quantization,
        decode_length=config.decode_length,
        warmup_count=config.warmup_count,
        repeat_count=config.repeat_count,
        context_lengths=list(config.context_lengths),
    )
    manifest_path = f"{manifest_dir}/{run_id}.json"
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest, manifest_path)
    return manifest_path


def run_ablation(
    context_lengths: list[int],
    warmup: int = 2,
    repeats: int = 5,
    decode_length: int = 20,
    output_dir: str = "results/raw/ablation",
) -> list[str]:
    """Run the full ablation grid. Returns list of CSV paths written."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_paths: list[str] = []

    for entry in ABLATION_GRID:
        name = entry["name"]
        print(f"\n  Running ablation: {name} ...")

        backend = SyntheticBackend(QWEN35_4B)
        config = SweepConfig(
            context_lengths=context_lengths,
            warmup_count=warmup,
            repeat_count=repeats,
            decode_length=decode_length,
            device="generic_aarch64",
            engine_gdn=entry["engine_gdn"],
            engine_full_attention=entry["engine_full_attention"],
            model_checkpoint=f"Qwen/Qwen3.5-4B@ablation-{name}",
            quantization=entry["quantization"],
            notes=f"ablation:{name}",
        )

        rows = run_sweep(backend, config)
        csv_file = str(output_path / f"ablation_{name}.csv")
        write_csv(rows, csv_file)
        csv_paths.append(csv_file)
        print(f"    {len(rows)} rows → {csv_file}")

        # Capture provenance manifest (ob-20t: previously missing)
        mpath = _write_manifest_for_rows(rows, config)
        if mpath:
            print(f"    manifest → {mpath}")

    return csv_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ablation matrix (ob-8qt.5)")
    parser.add_argument(
        "--context",
        default="4096,32768",
        help="Comma-separated context lengths (default: 4096,32768)",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--decode-length", type=int, default=20)
    parser.add_argument(
        "--table-output",
        default="results/figures/ablation_comparison.md",
        help="Where to write the comparison table",
    )
    parser.add_argument(
        "--output-dir",
        default="results/raw/ablation",
        help="Directory for ablation CSVs (default: results/raw/ablation)",
    )
    args = parser.parse_args(argv)

    context_lengths = [int(x) for x in args.context.split(",")]

    csv_paths = run_ablation(
        context_lengths,
        warmup=args.warmup,
        repeats=args.repeats,
        decode_length=args.decode_length,
        output_dir=args.output_dir,
    )

    # Generate the comparison table
    table = generate_comparison(csv_paths)

    # The ablation grid uses SyntheticBackend — the numbers are pipeline
    # placeholders, not real measurements. Make this explicit so no one
    # (including judges) mistakes them for empirical results.
    disclaimer = (
        "> ⚠ **Synthetic data.** These numbers are produced by `SyntheticBackend`, "
        "a deterministic analytical model — not measured on real hardware. "
        "They exist to validate the ablation pipeline end-to-end and to define "
        "the table structure for `ob-ami` (master comparison table). "
        "Real numbers require wiring optimized GDN kernels into a Qwen3.5 "
        "forward pass (`ob-8qt.9`) and running on the target device.\n\n"
    )
    table = disclaimer + table

    table_path = Path(args.table_output)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(table + "\n", encoding="utf-8")

    print(f"\n  Comparison table: {table_path}")
    print(f"  Configurations: {len(ABLATION_GRID)}")
    print(f"  Context lengths: {context_lengths}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
