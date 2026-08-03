#!/usr/bin/env python3
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
    args = parser.parse_args(argv)

    context_lengths = [int(x) for x in args.context.split(",")]

    csv_paths = run_ablation(
        context_lengths,
        warmup=args.warmup,
        repeats=args.repeats,
        decode_length=args.decode_length,
    )

    # Generate the comparison table
    table = generate_comparison(csv_paths)
    table_path = Path(args.table_output)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(table + "\n", encoding="utf-8")

    print(f"\n  Comparison table: {table_path}")
    print(f"  Configurations: {len(ABLATION_GRID)}")
    print(f"  Context lengths: {context_lengths}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
