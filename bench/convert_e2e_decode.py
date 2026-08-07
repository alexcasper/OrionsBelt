#!/usr/bin/env python3
"""Convert raw e2e decode CSV to RESULTS_SCHEMA-conformant tidy rows.

The C binary (gdn_e2e_decode.c) emits a simple wide CSV:

    model,tokens,ttft_ms,tok_per_sec_mean,p50_us,p95_us,p99_us,mean_us
    Qwen3.5-4B,8,13254.09,0.08,13248254,13254092,13254092,13245762

This script converts that into the standardized tidy/long format defined in
docs/RESULTS_SCHEMA.md, emitting two rows per invocation:

    - ttft_seconds (prefill phase)       — time to first token
    - decode_tokens_per_sec (decode)      — mean autoregressive throughput

Usage:
    python3 bench/convert_e2e_decode.py \\
        --raw results/raw/rk3588-t3_e2e_decode.csv \\
        --device rk3588-t3 \\
        --output results/raw/rk3588-t3_e2e_schema.csv \\
        --run-id rk3588-t3_e2e_20260806 \\
        --git-sha abc1234 \\
        --manifest-ref results/manifests/rk3588-t3_e2e.json \\
        --quantization fp32 \\
        --cluster big

Stdlib only — runs on edge devices. Bead ob-mrd.8.
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timezone


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert e2e decode CSV to schema-conformant tidy rows."
    )
    p.add_argument("--raw", required=True, help="Path to raw e2e decode CSV")
    p.add_argument("--device", required=True, help="Device enum value (e.g. rk3588-t3)")
    p.add_argument("--output", required=True, help="Output schema CSV path")
    p.add_argument("--run-id", required=True, help="Run identifier")
    p.add_argument("--git-sha", required=True, help="Short git SHA")
    p.add_argument(
        "--manifest-ref",
        required=True,
        help="Path to manifest (relative to repo root)",
    )
    p.add_argument("--quantization", default="fp32", help="Quantization code (default: fp32)")
    p.add_argument(
        "--model-checkpoint",
        default="Qwen/Qwen3.5-4B",
        help="Model checkpoint id (default: Qwen/Qwen3.5-4B)",
    )
    p.add_argument(
        "--context-length",
        type=int,
        default=0,
        help="Effective context length (default: use token count from raw CSV)",
    )
    p.add_argument(
        "--cluster",
        default="all",
        help="CPU cluster used (big/little/all) — goes in notes column",
    )
    return p.parse_args()


def convert(raw_path: str, args: argparse.Namespace) -> None:
    """Read the raw CSV and emit schema-conformant rows."""

    with open(raw_path, newline="") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    if not raw_rows:
        print(f"ERROR: no data rows in {raw_path}", file=sys.stderr)
        sys.exit(1)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cluster_note = f"cluster={args.cluster}" if args.cluster != "all" else ""

    # CSV fieldnames matching RESULTS_SCHEMA.md section 3
    fieldnames = [
        "run_id",
        "timestamp",
        "git_sha",
        "manifest_ref",
        "device",
        "engine_gdn",
        "engine_full_attention",
        "model_checkpoint",
        "quantization",
        "context_length",
        "phase",
        "metric_name",
        "metric_component",
        "value",
        "unit",
        "repeat_index",
        "repeat_count",
        "layer_class",
        "notes",
    ]

    output_rows = []

    for i, raw in enumerate(raw_rows):
        model = raw.get("model", args.model_checkpoint)
        if "/" not in model:
            model = f"Qwen/{model}"
        tokens = int(raw.get("tokens", 0))
        ttft_ms = float(raw.get("ttft_ms", 0))
        tok_per_sec = float(raw.get("tok_per_sec_mean", 0))

        repeat_count = len(raw_rows)
        # context_length: use explicit override, else fall back to token count
        # (the decode sequence length is the effective context for this run)
        ctx_len = args.context_length if args.context_length > 0 else tokens
        note = f"tokens={tokens}"
        if cluster_note:
            note += f";{cluster_note}"

        # TTFT → prefill phase
        output_rows.append(
            {
                "run_id": args.run_id,
                "timestamp": timestamp,
                "git_sha": args.git_sha,
                "manifest_ref": args.manifest_ref,
                "device": args.device,
                "engine_gdn": "cpu",
                "engine_full_attention": "cpu",
                "model_checkpoint": model,
                "quantization": args.quantization,
                "context_length": ctx_len,
                "phase": "prefill",
                "metric_name": "ttft_seconds",
                "metric_component": "",
                "value": f"{ttft_ms / 1000.0:.6f}",
                "unit": "seconds",
                "repeat_index": i,
                "repeat_count": repeat_count,
                "layer_class": "all",
                "notes": note,
            }
        )

        # decode_tokens_per_sec → decode phase
        output_rows.append(
            {
                "run_id": args.run_id,
                "timestamp": timestamp,
                "git_sha": args.git_sha,
                "manifest_ref": args.manifest_ref,
                "device": args.device,
                "engine_gdn": "cpu",
                "engine_full_attention": "cpu",
                "model_checkpoint": model,
                "quantization": args.quantization,
                "context_length": ctx_len,
                "phase": "decode",
                "metric_name": "decode_tokens_per_sec",
                "metric_component": "",
                "value": f"{tok_per_sec:.6f}",
                "unit": "tokens_per_sec",
                "repeat_index": i,
                "repeat_count": repeat_count,
                "layer_class": "all",
                "notes": note,
            }
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} schema-conformant rows to {args.output}")
    for r in output_rows:
        print(f"  {r['phase']:8s} {r['metric_name']:30s} = {r['value']:>12s} {r['unit']}")


if __name__ == "__main__":
    args = parse_args()
    convert(args.raw, args)
