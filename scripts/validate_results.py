#!/usr/bin/env python3
"""Validate benchmark result CSVs and their provenance manifests.

Scans results/raw/*.csv, checks each for:
  1. A corresponding manifest in results/manifests/
  2. CSV schema conformance (expected columns for each CSV type)
  3. Suspicious values (impossible throughput, negative latency, p95 < p50)
  4. Stale git_sha in manifest (flagged if != current HEAD)

Designed to run on any fleet device, including Python 3.6.9 (Jetson Nano).
Uses only the standard library — no numpy, no dataclasses, no future annotations.

Usage:
    python3 scripts/validate_results.py
    python3 scripts/validate_results.py --csv-dir results/raw --manifest-dir results/manifests
    python3 scripts/validate_results.py --quiet   # exit code only
"""

import argparse
import csv
import json
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

STANDARD_COLS = [
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

SUSTAINED_COLS = [
    "sustained_model",
    "sustained_kernel",
    "dispatch_path",
    "elapsed_s",
    "throughput_gibs",
    "thermal_c",
    "vs_first_pct",
]

POWER_COLS = [
    "timestamp_ms",
    "power_in_mw",
    "power_gpu_mw",
    "power_cpu_mw",
    "temp_milliC",
]

# Delta-rule matmul microbenchmark CSV (from bench_gdn --delta-matmul mode)
DELTA_MATMUL_COLS = [
    "kernel",
    "M",
    "K",
    "N",
    "repeats",
    "p50_us",
    "p95_us",
    "gib_per_s_p50",
]

# Per-layer latency profiling CSV (from profile_layers.py)
LAYER_PROFILE_COLS = [
    "phase",
    "ctx_len",
    "layer_idx",
    "layer_type",
    "p50_us",
    "p95_us",
    "mean_us",
    "n_samples",
]

# End-to-end C decode-loop CSV (from gdn_e2e_decode.c)
E2E_DECODE_COLS = [
    "model",
    "tokens",
    "ttft_ms",
    "tok_per_sec_mean",
    "p50_us",
    "p95_us",
    "p99_us",
    "mean_us",
    "gdn_proj_pct",
    "gdn_conv_pct",
    "gdn_decay_pct",
    "gdn_scan_pct",
    "gdn_oproj_pct",
    "full_pct",
    "ffn_pct",
]

# Context-length sweep CSV (from gdn_e2e_decode.c --ctx-sweep mode, ob-8qt.12)
CTX_SWEEP_COLS = [
    "model",
    "ctx_len",
    "gdn_layer_us",
    "full_attn_us",
    "ffn_us",
    "total_us",
    "tok_per_sec",
    "kv_cache_mb",
]

# End-to-end context-sweep CSV (from run_ablation.py / harness.py e2e mode)
E2E_SWEEP_COLS = [
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

# E2e context-sweep raw CSV (from gdn_e2e_decode.c --ctx-sweep mode)
E2E_CTXSWEEP_RAW_COLS = [
    "model",
    "ctx_len",
    "gdn_layer_us",
    "full_attn_us",
    "ffn_us",
    "total_us",
    "tok_per_sec",
    "kv_cache_mb",
]

# GPU microbenchmark CSV (from gpu/gdn_gpu_bench.c --csv, Mali-G610 OpenCL)
GPU_MICRO_COLS = [
    "kernel",
    "dim1",
    "dim2",
    "dim3",
    "p50_ms",
    "p95_ms",
    "bw_mibs",
]

# Device spec bandwidth (GiB/s) for sanity-check upper bounds.
# From DEVICE_RUNBOOK.md "What we are actually testing".
DEVICE_SPEC_BW = {
    "pi5": 17.0,
    "rk3588": 34.0,
    "jetson": 25.6,
}

# Absolute upper bound — no device should exceed this for a single kernel.
ABSURD_THROUGHPUT = 200.0  # GiB/s


def detect_csv_type(header):
    """Return CSV type: standard, sustained, power, layer_profile, delta_matmul, e2e_decode, ctx_sweep, e2e_sweep, or None."""
    cols = set(header)
    if cols >= set(STANDARD_COLS):
        return "standard"
    if cols >= set(SUSTAINED_COLS):
        return "sustained"
    if cols >= set(POWER_COLS):
        return "power"
    if cols >= {"layer_idx", "layer_type", "p50_us", "mean_us"}:
        return "layer_profile"
    if cols >= {"kernel", "M", "K", "N"}:
        return "delta_matmul"
    if cols >= {"tok_per_sec_mean", "gdn_proj_pct", "ffn_pct"}:
        return "e2e_decode"
    if cols >= {"ctx_len", "gdn_layer_us", "full_attn_us", "kv_cache_mb"}:
        return "ctx_sweep"
    if cols >= {"run_id", "metric_name", "metric_component", "repeat_index"}:
        return "e2e_sweep"
    if cols >= {"gdn_layer_us", "kv_cache_mb", "total_us"}:
        return "e2e_ctxsweep"
    if cols >= {"bw_mibs", "p50_ms", "dim1"}:
        return "gpu_micro"
    return None


def expected_columns(csv_type):
    if csv_type == "standard":
        return STANDARD_COLS
    if csv_type == "sustained":
        return SUSTAINED_COLS
    if csv_type == "power":
        return POWER_COLS
    if csv_type == "layer_profile":
        return LAYER_PROFILE_COLS
    if csv_type == "delta_matmul":
        return DELTA_MATMUL_COLS
    if csv_type == "e2e_decode":
        return E2E_DECODE_COLS
    if csv_type == "ctx_sweep":
        return CTX_SWEEP_COLS
    if csv_type == "e2e_sweep":
        return E2E_SWEEP_COLS
    if csv_type == "e2e_ctxsweep":
        return E2E_CTXSWEEP_RAW_COLS
    if csv_type == "gpu_micro":
        return GPU_MICRO_COLS
    return []


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


class Issue:
    """A single validation issue."""

    def __init__(self, severity, csv_name, message):
        self.severity = severity  # "ERROR", "WARNING", "NOTE"
        self.csv_name = csv_name
        self.message = message

    def __str__(self):
        return f"  {self.severity:s}: {self.csv_name:s} -- {self.message:s}"


def get_git_head_sha():
    """Return current HEAD sha, or None if not in a git repo."""
    try:
        result = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return result
    except Exception:
        return None


def find_device_spec(csv_name):
    """Guess the device from the CSV filename to look up spec bandwidth."""
    name_lower = csv_name.lower()
    for key, bw in DEVICE_SPEC_BW.items():
        if key in name_lower:
            return bw
    return None


def check_manifest_exists(csv_name, manifest_dir):
    """Check if a manifest exists for the given CSV basename."""
    # Try several manifest naming conventions
    base = os.path.splitext(csv_name)[0]
    candidates = [
        base + ".json",
        base.replace("_", "-") + ".json",
    ]
    # Also try without suffixes like _big, _little, _sustained_*, etc.
    parts = base.split("_")
    if len(parts) > 1:
        candidates.append(parts[0] + ".json")

    for candidate in candidates:
        path = os.path.join(manifest_dir, candidate)
        if os.path.isfile(path):
            return path
    return None


def check_ablation_manifests(ablation_dir, issues):
    """Validate that ablation CSVs' embedded manifest_ref paths exist (ob-20t).

    Ablation CSVs use the bench.schema.ResultRow format and embed the manifest
    path in each row's ``manifest_ref`` column.  This checks every distinct
    manifest_ref referenced across all ablation CSVs and verifies the file
    exists on disk.
    """
    if not os.path.isdir(ablation_dir):
        return

    for fname in sorted(os.listdir(ablation_dir)):
        if not fname.endswith(".csv"):
            continue
        csv_path = os.path.join(ablation_dir, fname)
        refs = set()
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ref = row.get("manifest_ref", "")
                    if ref:
                        refs.add(ref)
        except Exception:
            issues.append(Issue("ERROR", fname, "cannot read ablation CSV"))
            continue

        for ref in sorted(refs):
            if os.path.isfile(ref):
                issues.append(Issue("NOTE", fname, f"ablation manifest_ref OK: {ref}"))
            else:
                issues.append(
                    Issue(
                        "WARNING",
                        fname,
                        f"ablation manifest_ref points to MISSING file: {ref}",
                    )
                )


def load_manifest(path):
    """Load and parse a manifest JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def validate_standard_row(row, csv_name, issues, row_num):
    """Validate a single row of a standard benchmark CSV."""
    try:
        p50 = float(row["p50_us"])
        p95 = float(row["p95_us"])
        spread = float(row["spread_pct"])
        gib = float(row["gib_per_s_p50"])
        repeats = int(row["repeats"])
        # Parsed but not otherwise inspected: the conversion itself is the check,
        # so a malformed value lands in the except below as a parse error.
        float(row["gflop_per_s_p50"])
        int(row["seq"])
    except (ValueError, KeyError) as e:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: cannot parse numeric fields: {e}"))
        return

    # Negative latency
    if p50 <= 0 or p95 <= 0:
        issues.append(
            Issue("ERROR", csv_name, f"row {row_num}: non-positive latency (p50={p50}, p95={p95})")
        )

    # p95 < p50 is impossible
    if p95 < p50:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: p95 ({p95}) < p50 ({p50})"))

    # Absurd throughput
    if gib > ABSURD_THROUGHPUT:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: absurd throughput {gib} GiB/s"))

    # Very high spread
    if spread > 200:
        issues.append(Issue("WARNING", csv_name, f"row {row_num}: extreme spread {spread}%"))

    # Too few repeats
    if repeats < 5:
        issues.append(
            Issue("ERROR", csv_name, f"row {row_num}: repeats={repeats} (minimum 5 per METRICS.md)")
        )


def validate_sustained_row(row, csv_name, issues, row_num):
    """Validate a single row of a sustained-load CSV."""
    try:
        elapsed = float(row["elapsed_s"])
        tput = float(row["throughput_gibs"])
        thermal = float(row["thermal_c"])
    except (ValueError, KeyError) as e:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: cannot parse: {e}"))
        return

    if tput > ABSURD_THROUGHPUT:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: absurd throughput {tput} GiB/s"))
    if thermal > 120:
        issues.append(Issue("WARNING", csv_name, f"row {row_num}: very high thermal {thermal}C"))
    if elapsed <= 0:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: non-positive elapsed_s"))


def validate_layer_profile_row(row, csv_name, issues, row_num):
    """Validate a single row of a layer-profile CSV."""
    try:
        p50 = float(row["p50_us"])
        p95 = float(row["p95_us"])
        mean = float(row["mean_us"])
        n = int(row["n_samples"])
        ctx = int(row["ctx_len"])
        layer_idx = int(row["layer_idx"])
    except (ValueError, KeyError) as e:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: cannot parse: {e}"))
        return

    if p50 <= 0:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: non-positive p50_us"))
    if p95 < p50:
        issues.append(Issue("WARNING", csv_name, f"row {row_num}: p95 ({p95}) < p50 ({p50})"))
    if mean <= 0:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: non-positive mean_us"))
    if n < 1:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: n_samples={n}"))
    if ctx <= 0:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: non-positive ctx_len"))
    if layer_idx < 0:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: negative layer_idx"))


def validate_delta_matmul_row(row, csv_name, issues, row_num):
    """Validate a single row of a delta-rule matmul CSV."""
    try:
        m = int(row["M"])
        k = int(row["K"])
        n = int(row["N"])
        p50 = float(row["p50_us"])
        p95 = float(row["p95_us"])
    except (ValueError, KeyError) as e:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: cannot parse: {e}"))
        return

    if m < 1 or k < 1 or n < 1:
        issues.append(
            Issue("ERROR", csv_name, f"row {row_num}: non-positive matmul dim M={m},K={k},N={n}")
        )
    if p50 <= 0:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: non-positive p50_us"))
    if p95 < p50:
        issues.append(Issue("WARNING", csv_name, f"row {row_num}: p95 ({p95}) < p50 ({p50})"))


def validate_ctx_sweep_row(row, csv_name, issues, row_num):
    """Validate a single row of a context-length sweep CSV (gdn_e2e_decode.c --ctx-sweep)."""
    try:
        ctx = int(row["ctx_len"])
        gdn_us = float(row["gdn_layer_us"])
        full_us = float(row["full_attn_us"])
        ffn_us = float(row["ffn_us"])
        total_us = float(row["total_us"])
        tps = float(row["tok_per_sec"])
        kv_mb = float(row["kv_cache_mb"])
    except (ValueError, KeyError) as e:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: cannot parse numeric fields: {e}"))
        return

    if ctx < 1:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: non-positive ctx_len"))
    for label, val in [
        ("gdn_layer_us", gdn_us),
        ("ffn_us", ffn_us),
        ("total_us", total_us),
    ]:
        if val <= 0:
            issues.append(Issue("ERROR", csv_name, f"row {row_num}: non-positive {label}"))
    # full_attn_us is legitimately 0 for --pure-gdn sweeps (no full-attention
    # layers exist), so only a negative value is an error.
    if full_us < 0:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: negative full_attn_us"))
    if tps <= 0:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: non-positive tok_per_sec"))
    if kv_mb < 0:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: negative kv_cache_mb"))


def validate_e2e_sweep_row(row, csv_name, issues, row_num):
    """Validate a single row of an e2e context-sweep CSV (bench/schema.py format)."""
    try:
        value = float(row["value"])
        ctx = int(row["context_length"])
        repeat_idx = int(row["repeat_index"])
        repeat_count = int(row["repeat_count"])
    except (ValueError, KeyError) as e:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: cannot parse numeric fields: {e}"))
        return

    # Negative values are invalid for all our metrics (throughput, latency, memory)
    if value < 0:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: negative value {value}"))

    if ctx <= 0:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: non-positive context_length"))

    if repeat_count < 1:
        issues.append(
            Issue("ERROR", csv_name, f"row {row_num}: repeat_count={repeat_count} (must be >= 1)")
        )
    if repeat_idx < 0 or repeat_idx >= repeat_count:
        issues.append(
            Issue(
                "WARNING",
                csv_name,
                f"row {row_num}: repeat_index {repeat_idx} out of range [0,{repeat_count})",
            )
        )

    phase = row.get("phase", "")
    if phase not in ("prefill", "decode"):
        issues.append(Issue("WARNING", csv_name, f"row {row_num}: unexpected phase '{phase}'"))

    # Sanity-check throughput-like metrics
    unit = row.get("unit", "")
    metric = row.get("metric_name", "")
    if ("per_sec" in unit or "per_sec" in metric) and value <= 0:
        issues.append(Issue("WARNING", csv_name, f"row {row_num}: non-positive {metric} = {value}"))
    if "tokens_per_sec" in metric and value > 100000:
        issues.append(
            Issue("WARNING", csv_name, f"row {row_num}: very high throughput {value} tokens/sec")
        )


_GPU_KERNELS = {"gdn_gated_scan", "gdn_cumdecay", "gdn_causal_dwconv1d", "gdn_delta_rule_decode"}


def validate_gpu_micro_row(row, csv_name, issues, row_num):
    """Validate a single row of a GPU microbenchmark CSV (gpu/gdn_gpu_bench.c)."""
    kernel = row.get("kernel", "")
    if kernel not in _GPU_KERNELS:
        issues.append(Issue("WARNING", csv_name, f"row {row_num}: unknown kernel '{kernel}'"))

    try:
        dim1 = int(row["dim1"])
        dim2 = int(row["dim2"])
    except (ValueError, KeyError) as e:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: cannot parse dims: {e}"))
        return
    if dim1 < 1 or dim2 < 1:
        issues.append(
            Issue("ERROR", csv_name, f"row {row_num}: non-positive dims dim1={dim1},dim2={dim2}")
        )

    # dim3 is optional — some kernels only use 2 dims
    dim3_str = row.get("dim3", "")
    if dim3_str:
        try:
            dim3 = int(dim3_str)
            if dim3 < 1:
                issues.append(Issue("ERROR", csv_name, f"row {row_num}: non-positive dim3={dim3}"))
        except ValueError:
            issues.append(
                Issue("WARNING", csv_name, f"row {row_num}: non-integer dim3='{dim3_str}'")
            )

    try:
        p50 = float(row["p50_ms"])
        if p50 <= 0:
            issues.append(Issue("ERROR", csv_name, f"row {row_num}: non-positive p50_ms"))
    except (ValueError, KeyError):
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: cannot parse p50_ms"))
        return  # cannot validate p95 ordering without p50

    # p95 is optional — some kernels don't report it
    p95_str = row.get("p95_ms", "")
    if p95_str:
        try:
            p95 = float(p95_str)
            if p95 < p50:
                issues.append(
                    Issue("WARNING", csv_name, f"row {row_num}: p95 ({p95}) < p50 ({p50})")
                )
        except ValueError:
            issues.append(
                Issue("WARNING", csv_name, f"row {row_num}: non-numeric p95_ms='{p95_str}'")
            )

    try:
        bw = float(row["bw_mibs"])
        if bw <= 0:
            issues.append(Issue("ERROR", csv_name, f"row {row_num}: non-positive bw_mibs"))
    except (ValueError, KeyError):
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: cannot parse bw_mibs"))


def validate_csv(path, csv_name, issues):
    """Validate CSV schema and row-level data. Return (csv_type, row_count, manifest_ref)."""
    if not os.path.isfile(path):
        issues.append(Issue("ERROR", csv_name, "file not found"))
        return None, 0, None

    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            if not header:
                issues.append(Issue("ERROR", csv_name, "empty or unreadable CSV"))
                return None, 0, None

            csv_type = detect_csv_type(header)
            if csv_type is None:
                issues.append(
                    Issue("WARNING", csv_name, f"unrecognized CSV format ({len(header)} columns)")
                )
                return None, 0, None

            expected = expected_columns(csv_type)
            missing = [c for c in expected if c not in header]
            if missing:
                issues.append(
                    Issue(
                        "ERROR", csv_name, "missing required columns: {}".format(", ".join(missing))
                    )
                )

            row_count = 0
            manifest_ref = None
            for i, row in enumerate(reader, start=2):  # row 1 is header
                row_count += 1
                if csv_type == "standard":
                    validate_standard_row(row, csv_name, issues, i)
                elif csv_type == "sustained":
                    validate_sustained_row(row, csv_name, issues, i)
                elif csv_type == "layer_profile":
                    validate_layer_profile_row(row, csv_name, issues, i)
                elif csv_type == "delta_matmul":
                    validate_delta_matmul_row(row, csv_name, issues, i)
                elif csv_type == "e2e_decode":
                    # Basic sanity: tok_per_sec_mean must be positive
                    try:
                        tps = float(row.get("tok_per_sec_mean", 0))
                        if tps <= 0 or tps > 1000:
                            issues.append(
                                Issue("WARNING", csv_name, f"row {i}: implausible tok/s {tps}")
                            )
                    except ValueError:
                        pass
                elif csv_type == "ctx_sweep":
                    validate_ctx_sweep_row(row, csv_name, issues, i)
                elif csv_type == "e2e_sweep":
                    validate_e2e_sweep_row(row, csv_name, issues, i)
                    # e2e-sweep rows embed their own manifest path (bench.schema.ResultRow) —
                    # every row in one sweep CSV shares the same manifest, so the first is enough.
                    if manifest_ref is None:
                        manifest_ref = row.get("manifest_ref") or None
                elif csv_type == "gpu_micro":
                    validate_gpu_micro_row(row, csv_name, issues, i)

            return csv_type, row_count, manifest_ref

    except Exception as e:
        issues.append(Issue("ERROR", csv_name, f"cannot read CSV: {e}"))
        return None, 0, None


def validate_manifest(csv_name, csv_type, row_count, manifest_path, issues, head_sha):
    """Validate manifest content for a CSV."""
    if manifest_path is None:
        issues.append(Issue("WARNING", csv_name, "no manifest found"))
        return

    manifest = load_manifest(manifest_path)
    if manifest is None:
        issues.append(
            Issue("ERROR", csv_name, f"manifest exists but is invalid JSON: {manifest_path}")
        )
        return

    issues.append(Issue("NOTE", csv_name, f"manifest: {os.path.basename(manifest_path)}"))

    # Check git SHA staleness (NOTE, not WARNING — manifests are captured at run time)
    git_info = manifest.get("git", {})
    sha = git_info.get("sha", "")

    # A manifest with no git section at all has zero provenance — worse than
    # a dirty tree, because there is no SHA to trace back to.
    if not sha and not git_info.get("dirty"):
        issues.append(
            Issue(
                "WARNING",
                csv_name,
                "manifest has no git section -- no provenance (no SHA, no dirty flag)",
            )
        )

    if sha and head_sha and sha != head_sha:
        short_sha = sha[:7]
        issues.append(
            Issue(
                "NOTE",
                csv_name,
                f"manifest git_sha {short_sha} (run-time snapshot, not current HEAD)",
            )
        )

    # A dirty tree at capture time means the recorded SHA does NOT identify the code
    # that produced these numbers, so two runs labelled with the same commit may have
    # run different binaries. This is a WARNING rather than a NOTE because it silently
    # invalidates cross-run comparison, and it currently affects every fleet manifest.
    if git_info.get("dirty"):
        issues.append(
            Issue(
                "WARNING",
                csv_name,
                "captured from a DIRTY tree -- git_sha {} does not identify the "
                "code that ran, so this run is not safely comparable to another "
                "at the 'same' commit".format(sha[:7] if sha else "?"),
            )
        )

    # For standard CSVs, flag low row counts
    if csv_type == "standard" and row_count < 12:
        issues.append(
            Issue(
                "NOTE",
                csv_name,
                f"only {row_count} rows -- may be missing decode or mixed-precision variants",
            )
        )

    # Check device spec bandwidth vs achieved
    spec_bw = find_device_spec(csv_name)
    if spec_bw and csv_type in ("standard", "e2e_sweep") and row_count > 0:
        issues.append(Issue("NOTE", csv_name, f"device spec bandwidth: ~{spec_bw:.0f} GiB/s"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Validate benchmark CSVs and manifests.")
    parser.add_argument(
        "--csv-dir",
        default="results/raw",
        help="Directory containing CSV files (default: results/raw)",
    )
    parser.add_argument(
        "--manifest-dir",
        default="results/manifests",
        help="Directory containing manifest JSON files (default: results/manifests)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output, use exit code only")
    args = parser.parse_args()

    if not os.path.isdir(args.csv_dir):
        print(f"Error: CSV directory not found: {args.csv_dir}")
        return 2

    head_sha = get_git_head_sha()
    all_issues = []
    csv_files = sorted(f for f in os.listdir(args.csv_dir) if f.endswith(".csv"))

    if not csv_files:
        if not args.quiet:
            print(f"No CSV files found in {args.csv_dir}")
        return 0

    for csv_name in csv_files:
        csv_path = os.path.join(args.csv_dir, csv_name)
        csv_type, row_count, manifest_ref = validate_csv(csv_path, csv_name, all_issues)

        # e2e-sweep CSVs embed their manifest path in-row; prefer that, fall back
        # to filename-based matching (e.g. if the embedded path was moved/renamed).
        manifest_path = None
        if csv_type == "e2e_sweep" and manifest_ref:
            candidate = (
                manifest_ref if os.path.isabs(manifest_ref) else os.path.join(".", manifest_ref)
            )
            if os.path.isfile(candidate):
                manifest_path = candidate
        if manifest_path is None:
            manifest_path = check_manifest_exists(csv_name, args.manifest_dir)
        validate_manifest(csv_name, csv_type, row_count, manifest_path, all_issues, head_sha)

    # Check ablation CSVs (subdirectory) — different schema, manifest_ref in rows
    ablation_dir = os.path.join(args.csv_dir, "ablation")
    check_ablation_manifests(ablation_dir, all_issues)

    # Report
    errors = [i for i in all_issues if i.severity == "ERROR"]
    warnings = [i for i in all_issues if i.severity == "WARNING"]
    notes = [i for i in all_issues if i.severity == "NOTE"]

    if not args.quiet:
        for issue in all_issues:
            print(issue)

        print()
        print(f"{len(csv_files)} CSV(s) checked, {len(errors) + len(warnings)} issue(s) found")
        if errors:
            print(f"  {len(errors)} error(s)")
        if warnings:
            print(f"  {len(warnings)} warning(s)")
        if notes:
            print(f"  {len(notes)} note(s) (informational, not counted as issues)")

    # Exit code: 0 = clean, 1 = warnings, 2 = errors
    if errors:
        return 2
    if warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
