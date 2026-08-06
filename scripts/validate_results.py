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
    """Return 'standard', 'sustained', 'power', 'layer_profile', 'e2e_sweep', or None."""
    cols = set(header)
    if cols >= set(STANDARD_COLS):
        return "standard"
    if cols >= set(SUSTAINED_COLS):
        return "sustained"
    if cols >= set(POWER_COLS):
        return "power"
    if cols >= {"layer_idx", "layer_type", "p50_us", "mean_us"}:
        return "layer_profile"
    if cols >= {"run_id", "metric_name", "metric_component", "repeat_index"}:
        return "e2e_sweep"
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
    if csv_type == "e2e_sweep":
        return E2E_SWEEP_COLS
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


def validate_e2e_sweep_row(row, csv_name, issues, row_num):
    """Validate a single row of an e2e context-sweep CSV."""
    try:
        value = float(row["value"])
        ctx = int(row["context_length"])
        repeat_idx = int(row["repeat_index"])
        repeat_count = int(row["repeat_count"])
    except (ValueError, KeyError) as e:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: cannot parse: {e}"))
        return

    if ctx <= 0:
        issues.append(Issue("ERROR", csv_name, f"row {row_num}: non-positive context_length"))
    if repeat_idx < 0 or repeat_idx >= repeat_count:
        issues.append(
            Issue("WARNING", csv_name, f"row {row_num}: repeat_index {repeat_idx} out of range [0,{repeat_count})")
        )
    # Sanity-check throughput-like metrics
    unit = row.get("unit", "")
    metric = row.get("metric_name", "")
    if "per_sec" in unit or "per_sec" in metric:
        if value <= 0:
            issues.append(Issue("WARNING", csv_name, f"row {row_num}: non-positive {metric} = {value}"))


def validate_csv(path, csv_name, issues):
    """Validate CSV schema and row-level data. Return (csv_type, row_count)."""
    if not os.path.isfile(path):
        issues.append(Issue("ERROR", csv_name, "file not found"))
        return None, 0

    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            if not header:
                issues.append(Issue("ERROR", csv_name, "empty or unreadable CSV"))
                return None, 0

            csv_type = detect_csv_type(header)
            if csv_type is None:
                issues.append(
                    Issue("WARNING", csv_name, f"unrecognized CSV format ({len(header)} columns)")
                )
                return None, 0

            expected = expected_columns(csv_type)
            missing = [c for c in expected if c not in header]
            if missing:
                issues.append(
                    Issue(
                        "ERROR", csv_name, "missing required columns: {}".format(", ".join(missing))
                    )
                )

            row_count = 0
            for i, row in enumerate(reader, start=2):  # row 1 is header
                row_count += 1
                if csv_type == "standard":
                    validate_standard_row(row, csv_name, issues, i)
                elif csv_type == "sustained":
                    validate_sustained_row(row, csv_name, issues, i)
                elif csv_type == "layer_profile":
                    validate_layer_profile_row(row, csv_name, issues, i)
                elif csv_type == "e2e_sweep":
                    validate_e2e_sweep_row(row, csv_name, issues, i)

            return csv_type, row_count

    except Exception as e:
        issues.append(Issue("ERROR", csv_name, f"cannot read CSV: {e}"))
        return None, 0


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
    if spec_bw and csv_type == "standard" and row_count > 0:
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
        csv_type, row_count = validate_csv(csv_path, csv_name, all_issues)

        manifest_path = check_manifest_exists(csv_name, args.manifest_dir)
        validate_manifest(csv_name, csv_type, row_count, manifest_path, all_issues, head_sha)

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
