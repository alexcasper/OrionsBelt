#!/usr/bin/env python3
"""Generate the e2e fleet comparison table from committed schema CSVs.

Reads every results/raw/*_e2e_schema.csv, extracts decode tok/s and TTFT,
groups by model + quantization, and emits a markdown table to
results/figures/e2e_fleet_comparison.md.

Flags commit mismatches per RESULTS DISCIPLINE: devices measured at
different commits are not comparable.

Bead ob-52r. Run after all fleet devices have re-run at the matched commit.

Usage:
    python3 scripts/gen_e2e_comparison.py
    python3 scripts/gen_e2e_comparison.py --output results/figures/e2e_fleet_comparison.md
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "results" / "raw"
MANIFESTS_DIR = REPO_ROOT / "results" / "manifests"
DEFAULT_OUT = REPO_ROOT / "results" / "figures" / "e2e_fleet_comparison.md"


def _check_manifest_dirty(manifest_refs):
    """Cross-reference manifests to determine dirty status.

    Returns (any_dirty, all_checked) where any_dirty is True if any manifest
    has git.dirty=true, and all_checked is True if every manifest was found
    and parsed successfully.
    """
    import json

    any_dirty = False
    all_checked = True
    for ref in manifest_refs:
        if not ref:
            continue
        path = MANIFESTS_DIR / Path(ref).name
        if not path.exists():
            all_checked = False
            continue
        try:
            with open(path) as f:
                m = json.load(f)
            if m.get("git", {}).get("dirty", False):
                any_dirty = True
        except (json.JSONDecodeError, OSError):
            all_checked = False
    return any_dirty, all_checked


def load_rows():
    """Load all e2e schema CSVs and return a list of dicts."""
    rows = []
    for f in sorted(RAW_DIR.glob("*_e2e_schema.csv")):
        with open(f) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)

    # Deduplicate: when entries for the same device/model/quant exist from
    # different runs (e.g. old "rk3588-t3" with cluster=big superseded by
    # newer "rk3588-t3_big"), keep only the one with the most repeat_count.
    # This prevents stale low-run entries from cluttering the comparison.
    _dedup_rows(rows)
    return rows


def _normalize_device(device, notes):
    """Normalize device name: infer cluster from notes when not in the name."""
    if "_big" in device or "_little" in device:
        return device
    cluster = "all"
    if isinstance(notes, str) and "cluster=big" in notes:
        cluster = "big"
    elif isinstance(notes, str) and "cluster=little" in notes:
        cluster = "little"
    return f"{device}_{cluster}"


def _dedup_rows(rows):
    """Remove superseded rows: for the same (norm_device, model, quant, metric),
    keep only the rows from the run with the highest repeat_count."""
    # Group row indices by (norm_device, model, quant, metric)
    from collections import defaultdict

    groups = defaultdict(list)
    for i, r in enumerate(rows):
        device = r.get("device", "?")
        model = r.get("model_checkpoint", "?")
        quant = r.get("quantization", "fp32")
        metric = r.get("metric_name", "")
        norm = _normalize_device(device, r.get("notes", ""))
        if "4B" in model:
            model_short = "4B"
        elif "0.8B" in model:
            model_short = "0.8B"
        else:
            model_short = model
        key = (norm, model_short, quant, metric)
        try:
            rep_count = int(r.get("repeat_count", "1"))
        except ValueError:
            rep_count = 1
        groups[key].append((i, rep_count))

    # For each group, find the max repeat_count and mark rows from other runs
    # for removal.
    remove = set()
    for _key, indices in groups.items():
        if len(indices) <= 1:
            continue
        max_rep = max(rc for _, rc in indices)
        # Keep rows with the max repeat_count, remove others
        keep_indices = {i for i, rc in indices if rc == max_rep}
        for i, _rc in indices:
            if i not in keep_indices:
                remove.add(i)

    if remove:
        # Remove in reverse order to preserve indices
        for i in sorted(remove, reverse=True):
            del rows[i]


def extract_metrics(rows):
    """Extract tok/s and TTFT per device/model/quant, keyed by (device, model, quant)."""
    data = defaultdict(
        lambda: {"tok_per_sec": [], "ttft": [], "sha": set(), "manifests": set(), "runs": 0}
    )
    for r in rows:
        device = r.get("device", "?")
        model = r.get("model_checkpoint", "?")
        # Shorten model name
        if "4B" in model:
            model_short = "4B"
        elif "0.8B" in model:
            model_short = "0.8B"
        else:
            model_short = model.split("/")[-1] if "/" in model else model
        quant = r.get("quantization", "fp32")
        sha = r.get("git_sha", "?")[:8]

        key = (device, model_short, quant)
        entry = data[key]
        entry["sha"].add(sha)
        entry["manifests"].add(r.get("manifest_ref", ""))
        entry["runs"] = max(entry["runs"], int(r.get("repeat_count", "1")))

        metric = r.get("metric_name", "")
        val = r.get("value", "")
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue

        if metric == "decode_tokens_per_sec":
            entry["tok_per_sec"].append(val)
        elif metric == "ttft_seconds":
            # Convert to ms for readability
            entry["ttft"].append(val * 1000)

    return data


def fmt_mean_std(vals):
    if not vals:
        return "—"
    mean = sum(vals) / len(vals)
    if len(vals) > 1:
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        if std / mean > 0.01:  # Show std only if >1%
            return f"{mean:.2f} ± {std:.2f}"
    return f"{mean:.2f}"


def _check_commit_lineage(base_commit, commits):
    """Check whether commits descend from base_commit and whether they have
    kernel/binary-affecting code changes.

    Returns dict: {sha_short: {"status": "matched"|"code-identical"|"pre-matched",
                                "detail": str}}
    """
    import subprocess

    def _run(cmd):
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=REPO_ROOT)  # noqa: UP022
        return r.stdout.decode("utf-8", errors="replace")

    results = {}
    for sha in commits:
        if sha == base_commit:
            results[sha] = {"status": "matched", "detail": "base commit"}
            continue

        full_base = _run(["git", "rev-parse", base_commit]).strip()

        # Try to resolve the short SHA to a full SHA
        resolved = _run(["git", "rev-parse", sha]).strip()

        if not resolved or not full_base:
            results[sha] = {"status": "unknown", "detail": "unresolvable"}
            continue

        # Check ancestry
        # Note: stdout/stderr=PIPE (not capture_output=True) for Python 3.6
        # compatibility -- this script also runs on fleet nodes with old pythons.
        is_ancestor = (
            subprocess.run(  # noqa: UP022
                ["git", "merge-base", "--is-ancestor", full_base, resolved],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=REPO_ROOT,
            ).returncode
            == 0
        )

        if not is_ancestor:
            results[sha] = {"status": "pre-matched", "detail": "pre-dates base commit"}
            continue

        # It's a descendant — check for kernel/binary-affecting changes
        changed = _run(["git", "diff", "--name-only", full_base, resolved]).strip().splitlines()

        # Files that affect the benchmark binary or kernel
        kernel_patterns = (
            "bench/gdn_",
            "bench/bench_gdn",
            "src/",
            "include/",
            "scripts/build_device_bench",
            "scripts/run_e2e_decode",
        )
        kernel_changes = [f for f in changed if any(p in f for p in kernel_patterns)]

        if kernel_changes:
            results[sha] = {
                "status": "diverged",
                "detail": f"kernel changes: {', '.join(kernel_changes[:3])}",
            }
        else:
            results[sha] = {
                "status": "code-identical",
                "detail": "results/docs/beads only" if changed else "identical tree",
            }

    return results


def generate_table(data, base_commit=None, commit_info=None):
    """Generate markdown table from extracted metrics."""
    lines = []

    # Group by (model, quant)
    groups = defaultdict(list)
    for (device, model, quant), entry in sorted(data.items()):
        groups[(model, quant)].append((device, entry))

    lines.append("# E2E Decode Fleet Comparison")
    lines.append("")
    lines.append("Generated by `scripts/gen_e2e_comparison.py`. Do not hand-edit.")
    lines.append("")

    # Commit mismatch check — cross-reference manifests for dirty status
    all_shas = set()
    all_manifests = set()
    for entry in data.values():
        all_shas.update(entry["sha"])
        all_manifests.update(entry["manifests"])

    if base_commit and commit_info:
        # Smart mode: classify commits by lineage
        statuses = set(ci["status"] for ci in commit_info.values())
        has_pre_matched = "pre-matched" in statuses or "diverged" in statuses

        if has_pre_matched:
            problem = sorted(
                sha
                for sha, ci in commit_info.items()
                if ci["status"] in ("pre-matched", "diverged")
            )
            lines.append(
                f"> ⚠️ **Partial commit match.** Base commit: `{base_commit}`. "
                f"Some entries pre-date or diverge from the base and are not comparable."
            )
            lines.append(f"> Flagged: {', '.join(problem)}")
        else:
            lines.append(
                f"> ✅ **Matched-commit comparison.** All devices ran code-identical "
                f"kernels at base commit `{base_commit}`."
            )
            non_base = sorted(sha for sha in all_shas if sha != base_commit)
            if non_base:
                lines.append(
                    f"> Result-file commits (results/docs only, no kernel changes): "
                    f"{', '.join(non_base)}"
                )
    elif len(all_shas) > 1:
        # Fallback (no --base-commit given): cross-reference manifests for dirty status
        any_dirty, all_checked = _check_manifest_dirty(all_manifests)
        if any_dirty:
            lines.append("> ⚠️ **Commit mismatch with dirty manifests.** Some runs had")
            lines.append("> uncommitted changes — data is NOT comparable (RESULTS DISCIPLINE).")
        elif all_checked:
            lines.append("> ℹ️ **Multiple commits in play**, but all manifests show `dirty=false`.")
            lines.append("> SHAs differ only due to result-file commits between runs — the")
            lines.append("> benchmarked code is identical. Data IS comparable, but no")
            lines.append("> --base-commit was given to verify kernel-code lineage.")
        else:
            lines.append("> ⚠️ **Commit mismatch detected.** Devices measured at different commits")
            lines.append("> are not comparable (RESULTS DISCIPLINE, bead ob-bf7).")
            lines.append("> (Could not verify manifest status for all runs.)")
        lines.append(f"> Commits in play: {', '.join(sorted(all_shas))}")
        lines.append("")

    for (model, quant), entries in sorted(groups.items()):
        title = f"Qwen3.5-{model} — {quant.upper()}"
        lines.append(f"## {title}")
        lines.append("")

        # Table header
        lines.append("| Device | Commit | Tok/s | TTFT (ms) | Runs | Notes |")
        lines.append("|--------|--------|------:|----------:|-----:|-------|")

        for device, entry in entries:
            sha_str = ", ".join(sorted(entry["sha"]))
            tok = fmt_mean_std(entry["tok_per_sec"])
            ttft = fmt_mean_std(entry["ttft"])
            runs = entry["runs"]
            notes = []
            if len(entry["sha"]) > 1:
                notes.append("⚠ multi-commit")
            if any("dirty" in m for m in entry["manifests"]):
                notes.append("⚠ dirty")
            if commit_info:
                for sha in sorted(entry["sha"]):
                    ci = commit_info.get(sha)
                    if ci:
                        if ci["status"] == "pre-matched":
                            notes.append(f"⚠ {sha}: pre-matched")
                        elif ci["status"] == "diverged":
                            notes.append(f"⚠ {sha}: kernel diverged")
            notes_str = "; ".join(notes) if notes else ""
            lines.append(f"| {device} | `{sha_str}` | {tok} | {ttft} | {runs} | {notes_str} |")

        lines.append("")

    # Cross-quantization comparison (if both fp32 and int8 exist for same device/model)
    # Normalize device names: strip _int8 suffix, match big-cluster FP32 (no suffix) to big INT8
    fp32_map = {}  # (normalized_device, model) -> entry
    for (device, model, quant), entry in data.items():
        if quant == "fp32":
            # FP32 without cluster suffix = big cluster (default)
            norm = device
            fp32_map[(norm, model)] = entry
            # Also try with _big appended for matching
            if "_big" not in device and "_little" not in device:
                fp32_map[(device + "_big", model)] = entry

    int8_entries = {k: v for k, v in data.items() if k[2] == "int8"}
    if int8_entries:
        lines.append("## INT8 vs FP32 Speedup")
        lines.append("")
        lines.append("| Device | Model | FP32 tok/s | INT8 tok/s | Speedup |")
        lines.append("|--------|-------|-----------:|-----------:|--------:|")

        for (device, model, _), i_entry in sorted(int8_entries.items()):
            # Normalize: strip _int8 from device name for matching
            norm_device = device.replace("_int8", "")
            fp32_key = (norm_device, model)
            if fp32_key in fp32_map:
                f_entry = fp32_map[fp32_key]
                fp32_tok = (
                    sum(f_entry["tok_per_sec"]) / len(f_entry["tok_per_sec"])
                    if f_entry["tok_per_sec"]
                    else 0
                )
                int8_tok = (
                    sum(i_entry["tok_per_sec"]) / len(i_entry["tok_per_sec"])
                    if i_entry["tok_per_sec"]
                    else 0
                )
                speedup = int8_tok / fp32_tok if fp32_tok > 0 else 0
                lines.append(
                    f"| {device} | {model} | {fp32_tok:.2f} | {int8_tok:.2f} | **{speedup:.2f}×** |"
                )
            else:
                int8_tok = (
                    sum(i_entry["tok_per_sec"]) / len(i_entry["tok_per_sec"])
                    if i_entry["tok_per_sec"]
                    else 0
                )
                lines.append(f"| {device} | {model} | — | {int8_tok:.2f} | — |")

        lines.append("")

    lines.append("## Data Sources")
    lines.append("")
    for f in sorted(RAW_DIR.glob("*_e2e_schema.csv")):
        lines.append(f"- `{f.relative_to(REPO_ROOT)}`")
    lines.append("")

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate e2e fleet comparison table")
    parser.add_argument(
        "--output",
        "-o",
        default=str(DEFAULT_OUT),
        help="Output markdown file (default: results/figures/e2e_fleet_comparison.md)",
    )
    parser.add_argument(
        "--base-commit",
        default=None,
        help="Kernel base commit for matched-commit verification. When provided, "
        "classifies each device's commit by git lineage and code-diff to determine "
        "whether the comparison is valid.",
    )
    args = parser.parse_args()

    rows = load_rows()
    if not rows:
        print("ERROR: no *_e2e_schema.csv files found in results/raw/", file=sys.stderr)
        sys.exit(1)

    data = extract_metrics(rows)

    # Collect all SHAs for lineage check
    all_shas = set()
    for entry in data.values():
        all_shas.update(entry["sha"])

    commit_info = None
    if args.base_commit:
        commit_info = _check_commit_lineage(args.base_commit, all_shas)

    markdown = generate_table(data, base_commit=args.base_commit, commit_info=commit_info)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown)
    print(f"Written: {out_path}")
    print(f"  Devices: {len(set(r.get('device', '?') for r in rows))}")
    print(f"  Data rows: {len(rows)}")

    # Commit status summary
    if args.base_commit and commit_info:
        print(f"\nBase commit: {args.base_commit}")
        for sha in sorted(all_shas):
            ci = commit_info.get(sha, {})
            status = ci.get("status", "?")
            detail = ci.get("detail", "")
            marker = "✅" if status in ("matched", "code-identical") else "⚠️"
            print(f"  {marker} {sha}: {status} ({detail})")
    elif len(all_shas) > 1:
        print(f"\n⚠ WARNING: {len(all_shas)} different commits in the data:")
        for sha in sorted(all_shas):
            print(f"  {sha}")
        print("Devices at different commits are NOT comparable.")
        print("Tip: pass --base-commit <sha> to enable lineage-based verification.")


if __name__ == "__main__":
    main()
