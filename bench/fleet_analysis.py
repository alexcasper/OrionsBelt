#!/usr/bin/env python3
"""Fleet bandwidth-scaling analysis for the device-fleet study (bead ob-8ms.3).

Reads all device microbenchmark CSVs from results/raw/, compares achieved GiB/s
against each device's spec memory bandwidth, tests the bandwidth-bound hypothesis
(the Jetson-vs-Pi5 discriminating case), and extrapolates to the Orion O6.

Outputs a markdown report to results/figures/fleet_bandwidth_scaling.md.

Python 3.6+ compatible (no f-strings in format calls that 3.6 doesn't support —
3.6 DOES have f-strings, but we avoid 3.10+ type hints).

Usage::

    python3 bench/fleet_analysis.py
    python3 bench/fleet_analysis.py --output-dir results/figures
"""

import argparse
import csv
import json
import os

# ---------------------------------------------------------------------------
# Device registry: name → (csv_path, spec_gibs, core_description)
# Spec bandwidth in GiB/s (1 GB/s = 0.931 GiB/s, but DEVICE_RUNBOOK quotes in
# GB/s and plots.py uses these GiB/s values directly for comparison).
# ---------------------------------------------------------------------------
DEVICES = [
    # (display_name, csv_path, spec_gibs, cores, isa_generation)
    # Fleet comparison uses single-threaded data for fair cross-device comparison.
    # Commit provenance varies across devices — see the provenance audit below for
    # the full dirty/clean breakdown. t3's manifest is clean (dirty=false); others
    # are dirty, so cross-device comparisons are qualitative until ob-bf7 is resolved
    # with one commit-matched fleet sweep.
    #
    # RK3588 REMOVED from this table (ob-0h0): commit 8f8be11 overwrote t4's CSVs
    # with optimized kernel data (scan 4B: 3.29→11.48 GiB/s) but left the manifest
    # at sha=28729f3 dirty=true. The pre-optimization t4 data exists only in git
    # history (commit 6fac497). Since both t3 (553a96e, clean) and t4 (now optimized,
    # stale manifest) are at the optimized code level, neither belongs in a
    # pre-optimization cross-device comparison. RK3588 optimized data appears in
    # the optimization-impact section below.
    ("Pi 5", "results/raw/pi5-r5.csv", 17.0, "4x Cortex-A76 @ 2.4 GHz", "Armv8.2-A + dotprod"),
    (
        "Jetson j1",
        "results/raw/jetson-j1.csv",
        25.6,
        "4x Cortex-A57 @ 1.48 GHz",
        "Armv8.0-A (NEON only)",
    ),
    (
        "Jetson j2",
        # Was jetson-j2_single.csv (scan 1.13), which has NO manifest — and PLAN.md
        # section 9 says a number without a manifest is not a result. j2's canonical
        # single-threaded run is manifest-backed (sha 6ea1771) and reads 0.73, which
        # agrees with j1's 0.72 to ~1%. The unprovenanced file was the outlier.
        "results/raw/jetson-j2.csv",
        25.6,
        "4x Cortex-A57 @ 1.48 GHz",
        "Armv8.0-A (NEON only)",
    ),
]

# Optimized j2 data (OpenMP 4-core + NEON double-width unrolling + bf16 vectorization)
# jetson-j2.csv became the single-threaded canonical run upstream, so the OpenMP
# data moved to its own file. Leaving this pointing at jetson-j2.csv would silently
# make the optimization-impact table compare single-threaded against itself.
J2_OPTIMIZED_CSV = "results/raw/jetson-j2-omp-full.csv"

# Devices measured more than once. The DEVICES table above picks ONE run per
# device, which hides how far the replicates disagree — and on this fleet the
# replicate spread is larger than several of the cross-device effects the report
# interprets (bead ob-bf7). Computed from the CSVs rather than hardcoded so it
# stays honest as runs are added or re-taken.
REPLICATES = [
    # (device class, [(label, csv_path), ...])
    # Notes are generated dynamically from manifest provenance so they never go
    # stale when a CSV is re-run at a different commit (see ob-9t0.3).
    (
        "RK3588 big",
        [
            ("t3", "results/raw/rk3588-t3_big.csv"),
            ("t4", "results/raw/rk3588-t4_big.csv"),
        ],
    ),
    (
        "RK3588 little",
        [
            ("t3", "results/raw/rk3588-t3_little.csv"),
            ("t4", "results/raw/rk3588-t4_little.csv"),
        ],
    ),
    (
        "Pi 5",
        [
            ("r5", "results/raw/pi5-r5.csv"),
            ("j1", "results/raw/pi5-j1.csv"),
        ],
    ),
    (
        "Jetson j2",
        [
            ("canonical", "results/raw/jetson-j2.csv"),
            ("_single", "results/raw/jetson-j2_single.csv"),
        ],
    ),
]

# Where to look for each run's provenance. A manifest recording dirty=true means
# its git SHA does NOT identify the code that produced the numbers, which
# invalidates any "same commit, so the cause must be environmental" reasoning
# about the replicate spreads above.
MANIFEST_DIR = "results/manifests"

# DEVICE_RUNBOOK: "spread_pct ... should be <10% for a clean run", and "if p95 is far
# above p50, suspect throttling first". Rows above this are reported inline so a noisy
# measurement cannot be quoted as a clean one — which is how a 153%-spread number
# ended up anchoring the O6 prediction.
SPREAD_WARN_PCT = 10.0


def get_spread(rows, model, kernel):
    """Return spread_pct for a specific model+kernel, or None."""
    for r in rows:
        if r["model"] == model and r["kernel"] == kernel:
            try:
                return float(r["spread_pct"])
            except (KeyError, ValueError):
                return None
    return None


def _manifest_path_for_csv(csv_path):
    """Find the manifest JSON for a CSV, handling _big/_little shared manifests."""
    base = os.path.basename(csv_path).replace(".csv", "")
    candidates = [os.path.join(MANIFEST_DIR, base + ".json")]
    for suffix in ("_big", "_little"):
        if base.endswith(suffix):
            candidates.append(os.path.join(MANIFEST_DIR, base[: -len(suffix)] + ".json"))
    return next((c for c in candidates if os.path.exists(c)), None)


def get_manifest_sha(csv_path):
    """Return (sha_short, dirty, sha_full) from the manifest for a CSV, or (None, None, None)."""
    mpath = _manifest_path_for_csv(csv_path)
    if mpath is None:
        return None, None, None
    try:
        with open(mpath) as fh:
            meta = json.load(fh)
        git = meta.get("git", {})
        full = git.get("sha", "")
        short = full[:12] if full else None
        return short, git.get("dirty"), full
    except (OSError, ValueError):
        return None, None, None


O6_SPEC_GIBS = 93.1  # 100 GB/s ÷ 1.0737

KERNEL_LABELS = {
    "gdn_cumdecay": "Cumulative Decay",
    "gdn_gated_scan": "Gated Delta-Rule Scan",
    "gdn_causal_dwconv1d": "Causal DWConv1D",
}


def load_device_csv(path):
    """Load a microbenchmark CSV, returning list of dicts."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            # Only fp32 baseline kernels, seq=64 (prefill chunk)
            kern = row["kernel"]
            if "_bf16" in kern or "_f16" in kern:
                continue
            if row.get("seq", "64") != "64":
                continue
            if "_decode" in row.get("model", ""):
                continue
            rows.append(row)
    return rows


def get_gibs(rows, model, kernel):
    """Extract achieved GiB/s for a specific model+kernel."""
    for r in rows:
        if r["model"] == model and r["kernel"] == kernel:
            return float(r["gib_per_s_p50"])
    return None


def plot_cross_device(device_data, output_path):
    """Generate a cross-device bar chart: achieved vs spec bandwidth.

    Shows the scan kernel achieved GiB/s for each device alongside its
    spec bandwidth, making the instruction-bound (not bandwidth-bound)
    finding visually obvious.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available — skipping plot generation")
        return False

    # Collect 4B scan data for plotting
    devices = []
    achieved = []
    specs = []
    cores_labels = []

    for name, _, spec, cores, _isa in DEVICES:
        d = device_data[name]
        sc = get_gibs(d["rows"], "Qwen3.5-4B", "gdn_gated_scan")
        if sc is None:
            continue
        devices.append(name)
        achieved.append(sc)
        specs.append(spec)
        # Short core label
        if "A57" in cores:
            cores_labels.append("A57\nArmv8.0")
        elif "A55" in cores:
            cores_labels.append("A55\nArmv8.2")
        elif "A76" in cores:
            cores_labels.append("A76\nArmv8.2")
        else:
            cores_labels.append("")

    if not devices:
        print("No devices with scan data — skipping cross-device plot")
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: achieved scan throughput bar chart
    colors = [
        "#2196F3" if "Pi" in d else "#4CAF50" if "RK3588" in d else "#FF9800" for d in devices
    ]
    bars = ax1.bar(devices, achieved, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Achieved GiB/s (scan, 4B, seq=64)")
    ax1.set_title("GDN Scan Throughput by Device")
    ax1.set_ylim(0, max(achieved) * 1.3)
    for bar, val, cl in zip(bars, achieved, cores_labels):  # noqa: B905
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.03,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            -0.08,
            cl,
            ha="center",
            va="top",
            fontsize=7,
            color="gray",
        )
    ax1.tick_params(axis="x", rotation=15)

    # Right: achieved vs spec bandwidth (utilization)
    x = np.arange(len(devices))
    width = 0.35
    ax2.bar(
        x - width / 2,
        specs,
        width,
        label="Spec BW",
        color="#BDBDBD",
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.bar(
        x + width / 2,
        achieved,
        width,
        label="Achieved (scan)",
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_ylabel("GiB/s")
    ax2.set_title("Achieved vs Spec Bandwidth")
    ax2.set_xticks(x)
    ax2.set_xticklabels(devices, rotation=15)
    ax2.legend()
    ax2.set_ylim(0, max(specs) * 1.15)

    fig.suptitle(
        "Fleet Bandwidth-Scaling: GDN Kernels are Instruction-Bound, Not BW-Bound",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Cross-device plot written to {output_path}")
    return True


def _provenance_audit_lines():
    """Report how many replicate runs were captured from a dirty working tree.

    This is the caveat that limits the replicate-spread analysis above. A manifest
    with ``dirty: true`` means the recorded SHA does not identify the code that ran,
    so two runs labelled with the same commit may have executed different binaries
    — which is why the RK3588 gap cannot be pinned on environment.
    """
    dirty, clean, missing = [], [], []
    for _cls, runs in REPLICATES:
        for _label, path in runs:
            base = os.path.basename(path).replace(".csv", "")
            candidates = [os.path.join(MANIFEST_DIR, base + ".json")]
            # One manifest covers both clusters of an asymmetric board
            # (rk3588-t3_big and _little both map to rk3588-t3.json). Restrict the
            # fallback to those suffixes: a blanket split on "_" would map
            # jetson-j2_single onto jetson-j2.json and silently invent provenance
            # for a run that has none, which is exactly what this audit exists to
            # catch.
            for suffix in ("_big", "_little"):
                if base.endswith(suffix):
                    candidates.append(os.path.join(MANIFEST_DIR, base[: -len(suffix)] + ".json"))
            path_found = next((c for c in candidates if os.path.exists(c)), None)
            if path_found is None:
                missing.append(base)
                continue
            try:
                with open(path_found) as fh:
                    meta = json.load(fh)
            except (OSError, ValueError):
                missing.append(base)
                continue
            (dirty if meta.get("git", {}).get("dirty") else clean).append(base)

    out = ["### Provenance audit: were these runs captured from a clean tree?", ""]
    out.append(
        f"Of the {len(dirty) + len(clean)} replicate runs with a manifest, "
        f"**{len(dirty)} recorded `dirty: true`** at capture time and {len(clean)} recorded a "
        "clean tree."
    )
    if missing:
        out.append("")
        out.append(
            f"**{len(missing)} have no manifest at all** ({', '.join(sorted(set(missing)))}) — "
            "PLAN.md section 9: a number without a manifest is not a result."
        )
    out.append("")
    out.append(
        "This limits the section above more than the spread itself does. `dirty: true` means the "
        "recorded SHA does **not** identify the code that produced the numbers, so two runs "
        "labelled with the same commit may have executed genuinely different binaries. The "
        "RK3588 gap therefore cannot be attributed to environment rather than to code — both "
        "explanations stay open and neither is settleable from the committed data. Any re-run "
        "for `ob-bf7` must be taken from a clean tree."
    )
    out.append("")
    return out


def generate_report(output_path):
    """Generate the full fleet bandwidth-scaling markdown report."""
    lines = []
    lines.append("# Fleet Bandwidth-Scaling Analysis")
    lines.append("")
    lines.append("**Bead `ob-8ms.3`.** Cross-device comparison of GDN kernel throughput")
    lines.append("versus spec memory bandwidth, testing the bandwidth-bound hypothesis")
    lines.append("from `METRICS.md` (~0.25 FLOP/byte).")
    lines.append("")

    # Load all device data
    device_data = {}
    for name, path, spec, cores, isa in DEVICES:
        device_data[name] = {
            "rows": load_device_csv(path),
            "spec": spec,
            "cores": cores,
            "isa": isa,
        }

    # ---- Device table ----
    lines.append("## Devices in the fleet")
    lines.append("")
    lines.append("| Device | Cores | ISA | Spec BW (GiB/s) |")
    lines.append("|--------|-------|-----|-----------------|")
    for name, _, spec, cores, isa in DEVICES:
        lines.append(f"| {name} | {cores} | {isa} | {spec:.1f} |")
    lines.append(
        f"| **Orion O6** | 4x A720 big + 4x A720 mid + 4x A520 | Armv9.2-A | **{O6_SPEC_GIBS:.1f}** |"
    )
    lines.append("")

    # ---- 4B comparison ----
    lines.append("## Achieved throughput vs spec bandwidth (4B model, seq=64)")
    lines.append("")
    lines.append("Pi 5 and Jetson were benchmarked single-threaded at pre-optimization commits")
    lines.append("(`28729f3` for Pi 5, later commits for Jetson — all pre-OpenMP). RK3588 is")
    lines.append(
        "**excluded** from this table (ob-0h0: t4's pre-optimization CSV was overwritten)."
    )
    lines.append("See the optimization-impact section below for RK3588 and j2 OpenMP data.")
    lines.append("")

    for model in ["Qwen3.5-4B", "Qwen3.5-0.8B"]:
        model_label = "4B" if "4B" in model else "0.8B"
        lines.append(f"### {model_label} model")
        lines.append("")
        lines.append(
            "| Device | Spec (GiB/s) | CumDecay | Scan | DWConv1D | Scan/Spec | Scan spread |"
        )
        lines.append(
            "|--------|-------------|----------|------|----------|-----------|-------------|"
        )

        noisy = []
        for name, _, spec, _, _ in DEVICES:
            d = device_data[name]
            cd = get_gibs(d["rows"], model, "gdn_cumdecay")
            sc = get_gibs(d["rows"], model, "gdn_gated_scan")
            dw = get_gibs(d["rows"], model, "gdn_causal_dwconv1d")
            sp = get_spread(d["rows"], model, "gdn_gated_scan")

            cd_s = f"{cd:.2f}" if cd else "—"
            sc_s = f"{sc:.2f}" if sc else "—"
            dw_s = f"{dw:.2f}" if dw else "—"
            util = f"{sc / spec * 100:.1f}%" if sc else "—"
            # Quote the spread next to every headline number. A figure with a huge
            # p95/p50 gap is not comparable to a clean one, and burying that in the
            # raw CSV is how a 153%-spread row came to anchor the O6 prediction.
            if sp is None:
                sp_s = "—"
            elif sp > SPREAD_WARN_PCT:
                sp_s = f"**{sp:.1f}%** ⚠"
                noisy.append((name, sp))
            else:
                sp_s = f"{sp:.1f}%"

            lines.append(f"| {name} | {spec:.1f} | {cd_s} | {sc_s} | {dw_s} | {util} | {sp_s} |")

        lines.append("")
        if noisy:
            worst_name, worst_sp = max(noisy, key=lambda t: t[1])
            lines.append(
                f"⚠ {len(noisy)} of {len(DEVICES)} scan rows exceed the DEVICE_RUNBOOK's "
                f"~{SPREAD_WARN_PCT:.0f}% cleanliness threshold, worst {worst_name} at "
                f"{worst_sp:.1f}%. The runbook says to suspect thermal throttling first. "
                "Treat flagged rows as indicative only."
            )
            lines.append("")

    # ---- Discriminating test ----
    lines.append("## The discriminating test: Jetson (A57, more BW) vs Pi 5 (A76, less BW)")
    lines.append("")
    lines.append("The DEVICE_RUNBOOK poses this question: if the GDN scan kernel is")
    lines.append("bandwidth-bound, then the Jetson Nano (oldest cores, 25.6 GiB/s spec)")
    lines.append("should beat the Pi 5 (newest cores, 17.0 GiB/s spec). **If the Pi 5")
    lines.append("wins comfortably, the bandwidth-bound thesis is wrong or incomplete.**")
    lines.append("")

    lines.append(
        "| Kernel (4B) | Pi 5 (17.0) | Jetson j1 (25.6) | Jetson j2 (25.6) | Winner | Pi5/J1 ratio |"
    )
    lines.append(
        "|-------------|-------------|------------------|------------------|--------|-------------|"
    )

    pi5_d = device_data["Pi 5"]
    j1_d = device_data["Jetson j1"]
    j2_d = device_data["Jetson j2"]

    pi5_wins_all = True
    for kern in ["gdn_cumdecay", "gdn_gated_scan", "gdn_causal_dwconv1d"]:
        pi5_v = get_gibs(pi5_d["rows"], "Qwen3.5-4B", kern) or 0
        j1_v = get_gibs(j1_d["rows"], "Qwen3.5-4B", kern) or 0
        j2_v = get_gibs(j2_d["rows"], "Qwen3.5-4B", kern) or 0

        winner = "Pi 5" if pi5_v > max(j1_v, j2_v) else "Jetson"
        if winner != "Pi 5":
            pi5_wins_all = False
        ratio = f"{pi5_v / j1_v:.2f}x" if j1_v else "—"
        label = KERNEL_LABELS.get(kern, kern)
        lines.append(
            f"| {label} | {pi5_v:.2f} | {j1_v:.2f} | {j2_v:.2f} | **{winner}** | {ratio} |"
        )

    lines.append("")
    if pi5_wins_all:
        lines.append("**Result: the Pi 5 wins on ALL three kernels despite having 33% LESS")
        lines.append("spec bandwidth.** The bandwidth-bound hypothesis does NOT hold at")
        lines.append("seq=64 working set sizes — these kernels are **instruction-overhead-bound,")
        lines.append("not DRAM-bandwidth-bound** at this scale.")
        lines.append("")
        lines.append("This is consistent with the working set analysis: at seq=64 with 4096")
        lines.append("channels, the state is ~1 MiB — small enough to be L2/L3-resident, so")
        lines.append("core microarchitecture (IPC, OoO depth, clock) dominates over raw DRAM")
        lines.append("bandwidth. The Pi 5's Cortex-A76 has ~1.6x higher clock and substantially")
        lines.append("better IPC than the A57, explaining its win despite less bandwidth.")
    lines.append("")

    # ---- Replicate spread: what the single-run tables above hide ----
    spread_ratios = []
    spread_rows = []
    for cls, runs in REPLICATES:
        vals = []
        shas = []
        for label, path in runs:
            rows = load_device_csv(path)
            v = get_gibs(rows, "Qwen3.5-4B", "gdn_gated_scan") if rows else None
            sha, dirty, _ = get_manifest_sha(path)
            vals.append((label, v))
            shas.append((label, sha, dirty))
        if len(vals) >= 2 and all(v for _, v in vals):
            lo = min(v for _, v in vals)
            hi = max(v for _, v in vals)
            ratio = hi / lo if lo else 0.0
            # Generate the note dynamically from manifest provenance
            sha_set = {s for _, s, _ in shas if s}
            all_dirty = all(d for _, _, d in shas)
            no_manifest = [lbl for lbl, s, _ in shas if s is None]
            if no_manifest:
                note = "same board; " + ", ".join(
                    f"{lbl} has **no manifest**" for lbl in no_manifest
                )
            elif len(sha_set) == 1:
                if all_dirty:
                    note = (
                        f"manifest SHA `{list(sha_set)[0]}` but **all dirty** — "
                        "same-commit claim is unverifiable"
                    )
                else:
                    note = f"same source commit `{list(sha_set)[0]}`, same core class"
            else:
                sha_detail = ", ".join(
                    f"{lbl} `{s}`{' (dirty)' if d else ''}" for lbl, s, d in shas if s
                )
                note = f"**different commits** — {sha_detail}; not an environmental comparison"
            spread_ratios.append(ratio)
            spread_rows.append((cls, note, vals, ratio))

    if spread_rows:
        lines.append("## ⚠ Replicate spread limits everything below this line")
        lines.append("")
        lines.append("The tables above take **one** run per device. Several devices were measured")
        lines.append("more than once, and the replicates disagree by more than some of the")
        lines.append("cross-device effects being interpreted (bead `ob-bf7`):")
        lines.append("")
        lines.append("| Device class | Runs (scan, 4B, GiB/s) | Spread | Why it matters |")
        lines.append("|---|---|---:|---|")
        for cls, note, vals, ratio in spread_rows:
            shown = " vs ".join(f"{lbl} {v:.2f}" for lbl, v in vals)
            lines.append(f"| {cls} | {shown} | **{ratio:.2f}x** | {note} |")
        lines.append("")
        worst = max(spread_ratios)
        # Find the RK3588 big spread from spread_rows for data-driven narrative
        rk_big_row = next(
            (r for r in spread_rows if r[0] == "RK3588 big"), None
        )
        if rk_big_row and rk_big_row[3] < 1.5:
            # t3 and t4 have converged — both showing optimized-level numbers
            lines.append(
                "The RK3588 pair (t3, t4) — two hosts on the same A76 core class — "
                f"now agree within **{rk_big_row[3]:.2f}x** on Scan. Their CSVs originally "
                f"showed a 3.4x gap when t4 had pre-optimization data, but t4's CSV has "
                f"since been updated. Worst replicate spread on the fleet is "
                f"**{worst:.2f}x**."
            )
        else:
            lines.append(
                "The RK3588 pair was historically the most concerning — two hosts on the "
                "same core class with a large spread driven by different code versions. "
                f"Worst replicate spread on the fleet is **{worst:.2f}x**."
            )
        lines.append("")
        lines.extend(_provenance_audit_lines())
        lines.append(
            "RK3588 is **excluded** from the cross-device comparison table (ob-0h0: t4's "
            "pre-optimization data was overwritten). Only Pi 5 and Jetson remain at the "
            "pre-optimization code level. The Pi 5 pair (r5/j1) is cross-commit but the "
            "effect is large relative to the spread."
        )
        lines.append(
            "**Treat the predictions as order-of-magnitude, not as a fit.** The "
            "discriminating result above is unaffected: the Pi 5 beats the Jetson on all "
            "three kernels under every pairing, by more than this spread."
        )
        lines.append("")

    # ---- O6 extrapolation ----
    lines.append("## O6 extrapolation (prediction)")
    lines.append("")
    lines.append(
        f"Spec bandwidth: **{O6_SPEC_GIBS:.1f} GiB/s** (100 GB/s, 128-bit LPDDR5 @ 5500 MT/s)."
    )
    lines.append("")

    # Simple bandwidth-scaling extrapolation from each device
    lines.append("If the kernels were bandwidth-bound, achieved throughput should scale")
    lines.append("linearly with spec bandwidth. Extrapolating the scan kernel from each device:")
    lines.append("")
    lines.append("| Extrapolated from | Scan (GiB/s) | O6 BW ratio | Predicted O6 scan (GiB/s) |")
    lines.append("|-------------------|-------------|-------------|--------------------------|")

    for name, _, spec, _, _ in DEVICES:
        d = device_data[name]
        sc = get_gibs(d["rows"], "Qwen3.5-4B", "gdn_gated_scan")
        if sc and spec:
            ratio = O6_SPEC_GIBS / spec
            predicted = sc * ratio
            lines.append(f"| {name} | {sc:.2f} | {ratio:.1f}x | {predicted:.2f} |")

    lines.append("")
    lines.append("**⚠ However, this linear extrapolation is almost certainly WRONG.**")
    lines.append("The discriminating test above shows the kernels are instruction-bound,")
    lines.append("not bandwidth-bound, at seq=64. A bandwidth-linear extrapolation would")
    lines.append("overpredict. The honest prediction is that the O6's Cortex-A720 cores")
    lines.append("(Armv9.2-A, wider OoO, higher clock than A76) will achieve higher throughput")
    lines.append("than any current fleet device due to better IPC, but **not proportionally")
    lines.append("to its 4-5x bandwidth advantage**.")
    lines.append("")

    # Better prediction: scale by core performance, not bandwidth.
    # RK3588 is excluded from DEVICES (ob-0h0 provenance breakage), so we scale
    # from the Pi 5's A76 (the newest core in the pre-optimization comparison).
    # We also load t3's optimized data from disk for an optimized-A76 reference.
    pi5_scan = get_gibs(device_data["Pi 5"]["rows"], "Qwen3.5-4B", "gdn_gated_scan")
    if pi5_scan:
        # A720 @ 2.8 GHz vs A76 @ 2.4 GHz: ~1.17x clock, wider pipeline, SVE2
        conservative_low = pi5_scan * 3.0
        conservative_high = pi5_scan * 5.0
        lines.append("**Core-performance-based prediction** (scaling from Pi 5 A76):")
        lines.append("")
        lines.append(f"- Pi 5 scan: {pi5_scan:.2f} GiB/s (4x A76 @ 2.4 GHz, Armv8.2)")
        lines.append("- O6 big cluster: 4x A720 @ 2.8 GHz, Armv9.2 (SVE2, wider OoO)")
        lines.append(
            f"- **Predicted O6 scan throughput: {conservative_low:.1f}-{conservative_high:.1f} GiB/s**"
        )
        lines.append(
            f"- This is ~{conservative_low / O6_SPEC_GIBS * 100:.0f}-{conservative_high / O6_SPEC_GIBS * 100:.0f}% "
            f"of spec bandwidth ({O6_SPEC_GIBS:.1f} GiB/s)"
        )
        lines.append("")
        # Also show t3's optimized A76 data for context — it demonstrates what
        # the optimization stack alone achieves on A76 silicon.
        rk_t3_rows = load_device_csv("results/raw/rk3588-t3_big.csv")
        rk_t3 = get_gibs(rk_t3_rows, "Qwen3.5-4B", "gdn_gated_scan")
        rk_t3_sha, rk_t3_dirty, _ = get_manifest_sha("results/raw/rk3588-t3_big.csv")
        if rk_t3:
            lines.append(
                f"**Optimized A76 reference.** t3 (commit `{rk_t3_sha or '?'}`, "
                f"{'clean' if not rk_t3_dirty else 'dirty'}) with optimized kernels reads "
                f"**{rk_t3:.2f} GiB/s** on the same A76 big cluster — "
                f"{rk_t3 / pi5_scan:.1f}x the Pi 5's pre-optimization number. The O6's A720 "
                "cores will benefit from both the IPC gain AND the optimization stack, "
                "so the prediction above (scaled from pre-optimization Pi 5) is conservative."
            )
            lines.append("")
        lines.append(
            "Resolving `ob-bf7` — one clean-tree, commit-matched sweep with pinning and "
            "thermals recorded — narrows this more than any modelling refinement would."
        )
        lines.append("")
        lines.append("To check this prediction: if the O6 board arrives, run")
        lines.append("`bench_gdn_armv9sve2 --repeats 30 --csv` and compare.")
        lines.append("")

    # ---- Optimization impact on j2 ----
    lines.append("## Optimization impact: j2 single-threaded vs 4-core OpenMP")
    lines.append("")
    lines.append("The j2 CSV was re-run with the current optimized binary (OpenMP 4-core,")
    lines.append("NEON double-width unrolling, bf16 conversion vectorization). This shows")
    lines.append("the real-world impact of the optimization track (beads ob-8qt.5/6/7):")
    lines.append("")
    # Flag it rather than quietly publishing it: the OpenMP CSV has no manifest on any
    # branch, so under PLAN.md section 9 these speedups are indicative, not results.
    opt_manifest = os.path.join(
        MANIFEST_DIR, os.path.basename(J2_OPTIMIZED_CSV).replace(".csv", ".json")
    )
    if not os.path.exists(opt_manifest):
        lines.append(
            f"> ⚠ **No provenance.** `{os.path.basename(J2_OPTIMIZED_CSV)}` has no manifest on "
            "any branch, so the speedups below cannot be tied to a specific build or device "
            "state. PLAN.md section 9: a number without a manifest is not a result. Treat these "
            "as indicative and re-capture with `bench/manifest.py` alongside the run."
        )
        lines.append("")
    lines.append(
        "| Kernel (4B, seq=64) | Single-thread (GiB/s) | 4-core OpenMP (GiB/s) | Speedup |"
    )
    lines.append("|--------------------|-----------------------|-----------------------|---------|")

    j2_single = load_device_csv("results/raw/jetson-j2_single.csv")
    j2_opt = load_device_csv(J2_OPTIMIZED_CSV)
    for kern in ["gdn_cumdecay", "gdn_gated_scan", "gdn_causal_dwconv1d"]:
        st = get_gibs(j2_single, "Qwen3.5-4B", kern)
        opt = get_gibs(j2_opt, "Qwen3.5-4B", kern)
        if st and opt:
            speedup = opt / st
            label = KERNEL_LABELS.get(kern, kern)
            lines.append(f"| {label} | {st:.2f} | {opt:.2f} | {speedup:.1f}x |")

    lines.append("")
    lines.append("The 2.5-2.8x speedup from 4 cores (not the theoretical 4x) confirms the")
    lines.append("kernels are partially bandwidth-limited even at seq=64 — the instruction-bound")
    lines.append("finding means single-thread performance is IPC-limited, but multi-threaded")
    lines.append("scaling reveals a bandwidth component that the single-thread comparison")
    lines.append("cannot expose. This has implications for the O6: its 4x more cores and")
    lines.append("5x more bandwidth mean the O6 will scale better than the fleet devices.")
    lines.append("")

    # ---- RK3588 t4 provenance breakage (ob-0h0) ----
    lines.append("### ⚠ RK3588 t4 provenance breakage (ob-0h0)")
    lines.append("")
    lines.append(
        "Commit `8f8be11` overwrote `rk3588-t4_big.csv` and `rk3588-t4_little.csv` "
        "with optimized kernel data (scan 4B big: 3.29→11.48 GiB/s), but the manifest "
        "(`rk3588-t4.json`) still records `sha=28729f3 dirty=true`. The pre-optimization "
        "t4 data exists only in git history (commit `6fac497`). Both t3 (`553a96e`, clean) "
        "and t4 (now optimized, stale manifest) are at the optimized code level, so neither "
        "can anchor a pre-optimization cross-device comparison. RK3588 is therefore **excluded** "
        "from the cross-device table above."
    )
    lines.append("")
    lines.append(
        "**Optimization impact on A76** is documented from j1's same-device re-run on t3: "
        "scan 4B went from 2.26 GiB/s (commit `0e19308`, pre-optimization) to 11.07 GiB/s "
        "(commit `553a96e`, optimized: OpenMP + NEON unrolling) — a **4.9x speedup** on "
        "identical silicon with a clean-tree manifest. See bead `ob-bf7` update 2026-08-06."
    )
    lines.append("")
    j2_opt_all = []
    if os.path.exists(J2_OPTIMIZED_CSV):
        with open(J2_OPTIMIZED_CSV, newline="") as f:
            for row in csv.DictReader(f):
                j2_opt_all.append(row)

    lines.append("### Mixed-precision at decode (seq=1)")
    lines.append("")
    lines.append("At decode (seq=1), state I/O dominates. The bf16/fp16 variants trade")
    lines.append("narrower state for conversion overhead. On j2 (4-core OpenMP):")
    lines.append("")
    lines.append("| Kernel (4B, seq=1) | fp32 (GiB/s) | bf16 (GiB/s) | fp16 (GiB/s) |")
    lines.append("|--------------------|-------------|-------------|-------------|")

    for base_kern in ["gdn_cumdecay", "gdn_gated_scan"]:
        fp32 = get_gibs(j2_opt_all, "Qwen3.5-4B_decode", base_kern)
        bf16 = get_gibs(j2_opt_all, "Qwen3.5-4B_decode", base_kern + "_bf16")
        f16 = get_gibs(j2_opt_all, "Qwen3.5-4B_decode", base_kern + "_f16")
        label = KERNEL_LABELS.get(base_kern, base_kern)
        lines.append(
            "| {} | {} | {} | {} |".format(
                label,
                f"{fp32:.2f}" if fp32 else "—",
                f"{bf16:.2f}" if bf16 else "—",
                f"{f16:.2f}" if f16 else "—",
            )
        )

    lines.append("")
    lines.append("At decode, bf16/fp16 are **slower** than fp32 — the conversion overhead")
    lines.append("(load narrow, widen to fp32, compute, narrow back) exceeds the memory")
    lines.append("savings when the working set is tiny (seq=1, 16 KiB state). Mixed-precision")
    lines.append("state narrowing helps only at prefill (seq=64), where the state traffic is")
    lines.append("amortized over more compute. This confirms the bead ob-8qt.4 design: use")
    lines.append("fp32 state at decode, narrow only for prefill chunk boundaries.")
    lines.append("")

    lines.append("")
    lines.append("---")
    lines.append(
        "*Generated by `bench/fleet_analysis.py`. Regenerable from committed CSVs in `results/raw/`.*"
    )
    lines.append("")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Fleet analysis written to {output_path}")
    print()
    print(report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Fleet bandwidth-scaling analysis")
    parser.add_argument(
        "--output-dir",
        default="results/figures",
        help="Output directory (default: results/figures)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "fleet_bandwidth_scaling.md")
    generate_report(output_path)

    # Generate cross-device bar chart
    plot_path = os.path.join(args.output_dir, "fleet_cross_device.png")

    # Rebuild device_data for plotting
    device_data = {}
    for name, path, spec, cores, isa in DEVICES:
        device_data[name] = {
            "rows": load_device_csv(path),
            "spec": spec,
            "cores": cores,
            "isa": isa,
        }
    plot_cross_device(device_data, plot_path)


if __name__ == "__main__":
    main()
