#!/usr/bin/env python3
# ob-9t0.2: partial comparison table aggregation. stdlib only (csv, statistics, json).
# Reads committed CSVs + manifests, prints structured numbers with provenance.
# Does NOT import bench/ (this node is Py 3.6.9; bench/ needs 3.7+).
import csv
import json
import os
import statistics
from pathlib import Path

RAW = str(Path(__file__).resolve().parent.parent / "results" / "raw")
MAN = str(Path(__file__).resolve().parent.parent / "results" / "manifests")


def median(xs):
    return statistics.median(xs) if xs else None


def load_ablation():
    """tidy long -> p50 per (phase, metric, component) over the 5 repeats."""
    path = os.path.join(RAW, "ablation/ablation_cpu-only.csv")
    groups = {}
    run_ids = set()
    shas = set()
    manifest_refs = set()
    ctxs = set()
    quants = set()
    engines = set()
    model = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["phase"], r["metric_name"], r.get("metric_component") or "")
            groups.setdefault(key, []).append(float(r["value"]))
            run_ids.add(r["run_id"])
            shas.add(r["git_sha"])
            manifest_refs.add(r["manifest_ref"])
            ctxs.add(r["context_length"])
            quants.add(r["quantization"])
            engines.add((r["engine_gdn"], r["engine_full_attention"]))
            model.add(r["model_checkpoint"])
    return {
        "groups": {k: median(v) for k, v in groups.items()},
        "n": {k: len(v) for k, v in groups.items()},
        "run_id": sorted(run_ids),
        "git_sha": sorted(shas),
        "manifest_ref": sorted(manifest_refs),
        "context": sorted(ctxs),
        "quant": sorted(quants),
        "engines": sorted(engines),
        "model": sorted(model),
    }


def load_kernel(fname):
    """kernel micro-bench CSVs already store p50 over 30 repeats; read directly."""
    rows = []
    with open(os.path.join(RAW, fname)) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def kv(rows, model, kernel, seq):
    for r in rows:
        if r["model"] == model and r["kernel"] == kernel and int(r["seq"]) == seq:
            return float(r["gib_per_s_p50"]), float(r["p50_us"]), float(r["spread_pct"])
    return None


def load_sustained(fname):
    path = os.path.join(RAW, fname)
    if not os.path.isfile(path):
        return None
    gibs = []
    with open(path) as f:
        for r in csv.DictReader(f):
            gibs.append(float(r["throughput_gibs"]))
    if not gibs:
        return None
    return gibs[0], gibs[-1], median(gibs), len(gibs)


def fmt_mem(b):
    x = b / 1024.0
    for u in ("KiB", "MiB", "GiB"):
        if abs(x) < 1024:
            return f"{x:.1f} {u}"
        x /= 1024
    return f"{x:.1f} TiB"


def gib(fname, kernel):
    r = load_kernel(fname)
    v = kv(r, "Qwen3.5-4B", kernel, 64)
    return v[0] if v else None


# ---- provenance map (run_id, git_sha, dirty) from manifests ----
PROV = {
    "jetson-j1_clean.csv": ("j1_20260804T102230Z_ba7506d", "ba7506d", False),
    "jetson-j1.csv": ("j1_20260802T235238Z_2c9ac9f", "2c9ac9f", True),
    "jetson-j1_single.csv": ("j1_20260806T110812Z_829a9c3", "829a9c3", True),
    "jetson-j1_omp.csv": ("j1_20260806T110807Z_829a9c3", "829a9c3", True),
    "jetson-j2.csv": ("j2_20260803T144316Z_6ea1771", "6ea1771", True),
    "jetson-j2_single.csv": ("jetson-j2", "6ea1771", True),
    "jetson-j2-omp.csv": ("jetson-j2-omp_reconstructed", "152808b", True),
    "jetson-j2-omp-full.csv": ("jetson-j2-omp-full_reconstructed", "a085417", True),
    "jetson-j2-full-optimized.csv": ("jetson-j2-full-optimized", "8c9b3a9", True),
    "jetson-j2-conv-unroll.csv": ("jetson-j2-conv-unroll_reconstructed", "152808b", True),
    "jetson-j2-omp-unroll.csv": ("jetson-j2-omp-unroll_reconstructed", "152808b", True),
    "pi5-r5.csv": ("r5_20260802T201237Z_28729f3", "28729f3", True),
    "pi5-j1.csv": ("r5_20260803T083154Z_f127a11", "f127a11", True),
    "rk3588-t3_big.csv": ("t3_20260806T053612Z_553a96e", "553a96e", False),
    "rk3588-t3_little.csv": ("t3_20260806T053612Z_553a96e", "553a96e", False),
    "rk3588-t4_big.csv": ("t4_20260802T211249Z_28729f3", "28729f3", True),
    "rk3588-t4_little.csv": ("t4_20260802T211249Z_28729f3", "28729f3", True),
    # Fleet sweep (ob-bf7): commit-matched, clean-tree, single-threaded
    "jetson-j1-clean.csv": ("jetson-j1-clean_sweep_234807d", "234807d", False),
    "jetson-j2-clean.csv": ("jetson-j2-clean_sweep_234807d", "234807d", False),
    "rk3588-t3-clean.csv": ("rk3588-t3-clean_sweep_234807d", "234807d", False),
    "rk3588-t3-little-clean.csv": ("rk3588-t3-little-clean_sweep_234807d", "234807d", False),
    "rk3588-t4-clean.csv": ("rk3588-t4-clean_sweep_234807d", "234807d", False),
    "rk3588-t4-little-clean.csv": ("rk3588-t4-little-clean_sweep_234807d", "234807d", False),
}


def main() -> int:
    print("=" * 90)
    print("ABLATION (model-level, tidy long, cpu/cpu hybrid, 4K)")
    ab = load_ablation()
    print("  run_id    :", ab["run_id"])
    _abl_man_exists = all(os.path.isfile(m) for m in ab["manifest_ref"])
    print(
        "  git_sha   :",
        ab["git_sha"],
        " (manifest file exists)"
        if _abl_man_exists
        else " (manifest file MISSING from results/manifests/)",
    )
    print("  manifest  :", ab["manifest_ref"])
    print("  context   :", ab["context"], " quant:", ab["quant"], " engines:", ab["engines"])
    print("  model     :", ab["model"])
    print("  --- p50 over repeats (n) ---")
    g = ab["groups"]

    print(
        f"  prefill_tokens_per_sec : {g[('prefill', 'prefill_tokens_per_sec', '')]:<14.1f} "
        f"(n={ab['n'][('prefill', 'prefill_tokens_per_sec', '')]})"
    )
    print(
        f"  decode_tokens_per_sec  : {g[('decode', 'decode_tokens_per_sec', '')]:<14.1f} "
        f"(n={ab['n'][('decode', 'decode_tokens_per_sec', '')]})"
    )
    ttft = g[("prefill", "ttft_seconds", "")]
    ttft_n = ab["n"][("prefill", "ttft_seconds", "")]
    print(f"  ttft_seconds           : {ttft:<14.6f} s = {ttft * 1000:.4f} ms (n={ttft_n})")
    for comp in ("weights", "kv_cache", "recurrent_state"):
        for ph in ("prefill", "decode"):
            k = (ph, "peak_memory_bytes", comp)
            if k in g:
                print(
                    f"  mem[{comp}][{ph}] : {fmt_mem(g[k]):<12} ({g[k]:.0f} bytes, n={ab['n'][k]})"
                )

    print("\n" + "=" * 90)
    print("KERNEL FLEET BASELINE (4B/0.8B, seq=64, single-core canonical, fp32) — GiB/s @ p50")
    print("Device            run_id_sha            dirty   CumDecay  Scan    DWConv1D  GDN2")
    fleet = [
        ("Jetson j1 (canon)", "jetson-j1.csv"),
        ("Jetson j2 (canon)", "jetson-j2.csv"),
        ("Pi5 r5", "pi5-r5.csv"),
        ("Pi5 j1", "pi5-j1.csv"),
        ("RK3588 t4 big", "rk3588-t4_big.csv"),
        ("RK3588 t3 big", "rk3588-t3_big.csv"),
        ("RK3588 t4 little", "rk3588-t4_little.csv"),
        ("RK3588 t3 little", "rk3588-t3_little.csv"),
    ]
    for label, fname in fleet:
        rows = load_kernel(fname)
        rid, sha, dirty = PROV[fname]
        cd = kv(rows, "Qwen3.5-4B", "gdn_cumdecay", 64)
        sc = kv(rows, "Qwen3.5-4B", "gdn_gated_scan", 64)
        cv = kv(rows, "Qwen3.5-4B", "gdn_causal_dwconv1d", 64)
        g2 = kv(rows, "Qwen3.5-4B", "gdn2_gated_scan", 64)
        ds = "dirty" if dirty else "CLEAN"
        cd_s = f"{cd[0]:<5.2f}" if cd else "  —  "
        sc_s = f"{sc[0]:<5.2f}" if sc else "  —  "
        cv_s = f"{cv[0]:<5.2f}" if cv else "  —  "
        g2_s = f"{g2[0]:<5.2f}" if g2 else "  —  "
        print(f"  {label:<16} {sha:<20} {ds:<6}  {cd_s}  {sc_s}  {cv_s}  {g2_s}")
    print("  (t4 preferred over t3 per ob-bf7: t3 scan spread=153% contaminated; t4 spread=17%)")

    print("\n  --- Fleet sweep (ob-bf7): commit 234807d, clean tree, single-thread ---")
    sweep = [
        ("RK3588 t4 big (clean)", "rk3588-t4-clean.csv"),
        ("RK3588 t3 big (clean)", "rk3588-t3-clean.csv"),
        ("RK3588 t4 little (cln)", "rk3588-t4-little-clean.csv"),
        ("RK3588 t3 little (cln)", "rk3588-t3-little-clean.csv"),
        ("Jetson j1 (clean)", "jetson-j1-clean.csv"),
        ("Jetson j2 (clean)", "jetson-j2-clean.csv"),
    ]
    for label, fname in sweep:
        rows = load_kernel(fname)
        cd = kv(rows, "Qwen3.5-4B", "gdn_cumdecay", 64)
        sc = kv(rows, "Qwen3.5-4B", "gdn_gated_scan", 64)
        cv = kv(rows, "Qwen3.5-4B", "gdn_causal_dwconv1d", 64)
        g2 = kv(rows, "Qwen3.5-4B", "gdn2_gated_scan", 64)
        cd_s = f"{cd[0]:<5.2f}" if cd else "  —  "
        sc_s = f"{sc[0]:<5.2f}" if sc else "  —  "
        cv_s = f"{cv[0]:<5.2f}" if cv else "  —  "
        g2_s = f"{g2[0]:<5.2f}" if g2 else "  —  "
        print(f"  {label:<20} 234807d             CLEAN  {cd_s}  {sc_s}  {cv_s}  {g2_s}")
    print("  (All at commit 234807d, dirty=false, OMP_NUM_THREADS=1, governor=performance)")

    print("\n" + "=" * 90)
    print("OPTIMIZATION LADDER on Jetson (Qwen3.5-4B, seq=64) — GiB/s @ p50 / p50 µs")
    ladder = [
        ("baseline single (j1, manifest, dirty)", "jetson-j1.csv"),
        ("baseline single (j2, manifest, dirty)", "jetson-j2.csv"),
        ("CLEAN tree 4-core OMP (j1, dirty=false)", "jetson-j1_clean.csv"),
        ("single-core (j2_single, reconstructed)", "jetson-j2_single.csv"),
        ("4-core OMP (j2_omp, reconstructed)", "jetson-j2-omp.csv"),
        ("4-core OMP full (j2_omp-full, reconstructed)", "jetson-j2-omp-full.csv"),
        ("conv-unroll (j2, reconstructed)", "jetson-j2-conv-unroll.csv"),
        ("omp+unroll (j2, reconstructed)", "jetson-j2-omp-unroll.csv"),
        ("FULL-OPTIMIZED (j2, manifest, dirty)", "jetson-j2-full-optimized.csv"),
    ]
    print("Config                                        CumDecay       Scan          DWConv1D")
    for label, fname in ladder:
        rows = load_kernel(fname)
        cd = kv(rows, "Qwen3.5-4B", "gdn_cumdecay", 64)
        sc = kv(rows, "Qwen3.5-4B", "gdn_gated_scan", 64)
        cv = kv(rows, "Qwen3.5-4B", "gdn_causal_dwconv1d", 64)

        def cell(v):
            return f"{v[0]:.2f} GiB/s ({v[1]:.0f}us)" if v else "—"

        print(f"  {label:<42} {cell(cd):<14} {cell(sc):<13} {cell(cv):<14}")

    # speedup ratios (within j2 series, same device)
    print("\n  Within-j2-series speedups (4B seq=64, GiB/s):")

    for k in ("gdn_cumdecay", "gdn_gated_scan", "gdn_causal_dwconv1d"):
        base = gib("jetson-j2_single.csv", k)
        omp = gib("jetson-j2-omp.csv", k)
        full = gib("jetson-j2-full-optimized.csv", k)
        print(
            f"    {k:<20} single={base if base else 0:.2f}  "
            f"omp={omp if omp else 0:.2f} ({omp / base if base and omp else 0:.2f}x)  "
            f"full-opt={full if full else 0:.2f} ({full / base if base and full else 0:.2f}x)"
        )
    # manifest-backed clean-tree 4-core vs manifest-backed canonical single (j1 same device)
    j1_single = gib("jetson-j1.csv", "gdn_gated_scan")
    j1_clean = gib("jetson-j1_clean.csv", "gdn_gated_scan")
    print(
        f"  Manifest-backed (j1 device): single j1.csv scan={j1_single:.2f} "
        f"-> 4-core CLEAN j1_clean scan={j1_clean:.2f} "
        f"({j1_clean / j1_single:.2f}x, SUPERLINEAR=confounded)"
    )

    print("\n" + "=" * 90)
    print("DECODE (seq=1) KERNEL — Qwen3.5-4B — p50 µs/token + GiB/s (recurrence cost)")
    for label, fname in [
        ("j1 CLEAN 4-core (manifest)", "jetson-j1_clean.csv"),
        ("j2 full-optimized (manifest)", "jetson-j2-full-optimized.csv"),
    ]:
        rows = load_kernel(fname)
        print("  ", label)
        for k in ("gdn_cumdecay", "gdn_gated_scan", "gdn_causal_dwconv1d"):
            v = kv(rows, "Qwen3.5-4B_decode", k, 1)
            if v:
                print(f"     {k:<22} {v[1]:.3f} us/token   {v[0]:.2f} GiB/s   spread={v[2]:.1f}%")

    print("\n" + "=" * 90)
    print("SUSTAINED LOAD / THERMAL (gdn_gated_scan, Qwen3.5-4B, seq=64)")
    for label, fname in [
        ("j1 sustained 120s (manifest, dirty)", "jetson-j1_sustained.csv"),
        ("j2 1-core 30s (manifest, dirty)", "jetson-j2_sustained_1core.csv"),
        ("j2 4-core 30s (manifest, dirty)", "jetson-j2_sustained_4core.csv"),
        ("j2 optimized 120s (manifest, dirty)", "jetson-j2-sustained-optimized.csv"),
    ]:
        s = load_sustained(fname)
        if s:
            print(
                f"  {label:<38} first={s[0]:.2f} last={s[1]:.2f} median={s[2]:.2f} GiB/s  "
                f"drift={(s[1] - s[0]) / s[0] * 100:+.1f}%  (n={s[3]} samples)"
                if s[0]
                else "drift=N/A"
            )

    print("\n" + "=" * 90)
    print("POWER / ENERGY (jetson-j1_power.json, INA3221, 10s sustained, Qwen3.5-4B seq=64)")
    with open(os.path.join(MAN, "jetson-j1_power.json")) as f:
        pj = json.load(f)
    for run in pj["sustained_runs"]:
        print(
            f"  {run['kernel']:<22} {run['throughput_gib_per_s']:.2f} GiB/s  "
            f"delta_board={run['delta_power_mw']['board']:4d}mW  "
            f"delta_cpu={run['delta_power_mw']['cpu']:4d}mW  "
            f"energy={run['energy_per_gib_board_mj']:4d} mJ/GiB(board) "
            f"{run['energy_per_gib_cpu_mj']:4d} mJ/GiB(cpu)  "
            f"therm {run['thermal_c']['idle'] / 1000.0}->{run['thermal_c']['peak'] / 1000.0}C"
        )
    print("  provenance: bead ob-agf.1/ob-mrd.7, run_id jetson-j1 (no git sha in this manifest)")

    print("\nDONE.")


if __name__ == "__main__":
    main()
