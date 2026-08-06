#!/usr/bin/env python3
"""Feed the NPU operator-coverage ONNX probes to the Rockchip RKNN toolkit.

Bead ``ob-t3b.5``. The same seven probe graphs that were driven through the CIX
NOE Compiler (``ob-t3b.1``, see ``docs/FINDINGS.md`` §1) are now fed to the
entirely independent Rockchip RKNN toolchain on the RK3588.  If RKNN rejects
the same control-flow operators (Scan, runtime-length Loop), the finding
generalises from "a CIX limitation" to "edge NPU toolchains generally cannot
host a linear-attention recurrence".

Usage::

    python3 scripts/rknn_op_probe.py --probe-dir artifacts/npu_op_probe --out artifacts/npu_op_probe/audit_rknn

Outputs one log file per probe plus a summary JSON.
Requires rknn-toolkit2 (pip install rknn-toolkit2).
"""

import json
import os
import sys
import traceback
from pathlib import Path

PROBE_DESCRIPTIONS = {
    "01_causal_conv1d":      ("Conv",             "GDN depthwise causal Conv1D (asymmetric pads)"),
    "02_decay_cumprod":      ("Log,CumSum,Exp",   "Gated decay via exp(cumsum(log(a)))"),
    "03_delta_rule_update":  ("MatMul,Sub,Add",   "Delta-rule state update S←(I−kkᵀ)S+kvᵀ"),
    "04_gate_chain":         ("Sigmoid,Softplus", "Elementwise gate chain"),
    "05_scan_recurrence":    ("Scan",             "Chunk-to-chunk recurrence via ONNX Scan"),
    "06_loop_recurrence":    ("Loop (const)",     "Recurrence via Loop, compile-time trip count"),
    "07_loop_dynamic_trip":  ("Loop (runtime)",   "Recurrence via Loop, runtime trip count"),
}


def probe_one(rknn, onnx_path):
    """Run a single ONNX probe through the RKNN pipeline.

    Returns a dict with stage results.
    """
    result = {
        "probe": Path(onnx_path).stem,
        "onnx_ops": PROBE_DESCRIPTIONS.get(Path(onnx_path).stem, ("?", "?"))[0],
        "description": PROBE_DESCRIPTIONS.get(Path(onnx_path).stem, ("?", "?"))[1],
        "load_onnx": None,
        "build": None,
        "export": None,
        "verdict": None,
        "error": None,
        "stderr": None,
    }

    # Stage 1: load_onnx
    try:
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            result["load_onnx"] = f"FAILED (rc={ret})"
            result["verdict"] = "rejected"
            result["error"] = f"load_onnx returned {ret}"
            return result
        result["load_onnx"] = "OK"
    except Exception as e:
        msg = str(e)
        result["load_onnx"] = f"EXCEPTION: {msg[:200]}"
        result["verdict"] = "rejected"
        result["error"] = traceback.format_exc()[-500:]
        return result

    # Stage 2: build (no quantization — we are testing operator coverage)
    try:
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            result["build"] = f"FAILED (rc={ret})"
            result["verdict"] = "rejected"
            result["error"] = f"build returned {ret}"
            return result
        result["build"] = "OK"
    except Exception as e:
        msg = str(e)
        result["build"] = f"EXCEPTION: {msg[:200]}"
        result["verdict"] = "rejected"
        result["error"] = traceback.format_exc()[-500:]
        return result

    # Stage 3: export (confirms a valid RKNN model was produced)
    try:
        export_path = str(onnx_path).replace(".onnx", ".rknn")
        ret = rknn.export_rknn(export_path)
        if ret != 0:
            result["export"] = f"FAILED (rc={ret})"
            # Build succeeded but export failed — still counts as "compiles"
            result["verdict"] = "compiles (export failed)"
            return result
        result["export"] = "OK"
        result["verdict"] = "compiles"
    except Exception as e:
        msg = str(e)
        result["export"] = f"EXCEPTION: {msg[:200]}"
        result["verdict"] = "compiles (export failed)"
        return result

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RKNN NPU operator-coverage probe (ob-t3b.5)")
    parser.add_argument("--probe-dir", default="artifacts/npu_op_probe",
                        help="Directory containing the ONNX probe graphs")
    parser.add_argument("--out", default="artifacts/npu_op_probe/audit_rknn",
                        help="Output directory for logs and summary")
    args = parser.parse_args()

    probe_dir = Path(args.probe_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all .onnx files (sorted)
    onnx_files = sorted(probe_dir.glob("*.onnx"))
    if not onnx_files:
        print(f"ERROR: no .onnx files found in {probe_dir}")
        return 1

    print(f"RKNN operator-coverage probe (ob-t3b.5)")
    print(f"Toolkit: rknn-toolkit2")
    print(f"Probes: {len(onnx_files)}")
    print(f"Output: {out_dir}")
    print("=" * 72)

    from rknn.api import RKNN

    results = []
    for onnx_path in onnx_files:
        stem = onnx_path.stem
        print(f"\n--- {stem} ---")
        print(f"  Ops: {PROBE_DESCRIPTIONS.get(stem, ('?','?'))[0]}")
        print(f"  Desc: {PROBE_DESCRIPTIONS.get(stem, ('?','?'))[1]}")

        # Each probe needs a fresh RKNN instance
        rknn = RKNN(verbose=False)

        # config() must be called before load_onnx()
        # No mean/std — these are not image models (varying channel dims)
        rknn.config(
            target_platform='rk3588',
            float_dtype='float16',
            optimization_level=3,
        )

        # Capture stderr from the C library (RKNN prints to stderr)
        import io
        old_stderr = sys.stderr
        sys.stderr = captured = io.StringIO()

        result = probe_one(rknn, onnx_path)

        sys.stderr = old_stderr
        result["stderr"] = captured.getvalue()[-2000:] if captured.getvalue() else None

        rknn.release()

        # Print results
        print(f"  load_onnx: {result['load_onnx']}")
        print(f"  build:     {result['build']}")
        print(f"  export:    {result['export']}")
        print(f"  VERDICT:   {result['verdict']}")
        if result["stderr"]:
            # Extract relevant lines
            for line in result["stderr"].split("\n"):
                line = line.strip()
                if line and ("ERROR" in line.upper() or "UNSUPPORT" in line.upper()
                             or "WARN" in line.upper() or "FAIL" in line.upper()):
                    print(f"  log: {line[:120]}")

        # Write per-probe log
        log_path = out_dir / f"{stem}.rknn.log"
        with open(log_path, "w") as f:
            f.write(f"Probe: {stem}\n")
            f.write(f"ONNX ops: {result['onnx_ops']}\n")
            f.write(f"Description: {result['description']}\n")
            f.write(f"load_onnx: {result['load_onnx']}\n")
            f.write(f"build: {result['build']}\n")
            f.write(f"export: {result['export']}\n")
            f.write(f"verdict: {result['verdict']}\n")
            if result["error"]:
                f.write(f"\n--- Error/Traceback ---\n{result['error']}\n")
            if result["stderr"]:
                f.write(f"\n--- Captured stderr ---\n{result['stderr']}\n")
        print(f"  log saved: {log_path}")

        results.append(result)

    # Write summary JSON
    summary_path = out_dir / "summary.json"
    summary = {
        "toolkit": "rknn-toolkit2",
        "version": "2.3.2",
        "device": "rk3588-t4",
        "probe_count": len(results),
        "results": results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Probe':<30s} {'ONNX ops':<22s} {'Verdict'}")
    print("-" * 72)
    for r in results:
        print(f"{r['probe']:<30s} {r['onnx_ops']:<22s} {r['verdict']}")
    print("-" * 72)
    print(f"\nSummary JSON: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
