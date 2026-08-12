#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""
RKNN operator-coverage probe — the RK3588/RKNN counterpart to the CIX NOE audit.

Feeds the same seven ONNX probe graphs from artifacts/npu_op_probe/ through
rknn-toolkit2's load_onnx + build pipeline and records what it accepts,
silently falls back on, or rejects.

Bead: ob-t3b.5
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROBE_DIR = Path("artifacts/npu_op_probe")
OUT_DIR = Path("artifacts/npu_op_probe/rknn_audit")
MANIFEST = PROBE_DIR / "manifest.json"


def extract_op_table(output: str) -> list:
    """Parse the 'Network Layer Information Table' from RKNN verbose output."""
    ops = []
    in_table = False
    for line in output.split("\n"):
        # Strip RKNN log prefix: "D RKNN: [HH:MM:SS.mmm] actual content"
        clean = re.sub(r"^[DIEW]\s+RKNN:\s*\[[\d:.]+\]\s*", "", line.strip())
        # Also strip ANSI color codes
        clean = re.sub(r"\x1b\[[0-9;]*m", "", clean)

        if "Network Layer Information Table" in clean:
            in_table = True
            continue
        if in_table:
            if clean.startswith("---") and len(ops) > 0:
                break
            if clean.startswith("---"):
                continue
            # Skip header row
            if clean.startswith("ID"):
                continue
            parts = clean.split()
            if len(parts) >= 4 and parts[0].isdigit():
                op_id = parts[0]
                op_type = parts[1]
                dtype = parts[2]
                target = parts[3]
                ops.append(
                    {
                        "id": int(op_id),
                        "op_type": op_type,
                        "dtype": dtype,
                        "target": target,
                    }
                )
    return ops


def extract_key_evidence(output: str) -> list:
    """Extract the most informative lines for the results table."""
    keywords = [
        "unsupported",
        "UNSUPPORTED",
        "not support",
        "not found",
        "fallback",
        "FALLBACK",
        "error",
        "ERROR",
        "_RET=",
        "RKNN_FILE_SIZE=",
        "Graph is not DAG",
        "cannot",
        "failed",
    ]
    lines = []
    for line in output.split("\n"):
        stripped = line.strip()
        # Strip ANSI color codes
        clean = re.sub(r"\x1b\[[0-9;]*m", "", stripped)
        for kw in keywords:
            if kw.lower() in clean.lower():
                lines.append(clean[:200])
                break
    return lines[:30]


def probe_one(onnx_path: str, probe_name: str) -> dict:
    """Run one ONNX probe through RKNN toolkit2 with full verbose capture."""

    script = f'''
import sys, os
os.environ["RKNN_LOG_LEVEL"] = "3"
from rknn.api import RKNN

rknn = RKNN(verbose=True)

ret = rknn.config(
    mean_values=[],
    std_values=[],
    target_platform="rk3588",
    float_dtype="float16",
    optimization_level=3,
)
print(f"CONFIG_RET={{ret}}", file=sys.stderr)

ret = rknn.load_onnx(model="{onnx_path}")
print(f"LOAD_RET={{ret}}", file=sys.stderr)

if ret == 0:
    ret = rknn.build(do_quantization=False)
    print(f"BUILD_RET={{ret}}", file=sys.stderr)

    if ret == 0:
        ret = rknn.export_rknn("/tmp/{probe_name}.rknn")
        print(f"EXPORT_RET={{ret}}", file=sys.stderr)
        import os as _os
        if _os.path.exists("/tmp/{probe_name}.rknn"):
            print(f"RKNN_FILE_SIZE={{_os.path.getsize('/tmp/{probe_name}.rknn')}}", file=sys.stderr)
            _os.remove("/tmp/{probe_name}.rknn")

rknn.release()
'''

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        errors="replace",  # RKNN may emit non-UTF-8 bytes on rejection
    )

    combined = proc.stdout + "\n" + proc.stderr

    # Save full log
    log_path = OUT_DIR / f"{probe_name}.log"
    with open(log_path, "w") as f:
        f.write(combined)

    # Parse results
    config_ret = re.search(r"CONFIG_RET=(\d+)", combined)
    load_ret = re.search(r"LOAD_RET=(\d+)", combined)
    build_ret = re.search(r"BUILD_RET=(\d+)", combined)
    export_ret = re.search(r"EXPORT_RET=(\d+)", combined)
    file_size = re.search(r"RKNN_FILE_SIZE=(\d+)", combined)

    result = {
        "probe": probe_name,
        "config_ret": int(config_ret.group(1)) if config_ret else None,
        "load_ret": int(load_ret.group(1)) if load_ret else None,
        "build_ret": int(build_ret.group(1)) if build_ret else None,
        "export_ret": int(export_ret.group(1)) if export_ret else None,
        "rknn_file_size": int(file_size.group(1)) if file_size else None,
        "op_table": extract_op_table(combined),
        "evidence": extract_key_evidence(combined),
        "log": f"{probe_name}.log",
    }

    # Determine verdict
    if result["load_ret"] != 0:
        result["verdict"] = "REJECTED_AT_LOAD"
    elif result["build_ret"] != 0:
        result["verdict"] = "REJECTED_AT_BUILD"
    elif result["export_ret"] != 0:
        result["verdict"] = "COMPILED_NO_EXPORT"
    else:
        result["verdict"] = "COMPILED"

    # Check for CPU fallbacks in op table
    if result["op_table"]:
        cpu_ops = [
            op
            for op in result["op_table"]
            if op["target"] == "CPU" and op["op_type"] not in ("InputOperator", "OutputOperator")
        ]
        npu_ops = [op for op in result["op_table"] if op["target"] == "NPU"]
        result["cpu_fallback_ops"] = [f"{op['op_type']}:{op['id']}" for op in cpu_ops]
        result["npu_ops"] = [f"{op['op_type']}:{op['id']}" for op in npu_ops]

    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST) as f:
        manifest = json.load(f)

    results = []

    for entry in manifest:
        probe_name = entry["probe"]
        onnx_file = entry["file"]
        onnx_path = str(PROBE_DIR / onnx_file)
        ops = entry["ops"]

        print(f"\n{'=' * 70}")
        print(f"Probe: {probe_name} ({onnx_file})")
        print(f"  ONNX ops: {', '.join(ops)}")
        print(f"{'=' * 70}")

        if not os.path.exists(onnx_path):
            print("  SKIP: file not found")
            continue

        result = probe_one(onnx_path, probe_name)
        result["onnx_ops"] = ops
        result["description"] = entry["description"]

        print(f"  Verdict: {result['verdict']}")
        print(
            f"  load_ret={result['load_ret']} build_ret={result['build_ret']} export_ret={result['export_ret']}"
        )

        if result.get("op_table"):
            print("  Op placement:")
            for op in result["op_table"]:
                marker = (
                    " ← CPU FALLBACK"
                    if op["target"] == "CPU"
                    and op["op_type"] not in ("InputOperator", "OutputOperator")
                    else ""
                )
                print(f"    [{op['target']:3s}] {op['op_type']:<20s} {op['dtype']}{marker}")

        if result.get("cpu_fallback_ops"):
            print(f"  CPU FALLBACKS: {', '.join(result['cpu_fallback_ops'])}")

        if result["evidence"]:
            print("  Key evidence:")
            for e in result["evidence"][:8]:
                print(f"    {e}")

        results.append(result)

    # Write results JSON
    results_path = OUT_DIR / "rknn_audit_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {len(results)} probes completed")
    print(f"Results written to: {results_path}")
    print(f"{'=' * 70}\n")

    print(f"{'Probe':<25} {'Verdict':<20} {'CPU Fallbacks':<30} {'Notes'}")
    print("-" * 100)
    for r in results:
        fallbacks = ", ".join(r.get("cpu_fallback_ops", [])) or "—"
        notes = ""
        if r["verdict"] == "COMPILED" and not r.get("cpu_fallback_ops"):
            notes = "all-NPU"
        elif r["verdict"] == "COMPILED":
            notes = "silent fallback"
        print(f"{r['probe']:<25} {r['verdict']:<20} {fallbacks:<30} {notes}")


if __name__ == "__main__":  # pragma: no cover
    main()
