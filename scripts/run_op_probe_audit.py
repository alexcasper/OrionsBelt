#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Run the CIX NOE frontend against each operator probe and record coverage.

Bead ``ob-t3b.1``. Consumes the graphs emitted by ``scripts/npu_op_probe.py``.

Runs on an **x86 host with no Orion O6 board attached** — the NOE Compiler is a host-side
tool, so operator coverage is answerable before hardware exists. Only *executing* a
compiled artifact needs the board.

``cixparse`` is the frontend that lowers a framework graph into AIPU IR, so it is the
narrowest gate for "does this operator exist for this target at all". A probe that fails
here fails for operator-support reasons rather than anything downstream like quantization
calibration.

Usage:

    python3 scripts/run_op_probe_audit.py \
        --cixparse /path/to/noe310/bin/cixparse \
        --probe-dir artifacts/npu_op_probe \
        --out-dir artifacts/npu_op_probe/audit
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def cfg_for(model_path: Path, name: str) -> str:
    """Build a minimal Common-section cfg describing one probe graph."""
    import onnx

    model = onnx.load(str(model_path))
    graph = model.graph
    initializers = {i.name for i in graph.initializer}
    inputs = [i for i in graph.input if i.name not in initializers]

    def shape_of(vi) -> str:
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(str(d.dim_value) if d.dim_value else "1")
        return "[" + ",".join(dims) + "]" if dims else "[1]"

    return "\n".join(
        [
            "[Common]",
            "mode=build",
            f"model_name={name}",
            f"input_model={model_path.resolve()}",
            f"input={','.join(i.name for i in inputs)}",
            f"input_shape={','.join(shape_of(i) for i in inputs)}",
            f"output={','.join(o.name for o in graph.output)}",
            "",
        ]
    )


def classify(returncode: int, log: str) -> tuple[str, str]:
    """Map a cixparse invocation to (verdict, evidence).

    Deliberately conservative: anything not clearly a success is reported as such rather
    than smoothed over, and unsupported-operator errors are separated from other
    failures so the finding is about coverage rather than our own config mistakes.
    """
    low = log.lower()
    unsupported_markers = [
        "unsupported op",
        "unsupported operator",
        "not supported",
        "unsupported type",
        "no conversion",
        "cannot find",
        "unknown op",
        "unimplemented",
    ]
    hit = next((m for m in unsupported_markers if m in low), None)
    lines = [ln.strip() for ln in log.splitlines() if ln.strip()]

    def find(marker: str) -> str:
        for ln in lines:
            if marker in ln.lower():
                return ln[:300]
        return ""  # pragma: no cover (unreachable: marker confirmed in full-text low)

    if hit:
        return "UNSUPPORTED_OP", find(hit)
    if returncode != 0:
        err = next(
            (ln for ln in reversed(lines) if re.search(r"error|fail|exception", ln, re.I)),
            lines[-1] if lines else "",
        )
        return "FAILED", err[:300]
    if "warning" in low:
        return "OK_WITH_WARNINGS", find("warning")
    return "OK", lines[-1][:300] if lines else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cixparse", required=True, help="path to the cixparse executable")
    ap.add_argument("--probe-dir", default="artifacts/npu_op_probe")
    ap.add_argument("--out-dir", default="artifacts/npu_op_probe/audit")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    probe_dir = Path(args.probe_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((probe_dir / "manifest.json").read_text())
    results = []

    for entry in manifest:
        name = entry["probe"]
        model_path = probe_dir / entry["file"]
        cfg_path = out_dir / f"{name}.cfg"
        log_path = out_dir / f"{name}.log"
        cfg_path.write_text(cfg_for(model_path, name))

        try:
            proc = subprocess.run(
                [args.cixparse, "-c", str(cfg_path), "-v"],
                capture_output=True,
                text=True,
                timeout=args.timeout,
                cwd=str(out_dir),
            )
            log = (proc.stdout or "") + (proc.stderr or "")
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            log, rc = f"TIMEOUT after {args.timeout}s", 124

        log_path.write_text(log)
        verdict, evidence = classify(rc, log)
        results.append(
            {
                "probe": name,
                "ops": entry["ops"],
                "returncode": rc,
                "verdict": verdict,
                "evidence": evidence,
                "log": log_path.name,
            }
        )
        print(f"  {name:24s} {verdict:18s} rc={rc}")
        if evidence:
            print(f"      {evidence[:150]}")

    (out_dir / "audit_results.json").write_text(json.dumps(results, indent=2) + "\n")

    print("\nSummary")
    for verdict in ("OK", "OK_WITH_WARNINGS", "UNSUPPORTED_OP", "FAILED"):
        hits = [r["probe"] for r in results if r["verdict"] == verdict]
        if hits:
            print(f"  {verdict}: {', '.join(hits)}")
    print(f"\nDetail in {out_dir}/ (one .cfg and .log per probe, plus audit_results.json)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
