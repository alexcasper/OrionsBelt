# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Tests for bench/schema.py — the frozen results schema contract (bead ob-1lm).

Validates enum values, CSV header conformance, and round-trip serialization.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench.schema import (  # noqa: E402
    Device,
    Engine,
    MemoryComponent,
    MetricName,
    Phase,
)


def test_device_enum_values():
    assert Device.O6.value == "o6"
    assert Device.GENERIC_AARCH64.value == "generic_aarch64"
    assert Device.X86_REFERENCE.value == "x86_reference"


def test_engine_enum_values():
    assert Engine.NPU.value == "npu"
    assert Engine.GPU_VULKAN.value == "gpu_vulkan"
    assert Engine.CPU.value == "cpu"


def test_phase_enum_values():
    assert Phase.PREFILL.value == "prefill"
    assert Phase.DECODE.value == "decode"


def test_metric_name_enum_values():
    assert MetricName.PREFILL_TOKENS_PER_SEC.value == "prefill_tokens_per_sec"
    assert MetricName.DECODE_TOKENS_PER_SEC.value == "decode_tokens_per_sec"
    assert MetricName.TTFT_SECONDS.value == "ttft_seconds"


def test_memory_component_enum_values():
    vals = {c.value for c in MemoryComponent}
    assert "weights" in vals
    assert "kv_cache" in vals


def test_csv_header_from_existing_results():
    """Every committed microbenchmark CSV must carry the full column set.

    Selection is by *header*, not filename. The original version skipped files
    containing "_", intending to exclude sustained and power CSVs — but
    jetson-j2-sustained-optimized.csv is hyphen-separated, so it slipped through
    and failed for the right reason in the wrong place. results/raw/ legitimately
    holds seven different shapes, so detect the shape and assert accordingly.
    """
    import csv

    from bench.schema import COLUMNS as RESULT_ROW_COLUMNS

    base = os.path.join(os.path.dirname(__file__), "..", "results", "raw")
    expected = {
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
    }
    # Markers that identify the other six CSV shapes; see scripts/validate_results.py.
    sustained_marker = "sustained_kernel"
    power_marker = "power_in_mw"
    layer_profile_marker = "layer_idx"  # bench/profile_layers.py (ob-c9k)
    e2e_decode_marker = "tok_per_sec_mean"  # gdn_e2e_decode.c raw output (ob-mrd.8)
    e2e_ctxsweep_marker = "kv_cache_mb"  # ctx-sweep e2e raw (model,ctx_len,gdn_layer_us,...)
    ctx_sweep_marker = "gdn_layer_us"  # gdn_e2e_decode.c --ctx-sweep mode (ob-mrd.10)
    delta_matmul_marker = "M"  # bench_gdn --delta-matmul mode (ob-8qt.1)
    ctx_sweep_marker = "gdn_layer_us"  # context-length sweep (gdn_e2e_decode.c --ctx-sweep)
    kleidiai_marker = "shape"  # KleidiAI micro-kernel bench (bench_kai_gdn.c)
    prefill_gemm_marker = "prefill_M"  # prefill GEMM benchmark (gdn_e2e_decode.c --prefill)
    precision_cmp_marker = "variant"  # mixed-precision/prefill comparison CSVs (A/B variants)
    cross_tool_marker = "llamacpp_commit"  # cross-tool comparison (ob-mrd.15, validate_results.py "cross_tool_comparison")
    quant_accuracy_marker = "cos_sim"  # per-matmul quant accuracy validation (ob-8qt.18, validate_results.py "quant_accuracy")
    result_row_columns = set(RESULT_ROW_COLUMNS)

    checked = 0
    for fname in sorted(os.listdir(base)):
        if not fname.endswith(".csv"):
            continue
        if "_gpu_" in fname:
            continue  # GPU benchmark CSV (gdn_gpu_bench.c) — different format, no header row
        with open(os.path.join(base, fname)) as f:
            cols = set(csv.DictReader(f).fieldnames or [])
        if (
            sustained_marker in cols
            or power_marker in cols
            or layer_profile_marker in cols
            or delta_matmul_marker in cols
            or e2e_ctxsweep_marker in cols
            or kleidiai_marker in cols
            or prefill_gemm_marker in cols
            or precision_cmp_marker in cols
            or cross_tool_marker in cols
            or quant_accuracy_marker in cols
        ):
            continue  # different shape by design, not a conformance failure
        if e2e_decode_marker in cols:
            continue  # e2e decode raw CSV (simple format, converted by bench/convert_e2e_decode.py)
        if ctx_sweep_marker in cols:
            continue  # ctx-length sweep CSV (gdn_e2e_decode.c --ctx-sweep mode, ob-mrd.10)
        if result_row_columns <= cols:
            continue  # model-level ResultRow schema (bench/schema.py), not a microbenchmark CSV
        missing = expected - cols
        assert not missing, f"{fname} missing columns: {sorted(missing)}"
        checked += 1

    # Guard against the filter silently excluding everything.
    assert checked > 0, "no microbenchmark CSVs were checked — is the filter too broad?"
