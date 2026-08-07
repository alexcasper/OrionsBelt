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
    holds six different shapes, so detect the shape and assert accordingly.
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
    # Markers that identify the other five CSV shapes; see scripts/validate_results.py.
    sustained_marker = "sustained_kernel"
    power_marker = "power_in_mw"
    layer_profile_marker = "layer_idx"  # bench/profile_layers.py (ob-c9k)
    delta_matmul_marker = "M"  # bench_gdn --delta-matmul mode (ob-8qt.1)
    result_row_columns = set(RESULT_ROW_COLUMNS)

    checked = 0
    for fname in sorted(os.listdir(base)):
        if not fname.endswith(".csv"):
            continue
        with open(os.path.join(base, fname)) as f:
            cols = set(csv.DictReader(f).fieldnames or [])
        if sustained_marker in cols or power_marker in cols or layer_profile_marker in cols or delta_matmul_marker in cols:
            continue  # different shape by design, not a conformance failure
        if result_row_columns <= cols:
            continue  # model-level ResultRow schema (bench/schema.py), not a microbenchmark CSV
        missing = expected - cols
        assert not missing, f"{fname} missing columns: {sorted(missing)}"
        checked += 1

    # Guard against the filter silently excluding everything.
    assert checked > 0, "no microbenchmark CSVs were checked — is the filter too broad?"
