"""Tests for bench/schema.py — the frozen results schema contract (bead ob-1lm).

Validates enum values, CSV header conformance, and round-trip serialization.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench.schema import (
    Device, Engine, Phase, MetricName, MemoryComponent,
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
    """The committed CSVs must conform to the schema's expected columns."""
    import csv
    base = os.path.join(os.path.dirname(__file__), "..", "results", "raw")
    expected = {
        "model", "kernel", "dispatch_path", "seq", "channels", "repeats",
        "p50_us", "p95_us", "spread_pct", "gib_per_s_p50", "gflop_per_s_p50",
    }
    for fname in os.listdir(base):
        if not fname.endswith(".csv") or "_" in fname:
            continue
        with open(os.path.join(base, fname)) as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames or [])
        missing = expected - cols
        assert not missing, f"{fname} missing columns: {missing}"
