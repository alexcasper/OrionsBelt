"""Unit tests for the benchmark harness (ob-ljh).

Tests the measurement protocol (warmup → timed repeats → per-repeat rows),
CSV schema conformance, percentile reporting, and the synthetic backend.
These run in CI on Python 3.10/3.13 — no model weights or GPU required.
"""

import math
import os
import sys
import tempfile

import pytest

# Make bench/ importable when running from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import ResultRow, validate_row, validate_rows, write_csv, read_csv
from harness import (
    HarnessConfig,
    SyntheticBackend,
    TimingResult,
    RunResult,
    MemorySnapshot,
    _percentile,
    run_single,
    result_to_rows,
    run_sweep,
    summarize,
)


class TestPercentile:
    def test_single_value(self):
        assert _percentile([42.0], 50) == 42.0

    def test_p50_even(self):
        vals = [1.0, 2.0, 3.0, 4.0]
        p50 = _percentile(vals, 50)
        assert 2.0 <= p50 <= 3.0

    def test_p95(self):
        vals = list(range(1, 21))  # 1..20
        p95 = _percentile(vals, 95)
        assert p95 >= 18  # near the top

    def test_empty(self):
        assert math.isnan(_percentile([], 50))


class TestHarnessConfig:
    def test_min_repeats_enforced(self):
        """METRICS.md §9: never a single best run — min 5 repeats."""
        with pytest.raises(ValueError, match="must be >= 5"):
            HarnessConfig(
                model_checkpoint="test",
                device="x86_reference",
                engine_gdn="cpu",
                engine_full_attention="cpu",
                quantization="fp16",
                context_lengths=[4096],
                repeats=3,
            )

    def test_auto_run_id_generated(self):
        cfg = HarnessConfig(
            model_checkpoint="test",
            device="generic_aarch64",
            engine_gdn="cpu",
            engine_full_attention="cpu",
            quantization="fp16",
            context_lengths=[4096],
        )
        assert cfg.run_id.startswith("generic_aarch64_")
        assert cfg.manifest_ref.startswith("results/manifests/")


class TestResultToRows:
    """Verify that timing results produce correct schema-conformant rows."""

    def _make_config(self):
        return HarnessConfig(
            model_checkpoint="Qwen/Qwen3.5-4B",
            device="generic_aarch64",
            engine_gdn="cpu",
            engine_full_attention="cpu",
            quantization="fp16",
            context_lengths=[4096],
            run_id="test_run",
            manifest_ref="results/manifests/test_run.json",
        )

    def _make_result(self, prefill_dur=1.0, decode_dur=12.0):
        return RunResult(
            timing=TimingResult(
                t_submit=0.0,
                t_first_token=prefill_dur,
                t_last_token=prefill_dur + decode_dur,
                prompt_tokens=4096,
                decode_tokens=256,
            ),
        )

    def test_produces_prefill_throughput(self):
        rows = result_to_rows(self._make_result(), self._make_config(), 4096, 0, 5)
        prefill_rows = [r for r in rows if r.metric_name == "prefill_tokens_per_sec"]
        assert len(prefill_rows) == 1
        assert prefill_rows[0].phase == "prefill"
        assert prefill_rows[0].value == pytest.approx(4096.0)  # 4096 tokens / 1.0 sec
        assert prefill_rows[0].unit == "tokens_per_sec"
        assert prefill_rows[0].metric_component is None

    def test_produces_ttft(self):
        rows = result_to_rows(self._make_result(), self._make_config(), 4096, 0, 5)
        ttft_rows = [r for r in rows if r.metric_name == "ttft_seconds"]
        assert len(ttft_rows) == 1
        assert ttft_rows[0].value == pytest.approx(1.0)
        assert ttft_rows[0].phase == "prefill"

    def test_produces_decode_throughput(self):
        rows = result_to_rows(self._make_result(1.0, 12.0), self._make_config(), 4096, 0, 5)
        decode_rows = [r for r in rows if r.metric_name == "decode_tokens_per_sec"]
        assert len(decode_rows) == 1
        # 256 tokens / 12.0 sec ≈ 21.3 tok/s
        assert decode_rows[0].value == pytest.approx(256.0 / 12.0, rel=0.01)
        assert decode_rows[0].phase == "decode"

    def test_all_rows_validate(self):
        """Every row produced must pass schema validation."""
        rows = result_to_rows(self._make_result(), self._make_config(), 4096, 0, 5)
        for row in rows:
            validate_row(row)  # should not raise

    def test_memory_rows_have_component(self):
        result = self._make_result()
        result.memory = [
            MemorySnapshot(component="weights", phase="prefill", peak_bytes=2_500_000_000),
            MemorySnapshot(component="kv_cache", phase="decode", peak_bytes=134_217_728),
            MemorySnapshot(component="recurrent_state", phase="decode", peak_bytes=50_331_648),
        ]
        rows = result_to_rows(result, self._make_config(), 4096, 0, 5)
        mem_rows = [r for r in rows if r.metric_name == "peak_memory_bytes"]
        assert len(mem_rows) == 3
        components = {r.metric_component for r in mem_rows}
        assert components == {"weights", "kv_cache", "recurrent_state"}
        for row in mem_rows:
            validate_row(row)  # must pass including component validation


class TestSyntheticBackend:
    def test_smoke_sweep(self):
        """A minimal sweep with the synthetic backend should produce valid rows."""
        cfg = HarnessConfig(
            model_checkpoint="test-model",
            device="x86_reference",
            engine_gdn="cpu",
            engine_full_attention="cpu",
            quantization="fp16",
            context_lengths=[4096],
            warmups=1,
            repeats=5,
            run_id="smoke_test",
            manifest_ref="results/manifests/smoke_test.json",
        )
        backend = SyntheticBackend()
        rows = run_sweep(backend, cfg, progress=False)

        # 1 context length, 5 repeats, at least 3 metrics (prefill, ttft, decode)
        assert len(rows) >= 15  # 5 repeats × 3+ metrics
        validate_rows(rows)  # all must pass

    def test_csv_roundtrip(self):
        """Rows written to CSV must read back identically."""
        cfg = HarnessConfig(
            model_checkpoint="test-model",
            device="x86_reference",
            engine_gdn="cpu",
            engine_full_attention="cpu",
            quantization="fp16",
            context_lengths=[4096],
            warmups=0,
            repeats=5,
            run_id="csv_test",
            manifest_ref="results/manifests/csv_test.json",
        )
        backend = SyntheticBackend()
        rows = run_sweep(backend, cfg, progress=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            write_csv(rows, path)
            read_back = read_csv(path)
            assert len(read_back) == len(rows)
            # Spot-check a field
            assert read_back[0].run_id == "csv_test"
            assert read_back[0].device == "x86_reference"
        finally:
            os.unlink(path)


class TestSummarize:
    def test_summary_has_headers(self):
        cfg = HarnessConfig(
            model_checkpoint="test",
            device="x86_reference",
            engine_gdn="cpu",
            engine_full_attention="cpu",
            quantization="fp16",
            context_lengths=[4096],
            warmups=0,
            repeats=5,
            run_id="sum_test",
            manifest_ref="results/manifests/sum_test.json",
        )
        rows = run_sweep(SyntheticBackend(), cfg, progress=False)
        summary = summarize(rows)
        assert "p50" in summary
        assert "p95" in summary
        assert "spread" in summary
        assert "prefill_tokens_per_sec" in summary
        assert "decode_tokens_per_sec" in summary
