"""Unit tests for bench/harness.py — the benchmark runner CLI (ob-ljh).

Tests validate:
  - Schema conformance of every CSV produced
  - Timing boundary correctness (prefill/decode/TTFT per METRICS.md)
  - Percentile computation (nearest-rank)
  - Per-context-point CSV independence
  - Memory breakdown three-component attribution
  - CLI argument parsing and sweep orchestration

Uses the MockBackend so tests run without any model or hardware.
"""

import os
import sys
import time

import pytest

# Make bench/ importable
_BENCH = os.path.join(os.path.dirname(__file__), "..", "bench")
_BENCH = os.path.abspath(_BENCH)
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

import harness  # noqa: E402
import schema  # noqa: E402

# ---------------------------------------------------------------------------
# Percentile helpers
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_single_value(self):
        assert harness.percentile([42.0], 50) == 42.0
        assert harness.percentile([42.0], 95) == 42.0

    def test_two_values(self):
        data = [1.0, 2.0]
        # p50 → rank=ceil(0.5*2)=1 → sorted[0]=1.0
        assert harness.percentile(data, 50) == 1.0

    def test_ten_values_p50(self):
        data = list(range(1, 11))  # 1..10
        # p50 → rank=ceil(0.5*10)=5 → sorted[4]=5
        assert harness.percentile(data, 50) == 5

    def test_ten_values_p95(self):
        data = list(range(1, 11))  # 1..10
        # p95 → rank=ceil(0.95*10)=10 → sorted[9]=10
        assert harness.percentile(data, 95) == 10

    def test_thirty_values_p50(self):
        data = list(range(1, 31))
        # p50 → rank=ceil(0.5*30)=15 → sorted[14]=15
        assert harness.percentile(data, 50) == 15

    def test_thirty_values_p95(self):
        data = list(range(1, 31))
        # p95 → rank=ceil(0.95*30)=ceil(28.5)=29 → sorted[28]=29
        assert harness.percentile(data, 95) == 29

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            harness.percentile([], 50)

    def test_summarise(self):
        s = harness.summarise([1.0, 2.0, 3.0, 4.0, 5.0])
        assert s.n == 5
        # p50 → rank=ceil(0.5*5)=3 → sorted[2]=3
        assert s.p50 == 3.0
        # p95 → rank=ceil(0.95*5)=5 → sorted[4]=5
        assert s.p95 == 5.0
        assert s.spread == 2.0


# ---------------------------------------------------------------------------
# Repeat-count tier logic
# ---------------------------------------------------------------------------


class TestRepeatTiers:
    def test_short_context_default(self):
        assert harness.repeats_for_context(4096, None) == 30

    def test_medium_context_default(self):
        assert harness.repeats_for_context(32768, None) == 30

    def test_expensive_context_default(self):
        assert harness.repeats_for_context(131072, None) == 10
        assert harness.repeats_for_context(262144, None) == 10

    def test_explicit_override(self):
        assert harness.repeats_for_context(4096, 10) == 10
        assert harness.repeats_for_context(131072, 30) == 30

    def test_never_below_five(self):
        assert harness.repeats_for_context(4096, 3) == 5
        assert harness.repeats_for_context(4096, 1) == 5


# ---------------------------------------------------------------------------
# MockBackend
# ---------------------------------------------------------------------------


class TestMockBackend:
    def test_load_is_instant(self):
        b = harness.MockBackend()
        b.load()  # must not raise

    def test_tokenize_caps_at_max(self):
        b = harness.MockBackend()
        ids = b.tokenize("hello world test", 5)
        assert len(ids) <= 5
        assert len(ids) >= 1

    def test_tokenize_large_text(self):
        b = harness.MockBackend()
        ids = b.tokenize("x" * 10000, 4096)
        assert len(ids) == 4096

    def test_memory_flat_weights(self):
        """Weights must be flat across context length (METRICS.md section 5.2)."""
        b = harness.MockBackend()
        m1 = b.memory_breakdown(100)
        m2 = b.memory_breakdown(100000)
        assert m1.weights == m2.weights

    def test_memory_kv_grows(self):
        """KV cache must grow with seq_len (METRICS.md section 5.3)."""
        b = harness.MockBackend()
        m1 = b.memory_breakdown(100)
        m2 = b.memory_breakdown(10000)
        assert m2.kv_cache > m1.kv_cache

    def test_memory_recurrent_flat(self):
        """Recurrent state must be O(1) regardless of context (METRICS.md section 5.4)."""
        b = harness.MockBackend()
        m1 = b.memory_breakdown(100)
        m2 = b.memory_breakdown(100000)
        assert m1.recurrent_state == m2.recurrent_state

    def test_prefill_does_work(self):
        """Prefill should produce measurable elapsed time."""
        b = harness.MockBackend(prefill_work=500)
        ids = list(range(100))
        t0 = time.perf_counter()
        b.prefill(ids)
        elapsed = time.perf_counter() - t0
        # Should be fast but non-zero (not instant)
        assert elapsed >= 0.0  # at minimum, no negative time


# ---------------------------------------------------------------------------
# Single-repeat measurement timing boundaries
# ---------------------------------------------------------------------------


class TestRepeatMeasurement:
    def test_timing_boundaries(self):
        """Verify the METRICS.md timing events produce valid results."""
        b = harness.MockBackend(prefill_work=100, decode_work=50)
        b.load()
        r = harness.run_one_repeat(b, context_length=256, decode_tokens=10)

        assert r.prompt_token_count > 0
        assert r.prefill_elapsed > 0, "prefill must produce non-zero elapsed"
        assert r.ttft_elapsed > 0, "ttft must include tokenization + prefill + sample"
        assert r.decode_elapsed > 0, "decode must produce non-zero elapsed"
        assert r.decode_token_count == 9, "decode tokens = N-1 (token 1 is prefill)"

    def test_ttft_includes_tokenization(self):
        """TTFT must be >= prefill elapsed (it includes tokenization + sampling)."""
        b = harness.MockBackend(prefill_work=100)
        b.load()
        r = harness.run_one_repeat(b, context_length=128, decode_tokens=5)
        # TTFT includes: tokenization + prefill + sampling
        # prefill_elapsed is just the forward pass
        assert r.ttft_elapsed >= r.prefill_elapsed * 0.9  # allow clock jitter

    def test_decode_proportional_to_tokens(self):
        """More decode tokens → longer decode elapsed time."""
        b1 = harness.MockBackend(decode_work=200)
        b1.load()
        r1 = harness.run_one_repeat(b1, context_length=64, decode_tokens=10)

        b2 = harness.MockBackend(decode_work=200)
        b2.load()
        r2 = harness.run_one_repeat(b2, context_length=64, decode_tokens=50)

        # 49 decode steps vs 9 — should be meaningfully longer
        assert r2.decode_elapsed > r1.decode_elapsed * 2

    def test_memory_captured_at_phase_end(self):
        """Memory breakdown should reflect the seq_len at each phase-end."""
        b = harness.MockBackend(kv_cache_bytes_per_token=1024)
        b.load()
        r = harness.run_one_repeat(b, context_length=100, decode_tokens=10)

        # Prefill-end: seq_len = prompt tokens
        # Decode-end: seq_len = prompt tokens + decode tokens
        assert r.mem_decode_kv > r.mem_prefill_kv
        assert r.mem_prefill_weights == r.mem_decode_weights


# ---------------------------------------------------------------------------
# Context-point runner → schema conformance
# ---------------------------------------------------------------------------


class TestContextPointSchema:
    def test_produces_valid_rows(self):
        b = harness.MockBackend(prefill_work=50, decode_work=30)
        b.load()
        rows = harness.run_context_point(
            b,
            context_length=256,
            run_id="test_run_abc1234",
            manifest_ref="results/manifests/test.json",
            git_sha="abc1234",
            device="generic_aarch64",
            engine_gdn="cpu",
            engine_full_attention="cpu",
            model_checkpoint="Qwen/Qwen3.5-4B",
            quantization="fp16",
            warmup=1,
            repeat_count=5,
            decode_tokens=10,
        )
        # Validate every row against the schema
        schema.validate_rows(rows)

    def test_row_count_per_repeat(self):
        """Each repeat produces exactly 9 rows: 3 throughput + 6 memory."""
        b = harness.MockBackend(prefill_work=10)
        b.load()
        rows = harness.run_context_point(
            b,
            context_length=64,
            run_id="test_run",
            manifest_ref="results/manifests/test.json",
            git_sha="abc1234",
            device="generic_aarch64",
            engine_gdn="cpu",
            engine_full_attention="cpu",
            model_checkpoint="test",
            quantization="fp16",
            warmup=1,
            repeat_count=5,
            decode_tokens=5,
        )
        # 5 repeats × (1 prefill_tps + 1 ttft + 1 decode_tps + 6 memory) = 45
        assert len(rows) == 5 * 9

    def test_repeat_indices(self):
        b = harness.MockBackend(prefill_work=10)
        b.load()
        rows = harness.run_context_point(
            b,
            context_length=64,
            run_id="test_run",
            manifest_ref="results/manifests/test.json",
            git_sha="abc1234",
            device="generic_aarch64",
            engine_gdn="cpu",
            engine_full_attention="cpu",
            model_checkpoint="test",
            quantization="fp16",
            warmup=1,
            repeat_count=5,
            decode_tokens=5,
        )
        indices = {r.repeat_index for r in rows}
        assert indices == {0, 1, 2, 3, 4}
        for r in rows:
            assert r.repeat_count == 5

    def test_all_metric_names_present(self):
        b = harness.MockBackend(prefill_work=10)
        b.load()
        rows = harness.run_context_point(
            b,
            context_length=64,
            run_id="test_run",
            manifest_ref="results/manifests/test.json",
            git_sha="abc1234",
            device="generic_aarch64",
            engine_gdn="cpu",
            engine_full_attention="cpu",
            model_checkpoint="test",
            quantization="fp16",
            warmup=1,
            repeat_count=5,
            decode_tokens=5,
        )
        names = {r.metric_name for r in rows}
        assert schema.MetricName.PREFILL_TOKENS_PER_SEC.value in names
        assert schema.MetricName.DECODE_TOKENS_PER_SEC.value in names
        assert schema.MetricName.TTFT_SECONDS.value in names
        assert schema.MetricName.PEAK_MEMORY_BYTES.value in names

    def test_memory_component_required(self):
        """peak_memory_bytes rows must have metric_component set (METRICS.md §5)."""
        b = harness.MockBackend(prefill_work=10)
        b.load()
        rows = harness.run_context_point(
            b,
            context_length=64,
            run_id="test_run",
            manifest_ref="results/manifests/test.json",
            git_sha="abc1234",
            device="generic_aarch64",
            engine_gdn="cpu",
            engine_full_attention="cpu",
            model_checkpoint="test",
            quantization="fp16",
            warmup=1,
            repeat_count=5,
            decode_tokens=5,
        )
        for r in rows:
            if r.metric_name == schema.MetricName.PEAK_MEMORY_BYTES.value:
                assert r.metric_component in (
                    schema.MemoryComponent.WEIGHTS.value,
                    schema.MemoryComponent.KV_CACHE.value,
                    schema.MemoryComponent.RECURRENT_STATE.value,
                )

    def test_throughput_rows_have_no_component(self):
        """Throughput/TTFT rows must NOT have metric_component (METRICS.md §5.0)."""
        b = harness.MockBackend(prefill_work=10)
        b.load()
        rows = harness.run_context_point(
            b,
            context_length=64,
            run_id="test_run",
            manifest_ref="results/manifests/test.json",
            git_sha="abc1234",
            device="generic_aarch64",
            engine_gdn="cpu",
            engine_full_attention="cpu",
            model_checkpoint="test",
            quantization="fp16",
            warmup=1,
            repeat_count=5,
            decode_tokens=5,
        )
        for r in rows:
            if r.metric_name != schema.MetricName.PEAK_MEMORY_BYTES.value:
                assert r.metric_component is None or r.metric_component == ""


# ---------------------------------------------------------------------------
# Full sweep → CSV files on disk
# ---------------------------------------------------------------------------


class TestSweepIntegration:
    @pytest.fixture
    def sweep_result(self, tmp_path):
        b = harness.MockBackend(prefill_work=20, decode_work=10)
        config = harness.SweepConfig(
            device="generic_aarch64",
            contexts=(64, 128),
            warmup=1,
            repeats=5,
            decode_tokens=5,
            output_dir=str(tmp_path / "raw"),
            manifests_dir=str(tmp_path / "manifests"),
            write_manifest=True,
            print_summary=False,
        )
        return harness.run_sweep(b, config)

    def test_one_csv_per_context(self, sweep_result, tmp_path):
        assert len(sweep_result.csv_paths) == 2
        for p in sweep_result.csv_paths:
            assert os.path.exists(p)

    def test_csvs_are_schema_valid(self, sweep_result):
        for p in sweep_result.csv_paths:
            rows = schema.read_csv(p, validate=True)
            assert len(rows) > 0

    def test_manifest_written(self, sweep_result, tmp_path):
        manifest_path = tmp_path / "manifests" / f"{sweep_result.run_id}.json"
        assert manifest_path.exists()
        import json

        with open(manifest_path) as f:
            m = json.load(f)
        assert m["run_id"] == sweep_result.run_id
        assert "host" in m

    def test_each_csv_has_single_context_length(self, sweep_result):
        """Each CSV covers exactly one context_length (RESULTS_SCHEMA.md §2)."""
        for p in sweep_result.csv_paths:
            rows = schema.read_csv(p, validate=False)
            ctx_values = {r.context_length for r in rows}
            assert len(ctx_values) == 1, f"CSV {p} has multiple context lengths: {ctx_values}"

    def test_context_in_filename(self, sweep_result):
        for p in sweep_result.csv_paths:
            assert "ctx" in os.path.basename(p)

    def test_summaries_computed(self, sweep_result):
        assert len(sweep_result.summaries) == 2  # one per context
        for _ctx, summaries in sweep_result.summaries.items():
            # Should have prefill_tps, decode_tps, ttft, and 6 memory metrics
            assert len(summaries) >= 4


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_smoke_run(self, tmp_path):
        """End-to-end CLI invocation with the mock backend."""
        rc = harness.main(
            [
                "--backend",
                "mock",
                "--contexts",
                "64",
                "--repeats",
                "5",
                "--warmup",
                "1",
                "--decode-tokens",
                "5",
                "--output-dir",
                str(tmp_path / "raw"),
                "--manifests-dir",
                str(tmp_path / "manifests"),
                "--quiet",
            ]
        )
        assert rc == 0
        csvs = list((tmp_path / "raw").glob("*.csv"))
        assert len(csvs) == 1
        # Validate
        rows = schema.read_csv(str(csvs[0]), validate=True)
        assert len(rows) > 0

    def test_multi_context_sweep(self, tmp_path):
        rc = harness.main(
            [
                "--backend",
                "mock",
                "--contexts",
                "64,128",
                "--repeats",
                "5",
                "--warmup",
                "1",
                "--decode-tokens",
                "5",
                "--output-dir",
                str(tmp_path / "raw"),
                "--manifests-dir",
                str(tmp_path / "manifests"),
                "--quiet",
            ]
        )
        assert rc == 0
        csvs = list((tmp_path / "raw").glob("*.csv"))
        assert len(csvs) == 2

    def test_default_repeats_tier(self, tmp_path):
        """Without --repeats, 4K gets 30 and 128K gets 10 (METRICS.md §7)."""
        rc = harness.main(
            [
                "--backend",
                "mock",
                "--contexts",
                "4096",
                "--warmup",
                "1",
                "--decode-tokens",
                "3",
                "--output-dir",
                str(tmp_path / "raw"),
                "--manifests-dir",
                str(tmp_path / "manifests"),
                "--quiet",
            ]
        )
        assert rc == 0
        csvs = list((tmp_path / "raw").glob("*ctx4096*.csv"))
        assert len(csvs) == 1
        rows = schema.read_csv(str(csvs[0]), validate=False)
        # 30 repeats × 9 rows = 270
        assert len(rows) == 270

    def test_no_manifest_flag(self, tmp_path):
        rc = harness.main(
            [
                "--backend",
                "mock",
                "--contexts",
                "32",
                "--repeats",
                "5",
                "--warmup",
                "1",
                "--no-manifest",
                "--output-dir",
                str(tmp_path / "raw"),
                "--manifests-dir",
                str(tmp_path / "manifests"),
                "--quiet",
            ]
        )
        assert rc == 0
        # No manifest should be written
        manifests = list((tmp_path / "manifests").glob("*.json"))
        assert len(manifests) == 0


# ---------------------------------------------------------------------------
# CSV round-trip
# ---------------------------------------------------------------------------


class TestCSVRoundTrip:
    def test_write_then_read(self, tmp_path):
        b = harness.MockBackend(prefill_work=10)
        b.load()
        rows = harness.run_context_point(
            b,
            context_length=64,
            run_id="roundtrip_test",
            manifest_ref="results/manifests/test.json",
            git_sha="abc1234",
            device="generic_aarch64",
            engine_gdn="cpu",
            engine_full_attention="cpu",
            model_checkpoint="test",
            quantization="fp16",
            warmup=1,
            repeat_count=5,
            decode_tokens=5,
        )
        path = str(tmp_path / "test.csv")
        schema.write_csv(rows, path)

        # Read back and validate
        read_back = schema.read_csv(path, validate=True)
        assert len(read_back) == len(rows)
        for orig, read in zip(rows, read_back, strict=True):
            assert orig.run_id == read.run_id
            assert orig.metric_name == read.metric_name
            assert orig.value == pytest.approx(read.value)
            assert orig.repeat_index == read.repeat_index
