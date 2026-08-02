"""Tests for the benchmark harness (ob-ljh).

Exercises the timing protocol (METRICS.md sections 1-5), schema conformance
(RESULTS_SCHEMA.md), statistical protocol (section 7), and CLI. Uses only the
SyntheticBackend so these run anywhere without model weights.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Ensure repo root is importable when pytest hasn't already done so.
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.harness import (  # noqa: E402
    QWEN35_4B,
    RepeatTimings,
    SweepConfig,
    SyntheticBackend,
    _rows_from_timing,
    compute_summaries,
    generate_prompt,
    load_config_from_dict,
    main,
    run_one_repeat,
    run_sweep,
)
from bench.metrics import percentile, summarize  # noqa: E402
from bench.schema import read_csv, validate_rows  # noqa: E402

# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_p50_odd(self):
        assert percentile([3, 1, 2], 50) == 2

    def test_p50_even(self):
        # nearest-rank: ceil(50/100 * 4) = rank 2, value = sorted[1]
        assert percentile([4, 1, 3, 2], 50) == 2

    def test_p95(self):
        # 10 values, p95: ceil(95/100 * 10) = ceil(9.5) = rank 10
        vals = list(range(1, 11))
        assert percentile(vals, 95) == 10

    def test_p95_clamped(self):
        # With 3 values, rank = ceil(2.85) = 3, clamped to 3
        assert percentile([1, 2, 3], 95) == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            percentile([], 50)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            percentile([1, 2, 3], 101)

    def test_single_value(self):
        assert percentile([42], 50) == 42
        assert percentile([42], 95) == 42


class TestSummarize:
    def test_basic(self):
        s = summarize([10, 20, 30, 40, 50])
        assert s.n == 5
        assert s.p50 == 30
        assert s.spread == s.p95 - s.p50

    def test_normalized_spread(self):
        s = summarize([10, 20, 30, 40, 50])
        assert s.normalized_spread == pytest.approx((s.p95 - s.p50) / s.p50)

    def test_zero_p50(self):
        s = summarize([0, 0, 0])
        assert s.normalized_spread == math.inf

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            summarize([])


# ---------------------------------------------------------------------------
# SyntheticBackend
# ---------------------------------------------------------------------------


class TestSyntheticBackend:
    def test_tokenize_proportional_to_text_length(self):
        b = SyntheticBackend(QWEN35_4B)
        text = "x" * 400
        ids = b.tokenize(text)
        assert len(ids) == 100  # ~4 chars per token

    def test_tokenize_minimum_one(self):
        b = SyntheticBackend(QWEN35_4B)
        ids = b.tokenize("ab")
        assert len(ids) == 1

    def test_prefill_sets_seq_len(self):
        b = SyntheticBackend(QWEN35_4B)
        b.prefill(list(range(100)))
        mem = b.memory_bytes()
        assert mem["recurrent_state"] > 0

    def test_decode_increments_seq_len(self):
        b = SyntheticBackend(QWEN35_4B)
        b.prefill(list(range(100)))
        mem_before = b.memory_bytes()["kv_cache"]
        b.decode_step(0)
        b.decode_step(0)
        mem_after = b.memory_bytes()["kv_cache"]
        assert mem_after > mem_before  # kv_cache grows

    def test_reset_clears_state(self):
        b = SyntheticBackend(QWEN35_4B)
        b.prefill(list(range(100)))
        b.decode_step(0)
        b.reset()
        assert b.memory_bytes()["kv_cache"] == 0

    def test_memory_weights_constant_across_context(self):
        b = SyntheticBackend(QWEN35_4B)
        b.prefill(list(range(100)))
        w1 = b.memory_bytes()["weights"]
        b.reset()
        b.prefill(list(range(1000)))
        w2 = b.memory_bytes()["weights"]
        assert w1 == w2  # weights are flat (METRICS.md section 5.2)

    def test_memory_recurrent_state_constant_across_context(self):
        b = SyntheticBackend(QWEN35_4B)
        b.prefill(list(range(100)))
        rs1 = b.memory_bytes()["recurrent_state"]
        b.reset()
        b.prefill(list(range(1000)))
        rs2 = b.memory_bytes()["recurrent_state"]
        assert rs1 == rs2  # O(1) per token (METRICS.md section 5.4)

    def test_memory_values_match_formula(self):
        cfg = QWEN35_4B
        b = SyntheticBackend(cfg)
        b.prefill(list(range(256)))
        mem = b.memory_bytes()
        expected_rs = (
            cfg.num_gdn_layers
            * cfg.linear_num_value_heads
            * cfg.linear_key_head_dim
            * cfg.linear_value_head_dim
            * cfg.state_dtype_bytes
        )
        assert mem["recurrent_state"] == expected_rs
        expected_kv = (
            cfg.num_full_attention_layers
            * 2
            * 256
            * cfg.fa_n_kv_heads
            * cfg.fa_head_dim
            * cfg.cache_dtype_bytes
        )
        assert mem["kv_cache"] == expected_kv

    def test_sample_deterministic(self):
        b = SyntheticBackend(QWEN35_4B)
        assert b.sample(None) == 42

    def test_decode_step_advances(self):
        b = SyntheticBackend(QWEN35_4B)
        t1 = b.decode_step(42)
        assert t1 == 43


# ---------------------------------------------------------------------------
# Timing protocol (METRICS.md sections 1-5)
# ---------------------------------------------------------------------------


class TestTimingProtocol:
    def test_token1_belongs_to_prefill(self):
        """METRICS.md section 1: token 1 is counted toward prefill, not decode."""
        b = SyntheticBackend(QWEN35_4B)
        text = generate_prompt(64)
        t = run_one_repeat(b, text, decode_length=10)
        # decode_token_count = N - 1 (token 1 excluded)
        assert t.decode_token_count == 9

    def test_prompt_token_count_from_tokenizer(self):
        """METRICS.md section 2: numerator is len(input_ids), not context_length."""
        b = SyntheticBackend(QWEN35_4B)
        text = generate_prompt(64)
        t = run_one_repeat(b, text, decode_length=10)
        assert t.prompt_token_count == len(b.tokenize(text))

    def test_all_durations_positive(self):
        b = SyntheticBackend(QWEN35_4B)
        text = generate_prompt(64)
        t = run_one_repeat(b, text, decode_length=10)
        assert t.prefill_duration >= 0
        assert t.ttft_duration >= 0
        assert t.decode_duration >= 0

    def test_memory_sampled_at_correct_phases(self):
        """METRICS.md section 5.1: prefill memory at t_prefill_logits, decode at t_N."""
        b = SyntheticBackend(QWEN35_4B)
        text = generate_prompt(64)
        t = run_one_repeat(b, text, decode_length=10)
        # kv_cache should be larger at decode end (prompt + decode tokens)
        assert t.mem_decode["kv_cache"] > t.mem_prefill["kv_cache"]
        # recurrent_state is O(1) — same at both phases
        assert t.mem_decode["recurrent_state"] == t.mem_prefill["recurrent_state"]
        # weights are flat
        assert t.mem_decode["weights"] == t.mem_prefill["weights"]


# ---------------------------------------------------------------------------
# Row generation + schema validation
# ---------------------------------------------------------------------------


class TestRowGeneration:
    @staticmethod
    def _make_timing():
        return RepeatTimings(
            prompt_token_count=4096,
            prefill_duration=0.5,
            ttft_duration=0.55,
            decode_duration=1.0,
            decode_token_count=256,
            mem_prefill={"weights": 8e9, "kv_cache": 1e6, "recurrent_state": 5e5},
            mem_decode={"weights": 8e9, "kv_cache": 2e6, "recurrent_state": 5e5},
        )

    def test_nine_rows_per_repeat(self):
        """3 throughput/latency + 6 memory (3 components x 2 phases)."""
        config = SweepConfig(context_lengths=[4096])
        rows = _rows_from_timing(
            self._make_timing(),
            run_id="test_run",
            git_sha="abc1234",
            manifest_ref_str="results/manifests/test_run.json",
            config=config,
            context_length=4096,
            repeat_idx=0,
        )
        assert len(rows) == 9

    def test_all_rows_validate(self):
        config = SweepConfig(context_lengths=[4096])
        rows = _rows_from_timing(
            self._make_timing(),
            run_id="test_run",
            git_sha="abc1234",
            manifest_ref_str="results/manifests/test_run.json",
            config=config,
            context_length=4096,
            repeat_idx=0,
        )
        validate_rows(rows)  # raises if invalid

    def test_prefill_throughput_value(self):
        config = SweepConfig(context_lengths=[4096])
        rows = _rows_from_timing(
            self._make_timing(),
            run_id="test_run",
            git_sha="abc1234",
            manifest_ref_str="results/manifests/test_run.json",
            config=config,
            context_length=4096,
            repeat_idx=0,
        )
        prefill_row = [r for r in rows if r.metric_name == "prefill_tokens_per_sec"][0]
        assert prefill_row.value == pytest.approx(4096 / 0.5)

    def test_decode_throughput_value(self):
        config = SweepConfig(context_lengths=[4096])
        rows = _rows_from_timing(
            self._make_timing(),
            run_id="test_run",
            git_sha="abc1234",
            manifest_ref_str="results/manifests/test_run.json",
            config=config,
            context_length=4096,
            repeat_idx=0,
        )
        decode_row = [r for r in rows if r.metric_name == "decode_tokens_per_sec"][0]
        assert decode_row.value == pytest.approx(256 / 1.0)

    def test_memory_component_required_for_memory_metric(self):
        config = SweepConfig(context_lengths=[4096])
        rows = _rows_from_timing(
            self._make_timing(),
            run_id="test_run",
            git_sha="abc1234",
            manifest_ref_str="results/manifests/test_run.json",
            config=config,
            context_length=4096,
            repeat_idx=0,
        )
        mem_rows = [r for r in rows if r.metric_name == "peak_memory_bytes"]
        for r in mem_rows:
            assert r.metric_component in ("weights", "kv_cache", "recurrent_state")

    def test_throughput_metrics_have_no_component(self):
        config = SweepConfig(context_lengths=[4096])
        rows = _rows_from_timing(
            self._make_timing(),
            run_id="test_run",
            git_sha="abc1234",
            manifest_ref_str="results/manifests/test_run.json",
            config=config,
            context_length=4096,
            repeat_idx=0,
        )
        for r in rows:
            if r.metric_name != "peak_memory_bytes":
                assert not r.metric_component


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------


class TestSweep:
    def test_basic_sweep_validates(self):
        b = SyntheticBackend(QWEN35_4B)
        config = SweepConfig(
            context_lengths=[64, 128],
            warmup_count=2,
            repeat_count=5,
            decode_length=10,
        )
        rows = run_sweep(b, config)
        validate_rows(rows)  # raises if any invalid
        # 2 context lengths x 5 repeats x 9 rows = 90
        assert len(rows) == 90

    def test_warmup_not_in_output(self):
        """METRICS.md section 7: warmup repeats are discarded."""
        b = SyntheticBackend(QWEN35_4B)
        config = SweepConfig(
            context_lengths=[64],
            warmup_count=3,
            repeat_count=5,
            decode_length=10,
        )
        rows = run_sweep(b, config)
        # Only repeat_count (5) repeats should appear, not warmup + measured (8)
        repeat_indices = {r.repeat_index for r in rows}
        assert repeat_indices == {0, 1, 2, 3, 4}

    def test_two_context_lengths_distinct(self):
        b = SyntheticBackend(QWEN35_4B)
        config = SweepConfig(
            context_lengths=[64, 128],
            warmup_count=1,
            repeat_count=5,
            decode_length=10,
        )
        rows = run_sweep(b, config)
        ctx_values = {r.context_length for r in rows}
        assert ctx_values == {64, 128}

    def test_summaries_grouped_correctly(self):
        b = SyntheticBackend(QWEN35_4B)
        config = SweepConfig(
            context_lengths=[64],
            warmup_count=1,
            repeat_count=5,
            decode_length=10,
        )
        rows = run_sweep(b, config)
        summaries = compute_summaries(rows)
        # Groups: 2 ctx x (3 throughput + 3 mem_prefill + 3 mem_decode) = 9 groups
        assert len(summaries) == 9
        for ms in summaries:
            assert ms.summary.n == 5

    def test_run_id_format(self):
        """RESULTS_SCHEMA.md section 2: <device>_<timestamp>_<sha>."""
        b = SyntheticBackend(QWEN35_4B)
        config = SweepConfig(context_lengths=[64], device="generic_aarch64")
        rows = run_sweep(b, config)
        run_id = rows[0].run_id
        assert run_id.startswith("generic_aarch64_")

    def test_all_rows_share_manifest_ref(self):
        b = SyntheticBackend(QWEN35_4B)
        config = SweepConfig(context_lengths=[64])
        rows = run_sweep(b, config)
        refs = {r.manifest_ref for r in rows}
        assert len(refs) == 1  # all rows from one run share one manifest


# ---------------------------------------------------------------------------
# CSV round-trip
# ---------------------------------------------------------------------------


class TestCsvRoundTrip:
    def test_write_and_read_back(self, tmp_path):
        from bench.schema import write_csv

        b = SyntheticBackend(QWEN35_4B)
        config = SweepConfig(
            context_lengths=[64],
            warmup_count=1,
            repeat_count=5,
            decode_length=10,
        )
        rows = run_sweep(b, config)
        csv_path = str(tmp_path / "test.csv")
        write_csv(rows, csv_path)
        read_back = read_csv(csv_path)

        assert len(read_back) == len(rows)
        for original, read_row in zip(rows, read_back, strict=True):
            assert original.run_id == read_row.run_id
            assert original.context_length == read_row.context_length
            assert original.metric_name == read_row.metric_name
            assert original.value == pytest.approx(read_row.value)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_minimal_invocation(self, tmp_path, monkeypatch):
        """End-to-end: harness CLI writes valid CSV + manifest."""
        monkeypatch.chdir(tmp_path)
        rc = main(
            [
                "--backend",
                "synthetic",
                "--model",
                "0.8b",
                "--context-lengths",
                "64,128",
                "--warmup",
                "1",
                "--repeats",
                "5",
                "--decode-length",
                "10",
                "--device",
                "generic_aarch64",
            ]
        )
        assert rc == 0
        # CSV should exist
        csvs = list((tmp_path / "results" / "raw").glob("*.csv"))
        assert len(csvs) == 1
        # Manifest should exist
        manifests = list((tmp_path / "results" / "manifests").glob("*.json"))
        assert len(manifests) == 1
        # CSV should validate on read-back
        rows = read_csv(str(csvs[0]))
        validate_rows(rows)

    def test_no_csv_flag(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rc = main(
            [
                "--backend",
                "synthetic",
                "--context-lengths",
                "64",
                "--warmup",
                "1",
                "--repeats",
                "5",
                "--decode-length",
                "10",
                "--no-csv",
            ]
        )
        assert rc == 0
        csvs = list((tmp_path / "results" / "raw").glob("*.csv"))
        assert len(csvs) == 0

    def test_rejects_below_minimum_repeats(self):
        with pytest.raises(SystemExit):
            main(
                [
                    "--context-lengths",
                    "64",
                    "--repeats",
                    "3",  # below minimum of 5
                    "--no-csv",
                ]
            )

    def test_help_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


class TestLoadConfigFromDict:
    """Tests for load_config_from_dict (ob-xh3.2)."""

    def test_extracts_gdn_and_fa_counts(self):
        cfg = load_config_from_dict(
            {
                "text_config": {"layer_types": ["linear_attention"] * 3 + ["full_attention"]},
            }
        )
        assert cfg.num_gdn_layers == 3
        assert cfg.num_full_attention_layers == 1

    def test_extracts_linear_dimensions(self):
        cfg = load_config_from_dict(
            {
                "text_config": {
                    "layer_types": ["linear_attention", "full_attention"],
                    "linear_num_value_heads": 32,
                    "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128,
                },
            }
        )
        assert cfg.linear_num_value_heads == 32
        assert cfg.linear_key_head_dim == 128

    def test_extracts_fa_dimensions(self):
        cfg = load_config_from_dict(
            {
                "text_config": {
                    "layer_types": ["full_attention"],
                    "num_key_value_heads": 4,
                    "head_dim": 256,
                },
            }
        )
        assert cfg.fa_n_kv_heads == 4
        assert cfg.fa_head_dim == 256

    def test_maps_ssm_dtype(self):
        cfg = load_config_from_dict(
            {
                "text_config": {"layer_types": [], "mamba_ssm_dtype": "float32"},
            }
        )
        assert cfg.state_dtype_bytes == 4

    def test_no_text_config_nest(self):
        """Config without text_config wrapper should still work."""
        cfg = load_config_from_dict(
            {
                "layer_types": ["linear_attention", "full_attention"],
            }
        )
        assert cfg.num_gdn_layers == 1
        assert cfg.num_full_attention_layers == 1
