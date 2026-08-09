# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

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
    _busy_sleep,
    _fmt_bytes,
    _fmt_value,
    _rows_from_timing,
    compute_summaries,
    generate_prompt,
    load_config_from_dict,
    load_config_from_hub,
    load_corpus_prompt,
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
            * cfg.num_key_value_heads
            * cfg.full_attn_head_dim
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
                "--allow-missing-sha",
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
                "--allow-missing-sha",
            ]
        )
        assert rc == 0
        csvs = list((tmp_path / "results" / "raw").glob("*.csv"))
        assert len(csvs) == 0

    def test_refuses_unattributable_run_by_default(self, tmp_path, monkeypatch):
        """Outside a git repo the sweep must refuse rather than stamp a fake SHA.

        The frozen schema validates git_sha as 7-40 hex, so a placeholder like
        "0000000" would validate clean and produce a CSV that looks publishable
        with no provenance at all (docs/archive/PLAN.md section 9).
        """
        monkeypatch.chdir(tmp_path)
        with pytest.raises(RuntimeError, match="un-attributable"):
            main(
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

    def test_allow_missing_sha_stamps_every_row(self, tmp_path, monkeypatch):
        """With the override the run proceeds, but the caveat rides in the CSV."""
        monkeypatch.chdir(tmp_path)
        cfg = SweepConfig(
            context_lengths=[64],
            warmup_count=1,
            repeat_count=5,
            decode_length=10,
            allow_missing_sha=True,
            notes="pre-existing note",
        )
        rows = run_sweep(SyntheticBackend(QWEN35_4B), cfg)
        assert rows
        assert all("UNATTRIBUTABLE" in r.notes for r in rows)
        # The operator's own note must survive alongside the marker.
        assert all("pre-existing note" in r.notes for r in rows)

    def test_cli_choices_are_derived_from_the_frozen_schema(self):
        """The CLI's device/engine choices must not drift from schema.py.

        These were hardcoded literal lists duplicating the schema enums. They
        agreed, but nothing enforced it, so a schema change would have left the
        CLI accepting a value that validate_rows rejects only after a full sweep.
        """
        from bench.harness import _DEVICE_CHOICES, _ENGINE_CHOICES
        from bench.schema import Device, Engine

        assert set(_DEVICE_CHOICES) == {d.value for d in Device}
        assert set(_ENGINE_CHOICES) == {e.value for e in Engine}

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
        assert cfg.num_key_value_heads == 4
        assert cfg.full_attn_head_dim == 256

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


class TestLoadConfigFromHub:
    """Tests for load_config_from_hub (network function, mocked)."""

    def test_fetches_and_parses_config(self, monkeypatch):
        """load_config_from_hub fetches config.json from HF and builds a ModelConfig."""
        import io
        import json

        fake_config = {
            "text_config": {
                "layer_types": ["linear_attention"] * 2 + ["full_attention"],
                "linear_num_value_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
            }
        }
        fake_resp = io.BytesIO(json.dumps(fake_config).encode())

        captured = {}

        def fake_urlopen(url, timeout=10):
            captured["url"] = url
            captured["timeout"] = timeout
            return fake_resp

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        cfg = load_config_from_hub("test-org/test-model", timeout=15)
        assert cfg.num_gdn_layers == 2
        assert cfg.num_full_attention_layers == 1
        assert cfg.linear_num_value_heads == 16
        assert "test-org/test-model" in captured["url"]
        assert captured["timeout"] == 15

    def test_network_error_propagates(self, monkeypatch):
        """URLError from the network propagates as-is."""
        import urllib.request

        def fake_urlopen(url, timeout=10):
            raise urllib.error.URLError("simulated network failure")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        with pytest.raises(urllib.error.URLError):
            load_config_from_hub("test-org/test-model")


class TestSweepConfigValidation:
    """SweepConfig rejects schema-invalid values at construction.

    The CLI constrains these through argparse, but callers that build a config
    directly — scripts/run_ablation.py, bench/hf_backend.py, tests — bypassed
    that. Without validation here the first symptom of a typo'd device is
    validate_rows raising *after* the sweep, which at 262K context is expensive.
    """

    def _cfg(self, **overrides):
        kwargs = dict(context_lengths=[64], repeat_count=5)
        kwargs.update(overrides)
        return SweepConfig(**kwargs)

    def test_valid_config_constructs(self):
        cfg = self._cfg()
        assert cfg.device in {
            d.value for d in __import__("bench.schema", fromlist=["Device"]).Device
        }

    def test_rejects_unknown_device(self):
        with pytest.raises(ValueError, match="device must be one of"):
            self._cfg(device="raspberry-pi-9")

    def test_rejects_unknown_engine_gdn(self):
        with pytest.raises(ValueError, match="engine_gdn must be one of"):
            self._cfg(engine_gdn="tpu")

    def test_rejects_unknown_engine_full_attention(self):
        with pytest.raises(ValueError, match="engine_full_attention must be one of"):
            self._cfg(engine_full_attention="tpu")

    def test_rejects_repeat_count_below_five(self):
        with pytest.raises(ValueError, match="never report N < 5"):
            self._cfg(repeat_count=4)

    def test_accepts_every_schema_enum_value(self):
        """Whatever the schema allows, the config must allow — no second list."""
        from bench.schema import Device, Engine

        for d in Device:
            assert self._cfg(device=d.value).device == d.value
        for e in Engine:
            assert self._cfg(engine_gdn=e.value).engine_gdn == e.value


# ---------------------------------------------------------------------------
# _fmt_bytes + _fmt_value + _busy_sleep
# ---------------------------------------------------------------------------


class TestFmtBytes:
    def test_bytes(self):
        assert _fmt_bytes(512) == "512.0 B"

    def test_kib(self):
        assert _fmt_bytes(2048) == "2.0 KiB"

    def test_gib(self):
        assert _fmt_bytes(3 * 1024**3) == "3.0 GiB"

    def test_pib_overflow(self):
        assert _fmt_bytes(1024**5) == "1.0 PiB"


class TestFmtValue:
    def test_per_sec_format(self):
        result = _fmt_value("tokens_per_sec", 800.5)
        assert "800.50" in result

    def test_ttft_format(self):
        result = _fmt_value("ttft_seconds", 0.025)
        assert "25.00ms" in result

    def test_memory_format(self):
        result = _fmt_value("peak_memory_bytes", 2048)
        assert "KiB" in result

    def test_generic_format(self):
        result = _fmt_value("some_metric", 3.14159)
        assert "3.142" in result


class TestBusySleep:
    def test_returns_after_elapsed(self):
        import time

        start = time.perf_counter_ns()
        _busy_sleep(2_000_000)  # 2ms
        elapsed = time.perf_counter_ns() - start
        assert elapsed >= 1_500_000  # at least ~2ms (allow scheduling slack)

    def test_zero_is_instant(self):
        import time

        start = time.perf_counter_ns()
        _busy_sleep(0)
        elapsed = time.perf_counter_ns() - start
        assert elapsed < 1_000_000  # less than 1ms


class TestSyntheticBackendTiming:
    """Test prefill/decode with simulated timing."""

    def test_prefill_with_timing(self):
        b = SyntheticBackend(QWEN35_4B, prefill_ns_per_token=1000)
        b.prefill(list(range(50)))
        assert b._seq_len == 50

    def test_decode_with_timing(self):
        b = SyntheticBackend(QWEN35_4B, decode_ns_per_step=1000)
        b.prefill(list(range(10)))
        token = b.decode_step(5)
        assert token == 6
        assert b._seq_len == 11


class TestLoadCorpusPrompt:
    """Cover load_corpus_prompt fallback when file missing (line 377)."""

    def test_missing_file_falls_back_to_generate(self, tmp_path):
        """When prompt file doesn't exist, falls back to generate_prompt."""
        result = load_corpus_prompt("needle", 4096, str(tmp_path))
        assert len(result) > 0  # generate_prompt returns non-empty text

    def test_existing_file_loaded(self, tmp_path):
        """When prompt file exists, its content is returned."""
        prompt_file = tmp_path / "needle_4096.txt"
        prompt_file.write_text("This is a test prompt.\n")
        result = load_corpus_prompt("needle", 4096, str(tmp_path))
        assert result == "This is a test prompt."


class TestSweepExceptionHandling:
    """Cover the sweep exception handler (lines 655-656)."""

    def test_failing_context_length_continues(self, capsys):
        """A failing context_length is skipped, others still produce rows."""
        from unittest.mock import patch

        original_prefill = SyntheticBackend.prefill
        call_count = [0]

        def failing_prefill(self, input_ids):
            call_count[0] += 1
            if call_count[0] > 3:
                raise RuntimeError("simulated failure")
            return original_prefill(self, input_ids)

        b = SyntheticBackend(QWEN35_4B)
        config = SweepConfig(
            context_lengths=[64, 128],
            warmup_count=0,
            repeat_count=5,
            decode_length=2,
        )
        with patch.object(SyntheticBackend, "prefill", failing_prefill):
            rows = run_sweep(b, config)
        # First context_length succeeds, second fails
        assert len(rows) > 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "failed" in captured.err


class TestHFBackendImportError:
    """Cover the --backend hf ImportError fallback (lines 893-898).

    When the HuggingFace backend cannot instantiate (torch/transformers
    missing), the harness must give a clear message via parser.error
    rather than an unhandled traceback.
    """

    def test_hf_backend_import_error_gives_clear_message(self, monkeypatch):
        """main() with --backend hf + missing torch → parser.error (exit 2)."""

        def _raise_import_error(*_args, **_kwargs):
            raise ImportError("simulated: torch not installed")

        # HFTorchBackend is imported lazily inside main(), so patching the
        # attribute on the module is sufficient.
        monkeypatch.setattr("bench.hf_backend.HFTorchBackend", _raise_import_error)

        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--backend",
                    "hf",
                    "--context-lengths",
                    "64",
                    "--warmup",
                    "1",
                    "--repeats",
                    "5",
                    "--decode-length",
                    "10",
                    "--allow-missing-sha",
                ]
            )
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Coverage gap tests (ob-82d)
# ---------------------------------------------------------------------------


class TestCLIBadArgs:
    def test_bad_context_lengths(self, tmp_path, monkeypatch):
        """Non-integer context-lengths → parser.error → SystemExit(2)."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--backend", "synthetic",
                "--model", "0.8b",
                "--context-lengths", "64,abc",
                "--warmup", "1",
                "--repeats", "5",
                "--allow-missing-sha",
            ])
        assert exc_info.value.code == 2

    def test_unknown_backend(self, tmp_path, monkeypatch):
        """Unknown backend → parser.error → SystemExit(2)."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--backend", "quantum",
                "--context-lengths", "64",
                "--warmup", "1",
                "--repeats", "5",
                "--allow-missing-sha",
            ])
        assert exc_info.value.code == 2


class TestSweepNeedlePrompt:
    def test_needle_prompt_type_runs(self, tmp_path, monkeypatch):
        """SweepConfig with prompt_type='needle' exercises load_corpus_prompt path."""
        monkeypatch.chdir(tmp_path)
        # No corpus file exists → load_corpus_prompt falls back to generate_prompt
        config = SweepConfig(
            context_lengths=[64],
            warmup_count=1,
            repeat_count=5,
            decode_length=10,
            prompt_type="needle",
        )
        backend = SyntheticBackend(QWEN35_4B)
        config.allow_missing_sha = True
        rows = run_sweep(backend, config)
        assert len(rows) > 0


class TestMainEntryRunpy:
    def test_main_via_runpy(self, tmp_path, monkeypatch):
        """Running harness.py as __main__ covers the __main__ guard."""
        import runpy

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", [
            "harness.py",
            "--backend", "synthetic",
            "--model", "0.8b",
            "--context-lengths", "64",
            "--warmup", "1",
            "--repeats", "5",
            "--decode-length", "10",
            "--allow-missing-sha",
        ])
        script_path = str(Path(__file__).resolve().parent.parent / "bench" / "harness.py")
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(script_path, run_name="__main__")
        assert exc_info.value.code == 0
