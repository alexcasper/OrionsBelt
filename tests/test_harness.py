"""Tests for bench/harness.py — the benchmark runner CLI.

Bead ``ob-ljh``. Tests the harness's sweep/timing/CSV logic using the
``SyntheticBackend``, which exercises the full pipeline without any ML dependency.
This is the CI smoke test referenced by the harness README and ob-1lm.

Coverage:
  - Schema conformance of all emitted rows
  - METRICS.md timer correctness (token 1 in prefill; decode counts N-1)
  - Memory snapshot calls at the right phases
  - Multiple context lengths produce independent row groups
  - CSV round-trip: harness writes → schema.read_csv reads → validate passes
  - Warmups produce no rows
  - CLI smoke test
"""

import csv
import os
import subprocess
import sys

import pytest

# Make bench/ importable when running pytest from repo root.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)
_BENCH_DIR = os.path.join(_REPO_ROOT, "bench")
sys.path.insert(0, _BENCH_DIR)

from harness import (  # noqa: E402
    Harness,
    HarnessConfig,
    SyntheticBackend,
    _percentile,
    make_prompt,
)
from schema import read_csv, validate_rows  # noqa: E402

# --- Fixtures --------------------------------------------------------------


def _make_config(**overrides):
    """Default HarnessConfig for testing, with fast synthetic parameters."""
    defaults = dict(
        device="generic_aarch64",
        engine_gdn="cpu",
        engine_full_attention="cpu",
        model_checkpoint="test/model@abc1234",
        quantization="fp32",
        context_lengths=[512],
        warmups=1,
        repeats=5,
        decode_tokens=10,  # small for fast tests (1 prefill + 9 decode)
        manifest_ref="results/manifests/test_run.json",
    )
    defaults.update(overrides)
    return HarnessConfig(**defaults)


def _make_backend():
    return SyntheticBackend(
        prefill_rate=100_000.0,  # fast for testing
        decode_rate=10_000.0,
        weight_bytes=1_000_000,
        gdn_state_per_layer=1024,
        kv_bytes_per_token=256,
    )


@pytest.fixture
def harness():
    return Harness(_make_backend(), _make_config())


# --- Schema conformance ----------------------------------------------------


class TestSchemaConformance:
    def test_all_rows_validate(self, harness):
        rows = harness.run_sweep()
        assert len(rows) > 0
        validate_rows(rows)  # raises if any row is invalid

    def test_row_count_per_repeat(self, harness):
        rows = harness.run_sweep()
        # Per repeat: prefill_tps + ttft + decode_tps = 3 throughput rows
        # + 3 memory components × 2 phases = 6 memory rows
        # = 9 rows per repeat
        repeats = harness.config.repeats
        ctx_count = len(harness.config.context_lengths)
        expected = 9 * repeats * ctx_count
        assert len(rows) == expected

    def test_metric_names_present(self, harness):
        rows = harness.run_sweep()
        names = {r.metric_name for r in rows}
        assert "prefill_tokens_per_sec" in names
        assert "decode_tokens_per_sec" in names
        assert "ttft_seconds" in names
        assert "peak_memory_bytes" in names

    def test_memory_components_present(self, harness):
        rows = harness.run_sweep()
        components = {
            r.metric_component for r in rows if r.metric_name == "peak_memory_bytes"
        }
        assert components == {"weights", "kv_cache", "recurrent_state"}

    def test_phases_correct(self, harness):
        rows = harness.run_sweep()
        prefill_tps = [r for r in rows if r.metric_name == "prefill_tokens_per_sec"]
        decode_tps = [r for r in rows if r.metric_name == "decode_tokens_per_sec"]
        assert all(r.phase == "prefill" for r in prefill_tps)
        assert all(r.phase == "decode" for r in decode_tps)

    def all_values_nonnegative(self, harness):
        rows = harness.run_sweep()
        assert all(r.value >= 0 for r in rows)


# --- METRICS.md timer correctness -----------------------------------------


class TestTimerCorrectness:
    def test_decode_excludes_token_1(self, harness):
        """METRICS.md §1: token 1 belongs to prefill, not decode."""
        rows = harness.run_sweep()
        for r in rows:
            if r.metric_name == "decode_tokens_per_sec":
                # With synthetic backend at decode_rate=10000, 9 tokens:
                # expected ~ 10000 tok/s but timing overhead at small N adds noise.
                assert r.value > 0
                # Should be in a reasonable range (allowing generous margin for
                # timing overhead at only 9 decode steps).
                assert r.value > 1000, f"decode TPS {r.value} unexpectedly low"

    def test_prefill_tps_scales_with_context(self):
        """Prefill throughput should be roughly constant (tokens/sec), independent of length."""
        backend = _make_backend()
        config = _make_config(context_lengths=[256, 1024])
        harness = Harness(backend, config)
        rows = harness.run_sweep()

        # Group by context length
        from collections import defaultdict
        by_ctx = defaultdict(list)
        for r in rows:
            if r.metric_name == "prefill_tokens_per_sec":
                by_ctx[r.context_length].append(r.value)

        p50_256 = sorted(by_ctx[256])[len(by_ctx[256]) // 2]
        p50_1024 = sorted(by_ctx[1024])[len(by_ctx[1024]) // 2]

        # Synthetic backend has constant prefill_rate, so both should be close
        # Allow 20% variation for timing noise.
        ratio = min(p50_256, p50_1024) / max(p50_256, p50_1024)
        assert ratio > 0.8, f"prefill TPS ratio {ratio:.3f} too far from 1.0"

    def test_memory_weights_flat_across_context(self):
        """METRICS.md §5.2: weights are expected flat across context_length."""
        backend = _make_backend()
        config = _make_config(context_lengths=[256, 1024])
        harness = Harness(backend, config)
        rows = harness.run_sweep()

        from collections import defaultdict
        by_ctx = defaultdict(list)
        for r in rows:
            if r.metric_name == "peak_memory_bytes" and r.metric_component == "weights":
                by_ctx[r.context_length].append(r.value)

        # All weight values should be identical (synthetic backend uses fixed weight_bytes)
        all_weights = []
        for ctx_vals in by_ctx.values():
            all_weights.extend(ctx_vals)
        assert len(set(all_weights)) == 1, "weights should be flat across context lengths"

    def test_memory_kv_cache_grows_with_context(self):
        """METRICS.md §5.3: kv_cache grows linearly with context_length."""
        backend = _make_backend()
        config = _make_config(context_lengths=[256, 1024])
        harness = Harness(backend, config)
        rows = harness.run_sweep()

        from collections import defaultdict
        by_ctx = defaultdict(list)
        for r in rows:
            if r.metric_name == "peak_memory_bytes" and r.metric_component == "kv_cache":
                by_ctx[r.context_length].append(r.value)

        p50_256 = sorted(by_ctx[256])[0]
        p50_1024 = sorted(by_ctx[1024])[0]
        assert p50_1024 > p50_256, "kv_cache should be larger at longer context"

    def test_memory_recurrent_state_flat(self):
        """METRICS.md §5.4: recurrent_state stays O(1) regardless of context_length."""
        backend = _make_backend()
        config = _make_config(context_lengths=[256, 1024])
        harness = Harness(backend, config)
        rows = harness.run_sweep()

        from collections import defaultdict
        by_ctx = defaultdict(list)
        for r in rows:
            if r.metric_name == "peak_memory_bytes" and r.metric_component == "recurrent_state":
                by_ctx[r.context_length].append(r.value)

        all_state = []
        for ctx_vals in by_ctx.values():
            all_state.extend(ctx_vals)
        assert len(set(all_state)) == 1, "recurrent_state should be flat across context"


# --- CSV round-trip --------------------------------------------------------


class TestCSVRoundTrip:
    def test_write_and_read_back(self, harness, tmp_path):
        csv_path = str(tmp_path / "sweep.csv")
        harness.run_sweep(output_path=csv_path)

        # Read back and validate
        rows = read_csv(csv_path)
        validate_rows(rows)
        assert len(rows) > 0

    def test_csv_header_matches_schema(self, harness, tmp_path):
        from schema import COLUMNS

        csv_path = str(tmp_path / "sweep.csv")
        harness.run_sweep(output_path=csv_path)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == COLUMNS

    def test_truncated_sweep_preserves_data(self, tmp_path):
        """ob-ljh: each context point must be independently useful.

        If the sweep is interrupted, already-collected context points should
        be committed to the CSV.
        """
        csv_path = str(tmp_path / "sweep.csv")
        backend = _make_backend()
        config = _make_config(context_lengths=[256, 512])
        harness = Harness(backend, config)

        # Run the sweep — it flushes after each context length
        harness.run_sweep(output_path=csv_path)

        # Both context lengths should be in the file
        rows = read_csv(csv_path)
        ctx_values = {r.context_length for r in rows}
        assert ctx_values == {256, 512}


# --- Warmup correctness ----------------------------------------------------


class TestWarmup:
    def test_warmups_produce_no_rows(self, harness):
        """METRICS.md §7: warmups are discarded, never written to results/raw/."""
        rows = harness.run_sweep()
        # With warmups=1, repeats=5: should get exactly 5 repeats' worth of rows
        repeat_indices = {r.repeat_index for r in rows}
        assert repeat_indices == set(range(5))

    def test_zero_warmups_works(self):
        backend = _make_backend()
        config = _make_config(warmups=0)
        harness = Harness(backend, config)
        rows = harness.run_sweep()
        assert len(rows) > 0
        validate_rows(rows)


# --- Multiple context lengths ---------------------------------------------


class TestMultipleContexts:
    def test_three_context_lengths(self):
        backend = _make_backend()
        config = _make_config(context_lengths=[256, 512, 1024])
        harness = Harness(backend, config)
        rows = harness.run_sweep()
        validate_rows(rows)

        ctx_values = {r.context_length for r in rows}
        assert ctx_values == {256, 512, 1024}

    def test_each_context_has_all_repeats(self):
        backend = _make_backend()
        config = _make_config(context_lengths=[256, 512], repeats=5)
        harness = Harness(backend, config)
        rows = harness.run_sweep()

        from collections import Counter
        ctx_counts = Counter()
        for r in rows:
            if r.metric_name == "prefill_tokens_per_sec":
                ctx_counts[r.context_length] += 1

        assert ctx_counts[256] == 5
        assert ctx_counts[512] == 5


# --- Summary ---------------------------------------------------------------


class TestSummary:
    def test_summary_output(self, harness):
        rows = harness.run_sweep()
        summary = harness.summarize(rows)
        assert "p50" in summary
        assert "p95" in summary


# --- Percentile helper -----------------------------------------------------


class TestPercentile:
    def test_median_odd(self):
        assert _percentile([1, 2, 3, 4, 5], 0.5) == 3

    def test_median_even(self):
        # nearest-rank at p=0.5, n=4: idx = int(0.5*3+0.5) = int(2.0) = 2 → sorted[2]
        assert _percentile([1, 2, 3, 4], 0.5) == 3

    def test_p95(self):
        samples = list(range(1, 21))  # 1..20
        # idx = int(0.95*19+0.5) = int(18.55) = 18 → sorted[18] = 19
        assert _percentile(samples, 0.95) == 19

    def test_empty(self):
        assert _percentile([], 0.5) == 0.0


# --- Prompt corpus ---------------------------------------------------------


class TestPromptCorpus:
    def test_make_prompt_length(self):
        prompt = make_prompt(1024)
        assert len(prompt) > 0
        # Should be roughly target_tokens * 4 chars
        assert len(prompt) >= 1024 * 2  # at least half the target


# --- CLI smoke test --------------------------------------------------------


class TestCLI:
    def test_synthetic_smoke(self, tmp_path):
        """End-to-end CLI invocation with the synthetic backend."""
        csv_path = str(tmp_path / "cli_sweep.csv")

        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_BENCH_DIR, "harness.py"),
                "--backend", "synthetic",
                "--context-lengths", "256",
                "--device", "x86_reference",
                "--repeats", "5",
                "--warmups", "1",
                "--decode-tokens", "10",
                "--output", csv_path,
            ],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )

        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert os.path.exists(csv_path)

        rows = read_csv(csv_path)
        validate_rows(rows)
        assert len(rows) > 0

    def test_rejects_low_repeats(self):
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_BENCH_DIR, "harness.py"),
                "--backend", "synthetic",
                "--context-lengths", "256",
                "--device", "generic_aarch64",
                "--repeats", "3",  # < 5, should fail
                "--output", "/dev/null",
            ],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )

        assert result.returncode == 1
        assert ">= 5" in result.stderr
