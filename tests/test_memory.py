# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Tests for memory instrumentation (ob-vfp).

Verifies the three-way split: weights flat, KV cache linear, state flat.
"""

import pytest

from orionsbelt.engines.memory import (
    MemoryBreakdown,
    _rss_bytes,
    estimate_weights,
    format_breakdown_table,
    measure_delta,
    predict_breakdown,
    sweep_context,
)
from orionsbelt.model.gdn_layer_info import LAYER_INFO


class TestWeightsFlat:
    """Weights must not change with context length."""

    def test_weights_constant_across_context(self):
        b4k = predict_breakdown("4B", 4096)
        b32k = predict_breakdown("4B", 32768)
        b262k = predict_breakdown("4B", 262144)
        assert b4k.weights_bytes == b32k.weights_bytes == b262k.weights_bytes


class TestKVCacheLinear:
    """KV cache must grow linearly with context length."""

    def test_kv_grows_8x_from_32k_to_262k(self):
        b32k = predict_breakdown("4B", 32768)
        b262k = predict_breakdown("4B", 262144)
        ratio = b262k.kv_cache_bytes / b32k.kv_cache_bytes
        assert ratio == pytest.approx(262144 / 32768, rel=0.01)  # 8x

    def test_kv_at_4k_matches_audit(self):
        b = predict_breakdown("4B", 4096)
        assert b.kv_cache_bytes == pytest.approx(128 * 1024 * 1024, rel=0.01)  # 128 MiB FP16

    def test_kv_at_262k_matches_audit(self):
        b = predict_breakdown("4B", 262144)
        assert b.kv_cache_bytes == pytest.approx(8 * 1024**3, rel=0.01)  # 8 GiB FP16


class TestRecurrentStateFlat:
    """Recurrent state must be constant across context length."""

    def test_state_constant(self):
        b4k = predict_breakdown("4B", 4096)
        b262k = predict_breakdown("4B", 262144)
        assert b4k.recurrent_state_bytes == b262k.recurrent_state_bytes

    def test_state_48mib_fp32(self):
        b = predict_breakdown("4B", 4096, state_dtype="fp32")
        assert b.recurrent_state_bytes == pytest.approx(48 * 1024 * 1024, rel=0.01)

    def test_state_24mib_fp16(self):
        b = predict_breakdown("4B", 4096, state_dtype="fp16")
        assert b.recurrent_state_bytes == pytest.approx(24 * 1024 * 1024, rel=0.01)


class TestCentralClaim:
    """At 262K, KV cache dwarfs recurrent state — the project's headline number."""

    def test_kv_vs_state_ratio_at_262k(self):
        b = predict_breakdown("4B", 262144)
        ratio = b.kv_cache_bytes / b.recurrent_state_bytes
        assert ratio > 100  # expected ~170x

    def test_kv_exceeds_fp16_weights_at_262k(self):
        """At 262K the KV cache exceeds even FP16 text-model weights.

        Corrected 2026-08-06. The previous test assumed FP16 weights of 10.41 GiB
        (based on wrong intermediate_size=12288 and double-counted tied embeddings).
        The actual HuggingFace config.json has intermediate_size=9216 and
        tie_word_embeddings=True, giving FP16 text-model weights of 7.83 GiB.

        KV cache at 262K = 8.00 GiB (8 full-attention layers x K+V x 4 KV heads
        x head_dim 256 x 262144 x 2 bytes).

        So the KV cache at 262K now EXCEEDS the FP16 text-model weights — a
        stronger result than before. The recurrent state remains flat at 48 MiB
        regardless of context length. This is the architectural O(1) vs O(n) claim.
        """
        b = predict_breakdown("4B", 262144)
        # KV cache now exceeds FP16 text-model weights:
        assert b.kv_cache_bytes > b.weights_bytes, "KV at 262K should exceed FP16 text weights"
        # And it utterly dominates the recurrent state, which is the claim that matters.
        assert b.kv_cache_bytes > 100 * b.recurrent_state_bytes


class TestSweep:
    def test_sweep_produces_table(self):
        breakdowns = sweep_context("4B", [4096, 32768])
        table = format_breakdown_table(breakdowns)
        assert "ctx" in table
        assert "kv_cache" in table
        assert "4096" in table
        assert "32768" in table


class TestMemoryBreakdownProperties:
    """Cover total_bytes, total_mib, to_dict on MemoryBreakdown."""

    def test_total_bytes(self):
        b = MemoryBreakdown(
            weights_bytes=1000,
            kv_cache_bytes=2000,
            recurrent_state_bytes=500,
            conv_state_bytes=100,
            context_length=4096,
        )
        assert b.total_bytes == 3600

    def test_total_mib(self):
        b = MemoryBreakdown(
            weights_bytes=1024 * 1024,
            kv_cache_bytes=2 * 1024 * 1024,
            recurrent_state_bytes=0,
            conv_state_bytes=0,
            context_length=4096,
        )
        assert b.total_mib == pytest.approx(3.0)

    def test_to_dict(self):
        b = predict_breakdown("4B", 4096)
        d = b.to_dict()
        assert "weights" in d
        assert "kv_cache" in d
        assert "recurrent_state" in d
        assert "conv_state" in d
        assert "total" in d
        assert "context_length" in d
        assert d["total"] == b.total_bytes
        assert d["context_length"] == 4096


class TestRssBytes:
    """Cover _rss_bytes on Linux."""

    def test_returns_int_on_linux(self):
        """On Linux, _rss_bytes should return a positive int."""
        val = _rss_bytes()
        if val is not None:
            assert isinstance(val, int)
            assert val > 0


class TestMeasureDelta:
    """Cover measure_delta."""

    def test_returns_delta(self):
        """measure_delta returns an int (or None) after calling fn."""
        called = []

        def my_fn(x):
            called.append(x)

        result = measure_delta(my_fn, 42)
        assert called == [42]
        # On Linux, should return an int (could be 0 if RSS unchanged)
        if result is not None:
            assert isinstance(result, int)

    def test_none_when_rss_unavailable(self):
        """Returns None when _rss_bytes returns None."""
        from unittest.mock import patch

        with patch("orionsbelt.engines.memory._rss_bytes", return_value=None):
            result = measure_delta(lambda: None)
        assert result is None

    def test_none_when_after_unavailable(self):
        """Returns None when _rss_bytes returns None on the *second* call."""
        from unittest.mock import patch

        # First call succeeds, second returns None
        with patch(
            "orionsbelt.engines.memory._rss_bytes",
            side_effect=[1024, None],
        ):
            result = measure_delta(lambda: None)
        assert result is None


class TestRssBytesErrorPath:
    """Cover the exception/fallback path of _rss_bytes."""

    def test_returns_none_on_exception(self):
        """Returns None when /proc/self/status cannot be read."""
        from unittest.mock import patch

        with patch("builtins.open", side_effect=OSError("nope")):
            result = _rss_bytes()
        assert result is None

    def test_returns_none_when_no_vmrss(self):
        """Returns None when /proc/self/status has no VmRSS line."""
        from unittest.mock import mock_open, patch

        with patch("builtins.open", mock_open(read_data="some line\nanother\n")):
            result = _rss_bytes()
        assert result is None


class TestSweepDefault:
    """Cover sweep_context default context_lengths branch."""

    def test_default_context_lengths(self):
        """sweep_context without explicit context_lengths uses defaults."""
        breakdowns = sweep_context("4B")
        assert len(breakdowns) == 4
        assert breakdowns[0].context_length == 4096
        assert breakdowns[-1].context_length == 262144


class TestEstimateWeights:
    """Cover estimate_weights with different dtypes."""

    def test_fp32_double_of_fp16(self):
        """FP32 weights should be exactly 2x FP16 weights."""
        info = LAYER_INFO["4B"]
        w16 = estimate_weights(info, 2.0)
        w32 = estimate_weights(info, 4.0)
        assert w32 == 2 * w16

    def test_int4_half_of_int8(self):
        """INT4 weights should be half of INT8."""
        info = LAYER_INFO["4B"]
        w8 = estimate_weights(info, 1.0)
        w4 = estimate_weights(info, 0.5)
        assert w4 == w8 // 2  # integer division may lose 1 byte
