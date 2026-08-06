"""Tests for memory instrumentation (ob-vfp).

Verifies the three-way split: weights flat, KV cache linear, state flat.
"""

import pytest

from orionsbelt.engines.memory import (
    format_breakdown_table,
    predict_breakdown,
    sweep_context,
)


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

    def test_kv_vs_weights_at_262k_is_precision_dependent(self):
        """At 262K the KV cache is large but does NOT exceed FP16 weights.

        Corrected 2026-08-03. The original assertion (`kv_cache_bytes > weights_bytes`)
        was simply false for this checkpoint and the test failed on main. Verified
        independently: KV at 262K is 8.00 GiB (8 full-attention layers x K+V x 4 KV heads
        x head_dim 256 x 262144 x 2 bytes), while FP16 weights are 10.41 GiB for the full
        checkpoint including the vision tower and MTP head.

        So "the KV cache dwarfs the weights" is PRECISION-DEPENDENT, not absolute:
        false at FP16 (0.8x), true at INT4 (~3.1x). The honest framing, and the one the
        README now uses, is that the cache grows without bound while the recurrent state
        does not -- which is the architectural claim and holds at every precision.
        """
        b = predict_breakdown("4B", 262144)
        assert b.kv_cache_bytes < b.weights_bytes, "FP16 weights still exceed KV at 262K"
        # But the cache is the same order as the weights, which is the real point:
        assert b.kv_cache_bytes > 0.5 * b.weights_bytes
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
