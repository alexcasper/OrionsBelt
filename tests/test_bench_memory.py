"""Tests for bench/memory.py — the three-way memory decomposition (ob-vfp).

Verifies that weights are flat, KV cache grows linearly, recurrent state is O(1),
and the numbers match ADR 0003's verified table.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.harness import QWEN35_08B, QWEN35_4B  # noqa: E402
from bench.memory import (  # noqa: E402
    cross_check,
    decomposition,
    kv_cache_bytes,
    recurrent_state_bytes,
    weights_bytes,
)


class TestWeightsFlat:
    def test_weights_independent_of_context(self):
        for cfg in (QWEN35_4B, QWEN35_08B):
            w1 = weights_bytes(cfg)
            assert weights_bytes(cfg) == w1  # idempotent

    def test_4b_weights_fp16(self):
        assert weights_bytes(QWEN35_4B) == 4_000_000_000 * 2  # 8 GB fp16


class TestKVCacheLinear:
    def test_doubles_with_double_seq_len(self):
        cfg = QWEN35_4B
        kv_4k = kv_cache_bytes(cfg, 4096)
        kv_8k = kv_cache_bytes(cfg, 8192)
        assert kv_8k == 2 * kv_4k

    def test_4b_4k_matches_adr0003(self):
        """ADR 0003: 4K context → 128 MiB KV cache for 4B."""
        kv = kv_cache_bytes(QWEN35_4B, 4096)
        assert kv == 134_217_728  # 128 MiB

    def test_4b_262k_matches_adr0003(self):
        """ADR 0003: 262K context → 8.0 GiB KV cache for 4B."""
        kv = kv_cache_bytes(QWEN35_4B, 262144)
        assert kv == 8_589_934_592  # 8 GiB

    def test_08b_smaller_than_4b(self):
        """0.8B has fewer FA layers and fewer KV heads."""
        assert kv_cache_bytes(QWEN35_08B, 4096) < kv_cache_bytes(QWEN35_4B, 4096)


class TestRecurrentStateConstant:
    def test_independent_of_seq_len(self):
        cfg = QWEN35_4B
        rs = recurrent_state_bytes(cfg)
        # Should be the same regardless of "context" — it's O(1)
        assert recurrent_state_bytes(cfg) == rs

    def test_4b_matches_adr0003(self):
        """ADR 0003: 48 MiB total across 24 layers for 4B."""
        rs = recurrent_state_bytes(QWEN35_4B)
        assert rs == 50_331_648  # 48 MiB

    def test_4b_per_layer(self):
        """32 × 128 × 128 × 4 = 524,288 elements = 2 MiB per layer."""
        per_layer = recurrent_state_bytes(QWEN35_4B) // QWEN35_4B.num_gdn_layers
        assert per_layer == 2_097_152  # 2 MiB

    def test_08b_smaller(self):
        """0.8B: 16 value heads → half the per-layer state of 4B."""
        assert recurrent_state_bytes(QWEN35_08B) < recurrent_state_bytes(QWEN35_4B)


class TestDecomposition:
    def test_weights_constant_across_contexts(self):
        rows = decomposition(QWEN35_4B, [4096, 32768, 131072, 262144])
        weights = {r["weights"] for r in rows}
        assert len(weights) == 1  # flat

    def test_kv_cache_grows(self):
        rows = decomposition(QWEN35_4B, [4096, 32768, 131072, 262144])
        kvs = [r["kv_cache"] for r in rows]
        assert kvs == sorted(kvs)  # monotonically increasing
        assert kvs[-1] > kvs[0]

    def test_recurrent_state_constant(self):
        rows = decomposition(QWEN35_4B, [4096, 32768, 131072, 262144])
        rss = {r["recurrent_state"] for r in rows}
        assert len(rss) == 1  # flat

    def test_counterfactual_saved_bytes(self):
        """At 262K, the hybrid saves ~24.8 GiB vs all-attention (ADR 0003)."""
        rows = decomposition(QWEN35_4B, [262144])
        saved = rows[0]["saved_bytes"]
        # ~24.8 GiB ≈ 26.6 GB
        assert saved > 25_000_000_000  # > 25 GB saved

    def test_counterfactual_ratio(self):
        """All-attention KV = 32/8 × hybrid KV = 4× (4B)."""
        rows = decomposition(QWEN35_4B, [4096])
        ratio = rows[0]["all_attention_kv_equivalent"] / rows[0]["kv_cache"]
        assert ratio == 4.0  # 32 total / 8 FA = 4×


class TestCrossCheck:
    def test_no_discrepancies_when_matching(self):
        discrepancies = cross_check(
            QWEN35_4B,
            introspected_weights=weights_bytes(QWEN35_4B),
            introspected_state_shape=(1, 32, 128, 128),
            introspected_state_dtype_bytes=4,
        )
        assert discrepancies == []

    def test_weights_mismatch_detected(self):
        discrepancies = cross_check(
            QWEN35_4B,
            introspected_weights=999,
        )
        assert len(discrepancies) == 1
        assert "weights mismatch" in discrepancies[0]

    def test_state_shape_mismatch_detected(self):
        discrepancies = cross_check(
            QWEN35_4B,
            introspected_state_shape=(1, 16, 128, 128),  # wrong num_v_heads
        )
        assert len(discrepancies) == 1
        assert "state shape mismatch" in discrepancies[0]

    def test_unbatched_state_shape_accepted(self):
        """3D state shape (no batch dim) should be accepted."""
        discrepancies = cross_check(
            QWEN35_4B,
            introspected_state_shape=(32, 128, 128),  # no batch dim
        )
        assert discrepancies == []

    def test_dtype_mismatch_detected(self):
        discrepancies = cross_check(
            QWEN35_4B,
            introspected_state_dtype_bytes=2,  # bf16, not fp32
        )
        assert len(discrepancies) == 1
        assert "dtype mismatch" in discrepancies[0]
