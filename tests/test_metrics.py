"""Unit tests for bench/metrics.py — three-component memory attribution (ob-vfp).

Verifies the memory formulas match the ground-truth numbers from
docs/GDN_LAYER_AUDIT.md and ADR 0003 for both Qwen3.5-4B and 0.8B.
"""

import os
import sys

import pytest

_BENCH = os.path.join(os.path.dirname(__file__), "..", "bench")
_BENCH = os.path.abspath(_BENCH)
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

import metrics  # noqa: E402

# ---------------------------------------------------------------------------
# Qwen3.5-4B config (from config.json, verified in ADR 0003)
# ---------------------------------------------------------------------------

CFG_4B = metrics.ModelConfig(
    hidden_size=2560,
    num_hidden_layers=32,
    num_attention_heads=16,
    num_key_value_heads=4,
    full_attn_head_dim=256,
    linear_num_value_heads=32,
    linear_num_key_heads=16,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    intermediate_size=9216,
    vocab_size=248320,
    tie_word_embeddings=True,
)

CFG_08B = metrics.ModelConfig(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=8,
    num_key_value_heads=2,
    full_attn_head_dim=256,
    linear_num_value_heads=16,
    linear_num_key_heads=16,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    intermediate_size=2816,
    vocab_size=248320,
    tie_word_embeddings=True,
)


class TestLayerTypes:
    def test_4b_layer_counts(self):
        assert CFG_4B.num_hidden_layers == 32
        assert CFG_4B.num_gdn_layers == 24
        assert CFG_4B.num_full_attention_layers == 8

    def test_08b_layer_counts(self):
        assert CFG_08B.num_hidden_layers == 24
        assert CFG_08B.num_gdn_layers == 18
        assert CFG_08B.num_full_attention_layers == 6

    def test_3to1_ratio(self):
        for cfg in (CFG_4B, CFG_08B):
            assert cfg.num_gdn_layers == cfg.num_full_attention_layers * 3

    def test_attention_at_correct_positions(self):
        types = CFG_4B.layer_types
        attn_indices = [i for i, t in enumerate(types) if t == "full_attention"]
        assert attn_indices == [3, 7, 11, 15, 19, 23, 27, 31]


class TestRecurrentState:
    def test_4b_flat(self):
        """48 MiB across 24 layers, flat (GDN_LAYER_AUDIT.md section 2)."""
        rs = metrics.recurrent_state_bytes(CFG_4B)
        # 24 × 32 × 128 × 128 × 4 = 50,331,648
        assert rs == 24 * 32 * 128 * 128 * 4
        assert rs == 50_331_648
        assert rs / (1024**2) == pytest.approx(48.0)

    def test_08b_flat(self):
        """18 MiB across 18 layers (16 value heads)."""
        rs = metrics.recurrent_state_bytes(CFG_08B)
        # 18 × 16 × 128 × 128 × 4 = 18,874,368
        assert rs == 18 * 16 * 128 * 128 * 4

    def test_independent_of_context(self):
        """Recurrent state must be the same at any context length (METRICS.md §5.4)."""
        rs_short = metrics.recurrent_state_bytes(CFG_4B)
        assert rs_short == metrics.recurrent_state_bytes(CFG_4B)  # no seq_len param — always flat


class TestKVCache:
    def test_4b_grows_linearly(self):
        """KV cache must double when context doubles (METRICS.md §5.3)."""
        kv_4k = metrics.kv_cache_bytes(CFG_4B, 4096)
        kv_8k = metrics.kv_cache_bytes(CFG_4B, 8192)
        assert kv_8k == pytest.approx(kv_4k * 2)

    def test_4b_formula(self):
        """8 layers × 2 (KV) × 1 × seq × 4 kv_heads × 256 head_dim × 2 bytes."""
        kv = metrics.kv_cache_bytes(CFG_4B, 4096)
        expected = 8 * 2 * 1 * 4096 * 4 * 256 * 2
        assert kv == expected

    def test_4b_at_262k(self):
        """At 262K, KV cache should be ~8 GiB (ADR 0003)."""
        kv = metrics.kv_cache_bytes(CFG_4B, 262144)
        # 8 × 2 × 1 × 262144 × 4 × 256 × 2 = 8,589,934,592 bytes ≈ 8.0 GiB
        assert kv / (1024**3) == pytest.approx(8.0, abs=0.1)


class TestWeights:
    def test_4b_approximate(self):
        """Weights should be in the ~4B parameter range (× 2 bytes bf16 = ~8 GiB)."""
        w = metrics.weight_bytes(CFG_4B)
        # Allow tolerance — this is an analytical approximation
        assert 7e9 < w < 9e9, f"Expected ~8 GiB, got {w / 1e9:.2f} GiB"

    def test_weights_flat_across_context(self):
        """Weights must be independent of context length (METRICS.md §5.2)."""
        mb_short = metrics.memory_breakdown(CFG_4B, 4096)
        mb_long = metrics.memory_breakdown(CFG_4B, 262144)
        assert mb_short.weights == mb_long.weights


class TestMemoryBreakdown:
    def test_three_components(self):
        mb = metrics.memory_breakdown(CFG_4B, 32768)
        assert mb.weights > 0
        assert mb.kv_cache > 0
        assert mb.recurrent_state > 0

    def test_weights_dominant_at_short_context(self):
        """At 4K, weights should dominate total memory."""
        mb = metrics.memory_breakdown(CFG_4B, 4096)
        assert mb.weights > mb.kv_cache
        assert mb.weights > mb.recurrent_state

    def test_kv_dominant_at_long_context(self):
        """At 262K, KV cache should exceed weights (the scaling point)."""
        mb = metrics.memory_breakdown(CFG_4B, 262144)
        assert mb.kv_cache > mb.weights

    def test_recurrent_always_small(self):
        """Recurrent state should be negligible vs weights at any context."""
        for ctx in (4096, 32768, 131072, 262144):
            mb = metrics.memory_breakdown(CFG_4B, ctx)
            assert mb.recurrent_state < mb.weights * 0.01  # < 1% of weights

    def test_total(self):
        mb = metrics.memory_breakdown(CFG_4B, 4096)
        assert mb.total == mb.weights + mb.kv_cache + mb.recurrent_state


class TestContextSweep:
    def test_returns_all_contexts(self):
        sweep = metrics.context_sweep(CFG_4B)
        assert len(sweep) == 4
        assert [s["context_length"] for s in sweep] == [4096, 32768, 131072, 262144]

    def test_weights_constant_across_sweep(self):
        sweep = metrics.context_sweep(CFG_4B)
        w_values = [s["weights_gib"] for s in sweep]
        assert len(set(w_values)) == 1  # all the same

    def test_recurrent_constant_across_sweep(self):
        sweep = metrics.context_sweep(CFG_4B)
        rs_values = [s["recurrent_state_mib"] for s in sweep]
        assert len(set(rs_values)) == 1

    def test_kv_grows_monotonically(self):
        sweep = metrics.context_sweep(CFG_4B)
        kv_values = [s["kv_cache_gib"] for s in sweep]
        assert kv_values == sorted(kv_values)


class TestFromHFConfig:
    def test_parses_nested_config(self):
        """from_hf_config should handle text_config nesting."""
        config = {
            "text_config": {
                "hidden_size": 2560,
                "num_hidden_layers": 32,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "linear_num_value_heads": 32,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "intermediate_size": 9216,
                "vocab_size": 248320,
                "tie_word_embeddings": True,
            }
        }
        cfg = metrics.ModelConfig.from_hf_config(config)
        assert cfg.hidden_size == 2560
        assert cfg.num_gdn_layers == 24
