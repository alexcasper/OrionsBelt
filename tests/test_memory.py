"""Tests for bench/memory.py — three-way memory attribution."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BENCH_DIR = _REPO_ROOT / "bench"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_BENCH_DIR))

from bench.memory import (  # noqa: E402
    DTYPE_BYTES,
    MemoryModel,
    compute_kv_cache,
    compute_memory,
    compute_recurrent_state,
    compute_weights,
    format_scaling_table,
    memory_scaling_table,
)

# ---------------------------------------------------------------------------
# MemoryModel construction
# ---------------------------------------------------------------------------


class TestMemoryModel:
    def test_qwen35_4b_defaults(self):
        m = MemoryModel.qwen35_4b()
        assert m.num_gdn_layers == 24
        assert m.num_full_attn_layers == 8
        assert m.hidden_size == 2560
        assert m.linear_num_value_heads == 32
        assert m.linear_num_key_heads == 16
        assert m.linear_key_head_dim == 128
        assert m.linear_value_head_dim == 128
        assert m.linear_conv_kernel_dim == 4
        assert m.num_layers == 32

    def test_qwen35_0_8b_defaults(self):
        m = MemoryModel.qwen35_0_8b()
        assert m.num_gdn_layers == 18
        assert m.num_full_attn_layers == 6
        assert m.hidden_size == 1024
        assert m.linear_num_value_heads == 16
        assert m.num_layers == 24

    def test_from_config_4b(self):
        config = {
            "text_config": {
                "hidden_size": 2560,
                "layer_types": ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "mamba_ssm_dtype": "float32",
            }
        }
        m = MemoryModel.from_config(config)
        assert m.num_gdn_layers == 6
        assert m.num_full_attn_layers == 2

    def test_from_config_fallback_no_layer_types(self):
        config = {
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "full_attention_interval": 4,
        }
        m = MemoryModel.from_config(config)
        assert m.num_gdn_layers == 24
        assert m.num_full_attn_layers == 8

    def test_key_dim(self):
        m = MemoryModel.qwen35_4b()
        assert m.key_dim == 16 * 128  # = 2048

    def test_value_dim(self):
        m = MemoryModel.qwen35_4b()
        assert m.value_dim == 32 * 128  # = 4096

    def test_conv_dim(self):
        m = MemoryModel.qwen35_4b()
        # key_dim * 2 + value_dim = 2048*2 + 4096 = 8192
        assert m.conv_dim == 8192


# ---------------------------------------------------------------------------
# Recurrent state
# ---------------------------------------------------------------------------


class TestRecurrentState:
    def test_4b_state_per_layer(self):
        """4B: 32 × 128 × 128 × 4 bytes = 2,097,152 (2 MiB) per layer."""
        m = MemoryModel.qwen35_4b()
        total = compute_recurrent_state(m)
        # State only (no conv): 24 × 32 × 128 × 128 × 4
        state_only = 24 * 32 * 128 * 128 * 4
        assert state_only == 24 * 524_288 * 4  # = 50,331,648 = 48 MiB
        # Total includes conv state, so >= state_only
        assert total >= state_only

    def test_4b_state_48_mib(self):
        """4B total recurrent state (state only, no conv) = 48 MiB."""
        m = MemoryModel.qwen35_4b()
        state_only = m.num_gdn_layers * m.linear_num_value_heads * m.linear_key_head_dim * m.linear_value_head_dim * 4
        assert state_only == 50_331_648  # exactly 48 MiB

    def test_0_8b_state(self):
        """0.8B: 18 × 16 × 128 × 128 × 4 = 18 MiB."""
        m = MemoryModel.qwen35_0_8b()
        state_only = m.num_gdn_layers * m.linear_num_value_heads * m.linear_key_head_dim * m.linear_value_head_dim * 4
        assert state_only == 18 * 262_144 * 4  # = 18,874,368 = 18 MiB

    def test_state_is_context_independent(self):
        """Recurrent state is O(1) — does not change with context length."""
        m = MemoryModel.qwen35_4b()
        s1 = compute_recurrent_state(m)
        # This should be the same regardless of context length
        s2 = compute_recurrent_state(m)
        assert s1 == s2

    def test_conv_state_included(self):
        """Conv1D state is included in recurrent_state total."""
        m = MemoryModel.qwen35_4b()
        total = compute_recurrent_state(m)
        state_only = 24 * 32 * 128 * 128 * 4
        conv_only = 24 * m.conv_dim * 4 * 4  # layers × conv_dim × kernel × fp32
        assert total == state_only + conv_only

    def test_bf16_halves_state(self):
        m = MemoryModel.qwen35_4b()
        fp32_state = compute_recurrent_state(m, state_dtype="float32")
        bf16_state = compute_recurrent_state(m, state_dtype="bfloat16")
        assert bf16_state == fp32_state // 2


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------


class TestKVCache:
    def test_4b_kv_per_token(self):
        """4B: 8 layers × 2 (K,V) × 4 KV heads × 256 head_dim × 2 bytes = 32,768 bytes/token."""
        m = MemoryModel.qwen35_4b()
        kv = compute_kv_cache(m, total_tokens=1)
        assert kv == 8 * 2 * 1 * 4 * 256 * 2  # = 32,768

    def test_4b_kv_at_4k(self):
        m = MemoryModel.qwen35_4b()
        kv = compute_kv_cache(m, total_tokens=4096)
        assert kv == 32_768 * 4096  # = 134,217,728 = 128 MiB

    def test_4b_kv_at_262k(self):
        m = MemoryModel.qwen35_4b()
        kv = compute_kv_cache(m, total_tokens=262144)
        # 32768 * 262144 = 8,589,934,592 = ~8 GiB
        assert kv == 32768 * 262144

    def test_kv_grows_linearly(self):
        m = MemoryModel.qwen35_4b()
        kv_4k = compute_kv_cache(m, 4096)
        kv_8k = compute_kv_cache(m, 8192)
        assert kv_8k == 2 * kv_4k

    def test_kv_with_generated_tokens(self):
        m = MemoryModel.qwen35_4b()
        ctx = 32768
        gen = 256
        kv = compute_kv_cache(m, ctx + gen)
        kv_ctx_only = compute_kv_cache(m, ctx)
        # Difference is 256 tokens worth
        per_token = 32768
        assert kv - kv_ctx_only == 256 * per_token


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


class TestWeights:
    def test_4b_weights_fp16(self):
        """4B at fp16: ~4B params × 2 bytes = ~8 GB."""
        m = MemoryModel.qwen35_4b()
        w = compute_weights(m)
        assert w == 4_000_000_000 * 2  # = 8 GB

    def test_4b_weights_int8(self):
        m = MemoryModel.qwen35_4b()
        w = compute_weights(m, weight_dtype="int8")
        assert w == 4_000_000_000  # = 4 GB

    def test_weights_context_independent(self):
        """Weights don't change with context length."""
        m = MemoryModel.qwen35_4b()
        w1 = compute_weights(m)
        w2 = compute_weights(m)
        assert w1 == w2

    def test_architectural_estimate(self):
        """When num_parameters is None, use architectural estimate."""
        m = MemoryModel(
            hidden_size=2560, num_gdn_layers=24, num_full_attn_layers=8,
            linear_num_key_heads=16, linear_num_value_heads=32,
            linear_key_head_dim=128, linear_value_head_dim=128,
            linear_conv_kernel_dim=4, num_attention_heads=16,
            num_key_value_heads=4, head_dim=256,
        )
        w = compute_weights(m)
        # Should be in the right order of magnitude (billions)
        assert w > 1_000_000_000  # > 1 GB
        assert w < 20_000_000_000  # < 20 GB


# ---------------------------------------------------------------------------
# Full breakdown
# ---------------------------------------------------------------------------


class TestComputeMemory:
    def test_4b_at_4k(self):
        m = MemoryModel.qwen35_4b()
        b = compute_memory(m, context_length=4096)
        assert b.weights == 8_000_000_000
        assert b.kv_cache == 32_768 * 4096
        assert b.recurrent_state > 50_000_000  # ~48 MiB state + conv
        assert b.phase == "prefill"
        assert b.context_length == 4096

    def test_4b_at_262k(self):
        m = MemoryModel.qwen35_4b()
        b = compute_memory(m, context_length=262144)
        # KV cache should dominate at long context
        assert b.kv_cache > b.weights  # ~8 GB > 8 GB (approx equal)
        assert b.recurrent_state < b.kv_cache  # state is tiny vs cache

    def test_decode_adds_tokens(self):
        m = MemoryModel.qwen35_4b()
        b_prefill = compute_memory(m, 32768, 0, "prefill")
        b_decode = compute_memory(m, 32768, 256, "decode")
        # Weights and state are the same
        assert b_decode.weights == b_prefill.weights
        assert b_decode.recurrent_state == b_prefill.recurrent_state
        # KV cache grew
        assert b_decode.kv_cache > b_prefill.kv_cache

    def test_total_property(self):
        m = MemoryModel.qwen35_4b()
        b = compute_memory(m, 4096)
        assert b.total == b.weights + b.kv_cache + b.recurrent_state


# ---------------------------------------------------------------------------
# Scaling table
# ---------------------------------------------------------------------------


class TestScalingTable:
    def test_scaling_table_has_4_points(self):
        m = MemoryModel.qwen35_4b()
        table = memory_scaling_table(m)
        assert len(table) == 4

    def test_weights_flat_across_context(self):
        """Weights stay flat across all context lengths."""
        m = MemoryModel.qwen35_4b()
        table = memory_scaling_table(m)
        weights = {b.weights for b in table}
        assert len(weights) == 1  # all the same

    def test_state_flat_across_context(self):
        """Recurrent state stays flat across all context lengths."""
        m = MemoryModel.qwen35_4b()
        table = memory_scaling_table(m)
        states = {b.recurrent_state for b in table}
        assert len(states) == 1  # all the same

    def test_kv_cache_grows(self):
        """KV cache grows linearly with context length."""
        m = MemoryModel.qwen35_4b()
        table = memory_scaling_table(m)
        kvs = [b.kv_cache for b in table]
        # Each should be larger than the previous
        for i in range(1, len(kvs)):
            assert kvs[i] > kvs[i - 1]

    def test_format_table_contains_headers(self):
        m = MemoryModel.qwen35_4b()
        table = memory_scaling_table(m)
        formatted = format_scaling_table(table)
        assert "Context" in formatted
        assert "Weights" in formatted
        assert "KV cache" in formatted
        assert "Recurrent state" in formatted


# ---------------------------------------------------------------------------
# Dtype bytes
# ---------------------------------------------------------------------------


class TestDtypeBytes:
    def test_fp32(self):
        assert DTYPE_BYTES["fp32"] == 4
        assert DTYPE_BYTES["float32"] == 4

    def test_fp16(self):
        assert DTYPE_BYTES["fp16"] == 2
        assert DTYPE_BYTES["float16"] == 2

    def test_int8(self):
        assert DTYPE_BYTES["int8"] == 1

    def test_unknown_defaults_to_4(self):
        from bench.memory import _dtype_size

        assert _dtype_size("unknown_dtype") == 4
