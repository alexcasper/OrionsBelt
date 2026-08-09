# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

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
    MemoryBreakdown,
    ModelConfig,
    _fmt_bytes,
    cross_check,
    decomposition,
    kv_cache_bytes,
    context_sweep,
    memory_breakdown,
    print_decomposition,
    recurrent_state_bytes,
    weights_bytes,
)


class TestWeightsFlat:
    def test_weights_independent_of_context(self):
        for cfg in (QWEN35_4B, QWEN35_08B):
            w1 = weights_bytes(cfg)
            assert weights_bytes(cfg) == w1  # idempotent

    def test_4b_weights_analytical(self):
        """Weights are now config-derived (ob-7m6, ported from t4), not a round
        assumed num_params. The exact figure comes from the analytical formula
        over the verified 4B config dimensions (GDN_LAYER_AUDIT.md §8)."""
        w = weights_bytes(QWEN35_4B)
        # Exact analytical value: GDN + FA + MLP + embedding(248K×2560, tied) + final norm
        assert w == 8_411_693_056  # ≈ 7.83 GiB at fp16
        # Sanity: in the ~8 GiB band for a ~4B-parameter fp16 checkpoint
        assert 7e9 < w < 9e9


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


# ---------------------------------------------------------------------------
# _fmt_bytes
# ---------------------------------------------------------------------------


class TestFmtBytes:
    def test_bytes(self):
        assert _fmt_bytes(512) == "512.0 B"

    def test_kib(self):
        assert _fmt_bytes(2048) == "2.0 KiB"

    def test_mib(self):
        assert _fmt_bytes(5 * 1024 * 1024) == "5.0 MiB"

    def test_gib(self):
        assert _fmt_bytes(8 * 1024**3) == "8.0 GiB"

    def test_tib(self):
        assert _fmt_bytes(2 * 1024**4) == "2.0 TiB"

    def test_pib_overflow(self):
        assert _fmt_bytes(1024**5) == "1.0 PiB"

    def test_zero(self):
        assert _fmt_bytes(0) == "0.0 B"


# ---------------------------------------------------------------------------
# print_decomposition
# ---------------------------------------------------------------------------


class TestPrintDecomposition:
    def test_prints_header_and_rows(self, capsys):
        print_decomposition(QWEN35_4B, [4096, 32768])
        captured = capsys.readouterr()
        assert "Memory decomposition" in captured.out
        assert "GDN" in captured.out
        assert "FA layers" in captured.out
        # Both context lengths should appear
        assert "4,096" in captured.out
        assert "32,768" in captured.out

    def test_prints_weights_column(self, capsys):
        print_decomposition(QWEN35_4B, [4096])
        captured = capsys.readouterr()
        assert "Weights" in captured.out
        assert "KV cache" in captured.out
        assert "GDN state" in captured.out

    def test_prints_saved_column(self, capsys):
        print_decomposition(QWEN35_4B, [32768])
        captured = capsys.readouterr()
        assert "Saved" in captured.out

    def test_single_context_length(self, capsys):
        print_decomposition(QWEN35_08B, [4096])
        captured = capsys.readouterr()
        assert "4,096" in captured.out
        assert "Qwen3.5-0.8B" in captured.out

    def test_large_context(self, capsys):
        """131072 context should produce output with very large KV cache."""
        print_decomposition(QWEN35_4B, [131072])
        captured = capsys.readouterr()
        assert "131,072" in captured.out


# ---------------------------------------------------------------------------
# ModelConfig.from_hf_config
# ---------------------------------------------------------------------------


class TestFromHfConfig:
    """Test config-driven ModelConfig construction."""

    def test_explicit_layer_types_derives_interval(self):
        """When layer_types is present, interval is derived from first FA index."""
        cfg = ModelConfig.from_hf_config({
            "hidden_size": 256,
            "num_attention_heads": 8,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention",
                "full_attention",
            ],
            "vocab_size": 1000,
        })
        assert cfg.num_hidden_layers == 4
        assert cfg.full_attention_interval == 4
        assert cfg.layer_types.count("full_attention") == 1
        assert cfg.layer_types.count("linear_attention") == 3

    def test_implicit_interval_from_num_hidden_layers(self):
        """Without layer_types, reads num_hidden_layers + full_attention_interval."""
        cfg = ModelConfig.from_hf_config({
            "num_hidden_layers": 32,
            "full_attention_interval": 4,
            "hidden_size": 2560,
            "num_attention_heads": 32,
        })
        assert cfg.num_hidden_layers == 32
        assert cfg.full_attention_interval == 4

    def test_head_dim_explicit(self):
        """When head_dim is in config, it's used directly."""
        cfg = ModelConfig.from_hf_config({
            "num_hidden_layers": 4,
            "hidden_size": 256,
            "num_attention_heads": 8,
            "head_dim": 128,
        })
        assert cfg.full_attn_head_dim == 128

    def test_head_dim_derived(self):
        """When head_dim is absent, derived from hidden_size // num_attention_heads."""
        cfg = ModelConfig.from_hf_config({
            "num_hidden_layers": 4,
            "hidden_size": 256,
            "num_attention_heads": 8,
        })
        assert cfg.full_attn_head_dim == 32  # 256 // 8

    def test_head_dim_zero_when_no_heads(self):
        """Division guarded when num_attention_heads is 0."""
        cfg = ModelConfig.from_hf_config({
            "num_hidden_layers": 4,
            "hidden_size": 256,
            "num_attention_heads": 0,
        })
        assert cfg.full_attn_head_dim == 0

    def test_text_config_nesting(self):
        """Config nested under 'text_config' is read correctly."""
        cfg = ModelConfig.from_hf_config({
            "text_config": {
                "num_hidden_layers": 16,
                "hidden_size": 512,
                "num_attention_heads": 8,
                "model_type": "qwen3",
            },
        })
        assert cfg.num_hidden_layers == 16
        assert cfg.hidden_size == 512

    def test_state_dtype_mapping(self):
        """mamba_ssm_dtype maps to state_dtype_bytes."""
        cfg = ModelConfig.from_hf_config({
            "num_hidden_layers": 4,
            "mamba_ssm_dtype": "bfloat16",
        })
        assert cfg.state_dtype_bytes == 2

    def test_untied_embeddings(self):
        """tie_word_embeddings=False is read from config."""
        cfg = ModelConfig.from_hf_config({
            "num_hidden_layers": 4,
            "tie_word_embeddings": False,
        })
        assert cfg.tie_word_embeddings is False

    def test_name_from_model_type(self):
        """Name defaults to model_type when not provided."""
        cfg = ModelConfig.from_hf_config({
            "model_type": "qwen3",
            "num_hidden_layers": 4,
        })
        assert cfg.name == "qwen3"

    def test_no_fa_layers_in_explicit_list(self):
        """When layer_types has no full_attention, interval is large (all GDN)."""
        cfg = ModelConfig.from_hf_config({
            "layer_types": ["linear_attention"] * 4,
        })
        assert cfg.full_attention_interval == 5  # num_hidden_layers + 1
        assert cfg.layer_types.count("linear_attention") == 4


# ---------------------------------------------------------------------------
# Untied embeddings branch in weights_bytes
# ---------------------------------------------------------------------------


class TestUntiedWeights:
    def test_untied_doubles_embedding(self):
        """Untied embeddings: weights ≈ 2× the embedding portion."""
        base = ModelConfig(
            name="test",
            hidden_size=256,
            num_hidden_layers=4,
            vocab_size=1000,
            full_attention_interval=4,
            weight_dtype_bytes=2,
        )
        tied = weights_bytes(base)
        untied = ModelConfig(
            name="test",
            hidden_size=256,
            num_hidden_layers=4,
            vocab_size=1000,
            full_attention_interval=4,
            weight_dtype_bytes=2,
            tie_word_embeddings=False,
        )
        untied_w = weights_bytes(untied)
        # The difference is exactly vocab_size * hidden_size * dtype_bytes
        diff = untied_w - tied
        assert diff == 1000 * 256 * 2


# ---------------------------------------------------------------------------
# MemoryBreakdown dataclass
# ---------------------------------------------------------------------------


class TestMemoryBreakdown:
    def test_total_property(self):
        """MemoryBreakdown.total sums the three components."""
        mb = MemoryBreakdown(weights=1000, kv_cache=500, recurrent_state=200)
        assert mb.total == 1700

    def test_total_with_zero_kv(self):
        """At seq_len=0, kv_cache is 0 but total still sums."""
        mb = MemoryBreakdown(weights=1000, kv_cache=0, recurrent_state=200)
        assert mb.total == 1200

    def test_memory_breakdown_function(self):
        """memory_breakdown() returns a MemoryBreakdown at a given context."""
        mb = memory_breakdown(QWEN35_4B, seq_len=4096)
        assert isinstance(mb, MemoryBreakdown)
        assert mb.weights == weights_bytes(QWEN35_4B)
        assert mb.kv_cache == kv_cache_bytes(QWEN35_4B, 4096)
        assert mb.recurrent_state == recurrent_state_bytes(QWEN35_4B)
        assert mb.total == mb.weights + mb.kv_cache + mb.recurrent_state


# ---------------------------------------------------------------------------
# context_sweep
# ---------------------------------------------------------------------------


class TestContextSweep:
    def test_returns_list_of_dicts(self):
        """context_sweep returns a list of dicts with expected keys."""
        data = context_sweep(QWEN35_4B, [4096, 32768])
        assert len(data) == 2
        for entry in data:
            assert "context_length" in entry
            assert "weights_gib" in entry
            assert "kv_cache_gib" in entry
            assert "recurrent_state_mib" in entry
            assert "total_gib" in entry

    def test_weights_constant_across_contexts(self):
        """Weights are the same at every context length."""
        data = context_sweep(QWEN35_4B, [4096, 32768, 131072])
        weights = {entry["weights_gib"] for entry in data}
        assert len(weights) == 1

    def test_kv_cache_grows(self):
        """KV cache GiB increases with context length."""
        data = context_sweep(QWEN35_4B, [4096, 32768])
        assert data[1]["kv_cache_gib"] > data[0]["kv_cache_gib"]

    def test_recurrent_state_constant(self):
        """Recurrent state MiB is the same at every context length."""
        data = context_sweep(QWEN35_4B, [4096, 131072])
        rs = {entry["recurrent_state_mib"] for entry in data}
        assert len(rs) == 1

    def test_total_increases(self):
        """Total GiB increases with context (dominated by KV cache growth)."""
        data = context_sweep(QWEN35_4B, [4096, 32768])
        assert data[1]["total_gib"] > data[0]["total_gib"]

    def test_single_context(self):
        """Works with a single context length."""
        data = context_sweep(QWEN35_4B, [4096])
        assert len(data) == 1
        assert data[0]["context_length"] == 4096
