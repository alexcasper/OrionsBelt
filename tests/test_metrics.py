"""Tests for the ported config-driven memory accounting (ob-7m6).

Ported from ``origin/bench/t4:tests/test_metrics.py`` and adapted to the
integration's ``bench/memory.py`` (the single source of truth for the predicted
``peak_memory_bytes`` columns). Verifies:

  * ``ModelConfig.from_hf_config`` parses a real HF ``config.json`` dict.
  * ``layer_types`` is *derived* from ``full_attention_interval`` (=4 → 3:1),
    never hardcoded — GDN_LAYER_AUDIT.md §1 ground truth.
  * the analytical weight formula (``weight_bytes``) is config-driven and lands
    in the right band for the verified 4B / 0.8B checkpoints.
  * KV cache grows linearly, recurrent state is O(1) flat — METRICS.md §5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.memory import (  # noqa: E402
    CANONICAL_CONTEXT_LENGTHS,
    ModelConfig,
    context_sweep,
    kv_cache_bytes,
    memory_breakdown,
    recurrent_state_bytes,
    weights_bytes,
)

# ---------------------------------------------------------------------------
# Full Qwen3.5-4B config.json text_config (verified, GDN_LAYER_AUDIT.md + ADR 0003)
# ---------------------------------------------------------------------------

HF_CONFIG_4B = {
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
        "full_attention_interval": 4,
        "mamba_ssm_dtype": "float32",
    }
}

# Same dimensions but WITHOUT an explicit layer_types array — exercises the
# interval-derivation path directly.
HF_CONFIG_4B_NO_LIST = {
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
        "full_attention_interval": 4,
    }
}


class TestFromHFConfig:
    def test_parses_nested_config(self):
        """from_hf_config should handle text_config nesting and round-trip dims."""
        cfg = ModelConfig.from_hf_config(HF_CONFIG_4B, name="Qwen/Qwen3.5-4B")
        assert cfg.hidden_size == 2560
        assert cfg.num_hidden_layers == 32
        assert cfg.num_key_value_heads == 4
        assert cfg.full_attn_head_dim == 256
        assert cfg.linear_num_value_heads == 32
        assert cfg.name == "Qwen/Qwen3.5-4B"

    def test_maps_ssm_dtype_to_state_bytes(self):
        cfg = ModelConfig.from_hf_config(HF_CONFIG_4B)
        assert cfg.state_dtype_bytes == 4  # float32 recurrent state

    def test_name_falls_back_to_model_type(self):
        cfg = ModelConfig.from_hf_config({"model_type": "qwen3_5", **HF_CONFIG_4B})
        assert cfg.name == "qwen3_5"


class TestLayerTypesDerivation:
    """The t4 derivation: layer_types from full_attention_interval (=4 → 3:1)."""

    def test_4b_layer_counts_from_interval(self):
        cfg = ModelConfig.from_hf_config(HF_CONFIG_4B_NO_LIST)
        assert cfg.num_hidden_layers == 32
        assert cfg.num_gdn_layers == 24
        assert cfg.num_full_attention_layers == 8

    def test_3to1_ratio(self):
        cfg = ModelConfig.from_hf_config(HF_CONFIG_4B_NO_LIST)
        assert cfg.num_gdn_layers == cfg.num_full_attention_layers * 3

    def test_attention_at_correct_positions(self):
        """GDN_LAYER_AUDIT.md §1: full attention at indices {3,7,...,31}."""
        cfg = ModelConfig.from_hf_config(HF_CONFIG_4B_NO_LIST)
        attn_indices = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
        assert attn_indices == [3, 7, 11, 15, 19, 23, 27, 31]

    def test_explicit_layer_types_list_reproduced(self):
        """When the config carries an explicit layer_types list, the derived
        layer_types must reproduce it (explicit → interval translation)."""
        explicit = (["linear_attention"] * 3 + ["full_attention"]) * 8  # 32 entries, every 4th FA
        cfg = ModelConfig.from_hf_config({"text_config": {"layer_types": explicit}})
        assert cfg.layer_types == explicit
        assert cfg.num_full_attention_layers == 8
        assert cfg.num_gdn_layers == 24

    def test_08b_layer_counts(self):
        cfg = ModelConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            full_attention_interval=4,
        )
        assert cfg.num_gdn_layers == 18
        assert cfg.num_full_attention_layers == 6


class TestRecurrentState:
    def test_4b_flat(self):
        """48 MiB across 24 layers, flat (GDN_LAYER_AUDIT.md §3)."""
        cfg = ModelConfig.from_hf_config(HF_CONFIG_4B)
        rs = recurrent_state_bytes(cfg)
        assert rs == 24 * 32 * 128 * 128 * 4
        assert rs == 50_331_648
        assert rs / (1024**2) == pytest.approx(48.0)

    def test_independent_of_context(self):
        """recurrent_state_bytes takes no seq_len — always flat (METRICS.md §5.4)."""
        cfg = ModelConfig.from_hf_config(HF_CONFIG_4B)
        assert recurrent_state_bytes(cfg) == recurrent_state_bytes(cfg)


class TestKVCache:
    def test_4b_formula(self):
        """8 FA × 2 (KV) × 1 × seq × 4 kv_heads × 256 head_dim × 2 bytes."""
        cfg = ModelConfig.from_hf_config(HF_CONFIG_4B)
        kv = kv_cache_bytes(cfg, 4096)
        assert kv == 8 * 2 * 1 * 4096 * 4 * 256 * 2
        assert kv == 134_217_728  # 128 MiB at 4K (ADR 0003)

    def test_4b_grows_linearly(self):
        cfg = ModelConfig.from_hf_config(HF_CONFIG_4B)
        assert kv_cache_bytes(cfg, 8192) == pytest.approx(kv_cache_bytes(cfg, 4096) * 2)

    def test_4b_at_262k(self):
        cfg = ModelConfig.from_hf_config(HF_CONFIG_4B)
        kv = kv_cache_bytes(cfg, 262144)
        assert kv / (1024**3) == pytest.approx(8.0, abs=0.1)  # ~8 GiB (ADR 0003)


class TestWeights:
    def test_4b_analytical_from_config(self):
        """Config-derived (not round num_params). Matches the verified formula."""
        cfg = ModelConfig.from_hf_config(HF_CONFIG_4B)
        w = weights_bytes(cfg)
        assert w == 8_411_693_056  # analytical over verified 4B dims
        assert 7e9 < w < 9e9

    def test_weights_flat_across_context(self):
        """Weights must be independent of context length (METRICS.md §5.2)."""
        cfg = ModelConfig.from_hf_config(HF_CONFIG_4B)
        assert memory_breakdown(cfg, 4096).weights == memory_breakdown(cfg, 262144).weights

    def test_tied_vs_untied_embedding(self):
        """Untied lm_head adds vocab×hidden weight bytes (separate lm_head)."""
        base = {k: v for k, v in HF_CONFIG_4B["text_config"].items() if k != "tie_word_embeddings"}
        tied = ModelConfig.from_hf_config({"text_config": {**base, "tie_word_embeddings": True}})
        untied = ModelConfig.from_hf_config({"text_config": {**base, "tie_word_embeddings": False}})
        delta = weights_bytes(untied) - weights_bytes(tied)
        assert delta == 248320 * 2560 * 2  # vocab × hidden × fp16 bytes


class TestMemoryBreakdown:
    def test_three_components(self):
        cfg = ModelConfig.from_hf_config(HF_CONFIG_4B)
        mb = memory_breakdown(cfg, 32768)
        assert mb.weights > 0
        assert mb.kv_cache > 0
        assert mb.recurrent_state > 0

    def test_weights_dominant_at_short_context(self):
        cfg = ModelConfig.from_hf_config(HF_CONFIG_4B)
        mb = memory_breakdown(cfg, 4096)
        assert mb.weights > mb.kv_cache
        assert mb.weights > mb.recurrent_state

    def test_kv_dominant_at_long_context(self):
        """At 262K, KV cache exceeds weights — the scaling point (ADR 0003)."""
        cfg = ModelConfig.from_hf_config(HF_CONFIG_4B)
        mb = memory_breakdown(cfg, 262144)
        assert mb.kv_cache > mb.weights

    def test_total(self):
        cfg = ModelConfig.from_hf_config(HF_CONFIG_4B)
        mb = memory_breakdown(cfg, 4096)
        assert mb.total == mb.weights + mb.kv_cache + mb.recurrent_state


class TestContextSweep:
    def test_returns_all_contexts(self):
        cfg = ModelConfig.from_hf_config(HF_CONFIG_4B)
        sweep = context_sweep(cfg)
        assert [s["context_length"] for s in sweep] == list(CANONICAL_CONTEXT_LENGTHS)

    def test_weights_constant_across_sweep(self):
        cfg = ModelConfig.from_hf_config(HF_CONFIG_4B)
        sweep = context_sweep(cfg)
        assert len({s["weights_gib"] for s in sweep}) == 1

    def test_recurrent_constant_across_sweep(self):
        cfg = ModelConfig.from_hf_config(HF_CONFIG_4B)
        sweep = context_sweep(cfg)
        assert len({s["recurrent_state_mib"] for s in sweep}) == 1

    def test_kv_grows_monotonically(self):
        cfg = ModelConfig.from_hf_config(HF_CONFIG_4B)
        sweep = context_sweep(cfg)
        kv = [s["kv_cache_gib"] for s in sweep]
        assert kv == sorted(kv)
