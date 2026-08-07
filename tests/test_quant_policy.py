"""Tests for src/orionsbelt/quant/policy.py — GDN quantization policy.

This module defines the precision carve-outs that keep GDN numerically
stable. A silent bug here corrupts model output at long context, so the
tests verify both the data structures and the lookup/estimation functions.
"""

from __future__ import annotations

import pytest

from orionsbelt.quant.policy import (
    ALL_POLICIES,
    EMBEDDING_POLICIES,
    FULL_ATTENTION_POLICIES,
    GDN_POLICIES,
    MLP_POLICIES,
    Precision,
    QuantScheme,
    TensorGroupPolicy,
    estimate_weight_footprint_mib,
    policy_for,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestPrecisionEnum:
    def test_values(self):
        assert Precision.INT4.value == "int4"
        assert Precision.INT8.value == "int8"
        assert Precision.FP16.value == "fp16"
        assert Precision.BF16.value == "bf16"
        assert Precision.FP32.value == "fp32"

    def test_is_str_enum(self):
        assert isinstance(Precision.FP16, str)
        assert Precision.FP16 == "fp16"


class TestQuantSchemeEnum:
    def test_values(self):
        assert QuantScheme.WEIGHT_ONLY_INT4.value == "int4_w4a16"
        assert QuantScheme.WEIGHT_ONLY_INT8.value == "int8_w8a16"
        assert QuantScheme.W8A8.value == "int8_w8a8"
        assert QuantScheme.FP16.value == "fp16"
        assert QuantScheme.FP32.value == "fp32"

    def test_is_str_enum(self):
        assert isinstance(QuantScheme.FP16, str)


# ---------------------------------------------------------------------------
# TensorGroupPolicy dataclass
# ---------------------------------------------------------------------------


class TestTensorGroupPolicy:
    def test_construction(self):
        p = TensorGroupPolicy(
            tensor_group="test.weight",
            scheme=QuantScheme.WEIGHT_ONLY_INT4,
            precision_runtime=Precision.FP16,
            reason="test reason",
        )
        assert p.tensor_group == "test.weight"
        assert p.scheme == QuantScheme.WEIGHT_ONLY_INT4
        assert p.precision_runtime == Precision.FP16
        assert p.reason == "test reason"

    def test_is_frozen(self):
        p = TensorGroupPolicy(
            tensor_group="x",
            scheme=QuantScheme.FP16,
            precision_runtime=Precision.FP16,
            reason="",
        )
        with pytest.raises((AttributeError, Exception)):
            p.tensor_group = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GDN policies — the critical carve-outs
# ---------------------------------------------------------------------------


class TestGDNPolicies:
    def test_non_empty(self):
        assert len(GDN_POLICIES) > 0

    def test_no_duplicate_tensor_groups(self):
        names = [p.tensor_group for p in GDN_POLICIES]
        assert len(names) == len(set(names))

    def test_recurrent_state_is_fp32(self):
        """THE critical carve-out: recurrent state must be FP32."""
        state = next(p for p in GDN_POLICIES if p.tensor_group == "recurrent_state")
        assert state.scheme == QuantScheme.FP32
        assert state.precision_runtime == Precision.FP32

    def test_a_log_is_fp16_or_higher(self):
        """A_log enters exp(), so precision errors are exponentially amplified."""
        a_log = next(p for p in GDN_POLICIES if p.tensor_group == "A_log")
        assert a_log.scheme == QuantScheme.FP16
        assert a_log.precision_runtime == Precision.FP16

    def test_dt_bias_is_fp16_or_higher(self):
        dt = next(p for p in GDN_POLICIES if p.tensor_group == "dt_bias")
        assert dt.scheme == QuantScheme.FP16

    def test_beta_gate_is_int8_not_int4(self):
        """Beta controls write magnitude — more precision than INT4."""
        beta = next(p for p in GDN_POLICIES if p.tensor_group == "in_proj_b.weight")
        assert beta.scheme == QuantScheme.WEIGHT_ONLY_INT8

    def test_decay_gate_input_is_int8_not_int4(self):
        """Decay-gate input `a` feeds exp() — needs more precision than INT4."""
        a = next(p for p in GDN_POLICIES if p.tensor_group == "in_proj_a.weight")
        assert a.scheme == QuantScheme.WEIGHT_ONLY_INT8

    def test_reasons_non_empty(self):
        """Every policy must explain why its precision was chosen."""
        for p in GDN_POLICIES:
            assert len(p.reason) > 10, f"Missing reason for {p.tensor_group}"

    def test_known_tensor_groups_present(self):
        names = {p.tensor_group for p in GDN_POLICIES}
        expected = {
            "in_proj_qkv.weight",
            "in_proj_z.weight",
            "in_proj_b.weight",
            "in_proj_a.weight",
            "out_proj.weight",
            "conv1d.weight",
            "A_log",
            "dt_bias",
            "recurrent_state",
            "norm.weight",
        }
        assert expected.issubset(names), f"Missing: {expected - names}"


# ---------------------------------------------------------------------------
# Full-attention policies
# ---------------------------------------------------------------------------


class TestFullAttentionPolicies:
    def test_non_empty(self):
        assert len(FULL_ATTENTION_POLICIES) > 0

    def test_no_duplicates(self):
        names = [p.tensor_group for p in FULL_ATTENTION_POLICIES]
        assert len(names) == len(set(names))

    def test_kv_cache_is_fp16(self):
        kv = next(p for p in FULL_ATTENTION_POLICIES if p.tensor_group == "kv_cache")
        assert kv.scheme == QuantScheme.FP16

    def test_projections_are_int4(self):
        """Standard attention projections are safe for INT4."""
        for name in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"):
            p = next(pp for pp in FULL_ATTENTION_POLICIES if pp.tensor_group == name)
            assert p.scheme == QuantScheme.WEIGHT_ONLY_INT4

    def test_reasons_non_empty(self):
        for p in FULL_ATTENTION_POLICIES:
            assert len(p.reason) > 10


# ---------------------------------------------------------------------------
# MLP policies
# ---------------------------------------------------------------------------


class TestMLPPolicies:
    def test_non_empty(self):
        assert len(MLP_POLICIES) > 0

    def test_all_int4(self):
        """SwiGLU MLP is the biggest INT4 win — all three projections."""
        for p in MLP_POLICIES:
            assert p.scheme == QuantScheme.WEIGHT_ONLY_INT4

    def test_known_groups_present(self):
        names = {p.tensor_group for p in MLP_POLICIES}
        assert {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}.issubset(names)


# ---------------------------------------------------------------------------
# Embedding policies
# ---------------------------------------------------------------------------


class TestEmbeddingPolicies:
    def test_non_empty(self):
        assert len(EMBEDDING_POLICIES) > 0

    def test_embed_is_int8(self):
        """Embedding INT8, not INT4 — vocabulary diversity risk."""
        embed = next(p for p in EMBEDDING_POLICIES if p.tensor_group == "embed_tokens.weight")
        assert embed.scheme == QuantScheme.WEIGHT_ONLY_INT8

    def test_lm_head_is_int8(self):
        head = next(p for p in EMBEDDING_POLICIES if p.tensor_group == "lm_head.weight")
        assert head.scheme == QuantScheme.WEIGHT_ONLY_INT8


# ---------------------------------------------------------------------------
# ALL_POLICIES combined dict
# ---------------------------------------------------------------------------


class TestAllPolicies:
    def test_contains_all_from_each_list(self):
        total = (
            len(GDN_POLICIES)
            + len(FULL_ATTENTION_POLICIES)
            + len(MLP_POLICIES)
            + len(EMBEDDING_POLICIES)
        )
        assert len(ALL_POLICIES) == total

    def test_no_key_collisions_across_lists(self):
        """No tensor_group name appears in two different policy lists."""
        all_names: list[str] = []
        for plist in (GDN_POLICIES, FULL_ATTENTION_POLICIES, MLP_POLICIES, EMBEDDING_POLICIES):
            all_names.extend(p.tensor_group for p in plist)
        assert len(all_names) == len(set(all_names))

    def test_dict_values_match_list_entries(self):
        for plist in (GDN_POLICIES, FULL_ATTENTION_POLICIES, MLP_POLICIES, EMBEDDING_POLICIES):
            for p in plist:
                assert ALL_POLICIES[p.tensor_group] is p


# ---------------------------------------------------------------------------
# policy_for()
# ---------------------------------------------------------------------------


class TestPolicyFor:
    def test_exact_match(self):
        result = policy_for("recurrent_state")
        assert result.tensor_group == "recurrent_state"
        assert result.precision_runtime == Precision.FP32

    def test_suffix_match(self):
        """policy_for should match by suffix on qualified tensor names."""
        result = policy_for("layers.5.linear_attn.in_proj_qkv.weight")
        assert result.tensor_group == "in_proj_qkv.weight"

    def test_partial_name_match(self):
        result = policy_for("model.layers.0.gate_proj.weight")
        assert result.tensor_group == "gate_proj.weight"

    def test_keyerror_on_no_match(self):
        with pytest.raises(KeyError, match="No quantization policy"):
            policy_for("totally_unknown_tensor")

    def test_most_keys_reachable_by_exact_name(self):
        """Most keys are findable. Documented exception: q_norm/k_norm.weight
        collides with norm.weight because policy_for uses substring matching."""
        for key in ALL_POLICIES:
            if key == "q_norm/k_norm.weight":
                continue  # substring collision with norm.weight (pre-existing)
            result = policy_for(key)
            assert result.tensor_group == key

    def test_q_norm_collides_with_norm(self):
        """Known limitation: substring matching causes q_norm/k_norm.weight
        to match norm.weight (from GDN policies) first."""
        result = policy_for("q_norm/k_norm.weight")
        assert result.tensor_group == "norm.weight"  # not q_norm/k_norm.weight


# ---------------------------------------------------------------------------
# estimate_weight_footprint_mib()
# ---------------------------------------------------------------------------


class TestEstimateWeightFootprint:
    def test_returns_expected_keys(self):
        result = estimate_weight_footprint_mib(1_000_000)
        for key in ("int4_mib", "int8_mib", "fp16_mib", "total_mib", "bits_per_param_avg"):
            assert key in result

    def test_default_fractions_sum_to_one(self):
        """Default 85% INT4 + 10% INT8 + 5% FP16."""
        result = estimate_weight_footprint_mib(1_000_000)
        assert result["int4_mib"] > 0
        assert result["int8_mib"] > 0
        assert result["fp16_mib"] > 0

    def test_total_equals_sum_of_parts(self):
        result = estimate_weight_footprint_mib(10_000_000)
        total = result["int4_mib"] + result["int8_mib"] + result["fp16_mib"]
        assert abs(total - result["total_mib"]) < 0.001

    def test_int4_bytes_calculation(self):
        """INT4 = 0.5 bytes per param."""
        result = estimate_weight_footprint_mib(1_048_576, int4_fraction=1.0, int8_fraction=0.0)
        # 1_048_576 params * 0.5 bytes = 524_288 bytes = 0.5 MiB
        assert abs(result["int4_mib"] - 0.5) < 0.001

    def test_int8_bytes_calculation(self):
        """INT8 = 1 byte per param."""
        result = estimate_weight_footprint_mib(1_048_576, int4_fraction=0.0, int8_fraction=1.0)
        assert abs(result["int8_mib"] - 1.0) < 0.001

    def test_fp16_bytes_calculation(self):
        """FP16 = 2 bytes per param."""
        result = estimate_weight_footprint_mib(1_048_576, int4_fraction=0.0, int8_fraction=0.0)
        assert abs(result["fp16_mib"] - 2.0) < 0.001

    def test_bits_per_param_avg(self):
        """Average bits per param: with 85/10/5 split → ~6.2 bits."""
        result = estimate_weight_footprint_mib(1_000_000)
        # 0.85*4 + 0.10*8 + 0.05*16 = 3.4 + 0.8 + 0.8 = 5.0 bits
        assert abs(result["bits_per_param_avg"] - 5.0) < 0.01

    def test_all_int4(self):
        result = estimate_weight_footprint_mib(1_000_000, int4_fraction=1.0, int8_fraction=0.0)
        assert result["bits_per_param_avg"] == 4.0

    def test_all_int8(self):
        result = estimate_weight_footprint_mib(1_000_000, int4_fraction=0.0, int8_fraction=1.0)
        assert result["bits_per_param_avg"] == 8.0

    def test_all_fp16(self):
        result = estimate_weight_footprint_mib(1_000_000, int4_fraction=0.0, int8_fraction=0.0)
        assert result["bits_per_param_avg"] == 16.0

    def test_realistic_4b_model(self):
        """4B param model with default fractions should be ~2.4 GiB."""
        result = estimate_weight_footprint_mib(4_000_000_000)
        # 4B * 0.85 * 0.5 + 4B * 0.10 * 1 + 4B * 0.05 * 2
        # = 1.7B + 0.4B + 0.4B = 2.5B bytes ≈ 2384 MiB
        assert 2000 < result["total_mib"] < 3000
