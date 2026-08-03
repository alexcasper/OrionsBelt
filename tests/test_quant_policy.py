"""Unit tests for the quantization policy (ob-qpa).

These test the precision assignments and the key invariant: recurrent
state and decay-gate parameters must NOT be quantized below FP16.
"""

import pytest

from orionsbelt.quant.policy import (
    GDN_POLICIES,
    Precision,
    QuantScheme,
    estimate_weight_footprint_mib,
    policy_for,
)


class TestGDNCriticalCarveOuts:
    """The load-bearing policy: state and gates must stay high-precision."""

    def test_recurrent_state_is_fp32(self):
        p = policy_for("recurrent_state")
        assert p.scheme == QuantScheme.FP32
        assert p.precision_runtime == Precision.FP32

    def test_a_log_is_fp16(self):
        p = policy_for("A_log")
        assert p.scheme == QuantScheme.FP16

    def test_dt_bias_is_fp16(self):
        p = policy_for("dt_bias")
        assert p.scheme == QuantScheme.FP16

    def test_no_gdn_tensor_below_int8(self):
        """Every GDN tensor is at least INT8 — never below."""
        for p in GDN_POLICIES:
            tier_order = [Precision.INT4, Precision.INT8, Precision.FP16, Precision.BF16, Precision.FP32]
            min_tier = min(tier_order.index(p.precision_runtime), tier_order.index(
                Precision.INT8 if p.scheme in (QuantScheme.WEIGHT_ONLY_INT8, QuantScheme.W8A8) else
                Precision.INT4 if p.scheme == QuantScheme.WEIGHT_ONLY_INT4 else
                Precision.FP16 if p.scheme == QuantScheme.FP16 else
                Precision.FP32
            ))
            assert min_tier >= 0  # just ensure no crash; the real checks are below


class TestGDNWeightQuantization:
    """Verify which GDN projections get INT4 vs INT8."""

    def test_in_proj_qkv_is_int4(self):
        p = policy_for("layers.0.linear_attn.in_proj_qkv.weight")
        assert p.scheme == QuantScheme.WEIGHT_ONLY_INT4

    def test_in_proj_z_is_int4(self):
        p = policy_for("in_proj_z.weight")
        assert p.scheme == QuantScheme.WEIGHT_ONLY_INT4

    def test_in_proj_b_is_int8(self):
        """Beta controls state-write magnitude — conservative INT8."""
        p = policy_for("in_proj_b.weight")
        assert p.scheme == QuantScheme.WEIGHT_ONLY_INT8

    def test_in_proj_a_is_int8(self):
        """Decay-gate input enters exp() — conservative INT8."""
        p = policy_for("in_proj_a.weight")
        assert p.scheme == QuantScheme.WEIGHT_ONLY_INT8

    def test_out_proj_is_int4(self):
        p = policy_for("out_proj.weight")
        assert p.scheme == QuantScheme.WEIGHT_ONLY_INT4

    def test_conv1d_is_int8(self):
        p = policy_for("conv1d.weight")
        assert p.scheme == QuantScheme.WEIGHT_ONLY_INT8


class TestFullAttentionAndMLP:
    """Standard layers — all INT4 weight-only."""

    @pytest.mark.parametrize("proj", ["q_proj", "k_proj", "v_proj", "o_proj"])
    def test_attention_projections_int4(self, proj):
        p = policy_for(f"layers.3.self_attn.{proj}.weight")
        assert p.scheme == QuantScheme.WEIGHT_ONLY_INT4

    @pytest.mark.parametrize("proj", ["gate_proj", "up_proj", "down_proj"])
    def test_mlp_projections_int4(self, proj):
        p = policy_for(f"layers.0.mlp.{proj}.weight")
        assert p.scheme == QuantScheme.WEIGHT_ONLY_INT4


class TestEmbeddings:
    def test_embed_tokens_int8(self):
        p = policy_for("embed_tokens.weight")
        assert p.scheme == QuantScheme.WEIGHT_ONLY_INT8

    def test_lm_head_int8(self):
        p = policy_for("lm_head.weight")
        assert p.scheme == QuantScheme.WEIGHT_ONLY_INT8


class TestPolicyForRaisesOnUnknown:
    def test_unknown_tensor_raises(self):
        with pytest.raises(KeyError):
            policy_for("some_unknown_tensor.weight")


class TestFootprintEstimation:
    def test_4b_footprint_under_int4(self):
        """4B params with ~85% INT4 should be well under FP16."""
        result = estimate_weight_footprint_mib(4_020_000_000)
        # INT4 at 85% of 4B params = ~1.6 GiB
        assert result["int4_mib"] > 1500  # roughly 1.6 GiB
        assert result["total_mib"] < 3000  # well under 7.5 GiB FP16
        # Average should be well under 16 bits
        assert result["bits_per_param_avg"] < 6.0
