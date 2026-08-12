# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / AAIF
# SPDX-License-Identifier: Apache-2.0
"""Tests for bench/gdn2_swap.py — GDN-2 layer swap experiment (ob-68l).

Tests cover the pure-PyTorch recurrence, parameter counting, and manifest
capture. The full model-swap pipeline requires HuggingFace transformers +
downloaded weights and is exercised end-to-end on-device, not here.
"""

import os
import sys

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

nn = torch.nn if HAS_TORCH else None

_NNModule = nn.Module if HAS_TORCH else object
_sysmark = pytest.mark.skipif(not HAS_TORCH, reason="requires torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))

from gdn2_swap import (  # noqa: E402
    capture_manifest,
    count_parameters,
    gdn2_recurrent,
)

# ---------------------------------------------------------------------------
# gdn2_recurrent
# ---------------------------------------------------------------------------


def _make_inputs(B=2, H=4, T=6, K=8, V=5, dtype=None):
    """Create valid recurrence inputs."""
    if dtype is None:
        dtype = torch.float32
    torch.manual_seed(42)
    return dict(
        query=torch.randn(B, H, T, K, dtype=dtype),
        key=torch.randn(B, H, T, K, dtype=dtype),
        value=torch.randn(B, H, T, V, dtype=dtype),
        g=torch.randn(B, H, T).abs() * -0.1,  # negative log-decay
        b_gate=torch.rand(B, H, T, K, dtype=dtype),  # [0,1]
        w_gate=torch.rand(B, H, T, V, dtype=dtype),  # [0,1]
    )


@_sysmark
class TestGdn2Recurrent:
    def test_output_shape(self):
        """Output shape is [B, H, T, V] (transpose undoes the internal swap)."""
        inp = _make_inputs(B=2, H=4, T=6, K=8, V=5)
        out = gdn2_recurrent(**inp)
        assert out.shape == (2, 4, 6, 5)

    def test_output_shape_single_batch(self):
        inp = _make_inputs(B=1, H=1, T=1, K=3, V=2)
        out = gdn2_recurrent(**inp)
        assert out.shape == (1, 1, 1, 2)

    def test_output_dtype_preserved(self):
        """Output dtype should match input query dtype."""
        inp = _make_inputs(dtype=torch.float32)
        out = gdn2_recurrent(**inp)
        assert out.dtype == torch.float32

    def test_deterministic_given_seed(self):
        """Same inputs → same output."""
        inp = _make_inputs()
        out1 = gdn2_recurrent(**inp)
        out2 = gdn2_recurrent(**inp)
        assert torch.allclose(out1, out2)

    def test_no_nan_no_inf(self):
        """Output must be finite for well-conditioned inputs."""
        inp = _make_inputs()
        out = gdn2_recurrent(**inp)
        assert torch.isfinite(out).all()

    def test_decay_zero_resets_state(self):
        """When g is very negative (exp(g)≈0), state decays to zero each step.

        With exp(g)≈0, the recurrence becomes state := k⊗(w⊙v) each step,
        and output = qᵀ(k⊗(w⊙v)).
        """
        B, H, T, K, V = 1, 1, 3, 4, 3
        inp = _make_inputs(B=B, H=H, T=T, K=K, V=V)
        # Override decay to near-zero
        inp["g"] = torch.full((B, H, T), -50.0)  # exp(-50) ≈ 0
        out = gdn2_recurrent(**inp)
        assert torch.isfinite(out).all()

    def test_decay_one_accumulates(self):
        """When g=0 (exp(0)=1), no decay — state accumulates."""
        B, H, T, K, V = 1, 1, 2, 4, 3
        inp = _make_inputs(B=B, H=H, T=T, K=K, V=V)
        inp["g"] = torch.zeros(B, H, T)  # exp(0)=1, full retention
        out = gdn2_recurrent(**inp)
        assert torch.isfinite(out).all()

    def test_use_qk_l2norm_false(self):
        """Should work without L2 normalization."""
        inp = _make_inputs()
        out = gdn2_recurrent(**inp, use_qk_l2norm=False)
        assert out.shape == (2, 4, 6, 5)
        assert torch.isfinite(out).all()

    def test_different_batch_sizes(self):
        for B in [1, 3, 8]:
            inp = _make_inputs(B=B)
            out = gdn2_recurrent(**inp)
            assert out.shape[0] == B

    def test_l2norm_changes_output(self):
        """L2 norm should produce different output than raw qk."""
        inp = _make_inputs()
        out_norm = gdn2_recurrent(**inp, use_qk_l2norm=True)
        out_raw = gdn2_recurrent(**inp, use_qk_l2norm=False)
        assert not torch.allclose(out_norm, out_raw)

    def test_erase_gate_zeros_output_when_full_erase(self):
        """When b_gate=1 (full erase) and w_gate=0 (no write), output should
        be zero after the first step (state is fully erased and nothing written)."""
        B, H, T, K, V = 1, 1, 3, 4, 3
        inp = _make_inputs(B=B, H=H, T=T, K=K, V=V)
        inp["b_gate"] = torch.ones(B, H, T, K)
        inp["w_gate"] = torch.zeros(B, H, T, V)
        inp["g"] = torch.zeros(B, H, T)  # no decay
        out = gdn2_recurrent(**inp)
        # After step 0: state is erased and not written → state ≈ 0
        # Step 1+: same → all outputs ≈ 0
        assert out.abs().max() < 1e-4


# ---------------------------------------------------------------------------
# count_parameters
# ---------------------------------------------------------------------------


@_sysmark
class TestCountParameters:
    def test_simple_linear(self):
        layer = nn.Linear(10, 5)
        # 10*5 weights + 5 biases = 55
        assert count_parameters(layer) == 55

    def test_no_bias(self):
        layer = nn.Linear(10, 5, bias=False)
        assert count_parameters(layer) == 50

    def test_conv2d(self):
        layer = nn.Conv2d(3, 16, 3)
        # 3*16*3*3 + 16 = 448
        assert count_parameters(layer) == 448

    def test_empty_module(self):
        mod = nn.Module()
        assert count_parameters(mod) == 0

    def test_nested_modules(self):
        mod = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        # 4*4+4 + 4*2+2 = 20 + 10 = 30
        assert count_parameters(mod) == 30

    def test_frozen_params_excluded(self):
        layer = nn.Linear(10, 5)
        for p in layer.parameters():
            p.requires_grad = False
        assert count_parameters(layer) == 0

    def test_partial_freeze(self):
        mod = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        for p in mod[0].parameters():
            p.requires_grad = False
        # Only mod[1] counts: 4*2+2 = 10
        assert count_parameters(mod) == 10


# ---------------------------------------------------------------------------
# capture_manifest
# ---------------------------------------------------------------------------


class TestCaptureManifest:
    def test_returns_dict_with_required_fields(self):
        m = capture_manifest()
        assert isinstance(m, dict)
        assert "git" in m
        for field in ("sha", "dirty"):
            assert field in m["git"], f"missing git.{field}"
        for field in ("device", "machine", "python"):
            assert field in m, f"missing field: {field}"

    def test_git_sha_is_string(self):
        m = capture_manifest()
        assert isinstance(m["git"]["sha"], str)

    def test_git_dirty_is_bool(self):
        m = capture_manifest()
        assert isinstance(m["git"]["dirty"], bool)

    def test_python_version_present(self):
        import platform

        m = capture_manifest()
        assert m["python"] == platform.python_version()

    def test_timestamp_format(self):
        m = capture_manifest()
        # Format: YYYYMMDDTHHMMSSZ
        ts = m["timestamp"]
        assert ts.endswith("Z")
        assert "T" in ts
        assert len(ts) == 16  # 8+1+6+1


# ---------------------------------------------------------------------------
# Smart gate initialization (ob-t3b.9)
# ---------------------------------------------------------------------------


@_sysmark
class _MockGDN1(_NNModule):
    """Minimal mock of Qwen3_5GatedDeltaNet for testing gate initialization."""

    def __init__(self, hidden_size=64, num_v_heads=4, num_k_heads=4, head_k_dim=16, head_v_dim=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_v_heads = num_v_heads
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_kernel_size = 4
        self.layer_idx = 0
        self.layer_norm_epsilon = 1e-6

        # Core modules with random weights
        self.conv1d = nn.Conv1d(
            self.key_dim * 2 + self.value_dim,
            self.key_dim * 2 + self.value_dim,
            kernel_size=4,
            groups=self.key_dim * 2 + self.value_dim,
            padding=3,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(num_v_heads).uniform_(1, 8)))
        self.norm = nn.Identity()  # simplified
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, num_v_heads, bias=False)
        # This is the key weight we copy from:
        self.in_proj_b = nn.Linear(hidden_size, num_v_heads, bias=False)


@_sysmark
class TestSmartGateInit:
    """Tests for _init_gates_from_gdn1 / smart_init parameter (ob-t3b.9)."""

    def test_smart_init_flag_default_false(self):
        """Default is random init (smart_init=False)."""
        gdn1 = _MockGDN1()
        from gdn2_swap import Qwen3_5GDN2

        gdn2 = Qwen3_5GDN2(gdn1, smart_init=False)
        assert gdn2.smart_init is False

    def test_smart_init_flag_true(self):
        gdn1 = _MockGDN1()
        from gdn2_swap import Qwen3_5GDN2

        gdn2 = Qwen3_5GDN2(gdn1, smart_init=True)
        assert gdn2.smart_init is True

    def test_erase_gate_weight_copies_in_proj_b(self):
        """Each key channel's erase gate row should equal the corresponding
        head's in_proj_b row."""
        torch.manual_seed(123)
        gdn1 = _MockGDN1(hidden_size=32, num_v_heads=4, num_k_heads=4, head_k_dim=8, head_v_dim=8)
        from gdn2_swap import Qwen3_5GDN2

        gdn2 = Qwen3_5GDN2(gdn1, smart_init=True)
        b_weight = gdn1.in_proj_b.weight.data  # [4, 32]
        erase_weight = gdn2.in_proj_erase_gate.weight.data  # [32, 32] (4*8, 32)
        # For head 2, channel 5: should match in_proj_b row 2
        assert torch.allclose(erase_weight[2 * 8 + 5], b_weight[2].to(erase_weight.dtype))

    def test_write_gate_weight_copies_in_proj_b(self):
        """Each value channel's write gate row should equal the corresponding
        head's in_proj_b row."""
        torch.manual_seed(123)
        gdn1 = _MockGDN1(hidden_size=32, num_v_heads=4, num_k_heads=4, head_k_dim=8, head_v_dim=8)
        from gdn2_swap import Qwen3_5GDN2

        gdn2 = Qwen3_5GDN2(gdn1, smart_init=True)
        b_weight = gdn1.in_proj_b.weight.data  # [4, 32]
        write_weight = gdn2.in_proj_write_gate.weight.data  # [32, 32]
        assert torch.allclose(write_weight[3 * 8 + 7], b_weight[3].to(write_weight.dtype))

    def test_smart_init_gates_match_beta(self):
        """sigmoid(gate_output) should approximately equal GDN-1's beta
        when evaluated on the same hidden states."""
        torch.manual_seed(42)
        gdn1 = _MockGDN1(hidden_size=32, num_v_heads=4, num_k_heads=4, head_k_dim=8, head_v_dim=8)
        from gdn2_swap import Qwen3_5GDN2

        gdn2 = Qwen3_5GDN2(gdn1, smart_init=True)

        h = torch.randn(1, 10, 32, dtype=gdn1.in_proj_b.weight.dtype)
        # GDN-1 beta
        beta = torch.sigmoid(gdn1.in_proj_b(h))  # [1, 10, 4]
        # GDN-2 gates (pre-sigmoid → sigmoid)
        erase_raw = gdn2.in_proj_erase_gate(h)  # [1, 10, 32]
        erase_gate = torch.sigmoid(erase_raw)  # [1, 10, 32]
        # Reshape to [1, 10, 4, 8] to check per-head
        erase_gate_reshaped = erase_gate.reshape(1, 10, 4, 8)
        beta_expanded = beta.unsqueeze(-1).expand(1, 10, 4, 8)
        assert torch.allclose(erase_gate_reshaped.float(), beta_expanded.float(), atol=1e-5)

    def test_smart_init_differs_from_random(self):
        """Smart init should produce different weights than random init."""
        torch.manual_seed(42)
        gdn1 = _MockGDN1(hidden_size=32, num_v_heads=4, num_k_heads=4, head_k_dim=8, head_v_dim=8)
        from gdn2_swap import Qwen3_5GDN2

        gdn2_random = Qwen3_5GDN2(gdn1, smart_init=False)
        torch.manual_seed(42)
        gdn2_smart = Qwen3_5GDN2(gdn1, smart_init=True)
        assert not torch.allclose(
            gdn2_random.in_proj_erase_gate.weight.data.float(),
            gdn2_smart.in_proj_erase_gate.weight.data.float(),
        )

    def test_smart_init_with_key_grouping(self):
        """When num_v_heads > num_k_heads, erase gate should average in_proj_b
        across the group."""
        torch.manual_seed(99)
        gdn1 = _MockGDN1(hidden_size=16, num_v_heads=4, num_k_heads=2, head_k_dim=4, head_v_dim=4)
        from gdn2_swap import Qwen3_5GDN2

        gdn2 = Qwen3_5GDN2(gdn1, smart_init=True)
        b_weight = gdn1.in_proj_b.weight.data  # [4, 16]
        erase_weight = gdn2.in_proj_erase_gate.weight.data  # [8, 16] (2*4)
        # Key head 0 → value heads 0,1 (rep=2). Average them.
        expected = b_weight[0:2].mean(dim=0)
        assert torch.allclose(erase_weight[0], expected.to(erase_weight.dtype))
        assert torch.allclose(erase_weight[3], expected.to(erase_weight.dtype))

    def test_swap_function_passes_smart_init(self):
        """swap_gdn1_to_gdn2 should pass smart_init through."""
        from gdn2_swap import Qwen3_5GDN2, swap_gdn1_to_gdn2

        class _MockModel:
            class _TM:
                class _Layer:
                    def __init__(self):
                        self.linear_attn = _MockGDN1(
                            hidden_size=16, num_v_heads=2, num_k_heads=2, head_k_dim=4, head_v_dim=4
                        )

                layers = [_Layer(), _Layer()]

            model = _TM()

        swapped = swap_gdn1_to_gdn2(_MockModel(), [0], smart_init=True)
        assert swapped == [0]
        assert isinstance(_MockModel.model.layers[0].linear_attn, Qwen3_5GDN2)
        assert _MockModel.model.layers[0].linear_attn.smart_init is True
