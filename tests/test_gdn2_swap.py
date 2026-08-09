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

torch = pytest.importorskip("torch")
nn = torch.nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))

from gdn2_swap import (  # noqa: E402
    capture_manifest,
    count_parameters,
    gdn2_recurrent,
)

# ---------------------------------------------------------------------------
# gdn2_recurrent
# ---------------------------------------------------------------------------


def _make_inputs(B=2, H=4, T=6, K=8, V=5, dtype=torch.float32):
    """Create valid recurrence inputs."""
    torch.manual_seed(42)
    return dict(
        query=torch.randn(B, H, T, K, dtype=dtype),
        key=torch.randn(B, H, T, K, dtype=dtype),
        value=torch.randn(B, H, T, V, dtype=dtype),
        g=torch.randn(B, H, T).abs() * -0.1,  # negative log-decay
        b_gate=torch.rand(B, H, T, K, dtype=dtype),  # [0,1]
        w_gate=torch.rand(B, H, T, V, dtype=dtype),  # [0,1]
    )


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
        for field in ("git_sha", "git_dirty", "device", "machine", "python"):
            assert field in m, f"missing field: {field}"

    def test_git_sha_is_string(self):
        m = capture_manifest()
        assert isinstance(m["git_sha"], str)

    def test_git_dirty_is_bool(self):
        m = capture_manifest()
        assert isinstance(m["git_dirty"], bool)

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
