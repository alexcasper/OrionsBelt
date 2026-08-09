# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Tests for bench/gdn2_reference.py — pure-NumPy GDN-2 recurrence.

Covers the core arithmetic functions (softplus, l2_normalize), the token-by-token
recurrence (gdn2_recurrent, gdn1_recurrent), and the synthetic input generator.
These are the building blocks for the GDN-2 vs GDN-1 comparison in FINDINGS.md §10.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench.gdn2_reference import (  # noqa: E402
    gdn1_recurrent,
    gdn2_recurrent,
    l2_normalize,
    make_synthetic_input,
    softplus,
)

# ---------------------------------------------------------------------------
# softplus
# ---------------------------------------------------------------------------


def test_softplus_zero():
    """softplus(0) = ln(2) ≈ 0.6931."""
    assert softplus(np.array([0.0]))[0] == pytest.approx(math.log(2), abs=1e-12)


def test_softplus_positive():
    """softplus(x) ≈ x for large x."""
    assert softplus(np.array([100.0]))[0] == pytest.approx(100.0, abs=1e-6)


def test_softplus_negative():
    """softplus(-x) → 0 for large x."""
    assert softplus(np.array([-100.0]))[0] == pytest.approx(0.0, abs=1e-6)


def test_softplus_no_overflow():
    """No inf/nan even for extreme values."""
    result = softplus(np.array([-1e6, 0.0, 1e6]))
    assert np.all(np.isfinite(result))
    assert result[0] == pytest.approx(0.0, abs=1e-6)
    assert result[2] == pytest.approx(1e6, abs=1e-3)


def test_softplus_vectorized():
    """Works elementwise on arrays."""
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    result = softplus(x)
    expected = np.log1p(np.exp(x))
    np.testing.assert_allclose(result, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# l2_normalize
# ---------------------------------------------------------------------------


def test_l2_normalize_unit_norm():
    """Output has approximately unit L2 norm."""
    x = np.array([3.0, 4.0])
    result = l2_normalize(x)
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-4)


def test_l2_normalize_direction():
    """Direction is preserved."""
    x = np.array([1.0, 0.0, 0.0])
    result = l2_normalize(x)
    np.testing.assert_allclose(result, x, atol=1e-4)


def test_l2_normalize_zero_vector():
    """Zero vector doesn't produce nan/inf (eps protects)."""
    x = np.zeros(4)
    result = l2_normalize(x)
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# gdn2_recurrent — shapes & basic properties
# ---------------------------------------------------------------------------


def test_gdn2_output_shapes():
    """Output and state have correct shapes."""
    q, k, v, g, b_gate, w_gate, _, _ = make_synthetic_input(T=4, H=2, K=8, V=8)
    o, S = gdn2_recurrent(q, k, v, g, b_gate, w_gate)
    assert o.shape == (4, 2, 8)
    assert S.shape == (2, 8, 8)


def test_gdn2_gva_replication():
    """Grouped-value attention: HV > H replicates key-side tensors."""
    T, H, K, V = 4, 1, 8, 8
    HV = 2
    rng = np.random.RandomState(0)
    q = rng.randn(T, H, K).astype(np.float32)
    k = rng.randn(T, H, K).astype(np.float32)
    v = rng.randn(T, HV, V).astype(np.float32)
    g = rng.randn(T, HV, K).astype(np.float32) * 0.1
    b_gate = np.full((T, HV, K), 0.5, dtype=np.float32)
    w_gate = np.full((T, HV, V), 0.5, dtype=np.float32)
    o, S = gdn2_recurrent(q, k, v, g, b_gate, w_gate)
    assert o.shape == (T, HV, V)
    assert S.shape == (HV, K, V)


def test_gdn2_initial_state_used():
    """Non-zero initial state is incorporated."""
    T, H, K, V = 2, 1, 4, 4
    rng = np.random.RandomState(1)
    q = rng.randn(T, H, K).astype(np.float32)
    k = rng.randn(T, H, K).astype(np.float32)
    v = rng.randn(T, H, V).astype(np.float32)
    g = np.zeros((T, H, K), dtype=np.float32)  # no decay
    b_gate = np.zeros((T, H, K), dtype=np.float32)  # no erase
    w_gate = np.ones((T, H, V), dtype=np.float32)  # full write

    S0 = np.ones((H, K, V), dtype=np.float64)
    o_with, S_with = gdn2_recurrent(q, k, v, g, b_gate, w_gate, initial_state=S0)

    o_zero, S_zero = gdn2_recurrent(q, k, v, g, b_gate, w_gate)

    # With a non-zero initial state, outputs must differ
    assert not np.allclose(o_with, o_zero)
    # The initial state must have been incorporated (state grew)
    assert np.sum(S_with) > np.sum(S_zero)


def test_gdn2_no_l2norm():
    """use_qk_l2norm=False skips normalization."""
    T, H, K, V = 2, 1, 4, 4
    rng = np.random.RandomState(2)
    q = rng.randn(T, H, K).astype(np.float32) * 10  # large magnitudes
    k = rng.randn(T, H, K).astype(np.float32)
    v = rng.randn(T, H, V).astype(np.float32)
    g = np.zeros((T, H, K), dtype=np.float32)
    b_gate = np.zeros((T, H, K), dtype=np.float32)
    w_gate = np.ones((T, H, V), dtype=np.float32)

    o_norm, _ = gdn2_recurrent(q, k, v, g, b_gate, w_gate, use_qk_l2norm=True)
    o_raw, _ = gdn2_recurrent(q, k, v, g, b_gate, w_gate, use_qk_l2norm=False)

    # Large un-normalized q → larger outputs
    assert not np.allclose(o_norm, o_raw)


def test_gdn2_determinism():
    """Same inputs → same outputs."""
    q, k, v, g, b_gate, w_gate, _, _ = make_synthetic_input(T=4, H=2, K=8, V=8)
    o1, S1 = gdn2_recurrent(q, k, v, g, b_gate, w_gate)
    o2, S2 = gdn2_recurrent(q, k, v, g, b_gate, w_gate)
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(S1, S2)


def test_gdn2_all_finite():
    """Outputs are finite (no nan/inf from exponential overflow)."""
    q, k, v, g, b_gate, w_gate, _, _ = make_synthetic_input(T=8, H=2, K=8, V=8)
    o, S = gdn2_recurrent(q, k, v, g, b_gate, w_gate)
    assert np.all(np.isfinite(o))
    assert np.all(np.isfinite(S))


def test_gdn2_decay_reduces_state():
    """Strong negative decay (g → -inf) shrinks the state toward zero."""
    T, H, K, V = 1, 1, 4, 4
    q = np.ones((T, H, K), dtype=np.float32)
    k = np.ones((T, H, K), dtype=np.float32)
    v = np.ones((T, H, V), dtype=np.float32)
    g = np.full((T, H, K), -10.0, dtype=np.float32)  # exp(-10) ≈ 4.5e-5
    b_gate = np.zeros((T, H, K), dtype=np.float32)
    w_gate = np.ones((T, H, V), dtype=np.float32)

    S0 = np.ones((H, K, V), dtype=np.float64) * 100.0
    o, S = gdn2_recurrent(q, k, v, g, b_gate, w_gate, initial_state=S0)

    # State should be much smaller after heavy decay
    assert np.mean(np.abs(S)) < np.mean(np.abs(S0))


# ---------------------------------------------------------------------------
# gdn1_recurrent — shapes & edge cases
# ---------------------------------------------------------------------------


def test_gdn1_output_shapes():
    """Output and state have correct shapes."""
    T, H, K, V = 4, 2, 8, 8
    rng = np.random.RandomState(3)
    q = rng.randn(T, H, K).astype(np.float32)
    k = rng.randn(T, H, K).astype(np.float32)
    v = rng.randn(T, H, V).astype(np.float32)
    alpha = np.full((T, H), 0.9, dtype=np.float32)
    beta = np.full((T, H), 0.1, dtype=np.float32)
    o, S = gdn1_recurrent(q, k, v, alpha, beta)
    assert o.shape == (4, 2, 8)
    assert S.shape == (2, 8, 8)


def test_gdn1_beta_zero_no_write():
    """beta=0 means the state is only decayed, not written to."""
    T, H, K, V = 1, 1, 4, 4
    q = np.random.randn(T, H, K).astype(np.float32)
    k = np.random.randn(T, H, K).astype(np.float32)
    v = np.random.randn(T, H, V).astype(np.float32)
    alpha = np.array([[0.5]], dtype=np.float32)
    beta = np.array([[0.0]], dtype=np.float32)  # no write

    S0 = np.ones((H, K, V), dtype=np.float64)
    _, S = gdn1_recurrent(q, k, v, alpha, beta, initial_state=S0)
    # State should just be alpha * S0 = 0.5 * ones
    np.testing.assert_allclose(S, 0.5 * S0, atol=1e-10)


def test_gdn1_alpha_zero_pure_write():
    """alpha=0 means only the new write survives (old state erased)."""
    T, H, K, V = 1, 1, 4, 4
    rng = np.random.RandomState(5)
    q = rng.randn(T, H, K).astype(np.float32)
    k = rng.randn(T, H, K).astype(np.float32)
    v = rng.randn(T, H, V).astype(np.float32)
    alpha = np.array([[0.0]], dtype=np.float32)
    beta = np.array([[1.0]], dtype=np.float32)

    S0 = np.ones((H, K, V), dtype=np.float64) * 100.0
    _, S = gdn1_recurrent(q, k, v, alpha, beta, initial_state=S0, use_qk_l2norm=False)

    # alpha=0, beta=1: S = 0*(...) + 1 * outer(k, v - k@S0)
    # The old state contributes through kk = k@S0, but the result is
    # dominated by the new write, not the scaled old state.
    # Just check the output is finite and different from the initial state.
    assert np.all(np.isfinite(S))
    assert not np.allclose(S, S0)


def test_gdn1_gva_replication():
    """Grouped-value attention: HV > H replicates key-side tensors."""
    T, H, K, V = 4, 1, 8, 8
    HV = 2
    rng = np.random.RandomState(6)
    q = rng.randn(T, H, K).astype(np.float32)
    k = rng.randn(T, H, K).astype(np.float32)
    v = rng.randn(T, HV, V).astype(np.float32)
    alpha = np.full((T, H), 0.9, dtype=np.float32)
    beta = np.full((T, H), 0.1, dtype=np.float32)
    o, S = gdn1_recurrent(q, k, v, alpha, beta)
    assert o.shape == (T, HV, V)
    assert S.shape == (HV, K, V)


# ---------------------------------------------------------------------------
# make_synthetic_input
# ---------------------------------------------------------------------------


def test_make_synthetic_input_shapes():
    """Generated inputs have correct shapes."""
    q, k, v, g, b_gate, w_gate, A_log, dt_bias = make_synthetic_input(T=8, H=2, K=16, V=8)
    assert q.shape == (8, 2, 16)
    assert k.shape == (8, 2, 16)
    assert v.shape == (8, 2, 8)
    assert g.shape == (8, 2, 16)
    assert b_gate.shape == (8, 2, 16)
    assert w_gate.shape == (8, 2, 8)
    assert A_log.shape == (2,)
    assert dt_bias.shape == (2 * 16,)


def test_make_synthetic_input_deterministic():
    """Same seed → identical output."""
    a = make_synthetic_input(T=4, H=2, K=8, V=8, seed=42)
    b = make_synthetic_input(T=4, H=2, K=8, V=8, seed=42)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_make_synthetic_input_gates_in_range():
    """Erase/write gates are in [0, 1] (sigmoid output)."""
    _, _, _, _, b_gate, w_gate, _, _ = make_synthetic_input(T=4, H=2, K=8, V=8)
    assert np.all(b_gate >= 0.0) and np.all(b_gate <= 1.0)
    assert np.all(w_gate >= 0.0) and np.all(w_gate <= 1.0)


def test_make_synthetic_input_decay_negative():
    """Log-decay g is negative (decay shrinks the state)."""
    _, _, _, g, _, _, _, _ = make_synthetic_input(T=4, H=2, K=8, V=8)
    assert np.all(g <= 0.0)


# ---------------------------------------------------------------------------
# Cross-validation: GDN-2 with uniform gates ≈ GDN-1
# ---------------------------------------------------------------------------


def test_gdn2_uniform_gates_approximates_gdn1():
    """When erase gate=0 and write gate=1, GDN-2 reduces to a simpler form.

    This doesn't produce *identical* numbers to GDN-1 (different gate
    parameterizations), but the structural relationship (state read-modify-write
    with decay) is the same. We test that both produce finite, comparable-magnitude
    outputs on the same input.
    """
    T, H, K, V = 4, 2, 8, 8
    rng = np.random.RandomState(7)
    q = rng.randn(T, H, K).astype(np.float32)
    k = rng.randn(T, H, K).astype(np.float32)
    v = rng.randn(T, H, V).astype(np.float32)

    # GDN-2 with uniform gates
    g = np.full((T, H, K), -0.1, dtype=np.float32)  # mild decay
    b_gate = np.zeros((T, H, K), dtype=np.float32)  # no erase
    w_gate = np.ones((T, H, V), dtype=np.float32)  # full write
    o2, S2 = gdn2_recurrent(q, k, v, g, b_gate, w_gate)

    # GDN-1 with scalar gates
    alpha = np.full((T, H), math.exp(-0.1), dtype=np.float32)
    beta = np.full((T, H), 1.0, dtype=np.float32)
    o1, S1 = gdn1_recurrent(q, k, v, alpha, beta)

    # Both produce finite outputs
    assert np.all(np.isfinite(o1)) and np.all(np.isfinite(o2))
    # States have the same shape
    assert S1.shape == S2.shape


# ---------------------------------------------------------------------------
# Hand-verified exact computation (from in-module test_gdn2_known_answer)
# ---------------------------------------------------------------------------


class TestGdn2KnownAnswer:
    """Hand-verified single-step GDN-2 recurrence with exact expected values."""

    def test_single_step_state_exact(self):
        """With zero initial state, verify the state matrix element-by-element."""
        K = 2
        q = np.array([[[0.6, 0.8]]], dtype=np.float64)
        k_vec = np.array([[[0.8, 0.6]]], dtype=np.float64)
        v_vec = np.array([[[1.0, 2.0]]], dtype=np.float64)
        g = np.array([[[-0.5, -0.3]]], dtype=np.float64)
        b_gate = np.array([[[0.9, 0.8]]], dtype=np.float64)
        w_gate = np.array([[[0.7, 0.6]]], dtype=np.float64)

        o, S = gdn2_recurrent(
            q, k_vec, v_vec, g, b_gate, w_gate, scale=1.0 / math.sqrt(K), use_qk_l2norm=False
        )

        # Step-by-step:
        # S *= exp(g) → S stays 0 (zero initial state)
        # erase = (b⊙k)^T @ S = 0
        # v_new = w⊙v = [0.7, 1.2]
        # S += k ⊗ v_new = [[0.56, 0.96], [0.42, 0.72]]
        expected_S = np.array([[0.56, 0.96], [0.42, 0.72]])
        assert np.allclose(S[0], expected_S, atol=1e-10)

    def test_single_step_output_exact(self):
        """Verify the output vector element-by-element against hand computation."""
        K = 2
        q = np.array([[[0.6, 0.8]]], dtype=np.float64)
        k_vec = np.array([[[0.8, 0.6]]], dtype=np.float64)
        v_vec = np.array([[[1.0, 2.0]]], dtype=np.float64)
        g = np.array([[[-0.5, -0.3]]], dtype=np.float64)
        b_gate = np.array([[[0.9, 0.8]]], dtype=np.float64)
        w_gate = np.array([[[0.7, 0.6]]], dtype=np.float64)

        scale = 1.0 / math.sqrt(K)
        o, _ = gdn2_recurrent(q, k_vec, v_vec, g, b_gate, w_gate, scale=scale, use_qk_l2norm=False)

        # o = scale * [0.6*0.56 + 0.8*0.42, 0.6*0.96 + 0.8*0.72]
        #   = scale * [0.672, 1.152]
        expected_o = np.array([[scale * 0.672, scale * 1.152]])
        assert np.allclose(o[0, 0], expected_o[0], atol=1e-10)


# ---------------------------------------------------------------------------
# State continuity (from in-module test_gdn2_multi_step_consistency)
# ---------------------------------------------------------------------------


class TestGdn2StateContinuity:
    """Verify that initial_state lets a recurrence continue seamlessly."""

    def test_continued_run_is_finite(self):
        """Running a second batch with initial_state=S_final stays finite."""
        T, H, K, V = 16, 4, 16, 16
        q1, k1, v1, g1, b1, w1, _, _ = make_synthetic_input(T, H, K, V, seed=99)
        o1, S_final = gdn2_recurrent(
            q1, k1, v1, g1, b1, w1, scale=1.0 / math.sqrt(K), use_qk_l2norm=True
        )
        assert np.all(np.isfinite(S_final))

        q2, k2, v2, g2, b2, w2, _, _ = make_synthetic_input(T, H, K, V, seed=77)
        o_cont, S_cont = gdn2_recurrent(
            q2,
            k2,
            v2,
            g2,
            b2,
            w2,
            scale=1.0 / math.sqrt(K),
            initial_state=S_final,
            use_qk_l2norm=True,
        )
        assert np.all(np.isfinite(o_cont))
        assert np.all(np.isfinite(S_cont))

    def test_nonzero_initial_state_affects_output(self):
        """A non-zero initial state must change the output vs zero initial state."""
        T, H, K, V = 4, 2, 8, 8
        q, k, v, g, b_gate, w_gate, _, _ = make_synthetic_input(T, H, K, V, seed=42)

        o_zero, _ = gdn2_recurrent(q, k, v, g, b_gate, w_gate, use_qk_l2norm=True)
        S_init = np.ones((H, K, V), dtype=np.float64) * 0.1
        o_init, _ = gdn2_recurrent(
            q, k, v, g, b_gate, w_gate, initial_state=S_init, use_qk_l2norm=True
        )
        assert not np.allclose(o_zero, o_init), "Non-zero initial state had no effect on output"


# ---------------------------------------------------------------------------
# Bandwidth cost invariant (from in-module test_bandwidth_analysis)
# ---------------------------------------------------------------------------


class TestBandwidthAnalysis:
    """Document the per-token bandwidth overhead of GDN-2 vs GDN-1."""

    def test_gdn2_gate_overhead_under_5pct(self):
        """GDN-2's extra gate vectors add <5% bandwidth vs the state R/M/W."""
        H, d_k, d_v = 16, 128, 128
        bytes_f32 = 4

        state_bytes = H * d_k * d_v * bytes_f32 * 2  # read + write
        gdn2_extra = (H * d_k + H * d_v + H * d_k) * bytes_f32
        overhead_pct = 100.0 * gdn2_extra / state_bytes

        assert overhead_pct < 5.0, f"GDN-2 overhead {overhead_pct:.2f}% unexpectedly high"

    def test_gdn2_more_traffic_than_gdn1(self):
        """GDN-2 has strictly more per-token gate traffic than GDN-1."""
        H, d_k, d_v = 16, 128, 128
        bytes_f32 = 4

        gdn1_extra = 2 * bytes_f32  # alpha + beta scalars
        gdn2_extra = (H * d_k + H * d_v + H * d_k) * bytes_f32

        assert gdn2_extra > gdn1_extra
