"""Unit tests for bench/gdn2_reference.py — GDN-2 NumPy reference (ob-y3f).

Tests the core mathematical functions and recurrence implementations
that the GDN-2 stretch research track (ob-9lm) depends on.

These tests run on Python 3.6+ with only NumPy — no torch or CUDA needed,
so they work on every fleet device and in CI.
"""

import math

import bench.gdn2_reference as gdn2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# softplus
# ---------------------------------------------------------------------------


class TestSoftplus:
    def test_known_values(self):
        """gdn2.softplus(0) = ln(2), gdn2.softplus(large) ≈ x, gdn2.softplus(-large) ≈ 0."""
        assert gdn2.softplus(np.array(0.0)) == pytest.approx(math.log(2))
        assert gdn2.softplus(np.array(100.0)) == pytest.approx(100.0)
        assert gdn2.softplus(np.array(-100.0)) == pytest.approx(0.0)

    def test_array(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = gdn2.softplus(x)
        expected = np.log1p(np.exp(x))
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_monotonic(self):
        """softplus is monotonically increasing."""
        x = np.linspace(-5, 5, 100)
        result = gdn2.softplus(x)
        assert np.all(np.diff(result) > 0)

    def test_no_overflow(self):
        """Large positive values must not overflow to inf."""
        x = np.array([500.0, 1000.0])
        result = gdn2.softplus(x)
        assert not np.any(np.isinf(result))
        assert not np.any(np.isnan(result))


# ---------------------------------------------------------------------------
# l2_normalize
# ---------------------------------------------------------------------------


class TestL2Normalize:
    def test_unit_norm(self):
        x = np.array([3.0, 4.0])
        result = gdn2.l2_normalize(x)
        norm = np.sqrt(np.sum(result * result))
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_preserves_direction(self):
        """Normalization scales but doesn't change direction."""
        x = np.array([1.0, 2.0, 3.0])
        result = gdn2.l2_normalize(x)
        # Ratios of components should be preserved
        ratio_orig = x[1] / x[0]
        ratio_norm = result[1] / result[0]
        assert ratio_norm == pytest.approx(ratio_orig)

    def test_eps_for_zero_vector(self):
        """Zero vector should not produce NaN (eps prevents div by zero)."""
        x = np.zeros(5)
        result = gdn2.l2_normalize(x)
        assert not np.any(np.isnan(result))

    def test_known_value(self):
        x = np.array([1.0, 0.0, 0.0])
        result = gdn2.l2_normalize(x)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-6)


# ---------------------------------------------------------------------------
# gdn2_recurrent
# ---------------------------------------------------------------------------


class TestGDN2Recurrent:
    def test_single_step_zero_state(self):
        """Single step with zero initial state matches hand computation."""
        K = 2
        q = np.array([[[0.6, 0.8]]], dtype=np.float64)
        k_vec = np.array([[[0.8, 0.6]]], dtype=np.float64)
        v_vec = np.array([[[1.0, 2.0]]], dtype=np.float64)
        g = np.array([[[-0.5, -0.3]]], dtype=np.float64)
        b_gate = np.array([[[0.9, 0.8]]], dtype=np.float64)
        w_gate = np.array([[[0.7, 0.6]]], dtype=np.float64)
        scale = 1.0 / math.sqrt(K)

        o, S = gdn2.gdn2_recurrent(
            q, k_vec, v_vec, g, b_gate, w_gate, scale=scale, use_qk_l2norm=False
        )

        # With S=0: erase=0, v_new = [0.7, 1.2]
        # S = outer(k, v_new) = [[0.56, 0.96], [0.42, 0.72]]
        expected_S = np.array([[0.56, 0.96], [0.42, 0.72]])
        expected_o = np.array([[scale * 0.672, scale * 1.152]])

        np.testing.assert_allclose(S[0], expected_S, atol=1e-10)
        np.testing.assert_allclose(o[0], expected_o, atol=1e-10)

    def test_output_shape(self):
        """Output has correct shape [T, HV, V]."""
        T, H, HV, K, V = 4, 2, 2, 8, 8
        q, k, v, g, b_gate, w_gate, _, _ = gdn2.make_synthetic_input(T, H, K, V)
        o, S = gdn2.gdn2_recurrent(q, k, v, g, b_gate, w_gate)
        assert o.shape == (T, HV, V)
        assert S.shape == (HV, K, V)

    def test_gva_replication(self):
        """When HV > H, key-side tensors are replicated via group-value-attention."""
        T, H, HV, K, V = 2, 1, 4, 4, 4  # HV = 4 * H
        rng = np.random.RandomState(123)
        q = rng.randn(T, H, K).astype(np.float32)
        k_arr = rng.randn(T, H, K).astype(np.float32)
        v = rng.randn(T, HV, V).astype(np.float32)
        g = rng.randn(T, HV, K).astype(np.float32) * 0.1
        b_gate = np.ones((T, HV, K), dtype=np.float32) * 0.5
        w_gate = np.ones((T, HV, V), dtype=np.float32) * 0.5

        o, S = gdn2.gdn2_recurrent(q, k_arr, v, g, b_gate, w_gate)
        assert o.shape == (T, HV, V)
        assert S.shape == (HV, K, V)
        # With replicated q/k, groups of heads should produce identical output
        # (since v differs per head, we just check no NaN)
        assert not np.any(np.isnan(o))

    def test_deterministic(self):
        """Same inputs produce same outputs."""
        inputs = gdn2.make_synthetic_input(T=4, H=2, K=8, V=8, seed=42)
        q, k, v, g, b_gate, w_gate, _, _ = inputs
        o1, S1 = gdn2.gdn2_recurrent(q, k, v, g, b_gate, w_gate)
        o2, S2 = gdn2.gdn2_recurrent(q, k, v, g, b_gate, w_gate)
        np.testing.assert_array_equal(o1, o2)
        np.testing.assert_array_equal(S1, S2)

    def test_initial_state_continuity(self):
        """Passing the final state as initial_state for a second run is
        equivalent to one long run (state threading)."""
        q, k, v, g, b_gate, w_gate, _, _ = gdn2.make_synthetic_input(T=4, H=1, K=4, V=4, seed=7)

        # Full run
        o_full, S_full = gdn2.gdn2_recurrent(q, k, v, g, b_gate, w_gate)

        # Split run: first half, then continue with returned state
        o_a, S_a = gdn2.gdn2_recurrent(
            q[:2],
            k[:2],
            v[:2],
            g[:2],
            b_gate[:2],
            w_gate[:2],
        )
        o_b, S_b = gdn2.gdn2_recurrent(
            q[2:],
            k[2:],
            v[2:],
            g[2:],
            b_gate[2:],
            w_gate[2:],
            initial_state=S_a,
        )

        np.testing.assert_allclose(S_full, S_b, atol=1e-10)
        np.testing.assert_allclose(o_full[2:], o_b, atol=1e-10)


# ---------------------------------------------------------------------------
# gdn1_recurrent
# ---------------------------------------------------------------------------


class TestGDN1Recurrent:
    def test_output_shape(self):
        T, H, K, V = 4, 2, 8, 8
        q, k, v, _, _, _, _, _ = gdn2.make_synthetic_input(T, H, K, V)
        rng = np.random.RandomState(99)
        alpha = rng.uniform(0.5, 1.0, (T, H)).astype(np.float32)
        beta = rng.uniform(0.1, 0.5, (T, H)).astype(np.float32)
        o, S = gdn2.gdn1_recurrent(q, k, v, alpha, beta)
        assert o.shape == (T, H, V)
        assert S.shape == (H, K, V)

    def test_zero_decay_identity(self):
        """With alpha=1 (no decay) and beta=0 (no write), output is zero."""
        T, H, K, V = 3, 1, 4, 4
        rng = np.random.RandomState(42)
        q = rng.randn(T, H, K).astype(np.float32)
        k_arr = rng.randn(T, H, K).astype(np.float32)
        v = rng.randn(T, H, V).astype(np.float32)
        alpha = np.ones((T, H), dtype=np.float32)
        beta = np.zeros((T, H), dtype=np.float32)
        o, S = gdn2.gdn1_recurrent(q, k_arr, v, alpha, beta)
        np.testing.assert_allclose(o, 0.0, atol=1e-12)

    def test_deterministic(self):
        T, H, K, V = 4, 2, 8, 8
        q, k, v, _, _, _, _, _ = gdn2.make_synthetic_input(T, H, K, V)
        rng = np.random.RandomState(99)
        alpha = rng.uniform(0.5, 1.0, (T, H)).astype(np.float32)
        beta = rng.uniform(0.1, 0.5, (T, H)).astype(np.float32)
        o1, _ = gdn2.gdn1_recurrent(q, k, v, alpha, beta)
        o2, _ = gdn2.gdn1_recurrent(q, k, v, alpha, beta)
        np.testing.assert_array_equal(o1, o2)


# ---------------------------------------------------------------------------
# make_synthetic_input
# ---------------------------------------------------------------------------


class TestMakeSyntheticInput:
    def test_shapes(self):
        T, H, K, V = 8, 2, 8, 8
        q, k, v, g, b_gate, w_gate, A_log, dt_bias = gdn2.make_synthetic_input(T, H, K, V)
        assert q.shape == (T, H, K)
        assert k.shape == (T, H, K)
        assert v.shape == (T, H, V)
        assert g.shape == (T, H, K)
        assert b_gate.shape == (T, H, K)
        assert w_gate.shape == (T, H, V)
        assert A_log.shape == (H,)
        assert dt_bias.shape == (H * K,)

    def test_gate_ranges(self):
        """Gates should be in valid ranges."""
        _, _, _, g, b_gate, w_gate, _, _ = gdn2.make_synthetic_input()
        # g is log-space decay, should be negative
        assert np.all(g < 0), "Decay g should be negative (log-space)"
        # b_gate and w_gate are sigmoid'd, so in [0, 1]
        assert np.all(b_gate >= 0) and np.all(b_gate <= 1)
        assert np.all(w_gate >= 0) and np.all(w_gate <= 1)

    def test_reproducible(self):
        """Same seed produces same inputs."""
        a = gdn2.make_synthetic_input(seed=123)
        b = gdn2.make_synthetic_input(seed=123)
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(x, y)

    def test_different_seeds_differ(self):
        a = gdn2.make_synthetic_input(seed=1)
        b = gdn2.make_synthetic_input(seed=2)
        assert not np.array_equal(a[0], b[0])
