"""Tests for scripts/ort_gdn_probe.py — GDN reference implementation correctness.

numpy_gdn_reference() is the reference implementation for GDN's delta-rule
recurrence. It validates the ONNX Runtime model output in the feasibility probe
(ob-mrd.16). If this reference is wrong, the probe's correctness verdict is
invalid, so coverage here is deliberately thorough.

All tests use hand-computed small cases — no onnx/onnxruntime required.
"""

from __future__ import annotations

import math

import numpy as np
from scripts.ort_gdn_probe import numpy_gdn_reference

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(q, k, v, g, beta, state0=None):
    """Wrapper that defaults state0 to zeros matching q's column dim."""
    if state0 is None:
        state0 = np.zeros((q.shape[1], q.shape[1]), dtype=np.float32)
    return numpy_gdn_reference(
        q.astype(np.float32),
        k.astype(np.float32),
        v.astype(np.float32),
        g.astype(np.float32),
        beta.astype(np.float32),
        state0.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Single-token tests
# ---------------------------------------------------------------------------


class TestSingleToken:
    """Verify the delta-rule update for a single step with known values."""

    def test_zero_state_zero_attention(self):
        """With zero initial state and identity basis vectors, attention is zero.

        After the update, the state has an outer-product entry but q projects
        onto a zero column, so attn = 0.
        """
        V = 4
        q = np.eye(V, dtype=np.float32)[0:1]  # [1,0,0,0]
        k = np.eye(V, dtype=np.float32)[1:2]  # [0,1,0,0]
        v = np.eye(V, dtype=np.float32)[2:3]  # [0,0,1,0]
        g = np.zeros(1, dtype=np.float32)  # no decay
        beta = np.ones(1, dtype=np.float32)  # full learning rate

        attn, state = _run(q, k, v, g, beta)

        # State: outer(k,v) placed at row=1 (k-index), cols follow v
        expected_state = np.zeros((V, V), dtype=np.float32)
        expected_state[1, 2] = 1.0
        np.testing.assert_allclose(state, expected_state, atol=1e-6)

        # Attention: state @ q = state[:, 0] which is all zeros
        np.testing.assert_allclose(attn, np.zeros((1, V), dtype=np.float32), atol=1e-6)

    def test_known_values_single_token(self):
        """Hand-computed single token with V=2.

        q=[1,0], k=[1,0], v=[2,3], g=[0], beta=[0.5], state0=zeros
        → state = outer([1,0], [1,1.5]) = [[1,1.5],[0,0]]
        → attn = (state @ [1,0]) * 1/sqrt(2) = [1, 0] * 0.7071
        """
        q = np.array([[1.0, 0.0]])
        k = np.array([[1.0, 0.0]])
        v = np.array([[2.0, 3.0]])
        g = np.array([0.0])
        beta = np.array([0.5])
        scale = 1.0 / math.sqrt(2)

        attn, state = _run(q, k, v, g, beta)

        expected_state = np.array([[1.0, 1.5], [0.0, 0.0]], dtype=np.float32)
        expected_attn = np.array([[1.0 * scale, 0.0]], dtype=np.float32)

        np.testing.assert_allclose(state, expected_state, atol=1e-6)
        np.testing.assert_allclose(attn, expected_attn, atol=1e-6)


# ---------------------------------------------------------------------------
# Multi-token tests
# ---------------------------------------------------------------------------


class TestMultiToken:
    """Verify state accumulation and gate decay across multiple tokens."""

    def test_two_tokens_with_decay(self):
        """Hand-computed two-token sequence with gate decay.

        V=2, identity basis, g=[-0.5, 0], beta=[1.0, 0.5]
        See bead ob-9o7 for full derivation.
        """
        V = 2
        scale = 1.0 / math.sqrt(V)
        q = np.array([[1.0, 0.0], [0.0, 1.0]])
        k = np.array([[1.0, 0.0], [0.0, 1.0]])
        v = np.array([[1.0, 1.0], [2.0, 2.0]])
        g = np.array([-0.5, 0.0])
        beta = np.array([1.0, 0.5])

        attn, state = _run(q, k, v, g, beta)

        expected_state = np.array([[1.0, 1.0], [0.5, 1.0]], dtype=np.float32)
        expected_attn = np.array([[1.0 * scale, 0.0], [1.0 * scale, 1.0 * scale]], dtype=np.float32)

        np.testing.assert_allclose(state, expected_state, atol=1e-5)
        np.testing.assert_allclose(attn, expected_attn, atol=1e-5)

    def test_three_tokens_sequential(self):
        """Three-token run — verify state grows monotonically with beta=1, g=0."""
        V = 3
        q = np.eye(V, dtype=np.float32)
        k = np.eye(V, dtype=np.float32)
        v = np.ones((V, V), dtype=np.float32)
        g = np.zeros(V, dtype=np.float32)
        beta = np.ones(V, dtype=np.float32)

        attn, state = _run(q, k, v, g, beta)

        # After processing, state should have 1.0 on each diagonal position
        # (each token writes v=ones to its k-row).
        np.testing.assert_allclose(np.diag(state), [1.0, 1.0, 1.0], atol=1e-5)

    def test_gate_decay_erases_state(self):
        """A very large negative gate should erase prior state.

        g = -100 → exp(-100) ≈ 0, so state resets before the update.
        """
        # First token builds state, second token has massive decay
        q = np.array([[1.0, 0.0], [1.0, 0.0]])
        k = np.array([[1.0, 0.0], [1.0, 0.0]])
        v = np.array([[1.0, 1.0], [5.0, 5.0]])
        g = np.array([0.0, -100.0])
        beta = np.array([1.0, 1.0])

        attn, state = _run(q, k, v, g, beta)

        # After t=1: state ≈ outer([1,0], [5,5]) = [[5,5],[0,0]]
        expected_state = np.array([[5.0, 5.0], [0.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(state, expected_state, atol=1e-4)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Verify behaviour at boundary conditions."""

    def test_zero_beta_no_state_update(self):
        """beta=0 means no learning — state stays at initial value."""
        V = 3
        state0 = np.full((V, V), 0.5, dtype=np.float32)
        q = np.ones((1, V), dtype=np.float32)
        k = np.ones((1, V), dtype=np.float32)
        v = np.ones((1, V), dtype=np.float32)
        g = np.zeros(1, dtype=np.float32)
        beta = np.zeros(1, dtype=np.float32)

        attn, state = _run(q, k, v, g, beta, state0)

        np.testing.assert_allclose(state, state0, atol=1e-6)

    def test_state0_not_mutated(self):
        """The function must copy state0, not modify it in place."""
        V = 2
        state0 = np.zeros((V, V), dtype=np.float32)
        q = np.ones((1, V), dtype=np.float32)
        k = np.ones((1, V), dtype=np.float32)
        v = np.ones((1, V), dtype=np.float32)
        g = np.zeros(1, dtype=np.float32)
        beta = np.ones(1, dtype=np.float32)

        _ = numpy_gdn_reference(q, k, v, g, beta, state0)

        np.testing.assert_allclose(state0, np.zeros((V, V)), atol=1e-6)

    def test_scale_applied_correctly(self):
        """Verify the 1/sqrt(V) scaling is applied to attention output."""
        # scale = 1/sqrt(4) = 0.5
        q = np.array([[1.0, 0.0, 0.0, 0.0]])
        k = np.array([[1.0, 0.0, 0.0, 0.0]])
        v = np.array([[4.0, 0.0, 0.0, 0.0]])
        g = np.zeros(1, dtype=np.float32)
        beta = np.ones(1, dtype=np.float32)

        attn, state = _run(q, k, v, g, beta)

        # state = outer([1,0,0,0], [4,0,0,0]) = [[4,0,0,0],...]
        # attn = (state @ [1,0,0,0]) * 0.5 = [4,0,0,0] * 0.5 = [2,0,0,0]
        expected_attn = np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(attn, expected_attn, atol=1e-6)


# ---------------------------------------------------------------------------
# Property-based / invariant tests
# ---------------------------------------------------------------------------


class TestInvariants:
    """Mathematical properties that must hold for any input."""

    def test_output_shapes_match_input(self):
        """attn shape = q shape; state shape = (V, V)."""
        seq_len, V = 5, 8
        rng = np.random.default_rng(42)
        q = rng.standard_normal((seq_len, V)).astype(np.float32)
        k = rng.standard_normal((seq_len, V)).astype(np.float32)
        v = rng.standard_normal((seq_len, V)).astype(np.float32)
        g = rng.standard_normal(seq_len).astype(np.float32) * 0.01
        beta = rng.uniform(0.1, 1.0, seq_len).astype(np.float32)

        attn, state = _run(q, k, v, g, beta)

        assert attn.shape == (seq_len, V)
        assert state.shape == (V, V)

    def test_deterministic(self):
        """Same inputs → same outputs (no randomness inside the function)."""
        V = 4
        rng = np.random.default_rng(123)
        q = rng.standard_normal((3, V)).astype(np.float32)
        k = rng.standard_normal((3, V)).astype(np.float32)
        v = rng.standard_normal((3, V)).astype(np.float32)
        g = rng.standard_normal(3).astype(np.float32) * 0.01
        beta = rng.uniform(0.1, 1.0, 3).astype(np.float32)

        a1, s1 = _run(q, k, v, g, beta)
        a2, s2 = _run(q, k, v, g, beta)

        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(s1, s2)

    def test_seq_len_one_attn_uses_updated_state(self):
        """Attention at token t uses the *post-update* state, not pre-update."""
        # With beta=1, g=0: state = outer(k, v), attn = state @ q * scale
        q = np.array([[1.0, 0.0]])
        k = np.array([[1.0, 0.0]])
        v = np.array([[3.0, 0.0]])
        g = np.zeros(1, dtype=np.float32)
        beta = np.ones(1, dtype=np.float32)

        attn, state = _run(q, k, v, g, beta)

        # attn should reflect the updated state (which has v[0]=3 at [0,0]),
        # not the pre-update zero state.
        assert attn[0, 0] > 0, "Attention must use post-update state"
