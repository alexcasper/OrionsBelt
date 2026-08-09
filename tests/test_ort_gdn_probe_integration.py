#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for scripts/ort_gdn_probe.py — ONNX model building + ORT execution.

These tests require onnx + onnxruntime. They verify that:
1. build_gdn_loop_model() produces a valid ONNX model
2. The model executes correctly under ORT CPU EP
3. ORT output matches the numpy reference implementation

The existing test_ort_gdn_probe.py covers numpy_gdn_reference() only.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import checker as onnx_checker

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from scripts.ort_gdn_probe import build_gdn_loop_model, numpy_gdn_reference  # noqa: E402

# ---------------------------------------------------------------------------
# build_gdn_loop_model()
# ---------------------------------------------------------------------------


class TestBuildGdnLoopModel:
    """Verify the ONNX model builder produces valid, runnable models."""

    def test_returns_model_and_data(self):
        model, q, k, v, g, beta = build_gdn_loop_model(head_dim=4, seq_len=3)
        assert model is not None
        assert q.shape == (3, 4)
        assert k.shape == (3, 4)
        assert v.shape == (3, 4)
        assert g.shape == (3,)
        assert beta.shape == (3,)

    def test_model_passes_checker(self):
        model, *_ = build_gdn_loop_model(head_dim=4, seq_len=2)
        onnx_checker.check_model(model, full_check=True)

    def test_provided_data_used(self):
        q = np.array([[1, 0, 0, 0]], dtype=np.float32)
        k = np.array([[0, 1, 0, 0]], dtype=np.float32)
        v = np.array([[0, 0, 1, 0]], dtype=np.float32)
        g = np.array([0.0], dtype=np.float32)
        beta = np.array([1.0], dtype=np.float32)

        model, q_out, k_out, v_out, g_out, beta_out = build_gdn_loop_model(
            head_dim=4, seq_len=1, q_data=q, k_data=k, v_data=v, g_data=g, beta_data=beta
        )
        np.testing.assert_array_equal(q_out, q)
        np.testing.assert_array_equal(k_out, k)
        np.testing.assert_array_equal(v_out, v)

    def test_default_scale_is_1_over_sqrt_v(self):
        """Default scale should be 1/sqrt(head_dim)."""
        model, *_ = build_gdn_loop_model(head_dim=16, seq_len=2)
        # The scale is baked into the model — verify via ORT execution
        # against numpy reference with matching scale.
        import onnxruntime as ort

        model, q, k, v, g, beta = build_gdn_loop_model(head_dim=16, seq_len=4)
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        state0 = np.zeros((16, 16), dtype=np.float32)
        results = sess.run(
            None,
            {"trip_count": np.array(4, dtype=np.int64), "state0": state0},
        )
        attn_ort = results[1]

        attn_ref, _ = numpy_gdn_reference(q, k, v, g, beta, state0)
        np.testing.assert_allclose(attn_ort, attn_ref, atol=1e-4)


# ---------------------------------------------------------------------------
# ORT execution + correctness
# ---------------------------------------------------------------------------


class TestOrtExecution:
    """Verify the ONNX model runs under ORT and matches the numpy reference."""

    @staticmethod
    def _run_model(V, seq_len, **kwargs):
        """Helper: build model, run under ORT, return (ort_attn, ort_state, data)."""
        import onnxruntime as ort

        model, q, k, v, g, beta = build_gdn_loop_model(head_dim=V, seq_len=seq_len, **kwargs)
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        state0 = np.zeros((V, V), dtype=np.float32)
        results = sess.run(
            None,
            {"trip_count": np.array(seq_len, dtype=np.int64), "state0": state0},
        )
        state_final, attn_all = results
        return attn_all, state_final, (q, k, v, g, beta)

    def test_single_token_correctness(self):
        """V=4, seq_len=1: ORT must match numpy reference."""
        attn_ort, state_ort, (q, k, v, g, beta) = self._run_model(4, 1)
        state0 = np.zeros((4, 4), dtype=np.float32)
        attn_ref, state_ref = numpy_gdn_reference(q, k, v, g, beta, state0)

        np.testing.assert_allclose(attn_ort, attn_ref, atol=1e-5)
        np.testing.assert_allclose(state_ort, state_ref, atol=1e-5)

    def test_multi_token_correctness(self):
        """V=8, seq_len=5: ORT must match numpy reference over multiple steps."""
        attn_ort, state_ort, (q, k, v, g, beta) = self._run_model(8, 5)
        state0 = np.zeros((8, 8), dtype=np.float32)
        attn_ref, state_ref = numpy_gdn_reference(q, k, v, g, beta, state0)

        np.testing.assert_allclose(attn_ort, attn_ref, atol=1e-4)
        np.testing.assert_allclose(state_ort, state_ref, atol=1e-4)

    def test_output_shapes(self):
        attn_ort, state_ort, _ = self._run_model(16, 10)
        assert attn_ort.shape == (10, 16)
        assert state_ort.shape == (16, 16)

    def test_zero_state0_gives_same_as_numpy(self):
        """With zero initial state, ORT and numpy should be bit-for-bit close."""
        attn_ort, state_ort, (q, k, v, g, beta) = self._run_model(32, 8)
        state0 = np.zeros((32, 32), dtype=np.float32)
        attn_ref, state_ref = numpy_gdn_reference(q, k, v, g, beta, state0)

        # Relative error should be very small for float32
        rel_err = np.max(np.abs(attn_ort - attn_ref)) / max(np.max(np.abs(attn_ref)), 1e-10)
        assert rel_err < 1e-4, f"Relative error {rel_err:.2e} > 1e-4"

    def test_nonzero_state0(self):
        """Non-zero initial state: ORT must match numpy."""
        import onnxruntime as ort

        V, seq_len = 4, 3
        np.random.seed(99)
        state0 = np.random.randn(V, V).astype(np.float32) * 0.01

        model, q, k, v, g, beta = build_gdn_loop_model(head_dim=V, seq_len=seq_len)
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        results = sess.run(
            None,
            {"trip_count": np.array(seq_len, dtype=np.int64), "state0": state0},
        )
        attn_ort = results[1]
        state_ort = results[0]

        attn_ref, state_ref = numpy_gdn_reference(q, k, v, g, beta, state0)
        np.testing.assert_allclose(attn_ort, attn_ref, atol=1e-4)
        np.testing.assert_allclose(state_ort, state_ref, atol=1e-4)

    def test_deterministic(self):
        """Same inputs → same outputs (pass identical data to both calls)."""
        np.random.seed(42)
        q = np.random.randn(3, 8).astype(np.float32) * 0.1
        k = np.random.randn(3, 8).astype(np.float32) * 0.1
        v = np.random.randn(3, 8).astype(np.float32) * 0.1
        g = np.random.randn(3).astype(np.float32) * 0.01
        beta = np.ones(3, dtype=np.float32) * 0.1

        a1, s1, _ = self._run_model(8, 3, q_data=q, k_data=k, v_data=v, g_data=g, beta_data=beta)
        a2, s2, _ = self._run_model(8, 3, q_data=q, k_data=k, v_data=v, g_data=g, beta_data=beta)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(s1, s2)
