# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Tests for bench/gdn2_retrieval.py — associative recall capacity study.

Covers association generation, state building for both models, retrieval
scoring, the capacity ceiling, decay dynamics, and the matched-gate
equivalence proof.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.gdn2_retrieval import (  # noqa: E402
    build_state_gdn1,
    build_state_gdn2,
    capacity_sweep,
    decay_sweep,
    equivalence_check,
    main,
    make_associations,
    query_state,
    retrieval_accuracy,
)

# ---------------------------------------------------------------------------
# Association generation
# ---------------------------------------------------------------------------


class TestMakeAssociations:
    def test_shapes(self):
        assoc = make_associations(32, K=128, V=128, seed=0)
        assert assoc.N == 32
        assert assoc.keys.shape == (32, 128)
        assert assoc.values.shape == (32, 128)

    def test_keys_are_unit_norm(self):
        assoc = make_associations(16, K=64, V=64)
        norms = np.linalg.norm(assoc.keys, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_values_are_one_hot(self):
        assoc = make_associations(10, K=32, V=32)
        for i in range(10):
            assert np.sum(assoc.values[i]) == 1.0
            assert np.argmax(assoc.values[i]) == i

    def test_values_cycle_when_n_exceeds_v(self):
        assoc = make_associations(40, K=32, V=16)
        # keys 0..15 are unique slots; key 16 wraps to slot 0 again
        assert np.argmax(assoc.values[16]) == 0
        assert np.argmax(assoc.values[0]) == 0

    def test_deterministic(self):
        a1 = make_associations(8, seed=99)
        a2 = make_associations(8, seed=99)
        np.testing.assert_array_equal(a1.keys, a2.keys)

    def test_different_seeds_differ(self):
        a1 = make_associations(8, seed=1)
        a2 = make_associations(8, seed=2)
        assert not np.allclose(a1.keys, a2.keys)


# ---------------------------------------------------------------------------
# State builders
# ---------------------------------------------------------------------------


class TestStateBuilders:
    def test_gdn1_state_shape(self):
        assoc = make_associations(8, K=32, V=16)
        S = build_state_gdn1(assoc)
        assert S.shape == (1, 32, 16)

    def test_gdn2_state_shape(self):
        assoc = make_associations(8, K=32, V=16)
        S = build_state_gdn2(assoc)
        assert S.shape == (1, 32, 16)

    def test_gdn1_zero_state_for_zero_write(self):
        """With beta=0 nothing is written; state stays zero."""
        assoc = make_associations(4, K=16, V=16)
        S = build_state_gdn1(assoc, alpha=1.0, beta=0.0)
        assert np.max(np.abs(S)) == 0.0

    def test_gdn2_zero_state_for_zero_write(self):
        """With write_gate=0 nothing is written; state stays zero."""
        assoc = make_associations(4, K=16, V=16)
        S = build_state_gdn2(assoc, write_gate=0.0)
        assert np.max(np.abs(S)) == 0.0


# ---------------------------------------------------------------------------
# Matched-gate equivalence (the core mathematical result)
# ---------------------------------------------------------------------------


class TestEquivalence:
    def test_states_identical_matched_gates(self):
        """GDN-1 (α=1,β=1) and GDN-2 (g=0,b=1,w=1) produce identical states."""
        assoc = make_associations(32, K=64, V=64)
        S1 = build_state_gdn1(assoc)
        S2 = build_state_gdn2(assoc)
        np.testing.assert_array_equal(S1, S2)

    def test_equivalence_check_near_zero(self):
        diff = equivalence_check(num_keys=16, K=32, V=32)
        assert diff < 1e-12

    def test_states_differ_when_decay_mismatched(self):
        """If GDN-2 uses decay but GDN-1 uses no decay, states must differ."""
        assoc = make_associations(8, K=32, V=32)
        S1 = build_state_gdn1(assoc, alpha=1.0)  # no decay
        S2 = build_state_gdn2(assoc, decay=0.1)  # decay
        assert np.max(np.abs(S1 - S2)) > 0.0

    def test_states_match_when_decay_matched(self):
        """GDN-1 alpha=exp(-d) should match GDN-2 decay=d (within float64
        accumulation tolerance — the projection happens pre- vs post-decay)."""
        assoc = make_associations(8, K=32, V=32)
        S1 = build_state_gdn1(assoc, alpha=float(np.exp(-0.05)))
        S2 = build_state_gdn2(assoc, decay=0.05)
        np.testing.assert_allclose(S1, S2, atol=1e-6)


# ---------------------------------------------------------------------------
# Retrieval scoring
# ---------------------------------------------------------------------------


class TestRetrievalScoring:
    def test_perfect_recall_below_capacity(self):
        """N ≤ K associations should be 100% retrievable."""
        assoc = make_associations(32, K=64, V=64)
        S = build_state_gdn1(assoc)
        assert retrieval_accuracy(S, assoc) == 1.0

    def test_perfect_recall_at_capacity(self):
        """N = K should still be 100% (rank-limit boundary)."""
        assoc = make_associations(64, K=64, V=64)
        S = build_state_gdn1(assoc)
        assert retrieval_accuracy(S, assoc) == 1.0

    def test_recall_degrades_above_capacity(self):
        """N > K should see degradation."""
        assoc = make_associations(256, K=64, V=64)
        S = build_state_gdn1(assoc)
        assert retrieval_accuracy(S, assoc) < 1.0

    def test_query_returns_correct_count(self):
        assoc = make_associations(10, K=32, V=32)
        S = build_state_gdn1(assoc)
        preds = query_state(S, assoc)
        assert len(preds) == 10

    def test_gdn1_gdn2_same_accuracy_matched_gates(self):
        """Both models give identical accuracy at matched gates."""
        assoc = make_associations(128, K=128, V=128)
        S1 = build_state_gdn1(assoc)
        S2 = build_state_gdn2(assoc)
        assert retrieval_accuracy(S1, assoc) == retrieval_accuracy(S2, assoc)


# ---------------------------------------------------------------------------
# Study runners
# ---------------------------------------------------------------------------


class TestCapacitySweep:
    def test_returns_results_for_both_models(self):
        results = capacity_sweep(K=32, V=32, key_counts=(8, 16, 32))
        assert len(results) == 6  # 3 key counts × 2 models
        models = {r.model for r in results}
        assert models == {"gdn1", "gdn2"}

    def test_capacity_well_below_k_is_perfect(self):
        """N << K associations are 100% retrievable (keys near-orthogonal)."""
        results = capacity_sweep(K=32, V=32, key_counts=(4, 8, 16))
        for r in results:
            assert r.accuracy == 1.0

    def test_capacity_boundary_interference_for_small_k(self):
        """At N=K with small K, random-key coherence causes <100% (unlike
        K=128 where N=128 is still perfect — coherence ~sqrt(2ln(N)/K))."""
        results = capacity_sweep(K=32, V=32, key_counts=(32, 128))
        acc_32 = [r.accuracy for r in results if r.num_keys == 32][0]
        acc_128 = [r.accuracy for r in results if r.num_keys == 128][0]
        assert acc_32 < 1.0  # 93.75% at seed=42 — boundary interference
        assert acc_128 < acc_32  # monotone degradation well above capacity


class TestDecaySweep:
    def test_returns_results(self):
        results = decay_sweep(num_keys=16, K=32, V=32, decay_values=(0.0, 0.1))
        assert len(results) == 4  # 2 decay values × 2 models

    def test_zero_decay_is_perfect(self):
        results = decay_sweep(num_keys=16, K=32, V=32, decay_values=(0.0,))
        for r in results:
            assert r.accuracy == 1.0

    def test_high_decay_reduces_accuracy(self):
        results = decay_sweep(num_keys=32, K=64, V=64, decay_values=(0.0, 0.2))
        acc_0 = [r.accuracy for r in results if r.params["decay"] == 0.0][0]
        acc_high = [r.accuracy for r in results if r.params["decay"] == 0.2][0]
        assert acc_0 >= acc_high

    def test_gdn1_gdn2_identical_under_uniform_decay(self):
        """Uniform decay makes the models identical."""
        results = decay_sweep(num_keys=32, K=64, V=64, decay_values=(0.05,))
        a1 = [r.accuracy for r in results if r.model == "gdn1"][0]
        a2 = [r.accuracy for r in results if r.model == "gdn2"][0]
        assert a1 == a2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_human_output_exits_zero(self, capsys):
        rc = main(["--seed", "7"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Capacity sweep" in out
        assert "equivalence" in out.lower()

    def test_csv_output(self, capsys):
        rc = main(["--csv", "--seed", "7"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "test,model,num_keys,accuracy" in out
        lines = [line for line in out.strip().splitlines() if line and not line.startswith("test,")]
        assert len(lines) >= 10  # capacity + decay + equivalence rows
