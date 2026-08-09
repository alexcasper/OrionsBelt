# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Associative-recall capacity study for GDN-1 vs GDN-2 recurrent states.

Bead ``ob-zak``.  A portable, recurrence-state-level test of the retrieval
hypothesis recorded in ADR-0001: *does GDN-2's decoupled erase/write gating
improve multi-key retrieval?*

The full RULER multi-key evaluation (bead ``ob-zak`` as originally scoped)
requires loading an adapted GDN-2 checkpoint and running model-level
generation — infrastructure that does not exist on edge devices (no
torch/transformers; ob-68l's swap ran on x86/CUDA).  This module delivers the
**portable** part of the answer: it tests the recurrent state's associative
recall capacity directly, at the verified Qwen3.5-4B shapes
(K = V = 128, single head), without any model weights.

What it measures
----------------
1. **Capacity ceiling** — write *N* ``(key, value)`` associations into the state,
   then query every key and score top-1 retrieval accuracy.  Sweep *N* to find
   the rank-limit where the state saturates (expected ≈ K).
2. **Matched-gate equivalence** — with no decay and unit erase/write gates both
   GDN-1 and GDN-2 reduce to the *same* delta rule; the test verifies they are
   numerically identical, proving the retrieval advantage of GDN-2 can only come
   from **learned, input-dependent** per-channel gating.
3. **Decay dynamics** — with uniform exponential decay applied to old
   associations, measure how retrieval of early keys degrades.  Under uniform
   decay GDN-1 (scalar) and GDN-2 (per-channel, but all-equal) are still
   identical — the difference only appears with non-uniform learned gates.

The honest conclusion (see ``docs/research/ob-zak-retrieval-capacity.md``):
GDN-2's hypothesised multi-key retrieval benefit is **not testable without an
adapted model**, because at matched gate configurations the two architectures
are mathematically equivalent.  This module supplies the reproducible evidence
for that statement.

Usage::

    python -m bench.gdn2_retrieval                      # human-readable
    python -m bench.gdn2_retrieval --csv                 # CSV to stdout
    python -m bench.gdn2_retrieval --csv > results/raw/rk3588-t4_gdn2_retrieval.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field

import numpy as np

from bench.gdn2_reference import gdn1_recurrent, gdn2_recurrent

# Verified Qwen3.5-4B GDN shapes (docs/CLAIM_VERIFICATION.md §2.3)
DEFAULT_K = 128
DEFAULT_V = 128


# ---------------------------------------------------------------------------
# Synthetic association generators
# ---------------------------------------------------------------------------


@dataclass
class Associations:
    """A bundle of key-value associations for the recall test.

    Keys are random unit-norm vectors in R^K (near-orthogonal when N << K).
    Values are one-hot vectors in R^V so retrieval scoring is unambiguous:
    a query is correct iff ``argmax(output) == argmax(target_value)``.
    """

    keys: np.ndarray  # [N, K]  float32
    values: np.ndarray  # [N, V]  float32 (one-hot)
    N: int
    K: int
    V: int
    seed: int


def make_associations(
    num_keys: int,
    K: int = DEFAULT_K,
    V: int = DEFAULT_V,
    seed: int = 42,
) -> Associations:
    """Generate *num_keys* random near-orthogonal keys and one-hot values.

    Keys are drawn from a standard normal and L2-normalised so the delta-rule
    projection ``k k^T`` is well-conditioned.  If ``num_keys > V`` we reuse
    value slots cyclically (the capacity question is about keys, not values).
    """
    rng = np.random.RandomState(seed)
    keys = rng.randn(num_keys, K).astype(np.float32)
    # L2-normalise keys for well-conditioned projections
    norms = np.linalg.norm(keys, axis=1, keepdims=True)
    keys = keys / np.maximum(norms, 1e-8)
    # One-hot values; cycle if N > V
    values = np.zeros((num_keys, V), dtype=np.float32)
    for i in range(num_keys):
        values[i, i % V] = 1.0
    return Associations(keys=keys, values=values, N=num_keys, K=K, V=V, seed=seed)


# ---------------------------------------------------------------------------
# State builders: write all associations, return the final recurrent state
# ---------------------------------------------------------------------------


def build_state_gdn1(assoc: Associations, alpha: float = 1.0, beta: float = 1.0) -> np.ndarray:
    """Write all associations into a GDN-1 state and return S [1, K, V].

    With ``alpha=1`` (no decay) and ``beta=1`` (full write) this is the exact
    delta rule::

        S_t = (I - k k^T) S_{t-1} + k v^T
    """
    N = assoc.N
    # Layout: T=N, H=1, HV=1
    q = assoc.keys[:, None, :].copy()  # [N, 1, K]  (unused for writes)
    k = assoc.keys[:, None, :].copy()  # [N, 1, K]
    v = assoc.values[:, None, :].copy()  # [N, 1, V]
    alpha_arr = np.full((N, 1), alpha, dtype=np.float32)
    beta_arr = np.full((N, 1), beta, dtype=np.float32)
    _, S = gdn1_recurrent(q, k, v, alpha_arr, beta_arr, use_qk_l2norm=False)
    return S  # [1, K, V]


def build_state_gdn2(
    assoc: Associations,
    decay: float = 0.0,
    erase_gate: float = 1.0,
    write_gate: float = 1.0,
) -> np.ndarray:
    """Write all associations into a GDN-2 state and return S [1, K, V].

    Parameters control the (uniform) gate values:

    - ``decay``: log-space decay applied per channel (g = -decay, so
      ``exp(g) = exp(-decay)``; 0.0 = no decay).
    - ``erase_gate``: per-channel erase gate b (1.0 = full erase).
    - ``write_gate``: per-channel write gate w (1.0 = full write).

    With ``decay=0, erase_gate=1, write_gate=1`` this is identical to the
    GDN-1 delta rule with ``alpha=1, beta=1``.
    """
    N, K, V = assoc.N, assoc.K, assoc.V
    q = assoc.keys[:, None, :].copy()
    k = assoc.keys[:, None, :].copy()
    v = assoc.values[:, None, :].copy()
    g = np.full((N, 1, K), -decay, dtype=np.float32)  # exp(-decay)
    b_gate = np.full((N, 1, K), erase_gate, dtype=np.float32)
    w_gate = np.full((N, 1, V), write_gate, dtype=np.float32)
    _, S = gdn2_recurrent(q, k, v, g, b_gate, w_gate, use_qk_l2norm=False)
    return S  # [1, K, V]


# ---------------------------------------------------------------------------
# Retrieval scoring
# ---------------------------------------------------------------------------


def query_state(S: np.ndarray, assoc: Associations) -> np.ndarray:
    """Query every key against the state and return predicted value indices.

    ``o_i = q_i @ S`` → predicted value vector; correctness is
    ``argmax(o_i) == i % V``.

    Returns array of predicted indices [N].
    """
    Sh = S[0]  # [K, V]
    scale = 1.0 / np.sqrt(assoc.K)
    preds = np.empty(assoc.N, dtype=np.int64)
    for i in range(assoc.N):
        ki = assoc.keys[i] * scale
        o = ki @ Sh  # [V]
        preds[i] = int(np.argmax(o))
    return preds


def retrieval_accuracy(S: np.ndarray, assoc: Associations) -> float:
    """Top-1 retrieval accuracy over all associations."""
    preds = query_state(S, assoc)
    targets = np.array([i % assoc.V for i in range(assoc.N)], dtype=np.int64)
    correct = int(np.sum(preds == targets))
    return correct / assoc.N


# ---------------------------------------------------------------------------
# Study runners
# ---------------------------------------------------------------------------


@dataclass
class CapacityResult:
    model: str
    num_keys: int
    accuracy: float
    params: dict = field(default_factory=dict)


def capacity_sweep(
    K: int = DEFAULT_K,
    V: int = DEFAULT_V,
    key_counts: tuple[int, ...] | None = None,
    seed: int = 42,
) -> list[CapacityResult]:
    """Sweep number of keys and measure retrieval accuracy for both models.

    Default sweep: 8, 16, 32, 64, 96, 128, 160, 192, 256 — brackets the
    expected capacity ceiling at K=128.
    """
    if key_counts is None:
        key_counts = (8, 16, 32, 64, 96, 128, 160, 192, 256)
    results: list[CapacityResult] = []
    for n in key_counts:
        assoc = make_associations(n, K=K, V=V, seed=seed)
        S1 = build_state_gdn1(assoc)
        S2 = build_state_gdn2(assoc)
        a1 = retrieval_accuracy(S1, assoc)
        a2 = retrieval_accuracy(S2, assoc)
        results.append(CapacityResult("gdn1", n, a1))
        results.append(CapacityResult("gdn2", n, a2))
    return results


def decay_sweep(
    num_keys: int = 64,
    K: int = DEFAULT_K,
    V: int = DEFAULT_V,
    decay_values: tuple[float, ...] | None = None,
    seed: int = 42,
) -> list[CapacityResult]:
    """Sweep uniform decay and measure retrieval accuracy for both models.

    With uniform decay, GDN-1 (scalar alpha) and GDN-2 (per-channel g, all equal)
    should remain identical — verifying that the architectures differ only when
    gates are *non-uniform*.
    """
    if decay_values is None:
        decay_values = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1)
    assoc = make_associations(num_keys, K=K, V=V, seed=seed)
    results: list[CapacityResult] = []
    for d in decay_values:
        alpha = float(np.exp(-d))  # match decay: GDN-1 scalar alpha = exp(-decay)
        S1 = build_state_gdn1(assoc, alpha=alpha)
        S2 = build_state_gdn2(assoc, decay=d)
        a1 = retrieval_accuracy(S1, assoc)
        a2 = retrieval_accuracy(S2, assoc)
        results.append(CapacityResult("gdn1", num_keys, a1, {"decay": d}))
        results.append(CapacityResult("gdn2", num_keys, a2, {"decay": d}))
    return results


def equivalence_check(
    num_keys: int = 32,
    K: int = DEFAULT_K,
    V: int = DEFAULT_V,
    seed: int = 42,
) -> float:
    """Max absolute element-wise difference between GDN-1 and GDN-2 states.

    With matched gates (no decay, unit erase/write) both reduce to the delta
    rule, so this should be ~machine epsilon (float64 accumulation).
    """
    assoc = make_associations(num_keys, K=K, V=V, seed=seed)
    S1 = build_state_gdn1(assoc)
    S2 = build_state_gdn2(assoc)
    return float(np.max(np.abs(S1 - S2)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_human(capacity: list[CapacityResult], decay: list[CapacityResult], equiv: float) -> None:
    print("=" * 64)
    print("GDN-1 vs GDN-2 Associative Recall Capacity Study (ob-zak)")
    print(f"  Shapes: K=V={DEFAULT_K} (Qwen3.5-4B verified), single head")
    print("=" * 64)

    print("\n--- Capacity sweep (accuracy vs number of keys) ---")
    print(f"{'N keys':>8}  {'GDN-1':>8}  {'GDN-2':>8}")
    for i in range(0, len(capacity), 2):
        r1, r2 = capacity[i], capacity[i + 1]
        print(f"{r1.num_keys:>8}  {r1.accuracy:>7.1%}  {r2.accuracy:>7.1%}")

    print("\n--- Decay sweep (N=64 keys, uniform decay) ---")
    print(f"{'decay':>8}  {'GDN-1':>8}  {'GDN-2':>8}")
    for i in range(0, len(decay), 2):
        r1, r2 = decay[i], decay[i + 1]
        d = r1.params.get("decay", 0.0)
        print(f"{d:>8.4f}  {r1.accuracy:>7.1%}  {r2.accuracy:>7.1%}")

    print("\n--- Matched-gate equivalence (N=32, max|S1-S2|) ---")
    print(f"  max abs diff = {equiv:.2e}")
    status = "IDENTICAL (within float64 epsilon)" if equiv < 1e-9 else "DIFFERENT"
    print(f"  => {status}")

    print("\nConclusion: with matched gates GDN-1 and GDN-2 are numerically")
    print("identical.  The retrieval advantage of decoupled gating requires")
    print("learned, input-dependent per-channel gates (needs an adapted model).")


def _print_csv(capacity: list[CapacityResult], decay: list[CapacityResult], equiv: float) -> None:
    w = csv.writer(sys.stdout)
    w.writerow(["test", "model", "num_keys", "accuracy", "param", "param_value"])
    for r in capacity:
        w.writerow(["capacity", r.model, r.num_keys, f"{r.accuracy:.6f}", "", ""])
    for r in decay:
        d = r.params.get("decay", "")
        w.writerow(["decay", r.model, r.num_keys, f"{r.accuracy:.6f}", "decay", d])
    w.writerow(["equivalence", "gdn1_vs_gdn2", 32, "", "max_abs_diff", f"{equiv:.2e}"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GDN-1 vs GDN-2 retrieval capacity study")
    parser.add_argument("--csv", action="store_true", help="CSV output to stdout")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    capacity = capacity_sweep(seed=args.seed)
    decay = decay_sweep(seed=args.seed)
    equiv = equivalence_check(seed=args.seed)

    if args.csv:
        _print_csv(capacity, decay, equiv)
    else:
        _print_human(capacity, decay, equiv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
