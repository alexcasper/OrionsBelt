"""Tests for the prompt corpus generator (ob-del).

Verifies determinism (same seed → same output) and approximate token counts.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.generate_prompts import (  # noqa: E402
    MASTER_SEED,
    generate_needle,
    generate_ruler,
)


class TestDeterminism:
    def test_needle_deterministic(self):
        """Same seed produces identical output."""
        rng1 = random.Random(MASTER_SEED + 4096)
        rng2 = random.Random(MASTER_SEED + 4096)
        p1, m1 = generate_needle(4096, rng1)
        p2, m2 = generate_needle(4096, rng2)
        assert p1 == p2
        assert m1 == m2

    def test_ruler_deterministic(self):
        rng1 = random.Random(MASTER_SEED + 4096 + 100000)
        rng2 = random.Random(MASTER_SEED + 4096 + 100000)
        p1, m1 = generate_ruler(4096, rng1)
        p2, m2 = generate_ruler(4096, rng2)
        assert p1 == p2
        assert m1 == m2

    def test_different_seeds_different_output(self):
        rng1 = random.Random(MASTER_SEED + 4096)
        rng2 = random.Random(MASTER_SEED + 32768)
        p1, _ = generate_needle(4096, rng1)
        p2, _ = generate_needle(32768, rng2)
        assert p1 != p2


class TestTokenCounts:
    def test_needle_approximate_target(self):
        rng = random.Random(MASTER_SEED + 4096)
        _, meta = generate_needle(4096, rng)
        # Within ±20% of target
        assert 3000 <= meta["actual_tokens_approx"] <= 5000

    def test_ruler_approximate_target(self):
        rng = random.Random(MASTER_SEED + 32768 + 100000)
        _, meta = generate_ruler(32768, rng)
        assert 25000 <= meta["actual_tokens_approx"] <= 40000


class TestNeedleStructure:
    def test_needle_embedded_in_prompt(self):
        rng = random.Random(MASTER_SEED + 4096)
        prompt, meta = generate_needle(4096, rng)
        assert meta["needle"] in prompt
        assert meta["question"] in prompt

    def test_needle_depth_in_range(self):
        rng = random.Random(MASTER_SEED + 4096)
        _, meta = generate_needle(4096, rng)
        assert 0.3 <= meta["needle_depth_fraction"] <= 0.7


class TestRulerStructure:
    def test_keys_in_prompt(self):
        rng = random.Random(MASTER_SEED + 4096 + 100000)
        prompt, meta = generate_ruler(4096, rng)
        for key in meta["queried_keys"]:
            assert key in prompt

    def test_expected_answers_present(self):
        rng = random.Random(MASTER_SEED + 4096 + 100000)
        _, meta = generate_ruler(4096, rng)
        assert len(meta["expected_answers"]) == len(meta["queried_keys"])

    def test_num_keys_scales_with_context(self):
        rng_small = random.Random(MASTER_SEED + 4096 + 100000)
        rng_large = random.Random(MASTER_SEED + 32768 + 100000)
        _, small = generate_ruler(4096, rng_small)
        _, large = generate_ruler(32768, rng_large)
        assert large["num_keys"] >= small["num_keys"]
