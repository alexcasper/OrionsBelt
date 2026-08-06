"""Tests for scripts/generate_reference.py — testable functions without torch.

Tests the provenance utilities (_git_sha, _git_dirty, _governor, _thermals,
_hostname), the sequence builder (_build_sequence), prompt constants, and
the CLI error path when torch/transformers are unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.generate_reference import (  # noqa: E402
    CONTEXT_LENGTHS,
    PROMPTS,
    SMOKE_CONTEXT_LENGTHS,
    _build_sequence,
    _git_dirty,
    _git_sha,
    _governor,
    _hostname,
    _thermals,
    main,
)


# ---------------------------------------------------------------------------
# _build_sequence
# ---------------------------------------------------------------------------
class TestBuildSequence:
    def test_basic_truncation(self):
        """Sequence is truncated to target_length."""
        seed = [1, 2, 3]
        filler = list(range(10, 30))
        result = _build_sequence(seed, filler, 15)
        assert len(result) == 15
        assert result[:3] == seed

    def test_exact_length(self):
        """When combined length equals target, no truncation needed."""
        seed = [1, 2]
        filler = [3, 4, 5]
        result = _build_sequence(seed, filler, 5)
        assert result == [1, 2, 3, 4, 5]

    def test_target_shorter_than_seed(self):
        """Target shorter than seed alone → only seed prefix."""
        seed = list(range(10))
        filler = list(range(20, 30))
        result = _build_sequence(seed, filler, 3)
        assert result == [0, 1, 2]

    def test_filler_repeats_when_insufficient(self):
        """When seed + filler < target, filler is repeated."""
        seed = [1, 2]
        filler = [3, 4]
        result = _build_sequence(seed, filler, 10)
        assert len(result) == 10
        assert result[:2] == [1, 2]
        # Filler should be repeated: [3, 4, 3, 4, 3, 4, 3, 4]
        assert result[2:] == [3, 4, 3, 4, 3, 4, 3, 4]

    def test_single_filler_token_repeats(self):
        """Single filler token can be repeated to fill."""
        seed = [1]
        filler = [2]
        result = _build_sequence(seed, filler, 5)
        assert result == [1, 2, 2, 2, 2]

    def test_empty_seed(self):
        """Empty seed → all filler."""
        result = _build_sequence([], [1, 2, 3], 2)
        assert result == [1, 2]

    def test_empty_filler_repeats_zero(self):
        """Empty filler with target=0 → empty result (edge case)."""
        # With empty filler, reps = 0 // 0 + 1 = 1, but filler*1 = []
        result = _build_sequence([1], [], 0)
        assert result == []

    def test_preserves_seed_order(self):
        """Seed tokens always come first, in order."""
        seed = [100, 200, 300]
        filler = [10, 20, 30, 40]
        result = _build_sequence(seed, filler, 7)
        assert result[:3] == [100, 200, 300]
        assert result[3:] == [10, 20, 30, 40]

    @pytest.mark.parametrize("target", [1, 10, 50, 100, 500])
    def test_all_targets_exact_length(self, target):
        """All target lengths produce exactly target tokens."""
        seed = list(range(5))
        filler = list(range(10, 20))
        result = _build_sequence(seed, filler, target)
        assert len(result) == target


# ---------------------------------------------------------------------------
# Provenance utilities
# ---------------------------------------------------------------------------
class TestGitSha:
    def test_returns_string(self):
        result = _git_sha()
        assert isinstance(result, str)

    def test_not_empty(self):
        result = _git_sha()
        assert len(result) > 0

    def test_valid_format(self):
        """Either 'unknown' or a hex SHA."""
        result = _git_sha()
        if result != "unknown":
            assert all(c in "0123456789abcdef" for c in result)


class TestGitDirty:
    def test_returns_bool(self):
        result = _git_dirty()
        assert isinstance(result, bool)


class TestGovernor:
    def test_returns_string(self):
        result = _governor()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_known_governor_or_unknown(self):
        """Governor is a known Linux value or 'unknown'."""
        result = _governor()
        known = {"performance", "powersave", "schedutil", "ondemand", "conservative", "unknown"}
        assert result in known


class TestThermals:
    def test_returns_list_or_str(self):
        result = _thermals()
        assert isinstance(result, (list, str))

    def test_temps_reasonable(self):
        """Thermal readings should be in a plausible range (milli-Celsius)."""
        result = _thermals()
        if isinstance(result, list):
            for t in result:
                assert isinstance(t, int)
                # Plausible: -20°C to 120°C → 20000 to 120000 milli-°C
                assert 0 < t < 150000


class TestHostname:
    def test_returns_string(self):
        result = _hostname()
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# PROMPTS constant
# ---------------------------------------------------------------------------
class TestPrompts:
    def test_has_four_prompts(self):
        assert len(PROMPTS) == 4

    def test_prompt_ids(self):
        ids = [p["id"] for p in PROMPTS]
        assert "factual" in ids
        assert "code" in ids
        assert "sequential" in ids
        assert "reasoning" in ids

    def test_all_have_text(self):
        for p in PROMPTS:
            assert "text" in p
            assert len(p["text"]) > 20

    def test_all_ids_unique(self):
        ids = [p["id"] for p in PROMPTS]
        assert len(ids) == len(set(ids))

    def test_factual_mentions_gdn(self):
        factual = next(p for p in PROMPTS if p["id"] == "factual")
        assert "gating" in factual["text"].lower() or "delta" in factual["text"].lower()

    def test_code_prompt_has_python(self):
        code = next(p for p in PROMPTS if p["id"] == "code")
        assert "def " in code["text"]

    def test_sequential_prompt_has_numbers(self):
        seq = next(p for p in PROMPTS if p["id"] == "sequential")
        assert "one" in seq["text"] and "ten" in seq["text"]

    def test_reasoning_prompt_has_question(self):
        reasoning = next(p for p in PROMPTS if p["id"] == "reasoning")
        assert "Question:" in reasoning["text"]


# ---------------------------------------------------------------------------
# Context length constants
# ---------------------------------------------------------------------------
class TestContextLengths:
    def test_context_lengths_sorted(self):
        assert sorted(CONTEXT_LENGTHS) == CONTEXT_LENGTHS

    def test_context_lengths_positive(self):
        for cl in CONTEXT_LENGTHS:
            assert cl > 0

    def test_smoke_is_subset(self):
        """Smoke lengths should be shorter than full lengths."""
        for sl in SMOKE_CONTEXT_LENGTHS:
            assert sl <= min(CONTEXT_LENGTHS)

    def test_smoke_lengths_positive(self):
        for sl in SMOKE_CONTEXT_LENGTHS:
            assert sl > 0


# ---------------------------------------------------------------------------
# CLI / main()
# ---------------------------------------------------------------------------
class TestMain:
    def test_no_torch_returns_error(self, monkeypatch, capsys):
        """Without torch, main() returns 1 and prints an error."""
        # torch is not installed on this device, so _TORCH_AVAILABLE is False
        rc = main([])
        assert rc == 1
        captured = capsys.readouterr()
        assert "torch" in captured.err.lower() or "not available" in captured.err.lower()

    def test_missing_model_path(self, monkeypatch, capsys):
        """When torch is not available, exits before checking model path."""
        rc = main(["--model-path", "/nonexistent/path"])
        assert rc == 1

    def test_smoke_flag_still_fails_without_torch(self, capsys):
        """Smoke flag doesn't help without torch."""
        rc = main(["--smoke"])
        assert rc == 1
