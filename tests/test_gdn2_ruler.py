# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / AAIF
# SPDX-License-Identifier: Apache-2.0
"""Tests for bench/gdn2_ruler.py — RULER multi-key retrieval eval (ob-zak).

Tests cover prompt generation, log-likelihood scoring, and the full
evaluate_retrieval pipeline using mock model/tokenizer objects.

Pure-Python tests (generate_prompts, capture_manifest) run everywhere.
Torch-dependent tests (score_answer_logprob, evaluate_retrieval) are
skipped when torch is not installed (e.g. on edge CI runners).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))

from gdn2_ruler import (  # noqa: E402
    capture_manifest,
    generate_prompts,
)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# generate_prompts (pure Python — no torch required)
# ---------------------------------------------------------------------------


class TestGeneratePrompts:
    def test_returns_list(self):
        prompts = generate_prompts(num_prompts=3, context_length=512, num_keys=5)
        assert isinstance(prompts, list)
        assert len(prompts) == 3

    def test_prompt_has_required_fields(self):
        prompts = generate_prompts(num_prompts=1, context_length=512, num_keys=5)
        p = prompts[0]
        for field in (
            "prompt",
            "query_key",
            "correct_answer",
            "distractor_answers",
            "est_tokens",
            "num_keys",
            "needle_depth",
            "seed",
        ):
            assert field in p, f"missing field: {field}"

    def test_correct_answer_not_in_distractors(self):
        prompts = generate_prompts(num_prompts=5, context_length=512, num_keys=5)
        for p in prompts:
            assert p["correct_answer"] not in p["distractor_answers"]

    def test_distractors_from_other_keys(self):
        """With N keys, there should be N-1 distractors."""
        prompts = generate_prompts(num_prompts=3, context_length=512, num_keys=5)
        for p in prompts:
            assert len(p["distractor_answers"]) == 4

    def test_seed_increments(self):
        prompts = generate_prompts(num_prompts=3, context_length=512, num_keys=3, seed_base=200)
        seeds = [p["seed"] for p in prompts]
        assert seeds == [200, 201, 202]

    def test_seed_base_default(self):
        prompts = generate_prompts(num_prompts=2, context_length=512, num_keys=3)
        assert prompts[0]["seed"] == 100  # default seed_base

    def test_query_key_present_in_prompt(self):
        prompts = generate_prompts(num_prompts=2, context_length=512, num_keys=3)
        for p in prompts:
            assert p["query_key"] in p["prompt"]

    def test_correct_answer_present_in_prompt(self):
        prompts = generate_prompts(num_prompts=2, context_length=512, num_keys=3)
        for p in prompts:
            assert p["correct_answer"] in p["prompt"]

    def test_deterministic_with_same_seed(self):
        p1 = generate_prompts(num_prompts=1, context_length=512, num_keys=3, seed_base=42)
        p2 = generate_prompts(num_prompts=1, context_length=512, num_keys=3, seed_base=42)
        assert p1[0]["query_key"] == p2[0]["query_key"]
        assert p1[0]["correct_answer"] == p2[0]["correct_answer"]

    def test_num_keys_field_matches(self):
        prompts = generate_prompts(num_prompts=1, context_length=512, num_keys=7)
        assert prompts[0]["num_keys"] == 7

    def test_prompt_is_nonempty_string(self):
        prompts = generate_prompts(num_prompts=1, context_length=512, num_keys=3)
        assert isinstance(prompts[0]["prompt"], str)
        assert len(prompts[0]["prompt"]) > 100


# ---------------------------------------------------------------------------
# capture_manifest (pure Python — no torch required)
# ---------------------------------------------------------------------------


class TestCaptureManifest:
    def test_returns_dict_with_required_fields(self):
        m = capture_manifest()
        assert isinstance(m, dict)
        assert "git" in m
        assert "sha" in m["git"]
        assert "dirty" in m["git"]
        assert "host" in m
        assert "machine" in m["host"]
        assert "software" in m

    def test_git_sha_is_string(self):
        m = capture_manifest()
        assert isinstance(m["git"]["sha"], str)

    def test_git_dirty_is_bool(self):
        m = capture_manifest()
        assert isinstance(m["git"]["dirty"], bool)

    def test_timestamp_format(self):
        m = capture_manifest()
        ts = m["timestamp_utc"]
        assert ts.endswith("Z")
        assert "T" in ts


# ---------------------------------------------------------------------------
# score_answer_logprob (mock model/tokenizer — requires torch)
# ---------------------------------------------------------------------------


class _MockLogitsOutput:
    """Mimics model output with .logits attribute."""

    def __init__(self, logits):
        self.logits = logits


class _MockModel:
    """Returns deterministic logits. shape: [1, seq_len, vocab_size]."""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size

    def __call__(self, input_ids, **kwargs):
        seq_len = input_ids.shape[1]
        torch.manual_seed(42)
        logits = torch.randn(1, seq_len, self.vocab_size)
        return _MockLogitsOutput(logits)


class _MockTokenizer:
    """Maps each character to a token id; roundtrip preserves length."""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size

    def __call__(self, text, return_tensors="pt"):
        ids = [ord(c) % self.vocab_size for c in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
class TestScoreAnswerLogprob:
    def test_returns_tuple_of_two_floats(self):
        from gdn2_ruler import score_answer_logprob

        model = _MockModel()
        tokenizer = _MockTokenizer()
        total_lp, avg_lp = score_answer_logprob(model, tokenizer, "hello", "world")
        assert isinstance(total_lp, float)
        assert isinstance(avg_lp, float)

    def test_total_is_sum_of_token_logprobs(self):
        """total_lp should be <= 0 (log-probabilities are non-positive)."""
        from gdn2_ruler import score_answer_logprob

        model = _MockModel()
        tokenizer = _MockTokenizer()
        total_lp, _ = score_answer_logprob(model, tokenizer, "hello", "world")
        assert total_lp <= 0.0

    def test_avg_is_total_divided_by_answer_len(self):
        from gdn2_ruler import score_answer_logprob

        model = _MockModel()
        tokenizer = _MockTokenizer()
        total_lp, avg_lp = score_answer_logprob(model, tokenizer, "hello", "world")
        # answer " world" = 6 tokens
        assert abs(avg_lp - total_lp / 6) < 1e-5

    def test_deterministic(self):
        from gdn2_ruler import score_answer_logprob

        model = _MockModel()
        tokenizer = _MockTokenizer()
        lp1 = score_answer_logprob(model, tokenizer, "hello", "world")
        lp2 = score_answer_logprob(model, tokenizer, "hello", "world")
        assert lp1 == lp2

    def test_different_answers_different_scores(self):
        from gdn2_ruler import score_answer_logprob

        model = _MockModel()
        tokenizer = _MockTokenizer()
        lp1 = score_answer_logprob(model, tokenizer, "prompt", "answer1")
        lp2 = score_answer_logprob(model, tokenizer, "prompt", "answer2")
        assert lp1 != lp2


# ---------------------------------------------------------------------------
# evaluate_retrieval (mock model/tokenizer — requires torch)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
class TestEvaluateRetrieval:
    def _make_prompts(self, n=2):
        return generate_prompts(num_prompts=n, context_length=512, num_keys=3, seed_base=500)

    def test_returns_dict_with_required_fields(self):
        from gdn2_ruler import evaluate_retrieval

        model = _MockModel()
        tokenizer = _MockTokenizer()
        prompts = self._make_prompts(1)
        result = evaluate_retrieval(model, tokenizer, prompts)
        for field in ("accuracy", "num_prompts", "num_correct", "total_time_s", "details"):
            assert field in result

    def test_accuracy_in_valid_range(self):
        from gdn2_ruler import evaluate_retrieval

        model = _MockModel()
        tokenizer = _MockTokenizer()
        prompts = self._make_prompts(2)
        result = evaluate_retrieval(model, tokenizer, prompts)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_num_prompts_matches_input(self):
        from gdn2_ruler import evaluate_retrieval

        model = _MockModel()
        tokenizer = _MockTokenizer()
        prompts = self._make_prompts(3)
        result = evaluate_retrieval(model, tokenizer, prompts)
        assert result["num_prompts"] == 3

    def test_num_correct_consistent_with_accuracy(self):
        from gdn2_ruler import evaluate_retrieval

        model = _MockModel()
        tokenizer = _MockTokenizer()
        prompts = self._make_prompts(3)
        result = evaluate_retrieval(model, tokenizer, prompts)
        expected_acc = result["num_correct"] / result["num_prompts"]
        assert abs(result["accuracy"] - expected_acc) < 1e-9

    def test_details_have_required_fields(self):
        from gdn2_ruler import evaluate_retrieval

        model = _MockModel()
        tokenizer = _MockTokenizer()
        prompts = self._make_prompts(1)
        result = evaluate_retrieval(model, tokenizer, prompts)
        detail = result["details"][0]
        for field in (
            "seed",
            "query_key",
            "correct_answer",
            "hit",
            "correct_logprob",
            "margin",
            "num_candidates",
            "scores",
        ):
            assert field in detail

    def test_hit_is_bool(self):
        from gdn2_ruler import evaluate_retrieval

        model = _MockModel()
        tokenizer = _MockTokenizer()
        prompts = self._make_prompts(2)
        result = evaluate_retrieval(model, tokenizer, prompts)
        for d in result["details"]:
            assert isinstance(d["hit"], bool)

    def test_max_time_budget(self):
        """With negative max_time_secs, the break triggers on first iteration."""
        from gdn2_ruler import evaluate_retrieval

        model = _MockModel()
        tokenizer = _MockTokenizer()
        prompts = self._make_prompts(3)
        result = evaluate_retrieval(model, tokenizer, prompts, max_time_secs=-1)
        assert result["num_prompts"] == 0
        assert result["accuracy"] == 0.0

    def test_margin_zero_when_correct_is_best(self):
        """When the correct answer has the highest logprob, margin should be 0."""
        from gdn2_ruler import evaluate_retrieval

        model = _MockModel()
        tokenizer = _MockTokenizer()
        prompts = self._make_prompts(1)
        result = evaluate_retrieval(model, tokenizer, prompts)
        detail = result["details"][0]
        if detail["hit"]:
            assert detail["margin"] == 0.0

    def test_margin_negative_when_correct_not_best(self):
        """When the correct answer is NOT the highest, margin should be negative."""
        from gdn2_ruler import evaluate_retrieval

        model = _MockModel()
        tokenizer = _MockTokenizer()
        prompts = self._make_prompts(5)
        result = evaluate_retrieval(model, tokenizer, prompts)
        for d in result["details"]:
            if not d["hit"]:
                assert d["margin"] < 0.0

    def test_scores_sorted_descending(self):
        """Scores list should be sorted by total_logprob descending."""
        from gdn2_ruler import evaluate_retrieval

        model = _MockModel()
        tokenizer = _MockTokenizer()
        prompts = self._make_prompts(2)
        result = evaluate_retrieval(model, tokenizer, prompts)
        for d in result["details"]:
            lps = [s["total_logprob"] for s in d["scores"]]
            assert lps == sorted(lps, reverse=True)
