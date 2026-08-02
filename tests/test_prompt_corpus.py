"""Tests for bench/prompt_corpus.py — needle-in-haystack and multi-key prompts.

Bead ``ob-del``. Tests determinism, structure, and coverage of the prompt corpus.
"""

import json
import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)
_BENCH_DIR = os.path.join(_REPO_ROOT, "bench")
sys.path.insert(0, _BENCH_DIR)

from prompt_corpus import (  # noqa: E402
    CANONICAL_LENGTHS,
    generate_corpus,
    generate_multikey_prompt,
    generate_needle_prompt,
    write_csv,
    write_jsonl,
)


class TestDeterminism:
    def test_needle_prompt_reproducible(self):
        a = generate_needle_prompt(4096, 0, 0.5)
        b = generate_needle_prompt(4096, 0, 0.5)
        assert a.full_prompt == b.full_prompt

    def test_multikey_prompt_reproducible(self):
        a = generate_multikey_prompt(4096, 3)
        b = generate_multikey_prompt(4096, 3)
        assert a.full_prompt == b.full_prompt

    def test_different_seeds_differ(self):
        a = generate_multikey_prompt(4096, 3, seed=1)
        b = generate_multikey_prompt(4096, 3, seed=2)
        assert a.full_prompt != b.full_prompt

    def test_different_depths_differ(self):
        start = generate_needle_prompt(4096, 0, 0.0)
        mid = generate_needle_prompt(4096, 0, 0.5)
        end = generate_needle_prompt(4096, 0, 1.0)
        assert start.full_prompt != mid.full_prompt
        assert mid.full_prompt != end.full_prompt


class TestNeedlePrompt:
    def test_contains_needle_text(self):
        np = generate_needle_prompt(4096, 0, 0.5)
        assert np.needle_text in np.full_prompt

    def test_contains_question(self):
        np = generate_needle_prompt(4096, 0, 0.5)
        assert np.question in np.full_prompt

    def test_estimated_tokens_positive(self):
        np = generate_needle_prompt(4096, 0, 0.5)
        assert np.estimated_tokens > 0

    def test_estimated_tokens_near_target(self):
        np = generate_needle_prompt(4096, 0, 0.5)
        # Should be within 20% of target (filler is approximate)
        assert 3000 < np.estimated_tokens < 6000


class TestMultiKeyPrompt:
    def test_contains_all_needles(self):
        mk = generate_multikey_prompt(4096, 3)
        for kv in mk.needles:
            assert kv["needle"] in mk.full_prompt

    def test_expected_answers_populated(self):
        mk = generate_multikey_prompt(4096, 3)
        assert len(mk.expected_answers) == 3

    def test_unique_keys(self):
        mk = generate_multikey_prompt(4096, 5)
        keys = [kv["key"] for kv in mk.needles]
        assert len(set(keys)) == 5


class TestCorpus:
    def test_full_corpus_structure(self):
        entries = generate_corpus(context_lengths=[4096])
        # 5 depths + 3 multikey = 8 per context length
        assert len(entries) == 8

    def test_all_context_lengths(self):
        entries = generate_corpus()
        ctx_values = {e.context_length for e in entries}
        assert ctx_values == set(CANONICAL_LENGTHS)

    def test_prompt_ids_unique(self):
        entries = generate_corpus(context_lengths=[4096, 32768])
        ids = [e.prompt_id for e in entries]
        assert len(ids) == len(set(ids))


class TestSerialization:
    def test_csv_round_trip(self, tmp_path):
        entries = generate_corpus(context_lengths=[4096])
        path = str(tmp_path / "corpus.csv")
        write_csv(entries, path)
        assert os.path.exists(path)

    def test_jsonl_round_trip(self, tmp_path):
        entries = generate_corpus(context_lengths=[4096])
        path = str(tmp_path / "corpus.jsonl")
        write_jsonl(entries, path)
        assert os.path.exists(path)
        # Verify each line is valid JSON
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                assert "prompt_id" in obj
                assert "full_prompt" in obj
