"""Comprehensive tests for bench/prompts.py — long-context prompt corpus generator.

Tests determinism (same seed → identical output), needle positioning, multi-key
distribution, token estimation accuracy, corpus file generation, and CLI behavior.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

# Ensure repo root is importable
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.prompts import (  # noqa: E402
    BASE_SEED,
    CANONICAL_SWEEP_POINTS,
    DEFAULT_DEPTHS,
    DEFAULT_MULTIKEY_COUNTS,
    TARGET_FILLER_RATIO,
    TOKENS_PER_WORD,
    MultiKeyResult,
    NIAHResult,
    _estimate_tokens,
    _generate_filler,
    _sweep_label,
    _target_filler_words,
    generate_corpus,
    generate_multikey,
    generate_niah,
    main,
)


# ---------------------------------------------------------------------------
# _target_filler_words
# ---------------------------------------------------------------------------
class TestTargetFillerWords:
    def test_basic_conversion(self):
        """Standard token-to-word conversion with filler ratio."""
        result = _target_filler_words(4096)
        expected = int(4096 * TARGET_FILLER_RATIO / TOKENS_PER_WORD)
        assert result == expected

    def test_returns_at_least_one(self):
        """Even zero or negative targets should not return < 1."""
        assert _target_filler_words(0) >= 1
        assert _target_filler_words(-100) >= 1

    def test_scales_linearly(self):
        """Double the target tokens → double the filler words."""
        small = _target_filler_words(4096)
        large = _target_filler_words(8192)
        assert large == pytest.approx(small * 2, abs=1)

    @pytest.mark.parametrize("tokens", [128, 512, 2048, 4096, 32768])
    def test_all_canonical_lengths(self, tokens):
        """All canonical context lengths produce a positive integer."""
        result = _target_filler_words(tokens)
        assert isinstance(result, int)
        assert result > 0


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------
class TestEstimateTokens:
    def test_known_word_count(self):
        """10 words → ~13 tokens at 1.33 tokens/word."""
        text = "one two three four five six seven eight nine ten"
        assert _estimate_tokens(text) == int(10 * TOKENS_PER_WORD)

    def test_empty_string(self):
        assert _estimate_tokens("") == 0

    def test_single_word(self):
        assert _estimate_tokens("hello") == int(TOKENS_PER_WORD)

    def test_round_trip_approximate(self):
        """generate_filler at target N → estimate_tokens ≈ N (within ratio)."""
        rng = random.Random(BASE_SEED)
        words = _target_filler_words(2048)
        text = _generate_filler(rng, words)
        estimated = _estimate_tokens(text)
        # Should be in the ballpark of the target
        assert estimated > 1000
        assert estimated < 4000


# ---------------------------------------------------------------------------
# _generate_filler
# ---------------------------------------------------------------------------
class TestGenerateFiller:
    def test_deterministic(self):
        """Same seed produces identical output."""
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        assert _generate_filler(rng1, 50) == _generate_filler(rng2, 50)

    def test_different_seeds_differ(self):
        rng1 = random.Random(123)
        rng2 = random.Random(456)
        assert _generate_filler(rng1, 50) != _generate_filler(rng2, 50)

    def test_exact_word_count(self):
        """The filler text has exactly the requested number of words."""
        rng = random.Random(BASE_SEED)
        for n in [1, 5, 10, 50, 100]:
            text = _generate_filler(rng, n)
            assert len(text.split()) == n

    def test_empty_request(self):
        """Zero words → empty string."""
        rng = random.Random(BASE_SEED)
        assert _generate_filler(rng, 0) == ""

    def test_single_word(self):
        rng = random.Random(BASE_SEED)
        text = _generate_filler(rng, 1)
        assert len(text.split()) == 1

    def test_contains_real_words(self):
        """Filler text uses words from the word bank, not random characters."""
        rng = random.Random(BASE_SEED)
        text = _generate_filler(rng, 100)
        words = text.split()
        assert len(words) == 100
        # Should contain periods (from sentence templates)
        assert "." in text


# ---------------------------------------------------------------------------
# generate_niah
# ---------------------------------------------------------------------------
class TestGenerateNiah:
    def test_returns_named_tuple(self):
        result = generate_niah(2048)
        assert isinstance(result, NIAHResult)

    def test_fields_present(self):
        result = generate_niah(2048)
        for field in (
            "prompt",
            "needle",
            "expected_answer",
            "needle_depth_pct",
            "est_tokens",
            "seed",
        ):
            assert hasattr(result, field)

    def test_deterministic(self):
        """Same seed and depth → identical prompt."""
        r1 = generate_niah(2048, 50, seed=42)
        r2 = generate_niah(2048, 50, seed=42)
        assert r1.prompt == r2.prompt
        assert r1.needle == r2.needle
        assert r1.expected_answer == r2.expected_answer

    def test_different_seeds_produce_different_needles(self):
        r1 = generate_niah(2048, 50, seed=42)
        r2 = generate_niah(2048, 50, seed=99)
        assert r1.needle != r2.needle

    def test_needle_appears_in_prompt(self):
        """The needle text must be present in the generated prompt."""
        result = generate_niah(2048, 50, seed=42)
        assert result.needle in result.prompt

    def test_expected_answer_is_in_needle(self):
        """The expected answer is the color-id portion of the needle."""
        result = generate_niah(2048, 50, seed=42)
        assert result.expected_answer in result.needle

    def test_expected_answer_format(self):
        """Expected answer follows the 'color-number' pattern."""
        result = generate_niah(2048, 50, seed=42)
        parts = result.expected_answer.split("-")
        assert len(parts) == 2
        # First part is a color name (alphabetic)
        assert parts[0].isalpha()
        # Second part is a 5-digit number
        assert len(parts[1]) == 5
        assert parts[1].isdigit()

    def test_question_at_end(self):
        """Prompt ends with a retrieval question."""
        result = generate_niah(2048, seed=42)
        assert "Question:" in result.prompt
        assert "passcode" in result.prompt.lower()

    def test_est_tokens_positive(self):
        result = generate_niah(2048, seed=42)
        assert result.est_tokens > 0

    def test_seed_preserved(self):
        result = generate_niah(2048, seed=777)
        assert result.seed == 777

    @pytest.mark.parametrize("depth", [0, 10, 25, 50, 75, 90, 100])
    def test_all_default_depths(self, depth):
        """All default depths produce valid results."""
        result = generate_niah(512, depth, seed=42)
        assert result.needle_depth_pct == depth
        assert result.needle in result.prompt

    def test_depth_affects_needle_position(self):
        """At depth 0, needle is near the start; at 100, near the end."""
        r0 = generate_niah(2048, 0, seed=42)
        r100 = generate_niah(2048, 100, seed=42)
        pos0 = r0.prompt.index(r0.needle)
        pos100 = r100.prompt.index(r100.needle)
        assert pos0 < pos100

    def test_depth_0_needle_at_start(self):
        """At depth 0%, the needle should be very early in the text."""
        result = generate_niah(2048, 0, seed=42)
        # Needle position should be in the first 5% of the prompt
        needle_pos = result.prompt.index(result.needle)
        assert needle_pos / len(result.prompt) < 0.05

    def test_depth_100_needle_at_end(self):
        """At depth 100%, the needle should be near the end."""
        result = generate_niah(2048, 100, seed=42)
        prompt = result.prompt
        needle_end = prompt.index(result.needle) + len(result.needle)
        # The needle should be in the last 15% (before the question)
        assert needle_end / len(prompt) > 0.80

    def test_small_target_tokens(self):
        """Very small target tokens still produce a valid prompt."""
        result = generate_niah(128, 50, seed=42)
        assert result.needle in result.prompt
        assert result.est_tokens > 0

    def test_large_target_tokens(self):
        """Large target tokens produce a long prompt."""
        result = generate_niah(8192, 50, seed=42)
        assert len(result.prompt) > 10000  # at least 10K chars for 8K tokens


# ---------------------------------------------------------------------------
# generate_multikey
# ---------------------------------------------------------------------------
class TestGenerateMultikey:
    def test_returns_named_tuple(self):
        result = generate_multikey(2048)
        assert isinstance(result, MultiKeyResult)

    def test_fields_present(self):
        result = generate_multikey(2048)
        for field in ("prompt", "query_key", "expected_answer", "num_keys", "est_tokens", "seed"):
            assert hasattr(result, field)

    def test_deterministic(self):
        r1 = generate_multikey(2048, 10, seed=42)
        r2 = generate_multikey(2048, 10, seed=42)
        assert r1.prompt == r2.prompt
        assert r1.query_key == r2.query_key
        assert r1.expected_answer == r2.expected_answer

    def test_different_seeds_differ(self):
        r1 = generate_multikey(2048, 10, seed=42)
        r2 = generate_multikey(2048, 10, seed=99)
        assert r1.prompt != r2.prompt

    def test_query_key_in_prompt(self):
        result = generate_multikey(2048, 10, seed=42)
        assert result.query_key in result.prompt

    def test_expected_answer_in_prompt(self):
        result = generate_multikey(2048, 10, seed=42)
        assert result.expected_answer in result.prompt

    def test_query_key_format(self):
        """Query key follows 'item_XXXX' pattern."""
        result = generate_multikey(2048, 10, seed=42)
        assert result.query_key.startswith("item_")
        suffix = result.query_key.split("_")[1]
        assert suffix.isdigit()

    def test_expected_answer_format(self):
        """Expected answer follows 'value_XXXX' pattern."""
        result = generate_multikey(2048, 10, seed=42)
        assert result.expected_answer.startswith("value_")
        suffix = result.expected_answer.split("_")[1]
        assert suffix.isdigit()

    def test_num_keys_preserved(self):
        result = generate_multikey(2048, 15, seed=42)
        assert result.num_keys == 15

    def test_all_keys_in_prompt(self):
        """All num_keys key-value pairs should appear in the prompt."""
        result = generate_multikey(2048, 10, seed=42)
        # Count occurrences of "item_" and "= value_" patterns
        item_count = result.prompt.count("item_")
        assert item_count >= result.num_keys

    def test_question_at_end(self):
        result = generate_multikey(2048, 10, seed=42)
        assert "Question:" in result.prompt

    def test_unique_key_ids(self):
        """All generated key IDs are unique (query key appears in question too)."""
        import re

        result = generate_multikey(2048, 20, seed=42)
        ids = re.findall(r"item_(\d+)", result.prompt)
        # The query key appears twice: once in context, once in the question
        unique_ids = set(ids)
        assert len(unique_ids) == result.num_keys
        query_id = result.query_key.split("_")[1]
        assert ids.count(query_id) == 2

    def test_single_key(self):
        """num_keys=1 should still produce a valid prompt."""
        result = generate_multikey(2048, 1, seed=42)
        assert result.num_keys == 1
        assert result.query_key in result.prompt
        assert result.expected_answer in result.prompt

    def test_many_keys(self):
        """Large num_keys should distribute across the context."""
        result = generate_multikey(8192, 50, seed=42)
        assert result.num_keys == 50
        item_count = result.prompt.count("item_")
        assert item_count >= 50

    def test_est_tokens_positive(self):
        result = generate_multikey(2048, seed=42)
        assert result.est_tokens > 0

    def test_seed_preserved(self):
        result = generate_multikey(2048, seed=555)
        assert result.seed == 555


# ---------------------------------------------------------------------------
# _sweep_label
# ---------------------------------------------------------------------------
class TestSweepLabel:
    @pytest.mark.parametrize(
        "tokens,expected",
        [
            (4096, "4K"),
            (32768, "32K"),
            (131072, "128K"),
            (262144, "262K"),
        ],
    )
    def test_canonical_labels(self, tokens, expected):
        assert _sweep_label(tokens) == expected

    @pytest.mark.parametrize(
        "tokens,expected",
        [
            (1024, "1K"),
            (2048, "2K"),
            (8192, "8K"),
            (65536, "64K"),
        ],
    )
    def test_non_canonical_labels(self, tokens, expected):
        """Non-canonical multiples of 1024 get K labels."""
        assert _sweep_label(tokens) == expected

    def test_sub_1024(self):
        """Values below 1024 are returned as-is."""
        assert _sweep_label(512) == "512"
        assert _sweep_label(128) == "128"


# ---------------------------------------------------------------------------
# generate_corpus
# ---------------------------------------------------------------------------
class TestGenerateCorpus:
    def test_generates_files(self, tmp_path):
        written = generate_corpus(
            output_dir=tmp_path,
            sweep_points=[128],
            depths=[50],
            multikey_counts=[3],
        )
        assert len(written) > 0
        for p in written:
            assert p.exists()
            assert p.suffix == ".json"

    def test_niah_filenames(self, tmp_path):
        written = generate_corpus(
            output_dir=tmp_path,
            sweep_points=[4096],
            depths=[0, 50],
            multikey_counts=[],
        )
        names = [p.name for p in written]
        assert "niah_4K_d0.json" in names
        assert "niah_4K_d50.json" in names

    def test_multikey_filenames(self, tmp_path):
        written = generate_corpus(
            output_dir=tmp_path,
            sweep_points=[4096],
            depths=[],
            multikey_counts=[3, 10],
        )
        names = [p.name for p in written]
        assert "multikey_4K_k3.json" in names
        assert "multikey_4K_k10.json" in names

    def test_json_structure_niah(self, tmp_path):
        generate_corpus(
            output_dir=tmp_path,
            sweep_points=[128],
            depths=[50],
            multikey_counts=[],
        )
        data = json.loads((tmp_path / "niah_128_d50.json").read_text())
        assert data["task"] == "niah"
        assert "prompt" in data
        assert "needle" in data
        assert "expected_answer" in data
        assert "est_tokens" in data
        assert "seed" in data
        assert data["target_tokens"] == 128
        assert data["needle_depth_pct"] == 50

    def test_json_structure_multikey(self, tmp_path):
        generate_corpus(
            output_dir=tmp_path,
            sweep_points=[128],
            depths=[],
            multikey_counts=[5],
        )
        data = json.loads((tmp_path / "multikey_128_k5.json").read_text())
        assert data["task"] == "multikey"
        assert "prompt" in data
        assert "query_key" in data
        assert "expected_answer" in data
        assert "est_tokens" in data
        assert "seed" in data
        assert data["target_tokens"] == 128
        assert data["num_keys"] == 5

    def test_full_sweep_file_count(self, tmp_path):
        """Full sweep produces len(sweep) * (len(depths) + len(multikey)) files."""
        n_sweep = 1
        n_depths = len(DEFAULT_DEPTHS)
        n_multikey = len(DEFAULT_MULTIKEY_COUNTS)
        expected = n_sweep * (n_depths + n_multikey)

        written = generate_corpus(
            output_dir=tmp_path,
            sweep_points=[4096],
        )
        assert len(written) == expected

    def test_creates_output_dir(self, tmp_path):
        """Output directory is created if it doesn't exist."""
        new_dir = tmp_path / "new" / "subdir"
        generate_corpus(
            output_dir=new_dir,
            sweep_points=[128],
            depths=[50],
            multikey_counts=[3],
        )
        assert new_dir.exists()

    def test_json_serializable(self, tmp_path):
        """All generated JSON files are valid JSON."""
        written = generate_corpus(
            output_dir=tmp_path,
            sweep_points=[128],
            depths=[0, 50, 100],
            multikey_counts=[1, 10],
        )
        for p in written:
            data = json.loads(p.read_text())
            assert isinstance(data, dict)

    def test_default_depths_used(self, tmp_path):
        """When depths=None, all DEFAULT_DEPTHS are used."""
        written = generate_corpus(
            output_dir=tmp_path,
            sweep_points=[128],
            depths=None,
            multikey_counts=[],
        )
        niah_files = [p for p in written if p.name.startswith("niah_")]
        assert len(niah_files) == len(DEFAULT_DEPTHS)

    def test_default_multikey_used(self, tmp_path):
        """When multikey_counts=None, all DEFAULT_MULTIKEY_COUNTS are used."""
        written = generate_corpus(
            output_dir=tmp_path,
            sweep_points=[128],
            depths=[],
            multikey_counts=None,
        )
        mk_files = [p for p in written if p.name.startswith("multikey_")]
        assert len(mk_files) == len(DEFAULT_MULTIKEY_COUNTS)

    def test_default_sweep_points(self, tmp_path):
        """When sweep_points=None, CANONICAL_SWEEP_POINTS are used."""
        written = generate_corpus(
            output_dir=tmp_path,
            depths=[50],
            multikey_counts=[3],
        )
        # Should cover all 4 canonical points
        labels = {_sweep_label(s) for s in CANONICAL_SWEEP_POINTS}
        for label in labels:
            niah = [p for p in written if f"niah_{label}_" in p.name]
            assert len(niah) == 1

    def test_empty_depths_and_multikey(self, tmp_path):
        """Empty lists produce no files."""
        written = generate_corpus(
            output_dir=tmp_path,
            sweep_points=[128],
            depths=[],
            multikey_counts=[],
        )
        assert written == []


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------
class TestConstants:
    def test_canonical_sweep_points_sorted(self):
        assert sorted(CANONICAL_SWEEP_POINTS) == CANONICAL_SWEEP_POINTS

    def test_canonical_sweep_points_unique(self):
        assert len(CANONICAL_SWEEP_POINTS) == len(set(CANONICAL_SWEEP_POINTS))

    def test_default_depths_range(self):
        """Depths span 0-100."""
        assert DEFAULT_DEPTHS[0] == 0
        assert DEFAULT_DEPTHS[-1] == 100

    def test_default_depths_sorted(self):
        assert sorted(DEFAULT_DEPTHS) == DEFAULT_DEPTHS

    def test_default_multikey_counts_sorted(self):
        assert sorted(DEFAULT_MULTIKEY_COUNTS) == DEFAULT_MULTIKEY_COUNTS

    def test_default_multikey_positive(self):
        for c in DEFAULT_MULTIKEY_COUNTS:
            assert c > 0

    def test_tokens_per_word_reasonable(self):
        """Token-to-word ratio should be > 1 (English is sub-word tokenized)."""
        assert TOKENS_PER_WORD > 1.0
        assert TOKENS_PER_WORD < 3.0

    def test_filler_ratio_in_range(self):
        """Filler should be most of the context."""
        assert 0.8 < TARGET_FILLER_RATIO < 1.0

    def test_base_seed_is_int(self):
        assert isinstance(BASE_SEED, int)


# ---------------------------------------------------------------------------
# NamedTuple structure
# ---------------------------------------------------------------------------
class TestNamedTupleStructure:
    def test_niah_result_fields(self):
        assert NIAHResult._fields == (
            "prompt",
            "needle",
            "expected_answer",
            "needle_depth_pct",
            "est_tokens",
            "seed",
        )

    def test_multikey_result_fields(self):
        assert MultiKeyResult._fields == (
            "prompt",
            "query_key",
            "expected_answer",
            "num_keys",
            "est_tokens",
            "seed",
        )

    def test_niah_result_is_tuple(self):
        result = generate_niah(128, seed=42)
        assert isinstance(result, tuple)

    def test_multikey_result_is_tuple(self):
        result = generate_multikey(128, seed=42)
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# CLI / main()
# ---------------------------------------------------------------------------
class TestMain:
    def test_no_args_returns_error(self, monkeypatch, capsys):
        """No task flags → prints help, returns 1."""
        monkeypatch.setattr(sys, "argv", ["bench.prompts"])
        rc = main([])
        assert rc == 1

    def test_niah_flag(self, tmp_path, monkeypatch):
        """--niah generates NIAH files."""
        rc = main(["--niah", "--output", str(tmp_path), "--depths", "50"])
        assert rc == 0
        niah_files = list(tmp_path.glob("niah_*.json"))
        assert len(niah_files) > 0

    def test_multikey_flag(self, tmp_path):
        """--multikey generates multi-key files."""
        rc = main(["--multikey", "--output", str(tmp_path), "--keys", "5"])
        assert rc == 0
        mk_files = list(tmp_path.glob("multikey_*.json"))
        assert len(mk_files) > 0

    def test_all_flag(self, tmp_path):
        """--all generates both NIAH and multi-key files."""
        rc = main(["--all", "--output", str(tmp_path), "--depths", "50", "--keys", "5"])
        assert rc == 0
        niah_files = list(tmp_path.glob("niah_*.json"))
        mk_files = list(tmp_path.glob("multikey_*.json"))
        assert len(niah_files) > 0
        assert len(mk_files) > 0

    def test_custom_depths(self, tmp_path):
        """Custom depths are parsed correctly."""
        rc = main(["--niah", "--output", str(tmp_path), "--depths", "0,100"])
        assert rc == 0
        files = sorted(p.name for p in tmp_path.glob("niah_*.json"))
        assert any("d0.json" in f for f in files)
        assert any("d100.json" in f for f in files)

    def test_custom_keys(self, tmp_path):
        """Custom key counts are parsed correctly."""
        rc = main(["--multikey", "--output", str(tmp_path), "--keys", "1,50"])
        assert rc == 0
        files = sorted(p.name for p in tmp_path.glob("multikey_*.json"))
        assert any("k1.json" in f for f in files)
        assert any("k50.json" in f for f in files)

    def test_niah_only_skips_multikey(self, tmp_path):
        """--niah only should not produce multikey files."""
        rc = main(["--niah", "--output", str(tmp_path), "--depths", "50"])
        assert rc == 0
        mk_files = list(tmp_path.glob("multikey_*.json"))
        assert len(mk_files) == 0

    def test_multikey_only_skips_niah(self, tmp_path):
        """--multikey only should not produce NIAH files."""
        rc = main(["--multikey", "--output", str(tmp_path), "--keys", "5"])
        assert rc == 0
        niah_files = list(tmp_path.glob("niah_*.json"))
        assert len(niah_files) == 0

    def test_output_to_stdout(self, tmp_path, capsys):
        """CLI prints a summary of generated files."""
        rc = main(["--niah", "--output", str(tmp_path), "--depths", "50"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Generated" in captured.out
