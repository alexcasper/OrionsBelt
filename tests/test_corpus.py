"""Tests for bench/corpus.py — long-context prompt corpus generation.

Bead ob-del.
"""

import json
import os
import tempfile

from corpus import (
    DEFAULT_CHARS_PER_TOKEN,
    FILLER_PASSAGES,
    MASTER_SEED,
    MULTIKEY_COUNTS,
    NIAH_DEPTHS,
    CorpusConfig,
    _depth_seed,
    _task_seed,
    generate_corpus,
    generate_haystack,
    generate_niah_multikey,
    generate_niah_single,
    save_corpus,
    save_manifest,
)

# ---------------------------------------------------------------------------
# Haystack generation
# ---------------------------------------------------------------------------


class TestGenerateHaystack:
    def test_approximate_length(self):
        import random
        rng = random.Random(42)
        target = 10000
        text = generate_haystack(target, rng)
        # Should be approximately the target (within 10%)
        assert abs(len(text) - target) < target * 0.15

    def test_deterministic_with_seed(self):
        import random
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        text1 = generate_haystack(5000, rng1)
        text2 = generate_haystack(5000, rng2)
        assert text1 == text2

    def test_different_seeds_differ(self):
        import random
        rng1 = random.Random(42)
        rng2 = random.Random(99)
        text1 = generate_haystack(5000, rng1)
        text2 = generate_haystack(5000, rng2)
        assert text1 != text2

    def test_contains_known_passage(self):
        import random
        rng = random.Random(42)
        text = generate_haystack(2000, rng)
        # Should contain text from at least one filler passage
        assert any(p[:50] in text for p in FILLER_PASSAGES)


# ---------------------------------------------------------------------------
# NIAH single needle
# ---------------------------------------------------------------------------


class TestNiahSingle:
    def test_basic_generation(self):
        item = generate_niah_single(4096, 0.5, seed=12345)
        assert item.task_type == "niah_single"
        assert item.context_length == 4096
        assert len(item.prompt) > 0
        assert len(item.expected_answer) > 0
        assert item.needle_depth == 0.5
        assert item.seed == 12345

    def test_prompt_is_right_length(self):
        item = generate_niah_single(4096, 0.5, seed=12345)
        est_tokens = int(len(item.prompt) / DEFAULT_CHARS_PER_TOKEN)
        # Should be within 20% of target
        assert abs(est_tokens - 4096) < 4096 * 0.25

    def test_answer_appears_in_metadata(self):
        item = generate_niah_single(4096, 0.5, seed=12345)
        assert "needle_value" in item.metadata
        assert item.metadata["needle_value"] == item.expected_answer

    def test_deterministic(self):
        item1 = generate_niah_single(4096, 0.5, seed=12345)
        item2 = generate_niah_single(4096, 0.5, seed=12345)
        assert item1.prompt == item2.prompt
        assert item1.expected_answer == item2.expected_answer

    def test_different_depths_differ(self):
        item_start = generate_niah_single(4096, 0.0, seed=12345)
        item_mid = generate_niah_single(4096, 0.5, seed=12345)
        item_end = generate_niah_single(4096, 1.0, seed=12345)
        # The needle is at different positions, so the prompts differ
        assert item_start.prompt != item_mid.prompt
        assert item_mid.prompt != item_end.prompt

    def test_needle_is_in_prompt(self):
        item = generate_niah_single(4096, 0.5, seed=12345)
        # The needle value should appear somewhere in the prompt
        assert item.metadata["needle_value"] in item.prompt

    def test_question_at_end(self):
        item = generate_niah_single(4096, 0.5, seed=12345)
        # The question should be at the end of the prompt
        question_marker = "What is"
        assert question_marker in item.prompt[-200:]


# ---------------------------------------------------------------------------
# RULER Multi-key
# ---------------------------------------------------------------------------


class TestNiahMultikey:
    def test_basic_generation(self):
        item = generate_niah_multikey(4096, num_keys=10, seed=99999)
        assert item.task_type == "niah_multikey"
        assert item.context_length == 4096
        assert len(item.prompt) > 0
        assert len(item.expected_answer) > 0
        assert item.metadata["num_keys"] == 10

    def test_default_key_count(self):
        item = generate_niah_multikey(4096, seed=99999)
        assert item.metadata["num_keys"] == MULTIKEY_COUNTS[4096]

    def test_keys_are_unique(self):
        item = generate_niah_multikey(4096, num_keys=10, seed=99999)
        keys = item.metadata["all_keys"]
        assert len(keys) == len(set(keys))  # all unique

    def test_query_key_is_in_keys(self):
        item = generate_niah_multikey(4096, num_keys=10, seed=99999)
        assert item.metadata["query_key"] in item.metadata["all_keys"]

    def test_expected_answer_in_prompt(self):
        item = generate_niah_multikey(4096, num_keys=10, seed=99999)
        assert item.expected_answer in item.prompt

    def test_deterministic(self):
        item1 = generate_niah_multikey(4096, num_keys=10, seed=99999)
        item2 = generate_niah_multikey(4096, num_keys=10, seed=99999)
        assert item1.prompt == item2.prompt
        assert item1.expected_answer == item2.expected_answer

    def test_key_count_scales_with_context(self):
        item_4k = generate_niah_multikey(4096, seed=42)
        item_32k = generate_niah_multikey(32768, seed=42)
        assert item_32k.metadata["num_keys"] > item_4k.metadata["num_keys"]


# ---------------------------------------------------------------------------
# Corpus batch generation
# ---------------------------------------------------------------------------


class TestGenerateCorpus:
    def test_full_corpus(self):
        config = CorpusConfig(
            context_lengths=[4096, 32768],
            tasks=["niah_single", "niah_multikey"],
        )
        items = generate_corpus(config)
        # NIAH single: 2 lengths × 5 depths = 10
        # Multi-key: 2 lengths × 1 = 2
        assert len(items) == 12

    def test_niah_only(self):
        config = CorpusConfig(
            context_lengths=[4096],
            tasks=["niah_single"],
        )
        items = generate_corpus(config)
        assert len(items) == len(NIAH_DEPTHS)  # 5 depths
        assert all(i.task_type == "niah_single" for i in items)

    def test_multikey_only(self):
        config = CorpusConfig(
            context_lengths=[4096, 32768],
            tasks=["niah_multikey"],
        )
        items = generate_corpus(config)
        assert len(items) == 2  # one per context length
        assert all(i.task_type == "niah_multikey" for i in items)

    def test_deterministic_seeds(self):
        config = CorpusConfig(context_lengths=[4096])
        items1 = generate_corpus(config)
        items2 = generate_corpus(config)
        for i1, i2 in zip(items1, items2, strict=False):
            assert i1.seed == i2.seed
            assert i1.prompt == i2.prompt

    def test_all_depths_present(self):
        config = CorpusConfig(
            context_lengths=[4096],
            tasks=["niah_single"],
        )
        items = generate_corpus(config)
        depths = sorted(i.needle_depth for i in items)
        assert depths == sorted(NIAH_DEPTHS)


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


class TestSeedDerivation:
    def test_depth_seed_deterministic(self):
        s1 = _depth_seed(42, 4096, 0.5)
        s2 = _depth_seed(42, 4096, 0.5)
        assert s1 == s2

    def test_depth_seed_differs_by_depth(self):
        s1 = _depth_seed(42, 4096, 0.0)
        s2 = _depth_seed(42, 4096, 0.5)
        s3 = _depth_seed(42, 4096, 1.0)
        assert s1 != s2
        assert s2 != s3
        assert s1 != s3

    def test_depth_seed_differs_by_context(self):
        s1 = _depth_seed(42, 4096, 0.5)
        s2 = _depth_seed(42, 32768, 0.5)
        assert s1 != s2

    def test_task_seed_deterministic(self):
        s1 = _task_seed(42, 4096, "multikey")
        s2 = _task_seed(42, 4096, "multikey")
        assert s1 == s2


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


class TestSaveCorpus:
    def test_saves_json_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            items = [
                generate_niah_single(4096, 0.0, seed=42),
                generate_niah_multikey(4096, num_keys=10, seed=99),
            ]
            paths = save_corpus(items, tmpdir)

            assert len(paths) == 2
            for p in paths:
                assert os.path.exists(p)
                assert p.endswith(".json")
                with open(p) as f:
                    data = json.load(f)
                assert "prompt" in data
                assert "expected_answer" in data
                assert "task_type" in data

    def test_filename_includes_depth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            item = generate_niah_single(4096, 0.75, seed=42)
            paths = save_corpus([item], tmpdir)
            assert "d075" in os.path.basename(paths[0])

    def test_filename_includes_key_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            item = generate_niah_multikey(4096, num_keys=50, seed=99)
            paths = save_corpus([item], tmpdir)
            assert "k0050" in os.path.basename(paths[0])

    def test_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "sub", "dir")
            items = [generate_niah_single(4096, 0.5, seed=42)]
            save_corpus(items, nested)
            assert os.path.isdir(nested)


class TestSaveManifest:
    def test_manifest_summarises_corpus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            items = generate_corpus(CorpusConfig(
                context_lengths=[4096, 32768],
                tasks=["niah_single"],
            ))
            manifest_path = save_manifest(items, tmpdir)

            with open(manifest_path) as f:
                manifest = json.load(f)

            assert manifest["total_prompts"] == len(items)
            assert manifest["master_seed"] == MASTER_SEED
            assert manifest["by_task"]["niah_single"] == len(items)
            assert "4096" in manifest["by_context"]
            assert "32768" in manifest["by_context"]

    def test_manifest_lists_context_lengths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            items = generate_corpus(CorpusConfig(
                context_lengths=[4096, 32768],
            ))
            manifest_path = save_manifest(items, tmpdir)

            with open(manifest_path) as f:
                manifest = json.load(f)

            assert manifest["context_lengths"] == [4096, 32768]


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_to_dict_round_trips(self):
        item = generate_niah_single(4096, 0.5, seed=42)
        d = item.to_dict()

        assert d["task_type"] == "niah_single"
        assert d["context_length"] == 4096
        assert d["prompt"] == item.prompt
        assert d["expected_answer"] == item.expected_answer
        assert d["needle_depth"] == 0.5
        assert "prompt_chars" in d
        assert "est_tokens" in d

    def test_multikey_to_dict(self):
        item = generate_niah_multikey(4096, num_keys=10, seed=99)
        d = item.to_dict()

        assert d["task_type"] == "niah_multikey"
        assert d["metadata"]["num_keys"] == 10
        assert d["metadata"]["query_key"] == item.metadata["query_key"]
