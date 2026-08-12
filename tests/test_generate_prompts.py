# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/generate_prompts.py — long-context prompt corpus (ob-del)."""

import json
import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import random  # noqa: E402

from scripts.generate_prompts import (  # noqa: E402
    CANONICAL_LENGTHS,
    CHARS_PER_TOKEN,
    MASTER_SEED,
    _filler_text,
    _random_key,
    _random_value,
    generate_all,
    generate_needle,
    generate_ruler,
    main,
)

# ---------------------------------------------------------------------------
# _filler_text
# ---------------------------------------------------------------------------


class TestFillerText:
    def test_returns_string(self):
        rng = random.Random(42)
        text = _filler_text(500, rng)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_approximate_length(self):
        rng = random.Random(42)
        text = _filler_text(500, rng)
        # Allow generous tolerance since we truncate to target
        assert len(text) <= 500
        assert len(text) > 300  # at least most of the target

    def test_zero_target_returns_empty(self):
        rng = random.Random(42)
        assert _filler_text(0, rng) == ""

    def test_deterministic(self):
        a = _filler_text(500, random.Random(99))
        b = _filler_text(500, random.Random(99))
        assert a == b


# ---------------------------------------------------------------------------
# Key/value generators
# ---------------------------------------------------------------------------


class TestKeyValueGen:
    def test_key_format(self):
        rng = random.Random(42)
        key = _random_key(rng)
        parts = key.split("-")
        assert len(parts) == 2
        assert len(parts[0]) == 3
        assert len(parts[1]) == 4

    def test_value_format(self):
        rng = random.Random(42)
        val = _random_value(rng)
        parts = val.split("-")
        assert len(parts) == 2
        assert len(parts[0]) == 3
        assert len(parts[1]) == 4

    def test_key_deterministic(self):
        a = _random_key(random.Random(42))
        b = _random_key(random.Random(42))
        assert a == b


# ---------------------------------------------------------------------------
# generate_needle
# ---------------------------------------------------------------------------


class TestGenerateNeedle:
    @pytest.fixture
    def result(self):
        rng = random.Random(MASTER_SEED + 4096)
        return generate_needle(4096, rng)

    def test_returns_tuple(self, result):
        prompt, meta = result
        assert isinstance(prompt, str)
        assert isinstance(meta, dict)

    def test_prompt_contains_question(self, result):
        prompt, meta = result
        assert meta["question"] in prompt

    def test_prompt_contains_needle(self, result):
        prompt, meta = result
        assert meta["needle"] in prompt

    def test_metadata_type(self, result):
        _, meta = result
        assert meta["type"] == "needle_in_haystack"

    def test_metadata_has_target_tokens(self, result):
        _, meta = result
        assert meta["target_tokens"] == 4096

    def test_metadata_has_expected_answer(self, result):
        _, meta = result
        assert meta["expected_answer"]
        assert len(meta["expected_answer"]) > 0

    def test_needle_depth_near_middle(self, result):
        _, meta = result
        depth = meta["needle_depth_fraction"]
        assert 0.2 < depth < 0.8

    def test_prompt_approximate_length(self, result):
        prompt, _ = result
        actual_tokens = len(prompt) // CHARS_PER_TOKEN
        # Should be within ~30% of target (filler is approximate)
        assert 2500 < actual_tokens < 5500

    def test_deterministic(self):
        ctx = 4096
        a = generate_needle(ctx, random.Random(MASTER_SEED + ctx))
        b = generate_needle(ctx, random.Random(MASTER_SEED + ctx))
        assert a[0] == b[0]
        assert a[1] == b[1]

    def test_small_target_no_paragraph_boundary(self):
        """Very small target → single-paragraph filler, no \\n\\n boundary found."""
        rng = random.Random(MASTER_SEED + 40)
        prompt, meta = generate_needle(40, rng)
        assert isinstance(prompt, str)
        assert meta["question"] in prompt


# ---------------------------------------------------------------------------
# generate_ruler
# ---------------------------------------------------------------------------


class TestGenerateRuler:
    @pytest.fixture
    def result(self):
        rng = random.Random(MASTER_SEED + 4096 + 100000)
        return generate_ruler(4096, rng)

    def test_returns_tuple(self, result):
        prompt, meta = result
        assert isinstance(prompt, str)
        assert isinstance(meta, dict)

    def test_metadata_type(self, result):
        _, meta = result
        assert meta["type"] == "ruler_multi_key"

    def test_metadata_has_num_keys(self, result):
        _, meta = result
        assert meta["num_keys"] >= 3
        assert meta["num_keys"] <= 20

    def test_queried_keys_subset(self, result):
        _, meta = result
        assert len(meta["queried_keys"]) == 5

    def test_expected_answers_count(self, result):
        _, meta = result
        assert len(meta["expected_answers"]) == 5

    def test_prompt_contains_all_records(self, result):
        prompt, meta = result
        # Every key should appear somewhere in the prompt
        for key in meta["queried_keys"]:
            assert key in prompt

    def test_prompt_contains_question(self, result):
        prompt, meta = result
        # The question lists the queried keys
        for key in meta["queried_keys"]:
            assert key in prompt

    def test_num_keys_scales_with_context(self):
        small = generate_ruler(2048, random.Random(1))
        large = generate_ruler(65536, random.Random(1))
        assert small[1]["num_keys"] <= large[1]["num_keys"]

    def test_deterministic(self):
        ctx = 4096
        a = generate_ruler(ctx, random.Random(MASTER_SEED + ctx + 100000))
        b = generate_ruler(ctx, random.Random(MASTER_SEED + ctx + 100000))
        assert a[0] == b[0]
        assert a[1] == b[1]


# ---------------------------------------------------------------------------
# generate_all
# ---------------------------------------------------------------------------


class TestGenerateAll:
    def test_generates_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.generate_prompts.OUTPUT_DIR", tmp_path)
        files = generate_all((4096,))
        assert len(files) > 0
        for f in files:
            assert f.exists()

    def test_file_types_per_context(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.generate_prompts.OUTPUT_DIR", tmp_path)
        generate_all((4096,))
        # 2 types × 2 files (txt + json) + 1 manifest = 5
        names = sorted(p.name for p in tmp_path.iterdir())
        assert "needle_4096.txt" in names
        assert "needle_4096.json" in names
        assert "ruler_4096.txt" in names
        assert "ruler_4096.json" in names
        assert "manifest.json" in names

    def test_manifest_structure(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.generate_prompts.OUTPUT_DIR", tmp_path)
        generate_all((4096,))
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["generator"] == "scripts/generate_prompts.py"
        assert manifest["master_seed"] == MASTER_SEED
        assert manifest["chars_per_token"] == CHARS_PER_TOKEN
        assert 4096 in manifest["context_lengths"]

    def test_json_metadata_valid(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.generate_prompts.OUTPUT_DIR", tmp_path)
        generate_all((4096,))
        meta = json.loads((tmp_path / "needle_4096.json").read_text())
        assert meta["type"] == "needle_in_haystack"
        assert meta["target_tokens"] == 4096

    def test_multiple_context_lengths(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.generate_prompts.OUTPUT_DIR", tmp_path)
        generate_all((4096, 32768))
        names = sorted(p.name for p in tmp_path.iterdir())
        assert any("4096" in n for n in names)
        assert any("32768" in n for n in names)


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------


class TestMain:
    def test_small_flag(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.generate_prompts.OUTPUT_DIR", tmp_path)
        rc = main(["--small"])
        assert rc == 0
        names = sorted(p.name for p in tmp_path.iterdir())
        assert any("4096" in n for n in names)
        assert any("32768" in n for n in names)
        # --small should NOT generate 128K or 262K
        assert not any("131072" in n for n in names)
        assert not any("262144" in n for n in names)

    def test_default_generates_all_lengths(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.generate_prompts.OUTPUT_DIR", tmp_path)
        rc = main([])
        assert rc == 0
        names = sorted(p.name for p in tmp_path.iterdir())
        for ctx in CANONICAL_LENGTHS:
            assert any(str(ctx) in n for n in names)
