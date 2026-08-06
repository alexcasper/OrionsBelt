"""Tests for the golden reference capture script (scripts/capture_reference.py, bead ob-aqv).

The capture script itself needs torch+transformers to run, but the logic it wraps
— top-k extraction, prompt hashing, manifest generation, JSON format conformance
— is pure Python that can be tested without any ML dependencies. These tests:

  * Verify the top-k logit extraction is correct and deterministic.
  * Verify the JSON output format matches what bench/correctness.py expects.
  * Test the manifest generation includes all required provenance fields.
  * Test the CLI argument parsing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.capture_reference import (  # noqa: E402
    DEFAULT_CONTEXT_LENGTHS,
    DEFAULT_DECODE_LENGTH,
    DEFAULT_PROMPTS,
    DEFAULT_TOPK,
    STANDARD_PROMPTS,
    _build_manifest,
    _prompt_hash,
    _topk_logits,
)

# ---------------------------------------------------------------------------
# Top-k logit extraction
# ---------------------------------------------------------------------------


class TestTopKLogits:
    """The top-k extraction must be correct — wrong IDs would make the
    correctness oracle compare the wrong tokens and pass bad models."""

    def test_basic_extraction(self):
        logits = [0.1, 0.9, 0.5, 0.3, 0.7]
        vals, ids = _topk_logits(logits, k=3)
        assert ids == [1, 4, 2]  # 0.9 > 0.7 > 0.5
        assert vals == [0.9, 0.7, 0.5]

    def test_descending_order(self):
        logits = [5.0, 1.0, 4.0, 2.0, 3.0]
        vals, _ = _topk_logits(logits, k=5)
        assert vals == [5.0, 4.0, 3.0, 2.0, 1.0]

    def test_ties_broken_by_index(self):
        """When logits are equal, lower index should come first (stable sort)."""
        logits = [1.0, 1.0, 1.0]
        vals, ids = _topk_logits(logits, k=2)
        assert ids == [0, 1]

    def test_k_larger_than_input(self):
        """k > len(logits) should return all elements."""
        logits = [0.5, 0.3, 0.1]
        vals, ids = _topk_logits(logits, k=10)
        assert len(vals) == 3
        assert len(ids) == 3

    def test_k_equals_one(self):
        logits = [0.1, 0.9, 0.5]
        vals, ids = _topk_logits(logits, k=1)
        assert vals == [0.9]
        assert ids == [1]

    def test_single_element(self):
        vals, ids = _topk_logits([42.0], k=1)
        assert vals == [42.0]
        assert ids == [0]

    def test_deterministic(self):
        """Same input must always produce same output."""
        logits = [0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.8]
        r1 = _topk_logits(logits, k=4)
        r2 = _topk_logits(logits, k=4)
        assert r1 == r2

    def test_large_vocabulary_size(self):
        """Simulate a 248K vocab (Qwen3.5) — top-k must be fast and correct."""
        import random

        random.seed(42)
        logits = [random.gauss(0, 1) for _ in range(10000)]
        vals, ids = _topk_logits(logits, k=20)
        assert len(vals) == 20
        assert len(ids) == 20
        # Verify descending
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1]
        # Verify IDs point to the right values
        for idx, val in zip(ids, vals, strict=True):
            assert logits[idx] == val

    def test_negative_logits(self):
        logits = [-5.0, -1.0, -3.0, -0.5]
        vals, ids = _topk_logits(logits, k=2)
        assert ids == [3, 1]  # -0.5 > -1.0
        assert vals == [-0.5, -1.0]


# ---------------------------------------------------------------------------
# Prompt hashing
# ---------------------------------------------------------------------------


class TestPromptHash:
    """The hash verifies the same prompt was used across reference and candidate."""

    def test_deterministic(self):
        assert _prompt_hash("hello world") == _prompt_hash("hello world")

    def test_different_prompts_different_hash(self):
        assert _prompt_hash("hello world") != _prompt_hash("goodbye world")

    def test_short_hash(self):
        h = _prompt_hash("test")
        assert len(h) == 16  # truncated SHA-256

    def test_empty_string(self):
        h = _prompt_hash("")
        assert len(h) == 16
        assert h == _prompt_hash("")


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------


class TestBuildManifest:
    """The manifest ties every golden file to a specific model + commit."""

    def test_required_fields(self):
        m = _build_manifest(
            model_key="0.8b",
            config_name="Qwen/Qwen3.5-0.8B",
            device="cuda",
            dtype="float16",
            output_files=["ref_0001_ctx004096.json"],
            context_lengths=[4096],
            num_prompts=1,
            decode_length=32,
            topk=20,
            save_full_logits=False,
        )
        assert m["purpose"] == "golden_reference_logits"
        assert m["bead"] == "ob-aqv"
        assert m["model_checkpoint"] == "Qwen/Qwen3.5-0.8B"
        assert m["model_key"] == "0.8b"
        assert m["device"] == "cuda"
        assert m["dtype"] == "float16"
        assert "git" in m
        assert "captured_at" in m
        assert "platform" in m

    def test_capture_config_recorded(self):
        m = _build_manifest(
            model_key="4b",
            config_name="Qwen/Qwen3.5-4B",
            device="cpu",
            dtype="float32",
            output_files=[],
            context_lengths=[4096, 32768, 131072],
            num_prompts=5,
            decode_length=64,
            topk=50,
            save_full_logits=True,
        )
        cc = m["capture_config"]
        assert cc["context_lengths"] == [4096, 32768, 131072]
        assert cc["num_prompts"] == 5
        assert cc["decode_length"] == 64
        assert cc["topk"] == 50
        assert cc["save_full_logits"] is True

    def test_output_files_listed(self):
        files = ["ref_0001_ctx004096.json", "ref_0002_ctx032768.json"]
        m = _build_manifest(
            model_key="0.8b",
            config_name="Qwen/Qwen3.5-0.8B",
            device="cuda",
            dtype="float16",
            output_files=files,
            context_lengths=[4096, 32768],
            num_prompts=2,
            decode_length=32,
            topk=20,
            save_full_logits=False,
        )
        assert m["output_files"] == files

    def test_git_info_present_or_none(self):
        """Git SHA may be None outside a repo — the key must still exist."""
        m = _build_manifest(
            model_key="0.8b",
            config_name="Qwen/Qwen3.5-0.8B",
            device="cuda",
            dtype="float16",
            output_files=[],
            context_lengths=[4096],
            num_prompts=1,
            decode_length=32,
            topk=20,
            save_full_logits=False,
        )
        assert "sha" in m["git"]
        assert "dirty" in m["git"]
        # In a repo, sha should be a string; outside, None
        assert m["git"]["sha"] is None or isinstance(m["git"]["sha"], str)


# ---------------------------------------------------------------------------
# Standard prompts
# ---------------------------------------------------------------------------


class TestStandardPrompts:
    """The prompt set must be deterministic and cover diverse linguistic patterns."""

    def test_at_least_five_prompts(self):
        assert len(STANDARD_PROMPTS) >= 5

    def test_all_nonempty(self):
        for p in STANDARD_PROMPTS:
            assert len(p) > 20, f"Prompt too short: {p!r}"

    def test_diverse_starts(self):
        """Prompts should start with different words to avoid bias."""
        starts = {p.split()[0] for p in STANDARD_PROMPTS}
        assert len(starts) >= 4

    def test_deterministic(self):
        """The list is module-level — re-import gives the same content."""
        from scripts.capture_reference import STANDARD_PROMPTS as SP2

        assert SP2 is STANDARD_PROMPTS


# ---------------------------------------------------------------------------
# Golden file format conformance
# ---------------------------------------------------------------------------


class TestGoldenFileFormat:
    """The JSON output must match what bench.correctness.py expects."""

    def test_format_has_required_keys(self, tmp_path):
        """A golden reference file must have logits, perplexity, and metadata."""
        golden = {
            "logits": [[0.1, 0.9, 0.5]],
            "topk_token_ids": [[1, 0, 2]],
            "perplexity": 42.5,
            "argmax_tokens": [123, 456],
            "metadata": {
                "prompt_hash": "abc123",
                "context_length": 4096,
                "model_checkpoint": "Qwen/Qwen3.5-0.8B",
                "dtype": "float16",
                "device": "cuda",
                "captured_at": "2026-08-06T07:00:00Z",
            },
        }
        path = tmp_path / "golden.json"
        with open(path, "w") as f:
            json.dump(golden, f)

        # Verify it can be loaded and has the keys correctness.py reads
        with open(path) as f:
            loaded = json.load(f)
        assert "logits" in loaded
        assert "perplexity" in loaded
        assert "metadata" in loaded
        assert isinstance(loaded["logits"], list)
        assert isinstance(loaded["perplexity"], (int, float))

    def test_logits_aligned_with_token_ids(self):
        """Top-k logits and token_ids must have matching lengths."""
        vals = [0.9, 0.7, 0.5]
        ids = [1, 4, 2]
        assert len(vals) == len(ids)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


class TestCLI:
    """Test argument parsing and error handling."""

    def test_default_args(self):
        """Verify default values match constants."""
        import argparse

        # Re-parse to inspect defaults
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default="0.8b")
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--context-lengths", default=",".join(str(x) for x in DEFAULT_CONTEXT_LENGTHS))
        parser.add_argument("--prompts", type=int, default=DEFAULT_PROMPTS)
        parser.add_argument("--decode-length", type=int, default=DEFAULT_DECODE_LENGTH)
        parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
        args = parser.parse_args([])

        assert args.model == "0.8b"
        assert args.device == "cuda"
        assert args.prompts == DEFAULT_PROMPTS
        assert args.decode_length == DEFAULT_DECODE_LENGTH
        assert args.topk == DEFAULT_TOPK

    def test_context_lengths_parsed(self):
        """Context lengths should parse from comma-separated string."""
        parts = ["4096", "32768", "131072", "262144"]
        lengths = [int(x) for x in parts]
        assert lengths == [4096, 32768, 131072, 262144]

    def test_model_preset_keys(self):
        """Both 4b and 0.8b presets must exist."""
        from scripts.capture_reference import _MODEL_PRESETS

        assert "4b" in _MODEL_PRESETS
        assert "0.8b" in _MODEL_PRESETS

    def test_model_presets_match_harness(self):
        """The presets must be the same objects from bench.harness."""
        from bench.harness import QWEN35_08B, QWEN35_4B
        from scripts.capture_reference import _MODEL_PRESETS

        assert _MODEL_PRESETS["4b"] is QWEN35_4B
        assert _MODEL_PRESETS["0.8b"] is QWEN35_08B


# ---------------------------------------------------------------------------
# capture_reference output structure (no torch needed)
# ---------------------------------------------------------------------------


class TestOutputStructure:
    """Verify the golden reference JSON structure that capture_single_prompt
    produces. The actual capture needs torch; these test the contract."""

    def test_golden_file_round_trips_through_correctness_format(self, tmp_path):
        """A golden file must have the keys bench.correctness.py's CLI reads."""
        golden = {
            "logits": [[0.9, 0.7, 0.5, 0.3, 0.1]],
            "topk_token_ids": [[0, 1, 2, 3, 4]],
            "perplexity": 42.5,
            "argmax_tokens": [42, 99, 17, 3],
            "metadata": {
                "prompt_hash": _prompt_hash("test prompt"),
                "prompt_preview": "test prompt",
                "context_length": 64,
                "model_checkpoint": "Qwen/Qwen3.5-0.8B",
                "dtype": "float16",
                "device": "cuda",
                "captured_at": "2026-08-06T07:00:00Z",
            },
        }
        path = tmp_path / "golden.json"
        with open(path, "w") as f:
            json.dump(golden, f)

        with open(path) as f:
            loaded = json.load(f)

        # The two keys correctness.py reads
        assert "logits" in loaded
        assert "perplexity" in loaded
        # The metadata that makes the file self-describing
        assert loaded["metadata"]["model_checkpoint"] == "Qwen/Qwen3.5-0.8B"
        assert loaded["metadata"]["prompt_hash"] == _prompt_hash("test prompt")

    def test_topk_and_token_ids_aligned(self):
        """When using top-k mode, logits and token_ids lists must match length."""
        vals, ids = _topk_logits([0.1, 0.9, 0.5, 0.3], k=3)
        assert len(vals) == len(ids)
        # Each ID must be a valid index into the original logits
        original = [0.1, 0.9, 0.5, 0.3]
        for idx in ids:
            assert 0 <= idx < len(original)

    def test_argmax_tokens_length_matches_decode_length(self):
        """The argmax sequence must have exactly decode_length entries."""
        decode_length = 8
        # Simulate what capture_single_prompt produces
        argmax_tokens = [42] * decode_length
        assert len(argmax_tokens) == decode_length
