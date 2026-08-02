"""Tests for bench/prompts.py — long-context prompt corpus generator.

Bead ``ob-del``. Verifies determinism, token-length estimation, NIAH/multi-key
structure, and CLI behavior.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BENCH_DIR = _REPO_ROOT / "bench"
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))


class TestNIAH:
    """Needle-in-haystack generation."""

    def test_deterministic(self):
        """Same seed → same prompt."""
        import prompts

        a = prompts.generate_niah(4096, 50, seed=42)
        b = prompts.generate_niah(4096, 50, seed=42)
        assert a.prompt == b.prompt
        assert a.expected_answer == b.expected_answer

    def test_different_seed_different_output(self):
        import prompts

        a = prompts.generate_niah(4096, 50, seed=42)
        b = prompts.generate_niah(4096, 50, seed=99)
        assert a.prompt != b.prompt
        assert a.needle != b.needle

    def test_needle_present_in_prompt(self):
        """The needle text must appear in the prompt."""
        import prompts

        result = prompts.generate_niah(4096, 50, seed=42)
        assert result.needle in result.prompt

    def test_question_present(self):
        """The prompt must end with a retrieval question."""
        import prompts

        result = prompts.generate_niah(4096, 50, seed=42)
        assert "Question" in result.prompt
        assert "passcode" in result.prompt.lower()

    def test_depth_zero(self):
        """Depth 0% means needle at the very start."""
        import prompts

        result = prompts.generate_niah(4096, 0, seed=42)
        # The needle should appear early in the prompt
        assert result.needle in result.prompt
        needle_pos = result.prompt.index(result.needle)
        assert needle_pos < len(result.prompt) * 0.1

    def test_depth_hundred(self):
        """Depth 100% means needle at the very end (before the question)."""
        import prompts

        result = prompts.generate_niah(4096, 100, seed=42)
        assert result.needle in result.prompt
        needle_pos = result.prompt.index(result.needle)
        question_pos = result.prompt.index("Question")
        # Needle should be just before the question
        assert needle_pos < question_pos

    def test_expected_answer_in_needle(self):
        """The expected answer is derived from the needle."""
        import prompts

        result = prompts.generate_niah(4096, 50, seed=42)
        assert result.expected_answer in result.needle

    def test_est_tokens_reasonable(self):
        """Estimated tokens should be in the right ballpark for the target."""
        import prompts

        result = prompts.generate_niah(4096, 50, seed=42)
        # Allow 20% tolerance — the word heuristic is approximate
        assert 3000 < result.est_tokens < 5000


class TestMultiKey:
    """RULER-style multi-key generation."""

    def test_deterministic(self):
        import prompts

        a = prompts.generate_multikey(4096, 10, seed=42)
        b = prompts.generate_multikey(4096, 10, seed=42)
        assert a.prompt == b.prompt
        assert a.expected_answer == b.expected_answer

    def test_correct_num_keys(self):
        """Should embed exactly the specified number of key-value pairs."""
        import prompts

        result = prompts.generate_multikey(4096, 10, seed=42)
        # Count [item_XXXX] patterns in the prompt
        count = result.prompt.count("[item_")
        assert count == 10

    def test_query_key_in_prompt(self):
        """The query key must appear in the prompt."""
        import prompts

        result = prompts.generate_multikey(4096, 5, seed=42)
        assert result.query_key in result.prompt

    def test_expected_answer_in_prompt(self):
        """The expected answer value must appear in the prompt."""
        import prompts

        result = prompts.generate_multikey(4096, 5, seed=42)
        assert result.expected_answer in result.prompt

    def test_question_present(self):
        import prompts

        result = prompts.generate_multikey(4096, 5, seed=42)
        assert "Question" in result.prompt
        assert result.query_key in result.prompt


class TestCorpusGeneration:
    """End-to-end corpus generation to disk."""

    def test_generate_to_tmp(self, tmp_path):
        import prompts

        written = prompts.generate_corpus(
            output_dir=tmp_path,
            sweep_points=[4096, 32768],
            depths=[0, 50, 100],
            multikey_counts=[1, 10],
        )
        # 2 sweep points × (3 depths + 2 multikey counts) = 10 files
        assert len(written) == 10
        for p in written:
            assert p.exists()
            data = json.loads(p.read_text())
            assert "prompt" in data
            assert "expected_answer" in data
            assert "seed" in data

    def test_filenames_use_canonical_labels(self, tmp_path):
        """262144 → 262K, not 256K."""
        import prompts

        written = prompts.generate_corpus(
            output_dir=tmp_path,
            sweep_points=[262144],
            depths=[50],
            multikey_counts=[],
        )
        names = [p.name for p in written]
        assert any("262K" in n for n in names)
        assert not any("256K" in n for n in names)

    def test_json_structure_niah(self, tmp_path):
        import prompts

        prompts.generate_corpus(
            output_dir=tmp_path,
            sweep_points=[4096],
            depths=[50],
            multikey_counts=[],
        )
        niah_files = list(tmp_path.glob("niah_*.json"))
        assert len(niah_files) == 1
        data = json.loads(niah_files[0].read_text())
        assert data["task"] == "niah"
        assert data["target_tokens"] == 4096
        assert data["needle_depth_pct"] == 50

    def test_json_structure_multikey(self, tmp_path):
        import prompts

        prompts.generate_corpus(
            output_dir=tmp_path,
            sweep_points=[4096],
            depths=[],
            multikey_counts=[5],
        )
        mk_files = list(tmp_path.glob("multikey_*.json"))
        assert len(mk_files) == 1
        data = json.loads(mk_files[0].read_text())
        assert data["task"] == "multikey"
        assert data["num_keys"] == 5


class TestCLI:
    def test_no_args_prints_help(self, capsys):
        import prompts

        rc = prompts.main([])
        assert rc == 1
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "Generate" in captured.out

    def test_niah_only(self, tmp_path):
        import prompts

        rc = prompts.main(["--niah", "--depths", "0,50", "--output", str(tmp_path)])
        assert rc == 0
        niah_files = list(tmp_path.glob("niah_*.json"))
        multikey_files = list(tmp_path.glob("multikey_*.json"))
        assert len(niah_files) > 0
        assert len(multikey_files) == 0

    def test_multikey_only(self, tmp_path):
        import prompts

        rc = prompts.main(["--multikey", "--keys", "1,3", "--output", str(tmp_path)])
        assert rc == 0
        niah_files = list(tmp_path.glob("niah_*.json"))
        multikey_files = list(tmp_path.glob("multikey_*.json"))
        assert len(niah_files) == 0
        assert len(multikey_files) > 0
