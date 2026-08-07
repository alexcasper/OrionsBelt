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
        import scripts.generate_reference as gen

        monkeypatch.setattr(gen, "_TORCH_AVAILABLE", False)
        rc = main([])
        assert rc == 1
        captured = capsys.readouterr()
        assert "torch" in captured.err.lower() or "not available" in captured.err.lower()

    def test_missing_model_path(self, monkeypatch, capsys):
        """When torch is not available, exits before checking model path."""
        import scripts.generate_reference as gen

        monkeypatch.setattr(gen, "_TORCH_AVAILABLE", False)
        rc = main(["--model-path", "/nonexistent/path"])
        assert rc == 1

    def test_smoke_flag_still_fails_without_torch(self, monkeypatch, capsys):
        """Smoke flag doesn't help without torch."""
        import scripts.generate_reference as gen

        monkeypatch.setattr(gen, "_TORCH_AVAILABLE", False)
        rc = main(["--smoke"])
        assert rc == 1


# ---------------------------------------------------------------------------
# _LONG_FILLER constant
# ---------------------------------------------------------------------------
class TestLongFiller:
    def test_importable_and_nonempty(self):
        from scripts.generate_reference import _LONG_FILLER

        assert isinstance(_LONG_FILLER, str)
        assert len(_LONG_FILLER) > 500

    def test_mentions_rk3588(self):
        from scripts.generate_reference import _LONG_FILLER

        assert "RK3588" in _LONG_FILLER

    def test_mentions_gdn_concepts(self):
        from scripts.generate_reference import _LONG_FILLER

        assert "gated delta" in _LONG_FILLER.lower()

    def test_contains_python_code(self):
        """Filler includes code snippets for varied logit distributions."""
        from scripts.generate_reference import _LONG_FILLER

        assert "def " in _LONG_FILLER

    def test_mentions_memory_bandwidth(self):
        from scripts.generate_reference import _LONG_FILLER

        assert "bandwidth" in _LONG_FILLER.lower()


# ---------------------------------------------------------------------------
# collect_provenance() — mocked torch/transformers
# ---------------------------------------------------------------------------
class TestCollectProvenance:
    @pytest.fixture(autouse=True)
    def _mock_deps(self, monkeypatch):
        """Inject mock torch and transformers into the module namespace."""
        from unittest.mock import MagicMock

        import scripts.generate_reference as gen

        mock_torch = MagicMock()
        mock_torch.__version__ = "2.4.0"
        monkeypatch.setattr(gen, "torch", mock_torch, raising=False)

        mock_tf = MagicMock()
        mock_tf.__version__ = "4.44.0"
        monkeypatch.setitem(sys.modules, "transformers", mock_tf)

    def test_returns_dict_with_keys(self):
        """collect_provenance returns a dict with all expected keys."""
        import scripts.generate_reference as gen

        prov = gen.collect_provenance("/fake/model")
        assert isinstance(prov, dict)
        for key in [
            "timestamp",
            "git_sha",
            "git_dirty",
            "hostname",
            "platform",
            "python_version",
            "torch_version",
            "transformers_version",
            "numpy_version",
            "model_path",
            "model_repo",
            "device",
            "dtype",
            "cpu_governor",
            "thermals_pre",
        ]:
            assert key in prov, f"Missing key: {key}"

    def test_torch_version_captured(self, monkeypatch):
        """Override the fixture's mock to test a different version."""
        from unittest.mock import MagicMock

        import scripts.generate_reference as gen

        mock_torch = MagicMock()
        mock_torch.__version__ = "99.0.0"
        monkeypatch.setattr(gen, "torch", mock_torch, raising=False)

        prov = gen.collect_provenance("/fake/model")
        assert prov["torch_version"] == "99.0.0"

    def test_model_path_echoed(self):
        import scripts.generate_reference as gen

        prov = gen.collect_provenance("/some/path/model")
        assert prov["model_path"] == "/some/path/model"

    def test_fixed_fields(self):
        """Device and dtype are always cpu/float32."""
        import scripts.generate_reference as gen

        prov = gen.collect_provenance("/fake")
        assert prov["device"] == "cpu"
        assert prov["dtype"] == "float32"
        assert prov["model_repo"] == "Qwen/Qwen3.5-0.8B"

    def test_timestamp_is_iso_format(self):
        import scripts.generate_reference as gen

        prov = gen.collect_provenance("/fake")
        # ISO format contains 'T' separator
        assert "T" in prov["timestamp"]

    def test_numpy_version_captured(self):
        import scripts.generate_reference as gen

        prov = gen.collect_provenance("/fake")
        # numpy IS installed on this device
        assert prov["numpy_version"] is not None
        assert len(prov["numpy_version"]) > 0

    def test_transformers_version_captured(self):
        import scripts.generate_reference as gen

        prov = gen.collect_provenance("/fake")
        assert prov["transformers_version"] == "4.44.0"


# ---------------------------------------------------------------------------
# Provenance exception paths
# ---------------------------------------------------------------------------
class TestProvenanceExceptionPaths:
    """Cover the except branches in _git_sha, _git_dirty, _governor, _hostname."""

    def test_git_sha_returns_unknown_on_exception(self, monkeypatch):
        """_git_sha returns 'unknown' when subprocess fails."""
        import scripts.generate_reference as gen

        def boom(*a, **kw):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr(gen.subprocess, "check_output", boom)
        assert gen._git_sha() == "unknown"

    def test_git_dirty_returns_true_on_exception(self, monkeypatch):
        """_git_dirty returns True (conservative) when subprocess fails."""
        import scripts.generate_reference as gen

        def boom(*a, **kw):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr(gen.subprocess, "run", boom)
        assert gen._git_dirty() is True

    def test_governor_returns_unknown_on_oserror(self, monkeypatch):
        """_governor returns 'unknown' when /sys is unreadable."""
        import scripts.generate_reference as gen

        real_open = open

        def fake_open(path, *a, **kw):
            if "scaling_governor" in str(path):
                raise OSError("permission denied")
            return real_open(path, *a, **kw)

        monkeypatch.setattr("builtins.open", fake_open)
        assert gen._governor() == "unknown"

    def test_hostname_falls_back_to_platform_node(self, monkeypatch):
        """_hostname returns platform.node() when subprocess fails."""
        import scripts.generate_reference as gen

        def boom(*a, **kw):
            raise FileNotFoundError("hostname not found")

        monkeypatch.setattr(gen.subprocess, "check_output", boom)
        result = gen._hostname()
        assert isinstance(result, str)
        assert len(result) > 0  # platform.node() returns something


# ---------------------------------------------------------------------------
# load_model() — mocked torch/transformers
# ---------------------------------------------------------------------------
class TestLoadModelMocked:
    """Test load_model() with mocked AutoTokenizer/AutoModelForCausalLM."""

    def test_load_model_returns_model_and_tokenizer(self, monkeypatch):
        """load_model returns a (model, tokenizer) tuple."""
        from unittest.mock import MagicMock

        import scripts.generate_reference as gen

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.config.model_type = "qwen"
        mock_param = MagicMock()
        mock_param.numel.return_value = 800_000_000
        mock_model.parameters.return_value = iter([mock_param])

        monkeypatch.setattr(
            gen,
            "AutoTokenizer",
            MagicMock(from_pretrained=MagicMock(return_value=mock_tokenizer)),
            raising=False,
        )
        monkeypatch.setattr(
            gen,
            "AutoModelForCausalLM",
            MagicMock(from_pretrained=MagicMock(return_value=mock_model)),
            raising=False,
        )
        monkeypatch.setattr(gen, "torch", MagicMock(float32="float32"), raising=False)

        model, tokenizer = gen.load_model("fake/path")
        assert model is mock_model
        assert tokenizer is mock_tokenizer
        mock_model.eval.assert_called_once()


# ---------------------------------------------------------------------------
# run_reference_inference() — mocked with FakeTensor
# ---------------------------------------------------------------------------
class _FakeTensor:
    """Minimal tensor mock supporting indexing and numpy conversion."""

    def __init__(self, data):
        import numpy as np

        if isinstance(data, np.ndarray):
            self._data = data
        else:
            self._data = np.array(data)

    def __getitem__(self, key):
        return _FakeTensor(self._data[key])

    def contiguous(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._data

    def view(self, *a, **kw):
        return self

    def size(self, dim):
        return self._data.shape[dim] if dim < len(self._data.shape) else 1

    def numel(self):
        return self._data.size

    def item(self):
        return float(self._data.flat[0])


class _FakeTorch:
    """Minimal torch mock for run_reference_inference."""

    float32 = "float32"
    long = "long"

    @staticmethod
    def tensor(data, dtype=None):
        return _FakeTensor(data)

    class inference_mode:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    @staticmethod
    def argmax(t, dim=None):
        import numpy as np

        return _FakeTensor(np.array([int(np.argmax(t.numpy()))]))

    class nn:
        @staticmethod
        def CrossEntropyLoss(reduction="sum"):
            def loss_fn(logits, labels):
                return _FakeTensor([5.0])

            return loss_fn


class TestRunReferenceInferenceMocked:
    """Test run_reference_inference with FakeTorch."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Set up mocked torch and model/tokenizer for inference."""
        from unittest.mock import MagicMock

        import numpy as np
        import scripts.generate_reference as gen

        monkeypatch.setattr(gen, "torch", _FakeTorch, raising=False)

        # Mock model: returns outputs with fake logits
        def model_call(*args, **kwargs):
            seq_len = 128
            vocab = 100
            mock_out = MagicMock()
            mock_out.logits = _FakeTensor(np.random.randn(1, seq_len, vocab).astype(np.float32))
            mock_out.past_key_values = MagicMock()
            return mock_out

        mock_model = MagicMock(side_effect=model_call)

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode = MagicMock(side_effect=lambda text: list(range(min(len(text), 80))))
        mock_tokenizer.decode = MagicMock(return_value="decoded")

        return {"model": mock_model, "tokenizer": mock_tokenizer}

    def test_returns_list_of_dicts(self, mock_env):
        """run_reference_inference returns a list of result dicts."""
        import scripts.generate_reference as gen

        results = gen.run_reference_inference(
            mock_env["model"],
            mock_env["tokenizer"],
            gen.PROMPTS[:1],
            [128],
            top_k=5,
            decode_steps=2,
        )
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], dict)

    def test_result_has_required_keys(self, mock_env):
        """Each result dict has the expected keys."""
        import scripts.generate_reference as gen

        results = gen.run_reference_inference(
            mock_env["model"],
            mock_env["tokenizer"],
            gen.PROMPTS[:1],
            [128],
            top_k=5,
            decode_steps=2,
        )
        r = results[0]
        for key in [
            "entry_id",
            "prompt_id",
            "context_length",
            "prompt_text",
            "perplexity",
            "avg_nll",
            "forward_ms",
            "argmax_token",
            "argmax_token_text",
            "last_position_logits",
            "topk_window",
            "generated_token_ids",
            "generated_text",
        ]:
            assert key in r, f"Missing key: {key}"

    def test_multiple_prompts_and_lengths(self, mock_env):
        """Handles multiple prompts x context lengths."""
        import scripts.generate_reference as gen

        results = gen.run_reference_inference(
            mock_env["model"],
            mock_env["tokenizer"],
            gen.PROMPTS[:2],
            [128, 512],
            top_k=5,
            decode_steps=2,
        )
        assert len(results) == 4

    def test_generated_token_ids_length_matches_decode_steps(self, mock_env):
        """Generated token list length equals decode_steps."""
        import scripts.generate_reference as gen

        results = gen.run_reference_inference(
            mock_env["model"],
            mock_env["tokenizer"],
            gen.PROMPTS[:1],
            [128],
            decode_steps=4,
        )
        assert len(results[0]["generated_token_ids"]) == 4

    def test_perplexity_is_float(self, mock_env):
        """Perplexity is a float."""
        import scripts.generate_reference as gen

        results = gen.run_reference_inference(
            mock_env["model"],
            mock_env["tokenizer"],
            gen.PROMPTS[:1],
            [128],
            decode_steps=2,
        )
        assert isinstance(results[0]["perplexity"], float)

    def test_topk_window_has_entries(self, mock_env):
        """Top-k window contains position entries."""
        import scripts.generate_reference as gen

        results = gen.run_reference_inference(
            mock_env["model"],
            mock_env["tokenizer"],
            gen.PROMPTS[:1],
            [128],
            top_k=10,
            decode_steps=2,
        )
        window = results[0]["topk_window"]
        assert len(window) > 0
        for entry in window:
            assert "position_from_end" in entry
            assert "indices" in entry
            assert "values" in entry


# ---------------------------------------------------------------------------
# main() body — mocked load_model + run_reference_inference
# ---------------------------------------------------------------------------
class TestMainMocked:
    """Test main() with mocked internals (past torch check)."""

    def test_main_writes_output_json(self, monkeypatch, tmp_path, capsys):
        """main() writes a valid JSON output file and returns 0."""
        from unittest.mock import MagicMock

        import scripts.generate_reference as gen

        # Bypass torch check
        monkeypatch.setattr(gen, "_TORCH_AVAILABLE", True)

        # Create fake model directory so Path.exists() passes
        fake_model = tmp_path / "model"
        fake_model.mkdir()

        # Mock load_model
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        monkeypatch.setattr(gen, "load_model", MagicMock(return_value=(mock_model, mock_tokenizer)))

        # Mock run_reference_inference
        fake_results = [
            {
                "entry_id": "factual_128",
                "prompt_id": "factual",
                "context_length": 128,
                "prompt_text": "test",
                "perplexity": 12.5,
                "avg_nll": 2.5,
                "forward_ms": 100.0,
                "argmax_token": 42,
                "argmax_token_text": "test",
                "last_position_logits": [0.1, 0.2],
                "topk_window": [],
                "generated_token_ids": [1, 2],
                "generated_text": "hi",
            }
        ]
        monkeypatch.setattr(gen, "run_reference_inference", MagicMock(return_value=fake_results))

        # Mock collect_provenance to avoid subprocess issues
        monkeypatch.setattr(
            gen,
            "collect_provenance",
            MagicMock(
                return_value={
                    "git_sha": "abc",
                    "hostname": "test",
                    "torch_version": "2.0.0",
                    "transformers_version": "4.0.0",
                }
            ),
        )

        # Mock torch for provenance
        monkeypatch.setattr(gen, "torch", MagicMock(__version__="2.0.0"), raising=False)
        mock_tf = MagicMock()
        mock_tf.__version__ = "4.0.0"
        monkeypatch.setitem(sys.modules, "transformers", mock_tf)

        output_file = tmp_path / "out.json"
        rc = gen.main(
            [
                "--model-path",
                str(fake_model),
                "--output",
                str(output_file),
                "--smoke",
            ]
        )

        assert rc == 0
        assert output_file.exists()
        import json

        data = json.loads(output_file.read_text())
        assert "provenance" in data
        assert "entries" in data
        assert data["entries"][0]["entry_id"] == "factual_128"
        assert "summary" in data
        assert data["summary"]["num_entries"] == 1

    def test_main_missing_model_path_returns_1(self, monkeypatch, capsys):
        """main() returns 1 when model path doesn't exist (past torch check)."""
        import scripts.generate_reference as gen

        monkeypatch.setattr(gen, "_TORCH_AVAILABLE", True)

        rc = gen.main(
            [
                "--model-path",
                "/nonexistent/path/model",
                "--output",
                "/tmp/out.json",
            ]
        )
        assert rc == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower() or "model path" in captured.err.lower()
