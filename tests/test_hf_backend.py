"""Tests for bench/hf_backend.py.

Covers:
- Module import safety (imports without torch/transformers)
- Class hierarchy (HFTorchBackend extends Backend)
- Instantiation guard (ImportError without torch)
- Memory computation (analytic formulas via bench/memory.py)
- State management (reset, seq_len tracking)
- Backend interface conformance (all ABC methods present)
- Mock-based tests for torch-dependent paths
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from bench.harness import QWEN35_4B, Backend
from bench.hf_backend import HFTorchBackend

# ---------------------------------------------------------------------------
# Module-level tests
# ---------------------------------------------------------------------------


class TestModuleImports:
    """Verify the module imports cleanly in all environments."""

    def test_module_importable(self):
        """Module can always be imported, even without torch."""
        import bench.hf_backend as mod

        assert hasattr(mod, "HFTorchBackend")

    def test_torch_available_flag(self):
        """_TORCH_AVAILABLE is a boolean."""
        import bench.hf_backend as mod

        assert isinstance(mod._TORCH_AVAILABLE, bool)

    def test_all_exports(self):
        """__all__ exports HFTorchBackend."""
        import bench.hf_backend as mod

        assert "HFTorchBackend" in mod.__all__

    def test_exports_only_backend(self):
        """No extra names leaked via __all__."""
        import bench.hf_backend as mod

        assert mod.__all__ == ["HFTorchBackend"]


# ---------------------------------------------------------------------------
# Class structure
# ---------------------------------------------------------------------------


class TestClassHierarchy:
    """Verify class hierarchy and ABC conformance."""

    def test_inherits_backend(self):
        """HFTorchBackend is a subclass of Backend."""
        assert issubclass(HFTorchBackend, Backend)

    def test_all_abc_methods_present(self):
        """All abstract Backend methods are implemented."""
        required = {"load", "tokenize", "prefill", "sample", "decode_step", "memory_bytes", "reset"}
        for method_name in required:
            assert hasattr(HFTorchBackend, method_name), f"Missing method: {method_name}"

    def test_is_abstract_or_concrete(self):
        """HFTorchBackend should be concrete (no remaining abstract methods)."""
        # If it had unimplemented abstract methods, it couldn't be instantiated
        # at all. We test instantiation separately.
        assert not getattr(HFTorchBackend, "__abstractmethods__", set()), (
            "HFTorchBackend has unimplemented abstract methods"
        )


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    """Test constructor behavior."""

    def test_raises_without_torch(self):
        """Instantiation raises ImportError when torch is not available."""
        import bench.hf_backend as mod

        if mod._TORCH_AVAILABLE:
            pytest.skip("torch is installed — cannot test ImportError path")

        with pytest.raises(ImportError, match="torch"):
            HFTorchBackend(QWEN35_4B)

    def test_raises_with_torch_unavailable_mocked(self):
        """Even if the module was loaded with torch, mock-unavailable raises."""
        with (
            patch("bench.hf_backend._TORCH_AVAILABLE", False),
            pytest.raises(ImportError, match="torch"),
        ):
            HFTorchBackend(QWEN35_4B)

    def test_import_error_mentions_install_command(self):
        """Error message should tell the user how to install deps."""
        with (
            patch("bench.hf_backend._TORCH_AVAILABLE", False),
            pytest.raises(ImportError, match="pip install"),
        ):
            HFTorchBackend(QWEN35_4B)

    def test_quantization_mentioned_in_error(self):
        """If quantization requested, error should hint at bitsandbytes."""
        with (
            patch("bench.hf_backend._TORCH_AVAILABLE", False),
            pytest.raises(ImportError, match="bitsandbytes"),
        ):
            HFTorchBackend(QWEN35_4B, quantization="int4")


def make_backend_with_mock(config=QWEN35_4B, dtype="float16", quantization=None):
    """Create an HFTorchBackend with mocked torch/transformers.

    Bypasses the torch-availability check by patching _TORCH_AVAILABLE to True
    and providing mock objects for torch and transformers.
    """
    with patch("bench.hf_backend._TORCH_AVAILABLE", True):
        backend = HFTorchBackend(
            config,
            dtype=dtype,
            quantization=quantization,
        )
    return backend


# ---------------------------------------------------------------------------
# Memory computation (uses analytic formulas, no torch needed)
# ---------------------------------------------------------------------------


class TestMemoryBytes:
    """Test memory_bytes returns correct structure and values."""

    def test_returns_dict_with_three_keys(self):
        """memory_bytes returns weights, kv_cache, recurrent_state."""
        backend = make_backend_with_mock()
        mem = backend.memory_bytes()
        assert isinstance(mem, dict)
        assert set(mem.keys()) == {"weights", "kv_cache", "recurrent_state"}

    def test_all_values_non_negative_ints(self):
        """All memory values are non-negative integers (kv_cache=0 at seq_len=0)."""
        backend = make_backend_with_mock()
        mem = backend.memory_bytes()
        for key, val in mem.items():
            assert isinstance(val, int), f"{key} is not int: {type(val)}"
            assert val >= 0, f"{key} is negative: {val}"
        # Weights and recurrent_state are always positive
        assert mem["weights"] > 0
        assert mem["recurrent_state"] > 0

    def test_weights_match_config(self):
        """Weights bytes match config-derived formula."""
        from bench.memory import weights_bytes

        backend = make_backend_with_mock()
        mem = backend.memory_bytes()
        assert mem["weights"] == weights_bytes(QWEN35_4B)

    def test_recurrent_state_matches_config(self):
        """Recurrent state bytes match config-derived formula."""
        from bench.memory import recurrent_state_bytes

        backend = make_backend_with_mock()
        mem = backend.memory_bytes()
        assert mem["recurrent_state"] == recurrent_state_bytes(QWEN35_4B)

    def test_kv_cache_grows_with_seq_len(self):
        """KV cache should increase as seq_len grows."""
        backend = make_backend_with_mock()
        mem_0 = backend.memory_bytes()
        backend._seq_len = 100
        mem_100 = backend.memory_bytes()
        assert mem_100["kv_cache"] > mem_0["kv_cache"]
        # Weights and state should not change
        assert mem_100["weights"] == mem_0["weights"]
        assert mem_100["recurrent_state"] == mem_0["recurrent_state"]


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


class TestReset:
    """Test reset() clears state."""

    def test_reset_clears_kv_cache(self):
        """reset sets _past_key_values to None."""
        backend = make_backend_with_mock()
        backend._past_key_values = "some_cache_object"
        backend.reset()
        assert backend._past_key_values is None

    def test_reset_clears_seq_len(self):
        """reset sets _seq_len to 0."""
        backend = make_backend_with_mock()
        backend._seq_len = 500
        backend.reset()
        assert backend._seq_len == 0

    def test_reset_idempotent(self):
        """Calling reset twice is safe."""
        backend = make_backend_with_mock()
        backend.reset()
        backend.reset()
        assert backend._past_key_values is None
        assert backend._seq_len == 0


# ---------------------------------------------------------------------------
# Mocked torch-dependent paths
# ---------------------------------------------------------------------------


class TestLoadWithMock:
    """Test load() with mocked torch and transformers."""

    @pytest.fixture(autouse=True)
    def mock_memory_check(self):
        """Bypass the OOM pre-check so tests don't fail on low-RAM CI runners."""
        with patch.object(HFTorchBackend, "_check_memory", return_value=None):
            yield

    def test_load_calls_from_pretrained(self):
        """load() calls AutoModelForCausalLM.from_pretrained."""
        backend = make_backend_with_mock()

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("bench.hf_backend.torch") as mock_torch:
            mock_torch.float16 = "float16"
            mock_torch.float32 = "float32"
            mock_torch.bfloat16 = "bfloat16"
            with patch("bench.hf_backend.AutoModelForCausalLM") as mock_alm:
                mock_alm.from_pretrained.return_value = mock_model
                with patch("bench.hf_backend.AutoTokenizer") as mock_tok:
                    mock_tok.from_pretrained.return_value = mock_tokenizer

                    backend.load()

        mock_alm.from_pretrained.assert_called_once()
        mock_tok.from_pretrained.assert_called_once()
        assert backend._model is mock_model
        assert backend._tokenizer is mock_tokenizer

    def test_load_sets_model_to_eval(self):
        """load() calls model.eval()."""
        backend = make_backend_with_mock()

        mock_model = MagicMock()

        with patch("bench.hf_backend.torch") as mock_torch:
            mock_torch.float16 = "float16"
            with patch("bench.hf_backend.AutoModelForCausalLM") as mock_alm:
                mock_alm.from_pretrained.return_value = mock_model
                with patch("bench.hf_backend.AutoTokenizer") as mock_tok:
                    mock_tok.from_pretrained.return_value = MagicMock()

                    backend.load()

        mock_model.eval.assert_called_once()

    def test_load_dtype_mapping(self):
        """load() maps dtype string to torch dtype correctly."""
        backend = make_backend_with_mock(dtype="float32")

        mock_model = MagicMock()

        with patch("bench.hf_backend.torch") as mock_torch:
            mock_torch.float16 = "float16"
            mock_torch.float32 = "float32"
            mock_torch.bfloat16 = "bfloat16"
            with patch("bench.hf_backend.AutoModelForCausalLM") as mock_alm:
                mock_alm.from_pretrained.return_value = mock_model
                with patch("bench.hf_backend.AutoTokenizer") as mock_tok:
                    mock_tok.from_pretrained.return_value = MagicMock()

                    backend.load()

            call_kwargs = mock_alm.from_pretrained.call_args[1]
            assert call_kwargs["torch_dtype"] == "float32"


class TestTokenizeWithMock:
    """Test tokenize() with mocked tokenizer."""

    def test_tokenize_calls_encode(self):
        """tokenize() delegates to tokenizer.encode."""
        backend = make_backend_with_mock()
        backend._tokenizer = MagicMock()
        backend._tokenizer.encode.return_value = [1, 2, 3]

        result = backend.tokenize("hello world")

        backend._tokenizer.encode.assert_called_once_with("hello world")
        assert result == [1, 2, 3]


class TestPrefillWithMock:
    """Test prefill() with mocked model."""

    def test_prefill_returns_logits(self):
        """prefill() returns model logits."""
        backend = make_backend_with_mock()

        mock_logits = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_outputs.past_key_values = "new_cache"

        backend._model = MagicMock()
        backend._model.device = "cpu"
        backend._model.return_value = mock_outputs

        with patch("bench.hf_backend.torch") as mock_torch:
            mock_torch.tensor.return_value = MagicMock()
            mock_torch.inference_mode.return_value.__enter__ = MagicMock()
            mock_torch.inference_mode.return_value.__exit__ = MagicMock()

            result = backend.prefill([1, 2, 3])

        assert result is mock_logits
        assert backend._past_key_values == "new_cache"
        assert backend._seq_len == 3


class TestSampleWithMock:
    """Test sample() returns argmax token."""

    def test_sample_returns_int(self):
        """sample() returns an integer token ID."""
        backend = make_backend_with_mock()

        with patch("bench.hf_backend.torch") as mock_torch:
            mock_torch.argmax.return_value = MagicMock()
            mock_torch.argmax.return_value.item.return_value = 42

            result = backend.sample(MagicMock())

        assert result == 42


class TestDecodeStepWithMock:
    """Test decode_step() with mocked model."""

    def test_decode_step_increments_seq_len(self):
        """decode_step() increments _seq_len."""
        backend = make_backend_with_mock()
        backend._seq_len = 10

        mock_outputs = MagicMock()
        mock_outputs.past_key_values = "new_cache"
        backend._model = MagicMock()
        backend._model.device = "cpu"
        backend._model.return_value = mock_outputs

        with patch("bench.hf_backend.torch") as mock_torch:
            mock_torch.tensor.return_value = MagicMock()
            mock_torch.inference_mode.return_value.__enter__ = MagicMock()
            mock_torch.inference_mode.return_value.__exit__ = MagicMock()
            mock_torch.argmax.return_value = MagicMock()
            mock_torch.argmax.return_value.item.return_value = 99

            result = backend.decode_step(token_id=5)

        assert backend._seq_len == 11
        assert backend._past_key_values == "new_cache"
        assert result == 99


# ---------------------------------------------------------------------------
# _verify_memory edge cases
# ---------------------------------------------------------------------------


class TestVerifyMemory:
    """Test _verify_memory() handles edge cases gracefully."""

    def test_no_crash_without_cache(self):
        """_verify_memory does nothing when _past_key_values is None."""
        backend = make_backend_with_mock()
        backend._past_key_values = None
        # Should not raise
        backend._verify_memory()

    def test_no_crash_with_unsupported_cache(self):
        """_verify_memory handles cache objects without recurrent_states."""
        backend = make_backend_with_mock()
        backend._past_key_values = MagicMock()
        # Remove recurrent_states to trigger AttributeError path
        del backend._past_key_values.recurrent_states
        # Should not raise
        backend._verify_memory()

    def test_no_crash_with_empty_states(self):
        """_verify_memory handles empty recurrent_states."""
        backend = make_backend_with_mock()
        backend._past_key_values = MagicMock()
        backend._past_key_values.recurrent_states = []
        # Should not raise
        backend._verify_memory()


# ---------------------------------------------------------------------------
# Integration: backend conforms to harness Backend ABC
# ---------------------------------------------------------------------------


class TestBackendConformance:
    """Verify HFTorchBackend satisfies the Backend contract."""

    def test_can_be_used_as_backend_type(self):
        """An HFTorchBackend instance is recognized as a Backend."""
        backend = make_backend_with_mock()
        assert isinstance(backend, Backend)

    def test_memory_bytes_consistent_across_calls(self):
        """Repeated calls to memory_bytes return consistent values."""
        backend = make_backend_with_mock()
        m1 = backend.memory_bytes()
        m2 = backend.memory_bytes()
        assert m1 == m2

    def test_config_preserved(self):
        """Config is stored and accessible."""
        backend = make_backend_with_mock()
        assert backend.config is QWEN35_4B

    def test_seq_len_starts_at_zero(self):
        """New backend starts with seq_len = 0."""
        backend = make_backend_with_mock()
        assert backend._seq_len == 0
