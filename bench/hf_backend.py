# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace transformers Backend for the benchmark harness (ob-xh3.3).

Implements the ``Backend`` ABC (``bench/harness.py``) for a real Qwen3.5 model
via HuggingFace ``transformers``. This is the backend ``ob-aqv`` (x86/CUDA
reference) will use to produce the correctness oracle.

**Conditional imports:** this module imports cleanly without torch/transformers
installed — the ``HFTorchBackend`` class is always defined, but instantiation
raises ``ImportError`` with a clear message if the dependencies are missing.
This means CI can import and lint-check this file on any platform, while actual
inference requires an x86/CUDA (or Apple Silicon / Graviton) host.

See ``docs/BACKEND_GUIDE.md`` (ob-xh3.1) for the method-by-method mapping.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure repo root is importable
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.harness import Backend, ModelConfig  # noqa: E402
from bench.memory import (  # noqa: E402
    cross_check,
    kv_cache_bytes,
    recurrent_state_bytes,
    weights_bytes,
)

# Conditional imports — torch and transformers are NOT required to import this module.
try:
    import torch  # type: ignore[import-untyped]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


class HFTorchBackend(Backend):
    """Backend wrapping a HuggingFace transformers model.

    Implements the ``Backend`` ABC for Qwen3.5 (or any causal LM). The harness
    handles all timing (METRICS.md sections 1–5); this class provides only the
    compute operations the harness times around.

    Usage on an x86/CUDA host::

        from bench.harness import QWEN35_08B, SweepConfig, run_sweep
        from bench.hf_backend import HFTorchBackend

        backend = HFTorchBackend(QWEN35_08B)
        config = SweepConfig(
            context_lengths=[4096, 32768],
            device="x86_reference",
            model_checkpoint="Qwen/Qwen3.5-0.8B@x86-cuda",
        )
        rows = run_sweep(backend, config)
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        dtype: str = "float16",
        device_map: str = "auto",
        quantization: str | None = None,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "HFTorchBackend requires torch and transformers. "
                "Install with: pip install torch transformers"
                + (" bitsandbytes" if quantization else "")
            )
        self.config = config
        self._dtype = dtype
        self._device_map = device_map
        self._quantization = quantization
        self._model = None
        self._tokenizer = None
        self._past_key_values = None
        self._seq_len = 0

    # ------------------------------------------------------------------
    # OOM pre-check (ob-3lq): refuse to load if RAM is insufficient,
    # rather than letting the kernel OOM killer silently kill the
    # supervising tmux/goose session.
    # ------------------------------------------------------------------

    @staticmethod
    def _available_memory_bytes() -> int:
        """Read MemAvailable from /proc/meminfo (Linux only)."""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024  # KiB → bytes
        except (OSError, ValueError, IndexError):
            pass
        return 0  # unknown — skip the check

    def _check_memory(self) -> None:
        """Pre-flight: raise MemoryError if available RAM is insufficient.

        Prevents the kernel OOM killer from killing the entire tmux/goose
        supervising session (bead ob-3lq). The OOM killer targets the cgroup
        with the most anon-rss, which is the tmux-spawn process, not just the
        python benchmark subprocess.
        """
        avail = self._available_memory_bytes()
        if avail == 0:
            return  # can't check (non-Linux or /proc unreadable) — proceed

        label = self.config.name
        if "0.8B" in label or "0.8b" in label:
            weight_est = 3_010_000_000  # 752M params × 4 bytes (fp32 worst case)
        elif "4B" in label or "4b" in label:
            weight_est = 8_040_000_000  # ~2B params × 4 bytes
        else:
            weight_est = 4_000_000_000  # conservative default

        required = int(weight_est * 1.5)

        if avail < required:
            raise MemoryError(
                f"Insufficient memory to load {label}: "
                f"{avail / 1e9:.1f} GiB available, "
                f"~{required / 1e9:.1f} GiB required (weights + overhead). "
                f"This check prevents the kernel OOM killer from killing "
                f"the supervising session (ob-3lq). "
                f"Options: (1) free memory on this device, "
                f"(2) run on a higher-RAM node, "
                f"(3) use a smaller model."
            )

        print(
            f"  [hf] Memory check: {avail / 1e9:.1f} GiB available "
            f"(need ~{required / 1e9:.1f} GiB) — OK",
            flush=True,
        )

    def load(self) -> None:
        """Load the model and tokenizer. Excluded from all metrics (METRICS.md §1)."""
        self._check_memory()

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self._dtype, torch.float16)

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": self._device_map,
        }

        # INT4 quantization per QUANTIZATION_POLICY.md
        if self._quantization == "int4":
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                import warnings

                warnings.warn(
                    "bitsandbytes not installed — loading in fp16 instead of int4. "
                    "Install with: pip install bitsandbytes",
                    stacklevel=2,
                )

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.name)
        self._model = AutoModelForCausalLM.from_pretrained(self.config.name, **model_kwargs)
        self._model.eval()

        # Cross-check memory formulas against live tensors (METRICS.md §5.4)
        self._verify_memory()

    def _verify_memory(self) -> None:
        """Verify analytic memory formulas match the loaded model's actual tensors.

        Catches bugs where the config doesn't match reality (METRICS.md §5.4).
        Logs discrepancies as warnings rather than crashing — the harness can
        still run, but the numbers need the caveat.
        """
        if self._past_key_values is None:
            # State is only available after the first forward pass
            return

        try:
            state = self._past_key_values.recurrent_states
            if state and len(state) > 0:
                shape = tuple(state[0].shape)
                dtype_bytes = state[0].element_size()
                discrepancies = cross_check(
                    self.config,
                    introspected_state_shape=shape,
                    introspected_state_dtype_bytes=dtype_bytes,
                )
                if discrepancies:
                    import warnings

                    for d in discrepancies:
                        warnings.warn(f"Memory cross-check: {d}", stacklevel=2)
        except (AttributeError, IndexError, TypeError):
            # Cache object doesn't expose recurrent_states (different model arch)
            pass

    def tokenize(self, text: str) -> list[int]:
        """Tokenize prompt text. The harness records len(input_ids) (METRICS.md §2)."""
        return self._tokenizer.encode(text)

    def prefill(self, input_ids: list[int]) -> Any:
        """Full forward pass over the prompt. Returns logits.

        The harness times the interval [start, return] as prefill duration
        (METRICS.md §2).
        """
        input_tensor = torch.tensor([input_ids], device=self._model.device)
        self._seq_len = len(input_ids)
        with torch.inference_mode():
            outputs = self._model(
                input_tensor,
                use_cache=True,
                output_hidden_states=False,
            )
        self._past_key_values = outputs.past_key_values
        self._verify_memory()
        return outputs.logits

    def sample(self, logits: Any) -> int:
        """Argmax the last position. Token 1 belongs to prefill (METRICS.md §1)."""
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        return next_token.item()

    def decode_step(self, token_id: int) -> int:
        """Single-token decode. EOS does not stop generation (METRICS.md §4)."""
        self._seq_len += 1
        input_tensor = torch.tensor([[token_id]], device=self._model.device)
        with torch.inference_mode():
            outputs = self._model(
                input_tensor,
                past_key_values=self._past_key_values,
                use_cache=True,
            )
        self._past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        return next_token.item()

    def memory_bytes(self) -> dict[str, int]:
        """Three-way memory attribution from analytic formulas (METRICS.md §5).

        Uses bench/memory.py formulas verified against ADR 0003. The cross-check
        in ``_verify_memory`` catches any mismatch with live tensors.
        """
        return {
            "weights": weights_bytes(self.config),
            "kv_cache": kv_cache_bytes(self.config, self._seq_len),
            "recurrent_state": recurrent_state_bytes(self.config),
        }

    def reset(self) -> None:
        """Clear KV cache and recurrent state. Weights stay loaded."""
        self._past_key_values = None
        self._seq_len = 0


__all__ = ["HFTorchBackend"]
