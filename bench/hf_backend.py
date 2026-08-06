#!/usr/bin/env python3
"""HuggingFace transformers backend for the GDN benchmark harness.

Bead ``ob-mrd.2``. Implements the ``BenchmarkBackend`` ABC from
``bench/harness.py`` so the harness can time a real model end-to-end:
prefill throughput, TTFT, decode throughput, and three-component memory
attribution (weights / KV cache / recurrent state).

The model is Qwen3.5-0.8B — a **hybrid GDN** architecture with 18
linear-attention layers and 6 full-attention layers (every 4th).
This makes the memory decomposition directly interesting: the KV cache
grows only for the 6 full-attention layers, while the 18 linear layers
maintain a fixed-size recurrent state.

Usage::

    /tmp/model_venv/bin/python3 bench/harness.py \\
        --backend hf --model Qwen3.5-0.8B \\
        --contexts 512,1024 --repeats 3 --decode-tokens 33 \\
        --device rk3588-t4

Requires ``torch`` and ``transformers`` (not in stdlib — install into a
venv on the device).  Falls back to ``float32`` on ARM CPUs that lack
OneDNN bf16 support (RK3588 Cortex-A76).
"""

from __future__ import annotations

import os
import sys

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from harness import BenchmarkBackend, MemoryBreakdown  # noqa: E402

_WEIGHTS_DIR = os.environ.get("ORIONS_WEIGHTS_DIR", "weights")

_MODEL_PATHS = {
    "0.8b": os.path.join(_WEIGHTS_DIR, "Qwen--Qwen3.5-0.8B"),
    "4b": os.path.join(_WEIGHTS_DIR, "Qwen--Qwen3.5-4B"),
}

_LABEL_MAP = {
    "Qwen3.5-0.8B": "0.8b",
    "Qwen/Qwen3.5-0.8B": "0.8b",
    "Qwen3.5-4B": "4b",
    "Qwen/Qwen3.5-4B": "4b",
}


class HFTorchBackend(BenchmarkBackend):
    """Real model backend using HuggingFace transformers."""

    def __init__(self, model_checkpoint: str = "Qwen3.5-0.8B") -> None:
        import torch

        short = _LABEL_MAP.get(model_checkpoint, model_checkpoint)
        if short in _MODEL_PATHS:
            self._model_path = _MODEL_PATHS[short]
        elif os.path.isdir(model_checkpoint):
            self._model_path = model_checkpoint
        else:
            self._model_path = model_checkpoint

        self._torch = torch
        self._model = None
        self._tokenizer = None
        self._kv_cache = None
        self._last_logits = None
        self._last_token = 0
        self._dtype = None
        self._weights_bytes = 0
        self._num_full_attn = 0
        self._num_linear = 0
        self._cfg: dict = {}

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
        """Pre-flight: refuse to load if available RAM is insufficient.

        Prevents the kernel OOM killer from silently killing the entire
        tmux/goose supervising session (bead ob-3lq). The OOM killer targets
        the cgroup with the most anon-rss, which is the tmux-spawn process,
        not just the python benchmark subprocess.
        """
        avail = self._available_memory_bytes()
        if avail == 0:
            return  # can't check (non-Linux or /proc unreadable) — proceed

        # Estimate: model weights + 50% overhead for activations, KV cache,
        # gradient buffers, and tokenizer state. For Qwen3.5-0.8B fp32:
        # weights ≈ 3.0 GiB → need ≈ 4.5 GiB free.
        # The config's dtype field tells us the intended precision.
        label = os.path.basename(self._model_path)
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
                f"(2) run on a higher-RAM node (e.g. rk3588-t3 has 31 GB), "
                f"(3) use ORIONS_FORCE_FP32=1 with a smaller model."
            )

        print(
            f"  [hf] Memory check: {avail / 1e9:.1f} GiB available "
            f"(need ~{required / 1e9:.1f} GiB) — OK",
            flush=True,
        )

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._check_memory()

        print(f"  [hf] Loading model from {self._model_path} ...", flush=True)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=True
        )

        # Try bf16 (native dtype); fall back to fp32 on ARM CPUs
        # without OneDNN bf16 (RK3588 Cortex-A76).
        use_fp32 = os.environ.get("ORIONS_FORCE_FP32", "0") == "1"

        if not use_fp32:
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    attn_implementation="eager",
                )
                self._dtype = torch.bfloat16
                # Quick sanity check: can we do a bf16 matmul without hanging?
                t = torch.randn(4, 4, dtype=torch.bfloat16)
                _ = (t @ t.T).sum().item()
                print("  [hf] Loaded in bfloat16", flush=True)
            except Exception:
                use_fp32 = True

        if use_fp32:
            print("  [hf] Loading in float32 ...", flush=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            self._dtype = torch.float32
            print("  [hf] Loaded in float32", flush=True)

        self._model.eval()

        # Introspect architecture
        cfg = self._model.config
        text_cfg = getattr(cfg, "text_config", cfg)

        layer_types = list(getattr(text_cfg, "layer_types", []))
        self._num_full_attn = sum(1 for lt in layer_types if lt == "full_attention")
        self._num_linear = sum(1 for lt in layer_types if lt == "linear_attention")

        self._cfg = {
            "num_hidden_layers": getattr(text_cfg, "num_hidden_layers", 0),
            "hidden_size": getattr(text_cfg, "hidden_size", 0),
            "num_kv_heads": getattr(text_cfg, "num_key_value_heads", 0),
            "head_dim": getattr(text_cfg, "head_dim", 0),
            "lin_key_heads": getattr(text_cfg, "linear_num_key_heads", 0),
            "lin_key_dim": getattr(text_cfg, "linear_key_head_dim", 0),
            "lin_val_heads": getattr(text_cfg, "linear_num_value_heads", 0),
            "lin_val_dim": getattr(text_cfg, "linear_value_head_dim", 0),
            "conv_kernel": getattr(text_cfg, "linear_conv_kernel_dim", 0),
        }

        self._weights_bytes = sum(
            p.numel() * p.element_size() for p in self._model.parameters()
        )

        print(
            f"  [hf] Arch: {self._cfg['num_hidden_layers']} layers "
            f"({self._num_full_attn} full-attn, {self._num_linear} linear/GDN), "
            f"weights={self._weights_bytes / 1e9:.3f} GB, "
            f"dtype={self._dtype}",
            flush=True,
        )

    def tokenize(self, text: str, max_tokens: int) -> list[int]:
        ids = self._tokenizer.encode(text, add_special_tokens=True)
        # BPE tokenizers compress repeated chars (e.g. "xxx" -> 1 token),
        # so pad with diverse text to reach the target context length.
        if len(ids) < max_tokens:
            pad_text = "The quick brown fox jumps over the lazy dog. "
            pad_ids = self._tokenizer.encode(pad_text, add_special_tokens=False)
            while len(ids) < max_tokens:
                ids.extend(pad_ids)
        return ids[:max_tokens]

    def prefill(self, input_ids: list[int]) -> None:
        ids = self._torch.tensor([input_ids], dtype=self._torch.long)
        attn = self._torch.ones_like(ids)

        with self._torch.no_grad():
            out = self._model(
                input_ids=ids,
                attention_mask=attn,
                use_cache=True,
                return_dict=True,
            )

        self._kv_cache = out.past_key_values
        self._last_logits = out.logits[:, -1, :]

    def sample_first_token(self) -> int:
        with self._torch.no_grad():
            token = self._torch.argmax(self._last_logits, dim=-1).item()
        self._last_token = token
        return token

    def decode_step(self) -> int:
        tok = self._torch.tensor([[self._last_token]], dtype=self._torch.long)
        attn = self._torch.ones_like(tok)

        with self._torch.no_grad():
            out = self._model(
                input_ids=tok,
                attention_mask=attn,
                past_key_values=self._kv_cache,
                use_cache=True,
                return_dict=True,
            )

        self._kv_cache = out.past_key_values
        self._last_logits = out.logits[:, -1, :]

        with self._torch.no_grad():
            next_tok = self._torch.argmax(self._last_logits, dim=-1).item()
        self._last_token = next_tok
        return next_tok

    def memory_breakdown(self, seq_len: int) -> MemoryBreakdown:
        """Three-component memory from architecture introspection (METRICS.md §5.0).

        - **weights**: flat — model parameters.
        - **kv_cache**: linear in seq_len, but ONLY for full-attention layers.
          With GQA (2 KV heads × 256 head_dim), 6 of 24 layers.
        - **recurrent_state**: O(1) — fixed for 18 linear/GDN layers.
          Each holds key+value state (16×128 each) in float32, plus
          a small conv state (4 × hidden_size × dtype).
        """
        elem_size = self._torch.tensor(0, dtype=self._dtype).element_size()

        # KV cache: 2 (K+V) × full_attn_layers × seq_len × kv_heads × head_dim × elem
        kv = (
            2 * self._num_full_attn * seq_len
            * self._cfg["num_kv_heads"] * self._cfg["head_dim"] * elem_size
        )

        # Recurrent state per linear layer (float32 per mamba_ssm_dtype):
        #   key_state  = lin_key_heads × lin_key_dim × 4
        #   value_state = lin_val_heads × lin_val_dim × 4
        #   conv_state  = conv_kernel × hidden_size × elem
        rec_per_layer = (
            self._cfg["lin_key_heads"] * self._cfg["lin_key_dim"]
            + self._cfg["lin_val_heads"] * self._cfg["lin_val_dim"]
        ) * 4  # fp32
        conv_per_layer = (
            self._cfg["conv_kernel"] * self._cfg["hidden_size"] * elem_size
        )
        rec = self._num_linear * (rec_per_layer + conv_per_layer)

        return MemoryBreakdown(
            weights=self._weights_bytes,
            kv_cache=kv,
            recurrent_state=rec,
        )
