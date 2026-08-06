# Backend implementation guide: wrapping HuggingFace transformers in the Backend ABC

**Bead:** `ob-xh3.1` · **For:** `ob-aqv` (x86/CUDA reference) · **Backend ABC:** [`bench/harness.py`](../bench/harness.py)

This guide shows exactly how to implement the `Backend` protocol for a real Qwen3.5
model via HuggingFace `transformers`. When `ob-aqv` starts, the implementer should
be able to write the backend in under an hour by following this mapping.

---

## The contract: the harness handles ALL timing

The `Backend` ABC provides **compute operations only**. The harness wraps every
method call in `time.perf_counter()` boundaries according to the METRICS.md
timing protocol (sections 1–5). The backend never reads a clock, never writes a
CSV, and never needs to know about percentiles or the results schema.

```
Harness timing protocol (METRICS.md):
  t_submit → backend.tokenize() → t_prefill_start
           → backend.prefill()  → t_prefill_logits
           → backend.memory_bytes()  [prefill-end sample]
           → backend.sample()   → t_first_token
           → backend.decode_step() × (N-1) → t_N
           → backend.memory_bytes()  [decode-end sample]
```

---

## Method-by-method implementation

### `load(self) -> None`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(self):
    self.tokenizer = AutoTokenizer.from_pretrained(self.config.name)
    self.model = AutoModelForCausalLM.from_pretrained(
        self.config.name,
        torch_dtype=torch.float16,  # or bfloat16
        device_map="auto",  # or .cuda() / .to("npu")
    )
    self.model.eval()
    # Quantize weights per QUANTIZATION_POLICY.md if needed:
    #   from transformers import BitsAndBytesConfig
    #   quantization_config = BitsAndBytesConfig(load_in_4bit=True, ...)
```

Model load time is excluded from all metrics (METRICS.md §1). The harness calls
`load()` once before the sweep.

### `tokenize(self, text: str) -> list[int]`

```python
def tokenize(self, text: str) -> list[int]:
    return self.tokenizer.encode(text)
```

The harness records `len(input_ids)` as the prompt token count (METRICS.md §2).
Return a plain list — the harness passes it to `prefill()`.

### `prefill(self, input_ids: list[int]) -> Any`

```python
def prefill(self, input_ids: list[int]) -> Any:
    input_tensor = torch.tensor([input_ids], device=self.model.device)
    self._seq_len = len(input_ids)
    with torch.inference_mode():
        outputs = self.model(
            input_tensor,
            use_cache=True,  # populate KV cache + recurrent state
            output_hidden_states=False,
        )
    self._past_key_values = outputs.past_key_values
    return outputs.logits  # passed to sample()
```

For Qwen3.5, `past_key_values` is a `Qwen3_5Cache` object holding:
- **conv_states** — the causal Conv1D sliding window (GDN layers)
- **recurrent_states** — the GDN recurrent state `S` (shape: `(batch, n_v_heads, k_dim, v_dim)`, confirmed in ob-37v)
- **key_value_cache** — the standard KV cache (full-attention layers only)

These are carried forward into `decode_step()` — do NOT discard them.

### `sample(self, logits: Any) -> int`

```python
def sample(self, logits) -> int:
    # logits shape: (batch, seq_len, vocab_size) — take last position
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    return next_token.item()
```

Token 1 belongs to prefill, not decode (METRICS.md §1). The harness times the
interval from `t_prefill_logits` to here as part of TTFT.

### `decode_step(self, token_id: int) -> int`

```python
def decode_step(self, token_id: int) -> int:
    self._seq_len += 1
    input_tensor = torch.tensor([[token_id]], device=self.model.device)
    with torch.inference_mode():
        outputs = self.model(
            input_tensor,
            past_key_values=self._past_key_values,
            use_cache=True,
        )
    self._past_key_values = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    return next_token.item()
```

EOS does not stop generation during benchmarking (METRICS.md §4). The harness
runs exactly `decode_length - 1` steps regardless.

### `memory_bytes(self) -> dict[str, int]`

```python
from bench.memory import weights_bytes, kv_cache_bytes, recurrent_state_bytes, cross_check


def memory_bytes(self) -> dict[str, int]:
    # Analytic formulas (bench/memory.py, verified against ADR 0003):
    return {
        "weights": weights_bytes(self.config),
        "kv_cache": kv_cache_bytes(self.config, self._seq_len),
        "recurrent_state": recurrent_state_bytes(self.config),
    }
```

**Cross-check against live tensors** (METRICS.md §5.4) — call this once after
`load()` to verify the analytic formulas match reality:

```python
def verify_memory(self):
    # Introspect the actual recurrent state tensor shape
    # (Qwen3_5Cache.recurrent_states is a list of per-layer tensors)
    state_shape = self._past_key_values.recurrent_states[0].shape
    state_dtype_bytes = self._past_key_values.recurrent_states[0].element_size()
    discrepancies = cross_check(
        self.config,
        introspected_state_shape=tuple(state_shape),
        introspected_state_dtype_bytes=state_dtype_bytes,
        # introspected_weights=sum(p.numel() * p.element_size() for p in self.model.parameters()),
    )
    if discrepancies:
        raise RuntimeError(f"Memory cross-check failed: {discrepancies}")
```

This catches bugs that would silently falsify the project's central claim (e.g.
a state that grows with context instead of staying O(1)).

### `reset(self) -> None`

```python
def reset(self) -> None:
    self._past_key_values = None
    self._seq_len = 0
```

Called before each repeat. The model weights stay loaded — only the cache/state
is cleared.

---

## Quantization integration

Per [`docs/QUANTIZATION_POLICY.md`](./QUANTIZATION_POLICY.md), the recommended
configuration is INT4 weights with FP32 recurrent state. To load a quantized model:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, quantization_config=bnb_config, device_map="auto"
)
```

The recurrent state stays in FP32 automatically — `mamba_ssm_dtype: "float32"`
in the config controls this, and BitsAndBytes only quantizes weights, not
activations or recurrent state.

---

## Testing the implementation

Before running the full sweep, verify with a small smoke test:

```python
from bench.harness import QWEN35_08B, SweepConfig, run_sweep

backend = HFBatchBackend(QWEN35_08B)
config = SweepConfig(
    context_lengths=[64],
    warmup_count=1,
    repeat_count=5,
    decode_length=10,
    device="x86_reference",
    model_checkpoint="Qwen/Qwen3.5-0.8B@x86-cuda",
)
rows = run_sweep(backend, config)
# All rows should validate against the frozen schema automatically.
```

The `cross_check()` call in `verify_memory()` will catch any state-shape
mismatch immediately, before any timing data is collected.

---

## What the implementer needs

- An x86 machine with a CUDA GPU (or Apple Silicon / Graviton as fallback)
- `pip install transformers torch` (+ `bitsandbytes` for INT4)
- `python3 scripts/fetch_weights.py --model 0.8b` to download the checkpoint
- This guide + `bench/harness.py` (Backend ABC) + `bench/memory.py` (formulas)
- ~1 hour to implement, ~30 minutes to smoke-test

The rest of the pipeline (sweep, percentiles, CSV, manifest, plots, comparison
table) is already built and will work unchanged with the real backend.
