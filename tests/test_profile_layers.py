"""Tests for bench/profile_layers.py — per-layer latency profiling.

Focuses on write_csv() pure logic (p50/p95/mean computation, layer-type
assignment, CSV format) since load_model/run_profiling require PyTorch + weights.
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.profile_layers import tokenize_to_length, write_csv  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_times():
    """Build a small all_times dict with 3 layers, 2 phases, 1 ctx."""
    times = defaultdict(list)
    # Layer 0: linear_attention, prefill, ctx=64, 3 samples
    times[(0, "prefill", 64)] = [100.0, 120.0, 110.0]
    # Layer 1: full_attention, prefill, ctx=64, 3 samples
    times[(1, "prefill", 64)] = [200.0, 220.0, 210.0]
    # Layer 0: linear_attention, decode, ctx=64, 3 samples
    times[(0, "decode", 64)] = [50.0, 55.0, 52.0]
    # Layer 1: full_attention, decode, ctx=64, 3 samples
    times[(1, "decode", 64)] = [80.0, 90.0, 85.0]
    return times


def _read_csv(path):
    """Read a CSV file and return (fieldnames, list-of-row-dicts)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


# ---------------------------------------------------------------------------
# CSV output format
# ---------------------------------------------------------------------------


class TestWriteCsvFormat:
    def test_header_columns(self, tmp_path):
        times = _make_times()
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        fieldnames, rows = _read_csv(out)
        assert fieldnames == [
            "phase",
            "ctx_len",
            "layer_idx",
            "layer_type",
            "p50_us",
            "p95_us",
            "mean_us",
            "n_samples",
        ]

    def test_row_count(self, tmp_path):
        times = _make_times()
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        assert len(rows) == 4  # 2 layers × 2 phases

    def test_rows_sorted_by_layer_then_phase(self, tmp_path):
        times = _make_times()
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        # sorted(all_times) orders by (idx, phase, ctx)
        assert rows[0]["layer_idx"] == "0"
        assert rows[0]["phase"] == "decode"  # "decode" < "prefill" alphabetically
        assert rows[1]["layer_idx"] == "0"
        assert rows[1]["phase"] == "prefill"
        assert rows[2]["layer_idx"] == "1"
        assert rows[2]["phase"] == "decode"

    def test_empty_times_produces_empty_csv(self, tmp_path):
        out = tmp_path / "profile.csv"
        write_csv(defaultdict(list), set(), set(), str(out))
        fieldnames, rows = _read_csv(out)
        assert fieldnames is not None
        assert len(rows) == 0


# ---------------------------------------------------------------------------
# Layer-type assignment
# ---------------------------------------------------------------------------


class TestWriteCsvLayerType:
    def test_linear_attention_assigned(self, tmp_path):
        times = _make_times()
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        layer0_rows = [r for r in rows if r["layer_idx"] == "0"]
        assert all(r["layer_type"] == "linear_attention" for r in layer0_rows)

    def test_full_attention_assigned(self, tmp_path):
        times = _make_times()
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        layer1_rows = [r for r in rows if r["layer_idx"] == "1"]
        assert all(r["layer_type"] == "full_attention" for r in layer1_rows)

    def test_layer_not_in_either_set_defaults_to_linear(self, tmp_path):
        """A layer index not in full_attn defaults to linear_attention."""
        times = defaultdict(list)
        times[(5, "prefill", 32)] = [10.0, 20.0, 15.0]
        out = tmp_path / "profile.csv"
        write_csv(times, set(), set(), str(out))  # 5 not in either set
        _, rows = _read_csv(out)
        assert len(rows) == 1
        assert rows[0]["layer_type"] == "linear_attention"


# ---------------------------------------------------------------------------
# Statistics (p50, p95, mean)
# ---------------------------------------------------------------------------


class TestWriteCsvStats:
    def test_p50_is_median(self, tmp_path):
        times = defaultdict(list)
        times[(0, "prefill", 64)] = [100.0, 120.0, 110.0]
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        assert float(rows[0]["p50_us"]) == statistics.median([100.0, 120.0, 110.0])

    def test_mean_is_correct(self, tmp_path):
        times = defaultdict(list)
        times[(0, "prefill", 64)] = [100.0, 120.0, 110.0]
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        assert abs(float(rows[0]["mean_us"]) - statistics.mean([100.0, 120.0, 110.0])) < 0.1

    def test_p95_is_max_for_small_samples(self, tmp_path):
        """With < 20 samples, p95 should be max (not the percentile index)."""
        times = defaultdict(list)
        times[(0, "prefill", 64)] = [100.0, 200.0, 150.0]
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        assert float(rows[0]["p95_us"]) == 200.0

    def test_p95_percentile_for_large_samples(self, tmp_path):
        """With >= 20 samples, p95 should use the index-based percentile."""
        samples = list(range(100, 2100, 100))  # 20 samples: 100..2000
        times = defaultdict(list)
        times[(0, "prefill", 64)] = samples
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        expected_p95 = sorted(samples)[int(len(samples) * 0.95)]
        assert float(rows[0]["p95_us"]) == expected_p95

    def test_n_samples_recorded(self, tmp_path):
        times = _make_times()
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        for row in rows:
            assert int(row["n_samples"]) == 3

    def test_single_sample(self, tmp_path):
        """A single sample: p50 = p95 = mean = that value."""
        times = defaultdict(list)
        times[(0, "prefill", 64)] = [42.0]
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        assert float(rows[0]["p50_us"]) == 42.0
        assert float(rows[0]["p95_us"]) == 42.0
        assert float(rows[0]["mean_us"]) == 42.0


# ---------------------------------------------------------------------------
# Multiple contexts and phases
# ---------------------------------------------------------------------------


class TestWriteCsvMultiContext:
    def test_multiple_contexts(self, tmp_path):
        times = defaultdict(list)
        times[(0, "prefill", 32)] = [10.0, 12.0, 11.0]
        times[(0, "prefill", 64)] = [20.0, 22.0, 21.0]
        times[(0, "decode", 32)] = [5.0, 6.0, 5.5]
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        _, rows = _read_csv(out)
        assert len(rows) == 3
        ctx_values = {r["ctx_len"] for r in rows}
        assert ctx_values == {"32", "64"}

    def test_summary_prints_to_stdout(self, tmp_path, capsys):
        times = _make_times()
        out = tmp_path / "profile.csv"
        write_csv(times, {1}, {0}, str(out))
        captured = capsys.readouterr()
        assert "Summary" in captured.out
        assert "linear_attention" in captured.out
        assert "full_attention" in captured.out
        assert "Wrote 4 rows" in captured.out


# ---------------------------------------------------------------------------
# tokenize_to_length() — mock tokenizer
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Minimal tokenizer that returns predictable IDs."""

    def __init__(self, special_id=1, pad_ids=None):
        self._special = special_id
        self._pad_ids = pad_ids or [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    def encode(self, text, add_special_tokens=True):
        if add_special_tokens:
            return [self._special] + self._pad_ids[:]
        return self._pad_ids[:]


class TestTokenizeToLength:
    def test_returns_exact_length(self):
        tok = _MockTokenizer()
        result = tokenize_to_length(tok, 32)
        assert len(result) == 32

    def test_returns_exact_length_large(self):
        tok = _MockTokenizer()
        result = tokenize_to_length(tok, 500)
        assert len(result) == 500

    def test_seed_first_then_pad(self):
        """The first call encodes with special tokens, then padding repeats."""
        tok = _MockTokenizer(special_id=1, pad_ids=[10, 20])
        result = tokenize_to_length(tok, 6)
        # First call: [1, 10, 20], then extend with [10, 20], truncate to 6
        assert result[0] == 1
        assert result == [1, 10, 20, 10, 20, 10]

    def test_shorter_than_single_encode(self):
        """Target shorter than one encode → truncation."""
        tok = _MockTokenizer(special_id=1, pad_ids=[10, 20, 30])
        result = tokenize_to_length(tok, 2)
        assert len(result) == 2
        assert result == [1, 10]

    def test_large_repeat(self):
        """Many repetitions to reach a large target."""
        tok = _MockTokenizer(special_id=0, pad_ids=[1])
        result = tokenize_to_length(tok, 100)
        assert len(result) == 100
        assert result[0] == 0
        assert all(r == 1 for r in result[1:])

    def test_all_tokens_from_tokenizer(self):
        """No hardcoded IDs — all come from the tokenizer."""
        tok = _MockTokenizer(special_id=999, pad_ids=[888])
        result = tokenize_to_length(tok, 5)
        assert result == [999, 888, 888, 888, 888]


# ---------------------------------------------------------------------------
# write_csv — empty samples edge case
# ---------------------------------------------------------------------------


class TestWriteCsvEmptySamples:
    def test_empty_samples_skipped(self, tmp_path):
        """Entries with empty sample lists are skipped in CSV output."""
        times = defaultdict(list)
        times[(0, "prefill", 64)] = [10.0, 20.0]
        times[(1, "prefill", 64)] = []  # empty → skipped
        out = tmp_path / "profile.csv"
        write_csv(times, set(), {0, 1}, str(out))
        _, rows = _read_csv(out)
        assert len(rows) == 1
        assert rows[0]["layer_idx"] == "0"


# ---------------------------------------------------------------------------
# load_model() — mocked torch/transformers + fake config
# ---------------------------------------------------------------------------


class TestLoadModelMocked:
    """Test load_model() with mocked torch/transformers and a fake config."""

    def test_load_model_returns_tuple(self, monkeypatch, tmp_path):
        """load_model returns (model, tokenizer, cfg, layer_types)."""
        import json
        from unittest.mock import MagicMock

        import bench.profile_layers as pl

        # Create fake weights directory
        fake_weights = tmp_path / "weights"
        fake_model_dir = fake_weights / "Qwen--Qwen3.5-0.8B"
        fake_model_dir.mkdir(parents=True)
        config = {
            "num_hidden_layers": 4,
            "layer_types": [
                "linear_attention",
                "full_attention",
                "linear_attention",
                "full_attention",
            ],
        }
        (fake_model_dir / "config.json").write_text(json.dumps(config))

        monkeypatch.setattr(pl, "WEIGHTS", fake_weights)

        # Inject mock torch and transformers into sys.modules for inner imports
        mock_torch = MagicMock()
        mock_torch.float32 = "float32"
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        mock_tf = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tf.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_tf.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        monkeypatch.setitem(sys.modules, "transformers", mock_tf)

        model, tokenizer, cfg, layer_types = pl.load_model("Qwen3.5-0.8B")

        assert model is mock_model
        assert tokenizer is mock_tokenizer
        assert cfg["num_hidden_layers"] == 4
        assert len(layer_types) == 4
        # pad_token should be set to eos_token
        assert tokenizer.pad_token == "<eos>"
        mock_model.eval.assert_called_once()

    def test_load_model_extracts_text_config(self, monkeypatch, tmp_path):
        """load_model uses text_config if present in config.json."""
        import json
        from unittest.mock import MagicMock

        import bench.profile_layers as pl

        fake_weights = tmp_path / "weights"
        fake_model_dir = fake_weights / "Qwen--Qwen3.5-0.8B"
        fake_model_dir.mkdir(parents=True)
        config = {
            "text_config": {
                "num_hidden_layers": 2,
                "layer_types": ["linear_attention", "full_attention"],
            },
        }
        (fake_model_dir / "config.json").write_text(json.dumps(config))

        monkeypatch.setattr(pl, "WEIGHTS", fake_weights)
        monkeypatch.setitem(sys.modules, "torch", MagicMock(float32="float32"))

        mock_tf = MagicMock()
        mock_tf.AutoTokenizer.from_pretrained.return_value = MagicMock(pad_token="x")
        mock_tf.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()
        monkeypatch.setitem(sys.modules, "transformers", mock_tf)

        _, _, cfg, layer_types = pl.load_model("Qwen3.5-0.8B")
        assert cfg["num_hidden_layers"] == 2
        assert layer_types == ["linear_attention", "full_attention"]


# ---------------------------------------------------------------------------
# run_profiling() — mocked model + torch
# ---------------------------------------------------------------------------


class TestRunProfilingMocked:
    """Test run_profiling() with mocked model and torch."""

    def test_run_profiling_returns_times_and_sets(self, monkeypatch):
        """run_profiling returns (all_times, full_attn, linear) tuple."""
        from unittest.mock import MagicMock

        import bench.profile_layers as pl

        # Mock torch
        mock_torch = MagicMock()

        class _NoGradCtx:
            def __enter__(self):
                return None

            def __exit__(self, *a):
                return False

        mock_torch.no_grad.return_value = _NoGradCtx()
        mock_torch.tensor.return_value = MagicMock()
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        # Mock model with layers
        mock_model = MagicMock()
        mock_layers = [MagicMock() for _ in range(4)]
        mock_model.model.layers = mock_layers

        for layer in mock_layers:
            layer.register_forward_pre_hook.return_value = MagicMock()
            layer.register_forward_hook.return_value = MagicMock()

        mock_out = MagicMock()
        mock_out.past_key_values = MagicMock()
        mock_model.return_value = mock_out

        tok = _MockTokenizer()
        layer_types = [
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ]

        all_times, full_attn, linear = pl.run_profiling(
            mock_model, tok, layer_types, contexts=[32], repeats=1, decode_tokens=2
        )

        assert isinstance(all_times, dict)
        assert full_attn == {1, 3}
        assert linear == {0, 2}
        for layer in mock_layers:
            layer.register_forward_pre_hook.assert_called_once()
            layer.register_forward_hook.assert_called_once()

    def test_run_profiling_handles_multiple_contexts(self, monkeypatch):
        """run_profiling iterates over all context lengths."""
        from unittest.mock import MagicMock

        import bench.profile_layers as pl

        mock_torch = MagicMock()

        class _NoGradCtx:
            def __enter__(self):
                return None

            def __exit__(self, *a):
                return False

        mock_torch.no_grad.return_value = _NoGradCtx()
        mock_torch.tensor.return_value = MagicMock()
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        mock_model = MagicMock()
        mock_model.model.layers = [MagicMock() for _ in range(2)]
        mock_out = MagicMock()
        mock_out.past_key_values = MagicMock()
        mock_model.return_value = mock_out

        tok = _MockTokenizer()
        layer_types = ["linear_attention", "full_attention"]

        all_times, full_attn, linear = pl.run_profiling(
            mock_model, tok, layer_types, contexts=[32, 64, 128], repeats=1, decode_tokens=1
        )

        assert full_attn == {1}
        assert linear == {0}


# ---------------------------------------------------------------------------
# main() — mocked internals
# ---------------------------------------------------------------------------


class TestMainMocked:
    """Test main() with mocked load_model + run_profiling."""

    def test_main_writes_csv(self, monkeypatch, tmp_path, capsys):
        """main() calls load_model, run_profiling, write_csv."""
        from collections import defaultdict
        from unittest.mock import MagicMock

        import bench.profile_layers as pl

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_cfg = {"num_hidden_layers": 4}
        mock_layer_types = ["linear_attention", "full_attention"] * 2
        monkeypatch.setattr(
            pl,
            "load_model",
            MagicMock(return_value=(mock_model, mock_tokenizer, mock_cfg, mock_layer_types)),
        )

        fake_times = defaultdict(list)
        fake_times[(0, "prefill", 32)] = [10.0, 12.0]
        monkeypatch.setattr(
            pl,
            "run_profiling",
            MagicMock(return_value=(fake_times, {1, 3}, {0, 2})),
        )

        monkeypatch.setattr(pl, "write_csv", MagicMock())

        output_file = str(tmp_path / "profile.csv")
        monkeypatch.setattr(
            sys,
            "argv",
            ["profile_layers.py", "--model", "Qwen3.5-0.8B", "--output", output_file],
        )

        pl.main()

        pl.load_model.assert_called_once_with("Qwen3.5-0.8B")
        pl.run_profiling.assert_called_once()
        pl.write_csv.assert_called_once()

    def test_main_default_output_path(self, monkeypatch, tmp_path):
        """main() uses default output path when --output not given."""
        from collections import defaultdict
        from unittest.mock import MagicMock

        import bench.profile_layers as pl

        monkeypatch.setattr(
            pl,
            "load_model",
            MagicMock(
                return_value=(
                    MagicMock(),
                    MagicMock(),
                    {"num_hidden_layers": 2},
                    ["linear_attention"],
                )
            ),
        )
        fake_times = defaultdict(list)
        monkeypatch.setattr(
            pl,
            "run_profiling",
            MagicMock(return_value=(fake_times, set(), {0})),
        )
        monkeypatch.setattr(pl, "write_csv", MagicMock())
        monkeypatch.setattr(pl, "REPO", tmp_path)
        monkeypatch.setattr(sys, "argv", ["profile_layers.py"])

        pl.main()

        pl.write_csv.assert_called_once()
        output_arg = pl.write_csv.call_args[0][3]
        assert "rk3588-t4_layer_profile.csv" in str(output_arg)


# ---------------------------------------------------------------------------
# Hook callbacks — verify timing capture logic
# ---------------------------------------------------------------------------


class TestHookCallbacks:
    """Verify that the forward hooks actually record timing when fired."""

    def test_hooks_record_elapsed_time(self, monkeypatch):
        """Manually firing captured hooks records elapsed time in all_times."""
        from unittest.mock import MagicMock

        import bench.profile_layers as pl

        mock_torch = MagicMock()

        class _NoGradCtx:
            def __enter__(self):
                return None

            def __exit__(self, *a):
                return False

        mock_torch.no_grad.return_value = _NoGradCtx()
        mock_torch.tensor.return_value = MagicMock()
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        mock_model = MagicMock()
        mock_layer = MagicMock()
        mock_model.model.layers = [mock_layer]

        captured_pre: list = []
        captured_post: list = []

        def cap_pre(cb):
            captured_pre.append(cb)
            return MagicMock()

        def cap_post(cb):
            captured_post.append(cb)
            return MagicMock()

        mock_layer.register_forward_pre_hook.side_effect = cap_pre
        mock_layer.register_forward_hook.side_effect = cap_post

        mock_out = MagicMock()
        mock_out.past_key_values = MagicMock()
        mock_model.return_value = mock_out

        tok = _MockTokenizer()
        layer_types = ["linear_attention"]

        all_times, _, _ = pl.run_profiling(
            mock_model,
            tok,
            layer_types,
            contexts=[32],
            repeats=1,
            decode_tokens=1,
        )

        assert len(captured_pre) == 1
        assert len(captured_post) == 1

        # Fire pre then post to simulate a forward pass
        captured_pre[0](mock_layer, (MagicMock(),))
        captured_post[0](mock_layer, (MagicMock(),), MagicMock())

        # Timing should have been recorded (phase will be "decode" — last set value)
        any_recorded = any(len(v) > 0 for v in all_times.values())
        assert any_recorded
