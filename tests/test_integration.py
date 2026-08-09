# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Integration test: full pipeline end-to-end (harness → CSV → table → memory).

Exercises every module I built in a single end-to-end flow to catch regressions
in the interaction between components, not just within each component.

Run: pytest tests/test_integration.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.comparison_table import generate_comparison, load_and_summarize  # noqa: E402
from bench.harness import QWEN35_4B, SweepConfig, SyntheticBackend, run_sweep  # noqa: E402
from bench.memory import decomposition, recurrent_state_bytes, weights_bytes  # noqa: E402
from bench.schema import read_csv, validate_rows, write_csv  # noqa: E402


class TestFullPipeline:
    """End-to-end: harness produces CSV → CSV validates → table generates → memory decomposes."""

    def test_harness_csv_validates_on_readback(self, tmp_path):
        """The CSV written by the harness must round-trip through read_csv + validate."""
        backend = SyntheticBackend(QWEN35_4B)
        config = SweepConfig(
            context_lengths=[64, 128],
            warmup_count=1,
            repeat_count=5,
            decode_length=10,
        )
        rows = run_sweep(backend, config)

        csv_path = str(tmp_path / "integration.csv")
        write_csv(rows, csv_path)
        read_back = read_csv(csv_path)
        validate_rows(read_back)
        assert len(read_back) == len(rows)

    def test_comparison_table_from_harness_csv(self, tmp_path):
        """The comparison table generator must consume harness CSVs and produce a table."""
        backend = SyntheticBackend(QWEN35_4B)
        config = SweepConfig(
            context_lengths=[64],
            warmup_count=1,
            repeat_count=5,
            decode_length=10,
            engine_gdn="cpu",
            engine_full_attention="gpu_vulkan",
            quantization="fp16",
        )
        rows = run_sweep(backend, config)
        csv_path = str(tmp_path / "sweep.csv")
        write_csv(rows, csv_path)

        table = generate_comparison([csv_path])
        assert "cpu/gpu_vulkan" in table
        assert "prefill_tokens_per_sec" in table
        assert "peak_memory_bytes" in table

    def test_multi_config_comparison_table(self, tmp_path):
        """Multiple CSVs with different configs produce a multi-row comparison table."""
        csvs = []
        for engine in ("cpu", "gpu_vulkan", "npu"):
            backend = SyntheticBackend(QWEN35_4B)
            config = SweepConfig(
                context_lengths=[64],
                warmup_count=1,
                repeat_count=5,
                decode_length=10,
                engine_gdn="cpu",
                engine_full_attention=engine,
                quantization="fp16",
            )
            rows = run_sweep(backend, config)
            csv_path = str(tmp_path / f"sweep_{engine}.csv")
            write_csv(rows, csv_path)
            csvs.append(csv_path)

        summaries = load_and_summarize(csvs)
        engines = {s["engine_full_attention"] for s in summaries}
        assert engines == {"cpu", "gpu_vulkan", "npu"}

    def test_memory_decomposition_consistent_with_harness(self):
        """The standalone memory module must agree with the harness's memory rows."""
        backend = SyntheticBackend(QWEN35_4B)
        config = SweepConfig(
            context_lengths=[64],
            warmup_count=0,
            repeat_count=5,
            decode_length=10,
        )
        rows = run_sweep(backend, config)

        # Extract memory rows from the harness output
        mem_rows = [
            r for r in rows if r.metric_name == "peak_memory_bytes" and r.phase == "prefill"
        ]
        for component in ("weights", "kv_cache", "recurrent_state"):
            vals = {r.value for r in mem_rows if r.metric_component == component}
            assert len(vals) == 1, f"{component} should be constant across repeats"

            # Compare against standalone module
            if component == "weights":
                expected = weights_bytes(QWEN35_4B)
            elif component == "recurrent_state":
                expected = recurrent_state_bytes(QWEN35_4B)
            else:
                # kv_cache depends on seq_len (64 tokens ≈ 16 after tokenize at ~4 chars/token)
                continue

            actual = vals.pop()
            assert actual == expected, f"{component}: harness={actual}, standalone={expected}"

    def test_decomposition_shows_expected_scaling(self):
        """The memory decomposition must show the architectural claim: KV grows, state flat."""
        rows = decomposition(QWEN35_4B, [4096, 32768, 131072, 262144])

        # Weights: flat
        weights = [r["weights"] for r in rows]
        assert len(set(weights)) == 1

        # KV cache: grows linearly
        kv = [r["kv_cache"] for r in rows]
        assert kv == sorted(kv)
        assert kv[-1] > kv[0] * 10  # 262K should be >> 4K

        # Recurrent state: flat
        rs = [r["recurrent_state"] for r in rows]
        assert len(set(rs)) == 1

    def test_prompt_corpus_loads_in_harness(self):
        """The harness must load corpus prompts without crashing."""
        backend = SyntheticBackend(QWEN35_4B)
        config = SweepConfig(
            context_lengths=[4096],
            warmup_count=0,
            repeat_count=5,
            decode_length=10,
            prompt_type="needle",
        )
        rows = run_sweep(backend, config)
        assert len(rows) > 0
        validate_rows(rows)
