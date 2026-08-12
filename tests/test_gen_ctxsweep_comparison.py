# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/gen_ctxsweep_comparison.py — classification, dedup, and markdown generation.

Covers the CSV filename classifier, the _4t_fair-over-_1t dedup preference,
formatter helpers, and end-to-end markdown output validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from gen_ctxsweep_comparison import (  # noqa: E402
    classify,
    fmt_mb,
    fmt_tok,
    fmt_us,
    generate_markdown,
)

# ---------------------------------------------------------------------------
# classify()
# ---------------------------------------------------------------------------


class TestClassify:
    """Filename classification: device, cluster, model, quant, arch."""

    def test_t4_hybrid_4b_fp32(self):
        key = classify("rk3588-t4_e2e_ctxsweep_4t_fair.csv")
        assert key == ("rk3588-t4", "big", "Qwen3.5-4B", "FP32", "hybrid")

    def test_t4_puregdn_4b_fp32(self):
        key = classify("rk3588-t4_e2e_ctxsweep_puregdn_4t.csv")
        assert key == ("rk3588-t4", "big", "Qwen3.5-4B", "FP32", "puregdn")

    def test_t4_int8_sdot_4b(self):
        key = classify("rk3588-t4_e2e_ctxsweep_int8_sdot_4t.csv")
        assert key == ("rk3588-t4", "big", "Qwen3.5-4B", "INT8+SDOT", "hybrid")

    def test_t4_int4_sdot_puregdn_4b(self):
        key = classify("rk3588-t4_e2e_ctxsweep_int4_sdot_puregdn_4t.csv")
        assert key == ("rk3588-t4", "big", "Qwen3.5-4B", "INT4+SDOT", "puregdn")

    def test_t4_08b_hybrid(self):
        key = classify("rk3588-t4_e2e_ctxsweep_08b_4t_fair.csv")
        assert key == ("rk3588-t4", "big", "Qwen3.5-0.8B", "FP32", "hybrid")

    def test_t3_big_int8_puregdn(self):
        key = classify("rk3588-t3_big_int8_puregdn_ctxsweep_e2e_raw.csv")
        assert key == ("rk3588-t3", "big", "Qwen3.5-4B", "INT8", "puregdn")

    def test_t3_big_int8_sdot(self):
        key = classify("rk3588-t3_big_int8_sdot_ctxsweep_e2e_raw.csv")
        assert key == ("rk3588-t3", "big", "Qwen3.5-4B", "INT8+SDOT", "hybrid")

    def test_t3_little_int8(self):
        key = classify("rk3588-t3_08b_little_int8_ctxsweep_e2e_raw.csv")
        assert key == ("rk3588-t3", "little", "Qwen3.5-0.8B", "INT8", "hybrid")

    def test_jetson_j1_q80_puregdn(self):
        key = classify("jetson-j1_08b_q80_puregdn_ctxsweep_e2e_raw.csv")
        assert key == ("jetson-j1", "big", "Qwen3.5-0.8B", "Q8_0", "puregdn")

    def test_jetson_j1_4b_fp32(self):
        key = classify("jetson-j1_4b_fp32_ctxsweep_e2e_raw.csv")
        assert key == ("jetson-j1", "big", "Qwen3.5-4B", "FP32", "hybrid")

    def test_rejects_non_ctxsweep(self):
        assert classify("rk3588-t4_big.csv") is None

    def test_rejects_schema(self):
        assert classify("rk3588-t4_ctxsweep_schema.csv") is None

    def test_default_model_is_4b(self):
        """Files without an explicit model marker default to 4B."""
        key = classify("rk3588-t4_e2e_ctxsweep_4t_fair.csv")
        assert key[2] == "Qwen3.5-4B"


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


class TestFormatters:
    def test_fmt_tok_normal(self):
        assert fmt_tok("4.5231") == "4.52"

    def test_fmt_tok_zero(self):
        assert fmt_tok("0") == "0.00"

    def test_fmt_tok_none(self):
        assert fmt_tok(None) == "—"

    def test_fmt_tok_empty(self):
        assert fmt_tok("") == "—"

    def test_fmt_tok_invalid(self):
        assert fmt_tok("abc") == "—"

    def test_fmt_us_normal(self):
        assert fmt_us("2500") == "2.5"

    def test_fmt_us_none(self):
        assert fmt_us(None) == "—"

    def test_fmt_mb_normal(self):
        assert fmt_mb("96") == "96"

    def test_fmt_mb_none(self):
        assert fmt_mb(None) == "—"

    def test_fmt_mb_float(self):
        assert fmt_mb("24.7") == "25"


# ---------------------------------------------------------------------------
# load_ctxsweep_data — dedup preference
# ---------------------------------------------------------------------------


class TestDedupPreference:
    """_4t_fair variant should win over _1t when both exist."""

    def test_4t_fair_preferred_over_1t(self, tmp_path, monkeypatch):
        """Create two CSVs with the same classification key but
        different _1t vs _4t_fair naming, verify _4t_fair wins."""
        import gen_ctxsweep_comparison as mod

        # Minimal CSV with ctxsweep schema columns
        header = ["ctx_len", "tok_per_sec", "kv_cache_mb", "total_us"]
        row_1t = ["1", "3.0", "0", "333333"]
        row_4t = ["1", "4.5", "0", "222222"]

        (tmp_path / "rk3588-t4_e2e_ctxsweep_1t.csv").write_text(
            f"{','.join(header)}\n{','.join(row_1t)}\n"
        )
        (tmp_path / "rk3588-t4_e2e_ctxsweep_4t_fair.csv").write_text(
            f"{','.join(header)}\n{','.join(row_4t)}\n"
        )

        monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
        datasets, sources = mod.load_ctxsweep_data()

        key = ("rk3588-t4", "big", "Qwen3.5-4B", "FP32", "hybrid")
        assert key in datasets
        assert sources[key]["csv"] == "rk3588-t4_e2e_ctxsweep_4t_fair.csv"
        assert float(datasets[key][0]["tok_per_sec"]) == 4.5

    def test_no_collision_different_quants(self, tmp_path, monkeypatch):
        """Different quant levels should NOT collide."""
        import gen_ctxsweep_comparison as mod

        header = ["ctx_len", "tok_per_sec", "kv_cache_mb", "total_us"]

        for _quant, fname in [
            ("FP32", "rk3588-t4_e2e_ctxsweep_4t_fair.csv"),
            ("INT8", "rk3588-t4_e2e_ctxsweep_int8_4t_fair.csv"),
        ]:
            (tmp_path / fname).write_text(f"{','.join(header)}\n1,2.0,0,500000\n")

        monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
        datasets, _ = mod.load_ctxsweep_data()

        fp32_key = ("rk3588-t4", "big", "Qwen3.5-4B", "FP32", "hybrid")
        int8_key = ("rk3588-t4", "big", "Qwen3.5-4B", "INT8", "hybrid")
        assert fp32_key in datasets
        assert int8_key in datasets
        assert len(datasets) == 2

    def test_no_collision_hybrid_vs_puregdn(self, tmp_path, monkeypatch):
        """Hybrid and puregdn variants of same config should NOT collide."""
        import gen_ctxsweep_comparison as mod

        header = ["ctx_len", "tok_per_sec", "kv_cache_mb", "total_us"]
        (tmp_path / "rk3588-t4_e2e_ctxsweep_4t_fair.csv").write_text(
            f"{','.join(header)}\n1,3.0,0,333333\n"
        )
        (tmp_path / "rk3588-t4_e2e_ctxsweep_puregdn_4t.csv").write_text(
            f"{','.join(header)}\n1,3.1,0,322580\n"
        )

        monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
        datasets, _ = mod.load_ctxsweep_data()

        hybrid_key = ("rk3588-t4", "big", "Qwen3.5-4B", "FP32", "hybrid")
        pure_key = ("rk3588-t4", "big", "Qwen3.5-4B", "FP32", "puregdn")
        assert hybrid_key in datasets
        assert pure_key in datasets

    def test_empty_csv_skipped(self, tmp_path, monkeypatch):
        """Empty CSVs (no data rows) should be silently skipped."""
        import gen_ctxsweep_comparison as mod

        (tmp_path / "rk3588-t4_e2e_ctxsweep_4t_fair.csv").write_text(
            "ctx_len,tok_per_sec,kv_cache_mb,total_us\n"  # header only
        )
        monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
        datasets, _ = mod.load_ctxsweep_data()
        assert len(datasets) == 0


# ---------------------------------------------------------------------------
# generate_markdown — integration
# ---------------------------------------------------------------------------


class TestGenerateMarkdown:
    """Verify markdown output structure and content."""

    @staticmethod
    def _make_dataset(ctx_lens=(1, 1024, 4096)):
        """Build a minimal dataset for one hybrid+puregdn pair."""

        def make_rows(tok_per_sec, kv_growth):
            rows = []
            for i, ctx in enumerate(ctx_lens):
                rows.append(
                    {
                        "ctx_len": str(ctx),
                        "tok_per_sec": str(tok_per_sec),
                        "kv_cache_mb": str(kv_growth * i),
                        "total_us": str(int(1e6 / tok_per_sec)),
                    }
                )
            return rows

        key_h = ("rk3588-t4", "big", "Qwen3.5-4B", "FP32", "hybrid")
        key_p = ("rk3588-t4", "big", "Qwen3.5-4B", "FP32", "puregdn")
        datasets = {
            key_h: make_rows(4.0, 24),
            key_p: make_rows(4.0, 0),  # puregdn KV always 0
        }
        sources = {
            key_h: {"csv": "rk3588-t4_e2e_ctxsweep_4t_fair.csv"},
            key_p: {"csv": "rk3588-t4_e2e_ctxsweep_puregdn_4t.csv"},
        }
        return datasets, sources

    def test_has_title(self):
        ds, src = self._make_dataset()
        md = generate_markdown(ds, src)
        assert "# Master context-sweep comparison" in md

    def test_has_key_result_blockquote(self):
        ds, src = self._make_dataset()
        md = generate_markdown(ds, src)
        assert "Key result" in md
        assert "O(1)" in md

    def test_has_section_header(self):
        ds, src = self._make_dataset()
        md = generate_markdown(ds, src)
        assert "## 1. Qwen3.5-4B on rk3588-t4 (big cluster)" in md

    def test_has_throughput_table(self):
        ds, src = self._make_dataset()
        md = generate_markdown(ds, src)
        assert "### Throughput (tok/s)" in md
        assert "| ctx=1 |" in md
        assert "| ctx=4096 |" in md

    def test_has_kv_cache_table(self):
        ds, src = self._make_dataset()
        md = generate_markdown(ds, src)
        assert "### KV cache (MB)" in md

    def test_has_latency_table(self):
        ds, src = self._make_dataset()
        md = generate_markdown(ds, src)
        assert "### Per-token latency (ms/tok)" in md

    def test_has_retention_note(self):
        ds, src = self._make_dataset()
        md = generate_markdown(ds, src)
        assert "throughput retention" in md
        assert "FP32" in md  # quant level appears

    def test_puregdn_kv_always_zero(self):
        ds, src = self._make_dataset()
        md = generate_markdown(ds, src)
        # In KV cache section, Pure GDN row should show 0 for all ctx
        lines = md.split("\n")
        in_kv = False
        pure_line = None
        for line in lines:
            if "### KV cache" in line:
                in_kv = True
            elif in_kv and "Pure GDN" in line:
                pure_line = line
                break
        assert pure_line is not None
        # All KV values in the pure-GDN row should be 0
        cells = pure_line.split("|")
        kv_values = [c.strip() for c in cells[4:-2]]  # skip quant, arch, source
        assert all(v == "0" for v in kv_values)

    def test_has_data_sources_section(self):
        ds, src = self._make_dataset()
        md = generate_markdown(ds, src)
        assert "## Data sources" in md
        assert "rk3588-t4_e2e_ctxsweep_4t_fair.csv" in md

    def test_hybrid_kv_grows_with_context(self):
        ds, src = self._make_dataset()
        md = generate_markdown(ds, src)
        # Hybrid row in KV cache section should show growing values
        lines = md.split("\n")
        in_kv = False
        hybrid_line = None
        for line in lines:
            if "### KV cache" in line:
                in_kv = True
            elif in_kv and "Hybrid 3:1" in line:
                hybrid_line = line
                break
        assert hybrid_line is not None
        cells = hybrid_line.split("|")
        kv_values = [c.strip() for c in cells[4:-2]]
        # Values should be monotonically non-decreasing
        nums = [int(v) for v in kv_values if v.isdigit()]
        assert nums == sorted(nums)

    def test_multiple_sections_numbered(self):
        """Two different device+model combos produce sections 1 and 2."""

        def make_rows(tps):
            return [
                {"ctx_len": "1", "tok_per_sec": str(tps), "kv_cache_mb": "0", "total_us": "500000"}
            ]

        datasets = {
            ("jetson-j1", "big", "Qwen3.5-4B", "FP32", "hybrid"): make_rows(2.0),
            ("rk3588-t4", "big", "Qwen3.5-4B", "FP32", "hybrid"): make_rows(3.0),
        }
        sources = {
            ("jetson-j1", "big", "Qwen3.5-4B", "FP32", "hybrid"): {"csv": "j1.csv"},
            ("rk3588-t4", "big", "Qwen3.5-4B", "FP32", "hybrid"): {"csv": "t4.csv"},
        }
        md = generate_markdown(datasets, sources)
        assert "## 1." in md
        assert "## 2." in md


# ---------------------------------------------------------------------------
# classify — non-.csv extension (line 73)
# ---------------------------------------------------------------------------


class TestClassifyEdgeCases:
    def test_rejects_non_csv_extension(self):
        """A file with ctxsweep in name but not .csv extension returns None."""
        assert classify("rk3588-t4_e2e_ctxsweep_raw.txt") is None

    def test_rejects_non_csv_extension_raw(self):
        """Even _raw suffix doesn't help if not .csv."""
        assert classify("rk3588-t4_e2e_ctxsweep_raw.json") is None


# ---------------------------------------------------------------------------
# load_ctxsweep_data — schema skip, non-ctxsweep skip, key collision (lines 116, 119, 136)
# ---------------------------------------------------------------------------


class TestLoadCtxsweepSkipBranches:
    def test_schema_file_in_dir_skipped(self, tmp_path, monkeypatch):
        """Schema files in the directory are skipped."""
        import gen_ctxsweep_comparison as mod

        header = "ctx_len,tok_per_sec,kv_cache_mb,total_us"
        (tmp_path / "rk3588-t4_e2e_ctxsweep_4t_fair.csv").write_text(f"{header}\n1,3.0,0,333333\n")
        (tmp_path / "rk3588-t4_ctxsweep_schema.csv").write_text(f"{header}\n")
        monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
        datasets, sources = mod.load_ctxsweep_data()
        assert len(datasets) == 1  # schema file skipped

    def test_non_ctxsweep_file_in_dir_skipped(self, tmp_path, monkeypatch):
        """Non-ctxsweep CSVs are skipped by classify()."""
        import gen_ctxsweep_comparison as mod

        header = "ctx_len,tok_per_sec,kv_cache_mb,total_us"
        (tmp_path / "rk3588-t4_e2e_ctxsweep_4t_fair.csv").write_text(f"{header}\n1,3.0,0,333333\n")
        (tmp_path / "rk3588-t4_big.csv").write_text(f"{header}\n1,2.0,0,500000\n")
        monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
        datasets, sources = mod.load_ctxsweep_data()
        assert len(datasets) == 1  # non-ctxsweep file skipped

    def test_key_collision_non_4t_fair_skipped(self, tmp_path, monkeypatch):
        """When a second file with the same key is not _4t_fair, it is skipped."""
        import gen_ctxsweep_comparison as mod

        header = "ctx_len,tok_per_sec,kv_cache_mb,total_us"
        # Both files classify to the same key; sorted order means aaa comes first.
        # aaa sets the key; bbb (not _4t_fair) should be skipped via continue.
        (tmp_path / "rk3588-t4_e2e_ctxsweep_aaa.csv").write_text(f"{header}\n1,3.0,0,333333\n")
        (tmp_path / "rk3588-t4_e2e_ctxsweep_bbb.csv").write_text(f"{header}\n1,4.5,0,222222\n")
        monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
        datasets, sources = mod.load_ctxsweep_data()
        assert len(datasets) == 1  # bbb skipped, aaa wins
        key = ("rk3588-t4", "big", "Qwen3.5-4B", "FP32", "hybrid")
        # aaa was processed first and wins
        assert sources[key]["csv"] == "rk3588-t4_e2e_ctxsweep_aaa.csv"


# ---------------------------------------------------------------------------
# generate_markdown — malformed row exception handlers (lines 236-237, 260-261, 285-286, 327-328)
# ---------------------------------------------------------------------------


class TestGenerateMarkdownMalformedRows:
    def test_malformed_ctx_len_handled(self):
        """Malformed ctx_len rows are gracefully skipped in all table sections."""
        key = ("rk3588-t4", "big", "Qwen3.5-4B", "FP32", "hybrid")
        datasets = {
            key: [
                {"ctx_len": "bad", "tok_per_sec": "3.0", "kv_cache_mb": "10", "total_us": "333333"},
                {"ctx_len": "1", "tok_per_sec": "4.0", "kv_cache_mb": "20", "total_us": "250000"},
            ]
        }
        sources = {key: {"csv": "test_ctxsweep_4t_fair.csv"}}
        md = generate_markdown(datasets, sources)
        # Should not crash, and should produce valid markdown
        assert "### Throughput" in md
        assert "### KV cache" in md
        assert "### Per-token latency" in md

    def test_malformed_tok_per_sec_handled(self):
        """Malformed tok_per_sec is skipped without crashing."""
        key = ("rk3588-t4", "big", "Qwen3.5-4B", "FP32", "hybrid")
        datasets = {
            key: [
                {"ctx_len": "1", "tok_per_sec": "bad", "kv_cache_mb": "10", "total_us": "333333"},
                {
                    "ctx_len": "4096",
                    "tok_per_sec": "3.0",
                    "kv_cache_mb": "50",
                    "total_us": "333333",
                },
            ]
        }
        sources = {key: {"csv": "test_ctxsweep_4t_fair.csv"}}
        md = generate_markdown(datasets, sources)
        assert "### Throughput" in md

    def test_malformed_total_us_handled(self):
        """Malformed total_us in latency table is skipped without crashing."""
        key = ("rk3588-t4", "big", "Qwen3.5-4B", "FP32", "hybrid")
        datasets = {
            key: [
                {"ctx_len": "1", "tok_per_sec": "3.0", "kv_cache_mb": "10", "total_us": "bad"},
                {
                    "ctx_len": "4096",
                    "tok_per_sec": "3.0",
                    "kv_cache_mb": "50",
                    "total_us": "400000",
                },
            ]
        }
        sources = {key: {"csv": "test_ctxsweep_4t_fair.csv"}}
        md = generate_markdown(datasets, sources)
        assert "### Per-token latency" in md

    def test_retention_note_malformed_data(self):
        """Retention note with malformed tok_per_sec is skipped via except."""
        key_h = ("rk3588-t4", "big", "Qwen3.5-4B", "FP32", "hybrid")
        key_p = ("rk3588-t4", "big", "Qwen3.5-4B", "FP32", "puregdn")
        datasets = {
            key_h: [
                {"ctx_len": "1", "tok_per_sec": "bad", "kv_cache_mb": "10", "total_us": "333333"},
                {
                    "ctx_len": "4096",
                    "tok_per_sec": "bad",
                    "kv_cache_mb": "50",
                    "total_us": "333333",
                },
            ],
            key_p: [
                {"ctx_len": "1", "tok_per_sec": "4.0", "kv_cache_mb": "0", "total_us": "250000"},
                {"ctx_len": "4096", "tok_per_sec": "3.0", "kv_cache_mb": "0", "total_us": "333333"},
            ],
        }
        sources = {
            key_h: {"csv": "test_ctxsweep_4t_fair.csv"},
            key_p: {"csv": "test_ctxsweep_puregdn_4t.csv"},
        }
        md = generate_markdown(datasets, sources)
        # Should not crash; retention note may be absent due to malformed data
        assert "## Data sources" in md


# ---------------------------------------------------------------------------
# main() — both paths (lines 345-354)
# ---------------------------------------------------------------------------


class TestMainFunction:
    def test_main_no_datasets_returns_1(self, tmp_path, monkeypatch, capsys):
        """main() with no ctxsweep CSVs prints message and returns 1."""
        import gen_ctxsweep_comparison as mod

        monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
        monkeypatch.setattr(mod, "OUT_PATH", tmp_path / "out.md")
        result = mod.main()
        assert result == 1
        captured = capsys.readouterr()
        assert "No ctxsweep CSVs found" in captured.out

    def test_main_with_data_returns_0(self, tmp_path, monkeypatch, capsys):
        """main() with valid data writes output and returns 0."""
        import gen_ctxsweep_comparison as mod

        header = "ctx_len,tok_per_sec,kv_cache_mb,total_us"
        (tmp_path / "raw" / "rk3588-t4_e2e_ctxsweep_4t_fair.csv").parent.mkdir(parents=True)
        (tmp_path / "raw" / "rk3588-t4_e2e_ctxsweep_4t_fair.csv").write_text(
            f"{header}\n1,3.0,0,333333\n4096,2.5,50,400000\n"
        )
        out_path = tmp_path / "out.md"
        monkeypatch.setattr(mod, "RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(mod, "OUT_PATH", out_path)
        result = mod.main()
        assert result == 0
        captured = capsys.readouterr()
        assert "Wrote" in captured.out
        assert out_path.exists()
