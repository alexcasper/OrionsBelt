"""Tests for bench/plots.py — CSV parsing, table generation, figure pipeline.

Bead ob-9y8 (t-plots).
"""

import os
import tempfile

import pytest

from plots import (
    MICROBENCH_COLUMNS,
    detect_format,
    generate_all,
    microbench_bandwidth_table,
    microbench_to_markdown,
    parse_microbench,
    parse_schema,
    schema_memory_table,
    schema_throughput_table,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MICROBENCH_CSV = """model,kernel,dispatch_path,seq,channels,repeats,p50_us,p95_us,spread_pct,gib_per_s_p50,gflop_per_s_p50
Qwen3.5-4B,gdn_cumdecay,neon,64,4096,30,1869.927,2682.284,43.4,1.04,0.14
Qwen3.5-4B,gdn_gated_scan,neon,64,4096,30,4049.756,4231.425,4.5,0.73,0.13
Qwen3.5-4B,gdn_causal_dwconv1d,neon,64,4096,30,2238.839,2979.008,33.1,0.92,0.94
Qwen3.5-0.8B,gdn_cumdecay,neon,64,2048,30,481.519,1027.309,113.3,2.03,0.27
Qwen3.5-0.8B,gdn_gated_scan,neon,64,2048,30,910.536,1215.594,33.5,1.63,0.29
Qwen3.5-0.8B,gdn_causal_dwconv1d,neon,64,2048,30,519.696,691.835,33.1,1.98,2.02
"""


# Schema CSV with per-repeat rows (as the harness would produce)
SCHEMA_CSV = """run_id,timestamp,git_sha,manifest_ref,device,engine_gdn,engine_full_attention,model_checkpoint,quantization,context_length,phase,metric_name,metric_component,value,unit,repeat_index,repeat_count,layer_class,notes
run1,2026-08-10T14:30:05Z,a1b2c3d,results/manifests/run1.json,generic_aarch64,cpu,cpu,Qwen/Qwen3.5-4B,fp16,4096,prefill,prefill_tokens_per_sec,,800.0,tokens_per_sec,0,5,all,
run1,2026-08-10T14:30:05Z,a1b2c3d,results/manifests/run1.json,generic_aarch64,cpu,cpu,Qwen/Qwen3.5-4B,fp16,4096,prefill,prefill_tokens_per_sec,,820.0,tokens_per_sec,1,5,all,
run1,2026-08-10T14:30:05Z,a1b2c3d,results/manifests/run1.json,generic_aarch64,cpu,cpu,Qwen/Qwen3.5-4B,fp16,4096,prefill,prefill_tokens_per_sec,,810.0,tokens_per_sec,2,5,all,
run1,2026-08-10T14:30:05Z,a1b2c3d,results/manifests/run1.json,generic_aarch64,cpu,cpu,Qwen/Qwen3.5-4B,fp16,4096,prefill,prefill_tokens_per_sec,,790.0,tokens_per_sec,3,5,all,
run1,2026-08-10T14:30:05Z,a1b2c3d,results/manifests/run1.json,generic_aarch64,cpu,cpu,Qwen/Qwen3.5-4B,fp16,4096,prefill,prefill_tokens_per_sec,,805.0,tokens_per_sec,4,5,all,
run1,2026-08-10T14:30:05Z,a1b2c3d,results/manifests/run1.json,generic_aarch64,cpu,cpu,Qwen/Qwen3.5-4B,fp16,4096,decode,decode_tokens_per_sec,,14.0,tokens_per_sec,0,5,all,
run1,2026-08-10T14:30:05Z,a1b2c3d,results/manifests/run1.json,generic_aarch64,cpu,cpu,Qwen/Qwen3.5-4B,fp16,4096,decode,decode_tokens_per_sec,,14.5,tokens_per_sec,1,5,all,
run1,2026-08-10T14:30:05Z,a1b2c3d,results/manifests/run1.json,generic_aarch64,cpu,cpu,Qwen/Qwen3.5-4B,fp16,4096,decode,decode_tokens_per_sec,,13.8,tokens_per_sec,2,5,all,
run1,2026-08-10T14:30:05Z,a1b2c3d,results/manifests/run1.json,generic_aarch64,cpu,cpu,Qwen/Qwen3.5-4B,fp16,4096,decode,decode_tokens_per_sec,,14.2,tokens_per_sec,3,5,all,
run1,2026-08-10T14:30:05Z,a1b2c3d,results/manifests/run1.json,generic_aarch64,cpu,cpu,Qwen/Qwen3.5-4B,fp16,4096,decode,decode_tokens_per_sec,,14.1,tokens_per_sec,4,5,all,
run1,2026-08-10T14:30:05Z,a1b2c3d,results/manifests/run1.json,generic_aarch64,cpu,cpu,Qwen/Qwen3.5-4B,fp16,4096,prefill,peak_memory_bytes,weights,10737418240,bytes,0,5,all,
run1,2026-08-10T14:30:05Z,a1b2c3d,results/manifests/run1.json,generic_aarch64,cpu,cpu,Qwen/Qwen3.5-4B,fp16,4096,prefill,peak_memory_bytes,kv_cache,134217728,bytes,0,5,all,
run1,2026-08-10T14:30:05Z,a1b2c3d,results/manifests/run1.json,generic_aarch64,cpu,cpu,Qwen/Qwen3.5-4B,fp16,4096,prefill,peak_memory_bytes,recurrent_state,50331648,bytes,0,5,all,
run1,2026-08-10T14:30:05Z,a1b2c3d,results/manifests/run1.json,generic_aarch64,cpu,cpu,Qwen/Qwen3.5-4B,fp16,32768,prefill,peak_memory_bytes,kv_cache,1073741824,bytes,0,5,all,
run1,2026-08-10T14:30:05Z,a1b2c3d,results/manifests/run1.json,generic_aarch64,cpu,cpu,Qwen/Qwen3.5-4B,fp16,32768,prefill,peak_memory_bytes,weights,10737418240,bytes,0,5,all,
run1,2026-08-10T14:30:05Z,a1b2c3d,results/manifests/run1.json,generic_aarch64,cpu,cpu,Qwen/Qwen3.5-4B,fp16,32768,prefill,peak_memory_bytes,recurrent_state,50331648,bytes,0,5,all,
"""


def write_temp_csv(content: str) -> str:
    """Write content to a temp CSV file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


class TestDetectFormat:
    def test_microbench(self):
        header = MICROBENCH_COLUMNS
        assert detect_format(header) == "microbench"

    def test_schema(self):
        header = [
            "run_id", "timestamp", "git_sha", "manifest_ref", "device",
            "engine_gdn", "engine_full_attention", "model_checkpoint",
            "quantization", "context_length", "phase", "metric_name",
            "metric_component", "value", "unit", "repeat_index",
            "repeat_count", "layer_class", "notes",
        ]
        assert detect_format(header) == "schema"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unrecognised"):
            detect_format(["foo", "bar", "baz"])


# ---------------------------------------------------------------------------
# Microbenchmark parsing
# ---------------------------------------------------------------------------


class TestParseMicrobench:
    def test_parse_real_data(self):
        path = write_temp_csv(MICROBENCH_CSV)
        rows = parse_microbench(path)
        os.unlink(path)

        assert len(rows) == 6
        assert rows[0].model == "Qwen3.5-4B"
        assert rows[0].kernel == "gdn_cumdecay"
        assert rows[0].dispatch_path == "neon"
        assert rows[0].seq == 64
        assert rows[0].channels == 4096
        assert rows[0].repeats == 30
        assert rows[0].gib_per_s == pytest.approx(1.04)
        assert rows[3].model == "Qwen3.5-0.8B"
        assert rows[3].gib_per_s == pytest.approx(2.03)

    def test_all_three_kernels(self):
        path = write_temp_csv(MICROBENCH_CSV)
        rows = parse_microbench(path)
        os.unlink(path)

        kernels = {r.kernel for r in rows}
        assert kernels == {"gdn_cumdecay", "gdn_gated_scan", "gdn_causal_dwconv1d"}

    def test_two_models(self):
        path = write_temp_csv(MICROBENCH_CSV)
        rows = parse_microbench(path)
        os.unlink(path)

        models = {r.model for r in rows}
        assert models == {"Qwen3.5-4B", "Qwen3.5-0.8B"}


# ---------------------------------------------------------------------------
# Schema parsing with aggregation
# ---------------------------------------------------------------------------


class TestParseSchema:
    def test_aggregates_repeats_to_p50(self):
        path = write_temp_csv(SCHEMA_CSV)
        rows = parse_schema(path)
        os.unlink(path)

        # Find the prefill throughput row
        prefill = [r for r in rows if r.metric_name == "prefill_tokens_per_sec"]
        assert len(prefill) == 1
        # Values were [790, 800, 805, 810, 820] → sorted → p50 at index 2 = 805
        assert prefill[0].p50 == pytest.approx(805.0)
        assert prefill[0].repeat_count == 5

    def test_aggregates_repeats_to_p95(self):
        path = write_temp_csv(SCHEMA_CSV)
        rows = parse_schema(path)
        os.unlink(path)

        prefill = [r for r in rows if r.metric_name == "prefill_tokens_per_sec"]
        assert len(prefill) == 1
        # p95 at index 4 = 820 (nearest-rank with 5 samples)
        assert prefill[0].p95 == pytest.approx(820.0)

    def test_decode_throughput(self):
        path = write_temp_csv(SCHEMA_CSV)
        rows = parse_schema(path)
        os.unlink(path)

        decode = [r for r in rows if r.metric_name == "decode_tokens_per_sec"]
        assert len(decode) == 1
        # Values [13.8, 14.0, 14.1, 14.2, 14.5] → p50 = 14.1
        assert decode[0].p50 == pytest.approx(14.1)

    def test_memory_components(self):
        path = write_temp_csv(SCHEMA_CSV)
        rows = parse_schema(path)
        os.unlink(path)

        mem = [r for r in rows if r.metric_name == "peak_memory_bytes"]
        # 4K: weights, kv_cache, recurrent_state (3)
        # 32K: weights, kv_cache, recurrent_state (3)
        assert len(mem) == 6

        weights_4k = [r for r in mem if r.metric_component == "weights" and r.context_length == 4096]
        assert len(weights_4k) == 1
        assert weights_4k[0].p50 == pytest.approx(10737418240.0)

    def test_kv_cache_grows_with_context(self):
        path = write_temp_csv(SCHEMA_CSV)
        rows = parse_schema(path)
        os.unlink(path)

        kv_4k = [r for r in rows if r.metric_component == "kv_cache" and r.context_length == 4096]
        kv_32k = [r for r in rows if r.metric_component == "kv_cache" and r.context_length == 32768]
        assert kv_4k[0].p50 == pytest.approx(134217728.0)
        assert kv_32k[0].p50 == pytest.approx(1073741824.0)

    def test_weights_flat_across_context(self):
        path = write_temp_csv(SCHEMA_CSV)
        rows = parse_schema(path)
        os.unlink(path)

        w = [r for r in rows if r.metric_component == "weights"]
        assert all(r.p50 == pytest.approx(10737418240.0) for r in w)


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------


class TestMicrobenchTables:
    def test_markdown_table_has_headers(self):
        path = write_temp_csv(MICROBENCH_CSV)
        rows = parse_microbench(path)
        os.unlink(path)

        md = microbench_to_markdown(rows)
        assert "| Model |" in md
        assert "| Kernel |" in md
        assert "GiB/s" in md
        assert "Gated Cumulative Decay" in md  # label lookup
        assert "1.04" in md

    def test_bandwidth_table_includes_spec(self):
        path = write_temp_csv(MICROBENCH_CSV)
        rows = parse_microbench(path)
        os.unlink(path)

        md = microbench_bandwidth_table(rows, "jetson-j1")
        assert "25.6 GiB/s" in md
        assert "% of Spec" in md

    def test_bandwidth_table_unknown_device(self):
        path = write_temp_csv(MICROBENCH_CSV)
        rows = parse_microbench(path)
        os.unlink(path)

        md = microbench_bandwidth_table(rows, "unknown-device")
        assert "unknown" in md.lower()

    def test_empty_rows(self):
        md = microbench_to_markdown([])
        assert "no data" in md.lower()


class TestSchemaTables:
    def test_throughput_table(self):
        path = write_temp_csv(SCHEMA_CSV)
        rows = parse_schema(path)
        os.unlink(path)

        md = schema_throughput_table(rows)
        assert "Throughput" in md
        assert "4,096" in md
        assert "805.0" in md  # p50
        assert "prefill" in md.lower() or "Prefill" in md

    def test_memory_table(self):
        path = write_temp_csv(SCHEMA_CSV)
        rows = parse_schema(path)
        os.unlink(path)

        md = schema_memory_table(rows)
        assert "Memory Decomposition" in md
        assert "Weights" in md
        assert "KV Cache" in md
        assert "Recurrent State" in md
        # 10 GiB = 10240 MiB
        assert "10,240.0" in md

    def test_memory_table_shows_kv_growth(self):
        path = write_temp_csv(SCHEMA_CSV)
        rows = parse_schema(path)
        os.unlink(path)

        md = schema_memory_table(rows)
        # KV cache: 128 MiB at 4K, 1024 MiB at 32K
        assert "128.0" in md
        assert "1,024.0" in md


# ---------------------------------------------------------------------------
# Device spec bandwidth lookup
# ---------------------------------------------------------------------------


class TestDeviceSpecBandwidth:
    def test_jetson_lookup(self):
        from plots import _lookup_spec_bandwidth
        assert _lookup_spec_bandwidth("jetson-j1") == 25.6
        assert _lookup_spec_bandwidth("jetson") == 25.6

    def test_pi5_lookup(self):
        from plots import _lookup_spec_bandwidth
        assert _lookup_spec_bandwidth("pi5") == 17.0

    def test_rk3588_lookup(self):
        from plots import _lookup_spec_bandwidth
        assert _lookup_spec_bandwidth("rk3588") == 34.0

    def test_o6_lookup(self):
        from plots import _lookup_spec_bandwidth
        assert _lookup_spec_bandwidth("o6") == 93.1

    def test_unknown_returns_none(self):
        from plots import _lookup_spec_bandwidth
        assert _lookup_spec_bandwidth("mystery-board") is None


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestGenerateAll:
    def test_microbench_only_text_mode(self):
        """Pipeline works with only microbench data, no matplotlib."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = os.path.join(tmpdir, "raw")
            out_dir = os.path.join(tmpdir, "figures")
            os.makedirs(raw_dir)

            csv_path = os.path.join(raw_dir, "jetson-j1.csv")
            with open(csv_path, "w") as f:
                f.write(MICROBENCH_CSV)

            result = generate_all([csv_path], out_dir, text_only=True)

            assert len(result.figures) == 0  # text-only mode
            assert len(result.tables) >= 1
            # Check that table was written
            table_files = [t for t in result.tables if "jetson" in t]
            assert len(table_files) == 1
            with open(table_files[0]) as f:
                content = f.read()
            assert "Gated Cumulative Decay" in content
            assert "25.6 GiB/s" in content  # spec bandwidth

    def test_schema_and_microbench_text_mode(self):
        """Pipeline handles both formats in one run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "figures")

            mb_path = write_temp_csv(MICROBENCH_CSV)
            sc_path = write_temp_csv(SCHEMA_CSV)

            result = generate_all([mb_path, sc_path], out_dir, text_only=True)

            os.unlink(mb_path)
            os.unlink(sc_path)

            assert len(result.tables) >= 2  # microbench table + schema table

    def test_creates_output_dir(self):
        """Output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "nested", "figures")
            mb_path = write_temp_csv(MICROBENCH_CSV)

            generate_all([mb_path], out_dir, text_only=True)
            os.unlink(mb_path)

            assert os.path.isdir(out_dir)

    def test_missing_file_warns(self):
        """Missing files produce a warning, not a crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "figures")
            result = generate_all(["/nonexistent/file.csv"], out_dir, text_only=True)
            assert any("not found" in w.lower() for w in result.warnings)

    def test_readme_generated(self):
        """A summary README.md is always generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "figures")
            mb_path = write_temp_csv(MICROBENCH_CSV)

            generate_all([mb_path], out_dir, text_only=True)
            os.unlink(mb_path)

            readme = os.path.join(out_dir, "README.md")
            assert os.path.exists(readme)
            with open(readme) as f:
                content = f.read()
            assert "Generated Figures" in content

    def test_figure_generation_with_matplotlib(self):
        """If matplotlib is available, figures are generated."""
        pytest.importorskip("matplotlib")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "figures")
            mb_path = write_temp_csv(MICROBENCH_CSV)

            result = generate_all([mb_path], out_dir, text_only=False)
            os.unlink(mb_path)

            # Should have at least one PNG figure
            assert len(result.figures) >= 1
            assert any(f.endswith(".png") for f in result.figures)
