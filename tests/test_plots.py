"""Tests for bench/plots.py — CSV parsing, table generation, figure pipeline.

Bead ob-9y8 (t-plots).
"""

import os
import tempfile

import pytest

from plots import (
    MICROBENCH_COLUMNS,
    MicrobenchRow,
    PlotResult,
    SchemaRow,
    _lookup_spec_bandwidth,
    _plot_cross_device,
    _try_import_matplotlib,
    detect_format,
    generate_all,
    main,
    microbench_bandwidth_table,
    microbench_to_markdown,
    parse_microbench,
    parse_schema,
    plot_bandwidth_bars,
    plot_memory_decomposition,
    plot_throughput_curve,
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
            "run_id",
            "timestamp",
            "git_sha",
            "manifest_ref",
            "device",
            "engine_gdn",
            "engine_full_attention",
            "model_checkpoint",
            "quantization",
            "context_length",
            "phase",
            "metric_name",
            "metric_component",
            "value",
            "unit",
            "repeat_index",
            "repeat_count",
            "layer_class",
            "notes",
        ]
        assert detect_format(header) == "schema"

    def test_unknown_returns_none(self):
        """Unrecognised headers return None rather than raising.

        results/raw/ also holds power-monitor CSVs, and generate_all() walks the
        whole directory — so raising here took the entire plot run down over a
        file it never needed to read.
        """
        assert detect_format(["foo", "bar", "baz"]) is None


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

        weights_4k = [
            r for r in mem if r.metric_component == "weights" and r.context_length == 4096
        ]
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

    def test_bandwidth_table_empty_rows(self):
        md = microbench_bandwidth_table([])
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

    def test_throughput_table_empty_rows(self):
        md = schema_throughput_table([])
        assert "no throughput data" in md.lower()

    def test_memory_table_empty_rows(self):
        md = schema_memory_table([])
        assert "no memory data" in md.lower()


# ---------------------------------------------------------------------------
# Device spec bandwidth lookup
# ---------------------------------------------------------------------------


class TestDeviceSpecBandwidth:
    def test_jetson_lookup(self):

        assert _lookup_spec_bandwidth("jetson-j1") == 25.6
        assert _lookup_spec_bandwidth("jetson") == 25.6

    def test_pi5_lookup(self):

        assert _lookup_spec_bandwidth("pi5") == 17.0

    def test_rk3588_lookup(self):

        assert _lookup_spec_bandwidth("rk3588") == 34.0

    def test_o6_lookup(self):

        assert _lookup_spec_bandwidth("o6") == 93.1

    def test_unknown_returns_none(self):

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


# ---------------------------------------------------------------------------
# Helpers for figure tests
# ---------------------------------------------------------------------------


def _mb_row(**overrides):
    """Create a MicrobenchRow with sensible defaults."""
    defaults = dict(
        model="Qwen3.5-4B",
        kernel="gdn_gated_scan",
        dispatch_path="neon",
        seq=64,
        channels=4096,
        repeats=30,
        p50_us=4000.0,
        p95_us=4200.0,
        spread_pct=5.0,
        gib_per_s=0.73,
        gflop_per_s=0.13,
    )
    defaults.update(overrides)
    return MicrobenchRow(**defaults)


def _sc_row(**overrides):
    """Create a SchemaRow with sensible defaults."""
    defaults = dict(
        run_id="run1",
        device="rk3588-t4",
        engine_gdn="cpu",
        engine_full_attention="cpu",
        model_checkpoint="Qwen/Qwen3.5-4B",
        quantization="fp16",
        context_length=4096,
        phase="prefill",
        metric_name="prefill_tokens_per_sec",
        metric_component="",
        unit="tokens_per_sec",
        p50=800.0,
        p95=820.0,
        repeat_count=5,
    )
    defaults.update(overrides)
    return SchemaRow(**defaults)


# ---------------------------------------------------------------------------
# plot_bandwidth_bars
# ---------------------------------------------------------------------------


class TestPlotBandwidthBars:
    @pytest.fixture
    def plt(self):
        return pytest.importorskip("matplotlib.pyplot")

    def test_none_plt_returns_none(self):
        """If plt is None, returns None without crashing."""
        rows = [_mb_row()]
        assert plot_bandwidth_bars(rows, "rk3588-t4", "/tmp/out.png", plt=None) is None

    def test_creates_png(self, plt, tmp_path):
        """Valid rows produce a PNG file, returns the path."""
        out = str(tmp_path / "bw.png")
        rows = [_mb_row(kernel="gdn_cumdecay"), _mb_row(kernel="gdn_gated_scan")]
        result = plot_bandwidth_bars(rows, "rk3588-t4", out, plt)
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_spec_bandwidth_line_drawn(self, plt, tmp_path):
        """Known device draws a spec reference line (just verifies no error)."""
        out = str(tmp_path / "bw.png")
        rows = [_mb_row(gib_per_s=10.0)]
        result = plot_bandwidth_bars(rows, "jetson-j1", out, plt)
        assert result == out
        assert os.path.exists(out)

    def test_unknown_device_no_spec_line(self, plt, tmp_path):
        """Unknown device doesn't crash (no spec line drawn)."""
        out = str(tmp_path / "bw.png")
        rows = [_mb_row(gib_per_s=10.0)]
        result = plot_bandwidth_bars(rows, "mystery-board", out, plt)
        assert result == out
        assert os.path.exists(out)

    def test_multiple_models_same_kernel(self, plt, tmp_path):
        """Multiple models for the same kernel are shown side by side."""
        out = str(tmp_path / "bw.png")
        rows = [
            _mb_row(model="Qwen3.5-4B", gib_per_s=1.0),
            _mb_row(model="Qwen3.5-0.8B", gib_per_s=2.0),
        ]
        result = plot_bandwidth_bars(rows, "rk3588-t4", out, plt)
        assert result == out
        assert os.path.getsize(out) > 100  # non-trivial file


# ---------------------------------------------------------------------------
# plot_throughput_curve
# ---------------------------------------------------------------------------


class TestPlotThroughputCurve:
    @pytest.fixture
    def plt(self):
        return pytest.importorskip("matplotlib.pyplot")

    def test_none_plt_returns_none(self):
        assert plot_throughput_curve([_sc_row()], "/tmp/out.png", plt=None) is None

    def test_empty_rows_returns_none(self, plt, tmp_path):
        out = str(tmp_path / "tc.png")
        assert plot_throughput_curve([], out, plt) is None

    def test_no_throughput_metrics_returns_none(self, plt, tmp_path):
        """Rows with non-throughput metric → no plot."""
        out = str(tmp_path / "tc.png")
        rows = [_sc_row(metric_name="peak_memory_bytes")]
        assert plot_throughput_curve(rows, out, plt) is None

    def test_creates_png_with_throughput(self, plt, tmp_path):
        out = str(tmp_path / "tc.png")
        rows = [
            _sc_row(context_length=4096, p50=800.0, p95=820.0),
            _sc_row(context_length=32768, p50=300.0, p95=320.0),
            _sc_row(context_length=131072, p50=100.0, p95=110.0),
        ]
        result = plot_throughput_curve(rows, out, plt)
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_decode_metric(self, plt, tmp_path):
        """Decode throughput is plotted as a separate group."""
        out = str(tmp_path / "tc.png")
        rows = [
            _sc_row(metric_name="decode_tokens_per_sec", context_length=4096, p50=14.0),
            _sc_row(metric_name="decode_tokens_per_sec", context_length=32768, p50=12.0),
        ]
        result = plot_throughput_curve(rows, out, plt)
        assert result == out
        assert os.path.getsize(out) > 0


# ---------------------------------------------------------------------------
# plot_memory_decomposition
# ---------------------------------------------------------------------------


class TestPlotMemoryDecomposition:
    @pytest.fixture
    def plt(self):
        return pytest.importorskip("matplotlib.pyplot")

    def test_none_plt_returns_none(self):
        assert plot_memory_decomposition([_sc_row()], "/tmp/out.png", plt=None) is None

    def test_empty_rows_returns_none(self, plt, tmp_path):
        out = str(tmp_path / "md.png")
        assert plot_memory_decomposition([], out, plt) is None

    def test_no_memory_metrics_returns_none(self, plt, tmp_path):
        out = str(tmp_path / "md.png")
        rows = [_sc_row(metric_name="prefill_tokens_per_sec")]
        assert plot_memory_decomposition(rows, out, plt) is None

    def test_creates_png_with_memory(self, plt, tmp_path):
        out = str(tmp_path / "md.png")
        rows = [
            _sc_row(
                metric_name="peak_memory_bytes",
                metric_component="weights",
                p50=10737418240.0,
                unit="bytes",
            ),
            _sc_row(
                metric_name="peak_memory_bytes",
                metric_component="kv_cache",
                p50=134217728.0,
                unit="bytes",
            ),
            _sc_row(
                metric_name="peak_memory_bytes",
                metric_component="recurrent_state",
                p50=50331648.0,
                unit="bytes",
            ),
        ]
        result = plot_memory_decomposition(rows, out, plt)
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_multiple_context_lengths(self, plt, tmp_path):
        """Multiple context lengths produce a curve rather than a single point."""
        out = str(tmp_path / "md.png")
        rows = []
        for ctx in [4096, 32768, 131072]:
            for comp, val in [
                ("weights", 10737418240),
                ("kv_cache", ctx * 32768),
                ("recurrent_state", 50331648),
            ]:
                rows.append(
                    _sc_row(
                        context_length=ctx,
                        metric_name="peak_memory_bytes",
                        metric_component=comp,
                        p50=float(val),
                        unit="bytes",
                    )
                )
        result = plot_memory_decomposition(rows, out, plt)
        assert result == out
        assert os.path.getsize(out) > 0


# ---------------------------------------------------------------------------
# _plot_cross_device (tested via generate_all with multiple devices)
# ---------------------------------------------------------------------------


class TestCrossDevicePlot:
    def test_cross_device_generated(self, tmp_path):
        """Two devices' CSVs → cross_device_bandwidth.png is created."""
        pytest.importorskip("matplotlib")
        out_dir = str(tmp_path / "figures")

        csv_t3 = write_temp_csv(MICROBENCH_CSV)
        csv_t4 = write_temp_csv(MICROBENCH_CSV)

        result = generate_all([csv_t3, csv_t4], out_dir, text_only=False)

        os.unlink(csv_t3)
        os.unlink(csv_t4)

        cross = [f for f in result.figures if "cross_device" in os.path.basename(f)]
        assert len(cross) == 1
        assert os.path.exists(cross[0])

    def test_single_device_no_cross_plot(self, tmp_path):
        """Single device → no cross-device plot."""
        pytest.importorskip("matplotlib")
        out_dir = str(tmp_path / "figures")
        mb_path = write_temp_csv(MICROBENCH_CSV)

        result = generate_all([mb_path], out_dir, text_only=False)
        os.unlink(mb_path)

        cross = [f for f in result.figures if "cross_device" in os.path.basename(f)]
        assert len(cross) == 0

    def test_cross_device_text_only_skipped(self, tmp_path):
        """Text-only mode never produces cross-device figure."""
        out_dir = str(tmp_path / "figures")
        csv1 = write_temp_csv(MICROBENCH_CSV)
        csv2 = write_temp_csv(MICROBENCH_CSV)

        result = generate_all([csv1, csv2], out_dir, text_only=True)
        os.unlink(csv1)
        os.unlink(csv2)

        cross = [f for f in result.figures if "cross_device" in os.path.basename(f)]
        assert len(cross) == 0


# ---------------------------------------------------------------------------
# generate_all with schema figures
# ---------------------------------------------------------------------------


class TestGenerateAllSchemaFigures:
    def test_schema_figures_generated(self, tmp_path):
        """Schema CSV produces throughput + memory figures."""
        pytest.importorskip("matplotlib")
        out_dir = str(tmp_path / "figures")
        sc_path = write_temp_csv(SCHEMA_CSV)

        result = generate_all([sc_path], out_dir, text_only=False)
        os.unlink(sc_path)

        pngs = [f for f in result.figures if f.endswith(".png")]
        assert len(pngs) >= 2
        assert any("throughput" in f for f in pngs)
        assert any("memory" in f for f in pngs)

    def test_text_only_schema_tables(self, tmp_path):
        """Schema data in text-only mode still produces tables."""
        out_dir = str(tmp_path / "figures")
        sc_path = write_temp_csv(SCHEMA_CSV)

        result = generate_all([sc_path], out_dir, text_only=True)
        os.unlink(sc_path)

        assert len(result.figures) == 0
        assert len(result.tables) >= 1
        harness = [t for t in result.tables if "harness" in t]
        assert len(harness) == 1

    def test_unrecognised_csv_skipped(self, tmp_path):
        """Non-matching CSV format is silently skipped."""
        out_dir = str(tmp_path / "figures")
        unknown = write_temp_csv("foo,bar,baz\n1,2,3\n")

        result = generate_all([unknown], out_dir, text_only=True)
        os.unlink(unknown)

        assert len(result.tables) == 0
        assert len(result.warnings) == 0

    def test_summary_readme_has_figure_list(self, tmp_path):
        """README lists generated figures when matplotlib is available."""
        pytest.importorskip("matplotlib")
        out_dir = str(tmp_path / "figures")
        mb_path = write_temp_csv(MICROBENCH_CSV)

        generate_all([mb_path], out_dir, text_only=False)
        os.unlink(mb_path)

        readme = os.path.join(out_dir, "README.md")
        with open(readme) as f:
            content = f.read()
        assert "Figures" in content

    def test_summary_readme_no_warnings(self, tmp_path):
        """When everything works, README has no warnings section."""
        pytest.importorskip("matplotlib")
        out_dir = str(tmp_path / "figures")
        mb_path = write_temp_csv(MICROBENCH_CSV)

        generate_all([mb_path], out_dir, text_only=False)
        os.unlink(mb_path)

        readme = os.path.join(out_dir, "README.md")
        with open(readme) as f:
            content = f.read()
        assert "Warnings" not in content


# ---------------------------------------------------------------------------
# PlotResult dataclass
# ---------------------------------------------------------------------------


class TestPlotResult:
    def test_defaults_empty(self):
        r = PlotResult()
        assert r.figures == []
        assert r.tables == []
        assert r.warnings == []

    def test_can_append(self):
        r = PlotResult()
        r.figures.append("a.png")
        r.tables.append("b.md")
        r.warnings.append("oops")
        assert len(r.figures) == 1
        assert len(r.tables) == 1
        assert len(r.warnings) == 1


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------


class TestMainCLI:
    def test_no_csv_returns_error(self):
        """No CSV files found → returns 1."""
        rc = main(["--output-dir", "/tmp/empty_plots_test", "/nonexistent/"])
        assert rc == 1

    def test_microbench_text_only(self, tmp_path):
        """Basic text-only run succeeds and returns 0."""
        out_dir = str(tmp_path / "figures")
        csv_path = write_temp_csv(MICROBENCH_CSV)

        rc = main([csv_path, "--output-dir", out_dir, "--text-only"])
        os.unlink(csv_path)

        assert rc == 0
        assert os.path.isdir(out_dir)

    def test_microbench_with_figures(self, tmp_path):
        """Full run with matplotlib returns 0 and creates figures."""
        pytest.importorskip("matplotlib")
        out_dir = str(tmp_path / "figures")
        csv_path = write_temp_csv(MICROBENCH_CSV)

        rc = main([csv_path, "--output-dir", out_dir])
        os.unlink(csv_path)

        assert rc == 0
        pngs = [f for f in os.listdir(out_dir) if f.endswith(".png")]
        assert len(pngs) >= 1

    def test_directory_expansion(self, tmp_path):
        """Passing a directory expands to individual CSVs."""
        pytest.importorskip("matplotlib")
        raw_dir = str(tmp_path / "raw")
        out_dir = str(tmp_path / "figures")
        os.makedirs(raw_dir)

        csv_path = os.path.join(raw_dir, "rk3588-t4.csv")
        with open(csv_path, "w") as f:
            f.write(MICROBENCH_CSV)

        rc = main([raw_dir, "--output-dir", out_dir])
        assert rc == 0
        # Should have created the device table
        tables = [f for f in os.listdir(out_dir) if "rk3588-t4" in f]
        assert len(tables) >= 1

    def test_mixed_dir_and_file(self, tmp_path):
        """Passing both a directory and an explicit file works."""
        raw_dir = str(tmp_path / "raw")
        out_dir = str(tmp_path / "figures")
        os.makedirs(raw_dir)

        with open(os.path.join(raw_dir, "jetson-j1.csv"), "w") as f:
            f.write(MICROBENCH_CSV)

        sc_path = write_temp_csv(SCHEMA_CSV)

        rc = main([raw_dir, sc_path, "--output-dir", out_dir, "--text-only"])
        os.unlink(sc_path)

        assert rc == 0


# ---------------------------------------------------------------------------
# _try_import_matplotlib fallback
# ---------------------------------------------------------------------------


class TestTryImportMatplotlib:
    """Test the matplotlib import guard."""

    def test_returns_module_when_available(self):
        """When matplotlib is installed, returns the pyplot module."""
        pytest.importorskip("matplotlib")
        plt = _try_import_matplotlib()
        assert plt is not None

    def test_returns_none_on_import_error(self, monkeypatch):
        """When matplotlib is not available, returns None."""
        import sys

        monkeypatch.setitem(sys.modules, "matplotlib", None)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
        monkeypatch.setitem(sys.modules, "matplotlib.backends.backend_agg", None)
        result = _try_import_matplotlib()
        assert result is None


class TestGenerateAllMatplotlibUnavailable:
    """Cover generate_all warning when matplotlib is missing (line 601)."""

    def test_warns_when_matplotlib_missing(self, tmp_path, monkeypatch):
        """generate_all(text_only=False) warns when matplotlib unavailable."""
        import sys

        monkeypatch.setitem(sys.modules, "matplotlib", None)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
        monkeypatch.setitem(sys.modules, "matplotlib.backends.backend_agg", None)

        mb_path = write_temp_csv(MICROBENCH_CSV)
        result = generate_all([mb_path], str(tmp_path / "out"), text_only=False)
        os.unlink(mb_path)

        assert any("matplotlib not available" in w for w in result.warnings)


class TestPlotCrossDeviceEdgeCases:
    """Cover _plot_cross_device edge-case branches (lines 725, 757)."""

    @staticmethod
    def _scan_row(gib=2.0):
        return MicrobenchRow(
            model="Qwen3.5-4B",
            kernel="gdn_gated_scan",
            dispatch_path="neon",
            seq=64,
            channels=4096,
            repeats=30,
            p50_us=1000.0,
            p95_us=1100.0,
            spread_pct=5.0,
            gib_per_s=gib,
            gflop_per_s=0.2,
        )

    @staticmethod
    def _decay_row():
        return MicrobenchRow(
            model="Qwen3.5-4B",
            kernel="gdn_cumdecay",
            dispatch_path="neon",
            seq=64,
            channels=4096,
            repeats=30,
            p50_us=1000.0,
            p95_us=1100.0,
            spread_pct=5.0,
            gib_per_s=2.0,
            gflop_per_s=0.2,
        )

    def test_no_scan_data_returns_early(self, tmp_path):
        """_plot_cross_device returns early when no gated_scan rows (line 725)."""
        plt = _try_import_matplotlib()
        # Two devices but neither has gated_scan data
        by_device = {
            "dev-a": [self._decay_row()],
            "dev-b": [self._decay_row()],
        }
        _plot_cross_device(by_device, str(tmp_path / "cross.png"), plt)
        assert not os.path.exists(str(tmp_path / "cross.png"))

    def test_spec_bandwidth_line_drawn(self, tmp_path):
        """_plot_cross_device draws spec line for recognised device (line 757)."""
        pytest.importorskip("matplotlib")
        plt = _try_import_matplotlib()
        # "rk3588" prefix is in DEVICE_SPEC_BANDWIDTH → spec lookup hits
        by_device = {"rk3588-t4": [self._scan_row(gib=3.3)]}
        _plot_cross_device(by_device, str(tmp_path / "cross.png"), plt)
        assert os.path.exists(str(tmp_path / "cross.png"))
