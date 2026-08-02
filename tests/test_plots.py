"""Tests for bench/plots.py — plot and table generation from committed CSVs."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

# Ensure bench/ is importable
_BENCH_DIR = Path(__file__).resolve().parent.parent / "bench"
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_BENCH_DIR))

from bench.plots import (  # noqa: E402
    DeviceBenchRow,
    SchemaRow,
    _detect_format,
    _human_bytes,
    _human_throughput,
    _percentile,
    aggregate_schema_rows,
    generate_device_bench_table,
    generate_schema_table,
    load_all_csvs,
    load_csv,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """Write a CSV file with the given header and rows."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


SCHEMA_HEADER = [
    "run_id", "timestamp", "git_sha", "manifest_ref", "device",
    "engine_gdn", "engine_full_attention", "model_checkpoint",
    "quantization", "context_length", "phase", "metric_name",
    "metric_component", "value", "unit", "repeat_index", "repeat_count",
    "layer_class", "notes",
]

DEVICE_BENCH_HEADER = [
    "model", "kernel", "dispatch_path", "seq", "channels",
    "repeats", "p50_us", "p95_us", "spread_pct",
    "gib_per_s_p50", "gflop_per_s_p50",
]


def _make_schema_row(
    run_id="run1",
    device="o6",
    model="Qwen/Qwen3.5-4B@abc1234",
    quant="fp16",
    ctx=4096,
    phase="prefill",
    metric="prefill_tokens_per_sec",
    component="",
    value=1000.0,
    rep_idx=0,
    rep_count=5,
) -> list[str]:
    """Build a single schema-format CSV row."""
    return [
        run_id, "2026-08-02T12:00:00Z", "abc1234",
        "results/manifests/run1.json", device, "cpu", "npu",
        model, quant, str(ctx), phase, metric, component,
        str(value), _unit_for_metric(metric), str(rep_idx), str(rep_count),
        "all", "",
    ]


def _unit_for_metric(metric: str) -> str:
    units = {
        "prefill_tokens_per_sec": "tokens_per_sec",
        "decode_tokens_per_sec": "tokens_per_sec",
        "ttft_seconds": "seconds",
        "peak_memory_bytes": "bytes",
        "energy_joules_per_token": "joules_per_token",
    }
    return units.get(metric, "tokens_per_sec")


def _make_device_bench_row(
    model="Qwen3.5-4B",
    kernel="gdn_gated_scan",
    dispatch="neon",
    seq=64,
    channels=4096,
    repeats=30,
) -> list[str]:
    return [
        model, kernel, dispatch, str(seq), str(channels), str(repeats),
        "1514.145", "3832.468", "153.1", "1.96", "0.35",
    ]


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


class TestDetectFormat:
    def test_schema_format(self):
        assert _detect_format(list(SCHEMA_HEADER)) == "schema"

    def test_device_bench_format(self):
        assert _detect_format(list(DEVICE_BENCH_HEADER)) == "device_bench"

    def test_unknown_format(self):
        assert _detect_format(["foo", "bar", "baz"]) == "unknown"

    def test_empty(self):
        assert _detect_format([]) == "unknown"


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


class TestLoadCsv:
    def test_load_schema_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        _write_csv(csv_path, SCHEMA_HEADER, [_make_schema_row()])
        schema, device = load_csv(csv_path)
        assert len(schema) == 1
        assert len(device) == 0
        assert schema[0].device == "o6"
        assert schema[0].value == 1000.0
        assert schema[0].context_length == 4096

    def test_load_device_bench_csv(self, tmp_path):
        csv_path = tmp_path / "rk3588_big.csv"
        _write_csv(csv_path, DEVICE_BENCH_HEADER, [_make_device_bench_row()])
        schema, device = load_csv(csv_path)
        assert len(schema) == 0
        assert len(device) == 1
        assert device[0].model == "Qwen3.5-4B"
        assert device[0].kernel == "gdn_gated_scan"
        assert device[0].gib_per_s_p50 == 1.96
        assert device[0].source_file == "rk3588_big"

    def test_load_empty_csv(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        _write_csv(csv_path, ["col1", "col2"], [])
        schema, device = load_csv(csv_path)
        assert len(schema) == 0
        assert len(device) == 0

    def test_load_nonexistent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_csv(tmp_path / "missing.csv")

    def test_device_label_big(self):
        row = DeviceBenchRow(
            model="m", kernel="k", dispatch_path="d", seq=1, channels=1,
            repeats=1, p50_us=1, p95_us=1, spread_pct=1,
            gib_per_s_p50=1, gflop_per_s_p50=1, source_file="rk3588_t3_big",
        )
        assert "big" in row.device_label

    def test_device_label_little(self):
        row = DeviceBenchRow(
            model="m", kernel="k", dispatch_path="d", seq=1, channels=1,
            repeats=1, p50_us=1, p95_us=1, spread_pct=1,
            gib_per_s_p50=1, gflop_per_s_p50=1, source_file="rk3588_t3_little",
        )
        assert "little" in row.device_label


class TestLoadAllCsvs:
    def test_load_mixed_formats(self, tmp_path):
        _write_csv(
            tmp_path / "schema1.csv", SCHEMA_HEADER,
            [_make_schema_row()],
        )
        _write_csv(
            tmp_path / "device1.csv", DEVICE_BENCH_HEADER,
            [_make_device_bench_row()],
        )
        schema, device = load_all_csvs(tmp_path)
        assert len(schema) == 1
        assert len(device) == 1

    def test_load_empty_dir(self, tmp_path):
        schema, device = load_all_csvs(tmp_path)
        assert len(schema) == 0
        assert len(device) == 0

    def test_load_nonexistent_dir(self, tmp_path):
        schema, device = load_all_csvs(tmp_path / "nope")
        assert len(schema) == 0
        assert len(device) == 0


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_p50_even_count(self):
        assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0

    def test_p50_odd_count(self):
        assert _percentile([1.0, 2.0, 3.0], 50) == 2.0

    def test_p95_with_30_samples(self):
        vals = list(range(1, 31))
        # Nearest-rank: ceil(0.95 * 30) = 29th value (1-indexed) = 29
        assert _percentile(vals, 95) == 29

    def test_p95_with_10_samples(self):
        vals = list(range(1, 11))
        # ceil(0.95 * 10) = 10th value = 10
        assert _percentile(vals, 95) == 10

    def test_empty(self):
        assert _percentile([], 50) == 0.0

    def test_single_value(self):
        assert _percentile([42.0], 50) == 42.0
        assert _percentile([42.0], 95) == 42.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_basic_grouping(self):
        rows = [
            SchemaRow(
                run_id="r", device="o6", model_checkpoint="m@sha",
                quantization="fp16", context_length=4096, phase="prefill",
                metric_name="prefill_tokens_per_sec", metric_component="",
                value=1000.0 + i, repeat_index=i, repeat_count=5,
            )
            for i in range(5)
        ]
        agg = aggregate_schema_rows(rows)
        assert len(agg) == 1
        assert agg[0].p50 == 1002.0  # ceil(0.5 * 5) = 3rd → index 2
        assert agg[0].n_repeats == 5

    def test_multiple_groups(self):
        rows = []
        for ctx in [4096, 32768]:
            for phase in ["prefill", "decode"]:
                metric = f"{phase}_tokens_per_sec"
                for i in range(10):
                    rows.append(SchemaRow(
                        run_id="r", device="o6", model_checkpoint="m@sha",
                        quantization="fp16", context_length=ctx, phase=phase,
                        metric_name=metric, metric_component="",
                        value=float(100 * (ctx // 4096) + i),
                        repeat_index=i, repeat_count=10,
                    ))
        agg = aggregate_schema_rows(rows)
        assert len(agg) == 4

    def test_memory_component_grouping(self):
        rows = []
        for comp in ["weights", "kv_cache", "recurrent_state"]:
            for i in range(5):
                rows.append(SchemaRow(
                    run_id="r", device="o6", model_checkpoint="m@sha",
                    quantization="fp16", context_length=4096, phase="prefill",
                    metric_name="peak_memory_bytes", metric_component=comp,
                    value=float(1e9 + i * 1000),  # ~1 GiB
                    repeat_index=i, repeat_count=5,
                ))
        agg = aggregate_schema_rows(rows)
        assert len(agg) == 3
        components = {m.metric_component for m in agg}
        assert components == {"weights", "kv_cache", "recurrent_state"}

    def test_normalized_spread(self):
        rows = [
            SchemaRow(
                run_id="r", device="o6", model_checkpoint="m@sha",
                quantization="fp16", context_length=4096, phase="prefill",
                metric_name="prefill_tokens_per_sec", metric_component="",
                value=v, repeat_index=i, repeat_count=10,
            )
            for i, v in enumerate(range(100, 110))
        ]
        agg = aggregate_schema_rows(rows)
        assert agg[0].p50 == 104
        # p95 = 10th value (ceil(0.95*10)) = 109
        assert agg[0].p95 == 109
        assert agg[0].spread == 5
        assert agg[0].normalized_spread == pytest.approx(5 / 104)

    def test_empty(self):
        assert aggregate_schema_rows([]) == []


# ---------------------------------------------------------------------------
# Human formatting helpers
# ---------------------------------------------------------------------------


class TestHumanFormat:
    def test_bytes_gib(self):
        assert "GiB" in _human_bytes(1024 ** 3 * 2)

    def test_bytes_mib(self):
        assert "MiB" in _human_bytes(1024 ** 2 * 100)

    def test_bytes_kib(self):
        assert "KiB" in _human_bytes(1024 * 500)

    def test_bytes_b(self):
        assert "B" in _human_bytes(512)

    def test_throughput_high(self):
        assert "tok/s" in _human_throughput(5000, "prefill_tokens_per_sec")

    def test_throughput_low(self):
        assert "tok/s" in _human_throughput(14.2, "decode_tokens_per_sec")

    def test_ttft_ms(self):
        assert "ms" in _human_throughput(0.183, "ttft_seconds")

    def test_ttft_seconds(self):
        assert "s" in _human_throughput(5.2, "ttft_seconds")


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------


class TestSchemaTable:
    def test_generate_with_data(self, tmp_path):
        rows = []
        for ctx in [4096, 32768]:
            for i in range(5):
                rows.append(SchemaRow(
                    run_id="r", device="o6", model_checkpoint="Qwen/Qwen3.5-4B@sha",
                    quantization="fp16", context_length=ctx, phase="prefill",
                    metric_name="prefill_tokens_per_sec", metric_component="",
                    value=1000.0 * (4096 / ctx) + i,
                    repeat_index=i, repeat_count=5,
                ))
        for comp in ["weights", "kv_cache", "recurrent_state"]:
            for i in range(5):
                rows.append(SchemaRow(
                    run_id="r", device="o6", model_checkpoint="Qwen/Qwen3.5-4B@sha",
                    quantization="fp16", context_length=4096, phase="prefill",
                    metric_name="peak_memory_bytes", metric_component=comp,
                    value=1e9 + i * 100,
                    repeat_index=i, repeat_count=5,
                ))

        agg = aggregate_schema_rows(rows)
        out = tmp_path / "table.md"
        assert generate_schema_table(agg, out)
        content = out.read_text()
        assert "o6" in content
        assert "prefill" in content.lower()
        assert "Qwen3.5-4B" in content
        assert "weights" in content
        assert "kv_cache" in content
        assert "recurrent_state" in content

    def test_generate_empty(self, tmp_path):
        out = tmp_path / "table.md"
        assert not generate_schema_table([], out)
        # File may or may not be created, but function returns False
        assert not out.exists()

    def test_table_includes_repeat_count(self, tmp_path):
        rows = [
            SchemaRow(
                run_id="r", device="o6", model_checkpoint="m@sha",
                quantization="fp16", context_length=4096, phase="decode",
                metric_name="decode_tokens_per_sec", metric_component="",
                value=14.2, repeat_index=i, repeat_count=30,
            )
            for i in range(30)
        ]
        agg = aggregate_schema_rows(rows)
        out = tmp_path / "table.md"
        generate_schema_table(agg, out)
        content = out.read_text()
        assert "30" in content


class TestDeviceBenchTable:
    def test_generate_with_data(self, tmp_path):
        rows = [
            DeviceBenchRow(
                model="Qwen3.5-4B", kernel="gdn_gated_scan", dispatch_path="neon",
                seq=64, channels=4096, repeats=30,
                p50_us=1514.0, p95_us=3832.0, spread_pct=153.1,
                gib_per_s_p50=1.96, gflop_per_s_p50=0.35,
                source_file="rk3588_t3_big",
            ),
            DeviceBenchRow(
                model="Qwen3.5-0.8B", kernel="gdn_cumdecay", dispatch_path="neon",
                seq=64, channels=2048, repeats=30,
                p50_us=198.0, p95_us=204.0, spread_pct=3.1,
                gib_per_s_p50=4.92, gflop_per_s_p50=0.66,
                source_file="rk3588_t3_big",
            ),
        ]
        out = tmp_path / "kernel.md"
        assert generate_device_bench_table(rows, out)
        content = out.read_text()
        assert "gdn_gated_scan" in content
        assert "Qwen3.5-4B" in content
        assert "1.96" in content
        assert "big" in content.lower()

    def test_generate_empty(self, tmp_path):
        out = tmp_path / "kernel.md"
        assert not generate_device_bench_table([], out)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


class TestCLI:
    def test_list_with_device_bench_data(self, tmp_path, capsys):
        _write_csv(
            tmp_path / "dev.csv", DEVICE_BENCH_HEADER,
            [_make_device_bench_row()],
        )
        rc = main(["--raw-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"), "--list"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "device_bench" in captured.out
        assert "1" in captured.out  # 1 device-bench row

    def test_list_empty_dir(self, tmp_path, capsys):
        rc = main(["--raw-dir", str(tmp_path), "--list"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "0" in captured.out
        assert "(none)" in captured.out

    def test_tables_only(self, tmp_path):
        _write_csv(
            tmp_path / "dev.csv", DEVICE_BENCH_HEADER,
            [_make_device_bench_row()],
        )
        out_dir = tmp_path / "figures"
        rc = main([
            "--raw-dir", str(tmp_path),
            "--output-dir", str(out_dir),
            "--format", "tables",
        ])
        assert rc == 0
        assert (out_dir / "kernel_table.md").exists()

    def test_no_plots_flag(self, tmp_path):
        _write_csv(
            tmp_path / "dev.csv", DEVICE_BENCH_HEADER,
            [_make_device_bench_row()],
        )
        out_dir = tmp_path / "figures"
        rc = main([
            "--raw-dir", str(tmp_path),
            "--output-dir", str(out_dir),
            "--no-plots",
        ])
        assert rc == 0
        assert (out_dir / "kernel_table.md").exists()

    def test_schema_and_device_mixed(self, tmp_path):
        _write_csv(
            tmp_path / "schema.csv", SCHEMA_HEADER,
            [_make_schema_row(metric="prefill_tokens_per_sec", value=800.0)],
        )
        _write_csv(
            tmp_path / "dev.csv", DEVICE_BENCH_HEADER,
            [_make_device_bench_row()],
        )
        out_dir = tmp_path / "figures"
        rc = main([
            "--raw-dir", str(tmp_path),
            "--output-dir", str(out_dir),
            "--format", "tables",
        ])
        assert rc == 0
        assert (out_dir / "results_table.md").exists()
        assert (out_dir / "kernel_table.md").exists()

    def test_no_data_graceful(self, tmp_path, capsys):
        rc = main([
            "--raw-dir", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--format", "tables",
        ])
        captured = capsys.readouterr()
        assert rc == 0
        assert "No output generated" in captured.out
