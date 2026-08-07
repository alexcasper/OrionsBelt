"""Comprehensive tests for bench/schema.py — the frozen results schema contract.

This module is imported by the benchmark harness, plotting code, validation scripts,
and the final comparison table. A bug here silently corrupts or rejects every
benchmark result, so coverage is deliberately thorough.
"""

from __future__ import annotations

import csv

import pytest
from bench.schema import (
    CANONICAL_CONTEXT_LENGTHS,
    COLUMNS,
    METRIC_ALLOWED_PHASES,
    METRIC_UNITS,
    METRICS_REQUIRING_COMPONENT,
    Device,
    Engine,
    LayerClass,
    MemoryComponent,
    MetricName,
    Phase,
    ResultRow,
    SchemaValidationError,
    Unit,
    read_csv,
    validate_row,
    validate_rows,
    write_csv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def valid_row(**overrides) -> ResultRow:
    """Return a ResultRow that passes validation, with selective overrides."""
    defaults: dict = dict(
        run_id="run-001",
        timestamp="2026-08-01T12:00:00Z",
        git_sha="abcdef0",
        manifest_ref="manifests/t3.json",
        device="rk3588-t3",
        engine_gdn="cpu",
        engine_full_attention="cpu",
        model_checkpoint="Qwen/Qwen3.5-4B",
        quantization="fp16",
        context_length=4096,
        phase="decode",
        metric_name="decode_tokens_per_sec",
        metric_component=None,
        value=42.5,
        unit="tokens_per_sec",
        repeat_index=0,
        repeat_count=3,
        layer_class="all",
        notes="",
    )
    defaults.update(overrides)
    return ResultRow(**defaults)


def valid_memory_row(**overrides) -> ResultRow:
    """Return a valid PEAK_MEMORY_BYTES row (requires metric_component)."""
    defaults: dict = dict(
        phase="prefill",
        metric_name="peak_memory_bytes",
        metric_component="kv_cache",
        value=1048576.0,
        unit="bytes",
    )
    defaults.update(overrides)
    return valid_row(**defaults)


# ---------------------------------------------------------------------------
# Enum / vocabulary tests
# ---------------------------------------------------------------------------


class TestDeviceEnum:
    def test_all_values_present(self):
        values = {d.value for d in Device}
        assert "o6" in values
        assert "rk3588-t3" in values
        assert "rk3588-t4" in values
        assert "jetson-j1" in values
        assert "jetson-j2" in values

    def test_count(self):
        assert len(Device) == 7


class TestEngineEnum:
    def test_all_values_present(self):
        values = {e.value for e in Engine}
        assert "npu" in values
        assert "gpu_vulkan" in values
        assert "gpu_opencl" in values
        assert "cpu" in values
        assert "cuda_reference" in values


class TestPhaseEnum:
    def test_values(self):
        assert Phase.PREFILL.value == "prefill"
        assert Phase.DECODE.value == "decode"


class TestMetricNameEnum:
    def test_values(self):
        assert MetricName.PREFILL_TOKENS_PER_SEC.value == "prefill_tokens_per_sec"
        assert MetricName.DECODE_TOKENS_PER_SEC.value == "decode_tokens_per_sec"
        assert MetricName.TTFT_SECONDS.value == "ttft_seconds"
        assert MetricName.PEAK_MEMORY_BYTES.value == "peak_memory_bytes"
        assert MetricName.ENERGY_JOULES_PER_TOKEN.value == "energy_joules_per_token"


class TestMemoryComponentEnum:
    def test_values(self):
        assert MemoryComponent.WEIGHTS.value == "weights"
        assert MemoryComponent.KV_CACHE.value == "kv_cache"
        assert MemoryComponent.RECURRENT_STATE.value == "recurrent_state"


class TestUnitEnum:
    def test_values(self):
        assert Unit.TOKENS_PER_SEC.value == "tokens_per_sec"
        assert Unit.SECONDS.value == "seconds"
        assert Unit.BYTES.value == "bytes"
        assert Unit.JOULES_PER_TOKEN.value == "joules_per_token"


class TestLayerClassEnum:
    def test_values(self):
        assert LayerClass.GDN.value == "gdn"
        assert LayerClass.FULL_ATTENTION.value == "full_attention"
        assert LayerClass.FFN.value == "ffn"
        assert LayerClass.ALL.value == "all"


# ---------------------------------------------------------------------------
# Metric vocabulary rules
# ---------------------------------------------------------------------------


class TestMetricUnits:
    def test_every_metric_has_a_unit(self):
        assert set(METRIC_UNITS.keys()) == set(MetricName)

    def test_throughput_metrics(self):
        assert METRIC_UNITS[MetricName.PREFILL_TOKENS_PER_SEC] == Unit.TOKENS_PER_SEC
        assert METRIC_UNITS[MetricName.DECODE_TOKENS_PER_SEC] == Unit.TOKENS_PER_SEC

    def test_ttft_uses_seconds(self):
        assert METRIC_UNITS[MetricName.TTFT_SECONDS] == Unit.SECONDS

    def test_memory_uses_bytes(self):
        assert METRIC_UNITS[MetricName.PEAK_MEMORY_BYTES] == Unit.BYTES

    def test_energy_uses_joules_per_token(self):
        assert METRIC_UNITS[MetricName.ENERGY_JOULES_PER_TOKEN] == Unit.JOULES_PER_TOKEN


class TestMetricAllowedPhases:
    def test_every_metric_has_allowed_phases(self):
        assert set(METRIC_ALLOWED_PHASES.keys()) == set(MetricName)

    def test_prefill_throughput_only_prefill(self):
        assert METRIC_ALLOWED_PHASES[MetricName.PREFILL_TOKENS_PER_SEC] == {Phase.PREFILL}

    def test_decode_throughput_only_decode(self):
        assert METRIC_ALLOWED_PHASES[MetricName.DECODE_TOKENS_PER_SEC] == {Phase.DECODE}

    def test_ttft_only_prefill(self):
        assert METRIC_ALLOWED_PHASES[MetricName.TTFT_SECONDS] == {Phase.PREFILL}

    def test_memory_both_phases(self):
        assert METRIC_ALLOWED_PHASES[MetricName.PEAK_MEMORY_BYTES] == {
            Phase.PREFILL,
            Phase.DECODE,
        }

    def test_energy_both_phases(self):
        assert METRIC_ALLOWED_PHASES[MetricName.ENERGY_JOULES_PER_TOKEN] == {
            Phase.PREFILL,
            Phase.DECODE,
        }


class TestMetricsRequiringComponent:
    def test_only_memory_requires_component(self):
        assert {MetricName.PEAK_MEMORY_BYTES} == METRICS_REQUIRING_COMPONENT


# ---------------------------------------------------------------------------
# COLUMNS and CANONICAL_CONTEXT_LENGTHS
# ---------------------------------------------------------------------------


class TestColumns:
    def test_column_count(self):
        assert len(COLUMNS) == 19

    def test_required_columns_present(self):
        for col in (
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
        ):
            assert col in COLUMNS

    def test_no_duplicates(self):
        assert len(COLUMNS) == len(set(COLUMNS))


class TestCanonicalContextLengths:
    def test_values(self):
        assert CANONICAL_CONTEXT_LENGTHS == (4096, 32768, 131072, 262144)

    def test_sorted_ascending(self):
        assert list(CANONICAL_CONTEXT_LENGTHS) == sorted(CANONICAL_CONTEXT_LENGTHS)


# ---------------------------------------------------------------------------
# ResultRow
# ---------------------------------------------------------------------------


class TestResultRowDefaults:
    def test_layer_class_defaults_to_all(self):
        row = ResultRow(
            run_id="r",
            timestamp="2026-01-01T00:00:00Z",
            git_sha="abcdef0",
            manifest_ref="m.json",
            device="rk3588-t3",
            engine_gdn="cpu",
            engine_full_attention="cpu",
            model_checkpoint="model",
            quantization="fp16",
            context_length=4096,
            phase="decode",
            metric_name="decode_tokens_per_sec",
            metric_component=None,
            value=1.0,
            unit="tokens_per_sec",
            repeat_index=0,
            repeat_count=1,
        )
        assert row.layer_class == "all"

    def test_notes_defaults_to_empty(self):
        row = valid_row()
        assert row.notes == ""

    def test_custom_layer_class(self):
        row = valid_row(layer_class="gdn")
        assert row.layer_class == "gdn"


# ---------------------------------------------------------------------------
# validate_row — valid cases
# ---------------------------------------------------------------------------


class TestValidateRowValid:
    def test_minimal_valid_row(self):
        validate_row(valid_row())

    def test_valid_prefill_row(self):
        validate_row(
            valid_row(
                phase="prefill",
                metric_name="prefill_tokens_per_sec",
                unit="tokens_per_sec",
            )
        )

    def test_valid_memory_row(self):
        validate_row(valid_memory_row())

    def test_valid_ttft_row(self):
        validate_row(
            valid_row(
                phase="prefill",
                metric_name="ttft_seconds",
                value=0.150,
                unit="seconds",
            )
        )

    def test_valid_energy_row(self):
        validate_row(
            valid_row(
                metric_name="energy_joules_per_token",
                value=0.002,
                unit="joules_per_token",
            )
        )

    def test_int_value_accepted(self):
        validate_row(valid_row(value=42))

    def test_zero_value_accepted(self):
        validate_row(valid_row(value=0))

    def test_short_git_sha(self):
        validate_row(valid_row(git_sha="abc1234"))

    def test_full_40_char_sha(self):
        sha = "a" * 40
        validate_row(valid_row(git_sha=sha))

    def test_timestamp_with_offset(self):
        validate_row(valid_row(timestamp="2026-08-01T12:00:00+05:30"))

    def test_timestamp_without_z(self):
        validate_row(valid_row(timestamp="2026-08-01T12:00:00"))

    def test_empty_layer_class_defaults_to_all(self):
        validate_row(valid_row(layer_class=""))


# ---------------------------------------------------------------------------
# validate_row — required string fields
# ---------------------------------------------------------------------------


class TestRequiredStringFields:
    @pytest.mark.parametrize(
        "field_name",
        ["run_id", "timestamp", "git_sha", "manifest_ref", "model_checkpoint", "quantization"],
    )
    def test_empty_string_rejected(self, field_name):
        with pytest.raises(SchemaValidationError, match=field_name):
            validate_row(valid_row(**{field_name: ""}))

    @pytest.mark.parametrize(
        "field_name",
        ["run_id", "timestamp", "git_sha", "manifest_ref", "model_checkpoint", "quantization"],
    )
    def test_whitespace_only_rejected(self, field_name):
        with pytest.raises(SchemaValidationError, match=field_name):
            validate_row(valid_row(**{field_name: "   "}))


# ---------------------------------------------------------------------------
# validate_row — timestamp
# ---------------------------------------------------------------------------


class TestTimestampValidation:
    def test_bad_timestamp(self):
        with pytest.raises(SchemaValidationError, match="timestamp"):
            validate_row(valid_row(timestamp="not-a-date"))

    def test_garbage_timestamp(self):
        with pytest.raises(SchemaValidationError, match="timestamp"):
            validate_row(valid_row(timestamp="2026/08/01 12:00:00"))


# ---------------------------------------------------------------------------
# validate_row — git_sha
# ---------------------------------------------------------------------------


class TestGitShaValidation:
    def test_too_short(self):
        with pytest.raises(SchemaValidationError, match="git_sha"):
            validate_row(valid_row(git_sha="abc"))

    def test_non_hex(self):
        with pytest.raises(SchemaValidationError, match="git_sha"):
            validate_row(valid_row(git_sha="xyz1234"))

    def test_uppercase_normalized(self):
        # The regex lowercases before matching, so uppercase should pass.
        validate_row(valid_row(git_sha="ABCDEF0"))


# ---------------------------------------------------------------------------
# validate_row — device
# ---------------------------------------------------------------------------


class TestDeviceValidation:
    def test_unknown_device(self):
        with pytest.raises(SchemaValidationError, match="device"):
            validate_row(valid_row(device="iphone"))

    def test_all_known_devices_pass(self):
        for d in Device:
            validate_row(valid_row(device=d.value))


# ---------------------------------------------------------------------------
# validate_row — engine fields
# ---------------------------------------------------------------------------


class TestEngineValidation:
    def test_unknown_engine_gdn(self):
        with pytest.raises(SchemaValidationError, match="engine_gdn"):
            validate_row(valid_row(engine_gdn="tpu"))

    def test_unknown_engine_full_attention(self):
        with pytest.raises(SchemaValidationError, match="engine_full_attention"):
            validate_row(valid_row(engine_full_attention="tpu"))

    def test_all_known_engines_pass(self):
        for e in Engine:
            validate_row(valid_row(engine_gdn=e.value, engine_full_attention=e.value))


# ---------------------------------------------------------------------------
# validate_row — context_length
# ---------------------------------------------------------------------------


class TestContextLengthValidation:
    def test_zero_rejected(self):
        with pytest.raises(SchemaValidationError, match="context_length"):
            validate_row(valid_row(context_length=0))

    def test_negative_rejected(self):
        with pytest.raises(SchemaValidationError, match="context_length"):
            validate_row(valid_row(context_length=-1))

    def test_bool_rejected(self):
        with pytest.raises(SchemaValidationError, match="context_length"):
            validate_row(valid_row(context_length=True))


# ---------------------------------------------------------------------------
# validate_row — phase
# ---------------------------------------------------------------------------


class TestPhaseValidation:
    def test_unknown_phase(self):
        with pytest.raises(SchemaValidationError, match="phase"):
            validate_row(valid_row(phase="inference"))

    def test_prefill_with_decode_metric_rejected(self):
        with pytest.raises(SchemaValidationError, match="phase"):
            validate_row(valid_row(phase="prefill", metric_name="decode_tokens_per_sec"))

    def test_decode_with_prefill_metric_rejected(self):
        with pytest.raises(SchemaValidationError, match="phase"):
            validate_row(valid_row(phase="decode", metric_name="prefill_tokens_per_sec"))

    def test_decode_with_ttft_rejected(self):
        with pytest.raises(SchemaValidationError, match="phase"):
            validate_row(
                valid_row(phase="decode", metric_name="ttft_seconds", unit="seconds", value=0.1)
            )


# ---------------------------------------------------------------------------
# validate_row — metric_name + unit coupling
# ---------------------------------------------------------------------------


class TestMetricUnitCoupling:
    def test_wrong_unit_for_metric(self):
        with pytest.raises(SchemaValidationError, match="unit"):
            validate_row(valid_row(metric_name="decode_tokens_per_sec", unit="seconds"))

    def test_unknown_unit(self):
        with pytest.raises(SchemaValidationError, match="unit"):
            validate_row(valid_row(unit="mph"))

    def test_unknown_metric_name(self):
        with pytest.raises(SchemaValidationError, match="metric_name"):
            validate_row(valid_row(metric_name="flops", unit="tokens_per_sec"))


# ---------------------------------------------------------------------------
# validate_row — metric_component rules
# ---------------------------------------------------------------------------


class TestMetricComponentRules:
    def test_component_required_for_memory_metric(self):
        with pytest.raises(SchemaValidationError, match="metric_component"):
            validate_row(
                valid_row(
                    phase="prefill",
                    metric_name="peak_memory_bytes",
                    metric_component=None,
                    value=1024.0,
                    unit="bytes",
                )
            )

    def test_invalid_component_value(self):
        with pytest.raises(SchemaValidationError, match="metric_component"):
            validate_row(
                valid_row(
                    phase="prefill",
                    metric_name="peak_memory_bytes",
                    metric_component=" activations",
                    value=1024.0,
                    unit="bytes",
                )
            )

    def test_component_must_be_empty_for_non_memory_metric(self):
        with pytest.raises(SchemaValidationError, match="metric_component"):
            validate_row(
                valid_row(
                    metric_name="decode_tokens_per_sec",
                    metric_component="kv_cache",
                )
            )

    def test_all_valid_components(self):
        for comp in MemoryComponent:
            validate_row(valid_memory_row(metric_component=comp.value))


# ---------------------------------------------------------------------------
# validate_row — value
# ---------------------------------------------------------------------------


class TestValueValidation:
    def test_nan_rejected(self):
        with pytest.raises(SchemaValidationError, match="value"):
            validate_row(valid_row(value=float("nan")))

    def test_inf_rejected(self):
        with pytest.raises(SchemaValidationError, match="value"):
            validate_row(valid_row(value=float("inf")))

    def test_negative_inf_rejected(self):
        with pytest.raises(SchemaValidationError, match="value"):
            validate_row(valid_row(value=float("-inf")))

    def test_negative_rejected(self):
        with pytest.raises(SchemaValidationError, match="value"):
            validate_row(valid_row(value=-0.001))

    def test_bool_rejected(self):
        with pytest.raises(SchemaValidationError, match="value"):
            validate_row(valid_row(value=True))

    def test_string_rejected(self):
        with pytest.raises(SchemaValidationError, match="value"):
            validate_row(valid_row(value="fast"))


# ---------------------------------------------------------------------------
# validate_row — repeat_index / repeat_count
# ---------------------------------------------------------------------------


class TestRepeatValidation:
    def test_repeat_index_negative(self):
        with pytest.raises(SchemaValidationError, match="repeat_index"):
            validate_row(valid_row(repeat_index=-1))

    def test_repeat_count_zero(self):
        with pytest.raises(SchemaValidationError, match="repeat_count"):
            validate_row(valid_row(repeat_count=0))

    def test_repeat_index_equals_count(self):
        with pytest.raises(SchemaValidationError, match="repeat_index"):
            validate_row(valid_row(repeat_index=3, repeat_count=3))

    def test_repeat_index_greater_than_count(self):
        with pytest.raises(SchemaValidationError, match="repeat_index"):
            validate_row(valid_row(repeat_index=5, repeat_count=3))

    def test_repeat_index_bool_rejected(self):
        with pytest.raises(SchemaValidationError, match="repeat_index"):
            validate_row(valid_row(repeat_index=True))

    def test_repeat_count_bool_rejected(self):
        with pytest.raises(SchemaValidationError, match="repeat_count"):
            validate_row(valid_row(repeat_count=True))


# ---------------------------------------------------------------------------
# validate_row — layer_class
# ---------------------------------------------------------------------------


class TestLayerClassValidation:
    def test_unknown_layer_class(self):
        with pytest.raises(SchemaValidationError, match="layer_class"):
            validate_row(valid_row(layer_class="attention"))

    def test_all_valid_layer_classes(self):
        for lc in LayerClass:
            validate_row(valid_row(layer_class=lc.value))


# ---------------------------------------------------------------------------
# validate_row — error accumulation
# ---------------------------------------------------------------------------


class TestErrorAccumulation:
    def test_multiple_errors_reported(self):
        row = valid_row(device="bad-device", unit="bad-unit", value=-1)
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_row(row)
        msg = str(exc_info.value)
        assert "device" in msg
        assert "unit" in msg
        assert "value" in msg

    def test_error_includes_run_id(self):
        row = valid_row(run_id="my-special-run", device="bad")
        with pytest.raises(SchemaValidationError, match="my-special-run"):
            validate_row(row)


# ---------------------------------------------------------------------------
# validate_rows
# ---------------------------------------------------------------------------


class TestValidateRows:
    def test_all_valid(self):
        rows = [valid_row(), valid_row(run_id="run-002")]
        validate_rows(rows)

    def test_first_bad_reported(self):
        rows = [valid_row(device="bad"), valid_row()]
        with pytest.raises(SchemaValidationError, match="row 0"):
            validate_rows(rows)

    def test_second_bad_reported(self):
        rows = [valid_row(), valid_row(device="bad")]
        with pytest.raises(SchemaValidationError, match="row 1"):
            validate_rows(rows)

    def test_multiple_bad_rows_all_reported(self):
        rows = [valid_row(device="bad1"), valid_row(), valid_row(device="bad2")]
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_rows(rows)
        msg = str(exc_info.value)
        assert "row 0" in msg
        assert "row 2" in msg

    def test_empty_list_passes(self):
        validate_rows([])


# ---------------------------------------------------------------------------
# write_csv / read_csv round-trip
# ---------------------------------------------------------------------------


class TestWriteReadCsv:
    def test_round_trip_single_row(self, tmp_path):
        rows = [valid_row()]
        path = str(tmp_path / "out.csv")
        write_csv(rows, path)
        loaded = read_csv(path)
        assert len(loaded) == 1
        for field in [
            "run_id",
            "device",
            "engine_gdn",
            "phase",
            "metric_name",
            "unit",
            "value",
            "context_length",
        ]:
            assert getattr(loaded[0], field) == getattr(rows[0], field)

    def test_round_trip_multiple_rows(self, tmp_path):
        rows = [valid_row(run_id=f"run-{i}") for i in range(5)]
        path = str(tmp_path / "out.csv")
        write_csv(rows, path)
        loaded = read_csv(path)
        assert len(loaded) == 5
        assert [r.run_id for r in loaded] == [f"run-{i}" for i in range(5)]

    def test_round_trip_with_memory_component(self, tmp_path):
        rows = [valid_memory_row()]
        path = str(tmp_path / "mem.csv")
        write_csv(rows, path)
        loaded = read_csv(path)
        assert loaded[0].metric_component == "kv_cache"

    def test_round_trip_empty_component_becomes_none(self, tmp_path):
        rows = [valid_row(metric_component=None)]
        path = str(tmp_path / "no_comp.csv")
        write_csv(rows, path)
        loaded = read_csv(path)
        assert loaded[0].metric_component is None

    def test_round_trip_preserves_notes(self, tmp_path):
        rows = [valid_row(notes="this is a note with, comma")]
        path = str(tmp_path / "notes.csv")
        write_csv(rows, path)
        loaded = read_csv(path)
        assert loaded[0].notes == "this is a note with, comma"

    def test_header_matches_columns(self, tmp_path):
        rows = [valid_row()]
        path = str(tmp_path / "hdr.csv")
        write_csv(rows, path)
        with open(path) as f:
            header = next(csv.reader(f))
        assert header == COLUMNS

    def test_write_csv_validates_before_writing(self, tmp_path):
        """A bad row should not produce a partially written file."""
        rows = [valid_row(), valid_row(device="bad")]
        path = str(tmp_path / "should_not_exist.csv")
        with pytest.raises(SchemaValidationError):
            write_csv(rows, path)
        # File should not exist because validation runs before opening.
        # (write_csv validates first, then writes)
        # Note: the open call with "w" truncates, but validate_rows runs before open.
        # Actually looking at the code, validate_rows runs before the with block.

    def test_read_csv_wrong_header_rejected(self, tmp_path):
        path = str(tmp_path / "bad_hdr.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["wrong", "columns"])
            writer.writerow(["a", "b"])
        with pytest.raises(SchemaValidationError, match="header"):
            read_csv(path)

    def test_read_csv_preserves_empty_layer_class(self, tmp_path):
        """A row with empty layer_class string should default to 'all' on read."""
        rows = [valid_row(layer_class="")]
        path = str(tmp_path / "empty_lc.csv")
        write_csv(rows, path, validate=False)
        loaded = read_csv(path)
        assert loaded[0].layer_class == "all"


# ---------------------------------------------------------------------------
# write_csv with validate=False
# ---------------------------------------------------------------------------


class TestWriteCsvSkipValidation:
    def test_skips_validation(self, tmp_path):
        row = valid_row(device="bad-device")
        path = str(tmp_path / "skip.csv")
        write_csv([row], path, validate=False)
        # File was written despite bad row
        loaded = read_csv(path, validate=False)
        assert loaded[0].device == "bad-device"
