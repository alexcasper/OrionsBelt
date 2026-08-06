"""Comprehensive tests for bench/schema.py — validation, serialization, and round-trip.

Covers the core data-integrity layer: validate_row, validate_rows, write_csv,
read_csv, and all edge cases in the frozen results schema contract.  The existing
test_schema_conformance.py only checks enum values and CSV header shape; this file
exercises the actual validation logic that prevents bad data from silently
corrupting downstream analysis.
"""

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench.schema import (  # noqa: E402
    COLUMNS,
    METRIC_ALLOWED_PHASES,
    METRIC_UNITS,
    METRICS_REQUIRING_COMPONENT,
    MetricName,
    Phase,
    ResultRow,
    SchemaValidationError,
    Unit,
    _parse_iso8601,
    read_csv,
    validate_row,
    validate_rows,
    write_csv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_row(**overrides):
    """Create a fully valid ResultRow; override any field via kwargs."""
    defaults = dict(
        run_id="test_run_001",
        timestamp="2026-08-04T12:00:00Z",
        git_sha="abc1234",
        manifest_ref="test_run_001.json",
        device="o6",
        engine_gdn="cpu",
        engine_full_attention="cpu",
        model_checkpoint="Qwen3.5-4B",
        quantization="fp32",
        context_length=4096,
        phase="prefill",
        metric_name="prefill_tokens_per_sec",
        metric_component=None,
        value=42.0,
        unit="tokens_per_sec",
        repeat_index=0,
        repeat_count=5,
        layer_class="all",
        notes="",
    )
    defaults.update(overrides)
    return ResultRow(**defaults)


def make_memory_row(**overrides):
    """Create a valid ResultRow with peak_memory_bytes (requires metric_component)."""
    return make_row(
        metric_name="peak_memory_bytes",
        metric_component="weights",
        value=1_073_741_824.0,
        unit="bytes",
        **overrides,
    )


# ---------------------------------------------------------------------------
# _parse_iso8601
# ---------------------------------------------------------------------------


class TestParseISO8601:
    def test_valid_z_suffix(self):
        _parse_iso8601("2026-08-04T12:00:00Z")

    def test_valid_with_offset(self):
        _parse_iso8601("2026-08-04T12:00:00+00:00")

    def test_valid_no_tz(self):
        _parse_iso8601("2026-08-04T12:00:00")

    def test_valid_date_only(self):
        _parse_iso8601("2026-08-04")

    def test_invalid_garbage(self):
        with pytest.raises(ValueError):
            _parse_iso8601("not-a-date")

    def test_invalid_month(self):
        with pytest.raises(ValueError):
            _parse_iso8601("2026-13-04T12:00:00Z")

    def test_invalid_empty_string(self):
        with pytest.raises(ValueError):
            _parse_iso8601("")


# ---------------------------------------------------------------------------
# validate_row: required string fields
# ---------------------------------------------------------------------------


class TestValidateRequiredStrings:
    @pytest.mark.parametrize(
        "field",
        [
            "run_id",
            "timestamp",
            "git_sha",
            "manifest_ref",
            "model_checkpoint",
            "quantization",
        ],
    )
    def test_empty_string_rejected(self, field):
        row = make_row(**{field: ""})
        with pytest.raises(SchemaValidationError, match=field):
            validate_row(row)

    @pytest.mark.parametrize(
        "field",
        [
            "run_id",
            "timestamp",
            "git_sha",
            "manifest_ref",
            "model_checkpoint",
            "quantization",
        ],
    )
    def test_whitespace_only_rejected(self, field):
        row = make_row(**{field: "   "})
        with pytest.raises(SchemaValidationError, match=field):
            validate_row(row)

    @pytest.mark.parametrize(
        "field",
        [
            "run_id",
            "timestamp",
            "git_sha",
            "manifest_ref",
            "model_checkpoint",
            "quantization",
        ],
    )
    def test_none_rejected(self, field):
        row = make_row(**{field: None})
        with pytest.raises(SchemaValidationError, match=field):
            validate_row(row)


# ---------------------------------------------------------------------------
# validate_row: timestamp format
# ---------------------------------------------------------------------------


class TestValidateTimestamp:
    def test_valid_iso8601_z(self):
        validate_row(make_row(timestamp="2026-08-04T12:00:00Z"))

    def test_valid_iso8601_offset(self):
        validate_row(make_row(timestamp="2026-08-04T12:00:00-08:00"))

    def test_invalid_timestamp(self):
        row = make_row(timestamp="garbage")
        with pytest.raises(
            SchemaValidationError, match="timestamp.*not valid ISO 8601"
        ):
            validate_row(row)


# ---------------------------------------------------------------------------
# validate_row: git SHA
# ---------------------------------------------------------------------------


class TestValidateGitSha:
    @pytest.mark.parametrize(
        "sha",
        [
            "abc1234",  # 7 chars (short)
            "a" * 40,  # 40 chars (full)
            "ABC1234",  # uppercase is normalized
            "0123456789ab",  # 12 chars
        ],
    )
    def test_valid_sha(self, sha):
        validate_row(make_row(git_sha=sha))

    @pytest.mark.parametrize(
        "sha",
        [
            "abc123",  # too short (6 chars)
            "g" * 7,  # non-hex
            "xyz",  # non-hex
            "abc12345 ",  # trailing space
        ],
    )
    def test_invalid_sha(self, sha):
        with pytest.raises(SchemaValidationError, match="git_sha"):
            validate_row(make_row(git_sha=sha))


# ---------------------------------------------------------------------------
# validate_row: device, engine, phase, metric_name enums
# ---------------------------------------------------------------------------


class TestValidateEnumFields:
    def test_valid_device_o6(self):
        validate_row(make_row(device="o6"))

    def test_valid_device_generic_aarch64(self):
        validate_row(make_row(device="generic_aarch64"))

    def test_invalid_device(self):
        with pytest.raises(SchemaValidationError, match="device.*not in allowed"):
            validate_row(make_row(device="rpi4"))

    def test_valid_engine_gdn(self):
        validate_row(make_row(engine_gdn="npu"))

    def test_valid_engine_full_attention(self):
        validate_row(make_row(engine_full_attention="gpu_vulkan"))

    def test_invalid_engine(self):
        with pytest.raises(SchemaValidationError, match="engine_gdn.*not in allowed"):
            validate_row(make_row(engine_gdn="tpu"))

    def test_valid_phase_decode(self):
        validate_row(
            make_row(
                phase="decode",
                metric_name="decode_tokens_per_sec",
                unit="tokens_per_sec",
            )
        )

    def test_invalid_phase(self):
        with pytest.raises(SchemaValidationError, match="phase.*not in allowed"):
            validate_row(make_row(phase="inference"))

    def test_invalid_metric_name(self):
        with pytest.raises(SchemaValidationError, match="metric_name.*not in allowed"):
            validate_row(make_row(metric_name="tokens_per_second"))


# ---------------------------------------------------------------------------
# validate_row: metric/unit consistency
# ---------------------------------------------------------------------------


class TestMetricUnitConsistency:
    def test_prefill_tokens_correct_unit(self):
        validate_row(make_row(
            metric_name="prefill_tokens_per_sec", unit="tokens_per_sec"
        ))

    def test_decode_tokens_correct_unit(self):
        validate_row(make_row(
            phase="decode", metric_name="decode_tokens_per_sec", unit="tokens_per_sec"
        ))

    def test_ttft_correct_unit(self):
        validate_row(make_row(
            metric_name="ttft_seconds", unit="seconds"
        ))

    def test_memory_correct_unit(self):
        validate_row(make_memory_row())

    def test_energy_correct_unit(self):
        validate_row(make_row(
            metric_name="energy_joules_per_token", unit="joules_per_token"
        ))

    def test_wrong_unit_for_metric(self):
        with pytest.raises(SchemaValidationError, match="unit.*requires unit"):
            validate_row(make_row(
                metric_name="prefill_tokens_per_sec", unit="seconds"
            ))

    def test_invalid_unit_value(self):
        with pytest.raises(SchemaValidationError, match="unit.*not in allowed"):
            validate_row(make_row(unit="ms"))


# ---------------------------------------------------------------------------
# validate_row: metric/phase consistency
# ---------------------------------------------------------------------------


class TestMetricPhaseConsistency:
    def test_prefill_metric_in_decode_phase_rejected(self):
        with pytest.raises(SchemaValidationError, match="phase.*only valid for phase"):
            validate_row(make_row(
                phase="decode",
                metric_name="prefill_tokens_per_sec",
                unit="tokens_per_sec",
            ))

    def test_decode_metric_in_prefill_phase_rejected(self):
        with pytest.raises(SchemaValidationError, match="phase.*only valid for phase"):
            validate_row(make_row(
                phase="prefill",
                metric_name="decode_tokens_per_sec",
                unit="tokens_per_sec",
            ))

    def test_ttft_in_decode_rejected(self):
        with pytest.raises(SchemaValidationError, match="phase.*only valid for phase"):
            validate_row(make_row(
                phase="decode",
                metric_name="ttft_seconds",
                unit="seconds",
            ))

    def test_memory_metric_in_decode_phase_ok(self):
        """peak_memory_bytes is allowed in both phases."""
        validate_row(make_memory_row(phase="decode"))

    def test_energy_metric_in_both_phases_ok(self):
        """energy_joules_per_token is allowed in both phases."""
        validate_row(make_row(
            phase="prefill",
            metric_name="energy_joules_per_token",
            unit="joules_per_token",
        ))
        validate_row(make_row(
            phase="decode",
            metric_name="energy_joules_per_token",
            unit="joules_per_token",
        ))


# ---------------------------------------------------------------------------
# validate_row: metric_component rules
# ---------------------------------------------------------------------------


class TestMetricComponentRules:
    def test_memory_requires_component(self):
        row = make_row(
            metric_name="peak_memory_bytes",
            metric_component=None,
            value=1024.0,
            unit="bytes",
        )
        with pytest.raises(SchemaValidationError, match="metric_component.*required"):
            validate_row(row)

    def test_memory_component_empty_string_rejected(self):
        row = make_row(
            metric_name="peak_memory_bytes",
            metric_component="",
            value=1024.0,
            unit="bytes",
        )
        with pytest.raises(SchemaValidationError, match="metric_component.*required"):
            validate_row(row)

    def test_memory_component_whitespace_rejected(self):
        row = make_row(
            metric_name="peak_memory_bytes",
            metric_component="   ",
            value=1024.0,
            unit="bytes",
        )
        with pytest.raises(SchemaValidationError, match="metric_component.*required"):
            validate_row(row)

    @pytest.mark.parametrize("component", ["weights", "kv_cache", "recurrent_state"])
    def test_valid_memory_components(self, component):
        validate_row(make_row(
            metric_name="peak_memory_bytes",
            metric_component=component,
            value=1024.0,
            unit="bytes",
        ))

    def test_invalid_memory_component(self):
        with pytest.raises(
            SchemaValidationError, match="metric_component.*not in allowed"
        ):
            validate_row(make_row(
                metric_name="peak_memory_bytes",
                metric_component="activations",
                value=1024.0,
                unit="bytes",
            ))

    def test_non_memory_metric_with_component_rejected(self):
        """Non-memory metrics must NOT have a component."""
        with pytest.raises(
            SchemaValidationError, match="metric_component.*must be empty"
        ):
            validate_row(make_row(
                metric_name="prefill_tokens_per_sec",
                metric_component="weights",
                unit="tokens_per_sec",
            ))

    def test_non_memory_metric_with_none_component_ok(self):
        validate_row(make_row(metric_component=None))

    def test_non_memory_metric_with_empty_string_component_ok(self):
        validate_row(make_row(metric_component=""))


# ---------------------------------------------------------------------------
# validate_row: value constraints
# ---------------------------------------------------------------------------


class TestValidateValue:
    @pytest.mark.parametrize("value", [0, 0.0, 1, 100.5, 1e9])
    def test_valid_values(self, value):
        validate_row(make_row(value=value))

    def test_negative_value_rejected(self):
        with pytest.raises(SchemaValidationError, match="value.*must be >= 0"):
            validate_row(make_row(value=-1.0))

    def test_nan_rejected(self):
        with pytest.raises(SchemaValidationError, match="value.*must be finite"):
            validate_row(make_row(value=float("nan")))

    def test_inf_rejected(self):
        with pytest.raises(SchemaValidationError, match="value.*must be finite"):
            validate_row(make_row(value=float("inf")))

    def test_neg_inf_rejected(self):
        with pytest.raises(SchemaValidationError, match="value.*must be finite"):
            validate_row(make_row(value=float("-inf")))

    def test_bool_rejected(self):
        with pytest.raises(SchemaValidationError, match="value.*must be a number"):
            validate_row(make_row(value=True))

    def test_string_rejected(self):
        with pytest.raises(SchemaValidationError, match="value.*must be a number"):
            validate_row(make_row(value="42"))


# ---------------------------------------------------------------------------
# validate_row: context_length constraints
# ---------------------------------------------------------------------------


class TestValidateContextLength:
    @pytest.mark.parametrize("ctx", [1, 4096, 32768, 131072, 262144])
    def test_valid_context_lengths(self, ctx):
        validate_row(make_row(context_length=ctx))

    def test_zero_rejected(self):
        with pytest.raises(SchemaValidationError, match="context_length.*must be > 0"):
            validate_row(make_row(context_length=0))

    def test_negative_rejected(self):
        with pytest.raises(SchemaValidationError, match="context_length.*must be > 0"):
            validate_row(make_row(context_length=-1))

    def test_float_rejected(self):
        with pytest.raises(
            SchemaValidationError, match="context_length.*must be an int"
        ):
            validate_row(make_row(context_length=4096.0))

    def test_bool_rejected(self):
        with pytest.raises(
            SchemaValidationError, match="context_length.*must be an int"
        ):
            validate_row(make_row(context_length=True))


# ---------------------------------------------------------------------------
# validate_row: repeat_index / repeat_count constraints
# ---------------------------------------------------------------------------


class TestValidateRepeat:
    def test_valid_zero_index(self):
        validate_row(make_row(repeat_index=0, repeat_count=5))

    def test_valid_last_index(self):
        validate_row(make_row(repeat_index=4, repeat_count=5))

    def test_index_equal_to_count_rejected(self):
        with pytest.raises(
            SchemaValidationError, match="repeat_index.*must be < repeat_count"
        ):
            validate_row(make_row(repeat_index=5, repeat_count=5))

    def test_index_greater_than_count_rejected(self):
        with pytest.raises(
            SchemaValidationError, match="repeat_index.*must be < repeat_count"
        ):
            validate_row(make_row(repeat_index=10, repeat_count=5))

    def test_negative_index_rejected(self):
        with pytest.raises(
            SchemaValidationError, match="repeat_index.*must be an int >= 0"
        ):
            validate_row(make_row(repeat_index=-1, repeat_count=5))

    def test_zero_count_rejected(self):
        with pytest.raises(
            SchemaValidationError, match="repeat_count.*must be an int >= 1"
        ):
            validate_row(make_row(repeat_index=0, repeat_count=0))

    def test_negative_count_rejected(self):
        with pytest.raises(
            SchemaValidationError, match="repeat_count.*must be an int >= 1"
        ):
            validate_row(make_row(repeat_index=0, repeat_count=-1))

    def test_bool_index_rejected(self):
        with pytest.raises(
            SchemaValidationError, match="repeat_index.*must be an int >= 0"
        ):
            validate_row(make_row(repeat_index=True, repeat_count=5))

    def test_bool_count_rejected(self):
        with pytest.raises(
            SchemaValidationError, match="repeat_count.*must be an int >= 1"
        ):
            validate_row(make_row(repeat_index=0, repeat_count=True))


# ---------------------------------------------------------------------------
# validate_row: layer_class
# ---------------------------------------------------------------------------


class TestValidateLayerClass:
    @pytest.mark.parametrize("lc", ["all", "gdn", "full_attention", "ffn"])
    def test_valid_layer_classes(self, lc):
        validate_row(make_row(layer_class=lc))

    def test_empty_layer_class_defaults_to_all(self):
        validate_row(make_row(layer_class=""))

    def test_invalid_layer_class(self):
        with pytest.raises(SchemaValidationError, match="layer_class.*not in allowed"):
            validate_row(make_row(layer_class="attention"))


# ---------------------------------------------------------------------------
# validate_row: multiple errors in one call
# ---------------------------------------------------------------------------


class TestMultipleErrors:
    def test_reports_all_errors_not_just_first(self):
        """The validator collects every violation, not just the first."""
        row = make_row(
            device="bad_device",
            phase="bad_phase",
            value=-1.0,
        )
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_row(row)
        msg = str(exc_info.value)
        assert "device" in msg
        assert "phase" in msg
        assert "value" in msg

    def test_many_errors_all_reported(self):
        row = make_row(
            run_id="",
            git_sha="xyz",
            device="bad",
            engine_gdn="bad",
            context_length=0,
            value=float("nan"),
            repeat_count=0,
        )
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_row(row)
        msg = str(exc_info.value)
        assert "run_id" in msg
        assert "git_sha" in msg
        assert "device" in msg
        assert "engine_gdn" in msg
        assert "context_length" in msg
        assert "value" in msg
        assert "repeat_count" in msg


# ---------------------------------------------------------------------------
# validate_rows
# ---------------------------------------------------------------------------


class TestValidateRows:
    def test_all_valid_rows_pass(self):
        rows = [make_row(), make_row(metric_name="decode_tokens_per_sec",
                                      phase="decode", unit="tokens_per_sec")]
        validate_rows(rows)  # should not raise

    def test_single_bad_row_reports_index(self):
        rows = [make_row(), make_row(device="bad_device")]
        with pytest.raises(SchemaValidationError, match="row 1"):
            validate_rows(rows)

    def test_multiple_bad_rows_all_reported(self):
        rows = [
            make_row(device="bad_device"),
            make_row(),
            make_row(phase="bad_phase"),
        ]
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_rows(rows)
        msg = str(exc_info.value)
        assert "row 0" in msg
        assert "row 2" in msg
        assert "row 1" not in msg  # the valid row

    def test_empty_list_passes(self):
        validate_rows([])

    def test_first_error_does_not_short_circuit(self):
        """If row 0 is bad, row 2 is still checked."""
        rows = [
            make_row(value=-1),
            make_row(),
            make_row(device="bad"),
        ]
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_rows(rows)
        msg = str(exc_info.value)
        assert "row 0" in msg
        assert "row 2" in msg


# ---------------------------------------------------------------------------
# write_csv / read_csv round-trip
# ---------------------------------------------------------------------------


class TestCSVSerialization:
    def test_round_trip_single_row(self, tmp_path):
        row = make_row(notes="test note with, comma")
        path = str(tmp_path / "test.csv")
        write_csv([row], path)
        loaded = read_csv(path)
        assert len(loaded) == 1
        assert loaded[0].run_id == row.run_id
        assert loaded[0].notes == row.notes

    def test_round_trip_multiple_rows(self, tmp_path):
        rows = [
            make_row(),
            make_row(metric_name="decode_tokens_per_sec",
                      phase="decode", unit="tokens_per_sec"),
            make_memory_row(),
        ]
        path = str(tmp_path / "multi.csv")
        write_csv(rows, path)
        loaded = read_csv(path)
        assert len(loaded) == 3
        assert loaded[0].metric_name == "prefill_tokens_per_sec"
        assert loaded[1].metric_name == "decode_tokens_per_sec"
        assert loaded[2].metric_component == "weights"

    def test_round_trip_preserves_int_fields(self, tmp_path):
        row = make_row(context_length=131072, repeat_index=3, repeat_count=30)
        path = str(tmp_path / "ints.csv")
        write_csv([row], path)
        loaded = read_csv(path)
        assert loaded[0].context_length == 131072
        assert loaded[0].repeat_index == 3
        assert loaded[0].repeat_count == 30

    def test_round_trip_preserves_float_value(self, tmp_path):
        row = make_row(value=3.14159)
        path = str(tmp_path / "float.csv")
        write_csv([row], path)
        loaded = read_csv(path)
        assert loaded[0].value == pytest.approx(3.14159)

    def test_round_trip_none_component_to_none(self, tmp_path):
        """None metric_component should write as empty and read back as None."""
        row = make_row(metric_component=None)
        path = str(tmp_path / "none.csv")
        write_csv([row], path)
        loaded = read_csv(path)
        assert loaded[0].metric_component is None

    def test_round_trip_empty_component_to_none(self, tmp_path):
        """Empty string metric_component should also read back as None."""
        row = make_row(metric_component="")
        path = str(tmp_path / "empty_comp.csv")
        write_csv([row], path)
        loaded = read_csv(path)
        assert loaded[0].metric_component is None

    def test_round_trip_empty_layer_class_defaults(self, tmp_path):
        """Empty layer_class writes as empty, reads back as 'all'."""
        row = make_row(layer_class="")
        path = str(tmp_path / "empty_lc.csv")
        write_csv([row], path)
        loaded = read_csv(path)
        assert loaded[0].layer_class == "all"

    def test_write_csv_header_matches_columns(self, tmp_path):
        path = str(tmp_path / "header.csv")
        write_csv([make_row()], path)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == COLUMNS

    def test_write_csv_validates_before_writing(self, tmp_path):
        """A malformed row should not produce a partially-written file."""
        bad_row = make_row(value=-1.0)
        path = str(tmp_path / "bad.csv")
        with pytest.raises(SchemaValidationError):
            write_csv([bad_row], path)
        assert not os.path.exists(path)

    def test_write_csv_skip_validation(self, tmp_path):
        """write_csv(validate=False) skips validation."""
        bad_row = make_row(value=-1.0)
        path = str(tmp_path / "noval.csv")
        write_csv([bad_row], path, validate=False)
        assert os.path.exists(path)

    def test_read_csv_skip_validation(self, tmp_path):
        """read_csv(validate=False) reads without validating."""
        path = str(tmp_path / "noval_read.csv")
        write_csv([make_row()], path)  # write valid
        loaded = read_csv(path, validate=False)
        assert len(loaded) == 1

    def test_read_csv_wrong_header_rejected(self, tmp_path):
        """A CSV with columns not matching COLUMNS is rejected."""
        path = str(tmp_path / "wrong_header.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["a", "b", "c"])
            writer.writeheader()
            writer.writerow({"a": "1", "b": "2", "c": "3"})
        with pytest.raises(SchemaValidationError, match="CSV header.*does not match"):
            read_csv(path)

    def test_read_csv_bad_int_rejected(self, tmp_path):
        """Non-integer context_length in CSV raises parse error."""
        path = str(tmp_path / "bad_int.csv")
        row = make_row()
        write_csv([row], path)
        # Now corrupt the CSV: replace the context_length value
        with open(path) as f:
            content = f.read()
        content = content.replace("4096", "not_a_number")
        with open(path, "w") as f:
            f.write(content)
        with pytest.raises(
            SchemaValidationError, match="context_length.*could not parse"
        ):
            read_csv(path, validate=False)

    def test_read_csv_bad_float_rejected(self, tmp_path):
        """Non-float value in CSV raises parse error."""
        path = str(tmp_path / "bad_float.csv")
        row = make_row()
        write_csv([row], path)
        with open(path) as f:
            content = f.read()
        content = content.replace("42.0", "not_a_float")
        with open(path, "w") as f:
            f.write(content)
        with pytest.raises(SchemaValidationError, match="value.*could not parse"):
            read_csv(path, validate=False)


# ---------------------------------------------------------------------------
# Metric vocabulary rules consistency
# ---------------------------------------------------------------------------


class TestMetricVocabularyRules:
    def test_every_metric_has_a_unit(self):
        for metric in MetricName:
            assert metric in METRIC_UNITS, f"{metric} missing from METRIC_UNITS"

    def test_every_metric_has_allowed_phases(self):
        for metric in MetricName:
            assert (
                metric in METRIC_ALLOWED_PHASES
            ), f"{metric} missing from METRIC_ALLOWED_PHASES"

    def test_metrics_requiring_component_only_memory(self):
        assert {MetricName.PEAK_MEMORY_BYTES} == METRICS_REQUIRING_COMPONENT

    def test_unit_values_match_metric_units(self):
        for metric, unit in METRIC_UNITS.items():
            assert isinstance(metric, MetricName)
            assert isinstance(unit, Unit)

    def test_memory_allowed_both_phases(self):
        phases = METRIC_ALLOWED_PHASES[MetricName.PEAK_MEMORY_BYTES]
        assert Phase.PREFILL in phases
        assert Phase.DECODE in phases

    def test_prefill_tokens_only_prefill(self):
        phases = METRIC_ALLOWED_PHASES[MetricName.PREFILL_TOKENS_PER_SEC]
        assert phases == {Phase.PREFILL}

    def test_decode_tokens_only_decode(self):
        phases = METRIC_ALLOWED_PHASES[MetricName.DECODE_TOKENS_PER_SEC]
        assert phases == {Phase.DECODE}

    def test_ttft_only_prefill(self):
        phases = METRIC_ALLOWED_PHASES[MetricName.TTFT_SECONDS]
        assert phases == {Phase.PREFILL}


# ---------------------------------------------------------------------------
# ResultRow dataclass
# ---------------------------------------------------------------------------


class TestResultRowDataclass:
    def test_default_layer_class_is_all(self):
        row = ResultRow(
            run_id="x",
            timestamp="2026-01-01",
            git_sha="abcdef0",
            manifest_ref="ref",
            device="o6",
            engine_gdn="cpu",
            engine_full_attention="cpu",
            model_checkpoint="m",
            quantization="fp32",
            context_length=4096,
            phase="prefill",
            metric_name="prefill_tokens_per_sec",
            metric_component=None,
            value=1.0,
            unit="tokens_per_sec",
            repeat_index=0,
            repeat_count=1,
        )
        assert row.layer_class == "all"

    def test_default_notes_is_empty(self):
        row = make_row()
        assert row.notes == ""

    def test_notes_can_be_set(self):
        row = make_row(notes="some observation")
        assert row.notes == "some observation"


# ---------------------------------------------------------------------------
# Integration: committed CSVs
# ---------------------------------------------------------------------------


class TestCommittedCSVs:
    """Spot-check that committed fleet CSVs pass schema validation (if they
    match the schema format).  Fleet kernel CSVs use a different column set,
    so we only check files that have the schema columns.
    """

    def test_committed_csvs_with_schema_format_are_valid(self):
        base = os.path.join(os.path.dirname(__file__), "..", "results", "raw")
        if not os.path.isdir(base):
            pytest.skip("results/raw/ does not exist")

        schema_csvs = []
        ablation_dir = os.path.join(base, "ablation")
        if os.path.isdir(ablation_dir):
            for fname in os.listdir(ablation_dir):
                if fname.endswith(".csv"):
                    schema_csvs.append(os.path.join(ablation_dir, fname))

        if not schema_csvs:
            pytest.skip("no ablation CSVs found")

        for csv_path in schema_csvs:
            try:
                rows = read_csv(csv_path, validate=True)
                assert len(rows) > 0, f"{csv_path} has no data rows"
            except SchemaValidationError as exc:
                if "does not match expected columns" in str(exc):
                    continue  # different format, skip
                raise
