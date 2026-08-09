# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Executable form of docs/RESULTS_SCHEMA.md.

This module is the machine-enforced counterpart to the human contract in
``docs/RESULTS_SCHEMA.md``. The two describe the same frozen results schema from two
angles and MUST be changed together, in the same commit: if you add, rename, or remove a
column, an enum value, or a metric here, make the matching edit there, and vice versa.
Per docs/archive/PLAN.md section 2.4 and section 5, this schema is an early, frozen dependency of the
benchmark harness, the plotting code, and the final comparison table — renaming or
removing anything already in use invalidates already-collected data. See the "changing
this schema" section of the doc before touching either file.

Targets Python 3.10+, and uses no third-party dependencies -- only ``csv``,
``dataclasses``, ``enum``, ``datetime``, ``math``, and ``re`` from the standard library.
The stdlib-only rule is deliberate and still holds: this module has to import cleanly in
the NOE Compiler's own environment and on the board, where we do not control the
dependency set.

(This file was originally written to a Python 3.8 dialect to accommodate a reported 3.8
pin on the CIX NOE Compiler. That pin was wrong -- the SDK documents **Python 3.10**, and
Debian 12 on the board ships 3.11. See docs/CLAIM_VERIFICATION.md section 2.2a. The 3.8
restriction has been lifted accordingly.)
"""

import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# ---------------------------------------------------------------------------
# Enums (the closed vocabularies from docs/RESULTS_SCHEMA.md sections 3 and 4)
# ---------------------------------------------------------------------------


class Device(Enum):
    """Physical or hedge target a run executed on."""

    O6 = "o6"
    GENERIC_AARCH64 = "generic_aarch64"
    X86_REFERENCE = "x86_reference"
    RK3588_T3 = "rk3588-t3"
    RK3588_T4 = "rk3588-t4"
    JETSON_J1 = "jetson-j1"
    JETSON_J2 = "jetson-j2"


class Engine(Enum):
    """Execution engine a layer class was dispatched to.

    Used independently for ``engine_gdn`` and ``engine_full_attention`` because
    per-layer-class engine assignment is the central design question of E6: GDN's
    sequential recurrent scan and full attention's dense matmuls are expected to land on
    different engines, and a single "engine" column cannot express that split.
    """

    NPU = "npu"
    GPU_VULKAN = "gpu_vulkan"
    GPU_OPENCL = "gpu_opencl"
    CPU = "cpu"
    CUDA_REFERENCE = "cuda_reference"


class Phase(Enum):
    """Prefill vs decode. Never averaged into one throughput number -- see docs/archive/PLAN.md 2.4."""

    PREFILL = "prefill"
    DECODE = "decode"


class MetricName(Enum):
    """The frozen metric vocabulary. See METRIC_UNITS / METRIC_ALLOWED_PHASES below."""

    PREFILL_TOKENS_PER_SEC = "prefill_tokens_per_sec"
    DECODE_TOKENS_PER_SEC = "decode_tokens_per_sec"
    TTFT_SECONDS = "ttft_seconds"
    PEAK_MEMORY_BYTES = "peak_memory_bytes"
    ENERGY_JOULES_PER_TOKEN = "energy_joules_per_token"


class MemoryComponent(Enum):
    """The three-way memory attribution: the load-bearing measurement of this project.

    ``weights`` is expected flat, ``kv_cache`` is expected to grow linearly with context
    (full-attention layers only), ``recurrent_state`` is expected to stay O(1) per token
    (GDN layers only). Required when ``metric_name == PEAK_MEMORY_BYTES``, and must be
    absent for every other metric.
    """

    WEIGHTS = "weights"
    KV_CACHE = "kv_cache"
    RECURRENT_STATE = "recurrent_state"


class Unit(Enum):
    TOKENS_PER_SEC = "tokens_per_sec"
    SECONDS = "seconds"
    BYTES = "bytes"
    JOULES_PER_TOKEN = "joules_per_token"


class LayerClass(Enum):
    """Optional, additive dimension reserved for future per-layer-class metrics.

    Rows produced by the initial harness use ALL; a later bead may emit GDN /
    FULL_ATTENTION / FFN rows for finer-grained (e.g. per-layer-class latency) metrics
    without a schema migration, per docs/RESULTS_SCHEMA.md section 6.
    """

    GDN = "gdn"
    FULL_ATTENTION = "full_attention"
    FFN = "ffn"
    ALL = "all"


# ---------------------------------------------------------------------------
# Metric vocabulary rules (docs/RESULTS_SCHEMA.md section 4)
# ---------------------------------------------------------------------------

METRIC_UNITS: dict[MetricName, Unit] = {
    MetricName.PREFILL_TOKENS_PER_SEC: Unit.TOKENS_PER_SEC,
    MetricName.DECODE_TOKENS_PER_SEC: Unit.TOKENS_PER_SEC,
    MetricName.TTFT_SECONDS: Unit.SECONDS,
    MetricName.PEAK_MEMORY_BYTES: Unit.BYTES,
    MetricName.ENERGY_JOULES_PER_TOKEN: Unit.JOULES_PER_TOKEN,
}

METRIC_ALLOWED_PHASES: dict[MetricName, set[Phase]] = {
    MetricName.PREFILL_TOKENS_PER_SEC: {Phase.PREFILL},
    MetricName.DECODE_TOKENS_PER_SEC: {Phase.DECODE},
    MetricName.TTFT_SECONDS: {Phase.PREFILL},
    MetricName.PEAK_MEMORY_BYTES: {Phase.PREFILL, Phase.DECODE},
    MetricName.ENERGY_JOULES_PER_TOKEN: {Phase.PREFILL, Phase.DECODE},
}

# Metrics that require metric_component (currently only peak_memory_bytes -- the
# three-way memory split).
METRICS_REQUIRING_COMPONENT: set[MetricName] = {MetricName.PEAK_MEMORY_BYTES}

# Canonical context sweep points (documentary only -- context_length accepts any
# positive integer so exploratory/intermediate points are not schema violations).
CANONICAL_CONTEXT_LENGTHS = (4096, 32768, 131072, 262144)

# Column order as it appears in docs/RESULTS_SCHEMA.md section 3 and in every CSV.
COLUMNS: list[str] = [
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

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


class SchemaValidationError(ValueError):
    """Raised by validate_row/validate_rows when a row violates the frozen schema."""


# ---------------------------------------------------------------------------
# Row type
# ---------------------------------------------------------------------------


@dataclass
class ResultRow:
    """One tidy-format measurement row. See docs/RESULTS_SCHEMA.md section 3.

    Enum-valued fields are stored as plain strings (the corresponding Enum's ``.value``)
    rather than Enum instances, so a ResultRow round-trips through csv.DictReader /
    DictWriter without any custom (de)serialization. validate_row() is what enforces that
    those strings are actually members of the allowed vocabulary.
    """

    run_id: str
    timestamp: str
    git_sha: str
    manifest_ref: str
    device: str
    engine_gdn: str
    engine_full_attention: str
    model_checkpoint: str
    quantization: str
    context_length: int
    phase: str
    metric_name: str
    metric_component: str | None
    value: float
    unit: str
    repeat_index: int
    repeat_count: int
    layer_class: str = LayerClass.ALL.value
    notes: str = ""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_DEVICE_VALUES = set(d.value for d in Device)
_ENGINE_VALUES = set(e.value for e in Engine)
_PHASE_VALUES = set(p.value for p in Phase)
_METRIC_NAME_VALUES = set(m.value for m in MetricName)
_MEMORY_COMPONENT_VALUES = set(c.value for c in MemoryComponent)
_UNIT_VALUES = set(u.value for u in Unit)
_LAYER_CLASS_VALUES = set(lc.value for lc in LayerClass)

_REQUIRED_NONEMPTY_STRING_FIELDS = (
    "run_id",
    "timestamp",
    "git_sha",
    "manifest_ref",
    "model_checkpoint",
    "quantization",
)


def _parse_iso8601(value: str) -> None:
    # Accept a trailing "Z" (Python's fromisoformat before 3.11 does not).
    candidate = value
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    datetime.fromisoformat(candidate)


def validate_row(row: ResultRow) -> None:
    """Validate a single ResultRow against the frozen schema.

    Raises SchemaValidationError with a message describing every violation found (not
    just the first), so a caller fixing a bad row does not have to re-run repeatedly to
    discover each problem in turn.
    """
    errors = []  # type: List[str]

    for field_name in _REQUIRED_NONEMPTY_STRING_FIELDS:
        value = getattr(row, field_name)
        if value is None or not isinstance(value, str) or not value.strip():
            errors.append(f"{field_name}: required non-empty string, got {value!r}")

    if isinstance(row.timestamp, str) and row.timestamp.strip():
        try:
            _parse_iso8601(row.timestamp)
        except ValueError:
            errors.append(f"timestamp: not valid ISO 8601: {row.timestamp!r}")

    if (
        isinstance(row.git_sha, str)
        and row.git_sha.strip()
        and not _GIT_SHA_RE.match(row.git_sha.lower())
    ):
        errors.append(f"git_sha: expected 7-40 lowercase hex characters, got {row.git_sha!r}")

    if row.device not in _DEVICE_VALUES:
        errors.append(f"device: {row.device!r} not in allowed values {sorted(_DEVICE_VALUES)}")

    for engine_field in ("engine_gdn", "engine_full_attention"):
        value = getattr(row, engine_field)
        if value not in _ENGINE_VALUES:
            errors.append(
                f"{engine_field}: {value!r} not in allowed values {sorted(_ENGINE_VALUES)}"
            )

    if not isinstance(row.context_length, int) or isinstance(row.context_length, bool):
        errors.append(f"context_length: must be an int, got {row.context_length!r}")
    elif row.context_length <= 0:
        errors.append(f"context_length: must be > 0, got {row.context_length}")

    if row.phase not in _PHASE_VALUES:
        errors.append(f"phase: {row.phase!r} not in allowed values {sorted(_PHASE_VALUES)}")

    metric = None  # type: Optional[MetricName]
    if row.metric_name not in _METRIC_NAME_VALUES:
        errors.append(
            f"metric_name: {row.metric_name!r} not in allowed values {sorted(_METRIC_NAME_VALUES)}"
        )
    else:
        metric = MetricName(row.metric_name)

    if row.unit not in _UNIT_VALUES:
        errors.append(f"unit: {row.unit!r} not in allowed values {sorted(_UNIT_VALUES)}")
    elif metric is not None:
        expected_unit = METRIC_UNITS[metric]
        if row.unit != expected_unit.value:
            errors.append(
                f"unit: metric_name {row.metric_name!r} requires unit {expected_unit.value!r}, got {row.unit!r}"
            )

    if metric is not None and row.phase in _PHASE_VALUES:
        allowed_phases = METRIC_ALLOWED_PHASES[metric]
        if Phase(row.phase) not in allowed_phases:
            errors.append(
                f"phase: metric_name {row.metric_name!r} only valid for phase in {sorted(p.value for p in allowed_phases)}, got {row.phase!r}"
            )

    component_present = row.metric_component is not None and str(row.metric_component).strip() != ""
    if metric is not None:
        if metric in METRICS_REQUIRING_COMPONENT:
            if not component_present:
                errors.append(
                    f"metric_component: required when metric_name is {row.metric_name!r}, got {row.metric_component!r}"
                )
            elif row.metric_component not in _MEMORY_COMPONENT_VALUES:
                errors.append(
                    f"metric_component: {row.metric_component!r} not in allowed values {sorted(_MEMORY_COMPONENT_VALUES)}"
                )
        else:
            if component_present:
                errors.append(
                    f"metric_component: must be empty when metric_name is {row.metric_name!r}, got {row.metric_component!r}"
                )

    if isinstance(row.value, bool) or not isinstance(row.value, (int, float)):
        errors.append(f"value: must be a number, got {row.value!r}")
    else:
        value_float = float(row.value)
        if math.isnan(value_float) or math.isinf(value_float):
            errors.append(f"value: must be finite, got {row.value!r}")
        elif value_float < 0:
            errors.append(f"value: must be >= 0, got {row.value!r}")

    if (
        not isinstance(row.repeat_index, int)
        or isinstance(row.repeat_index, bool)
        or row.repeat_index < 0
    ):
        errors.append(f"repeat_index: must be an int >= 0, got {row.repeat_index!r}")

    if (
        not isinstance(row.repeat_count, int)
        or isinstance(row.repeat_count, bool)
        or row.repeat_count < 1
    ):
        errors.append(f"repeat_count: must be an int >= 1, got {row.repeat_count!r}")

    if (
        isinstance(row.repeat_index, int)
        and isinstance(row.repeat_count, int)
        and not isinstance(row.repeat_index, bool)
        and not isinstance(row.repeat_count, bool)
        and row.repeat_index >= row.repeat_count
    ):
        errors.append(
            f"repeat_index ({row.repeat_index}) must be < repeat_count ({row.repeat_count})"
        )

    layer_class = row.layer_class if row.layer_class else LayerClass.ALL.value
    if layer_class not in _LAYER_CLASS_VALUES:
        errors.append(
            f"layer_class: {row.layer_class!r} not in allowed values {sorted(_LAYER_CLASS_VALUES)}"
        )

    if errors:
        prefix = "invalid ResultRow (run_id={!r})".format(getattr(row, "run_id", None))
        raise SchemaValidationError(prefix + ": " + "; ".join(errors))


def validate_rows(rows: list[ResultRow]) -> None:
    """Validate every row, raising a single SchemaValidationError listing every bad row.

    Prefer this over calling validate_row() in a loop when checking a whole CSV: it
    reports all violations found across the dataset in one exception instead of stopping
    at the first one.
    """
    errors = []  # type: List[str]
    for index, row in enumerate(rows):
        try:
            validate_row(row)
        except SchemaValidationError as exc:
            errors.append(f"row {index}: {exc}")
    if errors:
        raise SchemaValidationError("\n".join(errors))


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


def _row_to_csv_dict(row: ResultRow) -> dict[str, str]:
    out = {}
    for field_name in COLUMNS:
        value = getattr(row, field_name)
        if value is None:
            out[field_name] = ""
        else:
            out[field_name] = str(value)
    return out


def write_csv(rows: list[ResultRow], path: str, validate: bool = True) -> None:
    """Write rows to path as a schema-conformant CSV.

    Validates all rows first (unless validate=False) so a malformed row is caught before
    anything is written to disk, rather than producing a partially-written invalid file.
    Uses only the csv module from the standard library.
    """
    if validate:
        validate_rows(rows)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_csv_dict(row))


def _parse_int(value: str, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise SchemaValidationError(f"{field_name}: could not parse {value!r} as int") from None


def _parse_float(value: str, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise SchemaValidationError(f"{field_name}: could not parse {value!r} as float") from None


def read_csv(path: str, validate: bool = True) -> list[ResultRow]:
    """Read a schema-conformant CSV back into a list of ResultRow.

    Uses only the csv module from the standard library. Raises SchemaValidationError if
    the header does not match COLUMNS, or (unless validate=False) if any row violates
    the schema.
    """
    rows = []  # type: List[ResultRow]
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise SchemaValidationError(
                f"CSV header {reader.fieldnames} does not match expected columns {COLUMNS}"
            )
        for raw in reader:
            metric_component = raw["metric_component"] if raw["metric_component"] else None
            row = ResultRow(
                run_id=raw["run_id"],
                timestamp=raw["timestamp"],
                git_sha=raw["git_sha"],
                manifest_ref=raw["manifest_ref"],
                device=raw["device"],
                engine_gdn=raw["engine_gdn"],
                engine_full_attention=raw["engine_full_attention"],
                model_checkpoint=raw["model_checkpoint"],
                quantization=raw["quantization"],
                context_length=_parse_int(raw["context_length"], "context_length"),
                phase=raw["phase"],
                metric_name=raw["metric_name"],
                metric_component=metric_component,
                value=_parse_float(raw["value"], "value"),
                unit=raw["unit"],
                repeat_index=_parse_int(raw["repeat_index"], "repeat_index"),
                repeat_count=_parse_int(raw["repeat_count"], "repeat_count"),
                layer_class=raw["layer_class"] if raw["layer_class"] else LayerClass.ALL.value,
                notes=raw["notes"] or "",
            )
            rows.append(row)
    if validate:
        validate_rows(rows)
    return rows


__all__ = [
    "Device",
    "Engine",
    "Phase",
    "MetricName",
    "MemoryComponent",
    "Unit",
    "LayerClass",
    "METRIC_UNITS",
    "METRIC_ALLOWED_PHASES",
    "METRICS_REQUIRING_COMPONENT",
    "CANONICAL_CONTEXT_LENGTHS",
    "COLUMNS",
    "SchemaValidationError",
    "ResultRow",
    "validate_row",
    "validate_rows",
    "write_csv",
    "read_csv",
]
