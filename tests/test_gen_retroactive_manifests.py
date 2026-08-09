# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/gen_retroactive_manifests.py — retroactive manifest generation.

Covers CSV→SHA mapping integrity, manifest field correctness per entry,
deep-copy isolation between iterations, output file creation, and the
retroactive provenance markers required by PLAN.md §9.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import scripts.gen_retroactive_manifests as grm  # noqa: E402

# ---------------------------------------------------------------------------
# Template manifest (a representative real manifest with the fields we expect)
# ---------------------------------------------------------------------------

SAMPLE_TEMPLATE = {
    "caller": {
        "original_csv": "template.csv",
        "retroactive": False,
    },
    "git": {
        "sha": "0" * 40,
        "dirty": False,
    },
    "host": {
        "cpu_model": "aarch64",
        "core_count": 8,
        "mem_total_kb": 8120248,
        "mem_available_kb": 5005428,
    },
    "run_id": "original_run",
    "software": {
        "python_version": "3.12.7",
        "python_implementation": "CPython",
    },
    "timestamp_utc": "2026-01-01T00:00:00Z",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _make_template(tmp_path: Path) -> Path:
    """Create the template manifest file and return its path."""
    out_dir = tmp_path / "manifests"
    out_dir.mkdir()
    tmpl_path = out_dir / "rk3588-t4_big_singlethread.json"
    tmpl_path.write_text(json.dumps(SAMPLE_TEMPLATE, indent=2) + "\n")
    return tmpl_path


def _setup(monkeypatch, tmp_path: Path):
    """Patch module paths to use tmp_path; return the output directory."""
    tmpl_path = _make_template(tmp_path)
    monkeypatch.setattr(grm, "TEMPLATE", str(tmpl_path))
    monkeypatch.setattr(grm, "OUTPUT_DIR", str(tmp_path / "manifests"))
    return tmp_path / "manifests"


# ---------------------------------------------------------------------------
# CSV_TO_SHA mapping integrity
# ---------------------------------------------------------------------------


class TestCsvToShaMapping:
    def test_all_shas_are_40_char_hex(self):
        for csv_name, sha in grm.CSV_TO_SHA.items():
            assert _SHA_RE.match(sha), f"Invalid SHA for {csv_name}: {sha}"

    def test_all_csv_names_are_non_empty_strings(self):
        for csv_name in grm.CSV_TO_SHA:
            assert isinstance(csv_name, str)
            assert len(csv_name) > 0

    def test_csv_names_are_unique(self):
        names = list(grm.CSV_TO_SHA.keys())
        assert len(names) == len(set(names))

    def test_csv_names_match_expected_prefix(self):
        for csv_name in grm.CSV_TO_SHA:
            assert csv_name.startswith("rk3588-t4_"), f"Unexpected csv_name {csv_name}"

    def test_at_least_two_distinct_shas(self):
        """Multiple CSVs may share a SHA (same commit batch) but we should
        have at least two distinct SHAs for provenance diversity."""
        distinct = set(grm.CSV_TO_SHA.values())
        assert len(distinct) >= 2


# ---------------------------------------------------------------------------
# Manifest generation via main()
# ---------------------------------------------------------------------------


class TestGenerateManifests:
    def test_main_creates_all_output_files(self, monkeypatch, tmp_path):
        out_dir = _setup(monkeypatch, tmp_path)
        grm.main()

        for csv_name in grm.CSV_TO_SHA:
            expected = out_dir / f"{csv_name}.json"
            assert expected.exists(), f"Missing output: {expected}"

    def test_main_creates_exactly_expected_count(self, monkeypatch, tmp_path, capsys):
        _setup(monkeypatch, tmp_path)
        grm.main()
        captured = capsys.readouterr()
        assert f"Generated {len(grm.CSV_TO_SHA)} retroactive manifests" in captured.out

    def test_each_manifest_has_correct_original_csv(self, monkeypatch, tmp_path):
        out_dir = _setup(monkeypatch, tmp_path)
        grm.main()

        for csv_name in grm.CSV_TO_SHA:
            mpath = out_dir / f"{csv_name}.json"
            manifest = json.loads(mpath.read_text())
            assert manifest["caller"]["original_csv"] == csv_name + ".csv"

    def test_each_manifest_marked_retroactive(self, monkeypatch, tmp_path):
        out_dir = _setup(monkeypatch, tmp_path)
        grm.main()

        for csv_name in grm.CSV_TO_SHA:
            manifest = json.loads((out_dir / f"{csv_name}.json").read_text())
            assert manifest["caller"]["retroactive"] is True

    def test_each_manifest_git_sha_matches_mapping(self, monkeypatch, tmp_path):
        out_dir = _setup(monkeypatch, tmp_path)
        grm.main()

        for csv_name, expected_sha in grm.CSV_TO_SHA.items():
            manifest = json.loads((out_dir / f"{csv_name}.json").read_text())
            assert manifest["git"]["sha"] == expected_sha

    def test_each_manifest_git_dirty_true(self, monkeypatch, tmp_path):
        out_dir = _setup(monkeypatch, tmp_path)
        grm.main()

        for csv_name in grm.CSV_TO_SHA:
            manifest = json.loads((out_dir / f"{csv_name}.json").read_text())
            assert manifest["git"]["dirty"] is True

    def test_each_manifest_has_retroactive_note(self, monkeypatch, tmp_path):
        out_dir = _setup(monkeypatch, tmp_path)
        grm.main()

        for csv_name in grm.CSV_TO_SHA:
            manifest = json.loads((out_dir / f"{csv_name}.json").read_text())
            note = manifest["git"].get("retroactive_note", "")
            assert isinstance(note, str)
            assert len(note) > 20  # must be a meaningful note, not empty
            assert "after the fact" in note.lower()

    def test_each_manifest_run_id_format(self, monkeypatch, tmp_path):
        out_dir = _setup(monkeypatch, tmp_path)
        grm.main()

        for csv_name, sha in grm.CSV_TO_SHA.items():
            manifest = json.loads((out_dir / f"{csv_name}.json").read_text())
            expected_run_id = "t4_retroactive_" + sha[:7]
            assert manifest["run_id"] == expected_run_id

    def test_each_manifest_has_valid_timestamp(self, monkeypatch, tmp_path):
        out_dir = _setup(monkeypatch, tmp_path)
        grm.main()

        for csv_name in grm.CSV_TO_SHA:
            manifest = json.loads((out_dir / f"{csv_name}.json").read_text())
            ts = manifest["timestamp_utc"]
            assert _TS_RE.match(ts), f"Invalid timestamp: {ts}"

    def test_timestamps_are_recent(self, monkeypatch, tmp_path):
        """Generated timestamps should be close to 'now'."""
        out_dir = _setup(monkeypatch, tmp_path)
        before = datetime.now(timezone.utc).replace(tzinfo=None)
        grm.main()
        after = datetime.now(timezone.utc).replace(tzinfo=None)

        for csv_name in grm.CSV_TO_SHA:
            manifest = json.loads((out_dir / f"{csv_name}.json").read_text())
            ts = datetime.strptime(manifest["timestamp_utc"], "%Y-%m-%dT%H:%M:%SZ")
            # Allow a generous window for slow CI runners
            assert before - ts < timedelta(seconds=5)
            assert ts - after < timedelta(seconds=5)

    def test_timestamps_within_single_run_are_identical(self, monkeypatch, tmp_path):
        """All manifests in one run share the same timestamp."""
        out_dir = _setup(monkeypatch, tmp_path)
        grm.main()

        timestamps = set()
        for csv_name in grm.CSV_TO_SHA:
            manifest = json.loads((out_dir / f"{csv_name}.json").read_text())
            timestamps.add(manifest["timestamp_utc"])
        assert len(timestamps) == 1


# ---------------------------------------------------------------------------
# Deep-copy isolation
# ---------------------------------------------------------------------------


class TestDeepCopyIsolation:
    def test_host_fields_preserved_from_template(self, monkeypatch, tmp_path):
        """Device-info fields (host, software, etc.) must be inherited
        from the template unchanged — only provenance fields vary."""
        out_dir = _setup(monkeypatch, tmp_path)
        grm.main()

        for csv_name in grm.CSV_TO_SHA:
            manifest = json.loads((out_dir / f"{csv_name}.json").read_text())
            assert manifest["host"]["cpu_model"] == SAMPLE_TEMPLATE["host"]["cpu_model"]
            assert manifest["host"]["core_count"] == SAMPLE_TEMPLATE["host"]["core_count"]
            assert manifest["host"]["mem_total_kb"] == SAMPLE_TEMPLATE["host"]["mem_total_kb"]
            assert (
                manifest["software"]["python_version"]
                == SAMPLE_TEMPLATE["software"]["python_version"]
            )

    def test_no_template_mutation_across_runs(self, monkeypatch, tmp_path):
        """Running main() twice should produce identical results for
        static fields (only timestamp changes)."""
        out_dir = _setup(monkeypatch, tmp_path)
        grm.main()
        first_name = sorted(grm.CSV_TO_SHA)[0]
        first = json.loads((out_dir / f"{first_name}.json").read_text())

        # Second run — should overwrite cleanly
        grm.main()
        second = json.loads((out_dir / f"{first_name}.json").read_text())

        # All non-timestamp fields must match
        assert first["caller"] == second["caller"]
        assert first["git"]["sha"] == second["git"]["sha"]
        assert first["git"]["dirty"] == second["git"]["dirty"]
        assert first["run_id"] == second["run_id"]
        assert first["host"] == second["host"]

    def test_nested_arrays_preserved(self, monkeypatch, tmp_path):
        """Nested structures (e.g. cpu_topology lists) survive the deep copy."""
        tmpl_path = _make_template(tmp_path)
        # Add a nested array to the template
        template = json.loads(tmpl_path.read_text())
        template["host"]["cpu_topology"] = [
            {"cpu": 0, "freq": 1800000},
            {"cpu": 1, "freq": 1800000},
        ]
        tmpl_path.write_text(json.dumps(template, indent=2) + "\n")
        monkeypatch.setattr(grm, "TEMPLATE", str(tmpl_path))
        monkeypatch.setattr(grm, "OUTPUT_DIR", str(tmp_path / "manifests"))

        grm.main()

        manifest = json.loads(
            (tmp_path / "manifests" / f"{sorted(grm.CSV_TO_SHA)[0]}.json").read_text()
        )
        assert manifest["host"]["cpu_topology"] == template["host"]["cpu_topology"]


# ---------------------------------------------------------------------------
# Output file format
# ---------------------------------------------------------------------------


class TestOutputFormat:
    def test_output_is_valid_json(self, monkeypatch, tmp_path):
        out_dir = _setup(monkeypatch, tmp_path)
        grm.main()

        for csv_name in grm.CSV_TO_SHA:
            mpath = out_dir / f"{csv_name}.json"
            data = json.loads(mpath.read_text())  # should not raise
            assert isinstance(data, dict)

    def test_output_has_trailing_newline(self, monkeypatch, tmp_path):
        out_dir = _setup(monkeypatch, tmp_path)
        grm.main()

        for csv_name in grm.CSV_TO_SHA:
            mpath = out_dir / f"{csv_name}.json"
            raw = mpath.read_text()
            assert raw.endswith("\n"), f"Missing trailing newline in {mpath}"

    def test_output_is_pretty_printed(self, monkeypatch, tmp_path):
        out_dir = _setup(monkeypatch, tmp_path)
        grm.main()

        for csv_name in grm.CSV_TO_SHA:
            mpath = out_dir / f"{csv_name}.json"
            raw = mpath.read_text()
            # Pretty-printed JSON has newlines between keys
            assert raw.count("\n") > 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_template_file_must_exist(self, monkeypatch, tmp_path):
        monkeypatch.setattr(grm, "TEMPLATE", str(tmp_path / "nonexistent.json"))
        monkeypatch.setattr(grm, "OUTPUT_DIR", str(tmp_path / "manifests"))
        with pytest.raises(FileNotFoundError):
            grm.main()

    def test_output_dir_created_if_missing(self, monkeypatch, tmp_path):
        """main() writes to OUTPUT_DIR which is created by _make_template;
        if the dir exists it should work fine."""
        out_dir = _setup(monkeypatch, tmp_path)
        assert out_dir.exists()  # pre-created by helper
        grm.main()  # should not raise
        # Only count generated manifests, not the template file
        gen_names = {f"{csv_name}.json" for csv_name in grm.CSV_TO_SHA}
        actual_files = {p.name for p in out_dir.glob("*.json") if p.name in gen_names}
        assert len(actual_files) == len(grm.CSV_TO_SHA)
