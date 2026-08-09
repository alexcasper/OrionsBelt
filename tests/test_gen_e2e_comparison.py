# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/gen_e2e_comparison.py — dedup logic and table generation.

Covers the device-name normalization and stale-entry dedup that prevents
old low-run data from cluttering the fleet comparison table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Import from scripts/ by adding it to the path
_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from gen_e2e_comparison import (  # noqa: E402
    _check_commit_lineage,
    _check_manifest_dirty,
    _dedup_rows,
    _normalize_device,
    extract_metrics,
    fmt_mean_std,
    generate_table,
    load_rows,
    main,
)

# ---------------------------------------------------------------------------
# Schema columns matching the e2e schema CSV format
# ---------------------------------------------------------------------------

COLUMNS = [
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


def _row(
    *,
    device="rk3588-t3_big",
    model="Qwen/Qwen3.5-4B",
    quant="fp32",
    metric="decode_tokens_per_sec",
    value=1.04,
    git_sha="8e7403c",
    repeat_count=3,
    notes="tokens=16;cluster=big",
):
    """Build a single e2e schema CSV row."""
    return {
        "run_id": f"{device}_e2e_{git_sha}",
        "timestamp": "2026-08-07T05:00:00Z",
        "git_sha": git_sha,
        "manifest_ref": f"results/manifests/{device}_e2e.json",
        "device": device,
        "engine_gdn": "cpu",
        "engine_full_attention": "cpu",
        "model_checkpoint": model,
        "quantization": quant,
        "context_length": "16",
        "phase": "decode" if "tokens_per_sec" in metric else "prefill",
        "metric_name": metric,
        "metric_component": "",
        "value": str(value),
        "unit": "tokens_per_sec" if "tokens_per_sec" in metric else "seconds",
        "repeat_index": "0",
        "repeat_count": str(repeat_count),
        "layer_class": "all",
        "notes": notes,
    }


class TestNormalizeDevice:
    """Test device-name normalization."""

    def test_already_has_big(self):
        assert _normalize_device("rk3588-t3_big", "cluster=big") == "rk3588-t3_big"

    def test_already_has_little(self):
        assert _normalize_device("rk3588-t3_little", "cluster=little") == "rk3588-t3_little"

    def test_infers_big_from_notes(self):
        assert _normalize_device("rk3588-t3", "tokens=16;cluster=big") == "rk3588-t3_big"

    def test_infers_little_from_notes(self):
        assert _normalize_device("rk3588-t3", "tokens=16;cluster=little") == "rk3588-t3_little"

    def test_no_cluster_in_notes(self):
        assert _normalize_device("jetson-j1", "tokens=16") == "jetson-j1_all"

    def test_empty_notes(self):
        assert _normalize_device("jetson-j1", "") == "jetson-j1_all"


class TestDedupRows:
    """Test that stale low-run entries are removed when a higher-run entry exists."""

    def test_removes_fewer_run_entry(self):
        """Old 2-run entry superseded by new 3-run entry for same device/model/quant."""
        rows = [
            _row(device="rk3588-t3", value=1.04, repeat_count=2, git_sha="2e752af"),
            _row(
                device="rk3588-t3",
                value=0.964,
                repeat_count=2,
                git_sha="2e752af",
                metric="ttft_seconds",
            ),
            _row(device="rk3588-t3_big", value=1.04, repeat_count=3, git_sha="8e7403c"),
            _row(
                device="rk3588-t3_big",
                value=0.960,
                repeat_count=3,
                git_sha="8e7403c",
                metric="ttft_seconds",
            ),
        ]
        _dedup_rows(rows)
        devices = {r["device"] for r in rows}
        assert devices == {"rk3588-t3_big"}, f"Expected only _big, got {devices}"

    def test_keeps_different_clusters(self):
        """big and little cluster entries should NOT be deduped."""
        rows = [
            _row(device="rk3588-t3_big", value=1.04, repeat_count=3),
            _row(device="rk3588-t3_big", value=0.960, repeat_count=3, metric="ttft_seconds"),
            _row(device="rk3588-t3_little", value=0.30, repeat_count=2),
            _row(device="rk3588-t3_little", value=2.5, repeat_count=2, metric="ttft_seconds"),
        ]
        _dedup_rows(rows)
        devices = {r["device"] for r in rows}
        assert devices == {"rk3588-t3_big", "rk3588-t3_little"}

    def test_keeps_different_models(self):
        """4B and 0.8B entries should NOT be deduped."""
        rows = [
            _row(device="rk3588-t3_big", model="Qwen/Qwen3.5-4B", value=1.04, repeat_count=3),
            _row(device="rk3588-t3_big", model="Qwen/Qwen3.5-0.8B", value=7.95, repeat_count=3),
        ]
        _dedup_rows(rows)
        assert len(rows) == 2

    def test_keeps_different_quant(self):
        """fp32 and int8 entries should NOT be deduped."""
        rows = [
            _row(device="rk3588-t3_big", quant="fp32", value=1.04, repeat_count=3),
            _row(device="rk3588-t3_big_int8", quant="int8", value=1.84, repeat_count=3),
        ]
        _dedup_rows(rows)
        assert len(rows) == 2

    def test_no_dedup_needed(self):
        """Single entry per group — nothing removed."""
        rows = [
            _row(device="jetson-j1", value=0.43, repeat_count=3),
            _row(device="jetson-j1", value=2.321, repeat_count=3, metric="ttft_seconds"),
        ]
        _dedup_rows(rows)
        assert len(rows) == 2

    def test_handles_missing_notes(self):
        """Rows with empty/missing notes should not crash."""
        rows = [
            _row(device="jetson-j1", value=0.43, repeat_count=3, notes=""),
        ]
        _dedup_rows(rows)
        assert len(rows) == 1


class TestFmtMeanStd:
    """Test the formatting helper."""

    def test_empty(self):
        assert fmt_mean_std([]) == "—"

    def test_single_value(self):
        assert fmt_mean_std([1.04]) == "1.04"

    def test_identical_values(self):
        assert fmt_mean_std([1.04, 1.04, 1.04]) == "1.04"

    def test_with_spread(self):
        result = fmt_mean_std([1.0, 1.1, 1.2])
        assert "±" in result

    def test_low_spread_no_std(self):
        """<1% spread should not show ± std."""
        result = fmt_mean_std([100.0, 100.5])
        assert "±" not in result


class TestCheckManifestDirty:
    """Test manifest dirty-status cross-referencing."""

    def test_all_clean(self, tmp_path, monkeypatch):
        """All manifests dirty=false → (False, True)."""
        import json

        manifests = tmp_path / "manifests"
        manifests.mkdir()
        for name in ["dev_a_e2e.json", "dev_b_e2e.json"]:
            (manifests / name).write_text(json.dumps({"git": {"sha": "abc1234", "dirty": False}}))
        monkeypatch.setattr("gen_e2e_comparison.MANIFESTS_DIR", manifests)
        refs = ["results/manifests/dev_a_e2e.json", "results/manifests/dev_b_e2e.json"]
        any_dirty, all_checked = _check_manifest_dirty(refs)
        assert any_dirty is False
        assert all_checked is True

    def test_some_dirty(self, tmp_path, monkeypatch):
        """One manifest dirty=true → (True, True)."""
        import json

        manifests = tmp_path / "manifests"
        manifests.mkdir()
        (manifests / "dev_a_e2e.json").write_text(
            json.dumps({"git": {"sha": "abc1234", "dirty": False}})
        )
        (manifests / "dev_b_e2e.json").write_text(
            json.dumps({"git": {"sha": "def5678", "dirty": True}})
        )
        monkeypatch.setattr("gen_e2e_comparison.MANIFESTS_DIR", manifests)
        refs = ["results/manifests/dev_a_e2e.json", "results/manifests/dev_b_e2e.json"]
        any_dirty, all_checked = _check_manifest_dirty(refs)
        assert any_dirty is True
        assert all_checked is True

    def test_missing_manifest(self, tmp_path, monkeypatch):
        """Manifest file not found → all_checked=False."""
        manifests = tmp_path / "manifests"
        manifests.mkdir()
        monkeypatch.setattr("gen_e2e_comparison.MANIFESTS_DIR", manifests)
        refs = ["results/manifests/nonexistent_e2e.json"]
        any_dirty, all_checked = _check_manifest_dirty(refs)
        assert any_dirty is False
        assert all_checked is False

    def test_empty_refs(self, tmp_path, monkeypatch):
        """No manifest refs → no dirty found, all_checked=True (vacuously)."""
        monkeypatch.setattr("gen_e2e_comparison.MANIFESTS_DIR", tmp_path)
        any_dirty, all_checked = _check_manifest_dirty(set())
        assert any_dirty is False
        assert all_checked is True

    def test_malformed_json(self, tmp_path, monkeypatch):
        """Malformed JSON manifest → all_checked=False."""
        manifests = tmp_path / "manifests"
        manifests.mkdir()
        (manifests / "bad_e2e.json").write_text("{not valid json")
        monkeypatch.setattr("gen_e2e_comparison.MANIFESTS_DIR", manifests)
        refs = ["results/manifests/bad_e2e.json"]
        any_dirty, all_checked = _check_manifest_dirty(refs)
        assert any_dirty is False
        assert all_checked is False

    def test_missing_git_key(self, tmp_path, monkeypatch):
        """Manifest without git key → treated as not dirty, but checked."""
        import json

        manifests = tmp_path / "manifests"
        manifests.mkdir()
        (manifests / "dev_e2e.json").write_text(json.dumps({"other": "data"}))
        monkeypatch.setattr("gen_e2e_comparison.MANIFESTS_DIR", manifests)
        refs = ["results/manifests/dev_e2e.json"]
        any_dirty, all_checked = _check_manifest_dirty(refs)
        assert any_dirty is False
        assert all_checked is True


# ---------------------------------------------------------------------------
# extract_metrics()
# ---------------------------------------------------------------------------


class TestExtractMetrics:
    """Test metric extraction from raw rows."""

    def test_single_decode_metric(self):
        rows = [_row(value=1.5)]
        data = extract_metrics(rows)
        assert len(data) == 1
        key = ("rk3588-t3_big", "4B", "fp32")
        assert key in data
        assert data[key]["tok_per_sec"] == [1.5]

    def test_ttft_converted_to_ms(self):
        """ttft_seconds should be converted to milliseconds (×1000)."""
        rows = [_row(value=0.5, metric="ttft_seconds")]
        data = extract_metrics(rows)
        key = ("rk3588-t3_big", "4B", "fp32")
        assert data[key]["ttft"] == [500.0]  # 0.5s = 500ms

    def test_model_shortening_4b(self):
        rows = [_row(model="Qwen/Qwen3.5-4B")]
        data = extract_metrics(rows)
        assert ("rk3588-t3_big", "4B", "fp32") in data

    def test_model_shortening_08b(self):
        rows = [_row(model="Qwen/Qwen3.5-0.8B")]
        data = extract_metrics(rows)
        assert ("rk3588-t3_big", "0.8B", "fp32") in data

    def test_unknown_model_uses_basename(self):
        rows = [_row(model="some-org/custom-model")]
        data = extract_metrics(rows)
        assert ("rk3588-t3_big", "custom-model", "fp32") in data

    def test_quant_grouping(self):
        """fp32 and int8 for same device go to separate keys."""
        rows = [
            _row(quant="fp32", value=1.0),
            _row(quant="int8", value=2.0, device="rk3588-t3_big_int8"),
        ]
        data = extract_metrics(rows)
        assert ("rk3588-t3_big", "4B", "fp32") in data
        assert ("rk3588-t3_big_int8", "4B", "int8") in data

    def test_sha_truncation(self):
        rows = [_row(git_sha="abcdef0123456789")]
        data = extract_metrics(rows)
        key = ("rk3588-t3_big", "4B", "fp32")
        assert "abcdef01" in data[key]["sha"]

    def test_repeat_count_tracked(self):
        rows = [_row(repeat_count=5)]
        data = extract_metrics(rows)
        key = ("rk3588-t3_big", "4B", "fp32")
        assert data[key]["runs"] == 5

    def test_invalid_value_skipped(self):
        rows = [_row(value="not_a_number")]
        data = extract_metrics(rows)
        key = ("rk3588-t3_big", "4B", "fp32")
        assert data[key]["tok_per_sec"] == []

    def test_multiple_values_accumulate(self):
        rows = [
            _row(value=1.0),
            _row(value=1.1),
            _row(value=1.2),
        ]
        data = extract_metrics(rows)
        key = ("rk3588-t3_big", "4B", "fp32")
        assert data[key]["tok_per_sec"] == [1.0, 1.1, 1.2]

    def test_manifest_ref_tracked(self):
        rows = [_row()]
        data = extract_metrics(rows)
        key = ("rk3588-t3_big", "4B", "fp32")
        assert "results/manifests/rk3588-t3_big_e2e.json" in data[key]["manifests"]

    def test_empty_rows(self):
        data = extract_metrics([])
        assert len(data) == 0

    def test_unknown_metric_ignored(self):
        rows = [_row(value=42.0, metric="custom_metric")]
        data = extract_metrics(rows)
        key = ("rk3588-t3_big", "4B", "fp32")
        assert data[key]["tok_per_sec"] == []
        assert data[key]["ttft"] == []


# ---------------------------------------------------------------------------
# generate_table()
# ---------------------------------------------------------------------------


class TestGenerateTable:
    """Test markdown table generation."""

    def test_generates_header(self):
        data = extract_metrics([_row()])
        md = generate_table(data)
        assert "# E2E Decode Fleet Comparison" in md

    def test_contains_device(self):
        data = extract_metrics([_row()])
        md = generate_table(data)
        assert "rk3588-t3_big" in md

    def test_contains_throughput(self):
        data = extract_metrics([_row(value=1.04)])
        md = generate_table(data)
        assert "1.04" in md

    def test_multiple_models_separate_sections(self):
        rows = [
            _row(model="Qwen/Qwen3.5-4B", value=1.04),
            _row(model="Qwen/Qwen3.5-0.8B", value=8.32, device="rk3588-t3_big"),
        ]
        data = extract_metrics(rows)
        md = generate_table(data)
        assert "Qwen3.5-4B" in md
        assert "Qwen3.5-0.8B" in md

    def test_int8_vs_fp32_speedup_section(self):
        """When both fp32 and int8 exist, a speedup section should appear."""
        rows = [
            _row(quant="fp32", value=1.0, device="rk3588-t3_big"),
            _row(quant="int8", value=2.0, device="rk3588-t3_big"),
        ]
        data = extract_metrics(rows)
        md = generate_table(data)
        assert "INT8 vs FP32" in md
        assert "2.00×" in md

    def test_q8_0_speedup_section(self):
        rows = [
            _row(quant="fp32", value=1.0, device="rk3588-t3_big"),
            _row(quant="q8_0", value=3.0, device="rk3588-t3_big"),
        ]
        data = extract_metrics(rows)
        md = generate_table(data)
        assert "Q8_0 vs FP32" in md

    def test_matched_commit_message(self):
        """With base_commit and matching commit_info, show matched-commit message."""
        rows = [_row(git_sha="abc12345")]
        data = extract_metrics(rows)
        commit_info = {"abc12345": {"status": "matched", "behind": 0, "ahead": 0}}
        md = generate_table(data, base_commit="abc12345", commit_info=commit_info)
        assert "Matched-commit" in md

    def test_empty_data(self):
        md = generate_table({})
        assert "# E2E Decode Fleet Comparison" in md

    def test_table_has_separator_row(self):
        data = extract_metrics([_row()])
        md = generate_table(data)
        assert "|--------|" in md

    def test_pre_matched_commit_warning(self):
        """When commit_info has pre-matched entries, show partial-match warning."""
        rows = [_row(git_sha="dead0000")]
        data = extract_metrics(rows)
        commit_info = {"dead0000": {"status": "pre-matched", "detail": "pre-dates base commit"}}
        md = generate_table(data, base_commit="aaaa1111", commit_info=commit_info)
        assert "Partial commit match" in md
        assert "dead0000" in md

    def test_diverged_commit_warning(self):
        """When commit_info has diverged entries, show partial-match warning."""
        rows = [_row(git_sha="beef2222")]
        data = extract_metrics(rows)
        commit_info = {"beef2222": {"status": "diverged", "detail": "kernel changes: src/foo.c"}}
        md = generate_table(data, base_commit="aaaa1111", commit_info=commit_info)
        assert "Partial commit match" in md
        assert "beef2222" in md

    def test_matched_commit_lists_non_base_shas(self):
        """When all commits are code-identical, non-base SHAs are listed."""
        rows = [_row(git_sha="cccc3333")]
        data = extract_metrics(rows)
        commit_info = {"cccc3333": {"status": "code-identical", "detail": "results/docs only"}}
        md = generate_table(data, base_commit="aaaa1111", commit_info=commit_info)
        assert "Matched-commit" in md
        assert "cccc3333" in md

    def test_multi_sha_dirty_manifest_warning(self):
        """With multiple SHAs and dirty manifests, show dirty warning."""
        rows = [
            _row(git_sha="aaaa1111", value=1.0, device="dev1_big"),
            _row(git_sha="bbbb2222", value=2.0, device="dev2_big"),
        ]
        data = extract_metrics(rows)
        md = generate_table(data)  # no base_commit → fallback path
        assert "Commit mismatch" in md or "dirty" in md.lower()

    def test_multi_commit_note_per_row(self):
        """When an entry has multiple SHAs, the note 'multi-commit' appears."""
        rows = [
            _row(git_sha="aaaa1111", device="dev_big", repeat_count=1),
            _row(git_sha="bbbb2222", device="dev_big", repeat_count=1),
        ]
        data = extract_metrics(rows)
        md = generate_table(data)
        assert "multi-commit" in md

    def test_int8_speedup_without_fp32_baseline(self):
        """When int8 exists but no fp32 baseline, speedup shows '—'."""
        rows = [_row(quant="int8", value=2.0, device="rk3588-t3_big")]
        data = extract_metrics(rows)
        md = generate_table(data)
        assert "INT8 vs FP32" in md
        # No fp32 data → em dash for speedup
        assert "—" in md

    def test_q8_0_speedup_calculation(self):
        """Q8_0 speedup is correctly computed as ratio."""
        rows = [
            _row(quant="fp32", value=1.0, device="rk3588-t3_big"),
            _row(quant="q8_0", value=3.0, device="rk3588-t3_big"),
        ]
        data = extract_metrics(rows)
        md = generate_table(data)
        assert "Q8_0 vs FP32" in md
        assert "3.00×" in md

    def test_data_sources_section(self):
        """Generated table ends with a Data Sources section."""
        data = extract_metrics([_row()])
        md = generate_table(data)
        assert "## Data Sources" in md


# ---------------------------------------------------------------------------
# _check_commit_lineage — git-provenance classification
# ---------------------------------------------------------------------------

# Real commits from this repo's history, chosen so that each branch of
# _check_commit_lineage is exercised.  These are stable SHAs — they reference
# the project's own history and will never be garbage-collected.
_BASE = "54c94df"  # "t4: device-fleet bench rk3588-t4 (ob-8ms.3)"
_DESC_CODE_IDENTICAL = "2619f8d"  # descendant of _BASE, only results/docs/tests changed
_ANCESTOR = "8227e98"  # ancestor of _BASE (pre-dates it)
# For the diverged case we use a different base: the very first commit has
# descendants that added kernel source files (src/, bench/).
_BASE_EARLY = "dd50acd"  # "Create project brief for Qwen3.5 optimization"
_DESC_DIVERGED = "9c8d239"  # descendant of _BASE_EARLY, touches src/ (kernel)


def _is_shallow_checkout() -> bool:
    """CI uses fetch-depth=1 (shallow); historical SHAs are unreachable there."""
    import subprocess

    r = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0 and r.stdout.strip() == "true"


class TestCheckCommitLineage:
    """Test the git-provenance commit classifier."""

    pytestmark = pytest.mark.skipif(
        _is_shallow_checkout(),
        reason="historical commit SHAs unreachable in shallow CI checkout",
    )

    def test_base_commit_self_matches(self):
        """The base commit itself is classified as 'matched'."""
        result = _check_commit_lineage(_BASE, {_BASE})
        assert result[_BASE]["status"] == "matched"

    def test_code_identical_descendant(self):
        """A descendant with no kernel-affecting changes is 'code-identical'."""
        result = _check_commit_lineage(_BASE, {_DESC_CODE_IDENTICAL})
        entry = result[_DESC_CODE_IDENTICAL]
        assert entry["status"] == "code-identical"

    def test_diverged_descendant(self):
        """A descendant with kernel-affecting changes is 'diverged'."""
        result = _check_commit_lineage(_BASE_EARLY, {_DESC_DIVERGED})
        entry = result[_DESC_DIVERGED]
        assert entry["status"] == "diverged"

    def test_pre_matched_ancestor(self):
        """A commit that predates the base is 'pre-matched'."""
        result = _check_commit_lineage(_BASE, {_ANCESTOR})
        entry = result[_ANCESTOR]
        assert entry["status"] == "pre-matched"

    def test_unresolvable_sha_is_unknown(self):
        """A SHA that git cannot resolve is classified as 'unknown'."""
        result = _check_commit_lineage(_BASE, {"zzzz999"})
        assert result["zzzz999"]["status"] == "unknown"

    def test_mixed_commits(self):
        """Multiple statuses appear when a mix of commits is passed."""
        # Use _BASE for matched/code-identical/pre-matched, _BASE_EARLY for diverged
        result_a = _check_commit_lineage(_BASE, {_BASE, _DESC_CODE_IDENTICAL, _ANCESTOR})
        statuses_a = {ci["status"] for ci in result_a.values()}
        assert "matched" in statuses_a
        assert "code-identical" in statuses_a
        assert "pre-matched" in statuses_a

        result_b = _check_commit_lineage(_BASE_EARLY, {_DESC_DIVERGED})
        assert result_b[_DESC_DIVERGED]["status"] == "diverged"

    def test_diverged_detail_lists_files(self):
        """The diverged detail string lists the changed kernel files."""
        result = _check_commit_lineage(_BASE_EARLY, {_DESC_DIVERGED})
        entry = result[_DESC_DIVERGED]
        assert entry["status"] == "diverged"
        # The detail should mention at least one of the kernel-pattern files
        assert any(
            kw in entry["detail"] for kw in ("bench", "src", "include", "build_device", "run_e2e")
        )

    def test_empty_manifest_ref_skipped(self, tmp_path, monkeypatch):
        """An empty string in manifest refs is skipped via 'continue'."""
        monkeypatch.setattr("gen_e2e_comparison.MANIFESTS_DIR", tmp_path)
        any_dirty, all_checked = _check_manifest_dirty([""])
        assert any_dirty is False
        assert all_checked is True


# ---------------------------------------------------------------------------
# load_rows()
# ---------------------------------------------------------------------------


def _write_e2e_csv(path: Path, rows: list[dict]) -> None:
    """Write rows as an e2e schema CSV file."""
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TestLoadRows:
    """Test load_rows() — reading CSVs from disk."""

    def test_loads_single_csv(self, tmp_path, monkeypatch):
        raw = tmp_path / "raw"
        raw.mkdir()
        _write_e2e_csv(raw / "dev_e2e_schema.csv", [_row()])
        monkeypatch.setattr("gen_e2e_comparison.RAW_DIR", raw)
        monkeypatch.setattr("gen_e2e_comparison.REPO_ROOT", tmp_path)
        rows = load_rows()
        assert len(rows) == 1
        assert rows[0]["device"] == "rk3588-t3_big"

    def test_loads_multiple_csvs(self, tmp_path, monkeypatch):
        raw = tmp_path / "raw"
        raw.mkdir()
        _write_e2e_csv(raw / "dev1_e2e_schema.csv", [_row(device="dev1_big")])
        _write_e2e_csv(raw / "dev2_e2e_schema.csv", [_row(device="dev2_big")])
        monkeypatch.setattr("gen_e2e_comparison.RAW_DIR", raw)
        monkeypatch.setattr("gen_e2e_comparison.REPO_ROOT", tmp_path)
        rows = load_rows()
        devices = {r["device"] for r in rows}
        assert devices == {"dev1_big", "dev2_big"}

    def test_empty_raw_dir(self, tmp_path, monkeypatch):
        raw = tmp_path / "raw"
        raw.mkdir()
        monkeypatch.setattr("gen_e2e_comparison.RAW_DIR", raw)
        monkeypatch.setattr("gen_e2e_comparison.REPO_ROOT", tmp_path)
        rows = load_rows()
        assert rows == []

    def test_dedup_applied_during_load(self, tmp_path, monkeypatch):
        """load_rows() calls _dedup_rows to remove superseded entries."""
        raw = tmp_path / "raw"
        raw.mkdir()
        _write_e2e_csv(
            raw / "dev_e2e_schema.csv",
            [
                _row(device="rk3588-t3", repeat_count=2, value=1.0),
                _row(device="rk3588-t3_big", repeat_count=3, value=1.1),
            ],
        )
        monkeypatch.setattr("gen_e2e_comparison.RAW_DIR", raw)
        monkeypatch.setattr("gen_e2e_comparison.REPO_ROOT", tmp_path)
        rows = load_rows()
        devices = {r["device"] for r in rows}
        assert devices == {"rk3588-t3_big"}


# ---------------------------------------------------------------------------
# _dedup_rows — edge cases
# ---------------------------------------------------------------------------


class TestDedupEdgeCases:
    """Cover remaining _dedup_rows branches."""

    def test_unknown_model_uses_full_name(self):
        """Model with neither '4B' nor '0.8B' falls back to full model name."""
        rows = [
            _row(device="dev_big", model="custom-model", repeat_count=2, value=1.0),
            _row(device="dev_big", model="custom-model", repeat_count=3, value=2.0),
        ]
        _dedup_rows(rows)
        # Higher repeat_count wins
        assert len(rows) == 1
        assert float(rows[0]["value"]) == 2.0

    def test_non_integer_repeat_count_defaults_to_one(self):
        """Non-integer repeat_count triggers ValueError, defaults to 1."""
        rows = [
            _row(device="dev_big", repeat_count="invalid", value=1.0),
            _row(device="dev_big", repeat_count="also_bad", value=2.0),
        ]
        _dedup_rows(rows)
        # Both default to rep_count=1, same group, same max_rep → both kept
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# generate_table — manifest, dirty, and cross-quant branches
# ---------------------------------------------------------------------------


class TestGenerateTableManifestBranches:
    """Cover manifest-related branches in generate_table()."""

    def test_multi_sha_dirty_manifests(self, monkeypatch):
        """Multiple SHAs with dirty manifests → dirty-manifest warning."""
        rows = [
            _row(git_sha="aaaa1111", device="dev1_big"),
            _row(git_sha="bbbb2222", device="dev2_big"),
        ]
        data = extract_metrics(rows)
        monkeypatch.setattr("gen_e2e_comparison._check_manifest_dirty", lambda refs: (True, True))
        md = generate_table(data)
        assert "dirty" in md.lower()

    def test_multi_sha_all_clean_manifests(self, monkeypatch):
        """Multiple SHAs, all manifests dirty=false → comparable message."""
        rows = [
            _row(git_sha="aaaa1111", device="dev1_big"),
            _row(git_sha="bbbb2222", device="dev2_big"),
        ]
        data = extract_metrics(rows)
        monkeypatch.setattr("gen_e2e_comparison._check_manifest_dirty", lambda refs: (False, True))
        md = generate_table(data)
        assert "dirty=false" in md.lower() or "comparable" in md.lower()

    def test_dirty_per_row_note(self):
        """Manifest ref containing 'dirty' triggers per-row ⚠ dirty note."""
        rows = [_row(device="dev_dirty")]
        data = extract_metrics(rows)
        md = generate_table(data)
        assert "⚠ dirty" in md

    def test_fp32_without_cluster_matches_int8_big(self):
        """FP32 without _big suffix still matches INT8 _big via auto-append."""
        rows = [
            _row(quant="fp32", value=1.0, device="rk3588-t3"),
            _row(quant="int8", value=2.0, device="rk3588-t3_big"),
        ]
        data = extract_metrics(rows)
        md = generate_table(data)
        assert "INT8 vs FP32" in md
        assert "2.00×" in md

    def test_q8_0_without_fp32_baseline(self):
        """Q8_0 with no matching FP32 baseline shows '—' for speedup."""
        rows = [_row(quant="q8_0", value=3.0, device="rk3588-t3_big")]
        data = extract_metrics(rows)
        md = generate_table(data)
        assert "Q8_0 vs FP32" in md
        assert "3.00" in md
        assert "—" in md


# ---------------------------------------------------------------------------
# main() — CLI entry point
# ---------------------------------------------------------------------------


class TestMain:
    """Test the main() CLI entry point."""

    def test_main_writes_table(self, tmp_path, monkeypatch):
        """main() reads CSVs and writes the comparison table."""
        raw = tmp_path / "raw"
        raw.mkdir()
        _write_e2e_csv(raw / "dev_e2e_schema.csv", [_row()])
        monkeypatch.setattr("gen_e2e_comparison.RAW_DIR", raw)
        monkeypatch.setattr("gen_e2e_comparison.REPO_ROOT", tmp_path)
        out = tmp_path / "output.md"
        monkeypatch.setattr("sys.argv", ["gen_e2e_comparison.py", "--output", str(out)])
        main()
        assert out.exists()
        content = out.read_text()
        assert "# E2E Decode Fleet Comparison" in content

    def test_main_no_csvs_exits_with_error(self, tmp_path, monkeypatch):
        """main() exits when no CSVs found."""
        raw = tmp_path / "raw"
        raw.mkdir()
        monkeypatch.setattr("gen_e2e_comparison.RAW_DIR", raw)
        monkeypatch.setattr("gen_e2e_comparison.REPO_ROOT", tmp_path)
        out = tmp_path / "output.md"
        monkeypatch.setattr("sys.argv", ["gen_e2e_comparison.py", "--output", str(out)])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_main_with_base_commit(self, tmp_path, monkeypatch):
        """main() with --base-commit runs lineage classification."""
        raw = tmp_path / "raw"
        raw.mkdir()
        _write_e2e_csv(raw / "dev_e2e_schema.csv", [_row(git_sha=_BASE[:8])])
        monkeypatch.setattr("gen_e2e_comparison.RAW_DIR", raw)
        monkeypatch.setattr("gen_e2e_comparison.REPO_ROOT", tmp_path)
        out = tmp_path / "output.md"
        monkeypatch.setattr(
            "sys.argv",
            ["gen_e2e_comparison.py", "--output", str(out), "--base-commit", _BASE],
        )
        main()
        assert out.exists()

    def test_main_multi_sha_no_base_commit(self, tmp_path, monkeypatch):
        """main() with multiple SHAs and no --base-commit prints warning."""
        raw = tmp_path / "raw"
        raw.mkdir()
        _write_e2e_csv(raw / "dev1_e2e_schema.csv", [_row(git_sha="aaaa1111", device="dev1_big")])
        _write_e2e_csv(raw / "dev2_e2e_schema.csv", [_row(git_sha="bbbb2222", device="dev2_big")])
        monkeypatch.setattr("gen_e2e_comparison.RAW_DIR", raw)
        monkeypatch.setattr("gen_e2e_comparison.REPO_ROOT", tmp_path)
        out = tmp_path / "output.md"
        monkeypatch.setattr("sys.argv", ["gen_e2e_comparison.py", "--output", str(out)])
        main()
        assert out.exists()
