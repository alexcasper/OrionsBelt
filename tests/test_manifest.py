# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Tests for bench/manifest.py — provenance capture (bead ob-1lm).

Validates that:
  1. capture() returns a complete, JSON-serializable dict with all required keys.
  2. manifest_ref() produces the documented path format.
  3. write() round-trips through JSON correctly.
  4. Graceful degradation: probe failures produce None, never exceptions.
  5. caller fields are passed through verbatim.

These run in CI on any platform (x86/aarch64) — no model weights or GPU required.
"""

import json
import os
import re
import sys
import tempfile

# Make bench/ importable when run from repo root or from tests/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import bench.manifest as manifest_mod
from bench.manifest import (
    _affinity_count,
    _core_count,
    _cpu_model,
    _cpufreq_topology,
    _default_run_id,
    _isa_features,
    _meminfo,
    _optional_package_versions,
    _parallelism,
    _read_int_file,
    _safe,
    _thermal_zones,
    _utc_timestamp,
    capture,
    manifest_ref,
    write,
)

# ---------------------------------------------------------------------------
# _safe wrapper
# ---------------------------------------------------------------------------


class TestSafe:
    def test_returns_value_on_success(self):
        assert _safe(lambda: 42) == 42

    def test_returns_none_on_exception(self):
        assert _safe(lambda: 1 / 0) is None

    def test_returns_none_on_keyerror(self):
        d: dict = {}
        assert _safe(lambda: d["missing"]) is None

    def test_passes_args(self):
        assert _safe(lambda x, y: x + y, 3, 4) == 7


# ---------------------------------------------------------------------------
# _utc_timestamp
# ---------------------------------------------------------------------------


class TestUtcTimestamp:
    def test_format_is_iso8601(self):
        ts = _utc_timestamp()
        # 2026-08-10T14:30:00Z
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", ts), ts

    def test_ends_with_z(self):
        assert _utc_timestamp().endswith("Z")


# ---------------------------------------------------------------------------
# capture()
# ---------------------------------------------------------------------------


class TestCapture:
    """capture() is the public entry point — it must always succeed and
    return a well-formed manifest dict regardless of platform."""

    def setup_method(self):
        self.manifest = capture()

    def test_returns_dict(self):
        assert isinstance(self.manifest, dict)

    def test_manifest_version(self):
        assert self.manifest["manifest_version"] == 1

    def test_has_run_id(self):
        assert "run_id" in self.manifest
        assert isinstance(self.manifest["run_id"], str)
        assert len(self.manifest["run_id"]) > 0

    def test_has_timestamp(self):
        assert "timestamp_utc" in self.manifest
        assert self.manifest["timestamp_utc"].endswith("Z")

    def test_has_git_block(self):
        git = self.manifest["git"]
        assert "sha" in git
        assert "dirty" in git
        # sha can be None (no git) or a hex string
        if git["sha"] is not None:
            assert re.match(r"^[0-9a-f]{40}$", git["sha"]), git["sha"]
        # dirty is None (no git) or bool
        if git["dirty"] is not None:
            assert isinstance(git["dirty"], bool)

    def test_has_host_block(self):
        host = self.manifest["host"]
        assert "hostname" in host
        assert "machine" in host
        assert "kernel" in host
        assert "os" in host
        assert "cpu_model" in host
        assert "core_count" in host

    def test_has_software_block(self):
        sw = self.manifest["software"]
        assert "python_version" in sw
        assert "python_implementation" in sw
        assert "packages" in sw
        assert isinstance(sw["packages"], dict)

    def test_is_json_serializable(self):
        """The whole point: this dict must be writable as JSON."""
        text = json.dumps(self.manifest, indent=2, sort_keys=True)
        roundtrip = json.loads(text)
        assert roundtrip == self.manifest

    def test_has_thermal_zones_key(self):
        # None (no /sys/class/thermal) or a list — key must exist either way
        assert "thermal_zones" in self.manifest

    def test_has_memory_key(self):
        assert "memory" in self.manifest

    def test_has_isa_features_key(self):
        # None on x86, dict on aarch64 — key must exist either way
        assert "isa_features" in self.manifest

    def test_has_cpu_topology_key(self):
        assert "cpu_topology" in self.manifest["host"]


class TestCaptureCallerFields:
    def test_caller_fields_passed_through(self):
        m = capture(device="test_device", engine="cpu", custom="value")
        assert m["caller"]["device"] == "test_device"
        assert m["caller"]["engine"] == "cpu"
        assert m["caller"]["custom"] == "value"

    def test_explicit_run_id_used(self):
        m = capture(run_id="my_custom_run_id")
        assert m["run_id"] == "my_custom_run_id"
        # run_id should NOT appear in caller (it's popped)
        assert "run_id" not in m["caller"]


# ---------------------------------------------------------------------------
# manifest_ref()
# ---------------------------------------------------------------------------


class TestManifestRef:
    def test_basic_format(self):
        ref = manifest_ref("o6_20260810T143000Z_a1b2c3d")
        assert ref == "results/manifests/o6_20260810T143000Z_a1b2c3d.json"

    def test_custom_dir(self):
        ref = manifest_ref("my_run", results_dir="custom/dir")
        assert ref == "custom/dir/my_run.json"

    def test_uses_forward_slashes(self):
        ref = manifest_ref("test")
        assert "/" in ref
        assert "\\" not in ref


# ---------------------------------------------------------------------------
# write()
# ---------------------------------------------------------------------------


class TestWrite:
    def test_writes_valid_json(self):
        manifest = capture()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "manifest.json")
            write(manifest, path)
            assert os.path.exists(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == manifest

    def test_creates_parent_directories(self):
        manifest = {"test": True}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a", "b", "c", "manifest.json")
            write(manifest, path)
            assert os.path.exists(path)

    def test_json_has_trailing_newline(self):
        manifest = {"test": True}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "manifest.json")
            write(manifest, path)
            with open(path) as f:
                content = f.read()
            assert content.endswith("\n")

    def test_sorts_keys(self):
        manifest = {"z_key": 1, "a_key": 2, "m_key": 3}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "manifest.json")
            write(manifest, path)
            with open(path) as f:
                content = f.read()
            # a_key should come before m_key before z_key
            assert content.index("a_key") < content.index("m_key") < content.index("z_key")


# ---------------------------------------------------------------------------
# _git_dirty — regression test for output-file filtering fix (30daf9c9)
# ---------------------------------------------------------------------------


class TestGitDirtyFiltering:
    """The dirty check must exclude results/ and .beads/ output files.

    Regression test for the bug where _git_dirty() counted ALL git status
    lines, causing every benchmark run that wrote its own output to report
    dirty=true (PLAN.md §9 violation). Fixed in 30daf9c9.
    """

    def test_clean_tree_returns_false(self, monkeypatch):
        """Empty git status → not dirty."""
        monkeypatch.setattr(manifest_mod, "_git_porcelain", lambda: "")
        from bench.manifest import _git_dirty

        assert _git_dirty() is False

    def test_only_results_files_returns_false(self, monkeypatch):
        """Untracked results/ files should NOT count as dirty."""
        monkeypatch.setattr(
            manifest_mod,
            "_git_porcelain",
            lambda: "?? results/test.csv\n?? results/manifests/foo.json\n",
        )
        from bench.manifest import _git_dirty

        assert _git_dirty() is False

    def test_only_beads_files_returns_false(self, monkeypatch):
        """Untracked .beads/ files should NOT count as dirty."""
        monkeypatch.setattr(
            manifest_mod,
            "_git_porcelain",
            lambda: "?? .beads/issues.jsonl\n?? .beads/dolt/HEAD\n",
        )
        from bench.manifest import _git_dirty

        assert _git_dirty() is False

    def test_modified_results_file_returns_false(self, monkeypatch):
        """Modified (tracked) results/ file should NOT count as dirty."""
        monkeypatch.setattr(
            manifest_mod,
            "_git_porcelain",
            lambda: " M results/raw/rk3588-t4_big.csv\n",
        )
        from bench.manifest import _git_dirty

        assert _git_dirty() is False

    def test_source_change_returns_true(self, monkeypatch):
        """Real source change should count as dirty."""
        monkeypatch.setattr(
            manifest_mod,
            "_git_porcelain",
            lambda: " M bench/manifest.py\n",
        )
        from bench.manifest import _git_dirty

        assert _git_dirty() is True

    def test_source_change_plus_output_returns_true(self, monkeypatch):
        """Source change + output files → still dirty (source change dominates)."""
        monkeypatch.setattr(
            manifest_mod,
            "_git_porcelain",
            lambda: " M bench/manifest.py\n?? results/test.csv\n?? .beads/issues.jsonl\n",
        )
        from bench.manifest import _git_dirty

        assert _git_dirty() is True

    def test_mixed_output_only_returns_false(self, monkeypatch):
        """Multiple output dirs, no source → not dirty."""
        monkeypatch.setattr(
            manifest_mod,
            "_git_porcelain",
            lambda: (
                " M results/raw/jetson-j1.csv\n"
                "?? results/manifests/new.json\n"
                "?? .beads/issues.jsonl\n"
                "?? .beads/dolt/HEAD\n"
            ),
        )
        from bench.manifest import _git_dirty

        assert _git_dirty() is False

    def test_beads_gate_lock_gitignored(self):
        """Root-level .beads.gate.lock (created by in-flight bd writes) must
        be in .gitignore so it never triggers false dirty=true in manifests.

        Regression test for ob-k0oz: the lock file is operational scratch,
        not a source change, but _git_dirty()'s output-dir regex only
        matches ``results/`` and ``.beads/`` (directory), not root-level
        ``.beads.gate.lock``.
        """
        from pathlib import Path

        gi = Path(__file__).resolve().parent.parent / ".gitignore"
        assert ".beads.gate.lock" in gi.read_text()

    def test_none_on_git_failure(self, monkeypatch):
        """When _git_porcelain returns None (git not available), _git_dirty returns None."""
        monkeypatch.setattr(manifest_mod, "_git_porcelain", lambda: None)
        from bench.manifest import _git_dirty

        assert _git_dirty() is None


# ---------------------------------------------------------------------------
# _default_run_id
# ---------------------------------------------------------------------------


class TestDefaultRunId:
    def test_format_contains_hostname_timestamp_sha(self):
        rid = _default_run_id()
        # hostname_timestamp_sha — all three parts joined by _
        parts = rid.split("_")
        # hostname might contain hyphens but the parts are: host, timestamp, sha
        assert len(parts) >= 3
        # Last part is short git sha or "nogit"
        last = parts[-1]
        assert len(last) == 7  # 7-char short sha or "nogit"

    def test_sanitizes_hostname(self):
        """Hostname with special chars should be replaced with hyphens."""
        # We can't control the hostname, but we can verify the ID is clean
        rid = _default_run_id()
        # Only alphanumerics, hyphens, dots, underscores
        assert re.match(r"^[A-Za-z0-9._-]+$", rid), f"Unexpected chars in run_id: {rid}"


# ---------------------------------------------------------------------------
# _read_int_file
# ---------------------------------------------------------------------------


class TestReadIntFile:
    def test_valid_integer(self, tmp_path):
        f = tmp_path / "freq"
        f.write_text("1800000\n")
        assert _read_int_file(str(f)) == 1800000

    def test_non_integer_returns_none(self, tmp_path):
        f = tmp_path / "bad"
        f.write_text("not_a_number\n")
        assert _read_int_file(str(f)) is None

    def test_missing_file_returns_none(self):
        assert _read_int_file("/nonexistent/path/file") is None


# ---------------------------------------------------------------------------
# _core_count
# ---------------------------------------------------------------------------


class TestCoreCount:
    def test_returns_int_or_none(self):
        result = _core_count()
        assert result is None or isinstance(result, int)


# ---------------------------------------------------------------------------
# _affinity_count — taskset-aware CPU count
# ---------------------------------------------------------------------------


class TestAffinityCount:
    def test_returns_int_or_none(self):
        result = _affinity_count()
        assert result is None or isinstance(result, int)

    def test_respects_affinity_mask(self):
        """On Linux, _affinity_count should be ≤ _core_count."""
        aff = _affinity_count()
        cores = _core_count()
        if aff is not None and cores is not None:
            assert aff <= cores


# ---------------------------------------------------------------------------
# _isa_features — non-aarch64 early return
# ---------------------------------------------------------------------------


class TestIsaFeatures:
    def test_returns_dict_on_aarch64_or_none(self):
        """On non-aarch64, _isa_features returns None."""
        result = _isa_features()
        # On this device (aarch64), should return a dict; on x86 CI, None
        assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# _meminfo
# ---------------------------------------------------------------------------


class TestMeminfo:
    def test_returns_dict_or_none(self):
        result = _meminfo()
        assert result is None or isinstance(result, dict)
        if result is not None:
            assert "mem_total_kb" in result
            assert "mem_available_kb" in result


# ---------------------------------------------------------------------------
# _optional_package_versions — PackageNotFoundError path
# ---------------------------------------------------------------------------


class TestOptionalPackageVersions:
    def test_returns_dict(self):
        result = _optional_package_versions()
        assert isinstance(result, dict)

    def test_missing_packages_are_none(self):
        """At least some optional packages may not be installed."""
        result = _optional_package_versions()
        # Values should be strings (version) or None (not installed)
        for v in result.values():
            assert v is None or isinstance(v, str)


# ---------------------------------------------------------------------------
# _parallelism — OMP_NUM_THREADS edge cases
# ---------------------------------------------------------------------------


class TestParallelism:
    def test_non_numeric_omp_threads(self, monkeypatch):
        """Non-numeric OMP_NUM_THREADS falls back to affinity or core count."""
        monkeypatch.setenv("OMP_NUM_THREADS", "not_a_number")
        result = _parallelism()
        assert result["omp_num_threads"] == "not_a_number"
        # effective_threads should fall back to affinity mask or core count
        assert result["threads_source"] in ("affinity_mask", "core_count_default")

    def test_valid_omp_threads(self, monkeypatch):
        """Numeric OMP_NUM_THREADS is parsed correctly."""
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        result = _parallelism()
        assert result["omp_num_threads"] == "4"
        assert result["effective_threads"] == 4
        assert result["threads_source"] == "OMP_NUM_THREADS"

    def test_unset_omp_threads(self, monkeypatch):
        """Unset OMP_NUM_THREADS falls back to affinity or core count."""
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        result = _parallelism()
        assert result["omp_num_threads"] is None
        # On Linux, falls back to affinity mask (respects taskset);
        # on other platforms, falls back to raw core count.
        assert result["threads_source"] in ("affinity_mask", "core_count_default")
        # New transparency fields should always be present
        assert "affinity_core_count" in result
        assert "system_core_count" in result


# ---------------------------------------------------------------------------
# Platform-dependent edge cases (lines 149, 187, 193, 232, 235, 238, 261, 283, 311-312)
# ---------------------------------------------------------------------------


class TestCpuModelFallback:
    """Cover _cpu_model fallback to platform.processor (line 149)."""

    def test_no_match_falls_back_to_processor(self, monkeypatch):
        """When /proc/cpuinfo has no model name, falls back to platform.processor."""
        monkeypatch.setattr(manifest_mod, "_read_text", lambda path: "unrelated: text\n")
        monkeypatch.setattr(manifest_mod.platform, "processor", lambda: "Generic CPU")
        assert _cpu_model() == "Generic CPU"

    def test_no_text_no_processor(self, monkeypatch):
        """When /proc/cpuinfo unreadable and processor empty, returns None."""
        monkeypatch.setattr(manifest_mod, "_read_text", lambda path: None)
        monkeypatch.setattr(manifest_mod.platform, "processor", lambda: "")
        assert _cpu_model() is None

    def test_hardware_key_match(self, monkeypatch):
        """When /proc/cpuinfo has 'Hardware' key, returns that value."""
        monkeypatch.setattr(manifest_mod, "_read_text", lambda path: "Hardware : Radxa RK3588\n")
        assert _cpu_model() == "Radxa RK3588"

    def test_model_name_key_match(self, monkeypatch):
        """When /proc/cpuinfo has 'model name' key (x86 style), returns it."""
        monkeypatch.setattr(
            manifest_mod, "_read_text", lambda path: "model name : Intel i7-12700K\n"
        )
        assert _cpu_model() == "Intel i7-12700K"


class TestCpufreqTopologyMissing:
    """Cover _cpufreq_topology when /sys tree absent (lines 187, 193)."""

    def test_no_sys_tree_returns_none(self, monkeypatch):
        monkeypatch.setattr(manifest_mod.os.path, "isdir", lambda p: False)
        assert _cpufreq_topology() is None

    def test_empty_cpu_dirs_returns_none(self, monkeypatch):
        """sys tree exists but no cpu[0-9]* dirs → None."""
        monkeypatch.setattr(manifest_mod.os.path, "isdir", lambda p: True)
        monkeypatch.setattr(manifest_mod, "glob", type("G", (), {"glob": lambda *a: []}))
        assert _cpufreq_topology() is None


class TestIsaFeaturesEdge:
    """Cover _isa_features non-aarch64 and empty paths (lines 232, 235, 238)."""

    def test_non_aarch64_returns_none(self, monkeypatch):
        monkeypatch.setattr(manifest_mod, "_machine_arch", lambda: "x86_64")
        assert _isa_features() is None

    def test_no_cpuinfo_returns_none(self, monkeypatch):
        monkeypatch.setattr(manifest_mod, "_machine_arch", lambda: "aarch64")
        monkeypatch.setattr(manifest_mod, "_read_text", lambda path: None)
        assert _isa_features() is None

    def test_no_features_line_returns_none(self, monkeypatch):
        monkeypatch.setattr(manifest_mod, "_machine_arch", lambda: "aarch64")
        monkeypatch.setattr(manifest_mod, "_read_text", lambda path: "processor : 0\n")
        assert _isa_features() is None


class TestThermalZonesMissing:
    """Cover _thermal_zones when /sys/class/thermal absent (line 261)."""

    def test_no_thermal_dir_returns_none(self, monkeypatch):
        monkeypatch.setattr(manifest_mod.os.path, "isdir", lambda p: False)
        assert _thermal_zones() is None


class TestMeminfoMissing:
    """Cover _meminfo when /proc/meminfo unreadable (line 283)."""

    def test_no_meminfo_returns_none(self, monkeypatch):
        monkeypatch.setattr(manifest_mod, "_read_text", lambda path: None)
        assert _meminfo() is None


class TestOptionalPackageVersionsException:
    """Cover _optional_package_versions generic exception handler (lines 311-312)."""

    def test_generic_exception_handled(self, monkeypatch):
        """A non-PackageNotFoundError exception is caught → value is None."""

        def boom(name):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(manifest_mod, "_pkg_version", boom)
        result = _optional_package_versions()
        # All should be None since every call raises RuntimeError
        assert all(v is None for v in result.values())
