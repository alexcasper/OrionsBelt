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

from bench.manifest import (
    capture,
    write,
    manifest_ref,
    _safe,
    _utc_timestamp,
    _default_run_id,
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
