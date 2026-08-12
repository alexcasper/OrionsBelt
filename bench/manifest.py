# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Run manifest — provenance capture for benchmark runs.

Per docs/archive/PLAN.md section 9: "a number without a manifest is not a result." On a
passively-cooled edge board, thermal state alone can move throughput enough to
invalidate a comparison, so this module exists to make that state visible and
recorded rather than assumed. ``docs/RESULTS_SCHEMA.md`` and ``bench/schema.py``
already define the CSV-side ``manifest_ref`` column that points at what this
module produces; this is the other half of that contract.

Targets Python 3.10+, and uses no third-party dependencies -- only stdlib
modules (``json``, ``os``, ``platform``, ``socket``, ``subprocess``, ``sys``,
``importlib.metadata``, ``pathlib``, ``re``, ``datetime``, ``glob``). Same
stdlib-only rule as ``bench/schema.py``: this has to import cleanly in the NOE
Compiler's Python 3.10 environment and on the board, where we do not control
the dependency set.

**Design rule: degrade gracefully on any platform.** This runs on an x86 dev
host today and an aarch64 board later, and it must never take a benchmark run
down. Every individual probe (a `/proc` read, a `/sys` walk, an import) is
wrapped in its own try/except and records ``None``/absent on failure rather
than raising. ``capture()`` itself never raises for probe failures; the worst
case is a manifest with more nulls in it, not a crashed harness. ``write()``
following the same rule is why callers can fire-and-forget it after a run.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

# ISA features we care about because the CPU kernel dispatch ladder is
# SVE (1 or 2) -> NEON -> scalar (docs/FINDINGS.md section 4), and the
# quantized matmul path additionally cares about i8mm/bf16/asimddp/sme
# (docs/FINDINGS.md section 3.1, 3.3).
_ISA_FEATURES_OF_INTEREST = ("sve", "sve2", "i8mm", "bf16", "asimddp", "sme")

# Packages whose versions are worth recording if importable, without ever
# requiring them (bench/schema.py and this module both run stdlib-only; the
# model/inference stack is a separate, optional concern).
_OPTIONAL_PACKAGES = ("torch", "transformers", "onnxruntime", "numpy")


def _safe(fn, *args, **kwargs):
    """Run fn, returning None on any exception instead of propagating it.

    The one helper this module leans on throughout: every probe is a call
    through this wrapper so a single missing /sys file, an absent binary, or
    a permissions error degrades to a null field rather than crashing the
    benchmark run that asked for a manifest.
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _read_text(path: str) -> str | None:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _run(cmd: list[str]) -> str | None:
    out = subprocess.run(  # noqa: S603 -- fixed argv lists, no shell, no user input
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=5,
        check=True,
    )
    return out.stdout.decode("utf-8", errors="replace").strip()


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------


def _utc_timestamp() -> str:
    """ISO 8601 UTC timestamp, e.g. 2026-08-10T14:30:00Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str | None:
    return _safe(_run, ["git", "rev-parse", "HEAD"])


def _git_dirty() -> bool | None:
    """Check whether the working tree has source changes.

    Excludes ``results/`` and ``.beads/`` because those are output data
    produced by benchmark runs, not source-code changes that invalidate
    provenance. This mirrors the filtering in ``scripts/capture_manifest.sh``
    (line ~119): writing a CSV or manifest JSON should not mark the tree
    dirty. Without this filter, every benchmark run that writes its own
    output reports ``git.dirty=true``, which per PLAN.md §9 invalidates the
    result for no real reason.
    """
    # NOTE: _run() calls .strip() which removes the leading space from git
    # status --porcelain lines (format: "XY path"). That leading space is what
    # the _OUTPUT_RE regex matches on to filter out results/ and .beads/
    # changes.  So we use _git_porcelain() which preserves leading whitespace.
    status = _git_porcelain()
    if status is None:
        return None
    # Filter lines that are only output dirs (results/ or .beads/).
    # git status --porcelain format: "XY path" where X/Y are status codes.
    _OUTPUT_RE = re.compile(r"^[ ?][M?] (results/|\.beads/)")
    filtered = [line for line in status.splitlines() if not _OUTPUT_RE.match(line)]
    return len(filtered) > 0


def _git_porcelain() -> str | None:
    """Return raw ``git status --porcelain`` output (leading whitespace preserved).

    Unlike :func:`_run`, this does **not** call ``.strip()`` so that the
    leading space in lines like ``" M results/..."`` is preserved for the
    :func:`_git_dirty` regex to match on.
    """
    try:
        result = subprocess.run(  # noqa: S603 -- fixed argv, no shell, no user input
            ["git", "status", "--porcelain"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=True,
        )
        return result.stdout.decode("utf-8", errors="replace")
    except Exception:
        return None


def _default_run_id() -> str:
    """Fallback run_id when the caller doesn't supply one.

    Mirrors the documented convention in docs/RESULTS_SCHEMA.md section 2:
    ``<device>_<yyyymmddTHHMMSSZ>_<short_git_sha>`` -- but this module doesn't
    know "device" (that's a harness/schema concept), so it uses the hostname
    as a stand-in when called standalone (e.g. via __main__).
    """
    host = _safe(socket.gethostname) or "unknown"
    host = re.sub(r"[^A-Za-z0-9_.-]", "-", host)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = _git_sha()
    short_sha = sha[:7] if sha else "nogit"
    return f"{host}_{stamp}_{short_sha}"


# ---------------------------------------------------------------------------
# Host / device
# ---------------------------------------------------------------------------


def _hostname() -> str | None:
    return _safe(socket.gethostname)


def _kernel_version() -> str | None:
    return _safe(platform.release)


def _machine_arch() -> str | None:
    return _safe(platform.machine)


def _cpu_model() -> str | None:
    """Best-effort human-readable CPU model string.

    /proc/cpuinfo's "model name" (x86) or the aarch64-equivalent "Hardware"/
    "model name" fields; falls back to platform.processor() which is often
    empty on Linux but costs nothing to try.
    """
    text = _safe(_read_text, "/proc/cpuinfo")
    if text:
        for key in ("model name", "Hardware", "Model"):
            match = re.search(rf"^{re.escape(key)}\s*:\s*(.+)$", text, re.MULTILINE)
            if match:
                return match.group(1).strip()
    return _safe(platform.processor) or None


def _core_count() -> int | None:
    return _safe(os.cpu_count)


def _read_int_file(path: str) -> int | None:
    text = _safe(_read_text, path)
    if text is None:
        return None
    try:
        return int(text.strip())
    except ValueError:
        return None


def _cpufreq_topology() -> list[dict[str, Any]] | None:
    """Per-CPU big.LITTLE-relevant topology from /sys/devices/system/cpu.

    Reads cpufreq/{scaling_max_freq,scaling_min_freq,scaling_governor} and
    cpu_capacity per logical CPU. On the CIX P1 the Cortex-A720 "big",
    A720 "medium", and A520 "little" clusters have distinct max frequencies,
    so this is how the cluster shape becomes visible without hardcoding
    a topology this module has no way to know in general. Returns None
    (rather than []) if the whole /sys/devices/system/cpu tree is absent
    (e.g. inside some containers), so callers can tell "no cpus found" from
    "found zero cpus" -- an absent tree is a different failure than an empty
    result.

    Individual per-CPU fields that are missing (e.g. no cpufreq on this
    platform, or a virtualized x86 host with no scaling driver) come back
    as None inside each entry rather than omitting the entry, so the entry
    count still reflects how many logical CPUs exist.
    """
    base = "/sys/devices/system/cpu"
    if not os.path.isdir(base):
        return None
    cpu_dirs = sorted(
        glob.glob(os.path.join(base, "cpu[0-9]*")),
        key=lambda p: int(re.search(r"cpu(\d+)$", p).group(1)),
    )
    if not cpu_dirs:
        return None

    topology = []
    for cpu_dir in cpu_dirs:
        cpu_id = int(re.search(r"cpu(\d+)$", cpu_dir).group(1))
        entry: dict[str, Any] = {
            "cpu": cpu_id,
            "max_freq_khz": _read_int_file(os.path.join(cpu_dir, "cpufreq", "scaling_max_freq")),
            "min_freq_khz": _read_int_file(os.path.join(cpu_dir, "cpufreq", "scaling_min_freq")),
            "governor": _safe(
                lambda d=cpu_dir: _read_text(os.path.join(d, "cpufreq", "scaling_governor")).strip()
            ),
            "cpu_capacity": _read_int_file(os.path.join(cpu_dir, "cpu_capacity")),
        }
        topology.append(entry)
    return topology


# ---------------------------------------------------------------------------
# ISA features (aarch64)
# ---------------------------------------------------------------------------


def _isa_features() -> dict[str, bool] | None:
    """Parse /proc/cpuinfo's aarch64 "Features" line for the ISA bits we care about.

    This matters because our CPU kernels dispatch SVE -> NEON -> scalar
    (docs/FINDINGS.md section 4), so a manifest that only records the target
    architecture and not which extensions are actually present at runtime
    cannot explain which path ran. Deliberately keyed on what actually
    appears in /proc/cpuinfo (not on platform.machine()=='aarch64' alone),
    since the same binary can run in an emulator that reports aarch64 but
    lacks a feature, or vice versa.

    Returns None on non-aarch64 (or if /proc/cpuinfo is unreadable / has no
    Features line) since the concept doesn't apply -- e.g. this always comes
    back None on the x86 dev host this module is verified against.
    """
    if _machine_arch() not in ("aarch64", "arm64"):
        return None
    text = _safe(_read_text, "/proc/cpuinfo")
    if not text:
        return None
    match = re.search(r"^Features\s*:\s*(.+)$", text, re.MULTILINE)
    if not match:
        return None
    present = set(match.group(1).split())
    return {feature: feature in present for feature in _ISA_FEATURES_OF_INTEREST}


# ---------------------------------------------------------------------------
# Thermal
# ---------------------------------------------------------------------------


def _thermal_zones() -> list[dict[str, Any]] | None:
    """Snapshot of /sys/class/thermal/thermal_zone*/{temp,type} at capture time.

    On a passively-cooled edge board this is the whole point (docs/archive/PLAN.md
    section 9): thermal state alone can move throughput enough to invalidate
    a comparison. temp is left in the raw millidegree-C integer /sys reports
    (not divided down) so this module makes no assumption about a particular
    thermal driver's scaling. Returns None if the thermal_zone tree doesn't
    exist at all (common on plain x86 dev hosts / some VMs); returns [] only
    if the tree exists but is empty.
    """
    zones = sorted(glob.glob("/sys/class/thermal/thermal_zone*"))
    if not os.path.isdir("/sys/class/thermal"):
        return None
    result = []
    for zone_dir in zones:
        result.append(
            {
                "zone": os.path.basename(zone_dir),
                "type": _safe(lambda d=zone_dir: _read_text(os.path.join(d, "type")).strip()),
                "temp_millicelsius": _read_int_file(os.path.join(zone_dir, "temp")),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def _meminfo() -> dict[str, int | None] | None:
    """MemTotal / MemAvailable (kB, as reported) from /proc/meminfo."""
    text = _safe(_read_text, "/proc/meminfo")
    if text is None:
        return None
    result: dict[str, int | None] = {"mem_total_kb": None, "mem_available_kb": None}
    for key, field in (("MemTotal", "mem_total_kb"), ("MemAvailable", "mem_available_kb")):
        match = re.search(rf"^{key}:\s*(\d+)\s*kB", text, re.MULTILINE)
        if match:
            result[field] = int(match.group(1))
    return result


# ---------------------------------------------------------------------------
# Software
# ---------------------------------------------------------------------------


def _optional_package_versions() -> dict[str, str | None]:
    """Versions of torch/transformers/onnxruntime/numpy, if importable.

    Uses importlib.metadata rather than importing the packages themselves,
    so this never pays the cost (or risk) of actually importing a heavy ML
    library just to ask its version, and never requires any of them to be
    installed -- per the task's stdlib-only / "do not require them" rule.
    """
    versions: dict[str, str | None] = {}
    for name in _OPTIONAL_PACKAGES:
        try:
            versions[name] = _pkg_version(name)
        except PackageNotFoundError:
            versions[name] = None
        except Exception:
            versions[name] = None
    return versions


# ---------------------------------------------------------------------------
# capture / write / manifest_ref
# ---------------------------------------------------------------------------


def _parallelism() -> dict[str, Any]:
    """Thread-count environment, because it is now a 4x experimental variable.

    Once the kernels gained OpenMP, a single-threaded and a 4-core run of the
    *same commit on the same device* differ by 3-4x — and nothing in the manifest
    recorded which one you were looking at. That is not hypothetical:
    ``jetson-j1_clean.csv`` was captured as a clean-tree run to answer ob-bf7 and
    reads 2.9-4.1x its predecessor, which is the OpenMP speedup rather than any
    provenance effect. Dropping it into the single-threaded fleet table would have
    inverted the project's central result by making the A57 look faster than the
    Pi 5.

    ``omp_num_threads`` is the environment variable as set (None if unset, which
    means OpenMP defaults to one thread per core); ``effective_threads`` is the
    best available guess at what actually ran.
    """
    env = os.environ.get("OMP_NUM_THREADS")
    threads: int | None = None
    if env:
        try:
            threads = int(env)
        except ValueError:
            threads = None
    return {
        "omp_num_threads": env,
        "omp_proc_bind": os.environ.get("OMP_PROC_BIND"),
        "omp_places": os.environ.get("OMP_PLACES"),
        # With OMP_NUM_THREADS unset, libgomp defaults to the number of available
        # CPUs, so record that as the effective count rather than leaving it null.
        "effective_threads": threads if threads is not None else _safe(_core_count),
        "threads_source": "OMP_NUM_THREADS" if threads is not None else "core_count_default",
    }


def capture(**caller_fields: Any) -> dict[str, Any]:
    """Capture a complete run manifest as a plain JSON-serializable dict.

    ``caller_fields`` are additional fields the caller wants recorded verbatim
    under ``manifest["caller"]`` -- this is how compiler flags and the
    layer-to-engine dispatch path actually used for a given run get into the
    manifest, since this module has no way to know them on its own (they are
    a harness/optimization-implementation concern, not a host-probing one).

    Never raises for probe failures: every probe below is individually
    wrapped, so the worst outcome is a manifest with more null fields, never
    an exception that takes a benchmark run down with it.
    """
    run_id = caller_fields.pop("run_id", None) or _safe(_default_run_id) or "unknown_run"

    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "run_id": run_id,
        "timestamp_utc": _safe(_utc_timestamp),
        "git": {
            "sha": _safe(_git_sha),
            "dirty": _safe(_git_dirty),
        },
        "host": {
            "hostname": _safe(_hostname),
            "machine": _safe(_machine_arch),
            "kernel": _safe(_kernel_version),
            "os": _safe(platform.platform),
            "cpu_model": _safe(_cpu_model),
            "core_count": _safe(_core_count),
            "cpu_topology": _safe(_cpufreq_topology),
        },
        "isa_features": _safe(_isa_features),
        "parallelism": _safe(_parallelism),
        "thermal_zones": _safe(_thermal_zones),
        "memory": _safe(_meminfo),
        "software": {
            "python_version": sys.version,
            "python_implementation": _safe(platform.python_implementation),
            "packages": _safe(_optional_package_versions) or {},
        },
        "caller": dict(caller_fields),
    }
    return manifest


def manifest_ref(run_id: str, *, results_dir: str = "results/manifests") -> str:
    """Relative-to-repo-root path for a run's manifest, for the CSV's manifest_ref column.

    docs/RESULTS_SCHEMA.md section 3 documents this exact form, e.g.
    ``results/manifests/o6_20260810T143000Z_a1b2c3d.json``. Uses posixpath-style
    forward slashes regardless of host OS since this string lands in a
    committed CSV, not a filesystem call.
    """
    return f"{results_dir}/{run_id}.json"


def write(manifest: dict[str, Any], path: str | os.PathLike) -> None:
    """Write manifest to path as JSON, creating parent directories as needed.

    Deliberately narrow: this function's only failure mode should be "disk
    write failed" (permissions, out of space, bad path), which is a real
    problem worth surfacing -- unlike the individual probes in capture(),
    which must never raise. Callers who want the "manifest must never take
    down a run" property end-to-end should wrap this call, not this function
    itself, e.g.::

        try:
            manifest.write(manifest.capture(...), path)
        except Exception:
            logging.exception("failed to write run manifest to %s", path)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = [
    "capture",
    "write",
    "manifest_ref",
]


if __name__ == "__main__":
    print(json.dumps(capture(), indent=2, sort_keys=True))
