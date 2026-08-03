"""Runtime ISA feature detection for aarch64 devices.

Per bead ob-ng6: confirm the feature flags are actually active at runtime
rather than merely compiled in.  This module reads ``/proc/cpuinfo`` and
maps the HWCAP feature strings to the dispatch decisions the kernel code
makes (NEON, dotprod, i8mm, SVE, SVE2, bf16).

Usage (CLI)::

    python3 -m orionsbelt.engines.cpu.isa_detect

Usage (library)::

    from orionsbelt.engines.cpu.isa_detect import detect_features, FeatureSet
    fs = detect_features()
    print(fs.summary())

The module is pure-stdlib so it runs on any Python 3.8+, including the
Jetson Nano's Python 3.6.9 (no f-strings needed for core logic).
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field

# ---------------------------------------------------------------------------
# HWCAP feature string → human-readable mapping
# (matches Linux kernel arch/arm64/include/uapi/asm/hwcap.h)
# ---------------------------------------------------------------------------
_HWCAP_MAP: dict[str, str] = {
    "fp": "Floating-point (fp)",
    "asimd": "NEON / Advanced SIMD",
    "evtstrm": "Event stream",
    "aes": "AES",
    "pmull": "PMULL",
    "sha1": "SHA1",
    "sha2": "SHA2 / SHA256",
    "crc32": "CRC32",
    "atomics": "Atomic instructions (LSE)",
    "fphp": "Half-precision FP",
    "asimdhp": "Half-precision SIMD",
    "cpuid": "CPUID register",
    "asimdrdm": "RDM (rounding double multiply-add)",
    "jscvt": "JSCVT",
    "fcma": "FCMA",
    "lrcpc": "Release-consistent PC",
    "dcpop": "DC POP (data cache clean to PoP)",
    "sha3": "SHA3",
    "sm3": "SM3",
    "sm4": "SM4",
    "asimddp": "Dot-product (dotprod / SDOT/UDOT)",
    "sha512": "SHA512",
    "sve": "SVE (Scalable Vector Extension)",
    "frint": "FRINT",
    "sve2": "SVE2",
    "svebf16": "SVE BF16",
    "i8mm": "Int8 matrix multiply (i8mm)",
    "bf16": "BF16 (BFloat16)",
    "dgh": "DGH",
    "rng": "Random number generator",
}

# Features that drive kernel dispatch decisions
_DISPATCH_FEATURES = [
    "asimd",  # NEON — baseline for all Armv8-A
    "asimddp",  # dotprod — Cortex-A55/A75+, enables SDOT/UDOT
    "i8mm",  # int8 matmul — needed for INT8 GEMM kernels
    "sve",  # SVE1 — Cortex-A76+? No, only Armv8.6+ / Armv9
    "sve2",  # SVE2 — Armv9-A
    "bf16",  # BF16 — for mixed-precision kernels
    "fphp",  # FP16 — half-precision FP
    "asimdhp",  # SIMD HP — half-precision SIMD
]


@dataclass
class FeatureSet:
    """Detected ISA feature set for the current device."""

    machine: str = ""
    features_raw: str = ""
    features: list[str] = field(default_factory=list)
    dispatch_features: dict[str, bool] = field(default_factory=dict)
    recommended_binary: str = ""
    cpu_model: str = ""
    core_count: int = 0
    cpu_part: str = ""
    cpu_implementation: str = ""

    def summary(self) -> str:
        """Human-readable one-line summary of the detected feature set."""
        active = [f for f in _DISPATCH_FEATURES if self.dispatch_features.get(f)]
        return "active ISA: " + ", ".join(active) if active else "active ISA: (none)"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _read_cpuinfo(path: str = "/proc/cpuinfo") -> str:
    """Read /proc/cpuinfo, returning empty string if unavailable."""
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def _parse_cpuinfo(cpuinfo: str) -> dict[str, str]:
    """Extract fields from the first CPU block of /proc/cpuinfo."""
    result: dict[str, str] = {}
    for line in cpuinfo.splitlines():
        line = line.strip()
        if not line:
            # First blank line = end of first CPU block
            if result:
                break
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lower().replace(" ", "_")
            val = val.strip()
            if key not in result:
                result[key] = val
    return result


def detect_features(cpuinfo_path: str = "/proc/cpuinfo") -> FeatureSet:
    """Detect ISA features at runtime from /proc/cpuinfo.

    This is the core function for bead ob-ng6: it confirms which features
    are *actually present* on the running device, not just what was compiled in.
    """
    raw = _read_cpuinfo(cpuinfo_path)
    parsed = _parse_cpuinfo(raw)

    features_str = parsed.get("features", "")
    features_list = features_str.split() if features_str else []

    # Map dispatch-critical features to booleans
    dispatch = {}
    for feat in _DISPATCH_FEATURES:
        dispatch[feat] = feat in features_list

    # Determine recommended binary variant based on features
    recommended = _recommend_binary(dispatch)

    # Count total CPU blocks by counting 'processor :' lines
    core_count = raw.count("processor")
    if core_count == 0:
        core_count = 1

    fs = FeatureSet(
        machine=platform.machine(),
        features_raw=features_str,
        features=features_list,
        dispatch_features=dispatch,
        recommended_binary=recommended,
        cpu_model=parsed.get("cpu_model", ""),
        core_count=core_count,
        cpu_part=parsed.get("cpu_part", ""),
        cpu_implementation=parsed.get("cpu_implementer", ""),
    )
    return fs


def _recommend_binary(dispatch: dict[str, bool]) -> str:
    """Recommend the most specific bench binary for the detected features."""
    if dispatch.get("sve2"):
        return "armv9sve2"
    if dispatch.get("i8mm"):
        return "armv8.6i8mm"
    if dispatch.get("asimddp"):
        return "armv8.2dot"
    if dispatch.get("asimd"):
        return "armv8a"
    return "scalar"


# CPU part number → common name mapping (for documentation)
_CPU_PARTS: dict[str, str] = {
    "0xd01": "Cortex-A32",
    "0xd03": "Cortex-A53",
    "0xd04": "Cortex-A35",
    "0xd05": "Cortex-A55",
    "0xd07": "Cortex-A57",
    "0xd08": "Cortex-A72",
    "0xd09": "Cortex-A73",
    "0xd0a": "Cortex-A75",
    "0xd0b": "Cortex-A76",
    "0xd0c": "Neoverse-N1",
    "0xd0d": "Cortex-A77",
    "0xd40": "Neoverse-V1",
    "0xd41": "Cortex-A78",
    "0xd44": "Cortex-X1C",
    "0xd4a": "Neoverse-E1",
    "0xd4b": "Cortex-A78C",
    "0xd4d": "Cortex-A715",
    "0xd4e": "Cortex-X4",
    "0xd4f": "Neoverse-V2",
    "0xd80": "Cortex-A520",
    "0xd81": "Cortex-A720",
}


def cpu_part_name(part_hex: str) -> str:
    """Return a human-readable CPU name for a CPU part number, or the raw value."""
    return _CPU_PARTS.get(part_hex.lower(), part_hex or "unknown")


def main() -> int:
    """CLI entry point — print detected features as JSON."""
    fs = detect_features()
    part_name = cpu_part_name(fs.cpu_part)

    # Enrich output with part name
    d = fs.to_dict()
    d["cpu_part_name"] = part_name
    d["active_dispatch_features"] = [f for f in _DISPATCH_FEATURES if fs.dispatch_features.get(f)]
    print(json.dumps(d, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
