"""Tests for runtime ISA feature detection (bead ob-ng6)."""

import os
import sys
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orionsbelt.engines.cpu.isa_detect import (
    FeatureSet,
    _recommend_binary,
    cpu_part_name,
    detect_features,
)

# Simulated /proc/cpuinfo for a Cortex-A76 (Pi 5)
_PI5_CPUINFO = textwrap.dedent("""\
    processor	: 0
    BogoMIPS	: 108.00
    Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp
    CPU implementer	: 0x41
    CPU architecture: 8
    CPU variant	: 0x4
    CPU part	: 0xd0b
    CPU revision	: 1

    processor	: 1
    BogoMIPS	: 108.00
    Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp
    CPU implementer	: 0x41
    CPU architecture: 8
    CPU variant	: 0x4
    CPU part	: 0xd0b
    CPU revision	: 1
""")

# Simulated /proc/cpuinfo for a Cortex-A57 (Jetson Nano)
_JETSON_CPUINFO = textwrap.dedent("""\
    processor	: 0
    Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics
    CPU implementer	: 0x41
    CPU architecture: 8
    CPU variant	: 0x1
    CPU part	: 0xd07
    CPU revision	: 0

    processor	: 1
    Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics
    CPU implementer	: 0x41
    CPU architecture: 8
    CPU variant	: 0x1
    CPU part	: 0xd07
    CPU revision	: 0
""")


def _write_tmp_cpuinfo(tmp_path, content):
    p = tmp_path / "cpuinfo"
    p.write_text(content)
    return str(p)


def test_pi5_detects_dotprod():
    """Pi 5 (A76) should detect asimd and dotprod but not i8mm/SVE."""
    path = _write_tmp_cpuinfo(__import__("pathlib").Path("/tmp"), _PI5_CPUINFO)
    fs = detect_features(path)
    assert fs.dispatch_features["asimd"] is True
    assert fs.dispatch_features["asimddp"] is True  # dotprod
    assert fs.dispatch_features["i8mm"] is False
    assert fs.dispatch_features["sve"] is False
    assert fs.recommended_binary == "armv8.2dot"
    assert fs.core_count == 2  # two processor blocks in the sample


def test_jetson_no_dotprod():
    """Jetson A57 (Armv8.0) should detect asimd but NOT dotprod."""
    import pathlib
    import tempfile

    d = pathlib.Path(tempfile.mkdtemp())
    path = _write_tmp_cpuinfo(d, _JETSON_CPUINFO)
    fs = detect_features(path)
    assert fs.dispatch_features["asimd"] is True
    assert fs.dispatch_features["asimddp"] is False  # no dotprod on A57
    assert fs.dispatch_features["i8mm"] is False
    assert fs.recommended_binary == "armv8a"
    assert fs.core_count == 2


def test_recommend_binary_hierarchy():
    """Most specific binary is recommended."""
    assert _recommend_binary({"sve2": True}) == "armv9sve2"
    assert _recommend_binary({"i8mm": True}) == "armv8.6i8mm"
    assert _recommend_binary({"asimddp": True}) == "armv8.2dot"
    assert _recommend_binary({"asimd": True}) == "armv8a"
    assert _recommend_binary({}) == "scalar"


def test_cpu_part_name():
    """Known CPU part numbers map to human names."""
    assert cpu_part_name("0xd0b") == "Cortex-A76"
    assert cpu_part_name("0xd07") == "Cortex-A57"
    assert cpu_part_name("0xd81") == "Cortex-A720"
    assert cpu_part_name("0x999") == "0x999"
    assert cpu_part_name("") == "unknown"


def test_featureset_summary():
    """summary() returns active features."""
    fs = FeatureSet(dispatch_features={"asimd": True, "asimddp": True, "i8mm": False})
    s = fs.summary()
    assert "asimd" in s
    assert "asimddp" in s
    assert "i8mm" not in s
