# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

"""Tests for runtime ISA feature detection (bead ob-ng6)."""

import json as _json
import os
import sys
import textwrap
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orionsbelt.engines.cpu.isa_detect import (
    FeatureSet,
    _parse_cpuinfo,
    _read_cpuinfo,
    _recommend_binary,
    cpu_part_name,
    detect_features,
    main,
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


class TestFeatureSetMethods:
    """Cover summary/to_dict/to_json on FeatureSet."""

    def test_summary_no_active_features(self):
        """summary() returns '(none)' when no dispatch features are active."""
        fs = FeatureSet(dispatch_features={"asimd": False, "asimddp": False})
        assert "(none)" in fs.summary()

    def test_to_dict(self):
        """to_dict returns all dataclass fields."""
        fs = FeatureSet(machine="aarch64", core_count=4)
        d = fs.to_dict()
        assert d["machine"] == "aarch64"
        assert d["core_count"] == 4
        assert "features_raw" in d
        assert "dispatch_features" in d

    def test_to_json(self):
        """to_json returns valid JSON string."""
        fs = FeatureSet(machine="aarch64", core_count=8)
        s = fs.to_json()
        parsed = _json.loads(s)
        assert parsed["machine"] == "aarch64"
        assert parsed["core_count"] == 8


class TestReadCpuinfo:
    """Cover _read_cpuinfo edge cases."""

    def test_missing_file_returns_empty(self):
        """Missing file returns empty string."""
        assert _read_cpuinfo("/nonexistent/path/cpuinfo") == ""

    def test_valid_file(self, tmp_path):
        """Valid file returns its content."""
        p = tmp_path / "cpuinfo"
        p.write_text("processor : 0\n")
        assert "processor" in _read_cpuinfo(str(p))


class TestParseCpuinfo:
    """Cover _parse_cpuinfo parsing."""

    def test_empty_string(self):
        """Empty string returns empty dict."""
        assert _parse_cpuinfo("") == {}

    def test_strips_whitespace(self):
        """Values are stripped."""
        text = "CPU part\t:  0xd0b \n"
        result = _parse_cpuinfo(text)
        assert result.get("cpu_part") == "0xd0b"

    def test_first_key_wins(self):
        """First occurrence of a key wins."""
        text = "processor : 0\n\nprocessor : 1\n"
        result = _parse_cpuinfo(text)
        assert result.get("processor") == "0"


class TestCpuPartNameLower:
    """Cover cpu_part_name lowercase normalization."""

    def test_uppercase_hex(self):
        """Uppercase hex is normalized."""
        assert cpu_part_name("0xD0B") == "Cortex-A76"

    def test_mixed_case(self):
        """Mixed case hex is normalized."""
        assert cpu_part_name("0xD0b") == "Cortex-A76"


class TestDetectFeaturesEmpty:
    """Cover detect_features with empty/missing cpuinfo."""

    def test_empty_cpuinfo(self, tmp_path):
        """Empty cpuinfo gives scalar recommendation, core_count=1."""
        p = tmp_path / "cpuinfo"
        p.write_text("")
        fs = detect_features(str(p))
        assert fs.recommended_binary == "scalar"
        assert fs.core_count == 1
        assert fs.features == []


class TestMainCLI:
    """Cover main() CLI entry point."""

    def test_main_prints_json(self, tmp_path, capsys):
        """main() prints JSON with enriched fields."""
        p = tmp_path / "cpuinfo"
        p.write_text(_PI5_CPUINFO)
        with patch("orionsbelt.engines.cpu.isa_detect._read_cpuinfo") as mock_read:
            mock_read.return_value = _PI5_CPUINFO
            rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        parsed = _json.loads(captured.out)
        assert "cpu_part_name" in parsed
        assert "active_dispatch_features" in parsed
        assert parsed["cpu_part_name"] == "Cortex-A76"
