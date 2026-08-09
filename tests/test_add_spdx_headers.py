#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/add_spdx_headers.py — SPDX header insertion logic."""

import sys
from pathlib import Path

# Make the script importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import add_spdx_headers as ash  # noqa: E402

# ─── _insert_spdx: core text transformation ────────────────────────────────────


class TestInsertSpdxBasic:
    """Verify _insert_spdx handles the common cases correctly."""

    def test_no_shebang(self):
        text = 'print("hello")\n'
        result = ash._insert_spdx(text)
        assert "SPDX-License-Identifier" in result
        assert result.startswith(ash.SPDX)
        assert 'print("hello")' in result

    def test_with_shebang(self):
        text = '#!/usr/bin/env python3\nprint("hello")\n'
        result = ash._insert_spdx(text)
        assert result.startswith("#!/usr/bin/env python3\n")
        assert "# SPDX-FileCopyrightText" in result
        # SPDX header comes after the shebang line
        shebang_pos = result.index("#!/usr/bin/env python3")
        spdx_pos = result.index("SPDX")
        assert spdx_pos > shebang_pos
        shebang_line = result.split("\n")[0]
        assert shebang_line == "#!/usr/bin/env python3"

    def test_blank_line_between_spdx_and_body(self):
        text = "#!/usr/bin/env bash\necho hi\n"
        result = ash._insert_spdx(text)
        lines = result.split("\n")
        # Line 0: shebang, 1-2: SPDX, 3: blank, 4+: body
        assert lines[0] == "#!/usr/bin/env bash"
        assert "SPDX-FileCopyrightText" in lines[1]
        assert "SPDX-License-Identifier" in lines[2]
        assert lines[3] == ""
        assert lines[4] == "echo hi"

    def test_multi_line_body_preserved(self):
        body_lines = ["import os", "", "def foo():", "    pass", ""]
        text = "\n".join(body_lines)
        result = ash._insert_spdx(text)
        # Every non-header line from the original must appear
        for line in body_lines:
            assert line in result

    def test_empty_file(self):
        result = ash._insert_spdx("")
        assert "SPDX-License-Identifier" in result

    def test_only_shebang(self):
        result = ash._insert_spdx("#!/usr/bin/env python3\n")
        assert result.startswith("#!/usr/bin/env python3")
        assert "SPDX-License-Identifier" in result


class TestInsertSpdxIdempotent:
    """_insert_spdx must be a no-op when an SPDX header is already present."""

    def test_already_has_spdx_unchanged(self):
        text = (
            "#!/usr/bin/env python3\n"
            "# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt\n"
            "# SPDX-License-Identifier: Apache-2.0\n"
            "\n"
            'print("hi")\n'
        )
        assert ash._insert_spdx(text) == text

    def test_idempotent_after_insertion(self):
        text = 'print("hello")\n'
        once = ash._insert_spdx(text)
        twice = ash._insert_spdx(once)
        assert once == twice

    def test_already_has_spdx_no_shebang(self):
        text = (
            "# SPDX-FileCopyrightText: Some other entity\n# SPDX-License-Identifier: MIT\n\nx = 1\n"
        )
        assert ash._insert_spdx(text) == text


class TestInsertSpdxEdgeCases:
    """Edge cases that could trip up the implementation."""

    def test_shebang_variants(self):
        for shebang in [
            "#!/usr/bin/env python3",
            "#!/usr/bin/env bash",
            "#!/bin/sh",
            "#!/usr/bin/python3",
        ]:
            text = shebang + "\necho ok\n"
            result = ash._insert_spdx(text)
            assert result.startswith(shebang + "\n")

    def test_comment_not_shebang(self):
        """A regular comment on line 1 is NOT a shebang."""
        text = "# just a comment\nx = 1\n"
        result = ash._insert_spdx(text)
        # No shebang detected, so SPDX goes first
        assert result.startswith("# SPDX-FileCopyrightText")

    def test_file_ending_without_newline(self):
        text = 'print("hi")'
        result = ash._insert_spdx(text)
        assert 'print("hi")' in result
        assert "SPDX-License-Identifier" in result


# ─── add_header_to_file: file-level operations ────────────────────────────────


class TestAddHeaderToFile:
    """Test the file-level wrapper with tmp_path."""

    def test_modifies_file_without_spdx(self, tmp_path):
        f = tmp_path / "script.py"
        f.write_text('print("hello")\n')
        assert ash.add_header_to_file(f) is True
        content = f.read_text()
        assert "SPDX-License-Identifier" in content
        assert 'print("hello")' in content

    def test_skips_file_with_spdx(self, tmp_path):
        f = tmp_path / "script.py"
        original = (
            "#!/usr/bin/env python3\n"
            "# SPDX-FileCopyrightText: Copyright (c) 2024\n"
            "# SPDX-License-Identifier: Apache-2.0\n"
            "\n"
            "x = 1\n"
        )
        f.write_text(original)
        assert ash.add_header_to_file(f) is False
        assert f.read_text() == original

    def test_shebang_file(self, tmp_path):
        f = tmp_path / "tool.sh"
        f.write_text("#!/usr/bin/env bash\necho hi\n")
        assert ash.add_header_to_file(f) is True
        content = f.read_text()
        assert content.startswith("#!/usr/bin/env bash\n")
        assert "SPDX-License-Identifier" in content

    def test_returns_false_for_binary(self, tmp_path):
        f = tmp_path / "binary.dat"
        f.write_bytes(b"\x00\x01\x02\xff\xfe")
        assert ash.add_header_to_file(f) is False

    def test_returns_false_for_nonexistent(self, tmp_path):
        assert ash.add_header_to_file(tmp_path / "nope.py") is False

    def test_does_not_double_insert_on_second_call(self, tmp_path):
        f = tmp_path / "s.py"
        f.write_text("x = 1\n")
        ash.add_header_to_file(f)
        assert ash.add_header_to_file(f) is False  # already has header


# ─── add_headers_to_files: batch operations ───────────────────────────────────


class TestAddHeadersToFiles:
    """Test the batch wrapper."""

    def test_multiple_files(self, tmp_path):
        files = []
        for i in range(5):
            f = tmp_path / f"f{i}.py"
            f.write_text(f"x = {i}\n")
            files.append(f)
        count = ash.add_headers_to_files(files)
        assert count == 5
        for f in files:
            assert "SPDX-License-Identifier" in f.read_text()

    def test_mixed_needing_and_not(self, tmp_path):
        needs = tmp_path / "needs.py"
        needs.write_text("x = 1\n")
        has = tmp_path / "has.py"
        has.write_text("# SPDX-License-Identifier: Apache-2.0\nx = 2\n")
        count = ash.add_headers_to_files([needs, has])
        assert count == 1

    def test_empty_list(self):
        assert ash.add_headers_to_files([]) == 0

    def test_skips_directories(self, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        f = tmp_path / "file.py"
        f.write_text("x = 1\n")
        count = ash.add_headers_to_files([d, f])
        assert count == 1


# ─── _glob_dirs: glob helper ───────────────────────────────────────────────────


class TestGlobDirs:
    def test_finds_matching_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("x")
        (tmp_path / "c.txt").write_text("x")
        result = ash._glob_dirs(tmp_path, ["*.py"])
        names = sorted(p.name for p in result)
        assert names == ["a.py", "b.py"]

    def test_multiple_patterns(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.sh").write_text("x")
        result = ash._glob_dirs(tmp_path, ["*.py", "*.sh"])
        names = sorted(p.name for p in result)
        assert names == ["a.py", "b.sh"]

    def test_no_matches(self, tmp_path):
        result = ash._glob_dirs(tmp_path, ["*.nonexistent"])
        assert result == []

    def test_recursive_glob(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.py").write_text("x")
        result = ash._glob_dirs(tmp_path, ["**/*.py"])
        assert any(p.name == "deep.py" for p in result)


# ─── main: integration smoke test ─────────────────────────────────────────────


class TestMain:
    """Verify main() processes a temp repo correctly."""

    def test_main_processes_files(self, tmp_path, monkeypatch, capsys):
        # Create a mini-repo structure
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "tool.py").write_text("x = 1\n")
        (tmp_path / "scripts" / "run.sh").write_text("#!/usr/bin/env bash\necho hi\n")
        (tmp_path / "bench").mkdir()
        (tmp_path / "bench" / "kernel.py").write_text("y = 2\n")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_foo.py").write_text(
            "# SPDX-License-Identifier: Apache-2.0\nz = 3\n"
        )

        monkeypatch.setattr(ash, "REPO", tmp_path)
        ash.main()

        captured = capsys.readouterr()
        assert (
            "Total: 3 files updated" in captured.out
        )  # tool.py, run.sh, kernel.py (test already has SPDX)

    def test_main_idempotent(self, tmp_path, monkeypatch):
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "tool.py").write_text("x = 1\n")

        monkeypatch.setattr(ash, "REPO", tmp_path)
        ash.main()
        ash.main()  # second run should be a no-op

        content = (tmp_path / "scripts" / "tool.py").read_text()
        assert content.count("SPDX-License-Identifier") == 1


# ─── SPDX constant correctness ─────────────────────────────────────────────────


class TestSpdxConstant:
    def test_spdx_has_copyright(self):
        assert "SPDX-FileCopyrightText" in ash.SPDX

    def test_spdx_has_license(self):
        assert "SPDX-License-Identifier" in ash.SPDX

    def test_spdx_ends_with_newline(self):
        assert ash.SPDX.endswith("\n")

    def test_spdx_two_lines(self):
        lines = ash.SPDX.strip().split("\n")
        assert len(lines) == 2
