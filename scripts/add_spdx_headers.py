#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0
"""Add SPDX license headers to Python and shell files that lack them.

Handles shebang lines correctly: the shebang stays on line 1, SPDX comments
follow, then a blank line, then the original content.
"""

from __future__ import annotations

from pathlib import Path

SPDX = (
    "# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
)

REPO = Path(__file__).resolve().parent.parent


def _insert_spdx(text: str) -> str:
    """Return *text* with the SPDX header inserted after any shebang line.

    If the text already contains an SPDX header it is returned unchanged.
    """
    if "SPDX-License-Identifier" in text:
        return text

    lines = text.split("\n")
    shebang = ""
    body_start = 0

    if lines and lines[0].startswith("#!"):
        shebang = lines[0] + "\n"
        body_start = 1

    body = "\n".join(lines[body_start:])
    return shebang + SPDX + "\n" + body


def add_header_to_file(fpath: Path, *, repo: Path | None = None) -> bool:
    """Add an SPDX header to *fpath* in-place.

    Returns ``True`` if the file was modified, ``False`` if it already had
    a header or could not be read as text.
    """
    try:
        text = fpath.read_text()
    except (UnicodeDecodeError, OSError):
        return False

    new = _insert_spdx(text)
    if new == text:
        return False

    fpath.write_text(new)
    if repo is not None:
        print(f"  + {fpath.relative_to(repo)}")
    return True


def add_headers_to_files(paths: list[Path], *, repo: Path | None = None) -> int:
    """Add SPDX headers to every file in *paths*, returning the count modified."""
    count = 0
    for fpath in paths:
        if fpath.is_dir():
            continue
        if add_header_to_file(fpath, repo=repo):
            count += 1
    return count


def _glob_dirs(repo: Path, patterns: list[str]) -> list[Path]:
    """Glob a list of patterns under *repo* and return the matching files."""
    result: list[Path] = []
    for pat in patterns:
        result.extend(repo.glob(pat))
    return result


def main() -> None:
    repo = REPO

    # --- Python files ---
    py_paths = _glob_dirs(
        repo, ["bench/*.py", "scripts/*.py", "tests/*.py", "src/orionsbelt/**/*.py"]
    )
    count = add_headers_to_files(py_paths, repo=repo)

    # --- Shell scripts (scripts/, bench/, tests/, top-level) ---
    sh_dirs = [repo / "scripts", repo / "bench", repo / "tests"]
    sh_paths: list[Path] = []
    for d in sh_dirs:
        sh_paths.extend(d.glob("*.sh"))
    # Top-level scripts
    for name in ("goose-loop.sh",):
        p = repo / name
        if p.exists():
            sh_paths.append(p)

    count += add_headers_to_files(sh_paths, repo=repo)

    print(f"\nTotal: {count} files updated")


if __name__ == "__main__":  # pragma: no cover
    main()
