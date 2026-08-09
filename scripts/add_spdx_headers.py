#!/usr/bin/env python3
"""Add SPDX license headers to Python and shell files that lack them.

Handles shebang lines correctly: the shebang stays on line 1, SPDX comments
follow, then a blank line, then the original content.
"""

from pathlib import Path

SPDX = (
    "# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
)

REPO = Path(__file__).resolve().parent.parent
SHEBANG_PY = "#!/usr/bin/env python3"
SHEBANG_SH = "#!/usr/bin/env bash"

count = 0

# --- Python files ---
for pattern in ("bench/*.py", "scripts/*.py", "tests/*.py", "src/orionsbelt/**/*.py"):
    for fpath in REPO.glob(pattern):
        if fpath.is_dir():
            continue
        text = fpath.read_text()
        if "SPDX-License-Identifier" in text:
            continue

        lines = text.split("\n")
        shebang = ""
        body_start = 0

        if lines and lines[0].startswith("#!"):
            shebang = lines[0] + "\n"
            body_start = 1

        body = "\n".join(lines[body_start:])
        # Ensure exactly one blank line between SPDX and body
        new = shebang + SPDX + "\n" + body
        fpath.write_text(new)
        print(f"  + {fpath.relative_to(REPO)}")
        count += 1

# --- Shell scripts ---
for fpath in (REPO / "scripts").glob("*.sh"):
    if "SPDX-License-Identifier" in fpath.read_text():
        continue
    text = fpath.read_text()
    lines = text.split("\n")
    shebang = ""
    body_start = 0
    if lines and lines[0].startswith("#!"):
        shebang = lines[0] + "\n"
        body_start = 1
    body = "\n".join(lines[body_start:])
    new = shebang + SPDX + "\n" + body
    fpath.write_text(new)
    print(f"  + {fpath.relative_to(REPO)}")
    count += 1

# Also handle top-level scripts
for name in ("goose-loop.sh",):
    fpath = REPO / name
    if not fpath.exists():
        continue
    if "SPDX-License-Identifier" in fpath.read_text():
        continue
    text = fpath.read_text()
    lines = text.split("\n")
    shebang = lines[0] + "\n" if lines and lines[0].startswith("#!") else ""
    body_start = 1 if shebang else 0
    body = "\n".join(lines[body_start:])
    new = shebang + SPDX + "\n" + body
    fpath.write_text(new)
    print(f"  + {fpath.relative_to(REPO)}")
    count += 1

# bench/*.sh
for fpath in (REPO / "bench").glob("*.sh"):
    if "SPDX-License-Identifier" in fpath.read_text():
        continue
    text = fpath.read_text()
    lines = text.split("\n")
    shebang = lines[0] + "\n" if lines and lines[0].startswith("#!") else ""
    body_start = 1 if shebang else 0
    body = "\n".join(lines[body_start:])
    new = shebang + SPDX + "\n" + body
    fpath.write_text(new)
    print(f"  + {fpath.relative_to(REPO)}")
    count += 1

# tests/*.sh
for fpath in (REPO / "tests").glob("*.sh"):
    if "SPDX-License-Identifier" in fpath.read_text():
        continue
    text = fpath.read_text()
    lines = text.split("\n")
    shebang = lines[0] + "\n" if lines and lines[0].startswith("#!") else ""
    body_start = 1 if shebang else 0
    body = "\n".join(lines[body_start:])
    new = shebang + SPDX + "\n" + body
    fpath.write_text(new)
    print(f"  + {fpath.relative_to(REPO)}")
    count += 1

print(f"\nTotal: {count} files updated")
