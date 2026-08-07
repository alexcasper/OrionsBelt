#!/usr/bin/env python3
"""Fix stale GB/s→GiB/s spec bandwidth in generated per-device table files.

t3's systematic unit fix (commit 78ffcdb) corrected DEVICE_SPEC_BANDWIDTH in
bench/plots.py but the generated results/figures/*_table.md files were never
regenerated (Python 3.10+ required). This script applies the same correction
in-place: updates the spec bandwidth line and recalculates every % of Spec.

Python 3.6 compatible (Jetson Nano rescue script).
"""
import glob
import os

# Correct spec bandwidth in GiB/s (from bench/plots.py DEVICE_SPEC_BANDWIDTH)
SPEC_GIBS = {
    "pi5": 15.8,
    "pi": 15.8,
    "rk3588": 31.7,
    "jetson": 23.8,
}

# Old (wrong) spec values that appear in stale generated files
OLD_SPEC = {
    "pi5": 17.0,
    "pi": 17.0,
    "rk3588": 34.0,
    "jetson": 25.6,
}


def lookup_spec(device_name):
    """Look up spec bandwidth for a device name (prefix match, case-insensitive)."""
    name = device_name.lower().strip()
    for prefix, bw in SPEC_GIBS.items():
        if name.startswith(prefix) or prefix.startswith(name):
            return bw, OLD_SPEC[prefix]
    return None, None


def fix_table_file(filepath):
    """Fix spec bandwidth and percentages in a single table file."""
    filename = os.path.basename(filepath).replace("_table.md", "")
    new_spec, old_spec = lookup_spec(filename)
    if new_spec is None:
        return False, "unknown device"

    with open(filepath) as f:
        content = f.read()

    original = content
    changes = 0

    # Fix the spec bandwidth line
    old_line = f"**Device spec bandwidth:** {old_spec:.1f} GiB/s"
    new_line = f"**Device spec bandwidth:** {new_spec:.1f} GiB/s"
    if old_line in content:
        content = content.replace(old_line, new_line)
        changes += 1

    # Also catch the case where it's written without decimal (e.g., "34.0" vs "34")
    old_line_nd = f"**Device spec bandwidth:** {int(old_spec)} GiB/s"
    if old_line_nd in content:
        content = content.replace(old_line_nd, new_line)
        changes += 1

    # Fix percentages in the "Achieved vs Spec Bandwidth" table.
    # Row format: | Kernel | Achieved (GiB/s) | % of Spec | p50 | Spread |
    # Recalculate % of Spec = Achieved / new_spec * 100
    lines = content.split("\n")
    in_bw_table = False
    fixed_lines = []
    for line in lines:
        if "Achieved vs Spec Bandwidth" in line:
            in_bw_table = True
            fixed_lines.append(line)
            continue
        if in_bw_table:
            # Stay in table until we hit a new section header or end of meaningful content
            if line.startswith("## ") or line.startswith("# "):
                in_bw_table = False
                fixed_lines.append(line)
                continue
            if line.startswith("|") and "%" in line:
                # Skip header and separator rows
                if "Achieved" in line or "---" in line:
                    fixed_lines.append(line)
                    continue
                # Split by | to access columns individually
                parts = line.split("|")
                if len(parts) >= 5:
                    try:
                        achieved = float(parts[2].strip())
                        new_pct = achieved / new_spec * 100.0
                        parts[3] = f" {new_pct:.1f}% "
                        line = "|".join(parts)
                        changes += 1
                    except ValueError:
                        pass  # not a data row

        fixed_lines.append(line)

    content = "\n".join(fixed_lines)

    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        return True, f"{changes} changes"
    return False, "no changes needed (already correct or pattern not found)"


def main():
    table_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
    table_files = sorted(glob.glob(os.path.join(table_dir, "*_table.md")))

    fixed = 0
    skipped = 0
    for tf in table_files:
        ok, msg = fix_table_file(tf)
        status = "FIXED" if ok else "SKIP"
        if ok:
            fixed += 1
        else:
            skipped += 1
        print(f"  {status} {os.path.basename(tf)}: {msg}")

    print(f"\n{fixed} files fixed, {skipped} skipped.")


if __name__ == "__main__":
    main()
