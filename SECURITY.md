# Security Policy

## Known Exposure: Device Sudo Password in Git History

**Severity:** Medium (repo is PRIVATE, but credential is committed)
**Status:** Tracked as bead `ob-3i5`
**Discovered:** 2026-08-03

### Summary

The sudo password for Arm test devices (value deliberately not repeated
here or in bead `ob-3i5` — see the remediation runbook below for how to
obtain it out-of-band if needed; used in a `sudo -S` pipeline to set CPU
governors) was committed to git in `.goose-task.md` and `.goose-loop.log`.
Both files are now gitignored and untracked (since commit `e2d1c7e`), but the
blobs remain reachable in history across all branches.

### Affected Branches

All branches: `main`, `bench/t4`, `bench/t3`, `bench/r5`, `bench/j2`

### Affected Commits (10 total)

| Commit | Branch | File |
|--------|--------|------|
| `87b1f05` | bench/t4 | .goose-task.md |
| `6a33d67` | bench/j2 | .goose-loop.log |
| `b96e12a` | bench/j2 | .goose-loop.log |
| `050b780` | bench/j2 | .goose-loop.log |
| `7705d79` | bench/j2 | .goose-loop.log |
| `b47a051` | bench/j2 | .goose-loop.log |
| `d4d5de0` | bench/j2 | .goose-loop.log |
| `f9ff0c9` | bench/r5 | .goose-task.md |
| `81b54b6` | bench/t4 | .goose-loop.log |
| `e2d1c7e` | integration | .goose-task.md (untrack commit; contains string in diff) |

### Remediation (requires human decision)

Two actions are needed:

1. **Rotate the device sudo password.** The credential must be treated as
   disclosed. Change it on every Arm device, then update the task assignment
   files that use it.

2. **Purge git history (optional but recommended).** Run the remediation
   script `scripts/purge_sudo_password.sh`, which rewrites history to replace
   the password string across all branches. This requires force-pushing all
   branches — coordinate with the team first, as it invalidates open PRs.

### Stale Branch: bench/r5

Branch `bench/r5` (413 commits diverged from `main`, work fully superseded)
exposes the rotated password at its **tip** in `.goose-task.md` (lines 7-8).
Because the password was already rotated (see §Remediation #1 above), this is
**acceptable historical exposure**:

- The credential is no longer valid on any device.
- The same string also exists in `main`'s history (commit `f9ff0c9`) — purging
  `bench/r5` alone would not remove the exposure.
- Truly purging requires `git-filter-repo` + force-push of all branches
  (owner-level action); this is tracked but not urgent given rotation.
- **Recommendation:** delete `bench/r5` (it is superseded work) once the
  team confirms no unique changes are needed. This requires a push the
  agent constraint prohibits, so it remains an owner action.

Tracked as bead `ob-0vu`.

### What Has Been Done

- `.goose-task.md` and `.goose-loop.log` are untracked and gitignored
  (commit `e2d1c7e`)
- This document (`SECURITY.md`) records the exposure
- Remediation script `scripts/purge_sudo_password.sh` is ready to execute
- The repo is private, limiting exposure to collaborators only
- **bench/r5 documented as acceptable historical exposure** of rotated
  credential (bead `ob-0vu`)
