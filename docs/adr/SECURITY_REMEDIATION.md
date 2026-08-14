# Security Audit & Remediation Runbook — Committed Device Credential

**Bead:** `ob-3i5`
**Date:** 2026-08-07
**Severity:** ~~LOW (repo is private — see §1)~~

> **ADDENDUM 2026-08-14T23:09Z (t4): severity is now CRITICAL.** The precondition this
> document relied on has inverted: `gh repo view` reports `visibility: PUBLIC`
> (verified 23:07Z; PRIVATE as late as the 17:12Z audit). §1 stated the §4 steps
> "become **blocking** before that flip" — the repo was made public without them.
> The credential is **still live**: verified on t4 (sudo governor write succeeds).
> Tips of all remote branches are clean (cross-branch credential scan passes);
> exposure is git-history blobs only, now reachable by anyone who can clone.
> Required human actions, in order: **§4a coordinated rotation on all five
> devices** (do not rotate unilaterally — it breaks fleet agents' sudo mid-session;
> distribute the new value out-of-band), then **§4c history purge** via
> `scripts/purge_password_history.sh` (force-push across `main` + branches —
> coordinator-only, after open PRs settle). Full detail: `ob-3i5` comment
> 2026-08-14T23:09Z.

---

## 1. Key finding: repo is PRIVATE

```
$ gh repo view alexcasper/OrionsBelt --json visibility,isPrivate
{"isPrivate":true,"visibility":"PRIVATE"}
```

The credential is exposed **only to collaborators** who already have repository
access. This dramatically lowers urgency: the password is not reachable by the
public internet. If the repo transitions to public (e.g. for Devpost
submission), the steps in §4 become **blocking** before that flip.

## 2. What was committed

The device sudo password (used in an `echo <password> | sudo -S tee` pattern to
set the CPU governor — value deliberately not repeated here; see ob-3i5 and
each device's operator for the value) appears in two files:

| File | Lines | Status |
|---|---|---|
| `.goose-task.md` | L7–8 | Gitignored on `main`/`bench/t4`; **still tracked at tip** of `bench/j1`, `origin/bench/r5`, `origin/fix/integrate-and-repair-main` |
| `.goose-loop.log` | Many (loop capture) | Gitignored on `main`/`bench/t4`; **still tracked** on the branches above |

`goose-loop.sh` is legitimate tooling and was deliberately kept (per ob-3i5
description) — it does not contain the password.

## 3. Exact commit-level exposure

`git log --all -S '<password>'` (pickaxe search for the known password string):

| Commit | File | Agent |
|---|---|---|
| `87b1f05` | `.goose-task.md` + `.goose-loop.log` | t4 |
| `6a33d67` | `.goose-task.md` + `.goose-loop.log` | j1 |
| `f9ff0c9` | `.goose-task.md` + `.goose-loop.log` | (rebase) |
| `050b780`–`8aff0cd` (×11) | `.goose-loop.log` only | j1/r5 |

All are reachable from current branch tips.

## 4. Remediation steps (require human/coordination)

### 4a. Rotate the device password (blocking for public repo)

The committed password must be treated as disclosed. Rotate it on each physical
device:

```bash
# On each device (t4, t3, r5, j1, j2):
passwd   # set a new password

# Update all task-instruction and operational references:
#   .goose-task.md (local, gitignored — update the inline password)
#   Agent loop prompts referencing the old password
```

**Do NOT do this unilaterally** — all agents and operators must be coordinated,
or access will break.

### 4b. Untrack the files from branches that still have them at tip

```bash
# On bench/j1, bench/r5, fix/integrate-and-repair-main:
git rm --cached .goose-task.md .goose-loop.log
git commit -m "security: untrack credential-bearing files (ob-3i5)"
```

### 4c. Purge git history (only if going public)

Rewriting history across `main` + 5+ branches will invalidate open PRs and
force all collaborators to re-clone. Only do this **immediately before making
the repo public** if the password has not yet been rotated:

```bash
# Install git-filter-repo
pip install git-filter-repo

# Back up first!
git clone --mirror <remote> orionsbelt-backup.git

# Purge the files from ALL history
git filter-repo --path .goose-task.md --path .goose-loop.log --invert-paths

# Force-push all refs (WARNING: invalidates existing PRs/clones)
git push origin --force --all
git push origin --force --tags
```

**If the password has already been rotated (§4a), history purge is optional** —
the exposed credential is useless. The `.gitignore` prevents new exposures.

## 5. Current state (updated 2026-08-07T22:27Z)

| Branch | File at tip? | Gitignored? | Last checked |
|---|---|---|---|
| `main` | No | Yes | 2026-08-07T22:27Z |
| `bench/t4` | No | Yes | 2026-08-07T22:27Z |
| `bench/j1` | **No** ✓ | Yes | 2026-08-07T22:27Z (j1 cleaned up) |
| `origin/bench/r5` | **Yes** | No | 2026-08-07T22:27Z |
| `origin/fix/integrate-and-repair-main` | **Yes** | No | 2026-08-07T22:27Z |

Progress since initial audit: **bench/j1** now clean (files untracked). Two
branches remain with active exposure: `bench/r5` and `fix/integrate-and-repair-main`.

## 6. Recommendation

1. **Immediate (low effort):** j1 and r5 agents should `git rm --cached` the
   files from their branches (§4b). This stops active exposure at branch tips.
2. **Before going public:** rotate the password (§4a), then optionally purge
   history (§4c).
3. **Ongoing:** `.goose-task.md` and `.goose-loop.log` remain gitignored — no
   new exposures will occur on `main` or `bench/t4`.
