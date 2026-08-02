# Using beads (`bd`) on OrionsBelt

This repo tracks all work in [beads](https://beads.gascity.com/) — a dependency-aware issue
tracker built for AI coding agents. Beads replaces markdown TODO lists with a real dependency
graph, so `bd ready` can tell you exactly what is unblocked right now.

The full plan these issues were generated from is [`PLAN.md`](../PLAN.md).

---

## 1. Install

```bash
npm install -g @beads/bd     # what this repo was set up with (v1.1.2)
# or
brew install beads
# or
curl -fsSL https://raw.githubusercontent.com/gastownhall/beads/main/scripts/install.sh | bash
```

Verify:

```bash
bd --version
```

## 2. First-time setup in a fresh clone

The `.beads/` Dolt database is **not** committed to git (see `.beads/.gitignore`) — only
`.beads/issues.jsonl`, `config.yaml`, and `metadata.json` are. So a fresh clone needs to
hydrate the database from the committed export:

```bash
cd OrionsBelt
bd bootstrap                      # non-destructive setup for fresh clones
bd import .beads/issues.jsonl     # hydrate issues from the committed export
bd ready                          # confirm: should list unblocked work
```

If `bd bootstrap` reports an existing database, you are already set up — skip to `bd ready`.

> **Note on `bd import`:** it is for *bootstrap and recovery only*. During normal work, the
> Dolt database is the source of truth and `issues.jsonl` is a passive export. Do not
> round-trip through JSONL as a workflow.

## 3. Repo conventions

### Issue prefix

All issues are `ob-<hash>` (e.g. `ob-a3f2dd`). Hash-based IDs, not sequential — they don't
collide when several agents create issues concurrently.

### Types

| Type | Used here for |
|---|---|
| `epic` | The ten workstreams from `PLAN.md` §4 (E0–E9) |
| `task` | Concrete units of work |
| `decision` | Forks that must be recorded as an ADR in `docs/adr/` |
| `chore` | Hygiene, compliance, tooling |
| `bug` | Defects found during implementation |

### Priorities

Priority is `0`–`4`, **0 is highest**. Not "high"/"medium"/"low".

| P | Meaning on this project |
|---|---|
| `0` | On the critical path to the Aug 14 deadline, or an external gate that must start now |
| `1` | Required for the minimum viable submission |
| `2` | Target submission — should have |
| `3` | Stretch / nice to have |
| `4` | Backlog, post-deadline follow-up |

### Labels

| Label | Meaning |
|---|---|
| `external-gate` | Blocked on a third party (procurement, CIX enrollment). **Cannot be worked by effort alone** — `bd ready` listing it does not mean it's actionable by us |
| `hardware` | Requires physical Orion O6 access |
| `portable` | Runs anywhere — safe to do before hardware arrives |
| `hedge` | Belongs to the Edge AI fallback track (generic aarch64) |
| `submission` | Directly required by Devpost rules |
| `stretch` | Cut first under the descope ladder (`PLAN.md` §7) |
| `research` | GDN-2 investigation |

### Dependency direction

This is the one thing people get backwards:

```bash
bd dep add <issue> <depends-on>     # <issue> is BLOCKED BY <depends-on>
```

In graph-plan JSON, the equivalent is:

```json
{"from_key": "the-dependent", "to_key": "the-prerequisite", "type": "blocks"}
```

`from_key` is the thing that has to wait. Verified empirically against `bd ready`.

### Hierarchy vs blocking

Epics own their tasks through **parent-child** (`--parent` / `parent_key`), never through
blocking edges. Blocking edges connect tasks to tasks. Mixing the two creates parent-child
blocking paths that `bd` will reject.

---

## 4. Daily workflow

```bash
bd ready                      # what can I work on right now?
bd show ob-a3f2dd             # full detail incl. blockers and blocked-by
bd update ob-a3f2dd --claim   # atomically claim it (assignee + in_progress)
# ... do the work ...
bd close ob-a3f2dd --reason "Landed in <commit>"
bd close ob-a3f2dd --suggest-next   # ...and show what that just unblocked
```

Creating new work as you discover it:

```bash
bd create "Vulkan scan kernel overflows at 128K" \
  --type bug --priority 1 \
  --description "Why this exists and what needs to happen" \
  --parent ob-<epic-id> \
  --deps ob-<blocker-id>
```

Useful views:

```bash
bd blocked                    # everything waiting on something
bd list --status=in_progress  # active work
bd dep tree ob-<id>           # visualize a dependency subtree
bd graph                      # whole dependency graph
bd stats                      # open/closed/blocked counts
bd search "quantization"      # text search
bd query 'label = "external-gate" AND status = "open"'
```

Health checks before a PR:

```bash
bd preflight                  # lint + stale + orphan checks
bd dep cycles                 # detect circular dependencies
bd doctor                     # sync problems, missing hooks
```

### Commands to avoid

- **`bd edit`** — opens `$EDITOR` (vim/nano) and will hang a non-interactive agent. Use
  `bd update <id> --description=... --notes=...` instead.
- **`bd create-form`** — likewise interactive.

---

## 5. Keeping the repo export in sync

`.beads/issues.jsonl` is the committed, human-reviewable snapshot of the issue graph. It is
what a fresh clone (and a reviewer reading the diff) sees. Refresh it whenever you change
issues:

```bash
bd export -o .beads/issues.jsonl
git add .beads/issues.jsonl
git commit -m "beads: <what changed>"
```

Cross-machine sync of the live Dolt database is a separate mechanism (`bd dolt push` /
`bd dolt pull`, riding `refs/dolt/data` on the git remote — not `refs/heads/*` where the code
lives). For a single-maintainer project the JSONL export in git is sufficient; reach for Dolt
sync only when two machines or agents need the live database.

See [SYNC_CONCEPTS.md](https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md)
for the anti-patterns — chiefly: don't treat JSONL as the source of truth, and don't
`bd import` during normal operation.

---

## 6. Agent instructions

`AGENTS.md` and `CLAUDE.md` (generated by `bd init`) point agents at beads. An agent starting
a session should run:

```bash
bd prime      # AI-optimized workflow context, ~80 lines
```

Claude Code and Codex hooks installed by `bd init` call this automatically when a beads
workspace is detected.

**Project-specific rules for agents:**

1. Read [`PLAN.md`](../PLAN.md) before claiming your first bead — the dependency graph only
   makes sense with the plan's context, especially the two-track (Physical AI / Edge AI)
   structure and the deadline pressure.
2. `bd ready` showing an `external-gate` bead does **not** mean it is actionable. Those wait
   on third parties. Prefer `portable`-labelled work when hardware is unavailable.
3. Do not close a benchmark bead without a committed run manifest. A number without
   provenance is not a result (`PLAN.md` §9).
4. Every `decision`-type bead must produce an ADR in `docs/adr/` before it closes; link it
   from the bead's notes.
5. Record durable insights with `bd remember "..."` rather than creating memory files.
   Retrieve with `bd memories <keyword>`.
