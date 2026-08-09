#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

# OrionsBelt goose loop: one per node, in tmux 'orion-bench'. Self-heals bloat-stalls.
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
cd "$HOME/OrionsBelt" || { echo "no ~/OrionsBelt"; exit 1; }
HOST=$(hostname); BRANCH="bench/$HOST"
LOG="$HOME/OrionsBelt/.goose-loop.log"; TASK="$HOME/OrionsBelt/.goose-task.md"; GOOSE="$HOME/.local/bin/goose"
# Guard against duplicate instances: if another goose-loop.sh is already
# running, exit silently. Prevents two agents working the same branch
# concurrently (seen on t3 2026-08-09: two sessions spawned two goose
# agents; the stale one generated 146 runaway gitignored manifests).
#
# Uses a PID-file lock rather than `pgrep -f <pattern>` matching on the
# script's own command line: that approach self-matched the command-
# substitution subshell bash forks to evaluate the pgrep|grep|head
# pipeline itself (its cmdline also starts with "bash" and contains the
# literal pattern text "goose-loop.sh"), so the guard saw a "duplicate"
# on every single invocation, including the very first one from a
# completely clean state -- this is why the loop could never stay up
# (ob-94u). kill -0 confirms the recorded PID is still alive, so a
# lockfile left behind by a crashed instance doesn't wedge future runs.
LOCKFILE="$HOME/OrionsBelt/.goose-loop.pid"
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
  echo "[guard] another goose-loop.sh (PID $(cat "$LOCKFILE")) is already running — exiting" >>"$LOG"
  exit 0
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT
TEMPLATE="$HOME/OrionsBelt/docs/agent-task.template.md"
# Max session age before forced fresh start. A resumed session can drift
# arbitrarily far behind main if it never re-reads the task template's
# sync step (bd dolt pull). 2h balances context-refresh cost against drift
# risk (ob-462: a >24h resumed session re-attempted a closed task for hours).
MAX_SESSION_AGE_SEC="${ORION_MAX_SESSION_AGE:-7200}"
git checkout "$BRANCH" >/dev/null 2>&1
echo "=== orion goose loop START host=$HOST branch=$BRANCH $(date) ===" | tee -a "$LOG"

# .goose-task.md is deliberately untracked (it once held a sudo password, bead
# ob-3i5). That means merging or rebasing onto main DELETES it from the working
# tree, and `goose run -i` then fails every iteration while the loop keeps
# spinning and committing nothing — silent death. That is exactly how the j2 node
# stopped: it rebased onto main at 14:38 UTC and went quiet ten minutes later.
# Regenerate from the committed template instead of dying.
ensure_task() {
  [ -f "$TASK" ] && return 0
  if [ ! -f "$TEMPLATE" ]; then
    echo "[FATAL] no $TASK and no template at $TEMPLATE" | tee -a "$LOG"; return 1
  fi
  echo "[self-heal: $TASK missing (rebase onto main?), regenerating from template]" | tee -a "$LOG"
  sed -e "s|__BRANCH__|$BRANCH|g" -e "s|__HOST__|$HOST|g" \
      -e "s|__DEVICE__|${ORION_DEVICE:-$HOST}|g" \
      -e "s|__BINARY__|${ORION_BINARY:-$HOST}|g" \
      -e "s|__RESULT_NAME__|${ORION_RESULT_NAME:-$HOST}|g" \
      "$TEMPLATE" > "$TASK"
  # Device specifics come from the environment; set them in the node's shell
  # profile (ORION_DEVICE / ORION_BINARY / ORION_RESULT_NAME) so a regenerated
  # task is still node-correct rather than generically wrong.
  return 0
}

while true; do
  echo "--- $HOST iter $(date) ---" | tee -a "$LOG"
  # Rotate log if it exceeds 10 MB to prevent disk fill (ob-502).
  # Keep last 2000 lines — enough for the tail -40 bloat-stall check below
  # and recent debugging context. Atomic via tmp+mv.
  _LOG_SIZE=$(stat -c%s "$LOG" 2>/dev/null || echo 0)
  if [ "$_LOG_SIZE" -gt 10485760 ]; then
    tail -2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
    echo "[log-rotate: truncated to last 2000 lines, was $((_LOG_SIZE / 1048576))MB]" >> "$LOG"
  fi
  # self-heal: if the last run tripped the bloat-stall signature, force a fresh session
  if tail -40 "$LOG" 2>/dev/null | grep -q "create a new session"; then
    rm -f "$HOME/OrionsBelt/.goose-session-created"; echo "[self-heal: fresh session]" >>"$LOG"
  fi
  # session aging: force a fresh session after MAX_SESSION_AGE_SEC so the
  # agent periodically re-reads the task template and re-syncs beads/main.
  # Without this, --resume can silently run a stale session for days (ob-462).
  MARKER="$HOME/OrionsBelt/.goose-session-created"
  if [ -f "$MARKER" ]; then
    AGE=$(( $(date +%s) - $(date +%s -r "$MARKER") ))
    if [ "$AGE" -gt "$MAX_SESSION_AGE_SEC" ]; then
      rm -f "$MARKER"
      echo "[session-age: $AGE sec > ${MAX_SESSION_AGE_SEC}s limit, forcing fresh session]" >>"$LOG"
    fi
  fi
  if ! ensure_task; then sleep 60; continue; fi
  if [ -f "$HOME/OrionsBelt/.goose-session-created" ]; then
    "$GOOSE" run -i "$TASK" -n "ob-$HOST" --resume >>"$LOG" 2>&1
  else
    "$GOOSE" run -i "$TASK" -n "ob-$HOST" >>"$LOG" 2>&1
    [ $? -eq 0 ] && touch "$HOME/OrionsBelt/.goose-session-created"
  fi
  # safety flush + push (goose usually commits its own work)
  git add -A 2>/dev/null
  git commit -q -m "$HOST: loop flush $(date -u +%FT%TZ)" 2>/dev/null || true
  git push -q origin "$BRANCH" 2>/dev/null || true
  sleep 20
done
