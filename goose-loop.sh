#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 OrionsBelt / Agentic AI Foundation
# SPDX-License-Identifier: Apache-2.0

# OrionsBelt goose loop: one per node, in tmux 'orion-bench'. Self-heals bloat-stalls.
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
cd "$HOME/OrionsBelt" || { echo "no ~/OrionsBelt"; exit 1; }
HOST=$(hostname); BRANCH="bench/$HOST"
LOG="$HOME/OrionsBelt/.goose-loop.log"; TASK="$HOME/OrionsBelt/.goose-task.md"; GOOSE="$HOME/.local/bin/goose"
TEMPLATE="$HOME/OrionsBelt/docs/agent-task.template.md"
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
  # self-heal: if the last run tripped the bloat-stall signature, force a fresh session
  if tail -40 "$LOG" 2>/dev/null | grep -q "create a new session"; then
    rm -f "$HOME/OrionsBelt/.goose-session-created"; echo "[self-heal: fresh session]" >>"$LOG"
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
