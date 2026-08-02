#!/usr/bin/env bash
# OrionsBelt goose loop: one per node, in tmux 'orion-bench'. Self-heals bloat-stalls.
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
cd "$HOME/OrionsBelt" || { echo "no ~/OrionsBelt"; exit 1; }
HOST=$(hostname); BRANCH="bench/$HOST"
LOG="$HOME/OrionsBelt/.goose-loop.log"; TASK="$HOME/OrionsBelt/.goose-task.md"; GOOSE="$HOME/.local/bin/goose"
git checkout "$BRANCH" >/dev/null 2>&1
echo "=== orion goose loop START host=$HOST branch=$BRANCH $(date) ===" | tee -a "$LOG"

while true; do
  echo "--- $HOST iter $(date) ---" | tee -a "$LOG"
  # self-heal: if the last run tripped the bloat-stall signature, force a fresh session
  if tail -40 "$LOG" 2>/dev/null | grep -q "create a new session"; then
    rm -f "$HOME/OrionsBelt/.goose-session-created"; echo "[self-heal: fresh session]" >>"$LOG"
  fi
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
