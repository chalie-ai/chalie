#!/usr/bin/env bash
# 2-phase cleanup loop:
#   Phase 1: ruff → vulture → writes queue (no LLM needed)
#   Phase 2: local model fixes items one at a time
#
# Crash-safe: queue file persists. Restart picks up where it left off.

cd /Volumes/llm/Chalie
QUEUE="scripts/cleanup_queue.json"

queue_size() {
  python3 -c "
import json, os
try:
    q = json.load(open('$QUEUE')) if os.path.exists('$QUEUE') else []
    print(len(q))
except:
    print(0)
" 2>/dev/null
}

log() { echo "$(date '+%H:%M:%S') $1"; }

# Parse ruff + vulture output into cleanup_queue.json (up to 10 items, 1 per file)
build_queue() {
  python3 << 'PYEOF'
import subprocess, json, re, os

seen_files = set()
queue = []

# ── Pass 1: ruff (unused imports, unused vars, redefined names) ──
result = subprocess.run(
    ["ruff", "check", "--select=F401,F841,F811", "--output-format=json",
     "--exclude=data,__pycache__,.venv,venv,tests", "backend/"],
    capture_output=True, text=True
)
issues = json.loads(result.stdout) if result.stdout.strip() else []

for issue in issues:
    if len(queue) >= 10:
        break
    fpath = issue["filename"]
    if fpath.startswith(os.getcwd() + "/"):
        fpath = fpath[len(os.getcwd()) + 1:]
    if fpath in seen_files:
        continue
    seen_files.add(fpath)

    code = issue.get("code", "")
    m = re.search(r"`([^`]+)`", issue.get("message", ""))
    sym = m.group(1) if m else "unknown"

    if code == "F401":
        desc = f"remove unused import `{sym}`"
        action = (f"Open `{fpath}` and remove the unused import `{sym}` from line {issue['location']['row']}. "
                  f"If `{sym}` is part of a multi-import line (e.g., `from x import a, {sym}, b`), "
                  f"remove only `{sym}` from the import list. If it is the only import on the line, delete the entire line.")
        verify = f"ruff check --select=F401 {fpath} 2>&1 | grep -i '{sym}'"
    elif code == "F841":
        desc = f"remove unused variable `{sym}`"
        action = (f"Open `{fpath}` and remove or replace the assignment to `{sym}` on line {issue['location']['row']}. "
                  f"If the right side has side effects (e.g., a function call), keep the call but remove the assignment "
                  f"(replace `{sym} = expr` with just `expr`). If it is a pure value, delete the entire line.")
        verify = f"ruff check --select=F841 {fpath} 2>&1 | grep -i '{sym}'"
    elif code == "F811":
        desc = f"remove redefined unused name `{sym}`"
        action = (f"Open `{fpath}` and examine the redefinition of `{sym}` on line {issue['location']['row']}. "
                  f"Remove the earlier (shadowed) definition if it is unused, or remove the later redefinition if it is the dead one.")
        verify = f"ruff check --select=F811 {fpath} 2>&1 | grep -i '{sym}'"
    else:
        continue

    queue.append({
        "type": "unused_import" if code == "F401" else "unused_variable" if code == "F841" else "redefined_name",
        "file": fpath,
        "symbol": sym,
        "description": desc,
        "action": action,
        "verify": verify,
    })

# ── Pass 2: vulture (dead code — unused vars, imports ruff missed) ──
if len(queue) < 10:
    result = subprocess.run(
        ["vulture", "backend/", "--min-confidence", "80",
         "--exclude=data,__pycache__,.venv,venv,tests"],
        capture_output=True, text=True
    )
    for line in result.stdout.strip().split("\n"):
        if len(queue) >= 10 or not line.strip():
            break
        m = re.match(r"(.+?):(\d+): (.+?) '([^']+)' \((\d+)% confidence\)", line)
        if not m:
            continue

        fpath, lineno, issue_type, sym, confidence = m.group(1), int(m.group(2)), m.group(3), m.group(4), int(m.group(5))
        if fpath.startswith(os.getcwd() + "/"):
            fpath = fpath[len(os.getcwd()) + 1:]
        if fpath in seen_files:
            continue
        seen_files.add(fpath)

        if "unused import" in issue_type:
            desc = f"remove unused import `{sym}`"
            action = (f"Open `{fpath}` and remove the unused import `{sym}` from line {lineno}. "
                      f"If `{sym}` is part of a multi-import line, remove only `{sym}`. "
                      f"If it is the only import on the line, delete the entire line.")
            verify = f"vulture {fpath} --min-confidence 80 2>&1 | grep '{sym}'"
            item_type = "unused_import"
        elif "unused variable" in issue_type:
            desc = f"remove unused variable `{sym}`"
            action = (f"Open `{fpath}` and examine the assignment to `{sym}` on line {lineno}. "
                      f"If the right side has side effects (e.g., a function call), keep the call but remove the variable assignment "
                      f"(replace `{sym} = expr` with just `expr`). If it is a pure value with no side effects, delete the entire line.")
            verify = f"vulture {fpath} --min-confidence 80 2>&1 | grep '{sym}'"
            item_type = "unused_variable"
        elif "unused function" in issue_type:
            desc = f"remove dead function `{sym}`"
            action = (f"Open `{fpath}` and remove the function `{sym}` starting at line {lineno}. "
                      f"Remove the entire function definition including its decorator(s) if any. "
                      f"After removing, check if any imports at the top of the file are now unused and remove those too.")
            verify = f"grep -n 'def {sym}' {fpath}"
            item_type = "dead_function"
        elif "unused method" in issue_type:
            desc = f"remove dead method `{sym}`"
            action = (f"Open `{fpath}` and remove the method `{sym}` starting at line {lineno}. "
                      f"Remove the entire method definition including its decorator(s) if any. "
                      f"After removing, check if any imports at the top of the file are now unused and remove those too.")
            verify = f"grep -n 'def {sym}' {fpath}"
            item_type = "dead_method"
        elif "unused class" in issue_type:
            desc = f"remove dead class `{sym}`"
            action = (f"Open `{fpath}` and remove the class `{sym}` starting at line {lineno}. "
                      f"Remove the entire class definition. "
                      f"After removing, check if any imports at the top of the file are now unused and remove those too.")
            verify = f"grep -n 'class {sym}' {fpath}"
            item_type = "dead_class"
        else:
            continue

        queue.append({
            "type": item_type,
            "file": fpath,
            "symbol": sym,
            "description": desc,
            "action": action,
            "verify": verify,
        })

json.dump(queue, open("scripts/cleanup_queue.json", "w"), indent=2)
print(len(queue))
PYEOF
}

while true; do
  # ── Phase 1: Build queue ────────────────────────────────────
  if [ "$(queue_size)" = "0" ]; then
    log "[AUDIT] Running ruff + vulture..."
    FOUND=$(build_queue)

    if [ "$FOUND" = "0" ]; then
      # Both tools clean — uncomment below to enable Sonnet deep analysis
      # log "[AUDIT] Tools clean. Calling Sonnet for deep scan..."
      # cd /Volumes/llm/Chalie && claude -p "/cleanup-audit" \
      #   --allowedTools 'Read,Grep,Glob,Bash,Write' \
      #   --dangerously-skip-permissions --verbose \
      #   --output-format stream-json 2>&1

      log "[AUDIT] No opportunities found. Sleeping 10m then exiting."
      sleep 600
      exit 0
    fi
    log "[AUDIT] $(queue_size) items queued."
  fi

  # ── Baseline ────────────────────────────────────────────────
  log "[BASELINE] Capturing eval baseline..."
  if ! cd /Volumes/llm/Chalie && python3 scripts/simplify_eval.py baseline 2>&1; then
    log "[BASELINE] Failed (tests broken?). Sleeping 30m."
    sleep 1800
    continue
  fi

  # ── Phase 2: Fix (local model) ─────────────────────────────
  ITEMS=$(queue_size)
  log "[FIX] Processing $ITEMS items..."
  RETRIES=0

  for i in $(seq 1 "$ITEMS"); do
    REMAINING=$(queue_size)
    [ "$REMAINING" = "0" ] && break

    # Clean up any leftover changes from a previous crash
    cd /Volumes/llm/Chalie && git checkout -- . 2>/dev/null || true

    log "[FIX] Item $i/$ITEMS (remaining: $REMAINING)"
    BEFORE=$REMAINING

    cd /Volumes/llm/Chalie && ~/.claude/local -p "/cleanup-fix" \
      --allowedTools 'Read,Edit,Write,Bash,Grep,Glob' \
      --dangerously-skip-permissions --verbose \
      --output-format stream-json 2>&1

    AFTER=$(queue_size)

    # If the item wasn't popped (model crashed), force-remove after 3 retries
    if [ "$BEFORE" = "$AFTER" ] && [ "$AFTER" != "0" ]; then
      RETRIES=$((RETRIES + 1))
      if [ "$RETRIES" -ge 3 ]; then
        log "[FIX] Item stuck after 3 retries. Force-removing."
        python3 -c "import json; q=json.load(open('$QUEUE')); q.pop(0); json.dump(q, open('$QUEUE','w'), indent=2)"
        cd /Volumes/llm/Chalie && git checkout -- . 2>/dev/null || true
        RETRIES=0
      fi
    else
      RETRIES=0
    fi
  done

  # ── Push ────────────────────────────────────────────────────
  UNPUSHED=$(cd /Volumes/llm/Chalie && git log @{u}..HEAD --oneline 2>/dev/null | wc -l | tr -d ' ')
  if [ "$UNPUSHED" -gt "0" ]; then
    log "[PUSH] Pushing $UNPUSHED commits."
    cd /Volumes/llm/Chalie && git push 2>&1
  fi

  log "[DONE] Batch complete."
done
