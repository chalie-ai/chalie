# Cleanup Fix

Process exactly ONE item from `scripts/cleanup_queue.json`, then stop.

## Steps

### 1. Read the queue

Read the file `scripts/cleanup_queue.json`. Take the FIRST item (index 0).

If the array is empty, stop immediately — there is nothing to do.

### 2. Verify the issue still exists

Run the item's `verify` command to confirm the issue is still present. Also read the file and confirm the symbol is actually there.

If the issue is gone (already fixed by someone else), skip straight to Step 6.

### 3. Verify safety

Before removing any symbol, grep to confirm nothing else uses it:

```bash
grep -rn 'SYMBOL' backend/ --include='*.py' | grep -v __pycache__ | grep -v 'def SYMBOL' | grep -v '# '
```

Replace SYMBOL with the actual symbol name from the queue item.

If there are unexpected references beyond the definition, skip to Step 6 without making changes.

### 4. Make the change

Follow the item's `action` field exactly. Rules:
- Make the MINIMUM change described
- Do NOT add comments, docstrings, or type hints
- Do NOT refactor or reformat surrounding code
- Do NOT touch test files or schema.sql
- For backend/api/ files: only lint fixes (unused imports/variables). Do NOT change function signatures, endpoints, or logic.

### 5. Run the eval

```bash
python scripts/simplify_eval.py check
```

**IMPORTANT: This command takes ~5 minutes. You MUST use timeout: 600000.**

**If exit code is 0 (PASS):**

Commit the change with the item's description:
```bash
git add -A && git commit -m "$(cat <<'EOF'
simplify: <paste the item description here>

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

Then refresh the baseline for the next item:
```bash
python scripts/simplify_eval.py refresh
```
**Use timeout: 600000.**

**If exit code is NOT 0 (FAIL):**

Revert all changes:
```bash
git checkout -- .
```

### 6. Remove the item from the queue

Regardless of whether the fix passed, failed, or was skipped — remove the first item:

```bash
python3 -c "import json; q=json.load(open('scripts/cleanup_queue.json')); q.pop(0); json.dump(q, open('scripts/cleanup_queue.json','w'), indent=2)"
```

### 7. STOP

Do NOT process any more items. You are done. Exit immediately.
