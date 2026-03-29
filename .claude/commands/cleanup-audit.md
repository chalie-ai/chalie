# Cleanup Audit

You are auditing the Chalie codebase for cleanup opportunities. Find up to 10 specific, atomic items and write them to `scripts/cleanup_queue.json`. Do NOT make any code changes yourself.

## Process

### Step 1: Check what was recently cleaned
```bash
git log --oneline --since="7 days" --grep="simplify" | head -30
```
Note which files/symbols were already handled. Do not duplicate these.

### Step 2: Find lint issues (fast wins)
```bash
ruff check --select=F401,F841,F811 -q --exclude=data,__pycache__,.venv,venv,tests backend/ 2>&1 | head -40
```
- F401 = unused imports
- F841 = unused variables
- F811 = redefined unused names

Each issue is a potential queue item. Verify each one is real by reading the file — sometimes an import is re-exported or used dynamically.

### Step 3: If fewer than 10 lint issues, find dead code

Find service files NOT touched recently:
```bash
git log --oneline --since="14 days" --name-only -- backend/services/ | grep '\.py$' | sort -u
```
Pick files NOT in this list. For each chosen file:
1. Read the file and identify functions/methods that look unused
2. Grep the **entire** codebase to confirm no callers:
```bash
grep -rn 'FUNCTION_NAME' backend/ --include='*.py' | grep -v __pycache__ | grep -v 'def FUNCTION_NAME'
```
3. Also check frontend and tests for references:
```bash
grep -rn 'FUNCTION_NAME' frontend/ --include='*.js' | head -5
```
4. Only add to the queue if there are zero references outside the definition

### Step 4: Write the queue

Write `scripts/cleanup_queue.json` with this exact JSON format:

```json
[
  {
    "type": "unused_import",
    "file": "backend/services/example_service.py",
    "symbol": "json",
    "description": "remove unused import `json`",
    "action": "Delete the line `import json` (around line 5). No other changes needed.",
    "verify": "grep -n 'import json' backend/services/example_service.py"
  }
]
```

### Field rules:
- **type**: `unused_import` | `unused_variable` | `dead_function` | `dead_method` | `dead_class` | `simplification`
- **file**: relative path from project root
- **symbol**: the exact name being removed/changed
- **description**: one-line, lowercase, starts with a verb (e.g., "remove unused import `os`")
- **action**: step-by-step instructions a junior developer could follow. Include approximate line numbers, exact code to remove, and any cascading changes (e.g., "if removing this function also makes the `typing.List` import unused, remove that import too")
- **verify**: a shell command that proves the issue exists (returns output when the issue is present, empty when already fixed)

## Rules

1. Maximum 10 items per audit
2. Each item MUST be in a DIFFERENT file — no two items touching the same file
3. Do NOT include items that touch test files or `schema.sql`. For `backend/api/` files, only lint fixes (unused imports/variables) are allowed — do NOT change function signatures, endpoints, or logic.
4. Do NOT include items already cleaned in the last 7 days
5. Items must be fully independent — fixing one must not break another
6. Be conservative — only include items you are confident are safe to remove
7. Write ONLY the queue file. Do NOT modify any source code
8. If you find zero opportunities, write an empty array: `[]`
