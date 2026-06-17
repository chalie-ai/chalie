# Typing Ratchet

The backend is migrated to `mypy --strict` incrementally using a ratchet
pattern. Strict mode is enabled globally in `pyproject.toml`, but unmigrated
packages are kept green via a single relax override block. The gate test
`tests/test_static_typing_gate.py` runs `mypy --strict` over the whole
first-party source tree on every CI run — it passes because the relax
override suppresses all errors in unmigrated packages, and tightens
automatically as each package is migrated.

---

## How the relax override works

`pyproject.toml` contains a `[[tool.mypy.overrides]]` block that lists every
unmigrated first-party package:

```toml
[[tool.mypy.overrides]]
module = [
    "abilities.*",
    "api.*",
    # ... all unmigrated packages ...
]
disallow_untyped_defs = false
disallow_incomplete_defs = false
disallow_untyped_calls = false
disallow_untyped_decorators = false
disallow_any_generics = false
disallow_subclassing_any = false
check_untyped_defs = false
warn_return_any = false
warn_unused_ignores = false
strict_equality = false
extra_checks = false
implicit_reexport = true
disable_error_code = [
    "arg-type",
    "assignment",
    "attr-defined",
    # ... other default-level codes that fire at migration baseline ...
]
```

The `disallow_*` / `check_*` / `warn_*` flags silence annotation-absence
errors (missing return types, missing generics, etc.). The `disable_error_code`
list suppresses the default-level correctness codes that still fire despite
those flags being off.

**Important:** `disable_error_code` entries are scoped to unmigrated packages
only — they are NOT global disables. Once a package is removed from the
`module` list, every code in `disable_error_code` becomes live for that
package again.

---

## Migrating a package

1. **Remove the package's glob** from the `module = [...]` list in the
   `[[tool.mypy.overrides]]` block in `pyproject.toml`.

2. **Run the gate:**
   ```sh
   cd backend
   python3 -m pytest tests/test_static_typing_gate.py -q
   ```
   Or run mypy directly for faster iteration:
   ```sh
   cd backend
   python3 -m mypy <package-name>
   ```

3. **Fix every surfaced error.** Three categories will appear:

   - **Strict annotation errors** (`no-untyped-def`, `disallow-any-generics`,
     etc.) — add missing type annotations, type arguments, and return types.

   - **Default-level correctness errors** (`arg-type`, `attr-defined`,
     `assignment`, `return-value`, etc.) — these may be **real latent bugs**.
     Fix the underlying code rather than adding `# type: ignore` suppression.
     Reserve `# type: ignore[<code>]` only for verified false positives (e.g.
     a third-party stub is wrong), always with a comment explaining why.

   - **Cross-package cascade errors** (`no-untyped-call` in a newly-migrated
     package that calls into a not-yet-migrated package) — these cannot be
     fixed until the dependency package is migrated. Track them in the
     `_MAX_RESIDUAL_ERRORS` ratchet ceiling in the gate test and document them
     in its comment block. The ceiling must only decrease; these errors resolve
     automatically when the dependency package's ticket lands.

4. **Run the full unit suite** to confirm no regressions:
   ```sh
   cd backend
   python3 -m pytest -m unit -q
   ```

5. **Commit** the package's type annotations and the updated `pyproject.toml`
   together in a single commit.

---

## Final flip

When the last package glob is removed from the `module` list, delete the
entire `[[tool.mypy.overrides]]` block (including `disable_error_code`), set
`_MAX_RESIDUAL_ERRORS = 0` in the gate test, and change the assertion back to
`assert proc.returncode == 0`. Verify `tests/test_static_typing_gate.py` still
passes — at that point the whole codebase runs under strict mode with no
exceptions.

---

## Type primitives

`services/typing_primitives.py` is the canonical home for recurring shape
aliases (`JSONDict`, `JSONList`) and future TypedDicts or Protocols. When
migrating a package that uses `dict[str, Any]` shapes, prefer importing from
`typing_primitives` and then replacing with a precise TypedDict once the
shape is fully understood.

---

## Gate test reference

The gate test is `tests/test_static_typing_gate.py`. It checks the same
package list that appears in the `[[tool.mypy.overrides]]` module list.
If you add a new top-level module to the codebase, add it to both the
overrides `module` list (to keep CI green immediately) and the `_PACKAGES` /
`_TOP_LEVEL_MODULES` lists in the gate test (so it is covered by the check).
