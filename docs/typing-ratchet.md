# Typing Ratchet

The backend type-checks clean under `mypy --strict` — every first-party
package, the top-level modules, **and the test suite**. Strict mode is enabled
globally in `pyproject.toml` and there is **no relax/override block**: new code
is held to the same bar from the first line. The gate test module
`tests/test_static_typing_gate.py` runs `mypy --strict` over the whole
first-party source tree on every CI run and fails the build on any error.

`mypy --strict` alone is **not** a hard-typed wall: even with no overrides it
still permits explicit `Any`, *used* `# type:` suppression comments, and only
checks the packages it is explicitly handed. Those escape hatches are closed by
companion gate tests so loosely-typed code hard-breaks the build rather than
slipping through — see [Enforcement gates](#enforcement-gates) below.

This document records how the codebase reached full strict and — more
importantly — the supported patterns for keeping it there without weakening a
type.

---

## The standing rule

`mypy --strict` must report **zero** errors over the package list in
`tests/test_static_typing_gate.py`. To check locally:

```sh
cd backend
python3 -m mypy abilities api capabilities configs mcp_server services \
    tools utils workers migrations scripts tests \
    consumer.py run.py runtime_config.py migrate_transcript_rebuild.py
```

or run the gate test directly:

```sh
cd backend
python3 -m pytest tests/test_static_typing_gate.py -q
```

Two absolutes, no exceptions — **both machine-enforced** (see
[Enforcement gates](#enforcement-gates)), not merely convention:

- **Never `Any`.** Reach for the most primitive concrete type that fits:
  `None`, `bool`, `int`, `str`, `bytes`, `float`, then `object`,
  `dict[str, object]`, `list[object]`, `tuple[object, ...]`, `sqlite3.Row`,
  and covariant `Mapping` / `Sequence` for read-only parameters. Promote to a
  precise `TypedDict` / `Protocol` once the shape is fully understood.
- **Never a `type:` suppression comment.** Resolve the underlying type problem
  instead. A default-level correctness error (`arg-type`, `attr-defined`,
  `assignment`, `return-value`, …) is often a real latent bug — fix the code,
  don't silence the checker. (`mypy --strict` only flags *unused* suppressions;
  a used one is invisible to it, which is why a separate gate bans them
  outright.)

---

## Supported narrowing patterns

When mypy needs help proving a type, narrow it **without changing runtime
behaviour**. The migration was guarded by an AST-equivalence check, so every
accepted pattern keeps the executed code byte-for-byte identical to its
untyped form.

### Inline `cast` — never an intermediate variable

Cast at the point of use, inside the expression. Do **not** introduce a
temporary variable or an `assert x is not None`, both of which add runtime
statements:

```python
# good — the cast erases to the original subscript chain
value = cast("dict[str, object]", payload["meta"])["id"]

# avoid — adds a runtime variable and a runtime assertion
meta = payload["meta"]
assert meta is not None
value = meta["id"]
```

### Cast a `Protocol` onto an assignment target

To satisfy mypy when a test injects an attribute onto an instance that
doesn't declare it, declare a write-only `Protocol` under `TYPE_CHECKING` and
cast the *target*. The cast erases at runtime, so the statement is exactly the
plain attribute assignment it always was:

```python
if TYPE_CHECKING:
    from typing import Protocol

    class _Injectable(Protocol):
        _db_path: object
        connection: object

cast("_Injectable", db)._db_path = tmp_path   # runs as: db._db_path = tmp_path
```

### `setattr` for method monkeypatching in tests

`obj.method = fn` trips `method-assign` under strict. `setattr(obj, "method",
fn)` is the same operation by Python's data model (a literal attribute name)
and type-checks clean. This is the **only** sanctioned form where the
type-only edit changes the AST — and it is sound because the language
guarantees the two are equivalent:

```python
original = Providers._resolve
setattr(Providers, "_resolve", lambda self, *_a, **_kw: _FakeLLMService(send_fn))
try:
    ...
finally:
    setattr(Providers, "_resolve", original)
```

Do **not** use `object.__setattr__(...)` for this — it bypasses descriptors
and is not equivalent to a normal assignment.

---

## Shared shapes

When a structural shape recurs across packages, give it a named alias or — once
the keys are well understood — a precise `TypedDict` / `Protocol`, and import
that instead of re-spelling `dict[str, object]` everywhere. Keep such shared
definitions in one module so the precise type can be tightened in a single
place. Never reach for `Any` as the alias's value: a JSON-decoded blob whose
shape is genuinely unknown is `object` (narrow at the read site with an inline
`cast`), not `Any`.

---

## Enforcement gates

`tests/test_static_typing_gate.py` holds four tests; together they make the
wall hard rather than advisory. All run under `pytest -m unit`:

| Test | Guarantees |
| --- | --- |
| `test_first_party_source_is_strict_clean` | `mypy --strict` reports **zero** errors over every package + top-level module + the test suite. |
| `test_no_explicit_any_in_annotations` | The literal `Any` token appears in **no** parameter, return, or variable annotation. AST-based, so `Any` in a docstring, comment, or identifier is not a false positive. This is the single largest hole `--strict` leaves open. |
| `test_no_type_ignore_comments_in_first_party_source` | **Zero** `type:` suppression comments tree-wide — a *used* one is invisible to `--strict`. |
| `test_typing_gate_covers_every_first_party_package_and_module` | The on-disk top-level layout matches the `_PACKAGES` / `_TOP_LEVEL_MODULES` roster exactly, so a new source package can never appear *outside* the type-checked set unnoticed. Py-free artifact dirs are listed in `_PY_FREE_DIRS`. |

`pyproject.toml` also sets `strict_bytes = true` on top of `--strict` (it is
not implied), forbidding the implicit `str` / `bytes` / `bytearray`
promiscuity mypy otherwise tolerates.

> **Note on `warn_unreachable`.** It is deliberately **not** enabled. It is a
> dead-code linter, not a type-soundness check, and on this codebase it fires
> almost entirely on *intentional* guards — double-checked-locking re-checks
> (reachable across threads, which mypy can't model) and defensive `isinstance`
> branches against real runtime input. Enabling it would force removing or
> obfuscating those guards for no type-safety gain.

When you add a new top-level package or module to the backend, add it to
`_PACKAGES` / `_TOP_LEVEL_MODULES` (or `_PY_FREE_DIRS` if it carries no `.py`) —
the completeness gate fails until you do.
