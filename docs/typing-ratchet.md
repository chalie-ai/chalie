# Typing Ratchet

The backend type-checks clean under `mypy --strict` — every first-party
package, the top-level modules, **and the test suite**. Strict mode is enabled
globally in `pyproject.toml` and there is **no relax/override block**: new code
is held to the same bar from the first line. The gate test
`tests/test_static_typing_gate.py` runs `mypy --strict` over the whole
first-party source tree on every CI run and fails the build on any error.

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

Two absolutes, no exceptions:

- **Never `Any`.** Reach for the most primitive concrete type that fits:
  `None`, `bool`, `int`, `str`, `bytes`, `float`, then `object`,
  `dict[str, object]`, `list[object]`, `tuple[object, ...]`, `sqlite3.Row`,
  and covariant `Mapping` / `Sequence` for read-only parameters. Promote to a
  precise `TypedDict` / `Protocol` once the shape is fully understood.
- **Never `# type: ignore`.** Resolve the underlying type problem instead. A
  default-level correctness error (`arg-type`, `attr-defined`, `assignment`,
  `return-value`, …) is often a real latent bug — fix the code, don't silence
  the checker. The only acceptable suppression is removing an existing
  *unused* ignore that mypy itself flags.

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

## Type primitives

`services/typing_primitives.py` is the canonical home for recurring shape
aliases (`JSONDict`, `JSONList`) and shared `TypedDict`s / `Protocol`s. Prefer
importing a named alias over re-spelling `dict[str, object]` in many places,
and graduate a hot shape to a precise `TypedDict` once it is well understood.

---

## Gate test reference

The gate test is `tests/test_static_typing_gate.py`. Its `_PACKAGES` and
`_TOP_LEVEL_MODULES` lists are the source of truth for what is checked. When
you add a new top-level package or module to the backend, add it to those
lists so the strict gate covers it.
