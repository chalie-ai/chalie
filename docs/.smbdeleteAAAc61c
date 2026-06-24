# Testing

## Running the Suite

```bash
cd backend

pytest -m unit          # pure functions only — fast
pytest -m integration   # services, DB, real models — the default tier
pytest -m e2e           # live network / real Chromium
pytest                  # everything
```

`pytest.ini` enforces `-x` (stop on first failure) and `--strict-markers`. Tests live in `backend/tests/` (most files at the root, marked `integration` by convention; `tests/unit/`, `tests/integration/`, and `tests/e2e/` hold the explicitly tiered ones).

The shared `conftest.py` builds one real schema-converged SQLite template per session and hands each test a fresh copy — tests run against the real database layer, not fakes.

## Philosophy

Every test answers one question: **does this service do what it claims?** Given real input X, through the real production stack, assert real observable output Y.

- Prefer real collaborators over mocks. In-memory or copied SQLite is fine — it's still SQLite. (The conftest patches only the auth/session boundary so API tests can run logged-in.)
- Keep it to a handful of feature tests per service — test what the service is responsible for, not how it's wired internally.
- No plumbing assertions: field lists, call counts, `isinstance` as the sole check, version pins.
- Name tests after the real-world scenario: `test_architecture_question_scores_high`, not `test_predict_returns_dict`.
- If a service can only be tested by mocking its collaborators, that's a design problem — add a real seam or cover it at a higher level instead.

## Example

```python
import pytest
pytestmark = pytest.mark.integration

class TestDeliberationScoreClassifier:
    def test_architecture_question_scores_high(self, onnx_svc):
        score = onnx_svc.predict_scalar(
            "deliberation_score",
            "Design a fault-tolerant multi-region distributed system "
            "for a high-traffic e-commerce platform.",
        )
        assert score >= 0.7
```

## Static typing gate

The whole backend — every first-party package, the top-level modules, and the
test suite — type-checks clean under `mypy --strict`. This is enforced as a
`pytest.mark.unit` test, `tests/test_static_typing_gate.py`, so a typing
regression fails the unit suite just like any other test:

```bash
cd backend
pytest tests/test_static_typing_gate.py -q   # runs mypy --strict over the tree
```

There is no relax/override block in `pyproject.toml`: new code is strict from
the first line. Two rules hold without exception — **never `Any`** (reach for
the most primitive concrete type: `object`, `dict[str, object]`,
`list[object]`, `sqlite3.Row`, covariant `Mapping`/`Sequence`) and **never
`# type: ignore`** (fix the underlying type problem). See
[typing-ratchet.md](typing-ratchet.md) for the supported narrowing patterns
(inline `cast`, write-only `Protocol`s, `setattr` for test monkeypatching).
