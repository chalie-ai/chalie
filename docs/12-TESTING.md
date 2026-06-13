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
