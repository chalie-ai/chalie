# Testing Guide

## Philosophy

Every test answers one question: **does this service do what it claims?**

Given real input X, through the real production stack, assert real observable output Y. If a test does not fit that shape, delete it.

### Hard rules

1. **Zero mocks.** No `MagicMock`, `Mock`, `patch`, `monkeypatch`, stubs, spies, or fakes against production code. In-memory SQLite is allowed — it is still SQLite, not a mock.
2. **3–10 feature tests per service. Hard cap.** More than 10 is a design smell. Escalate; do not write the 11th test.
3. **Real-world scenarios only.** Test what the service is responsible for, not how it is wired internally.
4. **No contract or plumbing tests.** No field-list, call-count, version-pin, shape, or enum-membership assertions. A real feature test catches those failures with better signal.
5. **No unit tests unless the function is pure.** Pure means: no IO, no collaborators, no state. Everything else is a feature test.
6. **The coder agent never writes or modifies tests.** The tester agent has exclusive ownership.

## Markers

| Marker | When to use |
|---|---|
| `@pytest.mark.unit` | Pure functions only — no IO, no collaborators, deterministic |
| `@pytest.mark.integration` | Anything touching a service, DB, model, or network — the default |

`integration` is the default. Apply `unit` only when all three pure-function criteria hold.

## Running the Suite

```bash
cd backend

pytest -m unit          # fast — pure functions, in-memory, ~1 second
pytest -m integration   # slow — real stack, real models
pytest                  # both
```

## Example Feature Test

```python
# backend/tests/test_classifier_features.py

import pytest
from pathlib import Path

pytestmark = pytest.mark.integration

_MODELS_DIR = str(Path(__file__).parent.parent / "data" / "models")


@pytest.fixture(scope="module")
def onnx_svc():
    from services.onnx_inference_service import OnnxInferenceService
    return OnnxInferenceService(_MODELS_DIR)


class TestThinkingLevelClassifier:

    def test_architecture_question_routes_to_high(self, onnx_svc):
        import numpy as np
        onehot = np.zeros((1, 4), dtype=np.float32)
        onehot[0, 0] = 1.0  # none → index 0

        label, confidence = onnx_svc.predict(
            "thinking_level",
            "Design a fault-tolerant multi-region distributed system for a "
            "high-traffic e-commerce platform.",
            extra_features=onehot,
        )

        assert label == "high"
        assert confidence > 0.4
```

Names describe the real-world scenario: `test_architecture_question_routes_to_high`, not `test_predict_returns_dict`.

## Banned Patterns

- `assert True` — tests nothing
- `assert isinstance(x, dict)` as the sole assertion
- `assert x is not None` without a content check
- Tests with zero assertions
- Any `MagicMock`, `Mock`, `patch`, or `monkeypatch` of production code
- Parametrised label pass-throughs where the test only asserts the mock returned what it was given
- Assertions on version strings, SHA pins, weight shapes, field lists, or `.called_with(...)`

## Three Testing Tiers

| Tier | Location | Purpose |
|---|---|---|
| Python feature tests | `backend/tests/` | Fast-feedback for individual service behaviour |
| Nightly YAML scenarios | `chalie-nightly-test/scenarios/` | Primary system tests — full stack, HTTP to response |
| Benchmarks | `chalie-nightly-test/benchmark_tasks/` | Scored quality measurement, not pass/fail |

Python-level tests are a fast-feedback supplement. They do not replace the nightly scenarios. See `.claude/commands/nightly-tests.md` for the nightly scenario spec.

## What If a Service Cannot Be Tested Without Mocks?

**Stop. Do not write the test. Escalate.**

If the only way to test a service is to mock its collaborators, the code is too coupled to test in production shape. Possible resolutions:

- Add a legitimate dependency-injection seam (via the coder agent, not the tester)
- Test the behaviour at a higher level where real collaborators can run
- Accept that the behaviour is covered by a nightly scenario and leave the Python-level test unwritten

The wrong answer is always: write a mock-heavy test that proves nothing.
