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

import paths

pytestmark = pytest.mark.integration

_MODELS_DIR = str(paths.MODELS_DIR)
_PRETRAINED_DIR = str(paths.PRETRAINED_DIR)


@pytest.fixture(scope="module")
def onnx_svc():
    from services.onnx_inference_service import OnnxInferenceService
    return OnnxInferenceService(_MODELS_DIR, _PRETRAINED_DIR)


class TestDeliberationScoreClassifier:

    def test_architecture_question_scores_high(self, onnx_svc):
        score = onnx_svc.predict_scalar(
            "deliberation_score",
            "Design a fault-tolerant multi-region distributed system for a "
            "high-traffic e-commerce platform.",
        )

        assert score >= 0.7
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
| System-level scenarios | Separate end-to-end harness | Primary system tests — full stack, HTTP to response |
| Benchmarks | Separate end-to-end harness | Scored quality measurement, not pass/fail |

Python-level tests are a fast-feedback supplement. They do not replace the system-level scenario tests, which run end to end in a separate harness against the full stack.

**Boundary-contract sentinels.** Some test files exist specifically to canary a cross-layer invariant that the end-to-end scenarios would catch too late and too silently. `backend/tests/test_rich_media_subagent_isolation.py` (12 tests across three classes) is the reference example. It locks the rich-media round-3 architectural contract: `ActDispatcherService` must not inject `_rich_media_ordinal` on non-user-channel dispatches, and `SubagentAbility` must strip every `<span id='name_N'>…</span>` wrapper from returned text before handing it to the parent. Without these guards, the rich-media end-to-end scenarios regress silently — the LLM stops receiving the rich-media instruction trailer, cards stop rendering, and the only observable symptom is a plain-text response where a card was expected. If a refactor of `act_dispatcher_service.py` or `abilities/subagent.py` causes these sentinels to fail, treat it as a must-fix before merging.

## What If a Service Cannot Be Tested Without Mocks?

**Stop. Do not write the test. Escalate.**

If the only way to test a service is to mock its collaborators, the code is too coupled to test in production shape. Possible resolutions:

- Add a legitimate dependency-injection seam (via the coder agent, not the tester)
- Test the behaviour at a higher level where real collaborators can run
- Accept that the behaviour is covered by an end-to-end system scenario and leave the Python-level test unwritten

The wrong answer is always: write a mock-heavy test that proves nothing.
