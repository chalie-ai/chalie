# Testing Guide

## Philosophy — FEATURE TESTS, ZERO MOCKS, REAL-WORLD ONLY

Every test answers one question: **does this service do what it claims?**

- Given **real** input X
- Through the **real** production stack (real DB, real services, real models, real cache, real config)
- Assert **real** observable output Y

If a test does not fit that shape, it does not belong in the suite. Delete it.

### Hard rules

1. **ZERO MOCKS.** No `MagicMock`, `Mock`, `patch`, `monkeypatch` of production code, stubs, spies, or fakes. The real production stack runs. In-memory SQLite is allowed because it is still SQLite, not a mock.

2. **3–10 feature tests per service. Hard cap.** More than 10 is a design smell — the service is doing too much. Escalate as a system design issue; do not write the 11th test.

3. **Real-world scenarios only.** Test what the service is responsible for, not how it is wired. Contradiction service → does it detect contradictions? Thinking-level classifier → does it route turns to the right level? Memory recall → does it return relevant memories?

4. **No contract / plumbing / version-pin / shape / field-list / call-count / enum-membership tests.** If the behaviour matters, a feature test catches it for free with a better signal. Delete these on sight.

5. **No unit tests unless the unit is pure.** A pure function (no IO, no collaborators, no state — e.g. math helpers, deterministic formatters) may have unit tests. Everything else is a feature test.

6. **The coder agent never writes or modifies tests.** The tester agent has exclusive ownership. See `~/.claude/agents/` for the full agent contracts.

## Quick Start

```bash
cd backend

# Unit tests (pure functions only — fast, in-memory, no external deps)
pytest -m unit

# Feature tests (real production stack — slow, loads real models/services)
pytest -m integration

# A specific test file
pytest tests/test_classifier_features.py

# Verbose output
pytest -m unit -v
```

## Markers

```python
@pytest.mark.unit          # Pure-function tests only — no IO, no collaborators
@pytest.mark.integration   # Real services, real DB, real models — the feature tests
```

**Integration is the default** for anything touching a service, the DB, a model, or the network. The unit marker is reserved for genuinely pure helpers.

## Anatomy of a Feature Test

```python
# backend/tests/test_classifier_features.py

import pytest
from pathlib import Path

pytestmark = pytest.mark.integration

_MODELS_DIR = str(Path(__file__).parent.parent / "data" / "models")


@pytest.fixture(scope="module")
def onnx_svc():
    """Real OnnxInferenceService pointed at production models."""
    from services.onnx_inference_service import OnnxInferenceService
    return OnnxInferenceService(_MODELS_DIR)


class TestThinkingLevelClassifier:

    def test_architecture_question_routes_to_high(self, onnx_svc):
        """Complex distributed systems question → 'high' deliberation level."""
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

What this test does:
- Loads the **real** 596 MB `gte-modernbert-base` encoder
- Loads the **real** thinking-level MLP head from disk (`.npz`)
- Feeds **real** text
- Asserts a **real** classification

What it does not do:
- Mock the ONNX service
- Assert the output dict has fields `{"label", "confidence"}` (that is a contract test)
- Assert the version pin is right (a real feature test catches misconfig faster)

## Naming

```
test_{behavior}_{condition_if_needed}
```

Names describe the **real-world scenario**, not the code path.

```python
# Good — describes what the feature does
def test_simple_factual_lookup_routes_to_low(self):
def test_architecture_question_routes_to_high(self):
def test_lut_temporal_rule_supersedes_old_value(self):

# Bad — describes the plumbing
def test_build_onnx_input_returns_dict(self):
def test_sha256_mismatch_raises_error(self):
def test_all_valid_labels_pass_through(self):
```

## Structure — Arrange / Act / Assert

```python
def test_architecture_question_routes_to_high(self, onnx_svc):
    # Arrange — real user turn, no mocks
    import numpy as np
    onehot = np.zeros((1, 4), dtype=np.float32)
    onehot[0, 0] = 1.0  # none

    # Act — run the real thinking-level classifier
    label, confidence = onnx_svc.predict(
        "thinking_level",
        "Design a fault-tolerant multi-region distributed system.",
        extra_features=onehot,
    )

    # Assert — observable output
    assert label == "high"
    assert confidence > 0.4
```

## Banned Patterns

- `assert True` — tests nothing
- `assert isinstance(x, dict)` as sole assertion — type checks
- `assert x is not None` without content check — truthiness theatre
- Tests with zero assertions
- Any `MagicMock`, `Mock`, `patch`, `monkeypatch` of production code
- Parametrised label pass-throughs (`@pytest.mark.parametrize("label", [...])` where the test just asserts the mock returned the label)
- Tests that assert version strings, SHA pins, weight shapes, field lists, or `.called_with(...)`

## Nightly Scenarios — the Primary Feature Tests

The black-box YAML scenarios at `/Volumes/llm/chalie-nightly-test/scenarios/` are the **primary feature tests** for Chalie. They exercise the entire stack: HTTP → auth → message processing → LLM → memory → response.

Python-level feature tests are a **fast-feedback supplement** for behaviours you want to catch before running a full nightly. They do not replace the nightly scenarios.

See `.claude/commands/nightly-tests.md` for the nightly scenario spec.

## What If a Service Cannot Be Tested Without Mocks?

**STOP.** Do not write the test. Escalate.

If the only way to test a service is to mock its collaborators, the code is too coupled to test in production shape. That is a design signal worth surfacing, not hiding behind a mock.

Possible outcomes:
- Add a legitimate dependency-injection seam (via the coder agent, not the tester)
- Test the service at a higher level where real collaborators can run
- Accept that the behaviour is only covered by a nightly scenario, and leave the Python-level test unwritten

The wrong answer is always: write a mock-heavy test that proves nothing.

## Service Size — the 10-Test Cap

If a service needs more than 10 feature tests to cover its behaviour, it is doing too much. Break it up.

Typical cap:
- Simple services: 3–5 feature tests
- Complex services: 6–10 feature tests
- More than 10: system design problem — escalate

Feature tests are expensive (they load real models and DBs). The cap is not a suggestion — it is a forcing function that keeps services small and focused.

## Pure-Function Unit Tests

The one exception to "feature tests only" — pure functions may have unit tests:

```python
# backend/tests/test_concept_lut_lookup.py

import pytest
pytestmark = pytest.mark.unit

from services.data_graph.lut_engine import _cosine_sim


class TestCosineSim:

    def test_identical_vectors_return_one(self):
        assert _cosine_sim([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        assert _cosine_sim([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
```

Criteria for a legitimate unit test:
- No IO (no disk, no network, no DB)
- No collaborators (no other service instances)
- No state (same input → same output, always)

If any of those are false, it is not a unit test — it is a feature test disguised as one. Write it under `@pytest.mark.integration`.

## Running the Suite

```bash
# Fast — pure functions only (~1 second)
pytest -m unit

# Slow — real stack, real models (loads 596 MB encoder once per module)
pytest -m integration

# Both
pytest

# Parallel (if using pytest-xdist)
pytest -m integration -n 4
```

## Examples in the Codebase

- `backend/tests/test_classifier_features.py` — reference feature test file (real encoder, real thinking-level head, zero mocks)
- `backend/tests/test_concept_lut_lookup.py` — reference pure-function file (LUT KNN, threshold gates)

Study these before writing a new test file. Match the pattern.
