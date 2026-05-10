"""Feature tests for Ability ABC contract (abilities/_base.py).

Asserts that the metaclass enforcement, attribute validation, and lifecycle
hooks behave exactly as the plan specifies — with no mocks, no stubs, no IO.

Subclass namespace isolation: every test-local subclass uses a unique class
name and is deleted with gc.collect() so Ability.__subclasses__() stays clean
between tests.
"""

import gc

import pytest

from abilities._base import Ability

pytestmark = pytest.mark.unit

_VALID_EXAMPLES = ["ex one", "ex two", "ex three", "ex four", "ex five", "ex six"]


def _noop_execute(self, channel, params, telemetry):  # noqa: ARG001
    return {}


def _make_subclass(**attrs):
    """Build an Ability subclass dynamically; triggers __init_subclass__ validation.

    Returns the created class so the construction expression is consumed by
    callers (or by the discard convention). Most call sites ignore the return
    because the test asserts on the side effect — TypeError raised mid-build —
    via ``pytest.raises``.
    """
    namespace = {"execute": _noop_execute, **attrs}
    return type("_TestSubclass", (Ability,), namespace)


# ---------------------------------------------------------------------------
# Required class attribute enforcement
# ---------------------------------------------------------------------------


def test_missing_name_raises_at_class_creation():
    """Subclass without NAME raises TypeError at class body evaluation time."""
    with pytest.raises(TypeError, match="NAME"):
        _make_subclass(SUMMARY="some summary", EXAMPLES=_VALID_EXAMPLES, INPUT_SCHEMA={})

    gc.collect()


def test_missing_summary_raises_at_class_creation():
    """Subclass without SUMMARY raises TypeError at class body evaluation time."""
    with pytest.raises(TypeError, match="SUMMARY"):
        _make_subclass(NAME="missing_summary", EXAMPLES=_VALID_EXAMPLES, INPUT_SCHEMA={})

    gc.collect()


def test_missing_examples_raises_at_class_creation():
    """Subclass without EXAMPLES raises TypeError at class body evaluation time."""
    with pytest.raises(TypeError, match="EXAMPLES"):
        _make_subclass(NAME="missing_examples", SUMMARY="some summary", INPUT_SCHEMA={})

    gc.collect()


def test_missing_input_schema_raises_at_class_creation():
    """Subclass without INPUT_SCHEMA raises TypeError at class body evaluation time."""
    with pytest.raises(TypeError, match="INPUT_SCHEMA"):
        _make_subclass(NAME="missing_schema", SUMMARY="some summary", EXAMPLES=_VALID_EXAMPLES)

    gc.collect()


# ---------------------------------------------------------------------------
# EXAMPLES type + length validation
# ---------------------------------------------------------------------------


def test_examples_non_list_raises():
    """EXAMPLES as a non-list raises TypeError."""
    with pytest.raises(TypeError, match="EXAMPLES"):
        _make_subclass(NAME="examples_string", SUMMARY="some summary", EXAMPLES="not a list", INPUT_SCHEMA={})

    gc.collect()


def test_examples_list_of_non_strings_raises():
    """EXAMPLES with non-string elements raises TypeError."""
    with pytest.raises(TypeError, match="EXAMPLES"):
        _make_subclass(NAME="examples_non_strings", SUMMARY="some summary", EXAMPLES=[1, 2, 3, 4, 5, 6], INPUT_SCHEMA={})

    gc.collect()


def test_examples_too_short_raises():
    """EXAMPLES with fewer than 6 entries raises TypeError."""
    with pytest.raises(TypeError, match="6"):
        _make_subclass(NAME="examples_short", SUMMARY="some summary", EXAMPLES=["a", "b", "c", "d"], INPUT_SCHEMA={})

    gc.collect()


def test_examples_too_long_raises():
    """EXAMPLES with more than 8 entries raises TypeError."""
    with pytest.raises(TypeError, match="8"):
        _make_subclass(
            NAME="examples_long",
            SUMMARY="some summary",
            EXAMPLES=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            INPUT_SCHEMA={},
        )

    gc.collect()


def test_examples_exactly_six_accepted():
    """EXAMPLES with exactly 6 entries is valid (lower boundary)."""

    class _ExamplesSix(Ability):
        NAME = "examples_six"
        SUMMARY = "some summary"
        EXAMPLES = ["a", "b", "c", "d", "e", "f"]
        INPUT_SCHEMA = {}

        def execute(self, channel, params, telemetry):
            return {}

    instance = _ExamplesSix()
    assert len(instance.EXAMPLES) == 6

    del _ExamplesSix
    gc.collect()


def test_examples_exactly_eight_accepted():
    """EXAMPLES with exactly 8 entries is valid (upper boundary)."""

    class _ExamplesEight(Ability):
        NAME = "examples_eight"
        SUMMARY = "some summary"
        EXAMPLES = ["a", "b", "c", "d", "e", "f", "g", "h"]
        INPUT_SCHEMA = {}

        def execute(self, channel, params, telemetry):
            return {}

    instance = _ExamplesEight()
    assert len(instance.EXAMPLES) == 8

    del _ExamplesEight
    gc.collect()


# ---------------------------------------------------------------------------
# execute() abstract enforcement
# ---------------------------------------------------------------------------


def test_subclass_without_execute_cannot_be_instantiated():
    """Concrete subclass missing execute() cannot be instantiated — ABC blocks it."""

    class _NoExecute(Ability):
        NAME = "no_execute"
        SUMMARY = "some summary"
        EXAMPLES = _VALID_EXAMPLES
        INPUT_SCHEMA = {}
        # execute() intentionally omitted

    with pytest.raises(TypeError):
        _NoExecute()

    del _NoExecute
    gc.collect()


# ---------------------------------------------------------------------------
# Valid concrete subclass construction
# ---------------------------------------------------------------------------


def test_valid_concrete_subclass_constructs_cleanly():
    """A fully specified concrete subclass instantiates without error."""

    class _ValidAbility(Ability):
        NAME = "valid_ability"
        SUMMARY = "Does something useful"
        EXAMPLES = ["do the thing", "run the thing", "thing now", "quick thing", "thing please", "start thing"]
        INPUT_SCHEMA = {"type": "object", "properties": {}}

        def execute(self, channel, params, telemetry):
            return {"text": "done"}

    instance = _ValidAbility()
    assert instance.NAME == "valid_ability"
    assert instance.SUMMARY == "Does something useful"
    assert len(instance.EXAMPLES) == 6

    del _ValidAbility
    gc.collect()


# ---------------------------------------------------------------------------
# Default class attribute values
# ---------------------------------------------------------------------------


def test_timeout_defaults_to_ten():
    """TIMEOUT defaults to 10 unless overridden."""

    class _DefaultTimeout(Ability):
        NAME = "default_timeout"
        SUMMARY = "Timeout default check"
        EXAMPLES = ["a", "b", "c", "d", "e", "f"]
        INPUT_SCHEMA = {}

        def execute(self, channel, params, telemetry):
            return {}

    assert _DefaultTimeout.TIMEOUT == 10

    del _DefaultTimeout
    gc.collect()


def test_timeout_overridable():
    """Subclass can override TIMEOUT (e.g. browser overrides to 90)."""

    class _LongTimeout(Ability):
        NAME = "long_timeout"
        SUMMARY = "Long timeout ability"
        EXAMPLES = ["a", "b", "c", "d", "e", "f"]
        INPUT_SCHEMA = {}
        TIMEOUT = 90

        def execute(self, channel, params, telemetry):
            return {}

    assert _LongTimeout.TIMEOUT == 90

    del _LongTimeout
    gc.collect()
