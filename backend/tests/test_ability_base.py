"""Feature tests for the Ability ABC contract (abilities/_ability.py).

The identity contract: each ability declares ``NAME: ClassVar[str]`` — the
single registry name every caller reads.
The remaining metadata is carried by four zero-arg ``@abstractmethod`` getters
(``get_summary`` / ``get_examples`` / ``get_search_tooltip`` /
``get_parameters``), and ``get_input_schema()`` is a ``@typing.final`` template
method — the SINGLE place a tool descriptor is built and the SINGLE injection
site for the framework field ``act_summary`` (always, required).
The MessageProcessor is constructor-injected (``Ability(mp=...)`` → ``self.mp``).

A subclass that omits a getter is *abstract* — it cannot be instantiated and so
never reaches the registry. A missing ``NAME`` is caught by the registry's
loud ValueError at load time (see test_ability_registry.py).
``get_input_schema`` is sealed by ``@typing.final``, which the type checker
enforces; there is no runtime guard duplicating it.

Subclass namespace isolation: every test-local subclass uses a unique class name
and is deleted with gc.collect() so Ability.__subclasses__() stays clean between
tests.
"""

import gc
from typing import TYPE_CHECKING, cast

import pytest

from abilities._ability import Ability

if TYPE_CHECKING:
    pass

pytestmark = pytest.mark.unit

_VALID_EXAMPLES = ["ex one", "ex two", "ex three", "ex four", "ex five", "ex six"]


class _Mp:
    """Minimal real MP-shaped context carrying a REAL channel config — not a
    mock, the real config object decides any gate that reads it."""

    def __init__(self, config: object) -> None:
        self.config = config


def _getters(
    name: str = "valid_ability",
    summary: str = "Does something useful",
    examples: "list[str] | None" = None,
    tooltip: str = "a useful tool",
    parameters: "dict[str, object] | None" = None,
) -> "dict[str, object]":
    """Build the concrete-ability namespace dict."""
    examples = list(_VALID_EXAMPLES) if examples is None else examples
    parameters = {"type": "object", "properties": {}, "required": []} if parameters is None else parameters
    return {
        "NAME": name,
        "get_summary": lambda self: summary,
        "get_examples": lambda self: examples,
        "get_search_tooltip": lambda self: tooltip,
        "get_parameters": lambda self: parameters,
        "run": lambda self, params: {"text": "done"},
    }


def _make_subclass(
    clsname: str,
    drop: "tuple[str, ...]" = (),
    base: "type[Ability]" = Ability,
    **overrides: object,
) -> "type[Ability]":
    """Build an Ability subclass dynamically.

    ``drop`` names getters to OMIT (leaving the abstractmethod unfilled, so the
    class stays abstract). ``base`` picks the parent (``Ability`` by default, or
    ``DelegateAbility`` to exercise the delegate-only ``async`` injection).
    ``overrides`` replace individual namespace members.
    """
    namespace = _getters()
    namespace.update(overrides)
    for member in drop:
        namespace.pop(member, None)
    return type(clsname, (base,), namespace)


# ---------------------------------------------------------------------------
# Missing getter / run → abstract → cannot be instantiated (ABC enforcement)
# ---------------------------------------------------------------------------


def test_missing_examples_getter_makes_class_abstract() -> None:
    """A subclass that omits get_examples is abstract — instantiation raises
    TypeError naming the missing getter, so it never reaches the registry."""
    cls = _make_subclass("_MissingExamples", drop=("get_examples",))
    with pytest.raises(TypeError, match="get_examples"):
        cls()
    del cls
    gc.collect()


def test_subclass_without_run_cannot_be_instantiated() -> None:
    """Concrete metadata but missing run() → still abstract → ABC blocks it."""
    cls = _make_subclass("_NoRun", drop=("run",))
    with pytest.raises(TypeError, match="run"):
        cls()
    del cls
    gc.collect()


# ---------------------------------------------------------------------------
# The single assembler: shape + framework-field injection
# ---------------------------------------------------------------------------


def test_valid_concrete_subclass_assembles_full_descriptor() -> None:
    """A fully specified subclass instantiates and get_input_schema() returns the
    full LLM-facing descriptor assembled from the getters."""
    cls = _make_subclass(
        "_ValidAbility",
        get_parameters=lambda self: {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    )
    instance = cls()

    assert instance.NAME == "valid_ability"
    assert instance.get_summary() == "Does something useful"
    assert len(instance.get_examples()) == 6

    schema = instance.get_input_schema()
    assert set(schema) == {"name", "description", "input_schema"}
    assert schema["name"] == "valid_ability"
    assert schema["description"] == "Does something useful"
    # The declared param survived assembly.
    assert cast("dict[str, object]", cast("dict[str, object]", schema["input_schema"])["properties"])["q"] == {"type": "string"}

    del cls
    gc.collect()


def test_act_summary_always_injected_and_required() -> None:
    """act_summary is injected into EVERY descriptor and marked required — the one
    framework field present regardless of channel or mp."""
    cls = _make_subclass("_ActSummaryProbe")
    schema = cast("dict[str, object]", cls().get_input_schema()["input_schema"])
    assert "act_summary" in cast("dict[str, object]", schema["properties"])
    assert cast("dict[str, object]", cast("dict[str, object]", schema["properties"])["act_summary"])["type"] == "string"
    assert "act_summary" in cast("list[str]", schema["required"])
    del cls
    gc.collect()
