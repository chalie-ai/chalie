"""Feature tests for the Ability ABC contract (abilities/_ability.py).

SPEC CHANGE: the five ``NAME`` / ``SUMMARY`` / ``EXAMPLES`` /
``SEARCH_TOOLTIP`` / ``INPUT_SCHEMA`` ClassVars were replaced by five zero-arg
``@abstractmethod`` getters (``get_name`` / ``get_summary`` / ``get_examples`` /
``get_search_tooltip`` / ``get_parameters``), and ``get_input_schema()`` became a
``@typing.final`` template method — the SINGLE place a tool descriptor is built
and the SINGLE injection site for the framework field ``act_summary`` (always,
required). The ``async`` backgrounding flag is NOT a base-``Ability`` field: it is
a delegate-only primitive added by ``DelegateAbility._inject_framework_fields``
(iff the bound processor's config sets ``SUPPORTS_ASYNC``), so plain tools never
carry it. The MessageProcessor is now constructor-injected
(``Ability(mp=...)`` → ``self.mp``).

Because the metadata is now carried by ``@abstractmethod`` getters, a subclass
that omits one is *abstract* — it cannot be instantiated and so never reaches the
registry. ``get_input_schema`` is sealed by ``@typing.final``, which the type
checker enforces; there is no runtime guard duplicating it.

Subclass namespace isolation: every test-local subclass uses a unique class name
and is deleted with gc.collect() so Ability.__subclasses__() stays clean between
tests.
"""

import gc
from typing import TYPE_CHECKING, cast

import pytest

from abilities._ability import Ability
from abilities._delegate import DelegateAbility
from configs.channels import DmnConfig, UserConfig

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

pytestmark = pytest.mark.unit

_VALID_EXAMPLES = ["ex one", "ex two", "ex three", "ex four", "ex five", "ex six"]


class _Mp:
    """Minimal real MP-shaped context — ``_inject_framework_fields`` reads only
    ``self.mp.config.SUPPORTS_ASYNC`` off it, carrying a REAL channel config
    (UserConfig backgrounds, DmnConfig does not). Not a mock — the real config
    object decides the gate."""

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
        "get_name": lambda self: name,
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


def test_missing_name_getter_makes_class_abstract() -> None:
    """A subclass that omits get_name is abstract — instantiation raises TypeError
    naming the missing getter. (Replaces the old 'missing NAME ClassVar' guard.)"""
    cls = _make_subclass("_MissingName", drop=("get_name",))
    with pytest.raises(TypeError, match="get_name"):
        cls()
    del cls
    gc.collect()


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
        get_name=lambda self: "valid_ability",
        get_summary=lambda self: "Does something useful",
        get_parameters=lambda self: {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    )
    instance = cls()

    assert instance.get_name() == "valid_ability"
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


def test_async_is_delegate_only_never_on_a_plain_tool() -> None:
    """async is a DELEGATE-ONLY primitive: a plain Ability NEVER exposes it, even
    on a SUPPORTS_ASYNC channel — that is the structural bound against a plain
    `read`/`shell` being backgrounded."""
    cls = _make_subclass("_PlainAsyncProbe")

    # Even on the SUPPORTS_ASYNC user channel, a plain tool has no async flag.
    user_props = cast("dict[str, object]", cast("dict[str, object]", cls(mp=cast("MessageProcessor", _Mp(UserConfig({})))).get_input_schema()["input_schema"])["properties"])
    assert "async" not in user_props
    assert "act_summary" in user_props  # the base framework field is still injected

    del cls
    gc.collect()


def test_async_injected_only_under_supports_async_config_for_delegates() -> None:
    """On a DelegateAbility, async appears ONLY when the bound mp's config sets
    SUPPORTS_ASYNC. It is a per-call deepcopy — never mutates get_parameters()."""
    shared_params: dict[str, object] = {"type": "object", "properties": {}, "required": []}
    cls = _make_subclass("_DelegateAsyncProbe", base=cast("type[Ability]", DelegateAbility), get_parameters=lambda self: shared_params)

    # SUPPORTS_ASYNC channel (UserConfig) → async exposed.
    user_props = cast("dict[str, object]", cast("dict[str, object]", cls(mp=cast("MessageProcessor", _Mp(UserConfig({})))).get_input_schema()["input_schema"])["properties"])
    assert "async" in user_props
    assert cast("dict[str, object]", user_props["async"])["type"] == "boolean"
    assert cast("dict[str, object]", user_props["async"])["default"] is False

    # Non-async channel (DmnConfig) → never exposed.
    dmn_props = cast("dict[str, object]", cast("dict[str, object]", cls(mp=cast("MessageProcessor", _Mp(DmnConfig()))).get_input_schema()["input_schema"])["properties"])
    assert "async" not in dmn_props

    # No live processor (mp=None — build/introspection) → never exposed.
    none_props = cast("dict[str, object]", cast("dict[str, object]", cls().get_input_schema()["input_schema"])["properties"])
    assert "async" not in none_props

    # The declared params dict was never mutated by any of the above.
    assert "async" not in cast("dict[str, object]", shared_params["properties"])
    assert "act_summary" not in cast("dict[str, object]", shared_params["properties"])

    del cls
    gc.collect()
