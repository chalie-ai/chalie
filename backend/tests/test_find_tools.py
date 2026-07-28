"""Feature tests for ``find_tools`` — the exact-match discovery contract.

``find_tools`` takes a ``query`` ARRAY of tool names — one name per entry — and
runs each through a single normalized lookup against the registry's alias map
(canonical names plus each ability's ``SEARCHABLE_AS`` tuple). A hit activates
the tool and returns it under ``body["injected"]`` with its tooltip; a miss
lands in ``body["not_found"]`` with a pointer back to the tool list.

No fuzzy search, no semantic vector rung, no MCP, no ``abilities.sqlite`` —
this tool is a precise menu.
"""

from typing import cast

import pytest

from abilities._result import ToolResult
from abilities._registry import AbilityRegistry
from abilities.find_tools import FindToolsAbility
from configs.channels import UserConfig
from configs.enums.ability_category import AbilityCategory
from contracts.params.find_tools_params_bag import FindToolsParamsBag
from controllers.message_processor import MessageProcessor
from services.dispatch_service import DispatchService

pytestmark = pytest.mark.unit


def _mp_with_loaded(names: list[str]) -> MessageProcessor:
    """A real MessageProcessor whose ``active_tools`` is exactly *names* — the one
    field ``get_summary()`` subtracts from the menu. Real, not a stub: the summary
    reads it off the live processor on every ``build_tools`` pass."""
    mp = MessageProcessor(UserConfig({"hidden_input": True}), raw_input="find a tool for me")
    mp.active_tools = list(names)
    return mp


def _run(query: object) -> tuple[list[str], dict[str, object], str]:
    """Drive ``FindToolsAbility.run()`` on a real ``MessageProcessor`` carrying
    the production ``UserConfig``. Returns ``(injected_names, body, rendered)``."""
    mp = MessageProcessor(UserConfig({"hidden_input": True}), raw_input="find a tool for me")
    mp.active_tools = list(mp.config.always_available or [])
    base = set(mp.active_tools)
    ability = FindToolsAbility()
    ability.mp = mp
    bag = FindToolsParamsBag.from_params({"query": query})
    result = bag if isinstance(bag, ToolResult) else ability.run(bag)
    injected = [t for t in mp.active_tools if t not in base]
    rendered = DispatchService(mp=cast("MessageProcessor", None))._render("find_tools", result)
    body = result.body if isinstance(result.body, dict) else {}
    return injected, cast("dict[str, object]", body), rendered


# ── rung 1 — exact name pins the tool ───────────────────────────────────────────


def test_exact_tool_name_pins_it() -> None:
    injected, _body, rendered = _run(["weather"])
    assert "weather" in injected, f"'weather' must pin the weather tool. injected={injected!r}"
    assert "status=success" in rendered


def test_exact_name_normalises_spacing() -> None:
    """The normaliser collapses punctuation, so 'chalie docs' resolves to
    the underscored ability name 'chalie_docs'."""
    injected, _body, _r = _run(["chalie docs"])
    assert "chalie_docs" in injected, f"'chalie docs' must resolve to chalie_docs. injected={injected!r}"


def test_no_duplicate_injection() -> None:
    injected, _body, _r = _run(["weather"])
    assert injected.count("weather") <= 1, f"weather must inject once, not duplicated. injected={injected!r}"


# ── aliases — SEARCHABLE_AS entries resolve to the canonical tool ───────────────


def test_alias_email_resolves_to_pim() -> None:
    """``email`` is declared in ``PimAbility.SEARCHABLE_AS`` — it must resolve to
    the ``pim`` canonical name, not the raw delegate-owned email tool."""
    injected, _body, _r = _run(["email"])
    assert "pim" in injected, f"'email' must resolve to pim. injected={injected!r}"


def test_alias_calendar_resolves_to_pim() -> None:
    """``calendar`` is declared in ``PimAbility.SEARCHABLE_AS`` — it must resolve
    to the ``pim`` canonical name."""
    injected, _body, _r = _run(["calendar"])
    assert "pim" in injected, f"'calendar' must resolve to pim. injected={injected!r}"


# ── chalie_docs aliases and prose rejection ─────────────────────────────────────


def test_alias_chalie_documentation_resolves_to_chalie_docs() -> None:
    """``chalie documentation`` is declared in ``ChalieDocsAbility.SEARCHABLE_AS``
    and must resolve to the canonical ``chalie_docs`` name."""
    injected, _body, _r = _run(["chalie documentation"])
    assert "chalie_docs" in injected, (
        f"'chalie documentation' must resolve to chalie_docs. injected={injected!r}"
    )


def test_alias_harness_documentation_resolves_to_chalie_docs() -> None:
    """``harness documentation`` is declared in ``ChalieDocsAbility.SEARCHABLE_AS``
    and must resolve to the canonical ``chalie_docs`` name."""
    injected, _body, _r = _run(["harness documentation"])
    assert "chalie_docs" in injected, (
        f"'harness documentation' must resolve to chalie_docs. injected={injected!r}"
    )


def test_prose_query_is_loud_no_results() -> None:
    """Prose like 'what can you do' no longer discovers anything — an all-miss
    is a loud no-results error with the pick-an-exact-name hint."""
    injected, _body, rendered = _run(["what can you do"])
    assert "chalie_docs" not in injected, (
        f"prose must NOT discover chalie_docs. injected={injected!r}"
    )
    assert "status=error" in rendered and "code=no-results" in rendered, (
        f"all-miss must be a loud no-results error. rendered={rendered!r}"
    )
    assert "Pick an exact tool name" in rendered, (
        f"the guidance hint is missing. rendered={rendered!r}"
    )


# ── junk queries are a loud all-miss ────────────────────────────────────────────


def test_junk_query_is_loud_no_results() -> None:
    """``donut`` matches no name and no alias — it must inject nothing and
    surface as a no-results error, never be forced in."""
    injected, _body, rendered = _run(["donut"])
    assert injected == [], f"junk 'donut' must inject nothing. injected={injected!r}"
    assert "status=error" in rendered and "code=no-results" in rendered, (
        f"all-miss must be a loud no-results error. rendered={rendered!r}"
    )
    assert "No results found." in rendered


def test_multi_intent_array_routes_each_entry() -> None:
    """Each array entry is searched independently: two resolvable intents inject
    their tools, the junk intent is reported under ``not_found`` — partial
    success is never silent."""
    injected, body, _r = _run(["weather", "email", "pizza"])
    assert "weather" in injected, f"injected={injected!r}"
    assert "pim" in injected, f"'email' must resolve to pim. injected={injected!r}"
    assert any("pizza" in e for e in cast("list[str]", body["not_found"])), (
        f"junk intent must be surfaced. body={body!r}"
    )


# ── result body shape, dedup, and no legacy tokens ─────────────────────────────


def test_result_body_shape_dedup_and_no_legacy_tokens() -> None:
    """Success body is ``{"injected": [{name, summary}, …], "not_found":
    [...]}``; the meta count matches the list length; the injected list is
    deduped; and none of the dropped legacy fields (input_schema / relevance /
    added_tools) appear."""
    _injected, body, rendered = _run(["chalie_docs"])

    assert "status=success" in rendered
    assert isinstance(body["injected"], list)
    assert isinstance(body["not_found"], list)
    names = [cast("dict[str, object]", row)["name"] for row in cast("list[object]", body["injected"])]
    for row in cast("list[object]", body["injected"]):
        assert set(cast("dict[str, object]", row).keys()) == {"name", "summary"}, f"row={row!r}"
    head = rendered.splitlines()[0]
    meta_count = int(head.split("injected=")[1].split(",")[0].rstrip(")]"))
    assert meta_count == len(names)
    assert len(names) == len(set(names)), f"injected must be deduped. names={names!r}"
    for legacy in ("input_schema", "relevance", "added_tools"):
        assert legacy not in rendered, f"legacy token {legacy!r} must be gone. rendered={rendered!r}"


def test_no_global_cap_large_array_returns_all_deduped() -> None:
    """The query array exists so the model can fetch every tool it needs in ONE
    pass: a large array of distinct exact names must inject ALL of them
    (more than the old global cap of 6), deduped, never truncated."""
    names = ["weather", "pim", "timer", "chalie_docs", "web_browse", "web_search", "vision"]
    injected, body, _r = _run(names + ["weather"])  # trailing dup must collapse
    for n in ("weather", "pim", "timer", "chalie_docs", "web_browse", "web_search", "vision"):
        assert n in injected, f"{n!r} must be injected (no global cap). injected={injected!r}"
    assert len(injected) > 6, f"a large array must exceed the old cap of 6. injected={injected!r}"
    assert len(injected) == len(set(injected)), f"injected must be deduped. injected={injected!r}"
    assert injected.count("weather") == 1, f"duplicate intent must collapse. injected={injected!r}"
    body_names = [cast("dict[str, object]", row)["name"] for row in cast("list[object]", body["injected"])]
    assert body_names.count("weather") == 1, (
        f"the BODY must not over-report a duplicate entry. body_names={body_names!r}"
    )


# ── edge-trim leniency — normalized forms all resolve ──────────────────────────


def test_edge_trim_normalized_forms_resolve() -> None:
    """The normaliser tolerates leading/trailing whitespace, mixed case, and
    punctuation: ``_weather_``, `` Weather:``, ``WEB_SEARCH``, ``web search!``
    all resolve to their canonical tools."""
    injected, _body, _r = _run(["_weather_", " Weather:", "WEB_SEARCH", "web search!"])
    assert "weather" in injected, f"'_weather_' must resolve to weather. injected={injected!r}"
    assert "web_search" in injected, f"'WEB_SEARCH'/'web search!' must resolve to web_search. injected={injected!r}"


# ── the discoverable roster lives in the tool DESCRIPTION ────────────────────────


def test_discoverable_roster_is_in_the_summary() -> None:
    """The roster of discoverable tools is surfaced in ``get_summary()`` (the
    field weak models actually read), so the model knows what exists before it
    queries. Off-spine (``mp is None``) nothing is loaded, so the FULL roster
    renders — which is also what keeps this text deterministic."""
    summary = FindToolsAbility().get_summary()
    assert "**Available Tools**" in summary, f"summary={summary!r}"
    for name in AbilityRegistry.discoverable_names():
        assert f"- `{name}`: " in summary, (
            f"discoverable tool '{name}' missing from the menu. summary={summary!r}"
        )


def test_summary_lists_each_tool_under_its_category_heading() -> None:
    """Every rendered line sits under its ability's own CATEGORY heading, and the
    headings come out in ``AbilityCategory`` declaration order — the enum is the
    only ordering authority, there is no second sort key to drift from it."""
    summary = FindToolsAbility().get_summary()

    headings = [c for c in AbilityCategory if f"\n{c.value}:\n" in f"\n{summary}\n"]
    positions = [summary.index(f"{c.value}:") for c in headings]
    assert positions == sorted(positions), (
        f"headings must render in enum declaration order. got={headings}"
    )

    for name in AbilityRegistry.discoverable_names():
        ability = AbilityRegistry.get(name)
        assert ability.CATEGORY is not None
        block = summary.split(f"{ability.CATEGORY.value}:\n", 1)[1].split("\n\n", 1)[0]
        assert f"- `{name}`: {ability.get_search_tooltip()}" in block, (
            f"'{name}' is not listed under '{ability.CATEGORY.value}'. block={block!r}"
        )


def test_summary_excludes_tools_already_loaded_in_context() -> None:
    """A tool the model can already call is noise in a menu whose only job is
    loading tools it cannot — so anything in ``mp.active_tools`` is dropped, and a
    category emptied by that drops with it rather than leaving a bare heading."""
    # Derived, not hardcoded: the "emptied category drops" half of this test
    # only holds if EVERY File Operations tool is loaded, and that roster grows
    # (write_file / make_dir / delete / set_permissions replaced one merged
    # tool). A literal list silently stops testing the second assertion the day
    # a tool is added to the category.
    file_ops = [
        name for name in AbilityRegistry.discoverable_names()
        if AbilityRegistry.get(name).CATEGORY is AbilityCategory.FILE_OPERATIONS
    ]
    assert len(file_ops) == 8, f"expected the 8 file primitives, got {sorted(file_ops)}"
    loaded = ["find_tools", *file_ops]
    summary = FindToolsAbility(mp=_mp_with_loaded(loaded)).get_summary()

    for name in loaded:
        assert f"- `{name}`: " not in summary, (
            f"'{name}' is already loaded and must not be offered. summary={summary!r}"
        )
    assert f"{AbilityCategory.FILE_OPERATIONS.value}:" not in summary, (
        f"an emptied category must not render a bare heading. summary={summary!r}"
    )
    # Everything NOT loaded is still on offer.
    assert "- `weather`: " in summary, f"summary={summary!r}"
    assert f"{AbilityCategory.INFORMATION.value}:" in summary, f"summary={summary!r}"


def test_summary_says_so_when_every_tool_is_already_loaded() -> None:
    """An empty "**Available Tools**" heading reads as "no tools exist" — the
    opposite of the truth. With nothing left to offer, say nothing is left."""
    everything = sorted(AbilityRegistry.discoverable_names())
    summary = FindToolsAbility(mp=_mp_with_loaded(everything)).get_summary()

    assert "**Available Tools**" not in summary, f"summary={summary!r}"
    assert "already loaded in context" in summary, f"summary={summary!r}"


def test_summary_tells_the_model_how_to_call_find_tools() -> None:
    """The menu is useless without the calling convention above it: names go in
    ``query``, verbatim."""
    summary = FindToolsAbility().get_summary()
    assert "`find_tools`" in summary, f"summary={summary!r}"
    assert "VERBATIM" in summary, f"summary={summary!r}"
    assert "`query`" in summary, f"summary={summary!r}"


# ── collision guard — duplicate SEARCHABLE_AS entries raise at discovery ─────────


class _AliasStub:
    """Duck-typed registry entry for the collision-guard test. Deliberately NOT
    an ``Ability`` subclass: a concrete subclass would stay visible to
    ``Ability.__subclasses__()`` for the rest of the process and poison every
    later registry rebuild with a permanent alias collision."""

    DISCOVERABLE = True
    SEARCHABLE_AS = ("shared_alias",)

    def __init__(self, name: str) -> None:
        self.NAME = name


def test_collision_guard_raises_runtime_error_on_duplicate_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two registry entries declaring the same SEARCHABLE_AS alias must make
    ``AbilityRegistry.discovery_aliases()`` raise ``RuntimeError`` naming both
    abilities and the alias. Only the registry CACHE is monkeypatched (auto-
    restored) — the real class table is never touched."""
    import abilities._registry as _reg_module

    monkeypatch.setattr(
        _reg_module,
        "_registry",
        {"collision_a": _AliasStub("collision_a"), "collision_b": _AliasStub("collision_b")},
    )
    with pytest.raises(RuntimeError, match="shared alias") as exc_info:
        AbilityRegistry.discovery_aliases()
    err = str(exc_info.value)
    assert "collision_a" in err and "collision_b" in err, (
        f"RuntimeError must name both abilities. got={err!r}"
    )


def test_collision_guard_leaves_real_registry_intact() -> None:
    """After the monkeypatched collision test, the real registry must resolve
    without collision and still carry the canonical names."""
    aliases = AbilityRegistry.discovery_aliases()
    assert isinstance(aliases, dict)
    assert "weather" in aliases
