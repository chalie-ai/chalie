"""Headline feature tests for find_tools — the shared discovery cascade over the
REAL production abilities.sqlite and the REAL EmbeddingService (ONNX). Zero mocks.

find_tools takes a ``query`` ARRAY of intents — one tool name or one described
action per entry — and runs each through the shared precise→broad cascade
(see ``abilities._search.SearchableAbility``): exact name → bm25 on the tool name
(segment-gated) → vector on the full prose, kept only under the fine-tuned
ceiling. The first rung that yields wins per entry; a row only the vector
terminus fails on is reported honestly under ``not_found``.

These drive the real ``FindToolsAbility.run()`` entry point bound to a real
MessageProcessor carrying the production UserConfig, then assert the downstream
``active_tools`` mutation and the model-facing rendered envelope.
"""

import sqlite3
from typing import cast

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.find_tools import FindToolsAbility
from configs.channels import UserConfig
from services.file_mapper_service import FileMapperService
from services.message_processor import MessageProcessor

pytestmark = pytest.mark.unit


def _run(db: sqlite3.Connection, query: object) -> tuple[list[str], dict[str, object], str]:
    """Drive FindToolsAbility.run() against the real production DB (the ``db``
    fixture's migrated temp SQLite) on a real MessageProcessor. ``hidden_input``
    skips the turn-0 input row — these tests only care about the discovery
    cascade, not transcript persistence. Returns (injected_names, body,
    rendered_envelope)."""
    mp = MessageProcessor(UserConfig({"hidden_input": True}), raw_input="find a tool for me")
    mp.active_tools = list(mp.config.always_available or [])
    base = set(mp.active_tools)
    ability = FindToolsAbility()
    ability.mp = mp
    result = ability.run({"query": query})
    injected = [t for t in mp.active_tools if t not in base]
    rendered = ToolDispatcher._render("find_tools", result)
    body = result.body if isinstance(result.body, dict) else {}
    return injected, cast("dict[str, object]", body), rendered


# ── rung 1 — exact name pins the tool ───────────────────────────────────────────


def test_exact_tool_name_pins_it(db: sqlite3.Connection) -> None:
    injected, _body, rendered = _run(db, ["weather"])
    assert "weather" in injected, f"exact 'weather' must pin the weather tool. injected={injected!r}"
    assert "status=success" in rendered


def test_exact_name_normalises_spacing(db: sqlite3.Connection) -> None:
    """The exact rung normalises punctuation, so the spaced display form 'chalie
    docs' resolves to the underscored ability name 'chalie_docs'."""
    injected, _body, _r = _run(db, ["chalie docs"])
    assert "chalie_docs" in injected, f"'chalie docs' must resolve to chalie_docs. injected={injected!r}"


def test_no_duplicate_injection(db: sqlite3.Connection) -> None:
    injected, _body, _r = _run(db, ["weather"])
    assert injected.count("weather") <= 1, f"weather must inject once, not duplicated. injected={injected!r}"


# ── rung 3 — semantic prose surfaces the right tool ─────────────────────────────


def test_described_action_surfaces_semantic_match(db: sqlite3.Connection) -> None:
    """A described action with no exact/segment name match escalates to the vector
    rung: 'add appointment' surfaces the calendar tool."""
    injected, _body, _r = _run(db, ["add appointment"])
    assert "calendar" in injected, f"'add appointment' must surface calendar. injected={injected!r}"


@pytest.mark.parametrize(
    "query",
    [
        "all available tools and capabilities",
        "what can you do",
        "help with chalie features",
        "documentation",
    ],
)
def test_implicit_prose_query_surfaces_chalie_docs(db: sqlite3.Connection, query: str) -> None:
    """The pre-rework regression: a real implicit prose query (no tool name, no
    markers) returned injected=0 and chalie_docs was never discovered. The vector
    rung must now surface chalie_docs so the model can use it in the same turn."""
    injected, _body, rendered = _run(db, [query])
    assert "chalie_docs" in injected, (
        f"implicit query {query!r} must discover chalie_docs (was injected=0 pre-rework). "
        f"injected={injected!r} rendered={rendered!r}"
    )


# ── ceiling — a junk query escalates to the vector terminus and is dropped ──────


def test_junk_query_is_honest_not_found(db: sqlite3.Connection) -> None:
    """'donut' matches no name and no prose under the fine-tuned ceiling — it must
    surface nothing and be reported honestly under not_found, never forced in."""
    injected, body, _r = _run(db, ["donut"])
    assert injected == [], f"junk 'donut' must inject nothing. injected={injected!r}"
    assert "donut" in cast("list[str]", body["not_found"]), f"junk query must be reported under not_found. body={body!r}"


# ── multi-intent array — each entry routed independently ────────────────────────


def test_multi_intent_array_routes_each_entry(db: sqlite3.Connection) -> None:
    """Each array entry is searched independently: two resolvable intents inject
    their tools, the junk intent is reported under not_found — partial success is
    never silent."""
    injected, body, _r = _run(db, ["weather", "send an email", "pizza"])
    assert "weather" in injected, f"injected={injected!r}"
    assert "email" in injected, f"injected={injected!r}"
    assert "pizza" in cast("list[str]", body["not_found"]), f"junk intent must be surfaced. body={body!r}"


# ── result body shape, dedup, and no legacy tokens ──────────────────────────────


def test_result_body_shape_dedup_and_no_legacy_tokens(db: sqlite3.Connection) -> None:
    """Success body is {"injected": [{name, summary}, …], "not_found": [...]}; the
    meta count matches the list length; the injected list is deduped; and none of
    the dropped legacy fields (input_schema / relevance / added_tools) appear."""
    _injected, body, rendered = _run(db, ["documentation"])

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


def test_no_global_cap_large_array_returns_all_deduped(db: sqlite3.Connection) -> None:
    """The query array exists so the model can fetch every tool it needs in ONE
    pass: a large array of distinct exact names must inject ALL of them (more than
    the old global cap of 6), deduped, never truncated."""
    names = ["weather", "calendar", "email", "chalie_docs", "web_browse", "web_search", "vision"]
    injected, _body, _r = _run(db, names + ["weather"])  # trailing dup must collapse
    for n in ("weather", "calendar", "email", "chalie_docs", "web_browse", "web_search", "vision"):
        assert n in injected, f"{n!r} must be injected (no global cap). injected={injected!r}"
    assert len(injected) > 6, f"a large array must exceed the old cap of 6. injected={injected!r}"
    assert len(injected) == len(set(injected)), f"injected must be deduped. injected={injected!r}"
    assert injected.count("weather") == 1, f"duplicate intent must collapse. injected={injected!r}"


# ── invalid query → loud error, nothing injected ────────────────────────────────


def test_missing_query_is_error_nothing_injected(db: sqlite3.Connection) -> None:
    injected, _body, rendered = _run(db, None)
    assert injected == [], f"a missing query must inject nothing. injected={injected!r}"
    assert "status=error" in rendered
    assert "missing-params" in rendered


def test_empty_query_array_is_error(db: sqlite3.Connection) -> None:
    injected, _body, rendered = _run(db, [])
    assert injected == []
    assert "status=error" in rendered


# ── graceful degradation when the index is unreachable ──────────────────────────


def test_semantic_query_degrades_gracefully_when_db_missing(db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the abilities index absent the bm25/vector rungs return empty (never
    raise); a purely-semantic intent then falls through to an honest not_found
    rather than crashing the turn."""
    monkeypatch.setattr(FindToolsAbility, "_DB_PATH", FileMapperService.get_abilities_db_path().parent / "gone.sqlite")
    _injected, body, rendered = _run(db, ["documentation"])
    assert "status=success" in rendered, f"a missing index must degrade, not error. rendered={rendered!r}"
    assert "documentation" in cast("list[str]", body["not_found"]), f"body={body!r}"


# ── the discoverable roster lives in the tool DESCRIPTION ────────────────────────


def test_discoverable_roster_is_in_the_summary() -> None:
    """The roster of discoverable tools is surfaced in get_summary() (the field
    weak models actually read), query-first, so the model knows what exists before
    it queries. Self-no-ops to bare guidance only if the roster were ever empty."""
    summary = FindToolsAbility().get_summary()
    assert "weather" in summary, f"a discoverable tool must appear in the roster. summary={summary!r}"
    assert "Available tools:" in summary, f"summary={summary!r}"
