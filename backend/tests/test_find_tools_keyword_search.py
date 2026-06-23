"""Headline feature tests for the find_tools keyword search rework.

The PRIMARY regression being pinned: before this rework, a query for
'chalie docs' or 'chalie documentation' returned injected=0 because the
FTS index only held prose summaries (no name entries) and the vector
distance to `chalie_docs` was insufficient for the old floor.

Under the new contract:
- The FTS index (trigram tokenizer) indexes both the tool summary AND
  the tool name via a ``kind='name'`` entry, so '+docs' substring-matches
  'chalie_docs' and 'programming_docs_search' directly.
- Keyword and vector search run independently; top results of each are
  deduped (keyword first). There is no relevance floor.
- The query grammar: +term=required, -term=excluded, bare=optional.

All tests run against the REAL shipped abilities.sqlite and the REAL
EmbeddingService (ONNX local model). Zero mocks.
"""

import json
from typing import cast

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.find_tools import FindToolsAbility
from services.message_processor import MessageProcessor
from tests.helpers import make_stub_config

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared stub-processor factory
# ---------------------------------------------------------------------------

def _stub_proc() -> MessageProcessor:
    proc = object.__new__(MessageProcessor)
    proc.config = make_stub_config()
    proc._active_tools = []
    return proc


def _run(query: str) -> tuple[list[str], str]:
    """Drive FindToolsAbility.run() against the real production DB.
    Returns (active_tools, rendered_result)."""
    ability = FindToolsAbility()
    proc = _stub_proc()
    ability.mp = proc
    result = ability.run({"query": query})
    rendered = ToolDispatcher._render("find_tools", result)
    return proc.active_tools, rendered


def _parse_body(rendered: str) -> dict[str, object]:
    """Extract the JSON body from the rendered tool result envelope."""
    start = rendered.index("]\n") + 2
    end = rendered.index("\n[end:find_tools]")
    return cast("dict[str, object]", json.loads(rendered[start:end]))


# ---------------------------------------------------------------------------
# Test 1 — 'chalie docs' injects chalie_docs (the pre-rework regression)
# ---------------------------------------------------------------------------


def test_chalie_docs_query_injects_chalie_docs() -> None:
    """Regression: 'chalie docs' returned injected=0 before the rework.

    Now the trigram name entry lets the keyword path find 'chalie_docs'
    via the 'docs' substring even though the prose summary may not
    contain the word 'docs'."""
    active, rendered = _run("chalie docs")

    assert "chalie_docs" in active, (
        f"'chalie docs' query must inject chalie_docs (pre-rework regression). "
        f"Got active_tools={active!r}. Rendered: {rendered!r}"
    )


# ---------------------------------------------------------------------------
# Test 2 — 'chalie documentation' injects chalie_docs (semantic path)
# ---------------------------------------------------------------------------


def test_chalie_documentation_query_injects_chalie_docs() -> None:
    """The vector path matches 'chalie documentation' to chalie_docs even
    when 'documentation' does not appear verbatim in the name."""
    active, rendered = _run("chalie documentation")

    assert "chalie_docs" in active, (
        f"'chalie documentation' query must inject chalie_docs. "
        f"Got active_tools={active!r}. Rendered: {rendered!r}"
    )


# ---------------------------------------------------------------------------
# Test 3 — '+docs' required substring matches chalie_docs via name entry
# ---------------------------------------------------------------------------


def test_required_docs_term_injects_chalie_docs() -> None:
    """'+docs' is a required keyword term. The trigram index on the name
    entry (kind='name') substring-matches 'chalie_docs' via 'docs'."""
    active, rendered = _run("+docs")

    assert "chalie_docs" in active, (
        f"'+docs' must inject chalie_docs via trigram name-entry substring match. "
        f"Got active_tools={active!r}. Rendered: {rendered!r}"
    )


# ---------------------------------------------------------------------------
# Test 4 — '+calendar -delete' injects calendar and respects exclusion
# ---------------------------------------------------------------------------


def test_calendar_minus_delete_injects_calendar_not_delete_tools() -> None:
    """+calendar injects the calendar tool; -delete excludes anything whose
    only match was the 'delete' substring. The exclusion must not prevent
    calendar from appearing (its match comes from 'calendar', not 'delete')."""
    active, rendered = _run("+calendar -delete")

    assert "calendar" in active, (
        f"'calendar' must be injected by '+calendar -delete'. "
        f"Got active_tools={active!r}"
    )
    delete_tools = [t for t in active if "delete" in t.lower()]
    assert not delete_tools, (
        f"No tool containing 'delete' should be injected when '-delete' is "
        f"in the query. Got: {delete_tools!r}. active_tools={active!r}"
    )


# ---------------------------------------------------------------------------
# Test 5 — result body shape and count invariant
# ---------------------------------------------------------------------------


def test_result_body_shape_and_count_invariant() -> None:
    """The success body is always {"injected": [...], "not_found": []} with
    the meta injected= count matching the list length and ≤ 6 tools."""
    active, rendered = _run("chalie docs")

    assert "status=success" in rendered, (
        f"Result must be success when tools are found. rendered={rendered!r}"
    )
    body = _parse_body(rendered)
    assert isinstance(body, dict), f"Body must be a dict, got {type(body)}"
    assert "injected" in body, f"Body must have 'injected' key. body={body!r}"
    assert "not_found" in body, f"Body must have 'not_found' key. body={body!r}"
    assert isinstance(body["injected"], list)
    assert isinstance(body["not_found"], list)
    for row in body["injected"]:
        assert "name" in row, f"Each injected row must have 'name'. row={row!r}"
        assert "summary" in row, f"Each injected row must have 'summary'. row={row!r}"
    # Meta count == list length
    head = rendered.splitlines()[0]
    meta_count = int(head.split("injected=")[1].split(",")[0].rstrip(")"))
    assert meta_count == len(body["injected"]), (
        f"Meta injected={meta_count} must equal len(injected)={len(body['injected'])}"
    )
    assert len(body["injected"]) <= 6, (
        f"At most 6 tools may be injected (2×_RESULT_CAP=3). "
        f"Got {len(body['injected'])}: {[r['name'] for r in body['injected']]}"
    )


# ---------------------------------------------------------------------------
# Test 6 — stopword-only / sub-3-char query does not crash
# ---------------------------------------------------------------------------


def test_stopword_only_query_does_not_crash_and_returns_valid_result() -> None:
    """A query consisting entirely of stopwords ('of to') has no FTS terms
    after build_keyword_query drops them. The vector path still runs with
    empty embed_text (skipped), so no crash — the result is injected=0 success."""
    active, rendered = _run("of to")

    # Must not raise; result must be a valid success (empty is OK)
    assert "status=success" in rendered, (
        f"Stopword-only query must not error. rendered={rendered!r}"
    )
    body = _parse_body(rendered)
    assert isinstance(body.get("injected"), list), (
        f"Body must have injected list even on zero hits. body={body!r}"
    )
    # active_tools is [] for this case — any extra tools from the vector
    # path should not be present when no positive terms exist.
    assert active == [], (
        f"Stopword-only query must inject nothing. active_tools={active!r}"
    )
