# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""list-specific business-logic tests migrated from the per-ability conformance
file removed in TKT-975. Drives the real ToolDispatcher end-to-end hot path with
zero mocks, exercising list name addressing, item disambiguation, rich card
payloads, and the enrich_rich_payload refresh hook."""

import json
import sqlite3
from typing import cast

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from services.act_trail import ActTrail
from tests._tool_result_harness import MP, allow_policy, parse_body, seed_transcript

pytestmark = pytest.mark.unit


@pytest.fixture
def chat_mp(db: sqlite3.Connection) -> MP:
    """A real chat-channel mp bound to the test database, with a seeded transcript
    anchor for the act-trail write and ``list.delete`` flipped to ``allow`` in the
    real policy table so the destructive gate passes through to run()."""
    allow_policy(db, "list.delete", "chat")
    return MP(seed_transcript(db, "chat", "manage my lists"), UserConfig({}))


@pytest.fixture
def dmn_mp(db: sqlite3.Connection) -> MP:
    """A real non-user-broadcast mp (``broadcast_to=None``). On this channel the
    dispatcher drops the rich card, so the rendered body head is the raw
    model-facing rows (id/text/done/position) — the shape the model reads —
    instead of the frontend card payload. The subconscious policy channel ships
    list.add/list_all as ``allow``."""
    from configs.channels import DmnConfig
    return MP(seed_transcript(db, "subconscious", "manage my lists"), DmnConfig())


def _service() -> object:
    """A real ``ListService`` bound to the test database — built off the same
    ``get_shared_db_service()`` singleton the ``db`` fixture patches, exactly as
    production ``ListAbility.run`` builds it."""
    from services.database_service import get_shared_db_service
    from services.list_service import ListService

    return ListService(get_shared_db_service())


def _parse_body(rendered: str, tool: str = "list") -> object:
    """Extract and JSON-parse the body between the open tag and ``[end:<tool>]`` —
    proves the envelope the model receives is structured and machine-parseable.

    For a rich card on a user-broadcast turn the body is ``<card_json>\\n\\n
    <instruction>``; the ``rich=True`` harness extractor returns the card payload
    (the JSON head before the blank line). On a non-broadcast turn the body is a
    single JSON block with no blank line, so the split is a no-op."""
    return parse_body(rendered, tool, rich=True)


# ── Automation: name addressing ────────────────────────────────────────────────


def test_add_by_exact_name_resolves_target(db: sqlite3.Connection, chat_mp: MP) -> None:
    """``add list="shopping" items=["milk"]`` resolves the target by NAME — no
    list-then-act round trip — and the item lands in the service-backed list."""
    from services.list_service import ListService
    service = cast(ListService, _service())
    list_id = service.create_list("shopping")

    out = ToolDispatcher(chat_mp).dispatch(
        "list", {"action": "add", "list": "shopping", "items": ["milk"], "act_summary": "x"}
    )

    assert "[list(status=success" in out, out
    # On a user turn the body head is the FE card; it still carries the resolved id.
    card = cast(dict[str, object], _parse_body(out))
    assert card["id"] == list_id, card
    assert card["name"] == "shopping", card

    # Downstream: the item really landed in the service-backed list.
    contents = [i["content"] for i in cast(list[dict[str, object]], cast(dict[str, object], service.get_list(list_id))["items"])]
    assert "milk" in contents


def test_add_renders_model_rows_with_id_text_done_position(db: sqlite3.Connection, dmn_mp: MP) -> None:
    """On a non-user-broadcast turn the dispatcher drops the card and the body
    head is the model-facing rows: id/text/done/position (content→text,
    checked→done). ``add`` resolves the target by NAME here too."""
    from services.list_service import ListService
    service = cast(ListService, _service())
    list_id = service.create_list("shopping")

    out = ToolDispatcher(dmn_mp).dispatch(
        "list", {"action": "add", "list": "shopping", "items": ["milk"], "act_summary": "x"}
    )

    assert "[list(status=success" in out, out
    body = cast(dict[str, object], _parse_body(out))
    assert body["id"] == list_id, body
    row = next(r for r in cast(list[dict[str, object]], body["items"]) if r["text"] == "milk")
    assert set(row) >= {"id", "text", "done", "position"}, row
    assert row["done"] is False, row


def test_add_by_partial_name_substring_resolves(db: sqlite3.Connection, chat_mp: MP) -> None:
    """A single substring match resolves: ``list="grocer"`` → "groceries"."""
    from services.list_service import ListService
    service = cast(ListService, _service())
    list_id = service.create_list("groceries")

    out = ToolDispatcher(chat_mp).dispatch(
        "list", {"action": "add", "list": "grocer", "items": ["eggs"], "act_summary": "x"}
    )

    assert "[list(status=success" in out, out
    assert cast(dict[str, object], _parse_body(out))["id"] == list_id
    assert "eggs" in [i["content"] for i in cast(list[dict[str, object]], cast(dict[str, object], service.get_list(list_id))["items"])]


def test_add_ambiguous_name_lists_candidates_and_mutates_nothing(db: sqlite3.Connection, chat_mp: MP) -> None:
    """Two lists both matching "shopp*"; ``add list="shopp"`` must return
    ``code=ambiguous-match`` with BOTH candidate ids+names, and add the item to
    NEITHER list. This is the consent-gating regression."""
    from services.list_service import ListService
    service = cast(ListService, _service())
    id_a = service.create_list("shopping")
    id_b = service.create_list("shopping backup")

    out = ToolDispatcher(chat_mp).dispatch(
        "list", {"action": "add", "list": "shopp", "items": ["milk"], "act_summary": "x"}
    )

    assert "[list(status=error" in out, out
    assert "code=ambiguous-match" in out, out
    for list_id in (id_a, id_b):
        assert list_id in out, out

    # The hard guarantee: NOTHING was added to either list.
    for list_id in (id_a, id_b):
        assert cast(dict[str, object], service.get_list(list_id))["items"] == [], list_id

    # The act-trail recorded the same error envelope against the transcript.
    trail = ActTrail().fetch_by_transcript_id(chat_mp.uid)
    assert any("code=ambiguous-match" in cast(str, row["result"]) for row in trail), trail


def test_unmatched_name_is_list_not_found(db: sqlite3.Connection, chat_mp: MP) -> None:
    """A name that matches no list returns ``code=list-not-found``."""
    out = ToolDispatcher(chat_mp).dispatch(
        "list", {"action": "view", "list": "does-not-exist", "act_summary": "x"}
    )
    assert "[list(status=error" in out, out
    assert "code=list-not-found" in out, out


def test_missing_target_when_no_list_or_id(db: sqlite3.Connection, chat_mp: MP) -> None:
    """``view`` with no list/id returns ``code=missing-target`` (not pre-gated —
    the target is checked ability-side so id-alias calls survive the pre-gate)."""
    out = ToolDispatcher(chat_mp).dispatch(
        "list", {"action": "view", "act_summary": "x"}
    )
    assert "[list(status=error" in out, out
    assert "code=missing-target" in out, out


# ── Automation: item-name disambiguation ───────────────────────────────────────


def test_check_ambiguous_item_lists_candidates_and_checks_nothing(db: sqlite3.Connection, chat_mp: MP) -> None:
    """A list with "whole milk" + "oat milk"; ``check items=["milk"]`` must return
    ``code=ambiguous-match`` naming both contents and check NOTHING."""
    from services.list_service import ListService
    service = cast(ListService, _service())
    list_id = service.create_list("groceries")
    service.add_items(list_id, ["whole milk", "oat milk"])

    out = ToolDispatcher(chat_mp).dispatch(
        "list",
        {"action": "check", "list": "groceries",
         "items": [{"content": "milk", "checked": True}], "act_summary": "x"},
    )

    assert "[list(status=error" in out, out
    assert "code=ambiguous-match" in out, out
    assert "whole milk" in out and "oat milk" in out, out

    # The hard guarantee: nothing was checked.
    for item in cast(list[dict[str, object]], cast(dict[str, object], service.get_list(list_id))["items"]):
        assert not item["checked"], item


def test_check_unmatched_item_is_item_not_found(db: sqlite3.Connection, chat_mp: MP) -> None:
    """An item term matching nothing returns ``code=item-not-found``."""
    from services.list_service import ListService
    service = cast(ListService, _service())
    list_id = service.create_list("groceries")
    service.add_items(list_id, ["bread"])

    out = ToolDispatcher(chat_mp).dispatch(
        "list",
        {"action": "check", "list": "groceries",
         "items": [{"content": "caviar", "checked": True}], "act_summary": "x"},
    )
    assert "[list(status=error" in out, out
    assert "code=item-not-found" in out, out


# ── FE contract: exact-content id-aliased check (the silent-action path) ─────────


def test_exact_item_check_via_id_alias_still_works(db: sqlite3.Connection, chat_mp: MP) -> None:
    """The frontend silent-action POSTs ``{action:'check', id:<id>,
    items:[{content:'whole milk', checked:true}]}`` through the REAL dispatcher.
    The ``id`` alias must resolve and the exact content must check — unchanged
    behaviour the card depends on."""
    from services.list_service import ListService
    service = cast(ListService, _service())
    list_id = service.create_list("groceries")
    service.add_items(list_id, ["whole milk", "oat milk"])

    out = ToolDispatcher(chat_mp).dispatch(
        "list",
        {"action": "check", "id": list_id,
         "items": [{"content": "whole milk", "checked": True}], "act_summary": "x"},
    )

    assert "[list(status=success" in out, out
    by_content = {i["content"]: bool(i["checked"]) for i in cast(list[dict[str, object]], cast(dict[str, object], service.get_list(list_id))["items"])}
    assert by_content["whole milk"] is True, by_content
    assert by_content["oat milk"] is False, by_content


def test_empty_list_all_is_success_count_zero(db: sqlite3.Connection, chat_mp: MP) -> None:
    """``list_all`` with no lists is a SUCCESS with ``count=0`` and an empty list —
    not an error, not prose."""
    out = ToolDispatcher(chat_mp).dispatch(
        "list", {"action": "list_all", "act_summary": "x"}
    )
    assert "[list(status=success" in out, out
    assert "count=0" in out, out
    assert _parse_body(out) == []


def test_list_all_renders_summary_rows(db: sqlite3.Connection, chat_mp: MP) -> None:
    """``list_all`` renders JSON rows with id/name/item_count/checked_count."""
    from services.list_service import ListService
    service = cast(ListService, _service())
    a = service.create_list("groceries")
    service.add_items(a, ["milk", "eggs"])
    service.check_items(a, ["milk"])

    out = ToolDispatcher(chat_mp).dispatch(
        "list", {"action": "list_all", "act_summary": "x"}
    )
    assert "[list(status=success" in out, out
    body = cast(list[dict[str, object]], _parse_body(out))
    row = next(r for r in body if r["id"] == a)
    assert row["name"] == "groceries"
    assert row["item_count"] == 2
    assert row["checked_count"] == 1


# ── Rich via contract: the FE card payload shape ────────────────────────────────


def test_create_renders_rich_trailer_with_legacy_fe_payload(db: sqlite3.Connection, chat_mp: MP) -> None:
    """A ``create`` on a user-broadcast turn renders the rich trailer (a
    ``<span id='list_1'>`` instruction) and the card payload in the trailer is the
    LEGACY FE shape (id, name, items with content/checked) the card consumes —
    distinct from the rows the model reads in the body."""
    out = ToolDispatcher(chat_mp).dispatch(
        "list",
        {"action": "create", "name": "shopping", "items": ["milk"], "act_summary": "x"},
    )

    assert "[list(status=success" in out, out
    assert "<span id='list_1'>" in out, out

    # The trailer's card payload is the first blank-line-delimited block of the body.
    head = out.index("]\n") + 2
    tail = out.index("\n[end:list]")
    body = out[head:tail]
    payload_json, instruction = body.split("\n\n", 1)
    payload = json.loads(payload_json)
    assert payload["name"] == "shopping", payload
    assert "id" in payload, payload
    item = payload["items"][0]
    # Legacy FE keys — NOT the model-facing text/done rows.
    assert item["content"] == "milk", item
    assert item["checked"] is False, item
    assert "list_1" in instruction


# ── enrich_rich_payload refresh hook (kept from the old suite, still valid) ──────


def test_enrich_rich_payload_returns_live_state_after_check(db: sqlite3.Connection) -> None:
    """The FE-refresh hook re-fetches live list state so a box ticked via the
    silent-action channel survives a conversation refresh — it operates on the
    legacy FE payload shape (content/checked)."""
    from abilities.list import ListAbility
    from services.list_service import ListService

    service = cast(ListService, _service())
    list_id = service.create_list("groceries")
    service.add_items(list_id, ["milk", "eggs"])
    service.check_items(list_id, ["milk"])

    stale: dict[str, object] = {
        "id": list_id,
        "name": "groceries",
        "items": [
            {"content": "milk", "checked": False},
            {"content": "eggs", "checked": False},
        ],
    }
    enriched = ListAbility.enrich_rich_payload(stale, row={})
    by_content = {i["content"]: i["checked"] for i in cast(list[dict[str, object]], enriched["items"])}
    assert by_content == {"milk": True, "eggs": False}
