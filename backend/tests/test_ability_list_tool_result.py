# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the list tool's ToolResult contract + name addressing (TKT-902).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch()`` chokepoint on the CHAT channel against a real
``mp``-shaped context, the real ``AbilityRegistry`` resolution of the production
``ListAbility``, the real ``PolicyManager.wrap`` gate (reading the real ``policy``
table the ``db`` fixture binds), and the real ``ListService`` (the ``db`` fixture
binds the singleton to a real SQLite database). The act-trail write is real too.

The automation regression this ticket fixes: id-only addressing forced a
list-then-act round trip. ``add item="milk" list="shopping"`` must now resolve the
target by NAME. When a partial name matches more than one list, it must return
``code=ambiguous-match`` listing every candidate (id + name) and mutate NOTHING —
never silently pick one. Item names resolve the same consent-safe way against the
fetched list contents.

The contract: every action returns a ``ToolResult`` rendered by the dispatcher;
list-returning actions emit JSON rows (id, text, done, position); errors carry a
stable kebab-case ``code`` (never ``code="error"``). The rich card path is via
``ToolResult(rich=…)`` carrying the LEGACY FE payload shape (id, name, items with
content/checked) the frontend card and ``enrich_rich_payload`` depend on.
"""

import json

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from services.act_trail import ActTrail

pytestmark = pytest.mark.unit


def _seed_transcript(db, channel: str) -> int:
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        (channel, "user", "manage my lists"),
    )
    db.commit()
    return cur.lastrowid


def _allow_delete(db, channel: str = "chat") -> None:
    """Flip the REAL ``policy`` table so ``list.delete`` is ``allow`` on the
    channel — the same row the real ``PolicyManager`` gate reads. ``delete`` ships
    as ``ask`` by seed; on a headless test channel an ``ask`` would block waiting
    for a human POST. Flipping the real row to ``allow`` (exactly what a user does
    when they pick "always allow") lets the gate pass through to the production
    ``run()``. No mock — this is the production policy store."""
    db.execute(
        "INSERT OR REPLACE INTO policy (channel, permission, setting) "
        "VALUES (?, ?, 'allow')",
        (channel, "list.delete"),
    )
    db.commit()


class _MP:
    """Minimal real MP-shaped context — exactly what dispatch reads off the live
    processor: ``config`` (the chat policy channel) and ``uid`` (the transcript
    anchor the trail records against)."""

    def __init__(self, uid: int, config) -> None:
        self.config = config
        self.uid = uid


@pytest.fixture
def chat_mp(db):
    """A real chat-channel mp bound to the test database, with a seeded transcript
    anchor for the act-trail write and ``list.delete`` flipped to ``allow`` in the
    real policy table so the destructive gate passes through to run()."""
    _allow_delete(db)
    return _MP(_seed_transcript(db, "chat"), UserConfig({}))


@pytest.fixture
def dmn_mp(db):
    """A real non-user-broadcast mp (``broadcast_to=None``). On this channel the
    dispatcher drops the rich card, so the rendered body head is the raw
    model-facing rows (id/text/done/position) — the shape the model reads —
    instead of the frontend card payload. The subconscious policy channel ships
    list.add/list_all as ``allow``."""
    from configs.channels import DmnConfig
    return _MP(_seed_transcript(db, "subconscious"), DmnConfig())


def _service():
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
    <instruction>``; this returns the card payload (the JSON head before the
    blank line)."""
    head = rendered.index("]\n") + 2
    tail = rendered.index(f"\n[end:{tool}]")
    body = rendered[head:tail]
    if "\n\n" in body:
        body = body.split("\n\n", 1)[0]
    return json.loads(body)


# ── Automation: name addressing ────────────────────────────────────────────────


def test_add_by_exact_name_resolves_target(db, chat_mp):
    """``add list="shopping" items=["milk"]`` resolves the target by NAME — no
    list-then-act round trip — and the item lands in the service-backed list."""
    service = _service()
    list_id = service.create_list("shopping")

    out = ToolDispatcher(chat_mp).dispatch(
        "list", {"action": "add", "list": "shopping", "items": ["milk"], "act_summary": "x"}
    )

    assert "[list(status=success" in out, out
    # On a user turn the body head is the FE card; it still carries the resolved id.
    card = _parse_body(out)
    assert card["id"] == list_id, card
    assert card["name"] == "shopping", card

    # Downstream: the item really landed in the service-backed list.
    contents = [i["content"] for i in service.get_list(list_id)["items"]]
    assert "milk" in contents


def test_add_renders_model_rows_with_id_text_done_position(db, dmn_mp):
    """On a non-user-broadcast turn the dispatcher drops the card and the body
    head is the model-facing rows: id/text/done/position (content→text,
    checked→done). ``add`` resolves the target by NAME here too."""
    service = _service()
    list_id = service.create_list("shopping")

    out = ToolDispatcher(dmn_mp).dispatch(
        "list", {"action": "add", "list": "shopping", "items": ["milk"], "act_summary": "x"}
    )

    assert "[list(status=success" in out, out
    body = _parse_body(out)
    assert body["id"] == list_id, body
    row = next(r for r in body["items"] if r["text"] == "milk")
    assert set(row) >= {"id", "text", "done", "position"}, row
    assert row["done"] is False, row


def test_add_by_partial_name_substring_resolves(db, chat_mp):
    """A single substring match resolves: ``list="grocer"`` → "groceries"."""
    service = _service()
    list_id = service.create_list("groceries")

    out = ToolDispatcher(chat_mp).dispatch(
        "list", {"action": "add", "list": "grocer", "items": ["eggs"], "act_summary": "x"}
    )

    assert "[list(status=success" in out, out
    assert _parse_body(out)["id"] == list_id
    assert "eggs" in [i["content"] for i in service.get_list(list_id)["items"]]


def test_add_ambiguous_name_lists_candidates_and_mutates_nothing(db, chat_mp):
    """Two lists both matching "shopp*"; ``add list="shopp"`` must return
    ``code=ambiguous-match`` with BOTH candidate ids+names, and add the item to
    NEITHER list. This is the consent-gating regression."""
    service = _service()
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
        assert service.get_list(list_id)["items"] == [], list_id

    # The act-trail recorded the same error envelope against the transcript.
    trail = ActTrail().fetch_by_transcript_id(chat_mp.uid)
    assert any("code=ambiguous-match" in row["result"] for row in trail), trail


def test_unmatched_name_is_list_not_found(db, chat_mp):
    """A name that matches no list returns ``code=list-not-found``."""
    out = ToolDispatcher(chat_mp).dispatch(
        "list", {"action": "view", "list": "does-not-exist", "act_summary": "x"}
    )
    assert "[list(status=error" in out, out
    assert "code=list-not-found" in out, out


def test_missing_target_when_no_list_or_id(db, chat_mp):
    """``view`` with no list/id returns ``code=missing-target`` (not pre-gated —
    the target is checked ability-side so id-alias calls survive the pre-gate)."""
    out = ToolDispatcher(chat_mp).dispatch(
        "list", {"action": "view", "act_summary": "x"}
    )
    assert "[list(status=error" in out, out
    assert "code=missing-target" in out, out


# ── Automation: item-name disambiguation ───────────────────────────────────────


def test_check_ambiguous_item_lists_candidates_and_checks_nothing(db, chat_mp):
    """A list with "whole milk" + "oat milk"; ``check items=["milk"]`` must return
    ``code=ambiguous-match`` naming both contents and check NOTHING."""
    service = _service()
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
    for item in service.get_list(list_id)["items"]:
        assert not item["checked"], item


def test_check_unmatched_item_is_item_not_found(db, chat_mp):
    """An item term matching nothing returns ``code=item-not-found``."""
    service = _service()
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


def test_exact_item_check_via_id_alias_still_works(db, chat_mp):
    """The frontend silent-action POSTs ``{action:'check', id:<id>,
    items:[{content:'whole milk', checked:true}]}`` through the REAL dispatcher.
    The ``id`` alias must resolve and the exact content must check — unchanged
    behaviour the card depends on."""
    service = _service()
    list_id = service.create_list("groceries")
    service.add_items(list_id, ["whole milk", "oat milk"])

    out = ToolDispatcher(chat_mp).dispatch(
        "list",
        {"action": "check", "id": list_id,
         "items": [{"content": "whole milk", "checked": True}], "act_summary": "x"},
    )

    assert "[list(status=success" in out, out
    by_content = {i["content"]: bool(i["checked"]) for i in service.get_list(list_id)["items"]}
    assert by_content["whole milk"] is True, by_content
    assert by_content["oat milk"] is False, by_content


# ── Contract: errors are errors, not status=success wrapping a fail string ───────


def test_unknown_id_is_error_list_not_found(db, chat_mp):
    """An unknown id returns ``status=error`` + ``code=list-not-found`` — the old
    code wrapped a json ``"status":"fail"`` body inside a status=success envelope."""
    out = ToolDispatcher(chat_mp).dispatch(
        "list", {"action": "view", "id": "deadbeef", "act_summary": "x"}
    )
    assert "[list(status=error" in out, out
    assert "code=list-not-found" in out, out


def test_unknown_action_renders_valid_ladder(db, chat_mp):
    """An unknown action returns ``code=unknown-action`` with the 9-action ladder."""
    out = ToolDispatcher(chat_mp).dispatch(
        "list", {"action": "frobnicate", "act_summary": "x"}
    )
    assert "[list(status=error" in out, out
    assert "code=unknown-action" in out, out
    assert "valid:" in out, out
    for action in ("create", "view", "add", "check", "remove", "clear", "rename", "delete"):
        assert action in out, out


def test_empty_list_all_is_success_count_zero(db, chat_mp):
    """``list_all`` with no lists is a SUCCESS with ``count=0`` and an empty list —
    not an error, not prose."""
    out = ToolDispatcher(chat_mp).dispatch(
        "list", {"action": "list_all", "act_summary": "x"}
    )
    assert "[list(status=success" in out, out
    assert "count=0" in out, out
    assert _parse_body(out) == []


def test_list_all_renders_summary_rows(db, chat_mp):
    """``list_all`` renders JSON rows with id/name/item_count/checked_count."""
    service = _service()
    a = service.create_list("groceries")
    service.add_items(a, ["milk", "eggs"])
    service.check_items(a, ["milk"])

    out = ToolDispatcher(chat_mp).dispatch(
        "list", {"action": "list_all", "act_summary": "x"}
    )
    assert "[list(status=success" in out, out
    body = _parse_body(out)
    row = next(r for r in body if r["id"] == a)
    assert row["name"] == "groceries"
    assert row["item_count"] == 2
    assert row["checked_count"] == 1


# ── Rich via contract: the FE card payload shape ────────────────────────────────


def test_create_renders_rich_trailer_with_legacy_fe_payload(db, chat_mp):
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


def test_enrich_rich_payload_returns_live_state_after_check(db):
    """The FE-refresh hook re-fetches live list state so a box ticked via the
    silent-action channel survives a conversation refresh — it operates on the
    legacy FE payload shape (content/checked)."""
    from abilities.list import ListAbility

    service = _service()
    list_id = service.create_list("groceries")
    service.add_items(list_id, ["milk", "eggs"])
    service.check_items(list_id, ["milk"])

    stale = {
        "id": list_id,
        "name": "groceries",
        "items": [
            {"content": "milk", "checked": False},
            {"content": "eggs", "checked": False},
        ],
    }
    enriched = ListAbility.enrich_rich_payload(stale, row={})
    by_content = {i["content"]: i["checked"] for i in enriched["items"]}
    assert by_content == {"milk": True, "eggs": False}
