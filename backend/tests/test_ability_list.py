"""Feature tests for ListAbility.enrich_rich_payload.

The list rich-media card has a refresh hazard: the canonical list state lives
in the ``lists`` / ``list_items`` tables (mutated by both the LLM-channel
``check`` action and the silent-action endpoint hit by FE checkboxes), while
the rendered card is reconstructed from ``tool_calls.result`` — a snapshot
frozen at LLM-call time. Without an enrichment hook, refresh would replay the
stale snapshot and visually un-tick boxes the user already ticked.

These tests verify ``ListAbility.enrich_rich_payload`` re-fetches the live
list via ``ListService`` so both render paths converge on the same data.
"""

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def service(db):
    from services.database_service import get_shared_db_service
    from services.list_service import ListService
    return ListService(get_shared_db_service())


def test_enrich_rich_payload_returns_live_state_after_check(service):
    """Snapshot says unchecked; live ListService says checked. Enrich must
    return the live state — that is the whole point of the hook."""
    from abilities.list import ListAbility

    service.create_list("Groceries")
    service.add_items("Groceries", ["milk", "eggs"])
    service.check_items("Groceries", ["milk"])

    stale_snapshot = {
        "name": "Groceries",
        "items": [
            {"content": "milk", "checked": False},
            {"content": "eggs", "checked": False},
        ],
    }
    enriched = ListAbility.enrich_rich_payload(stale_snapshot, row={})

    by_content = {item["content"]: item["checked"] for item in enriched["items"]}
    assert by_content == {"milk": True, "eggs": False}


def test_enrich_rich_payload_falls_back_when_list_missing(service):
    """If the named list has been deleted between snapshot and refresh, fall
    back to the snapshot rather than dropping the card from the conversation."""
    from abilities.list import ListAbility

    snapshot = {
        "name": "Deleted List",
        "items": [{"content": "item", "checked": False}],
    }
    enriched = ListAbility.enrich_rich_payload(snapshot, row={})

    assert enriched == snapshot


def test_enrich_rich_payload_passthrough_when_no_name(service):
    """A payload without ``name`` (e.g. malformed older snapshot) is passed
    through unchanged — there is nothing to look up."""
    from abilities.list import ListAbility

    payload = {"items": []}
    assert ListAbility.enrich_rich_payload(payload, row={}) == payload


def test_parser_uses_live_list_for_refresh(service):
    """End-to-end through the parser: a tool_calls row carries a stale snapshot
    string; the live list has additional check-state mutations from the FE
    silent-action channel; the parser must surface the live state."""
    from services.rich_media_parser import parse

    service.create_list("Chores")
    service.add_items("Chores", ["dishes", "laundry"])
    service.check_items("Chores", ["dishes"])

    stale_result = (
        '{"name": "Chores", "items": ['
        '{"content": "dishes", "checked": false}, '
        '{"content": "laundry", "checked": false}]}'
        "\n\n"
        "This tool supports rich-media. <span id='list_1'>x</span>"
    )
    tool_calls = [{
        "tool_name": "list",
        "params": "{}",
        "result": stale_result,
        "ephemeral": 1,
        "created_at": "2026-05-03 14:30:00",
    }]
    segments = parse("<span id='list_1'>Updated.</span>", tool_calls)
    assert len(segments) == 1
    payload = segments[0]["payload"]
    by_content = {item["content"]: item["checked"] for item in payload["items"]}
    assert by_content == {"dishes": True, "laundry": False}


def test_parser_uses_live_list_payload_shape(service):
    """The live re-fetched payload uses the canonical projection
    (``_list_json``): {name, items: [{content, checked}, ...]}."""
    from services.rich_media_parser import parse

    service.create_list("Ideas")
    service.add_items("Ideas", ["alpha", "beta", "gamma"])

    stale_result = (
        '{"name": "Ideas", "items": []}'
        "\n\nThis tool supports rich-media. <span id='list_2'>x</span>"
    )
    tool_calls = [{
        "tool_name": "list",
        "params": "{}",
        "result": stale_result,
        "ephemeral": 1,
        "created_at": "2026-05-03 14:30:00",
    }]
    segments = parse("<span id='list_2'>Saved.</span>", tool_calls)
    payload = segments[0]["payload"]
    assert payload["name"] == "Ideas"
    contents = [it["content"] for it in payload["items"]]
    assert contents == ["alpha", "beta", "gamma"]
    for it in payload["items"]:
        assert it["checked"] is False
