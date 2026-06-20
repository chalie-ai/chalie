# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Place-ability-specific business-logic tests migrated from the per-ability
conformance file removed in TKT-975.

Pins TKT-885's ToolResult contract: every action's happy path renders a
non-empty parseable success envelope, and every failure carries a stable
kebab-case code (NOT ``code="error"``). GPS is driven via real telemetry rows.
"""

import json
import sqlite3
from typing import cast

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from tests._tool_result_harness import MP, parse_body, seed_transcript

pytestmark = pytest.mark.unit


# ── GPS telemetry fixtures ───────────────────────────────────────────────────────


def _seed_gps(db: sqlite3.Connection, *, lat: float, lon: float, location_name: str) -> None:
    from services.heartbeat_service import heartbeat_service

    heartbeat_service._ctx = None
    db.execute("DELETE FROM telemetry")
    for key, value in {
        "location.lat": lat,
        "location.lon": lon,
        "location_name": location_name,
    }.items():
        db.execute(
            "INSERT INTO telemetry (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
    db.commit()
    heartbeat_service._ctx = None


def _clear_gps(db: sqlite3.Connection) -> None:
    from services.heartbeat_service import heartbeat_service

    heartbeat_service._ctx = None
    db.execute("DELETE FROM telemetry")
    db.commit()
    heartbeat_service._ctx = None


@pytest.fixture
def chat_mp(db: sqlite3.Connection) -> MP:
    return MP(seed_transcript(db, content="save this place"), UserConfig({}))


def _parse_body(rendered: str, tool: str = "place") -> object:
    return parse_body(rendered, tool)


# ── Happy paths: every action renders a non-empty, parseable success body ──────


def test_save_renders_structured_body_with_coords(db: sqlite3.Connection, chat_mp: MP) -> None:
    _seed_gps(db, lat=35.899, lon=14.514, location_name="Valletta, Malta")

    out = ToolDispatcher(chat_mp).dispatch(
        "place", {"action": "save", "name": "Home", "act_summary": "x"}
    )

    assert "[place(status=success" in out
    assert "[end:place]" in out
    body = cast(dict[str, object], _parse_body(out))
    assert body["name"] == "home"
    assert body["lat"] == 35.899
    assert body["lon"] == 14.514
    assert body["location_name"] == "Valletta, Malta"
    assert body["source"] == "place_ability"

    # Downstream: the row really landed in the data graph under kind='place'.
    from services.data_graph_service import KIND_PLACE, get_data_graph_service
    rows = get_data_graph_service().fetch(kinds=[KIND_PLACE])
    assert any(cast(dict[str, object], r).get("key") == "home" for r in rows)

    # The act-trail recorded the same non-empty envelope against the transcript.
    from services.act_trail import ActTrail
    trail = ActTrail().fetch_by_transcript_id(chat_mp.uid)
    assert "[place(status=success" in cast(str, trail[0]["result"])


def test_list_renders_count_meta_and_place_array(db: sqlite3.Connection, chat_mp: MP) -> None:
    _seed_gps(db, lat=10.0, lon=20.0, location_name="Alpha City")
    ToolDispatcher(chat_mp).dispatch(
        "place", {"action": "save", "name": "Office", "act_summary": "x"}
    )
    _seed_gps(db, lat=30.0, lon=40.0, location_name="Beta City")
    ToolDispatcher(chat_mp).dispatch(
        "place", {"action": "save", "name": "Cafe", "act_summary": "x"}
    )

    out = ToolDispatcher(chat_mp).dispatch(
        "place", {"action": "list", "act_summary": "x"}
    )

    assert "[place(status=success, count=2)]" in out
    body = _parse_body(out)
    assert isinstance(body, list)
    names = {cast(dict[str, object], p)["name"] for p in body}
    assert names == {"office", "cafe"}
    assert {cast(dict[str, object], p)["lat"] for p in body} == {10.0, 30.0}


def test_get_renders_the_single_place_record(db: sqlite3.Connection, chat_mp: MP) -> None:
    _seed_gps(db, lat=51.5, lon=-0.12, location_name="London, UK")
    ToolDispatcher(chat_mp).dispatch(
        "place", {"action": "save", "name": "Work", "act_summary": "x"}
    )

    out = ToolDispatcher(chat_mp).dispatch(
        "place", {"action": "get", "name": "work", "act_summary": "x"}
    )

    assert "[place(status=success" in out
    body = cast(dict[str, object], _parse_body(out))
    assert body["name"] == "work"
    assert body["lat"] == 51.5
    assert body["lon"] == -0.12
    assert body["location_name"] == "London, UK"


def test_delete_renders_confirmation_and_removes_row(db: sqlite3.Connection, chat_mp: MP) -> None:
    _seed_gps(db, lat=1.0, lon=2.0, location_name="Gym Town")
    ToolDispatcher(chat_mp).dispatch(
        "place", {"action": "save", "name": "Gym", "act_summary": "x"}
    )

    out = ToolDispatcher(chat_mp).dispatch(
        "place", {"action": "delete", "name": "gym", "act_summary": "x"}
    )

    assert "[place(status=success" in out
    body = cast(dict[str, object], _parse_body(out))
    assert body["name"] == "gym"
    assert body["deleted"] is True

    # Downstream: the row is really gone from the data graph.
    from services.data_graph_service import KIND_PLACE, get_data_graph_service
    rows = get_data_graph_service().fetch(kinds=[KIND_PLACE])
    assert not any(cast(dict[str, object], r).get("key") == "gym" for r in rows)

    follow = ToolDispatcher(chat_mp).dispatch(
        "place", {"action": "get", "name": "gym", "act_summary": "x"}
    )
    assert "[place(status=error, code=not-found" in follow


# ── Error paths: stable kebab-case codes, NOT the code="error" placeholder ─────


def test_save_without_gps_errors_with_no_location_code(db: sqlite3.Connection, chat_mp: MP) -> None:
    """``save`` with no client GPS errors with a STABLE ``code=no-location`` (NOT
    the ``code=error`` placeholder) and a one-line ``hint:`` recovery step."""
    _clear_gps(db)

    out = ToolDispatcher(chat_mp).dispatch(
        "place", {"action": "save", "name": "Home", "act_summary": "x"}
    )

    assert "[place(status=error, code=no-location" in out
    assert "code=error]" not in out
    assert "hint:" in out
    assert "[end:place]" in out


def test_get_missing_place_errors_with_not_found_code(db: sqlite3.Connection, chat_mp: MP) -> None:
    """``get`` for a name that was never saved errors with ``code=not-found`` and a
    ``hint:`` pointing the model at ``list``."""
    _clear_gps(db)

    out = ToolDispatcher(chat_mp).dispatch(
        "place", {"action": "get", "name": "nowhere", "act_summary": "x"}
    )

    assert "[place(status=error, code=not-found" in out
    assert "code=error]" not in out
    assert "hint:" in out
