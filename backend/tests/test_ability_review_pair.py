# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""review_tool_calls / review_transcript specific business-logic tests migrated
from the per-ability conformance file removed in TKT-975. Covers windowed row
retrieval, iter ordinals, empty-window hints, invalid-time errors, subagent
toggle, buffer clamping, invalid-param handling, and the query-failed loud path.
"""

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from tests._tool_result_harness import (
    MP,
    body,
    head,
    parse_body,
    seed_transcript,
)

pytestmark = pytest.mark.unit


# ── Fixtures / helpers ──────────────────────────────────────────────────────────

# A fixed anchor and ISO timestamps around it; the base opens ±buffer minutes.
_ANCHOR = "2026-04-07T14:30:00+00:00"
_INSIDE_A = "2026-04-07T14:28:00+00:00"   # -2 min
_INSIDE_B = "2026-04-07T14:31:00+00:00"   # +1 min
_OUTSIDE = "2026-04-07T14:50:00+00:00"    # +20 min (outside ±5)
_PLUS_20 = "2026-04-07T14:50:00+00:00"    # +20 min
_PLUS_40 = "2026-04-07T15:10:00+00:00"    # +40 min


def _seed_tool_call(db, *, transcript_id, tool_name, params, result, created_at):
    db.execute(
        "INSERT INTO tool_calls "
        "(transcript_id, tool_name, params, result, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (transcript_id, tool_name, params, result, created_at),
    )
    db.commit()


def _seed_transcript_row(db, *, channel, role, content, created_at):
    db.execute(
        "INSERT INTO transcript (channel, role, content, created_at) "
        "VALUES (?, ?, ?, ?)",
        (channel, role, content, created_at),
    )
    db.commit()


@pytest.fixture
def chat_mp(db):
    """Both review tools are in ``PolicyManager.INTERNAL`` so the gate short-circuits straight to run()."""
    return MP(seed_transcript(db, content="review the history"), UserConfig({}))


def _head(rendered: str, tool: str) -> str:
    return head(rendered, tool)


def _body(rendered: str, tool: str) -> str:
    return body(rendered, tool)


def _body_json(rendered: str, tool: str):
    return parse_body(rendered, tool)


# ── 1. review_tool_calls happy path: windowed rows, iter ordinals, ok flag ──────


def test_review_tool_calls_happy_path(db, chat_mp):
    tid = chat_mp.uid
    _seed_tool_call(
        db, transcript_id=tid, tool_name="search",
        params='{"query":"laptops"}',
        result="[search(status=success)]\nfound 3\n[end:search]",
        created_at=_INSIDE_A,
    )
    _seed_tool_call(
        db, transcript_id=tid, tool_name="weather",
        params='{"city":"London"}',
        result="[weather(status=error, code=missing-location)]\noops\n[end:weather]",
        created_at=_INSIDE_B,
    )
    _seed_tool_call(
        db, transcript_id=tid, tool_name="news",
        params='{"topic":"sports"}',
        result="[news(status=success)]\nirrelevant\n[end:news]",
        created_at=_OUTSIDE,
    )

    out = ToolDispatcher(chat_mp).dispatch(
        "review_tool_calls", {"date_time": _ANCHOR, "act_summary": "x"}
    )

    head = _head(out, "review_tool_calls")
    assert "status=success" in head
    assert "count=2" in head
    assert "code=error" not in out

    rows = _body_json(out, "review_tool_calls")
    assert isinstance(rows, list) and len(rows) == 2
    # ordered oldest first → search (-2 min) then weather (+1 min)
    assert [r["iter"] for r in rows] == [1, 2]
    assert [r["tool"] for r in rows] == ["search", "weather"]
    # envelope-aware ok: success envelope → True, error envelope → False
    assert rows[0]["ok"] is True
    assert rows[1]["ok"] is False
    # the +20 min row is outside ±5 and absent
    assert all(r["tool"] != "news" for r in rows)


# ── 2. review_tool_calls params_summary clipped for a long params string ────────


def test_review_tool_calls_params_summary_clipped(db, chat_mp):
    long_params = '{"q":"' + ("z" * 400) + '"}'
    _seed_tool_call(
        db, transcript_id=chat_mp.uid, tool_name="search", params=long_params,
        result="[search(status=success)]\nok\n[end:search]",
        created_at=_INSIDE_A,
    )

    out = ToolDispatcher(chat_mp).dispatch(
        "review_tool_calls", {"date_time": _ANCHOR, "act_summary": "x"}
    )

    rows = _body_json(out, "review_tool_calls")
    assert len(rows) == 1
    assert len(rows[0]["params_summary"]) <= 120


# ── 3. review_tool_calls empty window → success count=0 with a prose hint ────────


def test_review_tool_calls_empty_window_is_success_hint(db, chat_mp):
    out = ToolDispatcher(chat_mp).dispatch(
        "review_tool_calls", {"date_time": _ANCHOR, "act_summary": "x"}
    )

    head = _head(out, "review_tool_calls")
    assert "status=success" in head
    assert "count=0" in head
    assert "code=error" not in out

    body = _body(out, "review_tool_calls")
    # prose guidance, NOT an empty JSON list
    assert body.strip() != "[]"
    assert "No tool calls found" in body
    assert "timestamp" in body


# ── 4. review_tool_calls garbage date_time → invalid-time with ISO hint ─────────


def test_review_tool_calls_garbage_date_time_invalid_time(db, chat_mp):
    out = ToolDispatcher(chat_mp).dispatch(
        "review_tool_calls", {"date_time": "not a time", "act_summary": "x"}
    )

    head = _head(out, "review_tool_calls")
    assert "status=error" in head
    assert "code=invalid-time" in head
    assert "code=error" not in out
    hint_lines = [ln for ln in out.splitlines() if ln.startswith("hint:")]
    assert hint_lines and "2026-04-07T14:30:00+00:00" in hint_lines[0]


# ── 5. review_transcript happy path: rows with channel/role/ts, newline flat ────


def test_review_transcript_happy_path(db, chat_mp):
    _seed_transcript_row(
        db, channel="user", role="user",
        content="line one\nline two", created_at=_INSIDE_A,
    )
    _seed_transcript_row(
        db, channel="user", role="assistant",
        content="the draft", created_at=_INSIDE_B,
    )

    out = ToolDispatcher(chat_mp).dispatch(
        "review_transcript", {"date_time": _ANCHOR, "act_summary": "x"}
    )

    head = _head(out, "review_transcript")
    assert "status=success" in head
    assert "count=2" in head
    assert "code=error" not in out

    rows = _body_json(out, "review_transcript")
    assert [r["iter"] for r in rows] == [1, 2]
    assert {r["channel"] for r in rows} == {"user"}
    assert [r["role"] for r in rows] == ["user", "assistant"]
    # newline flattened to a single line
    assert rows[0]["content"] == "line one line two"
    assert "\n" not in rows[0]["content"]


# ── 6. review_transcript subagent toggle ────────────────────────────────────────


def test_review_transcript_subagent_toggle(db, chat_mp):
    _seed_transcript_row(
        db, channel="user", role="user",
        content="user said", created_at=_INSIDE_A,
    )
    _seed_transcript_row(
        db, channel="subagent", role="assistant",
        content="subagent found", created_at=_INSIDE_B,
    )

    # default: subagent row excluded
    default_out = ToolDispatcher(chat_mp).dispatch(
        "review_transcript", {"date_time": _ANCHOR, "act_summary": "x"}
    )
    default_rows = _body_json(default_out, "review_transcript")
    assert all(r["channel"] == "user" for r in default_rows)
    assert len(default_rows) == 1

    # toggle on: subagent row included
    sub_out = ToolDispatcher(chat_mp).dispatch(
        "review_transcript",
        {"date_time": _ANCHOR, "include_subagent_transcripts": True, "act_summary": "x"},
    )
    sub_rows = _body_json(sub_out, "review_transcript")
    assert {r["channel"] for r in sub_rows} == {"user", "subagent"}
    assert len(sub_rows) == 2


# ── 7. review_transcript buffer clamp: widen, and 999 clamps to 30 ──────────────


def test_review_transcript_buffer_clamp(db, chat_mp):
    _seed_transcript_row(
        db, channel="user", role="user",
        content="twenty minutes later", created_at=_PLUS_20,
    )
    _seed_transcript_row(
        db, channel="user", role="user",
        content="forty minutes later", created_at=_PLUS_40,
    )

    # default ±5 → +20 min row absent
    default_out = ToolDispatcher(chat_mp).dispatch(
        "review_transcript", {"date_time": _ANCHOR, "act_summary": "x"}
    )
    assert "count=0" in _head(default_out, "review_transcript")

    # ±25 → +20 min row present
    wide_out = ToolDispatcher(chat_mp).dispatch(
        "review_transcript",
        {"date_time": _ANCHOR, "buffer_minutes": 25, "act_summary": "x"},
    )
    wide_rows = _body_json(wide_out, "review_transcript")
    assert len(wide_rows) == 1
    assert wide_rows[0]["content"] == "twenty minutes later"

    # buffer_minutes=999 clamps to 30 → +20 present, +40 still absent
    clamp_out = ToolDispatcher(chat_mp).dispatch(
        "review_transcript",
        {"date_time": _ANCHOR, "buffer_minutes": 999, "act_summary": "x"},
    )
    head = _head(clamp_out, "review_transcript")
    assert "buffer=30" in head
    clamp_rows = _body_json(clamp_out, "review_transcript")
    assert [r["content"] for r in clamp_rows] == ["twenty minutes later"]


# ── 8. review_transcript invalid buffer → invalid-param, not unhandled-exception ─


def test_review_transcript_invalid_buffer_is_invalid_param(db, chat_mp):
    out = ToolDispatcher(chat_mp).dispatch(
        "review_transcript",
        {"date_time": _ANCHOR, "buffer_minutes": "lots", "act_summary": "x"},
    )

    head = _head(out, "review_transcript")
    assert "status=error" in head
    assert "code=invalid-param" in head
    assert "code=unhandled-exception" not in out
    assert "code=error" not in out
    assert any(ln.startswith("hint:") for ln in out.splitlines())


# ── 9. query-failed is LOUD and narrow: drop the table, dispatch → query-failed ─


def test_review_tool_calls_query_failed_is_loud(db, chat_mp):
    """A corrupt read must NOT masquerade as an empty window (the bug the deleted get_by_timerange swallowed)."""
    db.execute("DROP TABLE tool_calls")
    db.commit()

    out = ToolDispatcher(chat_mp).dispatch(
        "review_tool_calls", {"date_time": _ANCHOR, "act_summary": "x"}
    )

    head = _head(out, "review_tool_calls")
    assert "status=error" in head
    assert "code=query-failed" in head
    assert "code=error" not in out
    assert any(ln.startswith("hint:") for ln in out.splitlines())
