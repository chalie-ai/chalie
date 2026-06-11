# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the review-pair ToolResult contract (TKT-918).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch(...)`` chokepoint on the CHAT channel against a real
``mp``-shaped context, the real ``AbilityRegistry`` resolution of the production
``ReviewToolCallsAbility`` / ``ReviewTranscriptAbility``, the real
``PolicyManager.wrap`` gate (both names are in ``PolicyManager.INTERNAL`` → the
gate short-circuits straight through to ``run()``), the real shared base
``ReviewWindowAbility`` driving a REAL windowed SQLite query against the test
database, and the real ``ActTrail`` write.

TKT-918 extracts the two near-twin review tools' shared retrieval/formatting into
one base and migrates both to ToolResult: rows as compact-JSON lists with a
1-based ``iter`` ordinal, an empty window as ``count=0`` SUCCESS with a prose
hint (never an error), a LOUD narrow ``query-failed`` on a corrupt read, and
``code="error"`` gone. Seeds carry KNOWN ``created_at`` ISO timestamps so the
window arithmetic is exercised for real, fully offline.
"""

import json

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig

pytestmark = pytest.mark.unit


# ── Fixtures / helpers ──────────────────────────────────────────────────────────

# A fixed anchor and ISO timestamps around it; the base opens ±buffer minutes.
_ANCHOR = "2026-04-07T14:30:00+00:00"
_INSIDE_A = "2026-04-07T14:28:00+00:00"   # -2 min
_INSIDE_B = "2026-04-07T14:31:00+00:00"   # +1 min
_OUTSIDE = "2026-04-07T14:50:00+00:00"    # +20 min (outside ±5)
_PLUS_20 = "2026-04-07T14:50:00+00:00"    # +20 min
_PLUS_40 = "2026-04-07T15:10:00+00:00"    # +40 min


def _seed_transcript_anchor(db) -> int:
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        ("chat", "user", "review the history"),
    )
    db.commit()
    return cur.lastrowid


def _seed_tool_call(db, *, transcript_id, tool_name, params, result, created_at, ephemeral=0):
    db.execute(
        "INSERT INTO tool_calls "
        "(transcript_id, tool_name, params, result, ephemeral, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (transcript_id, tool_name, params, result, ephemeral, created_at),
    )
    db.commit()


def _seed_transcript_row(db, *, channel, role, content, created_at):
    db.execute(
        "INSERT INTO transcript (channel, role, content, created_at) "
        "VALUES (?, ?, ?, ?)",
        (channel, role, content, created_at),
    )
    db.commit()


class _MP:
    """Minimal real MP-shaped context — exactly what dispatch reads off the live
    processor: ``config`` (the chat policy channel + user broadcast) and ``uid``
    (the transcript anchor the act-trail records against)."""

    def __init__(self, uid: int, config) -> None:
        self.config = config
        self.uid = uid


@pytest.fixture
def chat_mp(db):
    """A real chat-channel mp bound to the test database. Both review tools are in
    ``PolicyManager.INTERNAL`` so the gate short-circuits straight to run()."""
    return _MP(_seed_transcript_anchor(db), UserConfig({}))


def _head(rendered: str, tool: str) -> str:
    line = rendered.splitlines()[0]
    assert line.startswith(f"[{tool}(")
    return line


def _body(rendered: str, tool: str) -> str:
    head = rendered.index("]\n") + 2
    tail = rendered.index(f"\n[end:{tool}]")
    return rendered[head:tail]


def _body_json(rendered: str, tool: str):
    return json.loads(_body(rendered, tool))


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


# ── 4. review_tool_calls missing date_time → pre-gate missing-params ────────────


def test_review_tool_calls_missing_date_time_pre_gate(db, chat_mp):
    out = ToolDispatcher(chat_mp).dispatch(
        "review_tool_calls", {"act_summary": "x"}
    )

    head = _head(out, "review_tool_calls")
    assert "status=error" in head
    assert "code=missing-params" in head
    assert "code=error" not in out
    assert "[end:review_tool_calls]" in out


# ── 5. review_tool_calls garbage date_time → invalid-time with ISO hint ─────────


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


# ── 6. review_transcript happy path: rows with channel/role/ts, newline flat ────


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


# ── 7. review_transcript subagent toggle ────────────────────────────────────────


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


# ── 8. review_transcript buffer clamp: widen, and 999 clamps to 30 ──────────────


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


# ── 9. review_transcript invalid buffer → invalid-param, not unhandled-exception ─


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


# ── 10. query-failed is LOUD and narrow: drop the table, dispatch → query-failed ─


def test_review_tool_calls_query_failed_is_loud(db, chat_mp):
    """A corrupt read must NOT masquerade as an empty window (the bug the deleted
    get_by_timerange swallowed). Drop the real table → real sqlite3.OperationalError
    → narrow query-failed envelope, zero mocks (TKT-911 corrupt-index precedent)."""
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


# ── 11. legacy-marker sweep across every dispatch above ─────────────────────────


def test_no_legacy_error_markers(db, chat_mp):
    """No dispatch in this pair emits the banned ``code=error`` head, and the
    structured success bodies are well-formed JSON lists — proving the migration
    off prose-with-code=error is complete."""
    _seed_tool_call(
        db, transcript_id=chat_mp.uid, tool_name="search", params='{"q":"x"}',
        result="[search(status=success)]\nok\n[end:search]",
        created_at=_INSIDE_A,
    )
    _seed_transcript_row(
        db, channel="user", role="user", content="hi", created_at=_INSIDE_A,
    )

    for tool in ("review_tool_calls", "review_transcript"):
        out = ToolDispatcher(chat_mp).dispatch(
            tool, {"date_time": _ANCHOR, "act_summary": "x"}
        )
        assert "code=error" not in out
        assert isinstance(_body_json(out, tool), list)
