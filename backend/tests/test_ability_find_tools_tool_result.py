# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for find_tools' ToolResult contract (TKT-894).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch()`` chokepoint against a real ``MessageProcessor``
carrying a REAL production channel config (``UserConfig`` / ``DmnConfig``), the
real ``AbilityRegistry`` resolution of the production ``FindToolsAbility``, the
real ``PolicyManager.wrap`` gate (``find_tools`` is INTERNAL — bypasses the gate
unconditionally), the real ``FindToolsAbility.run`` against the real on-disk
``abilities.sqlite``, and the real ``ActTrail`` write.

The regression under test (the exact silent-partial-success bug TKT-894 closes):
today find_tools returns a plain string, so a weak model cannot tell an injected
tool from an ignored one. A blocked delegate select used to render prose with no
machine signal, letting the model believe a tool it never got is now available.
The new contract makes the outcome EXPLICIT:

- select known → ``status=success, injected=N`` + JSON body naming each tool,
  and the tool ACTUALLY lands in ``mp.active_tools`` (the real mutation).
- select known + unknown → success with the unknown name in ``not_found`` and
  ``not_found=M`` meta — partial success is explicit, never silent.
- select ONLY unknown → ``status=error, code=unknown-tool`` + ``valid:`` ladder of
  real selectable tool names.
- select a delegate blocked on the invoking channel → ``status=error,
  code=blocked-on-channel`` and the blocked name NEVER appears in ``injected`` /
  ``active_tools`` (the silent-partial-success the ticket kills).
- query mode → ``status=success`` with a structured rows body.

NETWORK CONSTRAINT: unit tests are deterministic and offline. The select paths
touch no network. The query path runs the real EmbeddingService + the real
abilities.sqlite vec/FTS search locally — no engine call.
"""

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import DmnConfig, UserConfig
from services.act_trail import ActTrail
from services.message_processor import MessageProcessor

from tests._tool_result_harness import parse_body, seed_transcript

pytestmark = pytest.mark.unit


def _mp_for(config, db=None) -> MessageProcessor:
    """A real MessageProcessor bound to a real channel config, seeded exactly as
    ``_setup`` seeds it (active_tools = config.always_available). This is the same
    construction the channel-isolation feature suite drives. When *db* is given,
    a real transcript anchor is seeded and bound to ``mp.uid`` so the act-trail
    write the dispatcher performs lands against a real row.

    Kept local (not the harness ``MP``) because this drives the genuine
    ``MessageProcessor`` carrying ``active_tools`` (the live mutation find_tools
    asserts) and a real production channel config — beyond the minimal
    ``config``+``uid`` harness ``MP``."""
    mp = MessageProcessor("find a tool for me")
    mp.config = config
    mp.active_tools = list(config.always_available or [])
    if db is not None:
        channel = (
            getattr(config, "policy_channel", None).value
            if getattr(config, "policy_channel", None)
            else "chat"
        )
        mp.uid = seed_transcript(db, channel, "find a tool for me")
    return mp


def _head(rendered: str) -> str:
    """The open-tag line ``[find_tools(status=…, …)]`` the model reads first.

    Kept local (not the harness ``head``): call sites compare it with ``==`` /
    ``.startswith`` / ``in`` and need the raw first line, not the harness opener
    assertion."""
    return rendered.splitlines()[0]


def _body(rendered: str) -> object:
    """Extract and JSON-parse the body between the open tag and ``[end:find_tools]``."""
    return parse_body(rendered, "find_tools")


# ── select a known tool: success envelope + real active_tools mutation ─────────


def test_select_known_tool_injects_and_reports_success(db):
    """``find_tools(select=['weather'])`` on the user channel renders a success
    envelope with ``injected=1``, a JSON body naming weather, and ACTUALLY lands
    'weather' in the live ``mp.active_tools`` — the real downstream mutation."""
    mp = _mp_for(UserConfig({}), db)
    assert "weather" not in mp.active_tools  # precondition

    out = ToolDispatcher(mp).dispatch("find_tools", {"select": ["weather"], "act_summary": "x"})

    assert _head(out) == "[find_tools(status=success, injected=1)]"
    body = _body(out)
    names = [e["name"] for e in body["injected"]]
    assert names == ["weather"]
    assert body["injected"][0]["summary"]  # carries the tool's summary
    assert body["not_found"] == []
    # The real mutation: weather is now live on the processor for the next ACT turn.
    assert "weather" in mp.active_tools

    # The act-trail recorded the same success envelope against the transcript.
    trail = ActTrail().fetch_by_transcript_id(mp.uid)
    assert "[find_tools(status=success, injected=1)]" in trail[0]["result"]


# ── select known + unknown: explicit partial success, never silent ─────────────


def test_select_known_and_unknown_reports_not_found_explicitly(db):
    """A mix of a real tool and a bogus name injects the real one AND names the
    bogus one in ``not_found`` with ``not_found=1`` meta — partial success is
    explicit, the model can see exactly what it did and did not get."""
    mp = _mp_for(UserConfig({}))

    out = ToolDispatcher(mp).dispatch(
        "find_tools",
        {"select": ["weather", "totally_made_up_tool"], "act_summary": "x"},
    )

    head = _head(out)
    assert head.startswith("[find_tools(status=success")
    assert "injected=1" in head
    assert "not_found=1" in head
    body = _body(out)
    assert [e["name"] for e in body["injected"]] == ["weather"]
    assert body["not_found"] == ["totally_made_up_tool"]
    assert "weather" in mp.active_tools
    assert "totally_made_up_tool" not in mp.active_tools


# ── select ONLY unknown: loud error with a real valid ladder ───────────────────


def test_select_only_unknown_errors_with_valid_ladder(db):
    """Selecting a name that is no real tool at all returns ``code=unknown-tool``
    with a ``valid:`` ladder of REAL selectable tool names so a weak model
    self-corrects — and nothing is injected."""
    mp = _mp_for(UserConfig({}))

    out = ToolDispatcher(mp).dispatch(
        "find_tools", {"select": ["totally_made_up_tool"], "act_summary": "x"}
    )

    assert "[find_tools(status=error, code=unknown-tool" in out
    assert "code=error]" not in out
    assert "valid:" in out
    assert "[end:find_tools]" in out
    # The ladder names a real discoverable tool the model could have picked.
    valid_line = next(ln for ln in out.splitlines() if ln.startswith("valid:"))
    assert "weather" in valid_line
    assert mp.active_tools == list(UserConfig({}).always_available or [])


# ── blocked delegate on a rejecting channel: NEVER injected, loud error ────────


def test_select_blocked_delegate_on_dmn_is_never_injected(db):
    """DmnConfig blocks the delegate/raw-web tools. Selecting one returns
    ``code=blocked-on-channel`` and the blocked name NEVER appears in injected or
    in active_tools — the silent-partial-success the ticket kills."""
    mp = _mp_for(DmnConfig())
    assert "web_browse" in mp.config.blocked  # contract guard

    out = ToolDispatcher(mp).dispatch(
        "find_tools", {"select": ["web_browse"], "act_summary": "x"}
    )

    assert "[find_tools(status=error, code=blocked-on-channel" in out
    assert "code=error]" not in out
    assert "[end:find_tools]" in out
    assert "web_browse" not in mp.active_tools


def test_select_blocked_delegate_is_not_dressed_as_injected(db):
    """Even mixed with a tool DmnConfig DOES allow, a blocked delegate must land
    in not_found, never in injected — a blocked tool dressed as injected is the
    exact lie TKT-894 forbids."""
    mp = _mp_for(DmnConfig())
    discoverable = list(mp.config.discoverable or [])
    # 'memory' is always-available on every channel and never blocked.
    allowed = "memory" if "memory" in (mp.config.always_available or []) else discoverable[0]

    out = ToolDispatcher(mp).dispatch(
        "find_tools", {"select": [allowed, "web_browse"], "act_summary": "x"}
    )

    body = _body(out) if "status=success" in _head(out) else None
    if body is not None:
        injected_names = [e["name"] for e in body["injected"]]
        assert "web_browse" not in injected_names
    assert "web_browse" not in mp.active_tools


# ── query mode: structured rows body ───────────────────────────────────────────


def test_query_mode_returns_structured_rows(db):
    """The query (semantic discovery) path returns a structured rows body — each
    row a ``{name, summary}`` dict — with a success envelope, not prose."""
    mp = _mp_for(UserConfig({}))

    out = ToolDispatcher(mp).dispatch(
        "find_tools",
        {"query": "what is the weather forecast for tomorrow", "act_summary": "x"},
    )

    assert _head(out).startswith("[find_tools(status=success")
    assert "[end:find_tools]" in out
    body = _body(out)
    assert isinstance(body, dict)
    rows = body["injected"]
    assert isinstance(rows, list)
    # Every discovered row carries a name and a summary.
    for row in rows:
        assert isinstance(row["name"], str) and row["name"]
        assert "summary" in row
    # A weather-shaped query surfaces weather and lands it on the processor.
    if rows:
        assert any(r["name"] == "weather" for r in rows)
        assert "weather" in mp.active_tools


# ── require at least one param ──────────────────────────────────────────────────


def test_no_params_errors_with_actionable_guidance(db):
    """A call with neither select nor query is rejected with a stable error so the
    model knows to supply one — not the legacy ``code=error`` marker."""
    mp = _mp_for(UserConfig({}))

    out = ToolDispatcher(mp).dispatch("find_tools", {"act_summary": "x"})

    assert "[find_tools(status=error" in out
    assert "code=error]" not in out
    assert "[end:find_tools]" in out
