# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test: ``Ability.VERBATIM`` params reach ``run()`` byte for byte.

The dispatch seam scrubs every argument value — leaked provider sentinel tokens
out, surrounding whitespace trimmed (``ProviderService.sanitize_args``). That is
right for a value the tool INTERPRETS (a path, a query, a city) and wrong for one
it STORES. ``edit_file``'s ``search`` is an anchor that must match the file
character for character, and its ``replace`` is written into the file as-is; a
trim there is a silent rewrite of the user's file.

Measured before the fix, not inferred: ``edit_file(search="DELETE THIS LINE\\n",
replace="")`` deleted only ``"DELETE THIS LINE"`` and left a blank line behind,
because the trailing newline never reached the ability. The same trim made
indentation edits, whole-file clears, and whitespace-only replacements
impossible, and it silently dropped the trailing newline from every file
``write_file`` has ever written.

The fix is an opt-OUT: scrubbing stays the global rule and an ability names the
few params that must arrive untouched. Cases 1-4 drive real turns end to end;
case 5 asserts the seam itself, because a bag sitting behind the seam has
already masked one regression on this exact path (see
``test_edit_file_empty_replace.py``).

Cases 1-4 use the real production entry point: a real ``MessageProcessor`` turn
against the real SQLite DB (``db`` fixture), the real ``DispatchService``, the
real abilities, and real files under ``tmp_path``. The only substitution is the
LLM network boundary (``services.provider_service.build_client``), scripted to
replay one tool call per step. ``edit_file`` and ``write_file`` both carry a
seeded ``allow`` policy row on the ``chat`` channel
(``abilities/assets/policy_defaults.json``), so no policy seeding is needed.
"""

import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.provider_response import ProviderResponse
from models.tool_call import ToolCall

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

# ProviderService builds its thin transport client via this factory call — the
# real network boundary (see test_read_guard.py).
_BUILD_CLIENT = "services.provider_service.build_client"


def _tool(name: str, **params: object) -> dict[str, object]:
    """A provider-shaped tool call: ``{"name": ..., "input": {...}}``."""
    return {"name": name, "input": params}


class _ScriptedProvider:
    """Replays one scripted ``ProviderResponse`` per ``send()`` call, in order.

    Past the end of the script it returns a benign terminal (no-tool, empty)
    response rather than raising: a completed ``user`` turn spawns a
    fire-and-forget skill-suggestion turn that makes its own provider call on a
    daemon thread this test does not script for."""

    _TERMINAL = ProviderResponse(text="", model="scripted-verbatim", tool_calls=None)

    def __init__(self, *responses: ProviderResponse) -> None:
        self._responses = list(responses)
        self.sends = 0

    def get_context_limit(self) -> int:
        return 200000

    def estimate_request_tokens(self, _dto: object) -> int:
        return 1

    def send(self, _dto: object) -> ProviderResponse:
        if self.sends >= len(self._responses):
            return self._TERMINAL
        response = self._responses[self.sends]
        self.sends += 1
        return response


def _drain_background_turns(timeout_s: float = 10.0) -> None:
    """Join the fire-and-forget post-turn daemon turns a completed ``user`` turn
    spawns (skill suggestion, thread gist) so they run to completion inside THIS
    test's provider+DB patch and never leak into the next test."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pending = [
            t for t in threading.enumerate()
            if t.name in ("skill-suggest", "thread-gist") or t.name.startswith("turn-")
        ]
        if not pending:
            return
        for t in pending:
            t.join(timeout=deadline - time.monotonic())


def _run(provider: _ScriptedProvider, raw_input: str) -> MessageProcessor:
    """Drive a real user turn to termination against *provider*, then quiesce any
    post-turn daemon turns inside the patch so the test stays hermetic."""
    mp = MessageProcessor(UserConfig(), raw_input=raw_input)  # inert (I2)
    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        mp.result()
        _drain_background_turns()
    return mp


def _open_tag(result: str) -> str:
    """The envelope's first line — where ``code=...`` / ``status=...`` are
    rendered (dispatch_service._render)."""
    return result.split("\n", 1)[0]


def _calls(mp: MessageProcessor, tool_name: str) -> list[ToolCall]:
    return [c for c in mp.tool_call_service.by_turn() if c.tool_name == tool_name]


def _assert_succeeded(mp: MessageProcessor, tool_name: str) -> None:
    calls = _calls(mp, tool_name)
    assert len(calls) == 1
    assert calls[0].state == ToolCall.DONE
    assert "status=success" in _open_tag(calls[0].result), calls[0].result


# ── Case 1: a whole-line deletion keeps the newline in the anchor ─────────────


def test_edit_file_deletes_a_whole_line_including_its_newline(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """The measured failure. ``search`` carries the line's trailing ``\\n``, so the
    line disappears entirely instead of collapsing to a blank one."""
    assert db is not None
    target = tmp_path / "notes.txt"
    target.write_text("alpha\nDELETE THIS LINE\nomega\n", encoding="utf-8")

    provider = _ScriptedProvider(
        # The read is the protocol, not scaffolding: abilities/_read_guard.py
        # refuses an edit the model has not read for on this turn.
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("read", source=str(target))],
        ),
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool(
                "edit_file", path=str(target),
                search="DELETE THIS LINE\n", replace="",
            )],
        ),
        ProviderResponse(text="Line removed.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "delete the middle line of the notes file")

    _assert_succeeded(mp, "edit_file")
    assert target.read_text(encoding="utf-8") == "alpha\nomega\n"


# ── Case 2: an indentation fix keeps the leading whitespace ──────────────────


def test_edit_file_can_indent_a_line(db: sqlite3.Connection, tmp_path: Path) -> None:
    """``replace`` is written as-is, so leading whitespace survives and the line
    is genuinely indented rather than rewritten flush left."""
    assert db is not None
    target = tmp_path / "snippet.py"
    target.write_text("def f():\nreturn 1\n", encoding="utf-8")

    provider = _ScriptedProvider(
        # The read is the protocol, not scaffolding: abilities/_read_guard.py
        # refuses an edit the model has not read for on this turn.
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("read", source=str(target))],
        ),
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool(
                "edit_file", path=str(target),
                search="return 1", replace="    return 1",
            )],
        ),
        ProviderResponse(text="Indented.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "fix the indentation in the snippet")

    _assert_succeeded(mp, "edit_file")
    assert target.read_text(encoding="utf-8") == "def f():\n    return 1\n"


# ── Case 3: clearing a file leaves 0 bytes, not a stray newline ──────────────


def test_edit_file_can_clear_a_file_to_zero_bytes(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """Anchoring on the file's entire contents — trailing newline included —
    empties it completely."""
    assert db is not None
    target = tmp_path / "scratch.txt"
    target.write_text("only line\n", encoding="utf-8")

    provider = _ScriptedProvider(
        # The read is the protocol, not scaffolding: abilities/_read_guard.py
        # refuses an edit the model has not read for on this turn.
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("read", source=str(target))],
        ),
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool(
                "edit_file", path=str(target), search="only line\n", replace="",
            )],
        ),
        ProviderResponse(text="Cleared.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "clear out the scratch file")

    _assert_succeeded(mp, "edit_file")
    assert target.read_text(encoding="utf-8") == ""
    assert target.stat().st_size == 0


# ── Case 4: a written file keeps the trailing newline the model chose ────────


def test_write_file_keeps_the_trailing_newline(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """``contents`` is stored, not interpreted: the file on disk is byte-identical
    to what the model sent. The target does not exist, so the read-before-write
    guard does not apply."""
    assert db is not None
    target = tmp_path / "new.txt"

    provider = _ScriptedProvider(
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("write_file", path=str(target), contents="alpha\nomega\n")],
        ),
        ProviderResponse(text="Written.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "write a two-line file")

    _assert_succeeded(mp, "write_file")
    assert target.read_text(encoding="utf-8") == "alpha\nomega\n"


# ── Case 5: the seam's own contract, independent of the bags behind it ───────


def test_seam_scrubs_every_value_except_the_verbatim_params(
    db: sqlite3.Connection,
) -> None:
    """``DispatchService._prepare`` decides this on its own, before any ParamBag
    is built.

    Asserted directly because the bags behind these tools re-validate the same
    params and have already masked one regression on this path. The dispatcher
    and the abilities are the real production objects."""
    assert db is not None
    mp = MessageProcessor(UserConfig(), raw_input="verbatim seam contract")  # inert (I2)
    prepare = mp.dispatch_service._prepare  # noqa: SLF001 — the seam under test

    params, _summary, ability = prepare("edit_file", {
        "path": "  /tmp/x  ",
        "search": "\n\tanchor \n",
        "replace": "    padded    ",
    })
    assert ability is not None
    assert params["path"] == "/tmp/x"                 # interpreted → scrubbed
    assert params["search"] == "\n\tanchor \n"        # stored → untouched
    assert params["replace"] == "    padded    "      # a whitespace-only edit is real

    # An ability that declared no VERBATIM params is unaffected: the global rule
    # still applies to every one of its values.
    plain, _plain_summary, plain_ability = prepare("read", {"source": "  /tmp/x  "})
    assert plain_ability is not None
    assert plain["source"] == "/tmp/x"

    # Keys are healed BEFORE values are scrubbed, so the exemption — declared in
    # canonical names — still holds when the model reached the param through a
    # variant key.
    healed, _healed_summary, _healed_ability = prepare("edit_file", {
        "path": "/tmp/x", "search": "anchor", "Replace": "  padded  ",
    })
    assert healed["replace"] == "  padded  "

    # The exemption is total, sentinel stripping included: a leaked token written
    # into a file is visible and recoverable, whereas one silently deleted from an
    # anchor turns into an unexplainable not-found.
    sentinel, _sentinel_summary, _sentinel_ability = prepare("edit_file", {
        "path": "/tmp/x", "search": "a<|end|>b", "replace": "c",
    })
    assert sentinel["search"] == "a<|end|>b"

    # act_summary is a framework field, lifted out before healing and scrubbed on
    # its own — no ability declares it, so it must never be routed by VERBATIM.
    _lifted, summary, _lifted_ability = prepare("edit_file", {
        "path": "/tmp/x", "search": "a", "replace": "b", "act_summary": "  narrating  ",
    })
    assert summary == "narrating"
    assert "act_summary" not in _lifted
