# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test: one owner per file operation.

Two tools could both create a file. ``write_file`` with empty contents made a
0-byte file, and the merged file-CRUD tool's ``create`` action made either a
file or a folder depending on whether the path it was handed ended in a
trailing slash — a heuristic a small local model routinely got wrong, and the
reason a request for a folder could silently produce a file instead.

The merged tool is gone, split into ``make_dir`` (directories only), ``delete``
and ``set_permissions``. ``write_file`` is now the only way to create a file and
``make_dir`` the only way to create a directory, so the outcome no longer
depends on a punctuation mark.

``test_code_agent_tools.py`` covers each new ability's own logic by calling
``run()`` directly. This file covers what that cannot: that the three new names
are REGISTERED, POLICY-ALLOWED, and reachable through the real dispatcher with
their real param bags and the real ACTION_REQUIRED pre-gate. A tool whose class
is perfect but whose name never reaches the model is still a broken tool.

Every case drives the REAL production entry point end to end: a real
``MessageProcessor`` turn against the real SQLite DB (``db`` fixture), the real
``DispatchService``, the real abilities, and real paths on disk under
``tmp_path``. The only substitution is the LLM network boundary
(``services.provider_service.build_client``), scripted to replay one tool call
per step — the same harness as ``test_read_guard.py``. All four tools carry
seeded ``allow`` policy rows on the ``chat`` channel in the real policy seed
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
_BUILD_CLIENT = "services.provider_service.Factory.build_client"


def _tool(name: str, **params: object) -> dict[str, object]:
    """A provider-shaped tool call: ``{"name": ..., "input": {...}}``."""
    return {"name": name, "input": params}


def _step(*calls: dict[str, object]) -> ProviderResponse:
    return ProviderResponse(text="", model="scripted", tool_calls=list(calls))


class _ScriptedProvider:
    """Replays one scripted ``ProviderResponse`` per ``send()`` call, in order.

    Past the end of the script it returns a benign terminal response. Its text
    is NON-empty on purpose: an empty completion is steered and then crashes the
    turn with EmptyCompletionLoop, filling the log with a traceback that has
    nothing to do with the tools under test."""

    _TERMINAL = ProviderResponse(text="Nothing further.", model="scripted-split", tool_calls=None)

    def __init__(self, *responses: ProviderResponse) -> None:
        self._responses = list(responses)
        self.sends = 0

    def get_context_limit(self) -> int:
        return 200000


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
    """The envelope's first line — where ``status=`` / ``code=`` are rendered
    (dispatch_service._render)."""
    return result.split("\n", 1)[0]


def _calls(mp: MessageProcessor, *names: str) -> list[ToolCall]:
    return [c for c in mp.tool_call_service.by_turn() if c.tool_name in names]


# ── Case 1: all four tools reach the model and act on disk ───────────────────


def test_the_four_file_tools_run_end_to_end_in_one_turn(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """make_dir → write_file → set_permissions → delete, driven as a real turn.

    Each step is verified against the FILESYSTEM, not against the tool's own
    reply — a tool that reports success while doing nothing is exactly the
    failure this asserts against."""
    assert db is not None
    folder = tmp_path / "project" / "reports"
    target = folder / "summary.md"

    provider = _ScriptedProvider(
        _step(_tool("make_dir", path=str(folder))),
        _step(_tool("write_file", path=str(target), contents="# Summary\n")),
        _step(_tool("set_permissions", path=str(target), permission_code="0600")),
        _step(_tool("delete", path=str(folder))),
        ProviderResponse(text="Done.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "set up a reports folder, write a summary, lock it down, then clean up")

    calls = _calls(mp, "make_dir", "write_file", "set_permissions", "delete")
    assert [c.tool_name for c in calls] == [
        "make_dir", "write_file", "set_permissions", "delete",
    ], [(c.tool_name, c.result) for c in calls]
    for call in calls:
        assert call.state == ToolCall.DONE, (call.tool_name, call.result)
        assert "status=success" in _open_tag(call.result), (call.tool_name, call.result)

    # The delete removed the whole tree — the last observable state on disk.
    assert not folder.exists()
    assert not target.exists()


# ── Case 2: make_dir makes a DIRECTORY, trailing slash or not ────────────────


def test_make_dir_makes_a_directory_without_a_trailing_slash(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """The overlap is gone at the seam the model actually calls.

    The merged tool read a trailing slash to decide file-or-folder, so this
    exact argument used to produce an empty FILE. ``make_dir`` has one outcome
    and the punctuation no longer decides anything."""
    assert db is not None
    folder = tmp_path / "logs"

    provider = _ScriptedProvider(
        _step(_tool("make_dir", path=str(folder))),
        ProviderResponse(text="Created it.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "make a logs folder")

    calls = _calls(mp, "make_dir")
    assert len(calls) == 1
    assert calls[0].state == ToolCall.DONE, calls[0].result
    assert folder.is_dir()
    assert not folder.is_file()


# ── Case 3: the pre-gate guards the new names too ────────────────────────────


def test_the_new_tools_refuse_a_missing_path_at_the_pre_gate(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """``path`` omitted → a single ``missing-params`` refusal, and nothing runs.

    Registration alone is not wiring: ACTION_REQUIRED has to reach the
    dispatcher's pre-gate for each new name, or a malformed call falls through
    to ``run()`` and fails somewhere less legible."""
    assert db is not None
    stray = tmp_path / "stray.txt"
    stray.write_text("untouched", encoding="utf-8")

    provider = _ScriptedProvider(
        _step(_tool("make_dir")),
        _step(_tool("delete")),
        _step(_tool("set_permissions", path=str(stray))),  # permission_code missing
        ProviderResponse(text="I need a path.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "make a folder, delete something, change permissions")

    calls = _calls(mp, "make_dir", "delete", "set_permissions")
    assert len(calls) == 3, [(c.tool_name, c.result) for c in calls]
    for call in calls:
        assert call.state == ToolCall.ERROR, (call.tool_name, call.result)
        assert "code=missing-params" in _open_tag(call.result), (call.tool_name, call.result)

    assert "permission_code" in calls[2].result, calls[2].result
    assert stray.read_text(encoding="utf-8") == "untouched"
