# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test: the ``file_write`` truncated-read overwrite guard.

``abilities/file_write.py::FileWriteAbility._overwrite_guard`` inspects the
MOST RECENT ``read`` tool call on the current turn matching the write target:
no matching read → ``code=read-required``; the most recent matching read was
truncated (its persisted envelope open tag carries ``truncated=true``) →
``code=truncated-read``; otherwise the overwrite proceeds. Truncation itself
comes from ``ReadAbility``/``TextReader``: a ``max_chars`` smaller than the
file content clips the body and stamps ``truncated=true`` into the envelope
``dispatch_service._render`` writes into the persisted ``tool_calls.result``
row — the exact field the guard reads back.

Drives the REAL production entry point end to end: a real ``MessageProcessor``
turn against the real SQLite DB (``db`` fixture), the real ``DispatchService``,
the real ``read``/``file_write`` abilities, and real files on disk under
``tmp_path``. The only substitution is the LLM network boundary
(``services.provider_service.build_client``), scripted to replay one tool call
per step — mirrors ``test_dispatch_repeat_call_steer.py``. ``read``/``file_write``
both carry seeded ``allow`` policy rows on the ``chat`` channel in the real
policy seed (``abilities/assets/policy_defaults.json``), so no policy seeding
is needed here.
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

pytestmark = pytest.mark.unit

# ProviderService builds its thin transport client via this factory call — the
# real network boundary (see test_message_processor_runaway_loop.py,
# test_dispatch_repeat_call_steer.py).
_BUILD_CLIENT = "services.provider_service.build_client"

# Real file content, long enough (> 100 chars — the ReadParamsBag max_chars
# floor, see contracts/params/read_params_bag.py's clamp_int(lo=100)) that a
# max_chars=100 read is guaranteed to truncate it.
_ORIGINAL_CONTENT = (
    "Line one of the original file.\n"
    "Line two adds more real prose so the content comfortably exceeds the "
    "100-character minimum max_chars floor enforced by ReadParamsBag, "
    "guaranteeing a truncated read whenever max_chars=100 is requested.\n"
)


def _tool(name: str, **params: object) -> dict[str, object]:
    """A provider-shaped tool call: ``{"name": ..., "input": {...}}``."""
    return {"name": name, "input": params}


class _ScriptedProvider:
    """Replays one scripted ``ProviderResponse`` per ``send()`` call, in order.

    Past the end of the script it returns a benign terminal (no-tool, empty)
    response rather than raising: a completed ``user`` turn spawns a
    fire-and-forget skill-suggestion turn (``_proactive_suggestion``) that makes
    its own provider call on a daemon thread this test does not script for."""

    _TERMINAL = ProviderResponse(text="", model="scripted-guard", tool_calls=None)

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
    """The envelope's first line — where ``truncated=true`` / ``code=...`` /
    ``status=...`` are rendered (dispatch_service._render)."""
    return result.split("\n", 1)[0]


# ── Case 1: a truncated read refuses the overwrite, file left untouched ────────


def test_truncated_read_blocks_overwrite_file_unchanged(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """A read that truncated the file (max_chars=100 on content > 100 chars)
    refuses the very next file_write on the same path with
    code=truncated-read, and the file on disk is left byte-identical."""
    assert db is not None
    target = tmp_path / "notes.txt"
    target.write_text(_ORIGINAL_CONTENT, encoding="utf-8")

    provider = _ScriptedProvider(
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("read", source=str(target), max_chars=100)],
        ),
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("file_write", path=str(target), contents="attempted-overwrite-A")],
        ),
        ProviderResponse(text="All done.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "read the notes file then overwrite it")

    reads = [c for c in mp.tool_call_service.by_turn() if c.tool_name == "read"]
    writes = [c for c in mp.tool_call_service.by_turn() if c.tool_name == "file_write"]

    assert len(reads) == 1
    assert "truncated=true" in _open_tag(reads[0].result), reads[0].result

    assert len(writes) == 1
    assert writes[0].state == ToolCall.ERROR
    assert "code=truncated-read" in _open_tag(writes[0].result), writes[0].result

    assert target.read_text(encoding="utf-8") == _ORIGINAL_CONTENT


# ── Case 2: a newer full read overrides an older truncated one ─────────────────


def test_fresh_full_read_after_truncated_read_allows_overwrite(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """Most-recent-read semantics: after case 1's refusal, re-reading the SAME
    target with no truncation (max_chars large enough for the full content)
    makes the following file_write succeed and rewrite the file — the newer
    full read overrides the older truncated one."""
    assert db is not None
    target = tmp_path / "notes.txt"
    target.write_text(_ORIGINAL_CONTENT, encoding="utf-8")

    provider = _ScriptedProvider(
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("read", source=str(target), max_chars=100)],  # truncated
        ),
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("file_write", path=str(target), contents="attempted-overwrite-A")],  # refused
        ),
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("read", source=str(target))],  # full re-read, default max_chars
        ),
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("file_write", path=str(target), contents="final-content-B")],  # allowed
        ),
        ProviderResponse(text="All done.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "read the notes file, retry after a fresh full read")

    reads = [c for c in mp.tool_call_service.by_turn() if c.tool_name == "read"]
    writes = [c for c in mp.tool_call_service.by_turn() if c.tool_name == "file_write"]

    assert len(reads) == 2
    assert "truncated=true" in _open_tag(reads[0].result), reads[0].result
    assert "truncated=true" not in _open_tag(reads[1].result), reads[1].result

    assert len(writes) == 2
    assert writes[0].state == ToolCall.ERROR
    assert "code=truncated-read" in _open_tag(writes[0].result), writes[0].result
    assert writes[1].state == ToolCall.DONE
    assert "status=success" in _open_tag(writes[1].result), writes[1].result

    assert target.read_text(encoding="utf-8") == "final-content-B"


# ── Case 3: no prior read this turn refuses the overwrite (regression pin) ─────


def test_write_without_prior_read_this_turn_is_refused(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """Regression pin for the pre-existing behaviour: file_write on an existing
    file with NO prior read this turn is refused with code=read-required, and
    the file on disk is left untouched."""
    assert db is not None
    target = tmp_path / "notes.txt"
    target.write_text(_ORIGINAL_CONTENT, encoding="utf-8")

    provider = _ScriptedProvider(
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("file_write", path=str(target), contents="no-read-content")],
        ),
        ProviderResponse(text="All done.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "overwrite the notes file without reading it first")

    writes = [c for c in mp.tool_call_service.by_turn() if c.tool_name == "file_write"]

    assert len(writes) == 1
    assert writes[0].state == ToolCall.ERROR
    assert "code=read-required" in _open_tag(writes[0].result), writes[0].result

    assert target.read_text(encoding="utf-8") == _ORIGINAL_CONTENT
