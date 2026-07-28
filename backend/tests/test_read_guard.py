# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test: the shared read-before-modify guard (``abilities/_read_guard.py``).

One rule for every ability that changes a file: the model's most recent ``read``
of the target must be newer than the most recent SUCCESSFUL change to that target
on this turn. No read at all and a read invalidated by the model's own earlier
edit both come back ``code=read-required``.

Turn-scoped alone was not enough, and cases 4-6 are why. ``read -> edit(ok) ->
edit(blind)`` satisfied the old guard — any read of the path anywhere in the turn
counted, including one the first edit had already invalidated. The second edit
then anchors on text the file no longer contains, returns ``not-found``, and the
model retries it until the runaway backstop kills the turn.

``file_write`` alone opts into the extra ``truncated-read`` refusal (cases 1-2):
a truncated read is fatal to a whole-file overwrite, which would destroy the tail
the model never saw, and harmless to an anchored edit, which only touches text
the model quoted back verbatim. Truncation itself comes from
``ReadAbility``/``TextReader``: a ``max_chars`` smaller than the file content
clips the body and stamps ``truncated=true`` into the envelope
``dispatch_service._render`` writes into the persisted ``tool_calls.result``
row — the exact field the guard reads back.

Cases 1-7 drive the REAL production entry point end to end: a real
``MessageProcessor`` turn against the real SQLite DB (``db`` fixture), the real
``DispatchService``, the real ``read``/``file_write``/``edit_file`` abilities, and
real files on disk under ``tmp_path``. The only substitution is the LLM network
boundary (``services.provider_service.build_client``), scripted to replay one tool
call per step — mirrors ``test_dispatch_repeat_call_steer.py``. All three tools
carry seeded ``allow`` policy rows on the ``chat`` channel in the real policy seed
(``abilities/assets/policy_defaults.json``), so no policy seeding is needed here.
Case 8 asserts the two bypasses directly, since neither is reachable from a turn.
"""

import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from abilities._read_guard import read_guard
from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.provider_response import ProviderResponse
from models.tool_call import ToolCall

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

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

# Three distinct one-word anchors, so a sequence of edits is legible on disk and
# no anchor can accidentally match twice (which would be ``not-unique``, not the
# outcome under test).
_EDITABLE_CONTENT = "alpha\nbeta\ngamma\n"


def _tool(name: str, **params: object) -> dict[str, object]:
    """A provider-shaped tool call: ``{"name": ..., "input": {...}}``."""
    return {"name": name, "input": params}


class _ScriptedProvider:
    """Replays one scripted ``ProviderResponse`` per ``send()`` call, in order.

    Past the end of the script it returns a benign terminal (no-tool) response
    rather than raising: a completed ``user`` turn spawns a fire-and-forget
    skill-suggestion turn (``_proactive_suggestion``) that makes its own provider
    call on a daemon thread this test does not script for. Its text is NON-empty
    on purpose — an empty completion is steered and then crashes the turn with
    EmptyCompletionLoop (message_processor.py:355), which would fill this test's
    log with a crash traceback that has nothing to do with the guard."""

    _TERMINAL = ProviderResponse(text="Nothing further.", model="scripted-guard", tool_calls=None)

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


def _calls(mp: MessageProcessor, tool_name: str) -> list[ToolCall]:
    return [c for c in mp.tool_call_service.by_turn() if c.tool_name == tool_name]


def _head_parts(result: str) -> list[str]:
    """The open tag's comma-separated head parts. ``dispatch_service._render``
    writes ``[<tool>(status=success, <meta>)]`` or
    ``[<tool>(status=error, code=<kebab>, <meta>)]``."""
    tag = _open_tag(result)
    return [part.strip() for part in tag[tag.index("(") + 1 : tag.rindex(")")].split(",")]


def _codes(calls: list[ToolCall]) -> list[str]:
    """Each call's outcome as ``status=success`` or ``code=<kebab>`` — the whole
    sequence in one assertable list, so a wrong outcome anywhere is visible at
    once rather than one assertion at a time."""
    parts = [_head_parts(call.result) for call in calls]
    return [next((p for p in head if p.startswith("code=")), head[0]) for head in parts]


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


# ── Case 4: edit_file with no read this turn is refused ───────────────────────


def test_edit_without_prior_read_this_turn_is_refused(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """The guard is not file_write's alone: an anchored edit on a file the model
    has not read this turn is refused with code=read-required, and the file on
    disk is left untouched."""
    assert db is not None
    target = tmp_path / "notes.txt"
    target.write_text(_EDITABLE_CONTENT, encoding="utf-8")

    provider = _ScriptedProvider(
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("edit_file", path=str(target), search="beta", replace="BETA")],
        ),
        ProviderResponse(text="I should read it first.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "edit the notes file without reading it first")

    assert _codes(_calls(mp, "edit_file")) == ["code=read-required"]
    assert target.read_text(encoding="utf-8") == _EDITABLE_CONTENT


# ── Case 5: read then edit is allowed ─────────────────────────────────────────


def test_read_then_edit_is_allowed(db: sqlite3.Connection, tmp_path: Path) -> None:
    """The ordinary path: one read, one edit, one change on disk."""
    assert db is not None
    target = tmp_path / "notes.txt"
    target.write_text(_EDITABLE_CONTENT, encoding="utf-8")

    provider = _ScriptedProvider(
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("read", source=str(target))],
        ),
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("edit_file", path=str(target), search="beta", replace="BETA")],
        ),
        ProviderResponse(text="Edited.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "read the notes file then edit it")

    assert _codes(_calls(mp, "edit_file")) == ["status=success"]
    assert target.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"


# ── Case 6: a second edit behind the same read is stale ───────────────────────


def test_second_edit_behind_one_read_is_refused_as_stale(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """The defect this guard was tightened for. The first edit succeeds and
    invalidates the read; the second edit — still behind that one read — is
    refused with code=read-required instead of returning not-found and inviting
    a retry storm. Only the first edit reaches disk."""
    assert db is not None
    target = tmp_path / "notes.txt"
    target.write_text(_EDITABLE_CONTENT, encoding="utf-8")

    provider = _ScriptedProvider(
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("read", source=str(target))],
        ),
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("edit_file", path=str(target), search="beta", replace="BETA")],
        ),
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("edit_file", path=str(target), search="gamma", replace="GAMMA")],
        ),
        ProviderResponse(text="I need to re-read.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "read the notes file then make two edits")

    edits = _calls(mp, "edit_file")
    assert _codes(edits) == ["status=success", "code=read-required"]
    # The refusal names the culprit, so the model is told to look again rather
    # than to try harder.
    assert "changed since you last read it" in edits[1].result, edits[1].result

    assert target.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"


# ── Case 7: re-reading between edits clears the staleness ─────────────────────


def test_reread_between_edits_allows_the_second_edit(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """The remedy the refusal asks for actually works: read, edit, read again,
    edit again — both edits land."""
    assert db is not None
    target = tmp_path / "notes.txt"
    target.write_text(_EDITABLE_CONTENT, encoding="utf-8")

    provider = _ScriptedProvider(
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("read", source=str(target))],
        ),
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("edit_file", path=str(target), search="beta", replace="BETA")],
        ),
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("read", source=str(target))],
        ),
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("edit_file", path=str(target), search="gamma", replace="GAMMA")],
        ),
        ProviderResponse(text="Both edits done.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "read, edit, re-read and edit the notes file again")

    assert _codes(_calls(mp, "edit_file")) == ["status=success", "status=success"]
    assert target.read_text(encoding="utf-8") == "alpha\nBETA\nGAMMA\n"


# ── Case 8: a FAILED edit does not invalidate the read ───────────────────────


def test_failed_edit_does_not_invalidate_the_read(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """Only a SUCCEEDED change makes a read stale. An edit that returned
    not-found wrote nothing, so the model's view of the file still holds and the
    next edit behind the same read is allowed — otherwise one mistyped anchor
    would cost a mandatory re-read."""
    assert db is not None
    target = tmp_path / "notes.txt"
    target.write_text(_EDITABLE_CONTENT, encoding="utf-8")

    provider = _ScriptedProvider(
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("read", source=str(target))],
        ),
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("edit_file", path=str(target), search="absent", replace="X")],
        ),
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("edit_file", path=str(target), search="beta", replace="BETA")],
        ),
        ProviderResponse(text="Fixed my anchor.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "read the notes file, mistype an anchor, then edit properly")

    assert _codes(_calls(mp, "edit_file")) == ["code=not-found", "status=success"]
    assert target.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"


# ── Case 9: the staleness rule binds file_write too ──────────────────────────


def test_second_write_behind_one_read_is_refused_as_stale(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """One guard, both tools: a second overwrite behind a single read is refused
    for the same reason an edit is, and the file keeps the first write."""
    assert db is not None
    target = tmp_path / "notes.txt"
    target.write_text(_EDITABLE_CONTENT, encoding="utf-8")

    provider = _ScriptedProvider(
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("read", source=str(target))],
        ),
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("file_write", path=str(target), contents="first-write\n")],
        ),
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("file_write", path=str(target), contents="second-write\n")],
        ),
        ProviderResponse(text="I need to re-read.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "read the notes file then overwrite it twice")

    assert _codes(_calls(mp, "file_write")) == ["status=success", "code=read-required"]
    assert target.read_text(encoding="utf-8") == "first-write\n"


# ── Case 10: the two bypasses, asserted directly ─────────────────────────────


def test_guard_bypasses_when_the_act_trail_cannot_be_inspected(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """A guard that cannot see the evidence must not invent a refusal.

    Neither bypass is reachable from a driven turn — the guard only runs inside
    one — so both are asserted on the function. The second uses a REAL processor
    that has not begun a turn: that is exactly the "no live turn" state, not a
    stand-in for it."""
    assert db is not None
    target = tmp_path / "notes.txt"
    target.write_text(_EDITABLE_CONTENT, encoding="utf-8")

    # No processor at all — e.g. an ability invoked outside the ACT loop.
    assert read_guard(None, target, refuse_truncated=True) is None

    # A processor whose turn was never allocated: turn_id is the -1 sentinel
    # MessageProcessor.__init__ sets, and begin() would replace.
    inert = MessageProcessor(UserConfig(), raw_input="never begun")  # inert (I2)
    assert inert.turn_id == -1
    assert read_guard(inert, target, refuse_truncated=True) is None
