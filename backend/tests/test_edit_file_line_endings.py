# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test: ``edit_file`` preserves the file's original line endings.

Measured before the fix, not inferred: ``edit_file`` read its target with
``Path.read_text`` (default ``newline=None`` → universal newlines: every ``\r\n``
folded to ``\n`` on the way in) and persisted the result with ``write_text`` —
so a one-word edit of a CRLF file silently rewrote the line endings of EVERY
line in the file. The tool's summary promises "Every line outside the replaced
span is left byte-identical"; the bytes on disk said otherwise.

The fix (``services/file_text_service.py``) reads and writes with ``newline=""``
(no translation in either direction), matches the anchor on the LF-normalized
form — the same text the ``read`` tool's universal-newlines display shows the
model — and restores the file's detected ending before persisting. A file that
mixes ``\r\n`` and ``\n`` cannot be kept byte-identical under any single choice
of ending, so it is refused loudly (``code=mixed-line-endings``, file untouched)
rather than normalized. A file that is not valid UTF-8 is refused with
``code=decode-error`` — a clean error, never a stack trace: the ``read`` tool
decodes leniently (``errors='replace'``), so the model CAN read a file it must
not be able to edit.

Cases 1-5 drive the real production entry point end to end: a real
``MessageProcessor`` turn against the real SQLite DB (``db`` fixture), the real
``DispatchService``, the real ``read``/``edit_file`` abilities, and real files
under ``tmp_path``. The only substitution is the LLM network boundary
(``services.provider_service.build_client``), scripted to replay ``read``
(first — the read_guard requires it: the protocol, not scaffolding) and then
``edit_file``. ``edit_file`` and ``read`` both carry a seeded ``allow`` policy
row on the ``chat`` channel (``abilities/assets/policy_defaults.json``), so no
policy seeding is needed.

Every assertion is on the RAW BYTES (``Path.read_bytes``), never a decoded
string: a decoded comparison would hide exactly the byte-level corruption this
test exists to catch.
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
# real network boundary (see test_verbatim_param_values.py).
_BUILD_CLIENT = "services.provider_service.build_client"


def _tool(name: str, **params: object) -> dict[str, object]:
    """A provider-shaped tool call: ``{"name": ..., "input": {...}}``."""
    return {"name": name, "input": params}


class _ScriptedProvider:
    """Replays one scripted ``ProviderResponse`` per ``send()`` call, in order.

    Past the end of the script it returns a benign terminal (no-tool) response
    rather than raising: a completed ``user`` turn spawns a fire-and-forget
    skill-suggestion turn that makes its own provider call on a daemon thread
    this test does not script for. Its text is NON-empty on purpose — an empty
    completion is steered and then crashes the turn with EmptyCompletionLoop,
    which would fill this test's log with a crash traceback that has nothing to
    do with line endings."""

    _TERMINAL = ProviderResponse(text="Nothing further.", model="scripted-endings", tool_calls=None)

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


def _read_then_edit(target: Path, search: str, replace: str, raw_input: str) -> MessageProcessor:
    """The standard two-step dispatch: ``read`` then ``edit_file`` — the script
    replays exactly those two tool calls and a terminal reply."""
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
                search=search, replace=replace,
            )],
        ),
        ProviderResponse(text="Edited.", model="scripted", tool_calls=None),
    )
    return _run(provider, raw_input)


# ── Case 1: the measured failure — a CRLF file keeps every untouched \r\n ───


def test_crlf_file_only_the_span_changes(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """One-word anchor on line two of a CRLF file. The expected after-bytes are
    the before-bytes with ONLY ``beta`` → ``BETA`` — every untouched ``\\r\\n``
    intact. Asserted as exact byte equality of the WHOLE file."""
    assert db is not None
    target = tmp_path / "crlf.txt"
    target.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")

    mp = _read_then_edit(target, "beta", "BETA", "capitalize the middle word of the notes file")

    _assert_succeeded(mp, "edit_file")
    assert target.read_bytes() == b"alpha\r\nBETA\r\ngamma\r\n"


# ── Case 2: an LF file round-trips byte-identical (today's behaviour) ────────


def test_lf_file_only_the_span_changes(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """The same edit on an LF file — unchanged from today's behaviour: the
    after-bytes differ from the before-bytes in exactly the replaced span."""
    assert db is not None
    target = tmp_path / "lf.txt"
    target.write_bytes(b"alpha\nbeta\ngamma\n")

    mp = _read_then_edit(target, "beta", "BETA", "capitalize the middle word of the notes file")

    _assert_succeeded(mp, "edit_file")
    assert target.read_bytes() == b"alpha\nBETA\ngamma\n"


# ── Case 3: mixed endings are refused loudly, file left untouched ────────────


def test_mixed_line_endings_refused_file_unchanged(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """A file mixing ``\\r\\n`` and ``\\n`` cannot be restored byte-identical
    under any single choice of ending, so the edit is refused (the error code
    reaches the tool result) and the file's bytes are UNCHANGED."""
    assert db is not None
    target = tmp_path / "mixed.txt"
    before = b"alpha\r\nbeta\ngamma\r\n"
    target.write_bytes(before)

    mp = _read_then_edit(target, "beta", "BETA", "capitalize the middle word of the notes file")

    edits = _calls(mp, "edit_file")
    assert len(edits) == 1
    assert "code=mixed-line-endings" in _open_tag(edits[0].result), edits[0].result
    # Refused before any write — the file is not touched.
    assert target.read_bytes() == before


# ── Case 4: an undecodable file is refused with a clean error ────────────────


def test_undecodable_file_refused_with_clean_error(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """The ``read`` step must SUCCEED first — it decodes leniently
    (errors='replace'), which is the exact defect this case closes: the model
    can read a file it must not be able to edit. The edit then comes back
    ``code=decode-error`` — a clean, typed refusal, NOT the
    ``unhandled-exception`` catch-all that would mean an exception escaped the
    ability — and the file's bytes are UNCHANGED."""
    assert db is not None
    target = tmp_path / "binary.txt"
    before = b"alpha\r\n\xff\xfebeta\r\n"
    target.write_bytes(before)

    mp = _read_then_edit(target, "beta", "BETA", "capitalize the middle word of the notes file")
    # The read is the protocol step: it succeeded (lenient decode) before the
    # edit was refused on the strict decode.
    reads = _calls(mp, "read")
    assert len(reads) == 1
    assert "status=success" in _open_tag(reads[0].result), reads[0].result

    edits = _calls(mp, "edit_file")
    assert len(edits) == 1
    tag = _open_tag(edits[0].result)
    assert "code=decode-error" in tag, edits[0].result
    assert "code=unhandled-exception" not in tag, edits[0].result
    assert target.read_bytes() == before


# ── Case 5: a single-line file with no ending edits fine ─────────────────────


def test_single_line_file_without_ending_edits_fine(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """The ``none`` ending path: a file with no line ending at all has nothing
    to preserve — the edit lands and the bytes change only inside the span
    (the file still ends without a newline)."""
    assert db is not None
    target = tmp_path / "oneliner.txt"
    target.write_bytes(b"just one line")

    mp = _read_then_edit(target, "one", "two", "reword the single line of the notes file")

    _assert_succeeded(mp, "edit_file")
    assert target.read_bytes() == b"just two line"


# ── Case 6: params carrying literal \r\n are folded, never doubled ───────────


def test_crlf_params_with_literal_crlf_fold_not_double(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """A model that BELIEVES the file is CRLF may write literal ``\\r\\n``
    escapes into its own ``search``/``replace`` — ``read`` never shows it a
    ``\\r``, so those escapes are intent-for-a-newline, not bytes to honour.
    Both params are folded like the file content, so the match lands and
    ``restore`` writes the file's own ``\\r\\n`` — a naive passthrough would
    have doubled every param newline to ``\\r\\r\\n`` inside the span."""
    assert db is not None
    target = tmp_path / "windows.txt"
    target.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")

    mp = _read_then_edit(
        target, "beta\r\ngamma", "BETA\r\nGAMMA",
        "reword the middle and last lines of the notes file",
    )

    _assert_succeeded(mp, "edit_file")
    # No \r\r\n anywhere: the file stays clean CRLF, only the span changed.
    after = target.read_bytes()
    assert b"\r\r" not in after
    assert after == b"alpha\r\nBETA\r\nGAMMA\r\n"
