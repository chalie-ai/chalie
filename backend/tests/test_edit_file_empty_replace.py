# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test: ``edit_file`` deletes text via an empty ``replace``.

Deletion IS ``replace: ""``. Two independent layers rejected it, so the tool
structurally could not delete anything: ``DispatchService._prevalidate`` — the
ACTION_REQUIRED pre-gate, which runs FIRST and is therefore the layer that
fires in production — tested every required param with ``not params.get(p)``,
a truthiness test that cannot tell ``""`` from absent; and
``EditFileParamsBag._verbatim`` behind it. Both returned
``code=missing-params`` naming a parameter that WAS supplied, which misdirects
the model into re-sending the same correct call until the runaway guard kills
the turn.

The fix is an opt-OUT, not an opt-in: non-empty stays the global rule for every
required param of every ability, and ``Ability.ALLOW_EMPTY`` names the few
params that may legitimately be empty. An opted-out param is still REQUIRED TO
BE PRESENT — only its emptiness stops being a failure.

Cases 1-3 drive the whole turn; case 4 pins the pre-gate on its own, and it is
there for a measured reason. ``EditFileParamsBag`` re-checks the same three
params behind the pre-gate, so for THIS tool the two layers mask each other: an
implementation that relaxes the pre-gate to a bare presence test — dropping the
non-empty requirement for the required params of all the other abilities, whose
bags do not all re-check — still passes cases 1-3, verified by running them
against exactly that implementation. Only a direct assertion on
``_prevalidate`` catches it, and the pre-gate is a documented contract in its
own right ("a known action missing required params → a SINGLE
``code=missing-params`` error naming ALL missing params") that had no coverage
at all before this file.

Cases 1-3 drive the REAL production entry point end to end, like
``test_read_guard.py``: a real ``MessageProcessor`` turn
against the real SQLite DB (``db`` fixture), the real ``DispatchService`` (so
the pre-gate genuinely runs), the real ``edit_file`` ability, and a real file
on disk under ``tmp_path``. The only substitution is the LLM network boundary
(``services.provider_service.build_client``), scripted to replay one tool call
per step. ``edit_file`` carries a seeded ``allow`` policy row on the ``chat``
channel (``abilities/assets/policy_defaults.json``), so no policy seeding is
needed here.
"""

import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from abilities.edit_file import EditFileAbility
from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from models.provider_response import ProviderResponse
from models.tool_call import ToolCall

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

# ProviderService builds its thin transport client via this factory call — the
# real network boundary (see test_read_guard.py).
_BUILD_CLIENT = "services.provider_service.build_client"

# A span bracketed by keepers, so a deletion that removes too much (or the
# wrong span) is visible in the assertion rather than passing silently.
#
# The span deliberately carries NO leading or trailing whitespace: every string
# tool argument is passed through ``ProviderService.sanitize_args``, which ends
# in ``value.strip()``, so a ``search`` ending in ``\n`` reaches the ability
# without it. That is a SEPARATE defect (it defeats this bag's documented
# "kept VERBATIM — never stripped" contract and makes whole-line deletion
# impossible) and it is deliberately not entangled with this test.
_ORIGINAL_CONTENT = "keep-before[DELETE-ME]keep-after\n"
_AFTER_DELETION = "keep-beforekeep-after\n"


def _tool(name: str, **params: object) -> dict[str, object]:
    """A provider-shaped tool call: ``{"name": ..., "input": {...}}``."""
    return {"name": name, "input": params}


class _ScriptedProvider:
    """Replays one scripted ``ProviderResponse`` per ``send()`` call, in order.

    Past the end of the script it returns a benign terminal (no-tool, empty)
    response rather than raising: a completed ``user`` turn spawns a
    fire-and-forget skill-suggestion turn (``_proactive_suggestion``) that makes
    its own provider call on a daemon thread this test does not script for."""

    _TERMINAL = ProviderResponse(text="", model="scripted-empty-replace", tool_calls=None)

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


def _edits(mp: MessageProcessor) -> list[ToolCall]:
    return [c for c in mp.tool_call_service.by_turn() if c.tool_name == "edit_file"]


# ── Case 1: an empty replace deletes the matched text ─────────────────────────


def test_empty_replace_deletes_the_matched_text(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """``replace=""`` reaches the ability and performs the deletion: the call
    succeeds and the matched span is gone from the file on disk, with the
    surrounding lines byte-intact."""
    assert db is not None
    target = tmp_path / "notes.txt"
    target.write_text(_ORIGINAL_CONTENT, encoding="utf-8")

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
                "edit_file", path=str(target), search="[DELETE-ME]", replace="",
            )],
        ),
        ProviderResponse(text="Removed it.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "delete the deprecated span from the notes file")

    edits = _edits(mp)
    assert len(edits) == 1
    assert edits[0].state == ToolCall.DONE
    assert "status=success" in _open_tag(edits[0].result), edits[0].result
    assert "Replaced 1 occurrence" in edits[0].result

    assert target.read_text(encoding="utf-8") == _AFTER_DELETION


# ── Case 2: an ABSENT replace is still missing-params ─────────────────────────


def test_absent_replace_is_still_missing_params(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """Opting out of the non-empty rule does NOT make the param optional: with
    ``replace`` omitted entirely the pre-gate still refuses with
    code=missing-params, and the file is untouched."""
    assert db is not None
    target = tmp_path / "notes.txt"
    target.write_text(_ORIGINAL_CONTENT, encoding="utf-8")

    provider = _ScriptedProvider(
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("edit_file", path=str(target), search="[DELETE-ME]")],
        ),
        ProviderResponse(text="I need a replacement.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "edit the notes file without saying what to replace with")

    edits = _edits(mp)
    assert len(edits) == 1
    assert edits[0].state == ToolCall.ERROR
    assert "code=missing-params" in _open_tag(edits[0].result), edits[0].result
    assert "replace" in edits[0].result

    assert target.read_text(encoding="utf-8") == _ORIGINAL_CONTENT


# ── Case 3: a param that did NOT opt out is still guarded against empty ───────


def test_empty_search_is_still_missing_params(
    db: sqlite3.Connection, tmp_path: Path,
) -> None:
    """The non-empty rule is GLOBAL and only ``replace`` opted out of it.
    ``search=""`` — supplied but empty, and not in ``ALLOW_EMPTY`` — is still
    refused with code=missing-params.

    End to end this is enforced by the bag as well as the pre-gate; case 4 is
    what pins the pre-gate's own half of it."""
    assert db is not None
    target = tmp_path / "notes.txt"
    target.write_text(_ORIGINAL_CONTENT, encoding="utf-8")

    provider = _ScriptedProvider(
        ProviderResponse(
            text="", model="scripted",
            tool_calls=[_tool("edit_file", path=str(target), search="", replace="anything")],
        ),
        ProviderResponse(text="I need something to search for.", model="scripted", tool_calls=None),
    )

    mp = _run(provider, "edit the notes file with an empty search")

    edits = _edits(mp)
    assert len(edits) == 1
    assert edits[0].state == ToolCall.ERROR
    assert "code=missing-params" in _open_tag(edits[0].result), edits[0].result
    assert "search" in edits[0].result

    assert target.read_text(encoding="utf-8") == _ORIGINAL_CONTENT


# ── Case 4: the pre-gate's own contract, independent of the bag behind it ─────


def test_pregate_applies_non_empty_globally_and_opt_out_per_param(
    db: sqlite3.Connection,
) -> None:
    """``DispatchService._prevalidate`` — the ACTION_REQUIRED pre-gate — decides
    all three outcomes on its own, before the ParamBag is ever built.

    Asserted directly because the bag behind ``edit_file`` re-checks the same
    params and would mask a regression here (see the module docstring). The
    ability and the dispatcher are both the real production objects."""
    assert db is not None
    mp = MessageProcessor(UserConfig(), raw_input="pre-gate contract")  # inert (I2)
    gate = mp.dispatch_service._prevalidate  # noqa: SLF001 — the seam under test
    ability = EditFileAbility()
    base: dict[str, object] = {"path": "/tmp/x", "search": "needle", "replace": "thread"}

    # The opted-out param may be empty — this is the deletion call.
    assert gate(ability, {**base, "replace": ""}) is None

    # ...but it is still REQUIRED TO BE PRESENT.
    absent = gate(ability, {"path": "/tmp/x", "search": "needle"})
    assert absent is not None
    assert absent.code == "missing-params"
    assert "replace" in absent.body

    # A param that did NOT opt out is still refused when empty — the global
    # rule is untouched for every other ability's required params.
    empty = gate(ability, {**base, "search": ""})
    assert empty is not None
    assert empty.code == "missing-params"
    assert "search" in empty.body

    # Nothing missing, nothing empty → the gate stands aside.
    assert gate(ability, base) is None
