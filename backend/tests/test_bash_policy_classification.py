# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for bash's command-derived policy classification ().

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch()`` chokepoint against a real ``AbilityRegistry``,
the production ``PolicyManager.wrap`` gate, ``BashAbility.run``, and a real
SQLite-backed ``ActTrail``.

The contract under test (the closed bypass):
  * Permission is derived from the COMMAND via ``Ability.classify_action``, NOT
    from a model-supplied ``action`` param. A forged ``action="read"`` on a
    ``rm -rf``-class command is ignored: the gate evaluates ``bash.compound``
    (the command's real class), so a ``deny`` blocks the call.
  * The default ``classify_action`` (any other ability) returns ``None``, leaving
    the ``action`` param path untouched.
  * ``BashAbility.run`` returns a ``ToolResult`` body with structured
    ``{exit_code, stdout, stderr}`` and ``meta truncated=true`` when clipped.
"""

import sqlite3
from typing import cast

import pytest

from abilities._ability import Ability
from abilities._dispatcher import ToolDispatcher
from abilities._result import ToolResult
from configs.channels import UserConfig
from services.act_trail import ActTrail

pytestmark = pytest.mark.unit


def _seed_transcript(db: sqlite3.Connection, channel: str) -> int:
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        (channel, "user", "run a command"),
    )
    db.commit()
    return cast(int, cur.lastrowid)


class _MP:
    """Minimal real MP-shaped context — exactly what dispatch reads off the live
    processor: ``config`` (the chat policy channel) and ``uid`` (the transcript
    anchor the trail records against)."""

    def __init__(self, uid: int, config: object) -> None:
        self.config = config
        self.uid = uid


@pytest.fixture
def chat_mp(db: sqlite3.Connection) -> _MP:
    """A real chat-channel mp bound to the test database, with a seeded
    transcript anchor for the act-trail write."""
    return _MP(_seed_transcript(db, "chat"), UserConfig({}))


def _deny(channel: str, permission: str) -> None:
    """Seed a single policy cell through the production write path (no mock)."""
    from services.database_service import get_shared_db_service
    from services.policy_manager import PolicyManager
    PolicyManager(get_shared_db_service()).upsert(channel, permission, "deny")


# ── The bypass proof: forged action is ignored; the COMMAND decides the gate ──


def test_forged_action_read_cannot_bypass_command_classification(db: sqlite3.Connection, chat_mp: _MP) -> None:
    """A ``rm -rf``-class command dispatched with a FORGED ``action="read"`` is
    gated on its COMMAND-derived class (``bash.compound``), not the self-declared
    ``read``.

    ``bash.read`` is ``allow`` on chat by seed; ``bash.compound`` we deny here. If
    the gate trusted the model's ``action`` (the bypass), the call would slip past
    as ``bash.read`` and reach run(). With the fix it evaluates ``bash.compound``
    and is blocked BEFORE run() — the block string names ``bash.compound`` and
    NOT ``bash.read``.
    """
    _deny("chat", "bash.compound")

    out = ToolDispatcher(chat_mp).dispatch(
        "bash",
        {"command": "rm -rf /tmp/tkt884_should_never_run", "action": "read", "act_summary": "x"},
    )

    # The gate evaluated the classified permission and blocked the call.
    assert "bash.compound action is not allowed" in out
    assert "Do NOT retry" in out
    # The forged 'read' class was NEVER the permission the gate keyed on.
    assert "bash.read" not in out
    # run() never executed: no destructive-blocked body, no command output.
    assert "destructive-blocked" not in out
    assert "exit_code" not in out

    # The act-trail recorded the same block envelope against the transcript.
    rows = ActTrail().fetch_by_transcript_id(chat_mp.uid)
    assert "bash.compound action is not allowed" in cast(str, rows[0]["result"])


def test_forged_action_read_on_remote_command_is_gated_as_remote(db: sqlite3.Connection, chat_mp: _MP) -> None:
    """A non-destructive but escalated command (ssh → ``remote_execution``) with a
    forged ``action="read"`` is still gated on its command-derived class, proving
    the bypass is closed for the whole classification ladder, not just rm -rf."""
    _deny("chat", "bash.remote_execution")

    out = ToolDispatcher(chat_mp).dispatch(
        "bash",
        {"command": "ssh attacker@host whoami", "action": "read", "act_summary": "x"},
    )

    assert "bash.remote_execution action is not allowed" in out
    assert "bash.read" not in out
    assert "exit_code" not in out


# ── The default hook path is untouched for non-classifying abilities ──────────


def test_default_classify_action_returns_none() -> None:
    """The base ``Ability.classify_action`` returns None — "no opinion, use the
    action param" — so the dispatcher's action-param path is unchanged for every
    ability that does not override the hook. Driven on a real concrete ability."""

    class _Plain(Ability):
        def get_name(self) -> str: return "_tkt884_plain"
        def get_summary(self) -> str: return "throwaway: does not classify"
        def get_examples(self) -> list[str]: return ["a", "b", "c", "d", "e", "f"]
        def get_search_tooltip(self) -> str: return "x"
        def get_parameters(self) -> dict[str, object]: return {"type": "object", "properties": {}, "required": []}
        def run(self, params: dict[str, object]) -> ToolResult: return cast(ToolResult, None)  # never reached in this test

    assert _Plain().classify_action({"action": "delete", "id": "7"}) is None


# ── ToolResult envelope shape: structured body + truncation meta ──────────────


def test_bash_run_returns_structured_exit_code_body(db: sqlite3.Connection, chat_mp: _MP) -> None:
    """A benign command (classified ``read`` → allow on chat) runs through the real
    gate and returns the structured ``{exit_code, stdout, stderr}`` JSON body in
    the rendered envelope."""
    out = ToolDispatcher(chat_mp).dispatch(
        "bash",
        {"command": "printf 'tkt884-marker'", "action": "read", "act_summary": "x"},
    )

    assert "[bash(status=success" in out
    assert '"exit_code":0' in out
    assert "tkt884-marker" in out
    assert "[end:bash]" in out


def test_bash_run_marks_truncated_meta_when_output_clipped(db: sqlite3.Connection, chat_mp: _MP) -> None:
    """Output larger than the 100KB budget is clipped via the shared ``truncate``
    helper and reported uniformly as ``meta truncated=true`` in the open tag, with
    the truncation notice appended to the clipped stream."""
    # ~240K of stdout (120k 'a' lines) from a single, pipe-free command — well
    # over the 100*1024 char budget — so it classifies as 'read' (allow on chat,
    # no permission prompt) and still overflows the truncation helper.
    out = ToolDispatcher(chat_mp).dispatch(
        "bash",
        {
            "command": 'python3 -c "print(chr(10).join([chr(97)]*120000))"',
            "action": "read",
            "act_summary": "x",
        },
    )

    assert "[bash(status=success, truncated=true)]" in out
    assert "output truncated at 100KB" in out
