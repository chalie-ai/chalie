# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for email ability business logic migrated from test_tool_result_contract.py.

Covers: in_reply_to schema removal, recipient validation, and the not-connected surface."""

import sqlite3
from typing import cast

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from tests._tool_result_harness import MP, allow_policy, seed_transcript

pytestmark = pytest.mark.unit


def _allow_email_actions(db: sqlite3.Connection, channel: str = "chat") -> None:
    """Flip the REAL ``policy`` table so gated email actions are ``allow`` — prevents
    the real gate from parking on a human POST (outward-facing actions ship as ``ask``)."""
    for action in ("search", "read", "draft", "manage", "send", "reply", "forward"):
        allow_policy(db, f"email.{action}", channel)


@pytest.fixture
def chat_mp(db: sqlite3.Connection) -> MP:
    _allow_email_actions(db)
    return MP(seed_transcript(db, "chat", "check my email"), UserConfig({}))


def test_in_reply_to_absent_from_parameters() -> None:
    from abilities._registry import AbilityRegistry

    schema = AbilityRegistry.get("email").get_parameters()
    props = cast(dict[str, object], schema["properties"])
    assert "in_reply_to" not in props
    # The action enum still names all seven real actions.
    enum = cast(list[object], cast(dict[str, object], props["action"])["enum"])
    for action in ("search", "read", "draft", "manage", "send", "reply", "forward"):
        assert action in enum


def test_send_invalid_recipient_errors_before_connected_gate(db: sqlite3.Connection, chat_mp: MP) -> None:
    """A malformed ``to`` on ``send`` errors with a stable ``code=invalid-recipient``
    and a hint — BEFORE the capability-connected gate, so the loud error fires
    regardless of SMTP state (the same ordering calendar uses for dtstart)."""
    out = ToolDispatcher(chat_mp).dispatch(
        "email",
        {"action": "send", "to": "not-an-email", "subject": "Hi",
         "body": "hello", "act_summary": "x"},
    )

    assert "[email(status=error, code=invalid-recipient" in out
    assert "code=error]" not in out
    assert "hint:" in out
    # Not the not-connected error — validation ran first.
    assert "not-connected" not in out


def test_forward_invalid_recipient_errors_invalid_recipient(db: sqlite3.Connection, chat_mp: MP) -> None:
    """``forward`` validates its ``to`` recipient the same way."""
    out = ToolDispatcher(chat_mp).dispatch(
        "email",
        {"action": "forward", "uid": 3, "to": "garbage", "act_summary": "x"},
    )

    assert "[email(status=error, code=invalid-recipient" in out
    assert "code=error]" not in out


def test_draft_invalid_recipient_errors_invalid_recipient(db: sqlite3.Connection, chat_mp: MP) -> None:
    """``draft`` is outward-shaped too (writes a Drafts message) — its ``to`` is
    validated for shape before the connected gate."""
    out = ToolDispatcher(chat_mp).dispatch(
        "email",
        {"action": "draft", "to": "@@bad", "subject": "x", "body": "y",
         "act_summary": "x"},
    )

    assert "[email(status=error, code=invalid-recipient" in out


def test_send_valid_recipient_not_connected_errors_cleanly(db: sqlite3.Connection, chat_mp: MP) -> None:
    """A well-formed ``send`` (valid recipient, all params) passes validation and
    the pre-gate, then surfaces the base's ``code=not-connected`` — proving the
    guardrail does not block legitimate sends, only the connected gate does."""
    out = ToolDispatcher(chat_mp).dispatch(
        "email",
        {"action": "send", "to": "alex@example.com", "subject": "Hi",
         "body": "hello there", "act_summary": "x"},
    )

    assert "[email(status=error, code=not-connected" in out
    assert "code=error]" not in out
    assert "invalid-recipient" not in out
