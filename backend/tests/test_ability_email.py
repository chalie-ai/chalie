# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Email-specific business-logic tests migrated from the per-ability
conformance file removed in TKT-975. The full ToolResult wire contract is
pinned centrally in test_tool_result_contract.py; this file holds only the
email ability's genuine behaviour tests (in_reply_to schema removal, recipient
validation, not-connected surface) that have no coverage elsewhere."""

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig

from tests._tool_result_harness import MP, allow_policy, seed_transcript

pytestmark = pytest.mark.unit


def _allow_email_actions(db, channel: str = "chat") -> None:
    """Flip the REAL ``policy`` table so the gated email actions are ``allow`` on
    the channel — the same rows the real ``PolicyManager`` gate reads.

    ``send`` / ``reply`` / ``forward`` / ``manage`` ship as ``ask`` by seed
    (outward-facing / mutating), which on a headless test would park the real gate
    waiting for a human POST. Flipping the real policy row to ``allow`` (exactly
    what a user does when they pick "always allow") lets the gate pass through to
    the production ``run()`` so its recipient validation and the base's
    not-connected path actually execute. No mock — this is the production policy
    table driving the production gate. This loops EVERY email action (the harness
    ``allow_policy`` flips a single permission); that per-action breadth is why the
    local helper is kept."""
    for action in ("search", "read", "draft", "manage", "send", "reply", "forward"):
        allow_policy(db, f"email.{action}", channel)


@pytest.fixture
def chat_mp(db):
    """A real chat-channel mp bound to the test database, with a seeded transcript
    anchor for the act-trail write and every email action flipped to ``allow`` in
    the real policy table so the gate passes through to the production run()."""
    _allow_email_actions(db)
    return MP(seed_transcript(db, "chat", "check my email"), UserConfig({}))


def test_in_reply_to_absent_from_parameters():
    """``in_reply_to`` — the "do not pass manually" param — is REMOVED from the
    schema. Threading is auto-resolved inside the ability; the model never sees a
    field it must hold but never set."""
    from abilities._registry import AbilityRegistry

    schema = AbilityRegistry.get("email").get_parameters()
    props = schema["properties"]
    assert "in_reply_to" not in props
    # The action enum still names all seven real actions.
    enum = props["action"]["enum"]
    for action in ("search", "read", "draft", "manage", "send", "reply", "forward"):
        assert action in enum


def test_send_invalid_recipient_errors_before_connected_gate(db, chat_mp):
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


def test_forward_invalid_recipient_errors_invalid_recipient(db, chat_mp):
    """``forward`` validates its ``to`` recipient the same way."""
    out = ToolDispatcher(chat_mp).dispatch(
        "email",
        {"action": "forward", "uid": 3, "to": "garbage", "act_summary": "x"},
    )

    assert "[email(status=error, code=invalid-recipient" in out
    assert "code=error]" not in out


def test_draft_invalid_recipient_errors_invalid_recipient(db, chat_mp):
    """``draft`` is outward-shaped too (writes a Drafts message) — its ``to`` is
    validated for shape before the connected gate."""
    out = ToolDispatcher(chat_mp).dispatch(
        "email",
        {"action": "draft", "to": "@@bad", "subject": "x", "body": "y",
         "act_summary": "x"},
    )

    assert "[email(status=error, code=invalid-recipient" in out


def test_send_valid_recipient_not_connected_errors_cleanly(db, chat_mp):
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
