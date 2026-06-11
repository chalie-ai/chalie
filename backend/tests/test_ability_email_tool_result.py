# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the email tool's ToolResult contract (TKT-889).

Real hot path, zero mocks: every assertion drives the genuine
``ToolDispatcher(mp).dispatch()`` chokepoint on the CHAT channel against a real
``mp``-shaped context, the real ``AbilityRegistry`` resolution of the production
``EmailAbility`` (now a ``CapabilityAbility`` subclass), the real
``PolicyManager.wrap`` gate (with the real ``policy`` table from the ``db``
fixture), the real ``EmailAbility.run`` and the real ``ActTrail`` write.

The mail capability is NOT connected in the test environment (no IMAP/SMTP
credentials) — that is the production reality this suite leans on. Every dispatch
either:

* trips the dispatcher's ACTION_REQUIRED pre-gate (``code=missing-params`` /
  ``code=unknown-action`` with a ``valid:`` ladder), which fires BEFORE the
  policy gate and before the capability is ever loaded; or
* trips the ability's own input validation (``code=invalid-recipient`` for a
  malformed ``to`` on an outward-facing action), which runs BEFORE the base's
  connected gate so the loud error fires regardless of SMTP state; or
* reaches the base's ``code=not-connected`` surface (capability loaded, refuses
  because no credentials).

What TKT-889 changes, exercised end to end:

* **Schema shrink:** ``in_reply_to`` — the self-contradictory "do not pass
  manually" param — is GONE from ``get_parameters()``. Threading is auto-resolved
  inside ``reply`` by the capability (it reads the replied-to message's
  Message-ID and wires In-Reply-To/References itself).
* **Foundation:** ``EmailAbility`` is a ``CapabilityAbility`` subclass — the
  not-connected / unknown-action / handler-dispatch flow comes from the base.
* **Contract:** every action returns a ``ToolResult``; errors carry stable kebab
  codes (NOT the ``code="error"`` mechanical-wrap placeholder) + hints + a
  ``valid:`` ladder where applicable.
* **Guardrail:** outward-facing sends validate recipient shape pre-send with a
  stable ``code=invalid-recipient`` and a hint.
"""

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig
from services.act_trail import ActTrail

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


# ── Foundation: EmailAbility is a CapabilityAbility ─────────────────────────────


def test_email_is_a_capability_ability():
    """The ability is migrated onto the shared ``CapabilityAbility`` base — the
    hand-rolled capability-delegation block is gone."""
    from abilities._capability import CapabilityAbility
    from abilities._registry import AbilityRegistry

    ability = AbilityRegistry.get("email")
    assert isinstance(ability, CapabilityAbility)
    assert ability.CAPABILITY_KEY == "mail"


# ── Schema honesty: in_reply_to is gone ─────────────────────────────────────────


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


# ── Error ladder: unknown action (ACTION_REQUIRED pre-gate) ─────────────────────


def test_unknown_action_lists_real_actions_in_valid(db, chat_mp):
    """An unknown action is pre-gated into ONE ``code=unknown-action`` error whose
    ``valid:`` line names the REAL actions (NOT the placeholder)."""
    out = ToolDispatcher(chat_mp).dispatch(
        "email", {"action": "teleport", "act_summary": "x"}
    )

    assert "[email(status=error, code=unknown-action" in out
    assert "code=error]" not in out
    assert "valid:" in out
    for action in ("search", "read", "draft", "manage", "send", "reply", "forward"):
        assert action in out


# ── Error ladder: missing required params per action (pre-gate) ─────────────────


def test_send_without_recipient_reports_missing_params(db, chat_mp):
    """``send`` with no ``to``/``subject``/``body`` is pre-gated into a single
    ``code=missing-params`` error naming the missing params — BEFORE the policy
    gate and before the capability is ever loaded."""
    out = ToolDispatcher(chat_mp).dispatch(
        "email", {"action": "send", "act_summary": "x"}
    )

    assert "[email(status=error, code=missing-params" in out
    assert "code=error]" not in out
    assert "to" in out and "subject" in out and "body" in out


def test_read_without_uid_reports_missing_params(db, chat_mp):
    """``read`` with no ``uid`` is pre-gated into ``code=missing-params``."""
    out = ToolDispatcher(chat_mp).dispatch(
        "email", {"action": "read", "act_summary": "x"}
    )

    assert "[email(status=error, code=missing-params" in out
    assert "uid" in out


def test_manage_without_operation_reports_missing_params(db, chat_mp):
    """``manage`` with a uid but no ``operation`` is pre-gated into
    ``code=missing-params`` naming ``operation``."""
    out = ToolDispatcher(chat_mp).dispatch(
        "email", {"action": "manage", "uid": 42, "act_summary": "x"}
    )

    assert "[email(status=error, code=missing-params" in out
    assert "operation" in out


def test_forward_without_recipient_reports_missing_params(db, chat_mp):
    """``forward`` with a uid but no ``to`` is pre-gated into
    ``code=missing-params`` naming ``to``."""
    out = ToolDispatcher(chat_mp).dispatch(
        "email", {"action": "forward", "uid": 7, "act_summary": "x"}
    )

    assert "[email(status=error, code=missing-params" in out
    assert "to" in out


def test_reply_without_body_reports_missing_params(db, chat_mp):
    """``reply`` needs ``uid`` + ``body``; with only a uid it is pre-gated into
    ``code=missing-params`` naming ``body``."""
    out = ToolDispatcher(chat_mp).dispatch(
        "email", {"action": "reply", "uid": 9, "act_summary": "x"}
    )

    assert "[email(status=error, code=missing-params" in out
    assert "body" in out


# ── Guardrail: recipient shape validated pre-send (before connected gate) ───────


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


# ── Not-connected surface per action family (capability loaded, refuses) ────────


def test_search_not_connected_errors_with_hint(db, chat_mp):
    """``search`` (a valid action needing no params) passes the pre-gate, loads the
    capability, and surfaces the base's stable ``code=not-connected`` with a hint
    naming the mail integration — NOT the legacy JSON ``{status: error}`` body and
    NOT the ``code="error"`` placeholder."""
    out = ToolDispatcher(chat_mp).dispatch(
        "email", {"action": "search", "act_summary": "x"}
    )

    assert "[email(status=error, code=not-connected" in out
    assert "code=error]" not in out
    assert "not connected" in out.lower()
    assert "hint:" in out
    assert "mail integration" in out.lower()


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


def test_reply_valid_not_connected_errors_cleanly(db, chat_mp):
    """``reply`` with uid + body (no recipient to validate — threading is auto
    resolved) passes the pre-gate and surfaces the base's ``code=not-connected``.
    The model never supplies threading headers."""
    out = ToolDispatcher(chat_mp).dispatch(
        "email",
        {"action": "reply", "uid": 11, "body": "I agree", "act_summary": "x"},
    )

    assert "[email(status=error, code=not-connected" in out
    assert "code=error]" not in out


def test_read_valid_uid_not_connected_errors_cleanly(db, chat_mp):
    """``read`` with a uid passes the pre-gate and surfaces ``code=not-connected``."""
    out = ToolDispatcher(chat_mp).dispatch(
        "email", {"action": "read", "uid": 5, "act_summary": "x"}
    )

    assert "[email(status=error, code=not-connected" in out
    assert "code=error]" not in out


def test_manage_valid_not_connected_errors_cleanly(db, chat_mp):
    """``manage`` with uid + operation passes the pre-gate and surfaces
    ``code=not-connected``."""
    out = ToolDispatcher(chat_mp).dispatch(
        "email",
        {"action": "manage", "uid": 6, "operation": "mark_read", "act_summary": "x"},
    )

    assert "[email(status=error, code=not-connected" in out
    assert "code=error]" not in out


# ── Act-trail records the rendered envelope ─────────────────────────────────────


def test_act_trail_records_the_envelope(db, chat_mp):
    """The act-trail records the same non-empty email envelope against the
    transcript anchor — the cross-step write really lands."""
    ToolDispatcher(chat_mp).dispatch(
        "email", {"action": "search", "act_summary": "x"}
    )
    trail = ActTrail().fetch_by_transcript_id(chat_mp.uid)
    assert any("[email(status=" in row["result"] for row in trail)
