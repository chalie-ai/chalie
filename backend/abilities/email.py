"""
EmailAbility — Search, read, draft, send, reply, forward, and manage emails.

Delegates to MailCapability's IMAP/SMTP handler via the shared
:class:`CapabilityAbility` base, which owns capability loading, the
not-connected / unknown-action / handler-unavailable errors, handler dispatch,
and result wrapping. This ability supplies its metadata, the capability/action
wiring, the ACTION_REQUIRED pre-gate map, and one outward-facing guardrail.

Two design choices land here:

* **Threading is auto-resolved, never model-supplied.** ``reply`` wires
  In-Reply-To / References itself: the capability reads the replied-to message's
  Message-ID (by ``uid``) and threads the outgoing message. The schema therefore
  carries NO ``in_reply_to`` field — a param the model would have to hold but
  never set was pure cognitive load, and a model that invented one broke
  threading.
* **Recipient shape is validated pre-send, ahead of the connected gate.** The
  outward-facing actions (``send`` / ``draft`` / ``forward``) validate their
  ``to`` recipient through ``_validate_recipient`` BEFORE the base's connected
  gate, so a malformed address errors loudly with ``code=invalid-recipient`` even
  when SMTP is not wired up — the same validate-before-base-run ordering calendar
  uses for ``dtstart``.

Every action returns a :class:`ToolResult` built only via ``ok`` / ``err``; the
dispatcher renders the wire envelope. This ability never formats an envelope and
never imports the skill-tag formatter.
"""

import logging
import re
from typing import ClassVar, cast

from abilities._capability import CapabilityAbility
from configs.enums.param_key import Keys
from abilities._result import ToolResult, truncate
from contracts.params.capability_params_bag import CapabilityParamsBag

logger = logging.getLogger(__name__)
LOG_PREFIX = "[EMAIL ABILITY]"

# A pragmatic recipient-shape check: a single token containing one '@' with a
# non-empty local part and a dotted domain. Not RFC-5322-complete (the SMTP
# server is the real authority) — just enough to reject obvious garbage before a
# send is attempted, with a hint the model can self-correct from.
_RECIPIENT_RE = r"^[^@\s]+@(?=[^@\s]+\.[^@\s])[^@\s]++$"

# Outward-facing actions whose ``to`` recipient is validated for shape pre-send.
_RECIPIENT_ACTIONS = ("send", "draft", "forward")


class EmailAbility(CapabilityAbility):
    DISCOVERABLE: ClassVar[bool] = False  # pim-delegate-exclusive; pinned on PimConfig only
    NAME: ClassVar[str] = "email"
    CAPABILITY_KEY: ClassVar[str] = "mail"
    DEFAULT_ACTION: ClassVar[str] = "search"
    NOT_CONNECTED_HINT: ClassVar[str] = (
        "Configure the mail integration in the Brain dashboard."
    )
    # Maps the model-facing action onto the mail capability's handler name. The
    # base uses it to look up the handler and to build the valid= ladder for
    # unknown-action errors.
    ACTION_HANDLERS: ClassVar[dict[str, str]] = {
        "search": "search_email",
        "read": "read_email",
        "draft": "draft_email",
        "manage": "manage_email",
        "send": "send_email",
        "reply": "reply_email",
        "forward": "forward_email",
    }

    # Pre-gated by the dispatcher BEFORE run() (and before the policy gate): a
    # known action missing a required param yields one code=missing-params error
    # naming all missing params; an unknown action yields one code=unknown-action
    # whose valid= names every real action. search has no hard requirement (an
    # empty filter set lists the recent inbox).
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "search": (),
        "read": (Keys.uid,),
        "draft": (Keys.to, Keys.subject, Keys.body),
        "manage": (Keys.uid, Keys.operation),
        "send": (Keys.to, Keys.subject, Keys.body),
        "reply": (Keys.uid, Keys.body),
        "forward": (Keys.uid, Keys.to),
    }


    def get_summary(self) -> str:
        return (
            "Search, read, draft, send, reply, forward, and manage emails via the connected "
            "mail account. Available when the user asks to check, find, compose, send, "
            "reply to, forward, or manage email."
        )

    def get_examples(self) -> list[str]:
        return [
            "check my email",
            "search emails from John",
            "read that email from Sarah",
            "draft an email to the team about the meeting",
            "send an email to John about the meeting",
            "reply to that email saying I agree",
            "forward the newsletter to the marketing team",
            "mark that email as read",
        ]

    def get_search_tooltip(self) -> str:
        return "email inbox and sending"

    def get_follow_up(self, tr: ToolResult) -> str:
        """Point the model to read an email's full body by uid."""
        return "search results are snippets only, not the full message. If you need the complete body of an email before acting on it, call email again with action=read and that uid."

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {
                "type": "string",
                "enum": list(ACTION_HANDLERS),
                "description": (
                    "The email action to perform. "
                    "search — find emails by sender, subject, keyword, date range, triage, or unanswered flag (all optional). "
                    "read — fetch the full body of an email by its IMAP uid. "
                    "draft — compose an email and save to Drafts (needs to, subject, body). "
                    "manage — delete / mark read / mark important / move to spam (needs uid, operation). "
                    "send — send an email (needs to, subject, body). "
                    "reply — reply to an email by uid; threading is wired automatically (needs uid, body). "
                    "forward — forward an email by uid to a new recipient (needs uid, to)."
                ),
            },
            Keys.uid: {
                "type": "integer",
                "description": "read / manage / reply / forward: IMAP uid of the target email.",
            },
            Keys.to: {
                "type": "string",
                "description": "send / draft / forward: recipient email address.",
            },
            Keys.subject: {
                "type": "string",
                "description": "send / draft: subject line. search: filter by subject keyword.",
            },
            Keys.body: {
                "type": "string",
                "description": "send / draft / reply: plain-text email body.",
            },
            Keys.operation: {
                "type": "string",
                "enum": ["delete", "mark_read", "mark_important", "move_to_spam"],
                "description": "manage: the operation to perform on the email.",
            },
            Keys.sender: {
                "type": "string",
                "description": "search: filter by sender email address or name.",
            },
            Keys.keyword: {
                "type": "string",
                "description": "search: full-text keyword across email body and subject.",
            },
            Keys.date_from: {
                "type": "string",
                "description": "search: ISO date lower bound (YYYY-MM-DD).",
            },
            Keys.date_to: {
                "type": "string",
                "description": "search: ISO date upper bound (YYYY-MM-DD).",
            },
            Keys.triage: {
                "type": "string",
                "enum": ["actionable", "informational", "noise"],
                "description": "search: filter by triage category.",
            },
            Keys.unanswered: {
                "type": "boolean",
                "description": "search: when true, return only unanswered emails.",
            },
            Keys.cc: {
                "type": "string",
                "description": "send: CC recipient email address.",
            },
        },
        "required": [Keys.action],
    }

    # ── Entry point — guardrail BEFORE the base's connected gate ────────────────

    def run(self, params: CapabilityParamsBag) -> ToolResult:
        action = self._resolve_action(params)

        if action in _RECIPIENT_ACTIONS:
            err = _validate_recipient(params.extra)
            if err is not None:
                return err

        result = self._dispatch(action, dict(params.extra))
        if result.status != "success" or not isinstance(result.body, dict):
            # not-connected / unknown-action / handler-unavailable already carry a
            # canonical code from the base — surface them untouched.
            return result
        return _shape_result(action, result.body)


# ---------------------------------------------------------------------------
# Result shaping — a trimmed, model-friendly body per action family
# ---------------------------------------------------------------------------

# Cap on the read body so a giant email cannot blow the context window; the
# shared truncate primitive reports the clip as meta truncated=true.
_READ_BODY_LIMIT = 4000


def _shape_result(action: str, body: dict[str, object]) -> ToolResult:
    """Reshape a successful handler dict into the contract body for *action*.

    A handler that surfaced its own ``error`` key (a backend failure, NOT a bad
    input) is passed through as the success body unchanged — the contract reserves
    ``err()`` for anticipated input failures; the dispatcher wraps the unexpected."""
    if "error" in body:
        return ToolResult.ok(body, action=action)

    if action == "search":
        rows = [
            {
                "id": e.get("uid"),
                "from": e.get("from_name") or e.get("from_addr", ""),
                "subject": e.get("subject", ""),
                "date": e.get("date", ""),
                "snippet": e.get("snippet", ""),
            }
            for e in cast(list[dict[str, object]], body.get("emails", []))
        ]
        return ToolResult.ok({"emails": rows}, action=action, count=len(rows))

    if action == "read":
        clipped, truncated = truncate(str(body.get("body", "")), _READ_BODY_LIMIT)
        message = {
            "uid": body.get("uid"),
            "from": body.get("from_name") or body.get("from_addr", ""),
            "subject": body.get("subject", ""),
            "date": body.get("date", ""),
            "body": clipped,
        }
        return ToolResult.ok(message, action=action, truncated=truncated)

    if action in _RECIPIENT_ACTIONS or action == "reply":
        envelope = {
            "to": body.get("to", ""),
            "subject": body.get("subject", ""),
        }
        message_id = body.get("message_id")
        if message_id:
            envelope["message_id"] = message_id
        return ToolResult.ok(envelope, action=action)

    # manage and any future read-only action: echo the handler body.
    return ToolResult.ok(body, action=action)


# ---------------------------------------------------------------------------
# Guardrail — recipient shape validated pre-send
# ---------------------------------------------------------------------------

def _validate_recipient(params: dict[str, object]) -> ToolResult | None:
    """Runs ahead of the base's connected gate so the error fires regardless of SMTP state."""
    to = cast(str, params.get(Keys.to) or "").strip()
    if not to:
        return None
    if not re.match(_RECIPIENT_RE, to):
        return ToolResult.err(
            f"{to!r} does not look like an email address.",
            code="invalid-recipient",
            hint="pass a recipient like 'name@example.com'.",
        )
    return None
