"""
EmailAbility — Read, search, draft, send, reply, forward, and manage emails.

Delegates to MailCapability's IMAP/SMTP handler. The capability is lazy-loaded
inside execute() to avoid circular imports and to handle the case where the
capability is not yet connected.

Send, reply, and forward require SMTP to be configured. Reply and forward
additionally require that email.read was called on the target UID in the
current ACT turn.
"""

import json
import logging
from abilities._base import Ability

from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)
LOG_PREFIX = "[EMAIL ABILITY]"


class EmailAbility(Ability):
    NAME = "email"
    SEARCH_TOOLTIP = "email inbox and sending"
    SUMMARY = (
        "Read, search, draft, send, reply, forward, and manage emails via the connected "
        "mail account. Available when the user asks to check, find, compose, send, "
        "reply to, forward, or manage email."
    )
    EXAMPLES = [
        "check my email",
        "search emails from John",
        "do I have any unread messages",
        "read that email from Sarah",
        "draft an email to the team about the meeting",
        "send an email to John about the meeting",
        "reply to Alice's email saying I agree",
        "forward the newsletter to the marketing team",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "read", "draft", "manage", "send", "reply", "forward"],
                "description": (
                    "The email action to perform. "
                    "search — find emails by sender, subject, keyword, date range, triage, or unanswered flag. "
                    "read — fetch the full body of an email by its IMAP UID. "
                    "draft — compose an email and save to Drafts (requires to, subject, body). "
                    "manage — delete, mark as read, mark as important, or move an email to spam (requires uid, operation). "
                    "send — send an email via SMTP (requires to, subject, body). "
                    "reply — reply to an email by UID; must call read on the same UID first (requires uid, body). "
                    "forward — forward an email by UID to a new recipient; must call read on the same UID first (requires uid, to)."
                ),
            },
            "sender": {
                "type": "string",
                "description": "search: filter by sender email address or name.",
            },
            "subject": {
                "type": "string",
                "description": "search: filter by subject keyword. draft/send: email subject line.",
            },
            "keyword": {
                "type": "string",
                "description": "search: full-text keyword search across email body and subject.",
            },
            "date_from": {
                "type": "string",
                "description": "search: ISO date lower bound (YYYY-MM-DD).",
            },
            "date_to": {
                "type": "string",
                "description": "search: ISO date upper bound (YYYY-MM-DD).",
            },
            "triage": {
                "type": "string",
                "enum": ["actionable", "informational", "noise"],
                "description": "search: filter by triage category.",
            },
            "unanswered": {
                "type": "boolean",
                "description": "search: when true, return only unanswered emails.",
            },
            "limit": {
                "type": "integer",
                "description": "search: maximum number of results (default 20, max 100).",
            },
            "uid": {
                "type": "integer",
                "description": "read/manage/reply/forward: IMAP UID of the email.",
            },
            "operation": {
                "type": "string",
                "enum": ["delete", "mark_read", "mark_important", "move_to_spam"],
                "description": "manage: the operation to perform on the email.",
            },
            "to": {
                "type": "string",
                "description": "draft/send/forward: recipient email address.",
            },
            "body": {
                "type": "string",
                "description": "draft/send/reply: plain-text email body.",
            },
            "cc": {
                "type": "string",
                "description": "send: CC recipient email address.",
            },
            "in_reply_to": {
                "type": "string",
                "description": (
                    "draft: Message-ID for threading this as a reply. "
                    "reply: auto-set from the original email — do not pass manually."
                ),
            },
        },
        "required": ["action"],
    }

    def run(self, params: dict) -> dict | str:
        action = params.get("action", "search").lower()

        from capabilities import load_capabilities
        caps = load_capabilities()
        cap = caps.get("mail")

        if cap is None or not cap.is_connected():
            result = {
                "status": "error",
                "error": "Mail capability not connected. Configure it in the Brain dashboard.",
            }
            return {"text": _skill_tag("email", json.dumps(result), action=action)}

        tool_map = {t["name"]: t["handler"] for t in cap.get_tools()}

        _ACTION_TO_HANDLER = {
            "search": "search_email",
            "read": "read_email",
            "draft": "draft_email",
            "manage": "manage_email",
            "send": "send_email",
            "reply": "reply_email",
            "forward": "forward_email",
        }

        handler_name = _ACTION_TO_HANDLER.get(action)
        if handler_name is None:
            result = {"status": "error", "error": f"Unknown email action: {action}"}
            return {"text": _skill_tag("email", json.dumps(result), action=action)}

        handler = tool_map.get(handler_name)
        if handler is None:
            result = {
                "status": "error",
                "error": (
                    f"Handler '{handler_name}' not available. "
                    "The required protocol may not be connected."
                ),
            }
            return {"text": _skill_tag("email", json.dumps(result), action=action)}

        from services.innate_skills._capability import dispatch_capability_handler
        result = dispatch_capability_handler(handler, params, self.telemetry)

        return {"text": _skill_tag("email", json.dumps(result), action=action)}
