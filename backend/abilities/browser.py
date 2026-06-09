"""BrowserAbility — drive one persistent web page with nine flat verbs.

The model thinks as little as possible: it names WHAT to act on by visible text
("Sign in", "Email") and the code resolves WHERE mechanically
(tools/browser/session.PageSession). One page persists across calls within a
delegate run (key = the invoking mp's transcript uid); every call returns the
same JSON envelope with a mechanical `changed` diff. Screenshots ingest through
the document pipeline into a screenshots/ subdir and surface as doc_ids the
`vision` tool can read.
"""

import json
import logging
import os
from secrets import token_hex
from typing import ClassVar
from urllib.parse import urlparse

from abilities._ability import Ability
from tools.browser.session import error_envelope, record_screenshot, run_verb

logger = logging.getLogger(__name__)


class BrowserAbility(Ability):
    def get_name(self) -> str:
        return "browser"

    def get_summary(self) -> str:
        return (
            "Drive a web page step by step: open a URL, read it, find text on it, "
            "click buttons and links by their visible label, fill and submit forms, "
            "scroll, go back, and capture screenshots into the document store. The "
            "page stays open between calls; every action returns the same JSON "
            "envelope describing the page and what just changed."
        )

    def get_examples(self) -> list[str]:
        return [
            "open this webpage and tell me what it says",
            "click the Accept cookies button on this site",
            "fill in the search box and press the search button",
            "find the price on this product page",
            "take a screenshot of this page",
            "choose a country from the dropdown on this form",
            "scroll down and read the rest of the article",
            "go back to the previous page",
        ]

    def get_search_tooltip(self) -> str:
        return "interactive web browser"

    _PARAMETERS: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "open", "read", "find", "click", "fill",
                    "select", "scroll", "back", "screenshot",
                ],
                "description": (
                    "What to do. 'open' a URL first; every other action works on "
                    "the page that is already open."
                ),
            },
            "url": {
                "type": "string",
                "description": "Absolute http(s) URL. Only used by 'open'.",
            },
            "target": {
                "type": "string",
                "description": (
                    "Visible text of the element to act on — a button or link "
                    "label, or a form field's label/placeholder (e.g. 'Sign in', "
                    "'Email'). Used by click/fill/select."
                ),
            },
            "value": {
                "type": "string",
                "description": "Text to type ('fill') or option to choose ('select').",
            },
            "query": {
                "type": "string",
                "description": "Text to search for on the page. Only used by 'find'.",
            },
            "section": {
                "type": "string",
                "description": "Optional heading text — 'read' returns only that section.",
            },
            "direction": {
                "type": "string",
                "enum": ["down", "up"],
                "description": "Scroll direction. Only used by 'scroll'.",
            },
        },
        "required": ["action"],
    }

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    # verb -> (forwarded params, required params)
    _VERBS: ClassVar[dict] = {
        "open": (("url",), ("url",)),
        "read": (("section",), ()),
        "find": (("query",), ("query",)),
        "click": (("target",), ("target",)),
        "fill": (("target", "value"), ("target", "value")),
        "select": (("target", "value"), ("target", "value")),
        "scroll": (("direction",), ("direction",)),
        "back": ((), ()),
        "screenshot": ((), ()),
    }

    def run(self, params: dict) -> dict:
        action = (params.get("action") or "").strip().lower()
        if action not in self._VERBS:
            return self._reply(error_envelope(
                f"Unknown action {action!r}. Actions: {', '.join(self._VERBS)}."
            ))
        forwarded, required = self._VERBS[action]
        missing = [k for k in required if not str(params.get(k) or "").strip()]
        if missing:
            return self._reply(error_envelope(
                f"'{action}' requires {', '.join(missing)}."
            ))
        if action == "open":
            from tools.browser.security import validate_url  # noqa: PLC0415
            ok, reason = validate_url(str(params["url"]).strip())
            if not ok:
                return self._reply(error_envelope(f"URL blocked: {reason}"))
        if action == "screenshot":
            return self._screenshot()
        kwargs = {k: str(params[k]).strip() for k in forwarded if str(params.get(k) or "").strip()}
        return self._reply(self._run_verb(action, kwargs))

    def _run_verb(self, verb: str, kwargs: dict) -> dict:
        try:
            return run_verb(self._session_key(), verb, kwargs)
        except TimeoutError:
            return error_envelope("Browser operation timed out (90s).")
        except RuntimeError as e:
            if "chromium" in str(e).lower() or "executable" in str(e).lower():
                return error_envelope(f"Browser unavailable: {e}")
            return error_envelope(str(e)[:300])
        except Exception as e:
            logger.exception("[BROWSER] %s failed", verb)
            return error_envelope(f"Browser error: {str(e)[:300]}")

    def _screenshot(self) -> dict:
        grabbed = self._run_verb("grab", {})
        if grabbed.get("error"):
            return self._reply(grabbed)

        from abilities.document import ingest_file  # noqa: PLC0415
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        from services.document_service import DocumentService  # noqa: PLC0415
        from services.tmp_storage import new_tmp_path  # noqa: PLC0415

        page = grabbed["page"]
        host = urlparse(page["url"]).hostname or "page"
        tmp = new_tmp_path(f"{token_hex(4)}.png")
        try:
            with open(tmp, "wb") as fh:
                fh.write(grabbed["png"])
            ingested = ingest_file(
                DocumentService(get_shared_db_service()),
                tmp,
                name=f"screenshot-{host}.png",
                subdir="screenshots",
                source_type="screenshot",
            )
        finally:
            if os.path.isfile(tmp):
                os.remove(tmp)
        if ingested.get("error"):
            return self._reply(error_envelope(f"Screenshot ingest failed: {ingested['error']}"))

        record_screenshot(self._session_key(), ingested["id"], page["url"])
        return self._reply({
            "page": page,
            "data": {
                "doc_id": ingested["id"],
                "name": ingested["name"],
                "status": ingested["status"],
                "note": "Use the vision tool with image=<doc_id> to see this screenshot.",
            },
            "changed": {"navigated": False, "dialog": None, "popup": None, "summary": ""},
            "error": None,
        })

    def _session_key(self) -> int:
        return getattr(self.mp, "uid", None) or 0

    @staticmethod
    def _reply(envelope: dict) -> dict:
        return {
            "status": "error" if envelope.get("error") else "success",
            "result": json.dumps(envelope, ensure_ascii=False),
        }
