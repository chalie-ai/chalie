"""BrowserAbility — drive one persistent web page with nine flat verbs.

The model thinks as little as possible: it names WHAT to act on by visible text
("Sign in", "Email") and the code resolves WHERE mechanically
(tools/browser/session.PageSession). One page persists across calls within a
delegate run (key = the invoking mp's transcript uid); every successful call
returns the same JSON envelope DICT with a mechanical `changed` diff (the
dispatcher renders it as compact JSON — never a pre-dumped JSON string).
Screenshots ingest through the document pipeline into a screenshots/ subdir and
surface as doc_ids the `vision` tool can read.

Every result is an :class:`abilities._result.ToolResult` built only via
``ok()`` / ``err()``. Errors carry a STABLE KEBAB-CASE ``code`` (never the
``code="error"`` placeholder) plus a one-line ``hint`` so a weak model can
self-correct without re-reading the schema. Unknown actions and missing required
params are caught by the dispatcher's ACTION_REQUIRED pre-gate BEFORE the policy
gate and BEFORE ``run()`` — so they never reach this module.
"""

import logging
import os
from secrets import token_hex
from typing import ClassVar
from urllib.parse import urlparse

from abilities._ability import Ability
from abilities._params import Keys
from abilities._result import ToolResult
from tools.browser.session import record_screenshot, run_verb

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
            Keys.action: {
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
            Keys.url: {
                "type": "string",
                "description": "Absolute http(s) URL. Only used by 'open'.",
            },
            Keys.target: {
                "type": "string",
                "description": (
                    "Visible text of the element to act on — a button or link "
                    "label, or a form field's label/placeholder (e.g. 'Sign in', "
                    "'Email'). Used by click/fill/select."
                ),
            },
            Keys.value: {
                "type": "string",
                "description": "Text to type ('fill') or option to choose ('select').",
            },
            Keys.query: {
                "type": "string",
                "description": "Text to search for on the page. Only used by 'find'.",
            },
            Keys.section: {
                "type": "string",
                "description": "Optional heading text — 'read' returns only that section.",
            },
            Keys.direction: {
                "type": "string",
                "enum": ["down", "up"],
                "description": "Scroll direction. Only used by 'scroll'.",
            },
        },
        "required": [Keys.action],
    }

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    # Required params per action — consumed by the dispatcher's ACTION_REQUIRED
    # pre-gate (BEFORE the policy gate and BEFORE run()): an unknown action →
    # code=unknown-action whose valid= names all nine verbs; a known action
    # missing a required param → one code=missing-params naming it. run() never
    # sees either case.
    ACTION_REQUIRED: ClassVar[dict] = {
        "open": (Keys.url,),
        "read": (),
        "find": (Keys.query,),
        "click": (Keys.target,),
        "fill": (Keys.target, Keys.value),
        "select": (Keys.target, Keys.value),
        "scroll": (Keys.direction,),
        "back": (),
        "screenshot": (),
    }

    # verb -> params forwarded to the session layer (the required half is the
    # dispatcher pre-gate's job, above).
    _FORWARDED: ClassVar[dict] = {
        "open": (Keys.url,),
        "read": (Keys.section,),
        "find": (Keys.query,),
        "click": (Keys.target,),
        "fill": (Keys.target, Keys.value),
        "select": (Keys.target, Keys.value),
        "scroll": (Keys.direction,),
        "back": (),
        "screenshot": (),
    }

    def run(self, params: dict) -> ToolResult:
        action = (params.get(Keys.action) or "").strip().lower()
        if action == "open":
            from tools.browser.security import validate_url  # noqa: PLC0415
            ok, reason = validate_url(str(params[Keys.url]).strip())
            if not ok:
                return ToolResult.err(
                    f"URL blocked: {reason}",
                    code="url-blocked",
                    hint="Only public http(s) URLs are reachable; private, "
                         "loopback, link-local and non-http(s) URLs are refused.",
                )
        if action == "screenshot":
            return self._screenshot()
        kwargs = {
            k: str(params[k]).strip()
            for k in self._FORWARDED[action]
            if str(params.get(k) or "").strip()
        }
        envelope = self._run_verb(action, kwargs)
        return self._reply(envelope)

    def _run_verb(self, verb: str, kwargs: dict) -> dict:
        try:
            return run_verb(self._session_key(), verb, kwargs)
        except TimeoutError:
            return {"_code": "browser-timeout",
                    "_hint": "Retry, or simplify the action (a busy page can take >90s).",
                    "error": "Browser operation timed out (90s)."}
        except RuntimeError as e:
            text = str(e)
            if "chromium" in text.lower() or "executable" in text.lower():
                return {"_code": "browser-unavailable",
                        "_hint": "The browser runtime is missing; this environment "
                                 "cannot launch a page.",
                        "error": f"Browser unavailable: {text}"}
            return {"_code": "browser-error",
                    "_hint": "Re-open the page and retry the action.",
                    "error": text[:300]}
        except Exception as e:  # noqa: BLE001
            logger.exception("[BROWSER] %s failed", verb)
            return {"_code": "browser-error",
                    "_hint": "Re-open the page and retry the action.",
                    "error": f"Browser error: {str(e)[:300]}"}

    def _screenshot(self) -> ToolResult:
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
            return ToolResult.err(
                f"Screenshot ingest failed: {ingested['error']}",
                code="screenshot-ingest-failed",
                hint="The page was captured but could not be stored; retry the "
                     "screenshot.",
                url=page["url"],
            )

        record_screenshot(self._session_key(), ingested["id"], page["url"])
        return ToolResult.ok({
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
    def _reply(envelope: dict) -> ToolResult:
        """A successful envelope (``error`` is None) becomes ``ok`` with the DICT
        body — the dispatcher renders compact JSON. An error envelope becomes
        ``err`` with the plain human-readable message string and a stable kebab
        ``code`` derived from the failure (carrying any internal ``_code`` /
        ``_hint`` set by ``_run_verb``, or classified from the message text);
        useful page context (url/title) rides in flat meta.
        """
        error = envelope.get("error")
        if not error:
            return ToolResult.ok(envelope)

        code = envelope.get("_code") or _classify_error(error)
        hint = envelope.get("_hint") or _HINTS.get(code)
        meta: dict = {}
        page = envelope.get("page") or {}
        if page.get("url"):
            meta["url"] = str(page["url"])
        if page.get("title"):
            meta["title"] = str(page["title"])[:120]
        return ToolResult.err(str(error), code=code, hint=hint, **meta)


def _classify_error(message: str) -> str:
    """The session layer (tools/browser/session.py) returns its failures as the
    envelope ``error`` string; this is the single place those strings are mapped
    to a self-correction code. Anything unrecognised falls back to the generic —
    but still stable, never ``code=error`` — ``browser-action-failed``.
    """
    lowered = message.lower()
    if "no page is open" in lowered:
        return "no-open-page"
    if "matches" in lowered and "element" in lowered:
        return "ambiguous-target"
    if "no element matching" in lowered:
        return "element-not-found"
    return "browser-action-failed"


_HINTS = {
    "no-open-page": "Call action='open' with a url before any other action.",
    "ambiguous-target": "Use the fuller visible text of the element, or "
                        "action='find' to list candidates.",
    "element-not-found": "Use action='find' with the same text to see what is "
                        "actually on the page.",
    "browser-action-failed": "Re-read the page with action='read' and retry.",
}
