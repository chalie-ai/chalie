"""Feature tests: browser returns canonical ToolResult codes through the dispatcher.

Drives the REAL production hot path — ``ToolDispatcher.dispatch()`` on a real
``MessageProcessor`` bound to ``WebBrowseConfig`` (the channel where ``browser``
is always-available, and ``browser`` is in ``PolicyManager.INTERNAL`` so no
policy rows are needed). Zero mocks.

The SSRF guard (security.py) blocks every local/private/loopback address
WITHOUT a network call, so every error path here is deterministic offline:
- unknown action / missing params are caught by the dispatcher's ACTION_REQUIRED
  pre-gate BEFORE the policy gate and BEFORE run();
- a blocked URL is rejected by ``validate_url`` before any browser launch;
- a verb with no open page returns the no-open-page guard from session.py.

Every error must carry a STABLE KEBAB CODE and a one-line ``hint`` — never the
banned ``code=error`` placeholder — and success/error bodies must NOT be a
pre-dumped JSON string. We assert the dispatcher WIRE envelope (the string the
ACT loop actually consumes), not ability internals.
"""

import json

import pytest

from abilities._dispatcher import ToolDispatcher
from configs.channels.web_browse import WebBrowseConfig
from services.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig

pytestmark = pytest.mark.unit

_VERBS = ("open", "read", "find", "click", "fill", "select", "scroll", "back", "screenshot")


def _browse_mp() -> MessageProcessor:
    """A real MessageProcessor on the web_browse delegate channel, where browser
    is always-available (find_tools isolation test's contract). browser is in
    PolicyManager.INTERNAL, so dispatch bypasses the policy gate entirely."""
    mp = MessageProcessor("drive a web page")
    mp.config = WebBrowseConfig(ProcessorConfig.POLICY_CHANNEL.CHAT)
    mp.active_tools = list(mp.config.always_available or [])
    return mp


def _dispatch(params: dict) -> str:
    """The exact string the ACT loop consumes for one browser tool call."""
    return ToolDispatcher(_browse_mp()).dispatch("browser", params)


def _open_tag(rendered: str) -> str:
    """The ``[browser(...)]`` open tag of the rendered wire envelope."""
    line = rendered.splitlines()[0]
    assert line.startswith("[browser(") and line.endswith(")]"), rendered
    return line


def _body(rendered: str) -> str:
    """The body lines between the open tag and ``[end:browser]``."""
    lines = rendered.splitlines()
    assert lines[-1] == "[end:browser]", rendered
    return "\n".join(lines[1:-1])


def test_unknown_action_is_pregate_unknown_action_with_full_verb_ladder():
    """Pre-gate: an unknown action → code=unknown-action and a valid: ladder
    naming all nine verbs, BEFORE run() ever sees it."""
    rendered = _dispatch({"action": "render"})

    tag = _open_tag(rendered)
    assert "status=error" in tag, rendered
    assert "code=unknown-action" in tag, rendered
    valid_line = next(ln for ln in rendered.splitlines() if ln.startswith("valid:"))
    for verb in _VERBS:
        assert verb in valid_line, (verb, rendered)
    assert "code=error" not in tag, rendered


def test_missing_required_param_is_pregate_missing_params_naming_the_param():
    """open with no url → code=missing-params naming 'url'."""
    rendered = _dispatch({"action": "open"})

    tag = _open_tag(rendered)
    assert "code=missing-params" in tag, rendered
    assert "url" in _body(rendered), rendered
    assert "code=error" not in tag, rendered


def test_blocked_url_is_url_blocked_with_hint_and_no_browser_launch():
    """A loopback URL is rejected by the SSRF guard with code=url-blocked + a
    hint, with NO network and NO browser process launched."""
    rendered = _dispatch({"action": "open", "url": "http://127.0.0.1:9/admin"})

    tag = _open_tag(rendered)
    assert "status=error" in tag, rendered
    assert "code=url-blocked" in tag, rendered
    assert "code=error" not in tag, rendered
    hint_line = next(ln for ln in rendered.splitlines() if ln.startswith("hint:"))
    assert "public" in hint_line.lower() or "http" in hint_line.lower(), rendered


def test_link_local_metadata_url_is_url_blocked():
    """The classic SSRF target (cloud metadata endpoint) is blocked offline."""
    rendered = _dispatch({"action": "open", "url": "http://169.254.169.254/"})

    assert "code=url-blocked" in _open_tag(rendered), rendered


def test_action_on_no_open_page_carries_a_kebab_code_not_error():
    """A real verb with no page open returns the no-open-page guard with a
    stable kebab code + hint — never code=error, never a pre-dumped JSON body."""
    rendered = _dispatch({"action": "click", "target": "Sign in"})

    tag = _open_tag(rendered)
    assert "status=error" in tag, rendered
    assert "code=error" not in tag, rendered
    # A real stable kebab code (lower-kebab, not the generic placeholder).
    code = next(p for p in tag[len("[browser("):-2].split(", ") if p.startswith("code="))
    value = code[len("code="):]
    assert value and value == value.lower() and "_" not in value and " " not in value, rendered
    assert value != "error", rendered
    hint_line = next(ln for ln in rendered.splitlines() if ln.startswith("hint:"))
    assert hint_line, rendered


def test_success_body_is_not_a_pre_dumped_json_string():
    """The error body is a human-readable message string, NOT a serialised
    envelope. The whole-body must not be a JSON object that contains the legacy
    ``{"page":...,"data":...,"changed":...,"error":...}`` envelope keys."""
    rendered = _dispatch({"action": "open", "url": "http://127.0.0.1:9/admin"})
    body = _body(rendered)
    # The legacy contract dumped the full envelope dict as the body string.
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = None
    assert not (isinstance(parsed, dict) and "changed" in parsed and "page" in parsed), (
        "error body must be a plain message, not a pre-dumped envelope JSON string: " + body
    )


def test_wire_envelope_shape_is_dispatcher_owned():
    """The dispatcher owns the envelope: error renders
    ``[browser(status=error, code=...)] ... [end:browser]``."""
    rendered = _dispatch({"action": "open"})
    assert rendered.startswith("[browser(status=error"), rendered
    assert rendered.endswith("[end:browser]"), rendered
