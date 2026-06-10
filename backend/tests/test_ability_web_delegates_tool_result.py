"""Feature test: web_search / web_browse failure paths return ToolResult.err.

TKT-898. The two delegate twins already return the canonical ``ToolResult`` type;
this locks the FAILURE half of the contract that was missing:

  1. A missing/empty ``query`` (web_search) / ``goal`` (web_browse) must be
     rejected BEFORE a delegate is spawned. At HEAD an empty param flowed straight
     into ``delegate_goal`` and ``MessageProcessor.process`` spawned an expensive
     delegate loop on an empty goal. The fix wires each ability's
     ``ACTION_REQUIRED = {"": (...)}`` so the dispatcher's pre-gate (the ``""`` key
     covers action-less tools) rejects it with ``code=missing-params`` BEFORE the
     policy gate and BEFORE run() — no delegate, no LLM, no network.

  2. An empty delegate answer is not a success. ``MessageProcessor.process``
     returns ``""`` when the inner ACT loop exits without a final answer (it hit
     ``max_iterations`` or was cancelled — ``_loop`` returns ``""`` on every such
     exit). At HEAD web_search rendered that as ``ok("")`` (a blank success the
     outer model silently trusts) and web_browse as a success carrying
     failure-sounding prose. The fix maps it to ``code=delegate-no-answer`` via the
     shared ``delegate_result`` helper both abilities call.

Test seams, zero mocks:
  - The param-gate cases drive the REAL ``ToolDispatcher.dispatch()`` chokepoint
    against the real registry + real policy table. The pre-gate fires before run(),
    so the delegate is never spawned — fully deterministic, offline.
  - The empty/non-empty mapping is exercised through ``delegate_result``, the
    actual module-level production function both ``run()`` methods call (not a
    test re-implementation, not a mock). The only provider-free way to make the
    real inner ``_loop`` return ``""`` is a cancel/cap that the public tool path
    does not expose, so the mapping is asserted at its lowest real seam — the
    shared helper that owns the mapping — exactly as both abilities use it.
"""

import pytest

from abilities._delegate import delegate_result
from abilities._dispatcher import ToolDispatcher
from configs.channels import UserConfig

pytestmark = pytest.mark.unit


class _MP:
    """Minimal real MP-shaped context dispatch reads off the live processor:
    ``config`` (the chat policy channel) and ``uid`` (the act-trail anchor)."""

    def __init__(self, uid: int, config) -> None:
        self.config = config
        self.uid = uid


def _seed_transcript(db, channel: str = "chat") -> int:
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        (channel, "user", "research something"),
    )
    db.commit()
    return cur.lastrowid


@pytest.fixture
def chat_mp(db):
    """A real chat-channel mp bound to the test db, with a seeded transcript anchor
    so the dispatcher's act-trail record has a uid to key against."""
    return _MP(_seed_transcript(db), UserConfig({}))


def _render_envelope(rendered: str, tool: str) -> tuple[str, str]:
    """Split a rendered envelope into (open-tag-line, body+trailer)."""
    head_end = rendered.index("]\n") + 1
    open_tag = rendered[:head_end]
    body = rendered[head_end + 1 : rendered.index(f"\n[end:{tool}]")]
    return open_tag, body


# ── Param gate: missing query/goal is rejected BEFORE a delegate spawns ─────────


@pytest.mark.parametrize(
    "tool,param,params",
    [
        ("web_search", "query", {}),
        ("web_search", "query", {"query": ""}),
        ("web_browse", "goal", {}),
        ("web_browse", "goal", {"goal": ""}),
    ],
)
def test_missing_param_blocked_before_delegate(db, chat_mp, tool, param, params):
    """A missing/empty required param renders ``code=missing-params`` through the
    real dispatch chokepoint. The pre-gate fires before the policy gate and before
    run(), so no ``MessageProcessor.process`` delegate is ever spawned — this test
    needs no provider and cannot hang. At HEAD (no ACTION_REQUIRED on either tool)
    the empty param flowed into run() and spawned a real delegate loop."""
    rendered = ToolDispatcher(chat_mp).dispatch(tool, dict(params))

    open_tag, body = _render_envelope(rendered, tool)
    assert "status=error" in open_tag
    assert "code=missing-params" in open_tag
    assert param in body
    assert f"valid: {param}" in rendered


# ── Empty / non-empty delegate answer → err / ok mapping ────────────────────────


@pytest.mark.parametrize("empty", ["", "   ", "\n\t "])
def test_empty_delegate_answer_is_no_answer_error(empty):
    """An empty (or whitespace-only) delegate answer maps to an ERROR with the
    stable ``delegate-no-answer`` code and a self-correction hint — never a blank
    ``ok("")`` the outer model silently trusts."""
    tr = delegate_result(empty, hint="Narrow the query, then retry.")

    assert tr.status == "error"
    assert tr.code == "delegate-no-answer"
    assert tr.hint == "Narrow the query, then retry."


def test_real_delegate_answer_is_success_verbatim():
    """A real prose answer is returned as success with the body verbatim — the
    delegate's synthesis is never reshaped or summarised by the mapping."""
    answer = "Malta's population is about 540,000 (2024 estimate)."
    tr = delegate_result(answer, hint="Narrow the query, then retry.")

    assert tr.status == "success"
    assert tr.code is None
    assert tr.body == answer
