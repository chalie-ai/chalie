"""Feature test: web_search / web_browse delegates render their own act-trail.

Both delegate configs used to set ``skip_transcript=True`` AND
``skip_input_row=True`` — a "clean context" bundle that wrongly folded in the
async-only HiddenInput flag (``skip_input_row`` belongs to ``deliver_async_result``
/ ``with_hidden_input``, not to a delegate). Together those two flags trip the
``_setup`` uid guard (``not skip_transcript and not skip_input_row``), so
``mp.uid`` stayed ``None``, the act-trail render short-circuited to ``""``, and the
delegate's ``get_user_prompt`` was a constant, results-blind prompt on EVERY ACT
iteration. The loop never saw what it had already gathered, so it re-issued the
same searches turn after turn — minutes per delegate.

This drives the REAL production path with zero mocks: the real
``MessageProcessor.begin()`` synchronous turn/uid allocation (whose uid
assignment from the config flags is the fix), the real ``DispatchService``
dispatch chokepoint (which records each tool call under that uid), the real
``WebSearchConfig`` / ``WebBrowseConfig`` ``get_user_prompt`` +
``PromptService.act_trail()``, against the real SQLite test db.

The regression it locks, end to end: after the delegate executes a tool, its
NEXT-iteration user prompt must GROW to include that tool's result. With the old
flags ``mp.uid`` is ``None`` and the prompt stays constant — both assertions below
fail. With the fix the delegate sees its own work and can converge.
"""

import sqlite3
from typing import TYPE_CHECKING, Protocol, cast

import pytest

from configs.channels.web_browse import WebBrowseConfig
from configs.channels.web_search import WebSearchConfig
from controllers.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    class _DelegateConfigCls(Protocol):
        def __call__(self, policy_channel: ProcessorConfig.PolicyChannel) -> ProcessorConfig: ...

pytestmark = pytest.mark.unit

_CHAT = ProcessorConfig.PolicyChannel.CHAT

# (config class, expected transcript channel, expected role, user-prompt prefix)
_DELEGATES = [
    (WebSearchConfig, "delegate:web_search", "web_search", "Research query:"),
    (WebBrowseConfig, "delegate:web_browse", "web_browse", "Browsing goal:"),
]


def _delegate_mp(config: ProcessorConfig) -> MessageProcessor:
    """Build a real MessageProcessor bound to a delegate config.

    The constructor is "constructed inert" (I2: zero DB/WS side-effects) — it
    never allocates the turn or writes the anchoring input row itself (that's
    ``begin()``'s job, run on a background drive thread we don't want here).
    Replay the exact synchronous half of ``begin()`` (turn allocation + the
    anchoring input row) through the mp's own transcript_service, so ``mp.uid``
    is a real row, ``mp.turn_id`` a real allocated turn (the id
    ``ToolCall.by_turn`` joins ``transcript`` on), and the config's
    channel/role land on that row exactly as production writes them — no
    private-field poking, no invented API."""
    mp = MessageProcessor(config, raw_input="current population of Malta")
    mp.active_tools = list(mp.config.always_available or [])
    with mp.db.transaction():
        mp.turn_id = mp.transcript_service.allocate_turn()
        mp.uid = mp.transcript_service.append_input(mp.raw_input)
        mp.current_transcript_id = mp.uid
    return mp


@pytest.mark.parametrize("config_cls,channel,role,prefix", _DELEGATES)
def test_delegate_sees_its_own_act_trail(
    db: sqlite3.Connection,
    config_cls: "_DelegateConfigCls",
    channel: str,
    role: str,
    prefix: str,
) -> None:
    mp = _delegate_mp(config_cls(_CHAT))

    # ── Fix part 1: _setup assigned a uid and wrote the delegate's OWN
    #    transcript row (old flags left uid None and wrote nothing). ────────────
    assert mp.uid is not None, (
        "delegate _setup did not assign a uid — skip_transcript/skip_input_row "
        "regressed and the act-trail is dead again"
    )
    trow = db.execute(
        "SELECT channel, role FROM transcript WHERE id = ?", (mp.uid,)
    ).fetchone()
    assert trow is not None, "no transcript row written for the delegate turn"
    assert (cast(tuple[object, ...], trow)[0], cast(tuple[object, ...], trow)[1]) == (channel, role), (
        f"delegate row landed on the wrong channel/role: {tuple(trow)!r}"
    )

    # Before any tool executes, the prompt is the bare, results-blind query.
    config = mp.config
    blind = config.get_user_prompt(mp)
    assert blind.startswith(prefix)
    assert mp.prompt_service.act_trail() == "", "trail should be empty before any tool runs"

    # ── Execute one real tool through the production dispatch chokepoint. ──────
    #    `memory` is in the delegate's always_available tier; recall is
    #    deterministic (local embeddings + SQLite, no network).
    mp.dispatch_service.dispatch(
        "memory", {"action": "recall", "query": "current population of Malta"}
    )

    # ── Fix part 2: the call was recorded under the delegate's uid (the FK the
    #    old None-uid silently dropped) and now renders into the trail. ─────────
    names = [
        cast(tuple[object, ...], r)[0]
        for r in db.execute(
            "SELECT tool_name FROM tool_calls WHERE transcript_id = ?", (mp.uid,)
        ).fetchall()
    ]
    assert "memory" in names, (
        f"the delegate's tool call was not recorded under its uid: {names!r}"
    )

    trail = mp.prompt_service.act_trail()
    assert trail, "act-trail rendered empty even after a recorded tool call"

    # The NEXT-iteration prompt the delegate would send GREW to include the trail
    # — it is no longer blind to its own results.
    seeing = config.get_user_prompt(mp)
    assert seeing.startswith(prefix)
    assert len(seeing) > len(blind), (
        "delegate prompt did not grow across the iteration — still results-blind"
    )
    assert trail in seeing, "rendered act-trail was not appended to the user prompt"
