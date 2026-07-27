"""Feature test: ``llm_call_log.type`` — who a provider call's spend bills to.

The ledger answers one question: of the tokens spent, how many were the user's
own conversation ('chat') and how many were Chalie acting on its own behalf
('system')? That axis is a single class constant, ``ProcessorConfig.USAGE_TYPE``,
read by ``LlmLogService.record`` off the live config. It replaces the derivation
from ``policy_channel``, which answers an unrelated question (which policy gates
this turn's tools) and got the billing wrong wherever the two diverge.

Drives real turns through the real production entry point (construct inertly,
``begin()``, ``result()``) against the real, fully-migrated SQLite DB, with the
real ``ProviderService`` -> ``LlmLogService`` -> ``LlmCallLog`` write path
running, and asserts on the rows actually committed to ``llm_call_log`` — read
back with raw SQL, so a column the model no longer declares fails loud. The only
substitution is the LLM network boundary: ``services.provider_service.build_client``
(the same seam ``test_context_usage_signal.py`` uses).

Three claims, one per way the constant resolves:

1. a user turn bills 'chat' — the only spend the user actually asked for;
2. a delegate bills 'system' EVEN WHEN its ``policy_channel`` is CHAT — the
   divergence that proves the two axes are now genuinely independent;
3. DiscoveryConfig bills 'system' despite subclassing UserConfig — the
   inheritance trap, guarded.
"""

import sqlite3
import threading
import time
from unittest.mock import patch

import pytest

from configs.channels.discovery import DiscoveryConfig
from configs.channels.user import UserConfig
from configs.channels.web_search import WebSearchConfig
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from models.provider_response import ProviderResponse
from services.processor_config import ProcessorConfig

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("chat_provider")]

_BUILD_CLIENT = "services.provider_service.build_client"


class _ScriptedProvider:
    """A real functional double at the network boundary: implements the thin
    client protocol and answers with one benign terminal response (non-empty
    text, no tool calls) so the turn settles in a single step — one CHAT call,
    hence exactly one ledger row per turn. The text must be non-empty: an empty
    completion trips the processor's empty-completion steer, which re-sends and
    would append extra ledger rows."""

    def __init__(self) -> None:
        self.sends = 0

    def get_context_limit(self) -> int:
        return 120_000

    def estimate_request_tokens(self, _dto: object) -> int:
        return 1

    def send(self, _dto: object) -> ProviderResponse:
        self.sends += 1
        return ProviderResponse(
            text="ok.",
            model="scripted-usage-type",
            tool_calls=None,
            tokens_input=100,
            tokens_output=10,
        )


def _drain_background_turns(timeout_s: float = 10.0) -> None:
    """Join the fire-and-forget post-turn daemon turns a completed turn spawns
    (skill suggestion, thread gist) so they run to completion inside THIS test's
    provider+DB patch and never leak into the next. Each is a whole
    MessageProcessor turn on its own daemon; left running it would call
    ``build_client`` while the NEXT test holds the patch — and, here, would append
    its own 'system' rows to the ledger mid-assertion."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pending = [
            t for t in threading.enumerate()
            if t.name in ("skill-suggest", "thread-gist") or t.name.startswith("turn-")
        ]
        if not pending:
            return
        for t in pending:
            t.join(timeout=deadline - time.monotonic())


def _run(config: ProcessorConfig, raw_input: str) -> _ScriptedProvider:
    """Drive a real turn on *config* to termination, then quiesce any post-turn
    daemon turns inside the patch so the ledger settles before it is read."""
    provider = _ScriptedProvider()
    mp = MessageProcessor(config, raw_input=raw_input)  # inert (I2)
    with patch(_BUILD_CLIENT, return_value=provider):
        mp.begin()
        mp.result()
        _drain_background_turns()
    return provider


def _logged_types(db: sqlite3.Connection, model: str = "scripted-usage-type") -> list[str]:
    """The ``type`` of every ledger row this test's turns actually committed, read
    straight from SQL rather than through the model — a column the schema no
    longer carries fails here rather than passing silently."""
    return [
        r[0] for r in db.execute(
            "SELECT type FROM llm_call_log WHERE model = ? ORDER BY id", (model,),
        ).fetchall()
    ]


def test_user_turn_bills_chat(db: sqlite3.Connection) -> None:
    """A user turn is the user's own conversation: it bills 'chat'."""
    provider = _run(UserConfig(), "hello")

    assert provider.sends >= 1, "turn never reached the provider — an empty ledger proves nothing"
    assert _logged_types(db) == ["chat"]


def test_delegate_with_chat_policy_still_bills_system(db: sqlite3.Connection) -> None:
    """A delegate bills 'system' even when policy-gated as CHAT.

    ``WebSearchConfig`` takes its ``policy_channel`` from whoever invoked the tool,
    so a search fired from a user turn carries ``PolicyChannel.CHAT`` — which is
    correct for tool gating and was exactly what made the OLD derivation bill the
    user for Chalie's own delegate. Passing CHAT here reproduces that precise
    input; 'system' is the proof the spend axis no longer rides on the policy one.

    The DELEGATE lane resolves off the same selected row the ``chat_provider``
    fixture seeds (``_resolve_delegate_provider`` falls back to it when no
    explicit delegate is set), so the delegate reaches a real provider config —
    only the transport it then builds is the patched double.
    """
    provider = _run(WebSearchConfig(PolicyChannel.CHAT), "who won the match")

    assert provider.sends >= 1, "delegate never reached the provider"
    assert _logged_types(db) == ["system"]


def test_discovery_bills_system_not_chat(db: sqlite3.Connection) -> None:
    """DiscoveryConfig subclasses UserConfig — and so would INHERIT
    ``USAGE_TYPE = "chat"`` unless it re-declares it. It is a background research
    loop the user never asked for and never sees, so its spend is Chalie's, not
    theirs. This is the regression guard on that inheritance trap: it fails the
    moment the override in ``configs/channels/discovery.py`` is dropped."""
    provider = _run(DiscoveryConfig(), "quietly research something")

    assert provider.sends >= 1, "discovery turn never reached the provider"
    assert _logged_types(db) == ["system"]
