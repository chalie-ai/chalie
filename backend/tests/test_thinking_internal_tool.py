import sqlite3

import pytest
from abilities._registry import AbilityRegistry
from configs.channels import UserConfig

pytestmark = pytest.mark.integration


def test_thinking_never_discoverable(db: sqlite3.Connection) -> None:
    # thinking is registered but DISCOVERABLE=False, so it never enters the
    # global find_tools roster, and it is in no always_available list either.
    assert "thinking" in {a.get_name() for a in AbilityRegistry.all()}
    assert "thinking" not in AbilityRegistry.discoverable_names()
    cfg = UserConfig()
    assert "thinking" not in (cfg.always_available or [])


def test_thinking_config_mirrors_parent_tool_surface(db: sqlite3.Connection) -> None:
    from abilities.thinking import ThinkingConfig
    from configs.channels import UserConfig
    parent = UserConfig()
    active_tools_snapshot = list(parent.always_available or [])
    tc = ThinkingConfig(active_tools_snapshot, parent.policy_channel)
    assert tc.always_available == active_tools_snapshot
    assert tc.thinking_mode == "high"


def test_thinking_gate_writes_the_public_thinking_level_attr(db: sqlite3.Connection) -> None:
    """The deliberation gate MUST write the PUBLIC self.thinking_level that
    _seed_turn_zero / send() / Providers.send() read. Regression guard: when the
    gate wrote the private self._thinking_level instead, high-deliberation thinking
    never fired. We seed an out-of-band sentinel, run the REAL gate (real
    DeliberationScoreService + EMA, zero mocks), and require the gate to overwrite
    the sentinel on the public attr. On the pre-fix code (gate writes _thinking_level)
    the sentinel survives and this fails — exactly the regression.
    """
    from services.message_processor import MessageProcessor

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "Walk me through the trade-offs of three caching strategies.", None)
    mp.config = UserConfig()      # channel='user' → gate does NOT early-return
    mp.uid = None
    mp._uid = None                # gate's DB-persist block is skipped when _uid is None
    mp.thinking_level = "__SENTINEL__"   # not a valid bucket — proves the gate wrote it

    mp._run_thinking_gate()       # the REAL gate, real services, no mocks

    # The gate must have written the PUBLIC attribute (every reader uses this one).
    assert mp.thinking_level != "__SENTINEL__", (
        "gate did not write the public thinking_level — readers would see the stale value"
    )
    assert mp.thinking_level in {"low", "medium", "high"}


# ===========================================================================
# Migrated from test_ability_thinking_tool_result.py (TKT-975)
# Ability-specific business-logic tests for the NOTHING sentinel.
# ===========================================================================

from unittest.mock import patch  # noqa: E402 — appended to existing file

_PROVIDERS_RESOLVE = "services.providers.Providers._resolve"


class _ThinkingRecordingProvider:
    """Stand-in for the resolved LLM provider — the single sanctioned boundary."""

    CONTENT_FIELD_LABEL = "message.content"

    def __init__(self, reply_text: str) -> None:
        self._reply_text = reply_text

    def get_context_limit(self) -> int:
        return 200000

    def estimate_request_tokens(self, dto: object) -> int:
        return 1

    def send(self, dto: object) -> object:
        from services.provider_api import ProviderApiResponse
        return ProviderApiResponse(text=self._reply_text, model="recorder", tool_calls=None)


def _build_thinking_parent(raw_input: str) -> object:
    """A real UserConfig MessageProcessor in the exact state ``_seed_turn_zero``
    fires the thinking pass from."""
    from services.message_processor import MessageProcessor
    from services.transcript_service import write_input_row

    parent = object.__new__(MessageProcessor)
    MessageProcessor.__init__(parent, raw_input, None)
    parent.config = UserConfig()
    parent.uid = write_input_row("user", "user", raw_input)
    parent.active_tools = list(parent.config.always_available or [])
    return parent


def _dispatch_thinking(parent: object, reply_text: str) -> str:
    """Drive the real turn-0 dispatch with a recorder that returns *reply_text*."""
    from abilities._dispatcher import ToolDispatcher

    recorder = _ThinkingRecordingProvider(reply_text)
    with patch(_PROVIDERS_RESOLVE, return_value=recorder):
        return ToolDispatcher(parent).dispatch("thinking", {})


@pytest.mark.unit
def test_nothing_sentinel_yields_empty_body(db: sqlite3.Connection) -> None:
    """The ``NOTHING`` sentinel collapses to an empty body — ``ToolResult.ok("")``
    rendered as ``[thinking(status=success)]\\n\\n[end:thinking]`` (the body line
    is empty, the envelope still well-formed)."""
    out = _dispatch_thinking(_build_thinking_parent("Trivial request."), "NOTHING")

    assert out == "[thinking(status=success)]\n\n[end:thinking]"


@pytest.mark.unit
def test_nothing_sentinel_is_case_insensitive(db: sqlite3.Connection) -> None:
    """run() does ``text.strip().upper() == "NOTHING"`` — so lowercase and
    surrounding whitespace still resolve to the empty-body envelope."""
    lowercase = _dispatch_thinking(_build_thinking_parent("Trivial A."), "nothing")
    assert lowercase == "[thinking(status=success)]\n\n[end:thinking]"

    padded = _dispatch_thinking(_build_thinking_parent("Trivial B."), "  NOTHING  \n")
    assert padded == "[thinking(status=success)]\n\n[end:thinking]"
