import pytest
from abilities._registry import AbilityRegistry
from configs.channels import UserConfig

pytestmark = pytest.mark.integration


def test_thinking_never_discoverable(db):
    # thinking is registered but in no discoverable/always_available list.
    assert "thinking" in {a.NAME for a in AbilityRegistry.all()}
    cfg = UserConfig()
    assert "thinking" not in (cfg.always_available or [])
    assert "thinking" not in (cfg.discoverable or [])


def test_thinking_config_retains_parent_tool_surface(db):
    from abilities.thinking import ThinkingConfig
    from configs.channels import UserConfig
    parent = UserConfig()
    tc = ThinkingConfig(parent.always_available, parent.discoverable, parent.policy_channel)
    assert tc.always_available == list(parent.always_available or [])
    assert tc.discoverable == list(parent.discoverable or [])
    assert tc.thinking_mode == "high"
    assert tc.max_iterations == 1


def test_thinking_gate_writes_the_public_thinking_level_attr(db):
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
