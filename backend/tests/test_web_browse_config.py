"""Feature tests: the rebuilt web_browse delegate config.

Real config + real MessageProcessor._setup (uid assignment is prod's, not the
test's) + the real screenshot ledger module the browser ability writes to.
Locks: vision in the toolset, 200 iterations, the act-trail flags, the STOP RULE
prompt, the compaction-immune screenshot ledger in the user prompt, the
post-turn close hook wiring, and the cap-hit message.
"""

import pytest

from configs.channels.web_browse import WebBrowseConfig
from services.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig
from tools.browser.session import close_session, record_screenshot

pytestmark = pytest.mark.unit

_CHAT = ProcessorConfig.POLICY_CHANNEL.CHAT


def _mp():
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "find the cheapest flight to Malta", {})
    mp.config = WebBrowseConfig(_CHAT)
    mp._setup()
    return mp


def test_config_contract():
    cfg = WebBrowseConfig(_CHAT)
    assert cfg.max_iterations == 200
    assert set(cfg.always_available) == {"browser", "read", "vision", "memory"}
    assert cfg.discoverable == []
    # Both False or the delegate goes act-trail-blind again.
    assert cfg.skip_transcript is False
    assert cfg.skip_input_row is False
    assert cfg.suppress_history is True
    assert len(cfg.post_turn_hooks) == 1
    assert "STOP RULE" in cfg.get_system_prompt(None)


def test_screenshot_ledger_pins_doc_ids_into_every_prompt(db):
    """The ledger is mechanical state, not act-trail text — it survives
    compaction because get_user_prompt re-renders it deterministically."""
    mp = _mp()
    before = mp.config.get_user_prompt(mp)
    assert before.startswith("Browsing goal:")
    assert "doc_id" not in before

    record_screenshot(mp.uid, "ab12cd34", "https://example.com/checkout")
    after = mp.config.get_user_prompt(mp)
    assert "ab12cd34" in after
    assert "https://example.com/checkout" in after
    assert "vision" in after  # the line teaches the follow-up tool

    close_session(mp.uid)  # the post-turn hook's call — ledger gone
    assert "ab12cd34" not in mp.config.get_user_prompt(mp)


def test_post_turn_hook_clears_the_ledger(db):
    mp = _mp()
    record_screenshot(mp.uid, "ff00ff00", "https://example.com")
    hook = mp.config.post_turn_hooks[0]
    hook.run(mp, "final answer")
    assert "ff00ff00" not in mp.config.get_user_prompt(mp)
