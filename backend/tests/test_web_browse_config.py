

import pytest

from configs.channels.web_browse import WebBrowseConfig
from services.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig
from tools.browser.session import close_session, record_screenshot

pytestmark = pytest.mark.unit

_CHAT = ProcessorConfig.PolicyChannel.CHAT


def _mp():
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "find the cheapest flight to Malta", {})
    mp.config = WebBrowseConfig(_CHAT)
    mp._setup()
    return mp


def test_config_contract():
    cfg = WebBrowseConfig(_CHAT)
    assert set(cfg.always_available) == {"browser", "read", "vision", "memory"}
    # Both False or the delegate goes act-trail-blind again.
    assert cfg.skip_transcript is False
    assert cfg.skip_input_row is False
    assert cfg.suppress_history is True
    assert len(cfg.post_turn_hooks) == 1
    assert "STOP RULE" in cfg.get_system_prompt(None)


def test_screenshot_ledger_pins_doc_ids_into_every_prompt(db):
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


def test_screenshot_doc_ids_survive_session_close_into_the_callers_answer(db):
    from abilities._delegate import delegate_result
    from abilities.web_browse import WebBrowseAbility

    mp = _mp()
    record_screenshot(mp.uid, "ab12cd34", "https://example.com")
    mp.config.post_turn_hooks[0].run(mp, "final answer")

    # Stash survives the pop; the live ledger is gone.
    assert mp.config.final_screenshots() == [("ab12cd34", "https://example.com")]
    assert "ab12cd34" not in mp.config.get_user_prompt(mp)

    # The exact result the outer model receives from run().
    tr = WebBrowseAbility._with_screenshots(
        delegate_result("The page shows Example Domain.", hint="retry"),
        mp.config.final_screenshots(),
    )
    assert tr.status == "success"
    assert "The page shows Example Domain." in tr.body
    assert "doc_id=ab12cd34" in tr.body
    assert "vision" in tr.body
    assert tr.meta == {"screenshots": 1}

    # A delegate that died without an answer keeps its canonical error shape —
    # no ledger lines grafted onto an err body.
    dead = WebBrowseAbility._with_screenshots(
        delegate_result("", hint="retry"), mp.config.final_screenshots()
    )
    assert dead.status == "error"
    assert dead.code == "delegate-no-answer"
    assert "ab12cd34" not in dead.body
