from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import cast

import pytest

from configs.channels.web_browse import WebBrowseConfig
from controllers.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig
from tools.browser.session import close_session, record_screenshot

pytestmark = pytest.mark.unit

_CHAT = ProcessorConfig.PolicyChannel.CHAT


def _mp() -> MessageProcessor:
    """The constructor is "constructed inert" — it never allocates a turn or
    writes the anchoring input row itself (that's begin()'s job, run on a
    background drive thread we don't want here). Replay the exact synchronous
    half of begin() (turn allocation + the anchoring input row) through the
    mp's own transcript_service so mp.uid is a real row id — the delegate's
    transcript uid IS the browser session key (tools/browser/session.py), so a
    None uid would key the ledger on a value production never produces."""
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, WebBrowseConfig(_CHAT), raw_input="find the cheapest flight to Malta")
    with mp.db.transaction():
        mp.turn_id = mp.transcript_service.allocate_turn()
        mp.uid = mp.transcript_service.append_input(mp.raw_input)
        mp.current_transcript_id = mp.uid
    mp._setup()
    return mp


def test_config_contract() -> None:
    cfg = WebBrowseConfig(_CHAT)
    assert set(cfg.always_available) == {"browser", "read", "vision", "memory"}
    # Both False or the delegate goes act-trail-blind again.
    assert cfg.skip_transcript is False
    assert cfg.skip_input_row is False
    assert cfg.suppress_history is True
    assert len(cfg.post_turn_hooks) == 1


def test_screenshot_ledger_pins_doc_ids_into_every_prompt(db: sqlite3.Connection) -> None:
    mp = _mp()
    cfg = cast(WebBrowseConfig, mp.config)
    uid = cast(int, mp.uid)
    before = cfg.get_user_prompt(mp)
    assert before.startswith("Browsing goal:")
    assert "doc_id" not in before

    record_screenshot(uid, "ab12cd34", "https://example.com/checkout")
    after = cfg.get_user_prompt(mp)
    assert "ab12cd34" in after
    assert "https://example.com/checkout" in after

    close_session(uid)  # the post-turn hook's call — ledger gone
    assert "ab12cd34" not in cfg.get_user_prompt(mp)


def test_post_turn_hook_clears_the_ledger(db: sqlite3.Connection) -> None:
    mp = _mp()
    cfg = cast(WebBrowseConfig, mp.config)
    uid = cast(int, mp.uid)
    record_screenshot(uid, "ff00ff00", "https://example.com")
    hook = cfg.post_turn_hooks[0]
    hook.run(mp, "final answer")
    assert "ff00ff00" not in cfg.get_user_prompt(mp)


def test_screenshot_doc_ids_survive_session_close_into_the_callers_answer(db: sqlite3.Connection) -> None:
    from abilities._delegate import delegate_result
    from abilities.web_browse import WebBrowseAbility
    from services.document_service import DocumentService

    mp = _mp()
    cfg = cast(WebBrowseConfig, mp.config)
    uid = cast(int, mp.uid)

    # Ingest already ran the capture through vision and stored the description as
    # the document's clean_text — seed that exact state so we can assert it rides
    # back inline (the agent sees the page without a separate vision call).
    doc_svc = DocumentService()
    doc_svc.create_document(
        "screenshot-example.png", "image/png", 10,
        os.path.join(tempfile.gettempdir(), "x.png"), "deadbeef",
        source_type="screenshot", doc_id="ab12cd34",
    )
    doc_svc.update_clean_text("ab12cd34", "The page header reads Example Domain.")

    record_screenshot(uid, "ab12cd34", "https://example.com")
    cfg.post_turn_hooks[0].run(mp, "final answer")

    # Stash survives the pop; the live ledger is gone.
    assert cfg.final_screenshots() == [("ab12cd34", "https://example.com")]
    assert "ab12cd34" not in cfg.get_user_prompt(mp)

    # The exact result the outer model receives from run().
    tr = WebBrowseAbility._with_screenshots(
        delegate_result("The page shows Example Domain.", hint="retry"),
        cfg.final_screenshots(),
    )
    assert tr.status == "success"
    assert "The page shows Example Domain." in tr.body
    assert "doc_id=ab12cd34" in tr.body
    assert "https://example.com" in tr.body
    # The stored vision description rides back inline — no separate vision call.
    assert "The page header reads Example Domain." in tr.body
    assert tr.meta == {"screenshots": 1}

    # A delegate that died without an answer keeps its canonical error shape —
    # no ledger lines grafted onto an err body.
    dead = WebBrowseAbility._with_screenshots(
        delegate_result("", hint="retry"), cfg.final_screenshots()
    )
    assert dead.status == "error"
    assert dead.code == "delegate-no-answer"
    assert "ab12cd34" not in dead.body
