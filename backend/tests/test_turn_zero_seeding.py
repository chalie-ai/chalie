# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Turn-0 seeding (spec §4 / §4d / AC-30 / AC-31) — observable behaviour only.

``_seed_turn_zero()`` has two declarative behaviours, fired once before the
ACT loop's first LLM turn:
  - memory recall — one ``Ability.dispatch(memory, {action:recall, query:raw})``
    when ``config.memory_seed`` is set (and none when it is not);
  - attachment upload — one ``Ability.dispatch(document, {action:upload, ...})``
    per file in ``metadata['attachments']`` (and none when there are none), with
    no separate auto ``document.view``.

These tests cover those behaviours. They deliberately do NOT re-test config
shapes, field-storage contracts, content_type pass-through, or "function X does
not do Y" guards.
"""

import base64
import threading
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_flat_mp(memory_seed=False):
    """Build a flat-path MessageProcessor (config inline) without running _run().

    Single shared factory.  ``mp.config`` is exposed for the few tests that drive
    the full ``process()`` entry point rather than calling ``_seed_turn_zero``
    directly.
    """
    from configs.channels import ProcessorConfig
    from services.message_processor import MessageProcessor

    config = ProcessorConfig(
        channel="test_g_channel",
        role="test",
        usage_class="subconscious",
        build_user_prompt=lambda _mp: "hi",
        build_user_definition=lambda _mp: "test user",
        build_system_prompt=lambda _mp: "sys",
        always_available=[],
        discoverable=[],
        blocked=frozenset(),
        max_iterations=1,
        skip_transcript=True,
        skip_input_row=True,
        suppress_history=True,
        broadcast_to=None,
        memory_seed=memory_seed,
        post_turn=None,
    )
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "what is the capital of France?", None)
    mp.config = config
    mp.uid = None
    mp.current_iteration = 0
    mp.cancel_event = threading.Event()
    mp.deadline = None
    mp.thinking_level = "low"
    mp.thinking_exploration = None
    mp.discovered_tools = []
    return mp


def _patch_providers_noop():
    """Patch Providers.instance() so the LLM send returns immediately (no tools)."""
    from services.providers import Providers
    from services.llm_service import LLMResponse

    fake = MagicMock()
    fake.calculate.return_value = 0.0
    fake.send_messages.return_value = LLMResponse(
        text="done",
        model="test-model",
        provider="mock",
        tool_calls=[],
    )
    return patch.object(Providers, "instance", return_value=fake)


def _patch_read_attachment(name="attachment.txt", content=b"hello world", content_type="text/plain"):
    """Patch _read_attachment to return a fixed tuple, bypassing the OS path guard.

    The real guard resolves /tmp -> /private/tmp on macOS; patching it lets these
    tests drive _seed_turn_zero's dispatch logic with any placeholder path.
    """
    content_b64 = base64.b64encode(content).decode()

    def _fake_read_attachment(path):
        return (name, content_b64, content_type)

    return patch("services.message_processor._read_attachment", side_effect=_fake_read_attachment)


# ── memory recall seeding (declarative memory_seed flag) ───────────────────────


class TestMemoryRecallSeed:
    """memory_seed gates a single ``memory.recall`` dispatch carrying raw_input."""

    def test_memory_seed_true_fires_exactly_one_recall(self, db):
        """memory_seed=True fires exactly one memory recall.  Spec §4d / AC-30."""
        from abilities._base import Ability

        mp = _make_flat_mp(memory_seed=True)

        dispatches = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            dispatches.append((tool_name, dict(params)))
            return ""

        with patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()

        memory_recalls = [
            (tn, p) for tn, p in dispatches
            if tn == "memory" and p.get("action") == "recall"
        ]
        assert len(memory_recalls) == 1, (
            f"Exactly one memory recall must fire when memory_seed=True: "
            f"got {len(memory_recalls)} recalls in {dispatches}"
        )

    def test_memory_recall_query_equals_raw_input(self, db):
        """The recall query is exactly raw_input.  Spec §4d / AC-30."""
        from abilities._base import Ability

        raw = "what is the capital of France?"
        mp = _make_flat_mp(memory_seed=True)
        mp._raw_input = raw

        dispatches = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            dispatches.append((tool_name, dict(params)))
            return ""

        with patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()

        recall_params = [p for tn, p in dispatches if tn == "memory"]
        assert recall_params, "At least one memory dispatch must occur"
        assert recall_params[0].get("query") == raw, (
            f"memory recall query must equal raw_input exactly: "
            f"expected {raw!r}, got {recall_params[0].get('query')!r}"
        )

    def test_no_recall_when_memory_seed_false(self, db):
        """memory_seed=False fires no recall.  Spec §4d / AC-30."""
        from abilities._base import Ability

        mp = _make_flat_mp(memory_seed=False)

        dispatches = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            dispatches.append((tool_name, dict(params)))
            return ""

        with patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()

        memory_recalls = [(tn, p) for tn, p in dispatches if tn == "memory"]
        assert memory_recalls == [], (
            f"No memory recall must fire when memory_seed=False: got {memory_recalls}"
        )


# ── seed ordering — fires before the loop's first LLM call ─────────────────────


class TestSeedBeforeLoop:
    """_seed_turn_zero runs in _setup, before _loop's first send (so turn 0 sees it)."""

    def test_recall_fires_before_loop_entry(self, db):
        """memory recall is dispatched before the loop is entered.  Spec §4d / AC-30."""
        from services.message_processor import MessageProcessor
        from abilities._base import Ability

        config = _make_flat_mp(memory_seed=True).config
        order = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            order.append(f"dispatch:{tool_name}")
            return ""

        def _stub_loop(self):
            order.append("loop_entered")
            return "done"

        with patch.object(Ability, "dispatch", side_effect=_spy_dispatch), \
             patch.object(MessageProcessor, "_loop", _stub_loop), \
             patch.object(MessageProcessor, "_record", return_value=None), \
             _patch_providers_noop():
            MessageProcessor.process("hello", config)

        assert "loop_entered" in order, "loop must run"
        recall_indices = [i for i, e in enumerate(order) if e == "dispatch:memory"]
        loop_index = order.index("loop_entered")
        assert recall_indices, "memory dispatch must appear in order trace"
        assert recall_indices[0] < loop_index, (
            f"memory recall must fire BEFORE loop entry: order={order}"
        )


# ── attachment upload seeding (presence-gated, no auto-view) ───────────────────


class TestAttachmentUploadSeed:
    """metadata['attachments'] drives one document.upload per file, no auto-view."""

    def test_n_attachments_fire_n_upload_dispatches(self, db):
        """N attachments → exactly N document.upload dispatches.  Spec §4d / AC-31."""
        from abilities._base import Ability

        mp = _make_flat_mp(memory_seed=False)
        mp._metadata = {"attachments": ["/fake/a.txt", "/fake/b.txt", "/fake/c.txt"]}

        dispatches = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            dispatches.append(tool_name)
            return ""

        with _patch_read_attachment(), \
             patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()

        upload_dispatches = [tn for tn in dispatches if tn == "document"]
        assert len(upload_dispatches) == 3, (
            f"3 attachments must produce exactly 3 document.upload dispatches: "
            f"got {len(upload_dispatches)} in {dispatches}"
        )

    def test_missing_attachments_key_fires_no_upload(self, db):
        """No 'attachments' key → zero uploads, no error.  Spec §4d / AC-31."""
        from abilities._base import Ability

        mp = _make_flat_mp(memory_seed=False)
        mp._metadata = {"other_key": "value"}  # No 'attachments' key

        dispatches = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            dispatches.append(tool_name)
            return ""

        with patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()  # Must not raise

        assert "document" not in dispatches, (
            f"No document upload must fire without 'attachments' key: got {dispatches}"
        )

    def test_upload_params_carry_name_content_content_type(self, db):
        """document.upload carries action/name/content(base64)/content_type.  Spec §4d / AC-31."""
        from abilities._base import Ability

        file_bytes = b"Hello, document!"
        mp = _make_flat_mp(memory_seed=False)
        mp._metadata = {"attachments": ["/fake/hello.txt"]}

        captured = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            if tool_name == "document":
                captured.append(dict(params))
            return ""

        with _patch_read_attachment(name="hello.txt", content=file_bytes, content_type="text/plain"), \
             patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()

        assert captured, "document dispatch must have fired"
        p = captured[0]
        assert p["action"] == "upload", f"dispatch params must have action='upload': {p}"
        assert "name" in p, f"dispatch params must include 'name': {p}"
        assert "content" in p, f"dispatch params must include 'content': {p}"
        assert "content_type" in p, f"dispatch params must include 'content_type': {p}"
        # content is base64-encoded — decode and verify bytes match
        decoded = base64.b64decode(p["content"])
        assert decoded == file_bytes, (
            f"base64 content must decode back to original file bytes: "
            f"expected {file_bytes!r}, got {decoded!r}"
        )

    def test_no_document_view_dispatched_on_turn_zero(self, db):
        """Upload is the ingest — no auto document.view on turn 0.  Spec §4d / AC-31."""
        from abilities._base import Ability

        mp = _make_flat_mp(memory_seed=False)
        mp._metadata = {"attachments": ["/fake/ingest.txt"]}

        dispatches = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            dispatches.append((tool_name, dict(params)))
            return ""

        with _patch_read_attachment(), \
             patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()

        view_dispatches = [
            (tn, p) for tn, p in dispatches
            if tn == "document" and p.get("action") == "view"
        ]
        assert view_dispatches == [], (
            f"No document.view must be auto-dispatched on turn 0: got {view_dispatches}"
        )
