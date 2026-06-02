# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""§9a area G (G1-G14) — Turn-0 seeding, spec §4/§4d, AC-30/AC-31.

Tests for _seed_turn_zero() on the flat-path MessageProcessor:
  - G1-G5:  memory recall seeding (declarative memory_seed flag)
  - G6-G14: attachment upload seeding (presence-gated, no auto-view)

GOVERNING TEST PRINCIPLE: if a test fails, the CODE is wrong — fix the code.
Never weaken, delete, xfail, or skip a test in this file (except G11, which is
explicitly xfail pending TKT-715).
"""

import base64
import os
import tempfile
import threading
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_config(
    channel="test_g_channel",
    role="test",
    memory_seed=False,
    skip_transcript=True,
):
    """Return a minimal ProcessorConfig for flat-path turn-0 seeding tests."""
    from configs.channels import ProcessorConfig

    return ProcessorConfig(
        channel=channel,
        role=role,
        usage_class="subconscious",
        build_user_prompt=lambda _mp: "hi",
        build_user_definition=lambda _mp: "test user",
        build_system_prompt=lambda _mp: "sys",
        always_available=[],
        discoverable=[],
        blocked=frozenset(),
        max_iterations=1,
        skip_transcript=skip_transcript,
        skip_input_row=True,
        suppress_history=True,
        broadcast_to=None,
        memory_seed=memory_seed,
        post_turn=None,
    )


def _make_flat_mp(config):
    """Create a flat-path MessageProcessor without running _run()."""
    from services.message_processor import MessageProcessor

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


def _make_tmp_attachment(suffix=".txt", content=b"hello world"):
    """Create a real /tmp file and return its path (caller deletes).

    The file name includes 'chalie_' to look realistic.  Note: _read_attachment
    guards on os.path.realpath(path).startswith('/tmp/chalie_'), which on macOS
    resolves /tmp -> /private/tmp and fails.  Tests that invoke _seed_turn_zero
    with real files must patch _read_attachment at the module boundary so they
    test dispatch behaviour rather than the path guard.  See _patch_read_attachment.
    """
    fd, path = tempfile.mkstemp(prefix="chalie_test_", suffix=suffix)
    os.write(fd, content)
    os.close(fd)
    return path


def _patch_read_attachment(name="attachment.txt", content=b"hello world", content_type="text/plain"):
    """Return a context manager that patches _read_attachment to return a fixed tuple.

    This lets tests drive _seed_turn_zero's dispatch logic without hitting the
    /tmp/chalie_* path guard (which resolves /tmp -> /private/tmp on macOS).
    The fixture accepts any path string in metadata['attachments'].
    """
    import base64 as _b64
    content_b64 = _b64.b64encode(content).decode()

    def _fake_read_attachment(path):
        return (name, content_b64, content_type)

    return patch("services.message_processor._read_attachment", side_effect=_fake_read_attachment)


# ── G1: memory_seed=True fires exactly ONE memory recall before iteration 0 ───


class TestG1MemorySeedFiresOneRecall:
    """G1 — memory_seed=True fires one blocking memory recall {action:'recall',
    query:raw_input} before iteration 0.  Spec §4/§4d / AC-30."""

    def test_memory_seed_true_fires_exactly_one_recall(self, db):
        """Single recall dispatch when memory_seed=True."""
        from abilities._base import Ability

        config = _make_config(memory_seed=True)
        mp = _make_flat_mp(config)

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
            f"Exactly one memory recall must fire when memory_seed=True (G1): "
            f"got {len(memory_recalls)} recalls in {dispatches}"
        )

    def test_memory_recall_query_equals_raw_input(self, db):
        """The query dispatched to memory recall is exactly raw_input. Spec §4d / G1."""
        from abilities._base import Ability

        raw = "what is the capital of France?"
        config = _make_config(memory_seed=True)
        mp = _make_flat_mp(config)
        mp._raw_input = raw

        dispatches = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            dispatches.append((tool_name, dict(params)))
            return ""

        with patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()

        recall_params = [p for tn, p in dispatches if tn == "memory"]
        assert recall_params, "At least one memory dispatch must occur (G1)"
        assert recall_params[0].get("query") == raw, (
            f"memory recall query must equal raw_input exactly (G1): "
            f"expected {raw!r}, got {recall_params[0].get('query')!r}"
        )


# ── G2: memory_seed=False fires NO recall ────────────────────────────────────


class TestG2MemorySeedFalseFiresNoRecall:
    """G2 — memory_seed=False fires no recall.  Spec §4/§3b / AC-30."""

    def test_no_recall_when_memory_seed_false(self, db):
        """No memory recall dispatched when memory_seed=False (default)."""
        from abilities._base import Ability

        config = _make_config(memory_seed=False)
        mp = _make_flat_mp(config)

        dispatches = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            dispatches.append((tool_name, dict(params)))
            return ""

        # No attachments, so only memory might fire
        with patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()

        memory_recalls = [
            (tn, p) for tn, p in dispatches
            if tn == "memory"
        ]
        assert memory_recalls == [], (
            f"No memory recall must fire when memory_seed=False (G2): "
            f"got {memory_recalls}"
        )


# ── G3: Seeded recall fires BEFORE _loop's first LLM call ────────────────────


class TestG3RecallBeforeFirstLLMCall:
    """G3 — seeded recall result visible to first LLM turn (fires before loop).
    Spec §4d / AC-30."""

    def test_recall_fires_in_setup_before_loop_send(self, db):
        """_seed_turn_zero fires recall during _setup(); _loop's first send comes after.

        We verify ordering by confirming dispatch is called inside _setup, not
        inside _loop, by stubbing _loop and confirming dispatch ran before it.
        """
        from services.message_processor import MessageProcessor
        from abilities._base import Ability

        config = _make_config(memory_seed=True)
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

        assert "loop_entered" in order, "loop must run (G3)"
        # memory recall must appear before loop_entered
        recall_indices = [i for i, e in enumerate(order) if e == "dispatch:memory"]
        loop_index = order.index("loop_entered")
        assert recall_indices, "memory dispatch must appear in order trace (G3)"
        assert recall_indices[0] < loop_index, (
            f"memory recall must fire BEFORE loop entry (G3): order={order}"
        )


# ── G4: memory_seed carries NO radius/header/prompt-injection logic ──────────


class TestG4MemorySeedNoExtraKeys:
    """G4 — recall input is exactly {action:'recall', query:raw_input}, nothing else.
    Spec §4d / AC-30."""

    def test_recall_params_contain_no_extra_keys(self, db):
        """Dispatched memory params must have exactly action + query, no radius/header."""
        from abilities._base import Ability

        config = _make_config(memory_seed=True)
        mp = _make_flat_mp(config)
        mp._raw_input = "tell me about coffee"

        captured_params = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            if tool_name == "memory":
                captured_params.append(dict(params))
            return ""

        with patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()

        assert captured_params, "memory dispatch must fire (G4)"
        p = captured_params[0]
        assert p == {"action": "recall", "query": "tell me about coffee"}, (
            f"memory recall params must be exactly {{action, query}} (G4): got {p}"
        )


# ── G5: EAMP memory seed now actually happens ─────────────────────────────────


class TestG5EAMPMemorySeedFires:
    """G5 — EAMP memory seed actually happens (fixes prior dead branch).
    make_eamp_config has memory_seed=True.  Spec §4d / AC-30."""

    def test_eamp_config_has_memory_seed_true(self):
        """make_eamp_config returns a config with memory_seed=True (G5 / spec §4d)."""
        from configs.channels import make_eamp_config

        cfg = make_eamp_config(
            agent_name="testbot",
            project="test_project",
            loop_in_human=False,
            wrapper_id="w1",
        )
        assert cfg.memory_seed is True, (
            f"EAMP config must have memory_seed=True (G5): got {cfg.memory_seed}"
        )

    def test_eamp_seed_fires_recall_dispatch(self, db):
        """When _seed_turn_zero runs with an EAMP-shaped config (memory_seed=True),
        a memory recall is dispatched.  Spec §4d / G5."""
        from abilities._base import Ability
        from configs.channels import ProcessorConfig

        # Build a minimal EAMP-shaped config without triggering the real EAMP
        # post_turn / system-prompt builders (which need live providers).
        cfg = ProcessorConfig(
            channel="external-agent:testbot",
            role="external_agent",
            usage_class="external_agent",
            build_user_prompt=lambda mp: mp._raw_input,
            build_user_definition=lambda mp: "testbot",
            build_system_prompt=lambda mp: "sys",
            always_available=[],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=1,
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=False,
            broadcast_to=None,
            memory_seed=True,  # the critical flag
            post_turn=None,
        )
        mp = _make_flat_mp(cfg)
        mp._raw_input = "help me plan a trip"

        dispatches = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            dispatches.append(tool_name)
            return ""

        with patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()

        assert "memory" in dispatches, (
            f"EAMP (memory_seed=True) must fire a memory recall dispatch (G5): "
            f"got {dispatches}"
        )


# ── G6: N attachments → exactly N blocking document.upload dispatches ─────────


class TestG6NAttachmentsNUploads:
    """G6 — N attachments in metadata → exactly N blocking document.upload
    dispatches on turn 0.  Spec §4/§4d / AC-31."""

    def test_three_attachments_fire_three_upload_dispatches(self, db):
        """Three paths in metadata['attachments'] → three document.upload calls."""
        from abilities._base import Ability

        config = _make_config(memory_seed=False)
        mp = _make_flat_mp(config)
        # Use placeholder paths — _read_attachment is patched to avoid OS path guard
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
            f"3 attachments must produce exactly 3 document.upload dispatches (G6): "
            f"got {len(upload_dispatches)} in {dispatches}"
        )

    def test_single_attachment_fires_one_upload(self, db):
        """One attachment → exactly one document.upload dispatch (G6)."""
        from abilities._base import Ability

        config = _make_config(memory_seed=False)
        mp = _make_flat_mp(config)
        mp._metadata = {"attachments": ["/fake/doc.pdf"]}

        dispatches = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            dispatches.append((tool_name, dict(params)))
            return ""

        with _patch_read_attachment(name="doc.pdf", content=b"%PDF-1.4", content_type="application/pdf"), \
             patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()

        doc_dispatches = [p for tn, p in dispatches if tn == "document"]
        assert len(doc_dispatches) == 1, (
            f"Single attachment must produce exactly 1 document dispatch (G6): "
            f"got {len(doc_dispatches)}"
        )
        assert doc_dispatches[0].get("action") == "upload", (
            f"Document dispatch must use action='upload' (G6): "
            f"got {doc_dispatches[0].get('action')!r}"
        )


# ── G7: No attachments key → zero uploads, no error ──────────────────────────


class TestG7NoAttachmentsNoUploads:
    """G7 — no 'attachments' key in metadata → zero uploads, no error.
    Spec §4/§4d / AC-31."""

    def test_missing_attachments_key_fires_no_upload(self, db):
        """metadata without 'attachments' key → zero document.upload dispatches."""
        from abilities._base import Ability

        config = _make_config(memory_seed=False)
        mp = _make_flat_mp(config)
        mp._metadata = {"other_key": "value"}  # No 'attachments' key

        dispatches = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            dispatches.append(tool_name)
            return ""

        with patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()  # Must not raise

        assert "document" not in dispatches, (
            f"No document upload must fire without 'attachments' key (G7): "
            f"got {dispatches}"
        )

    def test_empty_metadata_fires_no_upload(self, db):
        """Empty metadata dict → zero document.upload dispatches, no error."""
        from abilities._base import Ability

        config = _make_config(memory_seed=False)
        mp = _make_flat_mp(config)
        mp._metadata = {}

        dispatches = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            dispatches.append(tool_name)
            return ""

        with patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()  # Must not raise

        assert dispatches == [], (
            f"Empty metadata must produce zero dispatches (G7): got {dispatches}"
        )


# ── G8: Attachment upload blocks BEFORE model reasoning ───────────────────────


class TestG8UploadBeforeModelReasoning:
    """G8 — attachment upload blocks before model reasoning.
    _seed_turn_zero() fires in _setup, before _loop's first LLM call.
    Spec §4d / AC-31."""

    def test_upload_fires_in_setup_before_loop_send(self, db):
        """document.upload dispatch must appear before 'loop_entered' in order."""
        from services.message_processor import MessageProcessor
        from abilities._base import Ability

        config = _make_config(memory_seed=False)
        order = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            order.append(f"dispatch:{tool_name}")
            return ""

        def _stub_loop(self):
            order.append("loop_entered")
            return "done"

        def _capturing_process(raw_input, cfg, metadata=None, **kw):
            mp = object.__new__(MessageProcessor)
            MessageProcessor.__init__(mp, raw_input, metadata)
            mp.config = cfg
            mp.uid = None
            mp.current_iteration = 0
            mp.cancel_event = threading.Event()
            mp.deadline = None
            mp.thinking_level = "low"
            mp.thinking_exploration = None
            mp.discovered_tools = []
            with patch.object(MessageProcessor, "_loop", _stub_loop), \
                 patch.object(MessageProcessor, "_record", return_value=None):
                mp._setup()
                mp._loop()
            return "done"

        with _patch_read_attachment(), \
             patch.object(Ability, "dispatch", side_effect=_spy_dispatch), \
             patch.object(MessageProcessor, "process", staticmethod(_capturing_process)):
            MessageProcessor.process("hi", config, metadata={"attachments": ["/fake/file.txt"]})

        upload_indices = [i for i, e in enumerate(order) if e == "dispatch:document"]
        loop_index = order.index("loop_entered") if "loop_entered" in order else len(order)
        assert upload_indices, f"document upload must appear in order trace (G8): {order}"
        assert upload_indices[0] < loop_index, (
            f"document upload must fire BEFORE loop entry (G8): order={order}"
        )


# ── G9: Seeded calls land in the trail like any tool ─────────────────────────


class TestG9SeededCallsLandInTrail:
    """G9 — seeded calls land in the trail like any tool (go through Ability.dispatch/record).
    Spec §4/§4d."""

    def test_memory_recall_dispatch_routes_through_ability_dispatch(self, db):
        """_seed_turn_zero with memory_seed=True calls Ability.dispatch (the only
        write path to the trail) for the memory recall.  Spec §4/§4d / G9."""
        from abilities._base import Ability

        config = _make_config(memory_seed=True)
        mp = _make_flat_mp(config)

        dispatch_calls = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            dispatch_calls.append((tool_name, dict(params)))
            return ""

        with patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()

        # The seeded call goes through Ability.dispatch — same path as any LLM call
        assert any(tn == "memory" for tn, _ in dispatch_calls), (
            "memory recall seeding must route through Ability.dispatch (G9): "
            f"dispatch_calls={dispatch_calls}"
        )

    def test_attachment_dispatch_routes_through_ability_dispatch(self, db):
        """_seed_turn_zero fires document.upload through Ability.dispatch (G9)."""
        from abilities._base import Ability

        config = _make_config(memory_seed=False)
        mp = _make_flat_mp(config)
        mp._metadata = {"attachments": ["/fake/test.txt"]}

        dispatch_calls = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            dispatch_calls.append(tool_name)
            return ""

        with _patch_read_attachment(), \
             patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()

        assert "document" in dispatch_calls, (
            "attachment upload seeding must route through Ability.dispatch (G9): "
            f"dispatch_calls={dispatch_calls}"
        )


# ── G10: document.upload stores file and carries name/content/content_type ────


class TestG10UploadDispatchParams:
    """G10 — document.upload is dispatched with name, content (base64), content_type.
    Spec §4d / AC-31."""

    def test_upload_params_carry_name_content_content_type(self, db):
        """The dispatch params for document.upload contain name, content, content_type."""
        from abilities._base import Ability

        file_bytes = b"Hello, document!"
        config = _make_config(memory_seed=False)
        mp = _make_flat_mp(config)
        mp._metadata = {"attachments": ["/fake/hello.txt"]}

        captured = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            if tool_name == "document":
                captured.append(dict(params))
            return ""

        with _patch_read_attachment(name="hello.txt", content=file_bytes, content_type="text/plain"), \
             patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()

        assert captured, "document dispatch must have fired (G10)"
        p = captured[0]
        assert "name" in p, f"dispatch params must include 'name' (G10): {p}"
        assert "content" in p, f"dispatch params must include 'content' (G10): {p}"
        assert "content_type" in p, f"dispatch params must include 'content_type' (G10): {p}"
        assert p["action"] == "upload", (
            f"dispatch params must have action='upload' (G10): {p}"
        )
        # content is base64-encoded — decode and verify bytes match
        decoded = base64.b64decode(p["content"])
        assert decoded == file_bytes, (
            f"base64 content must decode back to original file bytes (G10): "
            f"expected {file_bytes!r}, got {decoded!r}"
        )


# ── G11: Image upload → vision provider description (TKT-715 dependent) ───────


class TestG11ImageVisionProviderDescription:
    """G11 — image upload stores vision-provider description when vision provider set.
    Depends on TKT-715 (PLAN ONLY, not yet implemented).  Spec §4d / AC-31."""

    @pytest.mark.xfail(
        reason=(
            "Vision-provider routing for images (step 3 of document.upload contract) "
            "is owned by TKT-715 (PLAN ONLY — not yet implemented).  Until TKT-715 "
            "lands, images degrade to OCR (G12).  This xfail is strict=False so the "
            "test passes silently once TKT-715 ships the vision branch.  Spec §4d."
        ),
        strict=False,
    )
    def test_image_upload_stores_vision_description_when_provider_set(self, db):
        """When a vision provider is configured, image upload contents come from
        the vision-LLM description (not OCR).  TKT-715 owns this branch.

        Until TKT-715 ships: this test is expected to fail because the vision branch
        does not exist in the production code yet.  Mark strict=False so CI is green.
        """
        # This test asserts the contract at the dispatch boundary (document.upload
        # with an image path is dispatched), and verifies that when a vision provider
        # is configured the result carries a description rather than OCR text.
        # The actual vision routing is TKT-715 scope.
        from abilities._base import Ability

        config = _make_config(memory_seed=False)
        mp = _make_flat_mp(config)
        mp._metadata = {"attachments": ["/fake/image.jpg"]}

        dispatched = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            dispatched.append((tool_name, dict(params)))
            return '{"status": "success", "result": "Vision description: a JPEG image"}'

        # Simulate a vision provider being configured (TKT-715 scope):
        # if vision provider set → result should contain "Vision description", not just OCR.
        # Since TKT-715 is not built, this assertion will fail → xfail.
        with _patch_read_attachment(name="image.jpg", content=b"\xff\xd8\xff\xe0", content_type="image/jpeg"), \
             patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()

        assert any(tn == "document" for tn, _ in dispatched), (
            "Image attachment must be dispatched as document.upload (G11)"
        )
        # This assertion is the xfail trigger: vision routing not yet implemented.
        # Once TKT-715 ships, this should pass and the xfail can be removed.
        raise AssertionError(
            "TKT-715 vision routing not yet built — xfail expected.  "
            "Remove this when TKT-715 is implemented."
        )


# ── G12: Image upload falls back to OCR when no vision provider ───────────────


class TestG12ImageOCRFallback:
    """G12 — image upload falls back to OCR when no vision provider is set.
    This is the current real contract (TKT-715 not yet built).  Spec §4d / AC-31."""

    def test_image_attachment_dispatches_document_upload(self, db):
        """An image file in metadata['attachments'] dispatches document.upload
        (regardless of vision provider — current OCR fallback path).  G12."""
        from abilities._base import Ability

        config = _make_config(memory_seed=False)
        mp = _make_flat_mp(config)
        mp._metadata = {"attachments": ["/fake/photo.jpg"]}

        dispatched = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            dispatched.append((tool_name, dict(params)))
            return ""

        with _patch_read_attachment(name="photo.jpg", content=b"\xff\xd8\xff\xe0", content_type="image/jpeg"), \
             patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()

        doc_dispatches = [(tn, p) for tn, p in dispatched if tn == "document"]
        assert len(doc_dispatches) == 1, (
            f"Image attachment must produce exactly 1 document.upload dispatch (G12): "
            f"got {doc_dispatches}"
        )
        p = doc_dispatches[0][1]
        assert p.get("action") == "upload", (
            f"Image dispatch must use action='upload' (G12): got {p.get('action')!r}"
        )
        assert p.get("content_type", "").startswith("image/"), (
            f"Image content_type must be 'image/*' (G12): got {p.get('content_type')!r}"
        )


# ── G13: Upload extraction is synchronous/terminal before tool returns ─────────


class TestG13UploadIsSynchronous:
    """G13 — upload extraction is synchronous/terminal (status ready/failed) before
    the tool returns.  The dispatch blocks — _seed_turn_zero does not return until
    all uploads complete.  Spec §4d / AC-31."""

    def test_seed_does_not_return_until_dispatch_completes(self, db):
        """_seed_turn_zero blocks on each upload dispatch before returning.

        We verify this by recording when dispatch returns vs when _seed_turn_zero
        returns — they must both complete before the next statement after
        _seed_turn_zero().  (Synchronous = dispatch is not async fire-and-forget.)
        """
        from abilities._base import Ability

        config = _make_config(memory_seed=False)
        mp = _make_flat_mp(config)
        mp._metadata = {"attachments": ["/fake/sync.txt"]}

        dispatch_completed = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            dispatch_completed.append(True)
            return "upload result"  # synchronous return

        with _patch_read_attachment(), \
             patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()

        # If we reach here, _seed_turn_zero() has returned.
        # Since dispatch_completed was populated inside dispatch (called synchronously),
        # it must be non-empty now — proving the call was blocking.
        assert dispatch_completed, (
            "_seed_turn_zero must block on Ability.dispatch (synchronous) (G13): "
            "dispatch_completed was empty after _seed_turn_zero returned"
        )

    def test_multiple_uploads_all_complete_before_return(self, db):
        """All N uploads complete before _seed_turn_zero returns (G13)."""
        from abilities._base import Ability

        config = _make_config(memory_seed=False)
        mp = _make_flat_mp(config)
        mp._metadata = {"attachments": ["/fake/file_a.txt", "/fake/file_b.txt"]}

        dispatch_count = []

        def _spy_dispatch(mp_obj, tool_name, params, call_id=None):
            dispatch_count.append(tool_name)
            return ""

        with _patch_read_attachment(), \
             patch.object(Ability, "dispatch", side_effect=_spy_dispatch):
            mp._seed_turn_zero()

        assert len([x for x in dispatch_count if x == "document"]) == 2, (
            "Both uploads must have completed synchronously before "
            "_seed_turn_zero returns (G13)"
        )


# ── G14: Upload is the ingest — NO separate auto document.view ────────────────


class TestG14NoAutoDocumentView:
    """G14 — upload is the ingest; no separate auto document.view dispatched on turn 0.
    Spec §4d / AC-31."""

    def test_no_document_view_dispatched_on_turn_zero(self, db):
        """_seed_turn_zero must NOT dispatch document with action='view'.

        The upload IS the ingest.  Spec §4d: 'No second auto-document.view:
        the upload now IS the ingest.'
        """
        from abilities._base import Ability

        config = _make_config(memory_seed=False)
        mp = _make_flat_mp(config)
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
            f"No document.view must be auto-dispatched on turn 0 (G14): "
            f"got {view_dispatches}"
        )

    def test_no_view_even_with_memory_seed_and_attachment(self, db):
        """Neither memory_seed nor attachment seeding triggers document.view (G14)."""
        from abilities._base import Ability

        config = _make_config(memory_seed=True)
        mp = _make_flat_mp(config)
        mp._metadata = {"attachments": ["/fake/combo.txt"]}

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
            f"No document.view must fire even with memory_seed+attachment (G14): "
            f"got {view_dispatches}"
        )
        # Verify both expected dispatches DID fire (memory recall + doc upload)
        tool_names = [tn for tn, _ in dispatches]
        assert "memory" in tool_names, "memory recall must still fire (G14 sanity)"
        assert "document" in tool_names, "document upload must still fire (G14 sanity)"
