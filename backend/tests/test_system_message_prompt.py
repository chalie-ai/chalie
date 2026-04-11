# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for services.system_message_prompt (Commit 1).

All tests are purely additive — nothing in production code is changed.
The new module is imported directly; no callers are wired up yet.
"""

import logging
import pytest
from pathlib import Path
from unittest.mock import patch

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Import sanity
# ─────────────────────────────────────────────────────────────────────────────


class TestImport:
    """No subclass should raise on import."""

    def test_import_all_classes(self):
        from services.system_message_prompt import (
            SystemMessagePrompt,
            UnifiedSystemMessagePrompt,
            DMNSystemMessagePrompt,
            GoalPursuitSystemMessagePrompt,
            ScheduledSystemMessagePrompt,
        )
        assert SystemMessagePrompt is not None
        assert UnifiedSystemMessagePrompt is not None
        assert DMNSystemMessagePrompt is not None
        assert GoalPursuitSystemMessagePrompt is not None
        assert ScheduledSystemMessagePrompt is not None

    def test_import_does_not_hit_database(self):
        """Importing the module must not open any DB connections."""
        with patch('services.database_service.get_shared_db_service') as mock_db:
            import importlib
            import services.system_message_prompt as _mod
            importlib.reload(_mod)
            mock_db.assert_not_called()

    def test_import_does_not_spawn_threads(self):
        """Importing the module must not start background threads."""
        import threading
        before = threading.active_count()
        import importlib
        import services.system_message_prompt as _mod
        importlib.reload(_mod)
        after = threading.active_count()
        assert after == before, (
            f"Import spawned {after - before} threads — module must be side-effect free"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────


class TestSystemMessagePromptBase:
    """The abstract base class returns an empty string by default."""

    def test_get_prompt_returns_empty_string(self):
        from services.system_message_prompt import SystemMessagePrompt
        assert SystemMessagePrompt().getPrompt() == ''

    def test_get_prompt_return_type_is_str(self):
        from services.system_message_prompt import SystemMessagePrompt
        result = SystemMessagePrompt().getPrompt()
        assert isinstance(result, str)

    def test_get_prompt_idempotent(self):
        """Calling getPrompt() twice on the same instance returns the same value."""
        from services.system_message_prompt import SystemMessagePrompt
        sut = SystemMessagePrompt()
        assert sut.getPrompt() == sut.getPrompt()


# ─────────────────────────────────────────────────────────────────────────────
# UnifiedSystemMessagePrompt
# ─────────────────────────────────────────────────────────────────────────────


class TestUnifiedSystemMessagePrompt:
    """Unified prompt is an inlined Python constant (Decision Y1).

    ``getPrompt()`` returns ``_UNIFIED_PROMPT`` directly — no file reads,
    no config loading, no turn-specific state. ``{{adaptive_directives}}``
    rides through as a literal placeholder; ``UserMessageProcessor.getSystemPrompt()``
    weaves the per-turn value in before the prompt is sent.
    """

    # ── Zero-arg contract (Y1) ────────────────────────────────────────────────

    def test_constructible_with_zero_args(self):
        """UnifiedSystemMessagePrompt() takes no parameters."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        sut = UnifiedSystemMessagePrompt()
        assert sut is not None

    def test_init_rejects_unexpected_kwargs(self):
        """Passing legacy parameters (original_prompt / thread_id) must raise.

        Y1 tripwire: any caller still relying on the old parameterised shape
        must fail loudly, not silently degrade.
        """
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        with pytest.raises(TypeError):
            UnifiedSystemMessagePrompt(original_prompt='hello')  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            UnifiedSystemMessagePrompt(thread_id='t1')  # type: ignore[call-arg]

    def test_no_identity_or_adaptive_helper_methods(self):
        """Y1: weaving has moved up to UserMessageProcessor.getSystemPrompt()."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        assert not hasattr(UnifiedSystemMessagePrompt, '_get_identity_modulation'), (
            "_get_identity_modulation should have been removed — weaving now "
            "lives in UserMessageProcessor.getSystemPrompt()"
        )
        assert not hasattr(UnifiedSystemMessagePrompt, '_get_adaptive_directives'), (
            "_get_adaptive_directives should have been removed — weaving now "
            "lives in UserMessageProcessor.getSystemPrompt()"
        )

    # ── Prompt constant contract ──────────────────────────────────────────────

    def test_returns_non_empty_string(self):
        """getPrompt() returns a non-empty string without any mocking."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        result = UnifiedSystemMessagePrompt().getPrompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_idempotent(self):
        """Calling getPrompt() twice returns the same value."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        sut = UnifiedSystemMessagePrompt()
        assert sut.getPrompt() == sut.getPrompt()

    def test_adaptive_directives_placeholder_present(self):
        """``{{adaptive_directives}}`` is present for UserMessageProcessor to weave in."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        result = UnifiedSystemMessagePrompt().getPrompt()
        assert '{{adaptive_directives}}' in result

    def test_adaptive_directives_at_bottom(self):
        """``{{adaptive_directives}}`` sits at the end of the prompt (cache-busting suffix)."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        result = UnifiedSystemMessagePrompt().getPrompt()
        idx = result.rfind('{{adaptive_directives}}')
        assert idx != -1
        # Nothing substantive after the placeholder (may have whitespace/newline)
        assert result[idx + len('{{adaptive_directives}}'):].strip() == ''

    def test_no_identity_modulation_placeholder(self):
        """``{{identity_modulation}}`` must not appear — identity is prepended by
        getUserDefinition(), not injected via template substitution."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        result = UnifiedSystemMessagePrompt().getPrompt()
        assert '{{identity_modulation}}' not in result

    def test_no_dead_placeholders(self):
        """No stale {{...}} placeholders other than the two live ones.

        {{voice_modulation}} and {{adaptive_directives}} are intentionally live —
        UserMessageProcessor.getSystemPrompt() weaves them in per-turn.
        Any other {{...}} is a bug.
        """
        import re
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        result = UnifiedSystemMessagePrompt().getPrompt()
        placeholders = set(re.findall(r'\{\{(\w+)\}\}', result))
        allowed = {'adaptive_directives', 'voice_modulation'}
        unexpected = placeholders - allowed
        assert not unexpected, (
            f"Unexpected dead placeholders in _UNIFIED_PROMPT: {unexpected}"
        )

    # ── No side effects on DB / services ──────────────────────────────────────

    def test_get_prompt_does_not_touch_identity_service(self):
        """Y1: the class must not import or instantiate IdentityService."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        with patch('services.identity_service.IdentityService') as mock_ident:
            UnifiedSystemMessagePrompt().getPrompt()
        mock_ident.assert_not_called()

    def test_get_prompt_does_not_touch_adaptive_layer_service(self):
        """Y1: the class must not import or instantiate AdaptiveLayerService."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        with patch('services.adaptive_layer_service.AdaptiveLayerService') as mock_adl:
            UnifiedSystemMessagePrompt().getPrompt()
        mock_adl.assert_not_called()

    def test_get_prompt_does_not_touch_voice_mapper_service(self):
        """Y1: no voice mapper dependency inside the class."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        with patch('services.voice_mapper_service.VoiceMapperService') as mock_vms:
            UnifiedSystemMessagePrompt().getPrompt()
        mock_vms.assert_not_called()

    def test_get_prompt_does_not_touch_working_memory_service(self):
        """Y1: WorkingMemoryService lookup is gone."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        with patch('services.working_memory_service.WorkingMemoryService') as mock_wms:
            UnifiedSystemMessagePrompt().getPrompt()
        mock_wms.assert_not_called()

    def test_get_prompt_does_not_call_load_configs(self):
        """Prompt is a Python constant — no config loading at call time."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        with patch('workers.digest_singletons.load_configs') as mock_lc:
            UnifiedSystemMessagePrompt().getPrompt()
        mock_lc.assert_not_called()

    # ── Subclassing contract ──────────────────────────────────────────────────

    def test_is_subclass_of_system_message_prompt(self):
        from services.system_message_prompt import (
            SystemMessagePrompt,
            UnifiedSystemMessagePrompt,
        )
        assert issubclass(UnifiedSystemMessagePrompt, SystemMessagePrompt)


# ─────────────────────────────────────────────────────────────────────────────
# File-backed subclasses — real files present
# ─────────────────────────────────────────────────────────────────────────────


class TestFileBackedPromptsPresent:
    """Each file-backed subclass returns non-empty content when its file exists."""

    def test_dmn_returns_non_empty(self):
        from services.system_message_prompt import DMNSystemMessagePrompt
        result = DMNSystemMessagePrompt().getPrompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_goal_pursuit_returns_non_empty(self):
        from services.system_message_prompt import GoalPursuitSystemMessagePrompt
        result = GoalPursuitSystemMessagePrompt().getPrompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_scheduled_returns_non_empty(self):
        from services.system_message_prompt import ScheduledSystemMessagePrompt
        result = ScheduledSystemMessagePrompt().getPrompt()
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize("cls_name,expected_tag", [
        ("DMNSystemMessagePrompt", "DMN"),
        ("GoalPursuitSystemMessagePrompt", "GOAL PURSUIT"),
        ("ScheduledSystemMessagePrompt", "SCHEDULED PROMPT"),
    ])
    def test_happy_path_parametrized(self, cls_name, expected_tag):
        """All three file-backed subclasses return a non-empty str from real files."""
        import services.system_message_prompt as mod
        cls = getattr(mod, cls_name)
        result = cls().getPrompt()
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    @pytest.mark.parametrize("cls_name,expected_tag", [
        ("DMNSystemMessagePrompt", "DMN"),
        ("GoalPursuitSystemMessagePrompt", "GOAL PURSUIT"),
        ("ScheduledSystemMessagePrompt", "SCHEDULED PROMPT"),
    ])
    def test_file_content_is_stripped(self, cls_name, expected_tag):
        """getPrompt() returns stripped content — no leading/trailing whitespace."""
        import services.system_message_prompt as mod
        cls = getattr(mod, cls_name)
        result = cls().getPrompt()
        assert result == result.strip(), (
            f"{cls_name}.getPrompt() returned non-stripped content"
        )

    @pytest.mark.parametrize("cls_name,expected_tag", [
        ("DMNSystemMessagePrompt", "DMN"),
        ("GoalPursuitSystemMessagePrompt", "GOAL PURSUIT"),
        ("ScheduledSystemMessagePrompt", "SCHEDULED PROMPT"),
    ])
    def test_return_type_is_str(self, cls_name, expected_tag):
        """Each file-backed subclass always returns a str, never bytes."""
        import services.system_message_prompt as mod
        cls = getattr(mod, cls_name)
        result = cls().getPrompt()
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# File-backed subclasses — missing file fallback
# ─────────────────────────────────────────────────────────────────────────────


# Exact legacy fallback strings, lifted byte-for-byte from the pre-refactor
# _load_system_prompt() functions:
#   backend/services/dmn_message_processor.py
#   backend/services/goal_pursuit_processor.py
#   backend/services/scheduled_message_processor.py
#
# Any change to these strings is a breaking change — Commit 9 wires the
# subclasses into the background processors, and a divergent fallback would
# ship blank/partial DMN / goal-pursuit / scheduled system prompts on any
# prompt-file read failure. These are regression locks.
_DMN_LEGACY_FALLBACK = (
    "You are Chalie, running a background review of recent activity. "
    "Based on the episodes provided, do you detect any patterns, unresolved threads, "
    "or opportunities to be proactive? If so, act on it using the tools available to you. "
    "If nothing actionable stands out, respond with exactly: DMN_NO_ACTION"
)
_GOAL_PURSUIT_LEGACY_FALLBACK = (
    "You are Chalie, a determined assistant. Pursue the goal provided to the best "
    "of your ability. If tool calls fail or errors occur, try alternatives. "
    "When complete, provide a clear summary of what you accomplished."
)
_SCHEDULED_LEGACY_FALLBACK = (
    "You are Chalie, executing a scheduled task. The user set this up earlier "
    "and it is now due. Execute the task to the best of your ability using the "
    "tools available to you. Be concise and action-oriented in your response."
)


class TestFileBackedPromptsMissing:
    """Each file-backed subclass falls back to its legacy hardcoded string and
    logs a warning when its prompt file cannot be read.

    DMN, GoalPursuit, and Scheduled must each return the exact legacy fallback
    string from their pre-refactor ``_load_system_prompt()`` function, so that
    Commit 9 can wire them up without silently blanking the system prompt.
    """

    def _assert_fallback_and_warning(self, cls, expected_fallback, caplog):
        """Patch Path.read_text to raise FileNotFoundError, assert the legacy
        fallback string is returned AND a [TAG] warning is logged."""
        with patch.object(
            Path, 'read_text', side_effect=FileNotFoundError("no such file")
        ):
            with caplog.at_level(logging.WARNING, logger='services.system_message_prompt'):
                result = cls().getPrompt()
        assert result == expected_fallback, (
            f"Expected legacy fallback string, got: {result!r}"
        )
        assert any(
            'Failed to load system prompt' in record.message
            for record in caplog.records
        ), f"Expected WARNING not found in records: {[r.message for r in caplog.records]}"

    def test_dmn_fallback_on_missing_file(self, caplog):
        from services.system_message_prompt import DMNSystemMessagePrompt
        self._assert_fallback_and_warning(
            DMNSystemMessagePrompt, _DMN_LEGACY_FALLBACK, caplog
        )

    def test_goal_pursuit_fallback_on_missing_file(self, caplog):
        from services.system_message_prompt import GoalPursuitSystemMessagePrompt
        self._assert_fallback_and_warning(
            GoalPursuitSystemMessagePrompt, _GOAL_PURSUIT_LEGACY_FALLBACK, caplog
        )

    def test_scheduled_fallback_on_missing_file(self, caplog):
        from services.system_message_prompt import ScheduledSystemMessagePrompt
        self._assert_fallback_and_warning(
            ScheduledSystemMessagePrompt, _SCHEDULED_LEGACY_FALLBACK, caplog
        )

    def test_dmn_fallback_logs_warning_with_tag(self, caplog):
        from services.system_message_prompt import DMNSystemMessagePrompt
        with patch.object(Path, 'read_text', side_effect=OSError("permission denied")):
            with caplog.at_level(logging.WARNING, logger='services.system_message_prompt'):
                DMNSystemMessagePrompt().getPrompt()
        assert any('[DMN]' in r.message for r in caplog.records)

    def test_goal_pursuit_fallback_logs_warning_with_tag(self, caplog):
        from services.system_message_prompt import GoalPursuitSystemMessagePrompt
        with patch.object(Path, 'read_text', side_effect=OSError("permission denied")):
            with caplog.at_level(logging.WARNING, logger='services.system_message_prompt'):
                GoalPursuitSystemMessagePrompt().getPrompt()
        assert any('[GOAL PURSUIT]' in r.message for r in caplog.records)

    def test_scheduled_fallback_logs_warning_with_tag(self, caplog):
        from services.system_message_prompt import ScheduledSystemMessagePrompt
        with patch.object(Path, 'read_text', side_effect=OSError("permission denied")):
            with caplog.at_level(logging.WARNING, logger='services.system_message_prompt'):
                ScheduledSystemMessagePrompt().getPrompt()
        assert any('[SCHEDULED PROMPT]' in r.message for r in caplog.records)

    def test_dmn_fallback_on_permission_error(self, caplog):
        """PermissionError (distinct from FileNotFoundError) also returns the
        legacy fallback string."""
        from services.system_message_prompt import DMNSystemMessagePrompt
        with patch.object(Path, 'read_text', side_effect=PermissionError("access denied")):
            with caplog.at_level(logging.WARNING, logger='services.system_message_prompt'):
                result = DMNSystemMessagePrompt().getPrompt()
        assert result == _DMN_LEGACY_FALLBACK
        assert any('[DMN]' in r.message for r in caplog.records)

    def test_goal_pursuit_fallback_on_permission_error(self, caplog):
        """PermissionError on goal-pursuit prompt file returns the legacy
        fallback string with the correct tag."""
        from services.system_message_prompt import GoalPursuitSystemMessagePrompt
        with patch.object(Path, 'read_text', side_effect=PermissionError("access denied")):
            with caplog.at_level(logging.WARNING, logger='services.system_message_prompt'):
                result = GoalPursuitSystemMessagePrompt().getPrompt()
        assert result == _GOAL_PURSUIT_LEGACY_FALLBACK
        assert any('[GOAL PURSUIT]' in r.message for r in caplog.records)

    def test_scheduled_fallback_on_permission_error(self, caplog):
        """PermissionError on scheduled prompt file returns the legacy
        fallback string with the correct tag."""
        from services.system_message_prompt import ScheduledSystemMessagePrompt
        with patch.object(Path, 'read_text', side_effect=PermissionError("access denied")):
            with caplog.at_level(logging.WARNING, logger='services.system_message_prompt'):
                result = ScheduledSystemMessagePrompt().getPrompt()
        assert result == _SCHEDULED_LEGACY_FALLBACK
        assert any('[SCHEDULED PROMPT]' in r.message for r in caplog.records)

    def test_empty_file_content_returns_empty_string(self):
        """When prompt file exists but is empty on disk, getPrompt() returns ''
        — the empty-but-present file is treated as a deliberately-blank body,
        distinct from the read-failure path which returns the legacy fallback.
        """
        from services.system_message_prompt import DMNSystemMessagePrompt
        with patch.object(Path, 'read_text', return_value=''):
            result = DMNSystemMessagePrompt().getPrompt()
        assert result == '', f"Expected '' for empty file, got {result!r}"

    def test_whitespace_only_file_content_returns_empty_string(self):
        """When prompt file contains only whitespace, getPrompt() returns ''
        because .strip() normalises it to empty. This is distinct from the
        read-failure path which returns the legacy hardcoded fallback.
        """
        from services.system_message_prompt import GoalPursuitSystemMessagePrompt
        with patch.object(Path, 'read_text', return_value='   \n\t\n   '):
            result = GoalPursuitSystemMessagePrompt().getPrompt()
        assert result == '', (
            f"Expected '' for whitespace-only file, got {result!r}"
        )

    @pytest.mark.parametrize("cls_name,exc_type,expected_tag,expected_fallback", [
        ("DMNSystemMessagePrompt",             FileNotFoundError, "DMN",              _DMN_LEGACY_FALLBACK),
        ("GoalPursuitSystemMessagePrompt",     PermissionError,   "GOAL PURSUIT",     _GOAL_PURSUIT_LEGACY_FALLBACK),
        ("ScheduledSystemMessagePrompt",       OSError,           "SCHEDULED PROMPT", _SCHEDULED_LEGACY_FALLBACK),
    ])
    def test_any_exception_type_falls_back_parametrized(
        self, cls_name, exc_type, expected_tag, expected_fallback, caplog
    ):
        """Any exception from Path.read_text falls back to the legacy string
        and logs the right tag."""
        import services.system_message_prompt as mod
        cls = getattr(mod, cls_name)
        with patch.object(Path, 'read_text', side_effect=exc_type("boom")):
            with caplog.at_level(logging.WARNING, logger='services.system_message_prompt'):
                result = cls().getPrompt()
        assert result == expected_fallback
        assert any(f'[{expected_tag}]' in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# Inheritance
# ─────────────────────────────────────────────────────────────────────────────


class TestInheritanceChain:
    """All concrete subclasses are proper subclasses of SystemMessagePrompt."""

    def test_dmn_is_subclass(self):
        from services.system_message_prompt import SystemMessagePrompt, DMNSystemMessagePrompt
        assert issubclass(DMNSystemMessagePrompt, SystemMessagePrompt)

    def test_goal_pursuit_is_subclass(self):
        from services.system_message_prompt import SystemMessagePrompt, GoalPursuitSystemMessagePrompt
        assert issubclass(GoalPursuitSystemMessagePrompt, SystemMessagePrompt)

    def test_scheduled_is_subclass(self):
        from services.system_message_prompt import SystemMessagePrompt, ScheduledSystemMessagePrompt
        assert issubclass(ScheduledSystemMessagePrompt, SystemMessagePrompt)

    def test_each_instance_is_system_message_prompt(self):
        from services.system_message_prompt import (
            SystemMessagePrompt,
            UnifiedSystemMessagePrompt,
            DMNSystemMessagePrompt,
            GoalPursuitSystemMessagePrompt,
            ScheduledSystemMessagePrompt,
        )
        for cls in (
            SystemMessagePrompt,
            UnifiedSystemMessagePrompt,
            DMNSystemMessagePrompt,
            GoalPursuitSystemMessagePrompt,
            ScheduledSystemMessagePrompt,
        ):
            assert isinstance(cls(), SystemMessagePrompt)

    def test_unified_is_not_file_backed(self):
        """UnifiedSystemMessagePrompt must NOT inherit from _FileBackedSystemMessagePrompt —
        it has a completely different construction path."""
        import services.system_message_prompt as mod
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        assert not issubclass(
            UnifiedSystemMessagePrompt, mod._FileBackedSystemMessagePrompt
        ), "UnifiedSystemMessagePrompt should not inherit _FileBackedSystemMessagePrompt"

    def test_dmn_inherits_file_backed(self):
        import services.system_message_prompt as mod
        from services.system_message_prompt import DMNSystemMessagePrompt
        assert issubclass(DMNSystemMessagePrompt, mod._FileBackedSystemMessagePrompt)

    def test_goal_pursuit_inherits_file_backed(self):
        import services.system_message_prompt as mod
        from services.system_message_prompt import GoalPursuitSystemMessagePrompt
        assert issubclass(GoalPursuitSystemMessagePrompt, mod._FileBackedSystemMessagePrompt)

    def test_scheduled_inherits_file_backed(self):
        import services.system_message_prompt as mod
        from services.system_message_prompt import ScheduledSystemMessagePrompt
        assert issubclass(ScheduledSystemMessagePrompt, mod._FileBackedSystemMessagePrompt)


# ─────────────────────────────────────────────────────────────────────────────
# File-backed _FILE_NAME and _LOG_TAG contract
# ─────────────────────────────────────────────────────────────────────────────


class TestFileBackedMetadata:
    """File names and log tags are set correctly on each subclass."""

    def test_dmn_file_name(self):
        from services.system_message_prompt import DMNSystemMessagePrompt
        assert DMNSystemMessagePrompt._FILE_NAME == 'dmn.md'

    def test_goal_pursuit_file_name(self):
        from services.system_message_prompt import GoalPursuitSystemMessagePrompt
        assert GoalPursuitSystemMessagePrompt._FILE_NAME == 'goal-pursuit.md'

    def test_scheduled_file_name(self):
        from services.system_message_prompt import ScheduledSystemMessagePrompt
        assert ScheduledSystemMessagePrompt._FILE_NAME == 'scheduled-prompt.md'

    def test_dmn_log_tag(self):
        from services.system_message_prompt import DMNSystemMessagePrompt
        assert DMNSystemMessagePrompt._LOG_TAG == 'DMN'

    def test_goal_pursuit_log_tag(self):
        from services.system_message_prompt import GoalPursuitSystemMessagePrompt
        assert GoalPursuitSystemMessagePrompt._LOG_TAG == 'GOAL PURSUIT'

    def test_scheduled_log_tag(self):
        from services.system_message_prompt import ScheduledSystemMessagePrompt
        assert ScheduledSystemMessagePrompt._LOG_TAG == 'SCHEDULED PROMPT'

    def test_prompts_dir_resolves_to_backend_prompts(self):
        """The _PROMPTS_DIR constant must resolve to backend/prompts/."""
        import services.system_message_prompt as mod
        resolved = mod._PROMPTS_DIR.resolve()
        assert resolved.name == 'prompts', (
            f"_PROMPTS_DIR should point to 'prompts/', got: {resolved}"
        )
        assert resolved.is_dir(), f"_PROMPTS_DIR {resolved} does not exist"


# Regression: exact fallback strings locked in
#
# REMOVED (Commit 2a, Decision Y1): identity modulation and adaptive directive
# fallback strings no longer live inside UnifiedSystemMessagePrompt. The
# weaving (and therefore the fallbacks) moves up to
# UserMessageProcessor.getSystemPrompt() in Commit 8, where a new regression
# lock should be introduced against that override.
