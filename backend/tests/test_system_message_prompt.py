# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for services.system_message_prompt.

The module exposes one abstract base (``SystemMessagePrompt``) and four
concrete subclasses, each overriding the abstract ``_SYSTEM_PROMPT`` class
constant with an inlined Python string literal. No file I/O, no fallbacks.
"""

import pytest
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
            ScheduledSystemMessagePrompt,
        )
        assert callable(SystemMessagePrompt)
        assert callable(UnifiedSystemMessagePrompt)
        assert callable(DMNSystemMessagePrompt)
        assert callable(ScheduledSystemMessagePrompt)

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
# Abstract base contract
# ─────────────────────────────────────────────────────────────────────────────


class TestSystemMessagePromptBase:
    """The base class is abstract — instantiation must fail, and any subclass
    that forgets to override ``_SYSTEM_PROMPT`` must also fail at construction."""

    def test_base_is_not_instantiable(self):
        """Instantiating the abstract base directly raises TypeError."""
        from services.system_message_prompt import SystemMessagePrompt
        with pytest.raises(TypeError):
            SystemMessagePrompt()  # type: ignore[abstract]

    def test_subclass_without_override_is_not_instantiable(self):
        """A subclass that does not supply ``_SYSTEM_PROMPT`` inherits the
        abstract contract and must raise TypeError on instantiation."""
        from services.system_message_prompt import SystemMessagePrompt

        class _ForgetfulPrompt(SystemMessagePrompt):
            pass

        with pytest.raises(TypeError):
            _ForgetfulPrompt()  # type: ignore[abstract]

    def test_subclass_with_plain_class_attr_is_instantiable(self):
        """A subclass that sets ``_SYSTEM_PROMPT`` as a plain class attribute
        satisfies the abstract contract — that's the whole point of the
        pattern (no @property boilerplate in subclasses)."""
        from services.system_message_prompt import SystemMessagePrompt

        class _GoodPrompt(SystemMessagePrompt):
            _SYSTEM_PROMPT = 'hello'

        assert _GoodPrompt().getPrompt() == 'hello'

    def test_get_prompt_returns_subclass_constant(self):
        """``getPrompt()`` returns ``self._SYSTEM_PROMPT`` verbatim."""
        from services.system_message_prompt import SystemMessagePrompt

        class _Echo(SystemMessagePrompt):
            _SYSTEM_PROMPT = 'echo-body'

        assert _Echo().getPrompt() == 'echo-body'

    def test_get_prompt_idempotent(self):
        """Calling getPrompt() twice on the same instance returns the same value."""
        from services.system_message_prompt import SystemMessagePrompt

        class _Stable(SystemMessagePrompt):
            _SYSTEM_PROMPT = 'stable'

        sut = _Stable()
        assert sut.getPrompt() == sut.getPrompt()


# UnifiedSystemMessagePrompt
# ─────────────────────────────────────────────────────────────────────────────


class TestUnifiedSystemMessagePrompt:
    """Unified prompt is an inlined Python constant (Decision Y1).

    ``getPrompt()`` returns ``_SYSTEM_PROMPT`` directly — no file reads,
    no config loading, no turn-specific state.
    """

    # ── Zero-arg contract (Y1) ────────────────────────────────────────────────

    def test_constructible_with_zero_args(self):
        """UnifiedSystemMessagePrompt() takes no parameters."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        sut = UnifiedSystemMessagePrompt()
        assert isinstance(sut, UnifiedSystemMessagePrompt)

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

    def test_identity_section_present(self):
        """Identity section is inlined at the top of the unified prompt."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        result = UnifiedSystemMessagePrompt().getPrompt()
        assert '## Identity' in result
        assert 'Chalie' in result

    def test_hard_boundaries_section_present(self):
        """Hard Boundaries section is inlined in the unified prompt."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        result = UnifiedSystemMessagePrompt().getPrompt()
        assert '## Hard Boundaries' in result

    def test_core_principles_section_present(self):
        """Core Principles section is inlined in the unified prompt."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        result = UnifiedSystemMessagePrompt().getPrompt()
        assert '## Core Principles' in result

    def test_no_identity_modulation_placeholder(self):
        """``{{identity_modulation}}`` must not appear — identity is inlined,
        not injected via template substitution."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        result = UnifiedSystemMessagePrompt().getPrompt()
        assert '{{identity_modulation}}' not in result

    # ── No side effects on DB / services ──────────────────────────────────────

    def test_get_prompt_returns_non_empty_string(self):
        """Prompt is a Python constant — getPrompt() returns a non-empty string."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        result = UnifiedSystemMessagePrompt().getPrompt()
        assert isinstance(result, str) and len(result) > 0

    # ── Subclassing contract ──────────────────────────────────────────────────

    def test_is_subclass_of_system_message_prompt(self):
        from services.system_message_prompt import (
            SystemMessagePrompt,
            UnifiedSystemMessagePrompt,
        )
        assert issubclass(UnifiedSystemMessagePrompt, SystemMessagePrompt)


# ─────────────────────────────────────────────────────────────────────────────
# Background-channel subclasses — inlined Python constants
# ─────────────────────────────────────────────────────────────────────────────


class TestBackgroundChannelPrompts:
    """DMN, Subagent, and Scheduled each return a non-empty inlined string."""

    def test_dmn_returns_non_empty(self):
        from services.system_message_prompt import DMNSystemMessagePrompt
        result = DMNSystemMessagePrompt().getPrompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_scheduled_returns_non_empty(self):
        from services.system_message_prompt import ScheduledSystemMessagePrompt
        result = ScheduledSystemMessagePrompt().getPrompt()
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize("cls_name", [
        "DMNSystemMessagePrompt",
        "ScheduledSystemMessagePrompt",
    ])
    def test_return_type_is_str(self, cls_name):
        """Each background-channel subclass always returns a str."""
        import services.system_message_prompt as mod
        cls = getattr(mod, cls_name)
        result = cls().getPrompt()
        assert isinstance(result, str)

    @pytest.mark.parametrize("cls_name", [
        "DMNSystemMessagePrompt",
        "ScheduledSystemMessagePrompt",
    ])
    def test_mentions_chalie(self, cls_name):
        """Each background-channel prompt introduces the agent as Chalie."""
        import services.system_message_prompt as mod
        cls = getattr(mod, cls_name)
        result = cls().getPrompt()
        assert 'Chalie' in result

    @pytest.mark.parametrize("cls_name", [
        "DMNSystemMessagePrompt",
        "ScheduledSystemMessagePrompt",
    ])
    def test_idempotent(self, cls_name):
        """Each background-channel prompt is deterministic."""
        import services.system_message_prompt as mod
        cls = getattr(mod, cls_name)
        sut = cls()
        assert sut.getPrompt() == sut.getPrompt()

    @pytest.mark.parametrize("cls_name", [
        "DMNSystemMessagePrompt",
        "ScheduledSystemMessagePrompt",
    ])
    def test_no_file_reads(self, cls_name):
        """Background-channel prompts are inlined Python constants —
        constructing and calling getPrompt() must not read any files."""
        import services.system_message_prompt as mod
        cls = getattr(mod, cls_name)
        with patch('builtins.open') as mock_open:
            cls().getPrompt()
        mock_open.assert_not_called()

# ─────────────────────────────────────────────────────────────────────────────
# Inheritance
# ─────────────────────────────────────────────────────────────────────────────


class TestInheritanceChain:
    """All concrete subclasses are proper subclasses of SystemMessagePrompt."""

    def test_dmn_is_subclass(self):
        from services.system_message_prompt import SystemMessagePrompt, DMNSystemMessagePrompt
        assert issubclass(DMNSystemMessagePrompt, SystemMessagePrompt)

    def test_scheduled_is_subclass(self):
        from services.system_message_prompt import SystemMessagePrompt, ScheduledSystemMessagePrompt
        assert issubclass(ScheduledSystemMessagePrompt, SystemMessagePrompt)

    def test_each_concrete_instance_is_system_message_prompt(self):
        from services.system_message_prompt import (
            SystemMessagePrompt,
            UnifiedSystemMessagePrompt,
            DMNSystemMessagePrompt,
            ScheduledSystemMessagePrompt,
        )
        for cls in (
            UnifiedSystemMessagePrompt,
            DMNSystemMessagePrompt,
            ScheduledSystemMessagePrompt,
        ):
            assert isinstance(cls(), SystemMessagePrompt)
