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
        )
        assert callable(SystemMessagePrompt)
        assert callable(UnifiedSystemMessagePrompt)
        assert callable(DMNSystemMessagePrompt)



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

    def test_subclass_with_plain_class_attr_is_instantiable(self):
        """A subclass that sets ``_SYSTEM_PROMPT`` as a plain class attribute
        satisfies the abstract contract — that's the whole point of the
        pattern (no @property boilerplate in subclasses)."""
        from services.system_message_prompt import SystemMessagePrompt

        class _GoodPrompt(SystemMessagePrompt):
            _SYSTEM_PROMPT = 'hello'

        assert _GoodPrompt().get_prompt() == 'hello'



# UnifiedSystemMessagePrompt
# ─────────────────────────────────────────────────────────────────────────────


class TestUnifiedSystemMessagePrompt:
    """Unified prompt is an inlined Python constant (Decision Y1).

    ``getPrompt()`` returns ``_SYSTEM_PROMPT`` directly — no file reads,
    no config loading, no turn-specific state.
    """

    # ── Zero-arg contract (Y1) ────────────────────────────────────────────────

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
        """Y1: weaving has moved up to UserMessageProcessor.get_system_prompt()."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        assert not hasattr(UnifiedSystemMessagePrompt, '_get_identity_modulation'), (
            "_get_identity_modulation should have been removed — weaving now "
            "lives in UserMessageProcessor.get_system_prompt()"
        )
        assert not hasattr(UnifiedSystemMessagePrompt, '_get_adaptive_directives'), (
            "_get_adaptive_directives should have been removed — weaving now "
            "lives in UserMessageProcessor.get_system_prompt()"
        )

    # ── Prompt constant contract ──────────────────────────────────────────────

    def test_identity_section_present(self):
        """Identity section is inlined at the top of the unified prompt."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        result = UnifiedSystemMessagePrompt().get_prompt()
        assert '## Identity' in result
        assert 'Chalie' in result

    def test_no_identity_modulation_placeholder(self):
        """``{{identity_modulation}}`` must not appear — identity is inlined,
        not injected via template substitution."""
        from services.system_message_prompt import UnifiedSystemMessagePrompt
        result = UnifiedSystemMessagePrompt().get_prompt()
        assert '{{identity_modulation}}' not in result

    # ── No side effects on DB / services ──────────────────────────────────────

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
    """DMN returns a non-empty inlined string."""

    def test_dmn_returns_non_empty(self):
        from services.system_message_prompt import DMNSystemMessagePrompt
        result = DMNSystemMessagePrompt().get_prompt()
        assert isinstance(result, str)
        assert len(result) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Inheritance
# ─────────────────────────────────────────────────────────────────────────────


