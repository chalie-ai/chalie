"""Guard tests for TKT-731 — the compaction ability is INTERNAL and never LLM-callable.

Compaction execution is routed through the tool machinery so it earns an ACT-trail
pill + a durable audit row, but the ability MUST stay invisible to the model:
  - ``INTERNAL=True``        → excluded from abilities.sqlite (the find_tools index) at build time
  - not in ``policy_visible()`` → never shown in the policy UI
  - not in any tool-scope list  → never presented to the model as a callable tool

These are structural invariants: if a future change drops ``INTERNAL`` or adds
``compaction`` to a scope list, the model could invoke it directly — which the
ticket explicitly forbids.
"""

import pytest

pytestmark = pytest.mark.unit


def test_compaction_ability_is_registered_and_internal():
    """The ability self-registers under NAME 'compaction' and is marked INTERNAL."""
    from abilities._registry import AbilityRegistry
    ability = AbilityRegistry.get('compaction')
    assert ability.NAME == 'compaction'
    assert ability.INTERNAL is True


def test_compaction_excluded_from_policy_visible():
    """INTERNAL abilities never reach the policy UI (their denial would break the loop)."""
    from abilities._registry import AbilityRegistry
    names = {a.NAME for a in AbilityRegistry.policy_visible()}
    assert 'compaction' not in names


def test_compaction_not_presented_to_the_llm():
    """Absent from both tool-scope lists on the chat processor → the model can neither
    call it directly nor surface it through find_tools discovery."""
    from services.message_processor import MessageProcessor
    assert 'compaction' not in MessageProcessor.ALWAYS_AVAILABLE
    assert 'compaction' not in MessageProcessor.DISCOVERABLE
