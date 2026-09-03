"""Phase 4 dispatch-cutover invariant tests."""

import pytest

import abilities._registry as _reg_module
from abilities._ability import Ability

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test 1 — SaveGraph is a real, registered Ability subclass
# ---------------------------------------------------------------------------


def test_save_graph_is_registered_ability() -> None:
    """SaveGraph is a first-class Ability subclass registered"""
    from abilities.save_graph import SaveGraph

    assert issubclass(SaveGraph, Ability)

    registry_names = {a.NAME for a in _reg_module.AbilityRegistry.all()}
    assert "save_graph" in registry_names

    assert isinstance(_reg_module.AbilityRegistry.get("save_graph"), SaveGraph)
