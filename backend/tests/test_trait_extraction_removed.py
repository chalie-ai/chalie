"""
Regression tests — trait-extraction pipeline removal (2026-04-11).

Every test in this file asserts that a specific piece of the legacy background
trait-extraction pipeline is gone and cannot silently come back.  The tests
inspect imported modules directly; they do not test behaviour through a public
API because there is no public API — the whole pipeline was deleted.

All tests are marked ``unit`` — no network, no file I/O beyond the local
source tree, no external processes.

Architecture decision backing these tests:
  /Volumes/llm/chalie-plans/message-processing.md — postTurn() fan-out
  /Volumes/llm/chalie-plans/v0.3.2/trait-extraction-rip-plan.md
"""

import pytest

pytestmark = pytest.mark.unit


class TestMemorySkillTraitContradictionLift:

    def test_check_trait_contradiction_removed_from_memory_ability(self):
        """_check_trait_contradiction must NOT exist in abilities/memory.py.

        After the data_graph refactor (Part 2), contradiction checking is handled
        entirely inside DataGraphService.store() — the standalone helper was dead
        code and was deleted. The innate_skills/memory_skill.py module itself was
        subsequently deleted and replaced by abilities/memory.py (Phase 4).
        """
        import abilities.memory as mod
        assert not hasattr(mod, "_check_trait_contradiction"), (
            "_check_trait_contradiction still present in abilities/memory — "
            "it should have been deleted as dead code in the data_graph refactor"
        )

    def test_memory_ability_does_not_import_check_trait_contradiction_from_digest_worker(self):
        """Explicit string check: the dead import line must not appear in abilities/memory source."""
        from pathlib import Path
        import abilities.memory as mod
        source = Path(mod.__file__).read_text()
        assert "from workers.digest_worker import _check_trait_contradiction" not in source, (
            "Dead import 'from workers.digest_worker import _check_trait_contradiction' "
            "found in abilities/memory.py — this line was supposed to be removed"
        )
