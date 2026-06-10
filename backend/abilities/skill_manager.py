"""SkillManagerAbility — SYSTEM-level skill CRUD for background processors."""

from abilities.skill_builder import SkillBuilderAbility


class SkillManagerAbility(SkillBuilderAbility):
    """SYSTEM variant of SkillBuilderAbility that bypasses policy enforcement.

    Used exclusively by SkillSuggestionMessageProcessor. Inherits all handlers
    — and ``run()`` — from SkillBuilderAbility; only get_name() and SYSTEM are
    overridden.

    The ACT trail shows a consistent ``skill_manager`` identity because the
    dispatcher renders the result envelope under ``get_name()``, not a tag name
    baked into the ability body.
    """

    SYSTEM = True

    def get_name(self) -> str:
        return "skill_manager"
