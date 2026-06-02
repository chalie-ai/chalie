"""SkillManagerAbility — SYSTEM-level skill CRUD for background processors."""

from abilities.skill_builder import SkillBuilderAbility


class SkillManagerAbility(SkillBuilderAbility):
    """SYSTEM variant of SkillBuilderAbility that bypasses policy enforcement.

    Used exclusively by SkillSuggestionMessageProcessor. Inherits all handlers
    from SkillBuilderAbility — only NAME and SYSTEM are overridden.

    ``run()`` post-processes the result to replace the parent's hardcoded
    ``skill_builder`` tag name with ``skill_manager`` so the ACT trail shows
    a consistent tool identity.
    """

    NAME = "skill_manager"
    SYSTEM = True

    def run(self, channel: str, params: dict, telemetry: "dict | None") -> dict:
        """Run inherited handler, then fix tag name in result text."""
        result = super().run(channel, params, telemetry)
        if 'text' in result:
            result['text'] = result['text'].replace(
                'skill_builder', self.NAME,
            )
        return result
