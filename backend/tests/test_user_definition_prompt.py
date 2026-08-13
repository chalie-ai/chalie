"""Feature tests for the prompt spine's user-synthesis reads.

``user_definition()`` and the external-agent first-name resolution both read
:class:`models.user_synthesis.UserSynthesisRow` — the versioned table — so the
newest saved version is what every system prompt serves. Real rows on the real
DB; the mp stand-ins carry only the state each method actually touches
(the ``_bare_mp`` idiom).
"""

import sqlite3
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from models.user_synthesis import UserSynthesisRow
from services.prompt_service import _USER_DEFINITION_FALLBACK, PromptService

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

pytestmark = pytest.mark.unit


def _service() -> PromptService:
    """``user_definition`` touches no mp state — an empty stand-in suffices."""
    return PromptService(cast("MessageProcessor", SimpleNamespace()))


def _external_service() -> PromptService:
    """An mp stand-in carrying the external-agent config fields
    ``_external_agent_system_prompt`` reads: the captured agent/project names
    and the producer body with its ``{user_name}`` placeholder."""
    config = SimpleNamespace(
        _agent_name="ferret",
        _project="release notes",
        system_prompt="Address {user_name} on {project_or_task_name} for {agent_name}.",
    )
    return PromptService(cast("MessageProcessor", SimpleNamespace(config=config)))


class TestUserDefinition:
    def test_newest_version_is_served(self, db: sqlite3.Connection) -> None:
        UserSynthesisRow.write("- Name: Sam\n- Keeps bees")
        UserSynthesisRow.write("- Name: Sam\n- Keeps bees and chickens")

        assert _service().user_definition() == "- Name: Sam\n- Keeps bees and chickens"

    def test_empty_table_falls_back_to_peer_line(self, db: sqlite3.Connection) -> None:
        assert _service().user_definition() == _USER_DEFINITION_FALLBACK


class TestExternalAgentUserName:
    def test_prose_synthesis_resolves_the_leading_name(self, db: sqlite3.Connection) -> None:
        """The legacy prose form opened with the name — a clean leading word
        still resolves."""
        UserSynthesisRow.write("Sam is a beekeeper in Gozo.")

        rendered = _external_service()._external_agent_system_prompt()
        assert "Address Sam on release notes for ferret." in rendered

    def test_bullet_synthesis_falls_back_to_the_user(self, db: sqlite3.Connection) -> None:
        """Bullet-form synthesis opens with punctuation — never read a bullet
        marker as the user's name."""
        UserSynthesisRow.write("- Name: Sam\n- Keeps bees")

        rendered = _external_service()._external_agent_system_prompt()
        assert "Address the user on release notes for ferret." in rendered

    def test_empty_synthesis_falls_back_to_the_user(self, db: sqlite3.Connection) -> None:
        rendered = _external_service()._external_agent_system_prompt()
        assert "Address the user on release notes for ferret." in rendered
