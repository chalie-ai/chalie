"""Feature tests — the ``vision`` delegate tool is fully registered.

Policy defaults seeded by ``policy_defaults.json``:
    (chat, vision) -> allow · (external_agent, vision) -> allow ·
    (subconscious, vision) -> deny.

``vision`` is in ``DELEGATE_TOOLS`` (subtracted on DmnConfig) and in
``DEFAULT_DISCOVERABLE`` (offered on UserConfig).
"""

import sqlite3
from pathlib import Path
from typing import cast

import pytest

from abilities._registry import AbilityRegistry
from abilities.find_tools import FindToolsAbility
from abilities.vision import VisionAbility
from configs.channels import DmnConfig, UserConfig
from configs.channels._common import DELEGATE_TOOLS
from run import _migrate_legacy_policy_rules
from services.database_service import DatabaseService
from services.file_mapper_service import FileMapperService
from services.policy_manager import PolicyManager
from services.processor_config import ProcessorConfig
from services.schema_convergence_service import SchemaConvergenceService
from services.message_processor import MessageProcessor

pytestmark = pytest.mark.unit

_CHANNEL = ProcessorConfig.PolicyChannel


# ---------------------------------------------------------------------------
# Shared real-stack helpers (mirror the channel-isolation feature suite).
# ---------------------------------------------------------------------------

def _mp_for(config: ProcessorConfig) -> MessageProcessor:
    mp = MessageProcessor("find a vision tool for me")
    mp.config = config
    mp.active_tools = list(config.always_available or [])
    return mp


def _find_tools_on(mp: MessageProcessor, params: dict[str, object]) -> object:
    ability = FindToolsAbility()
    ability.mp = mp
    return ability.run(params).body


def _seeded_policy_db(tmp_path: Path) -> PolicyManager:
    db = DatabaseService(str(tmp_path / "vision_policy.db"))
    _migrate_legacy_policy_rules(db)                                  # no-op on fresh
    SchemaConvergenceService(db, embedding_dimensions=256).converge()  # creates policy
    PolicyManager(db).apply_seed()                                    # reads the JSON
    return PolicyManager(db)


# ---------------------------------------------------------------------------
# 1. Policy gate — real seed path, real lookup.
# ---------------------------------------------------------------------------

class TestVisionPolicyDefaults:

    def test_chat_vision_is_allow(self, tmp_path: Path) -> None:
        pm = _seeded_policy_db(tmp_path)
        assert pm._setting(_CHANNEL.CHAT.value, "vision") == "allow"

    def test_external_agent_vision_is_allow(self, tmp_path: Path) -> None:
        """Intentional policy: external_agent vision is allow even though
        external_agent web_search is deny — do not regress this to deny."""
        pm = _seeded_policy_db(tmp_path)
        assert pm._setting(_CHANNEL.EXTERNAL_AGENT.value, "vision") == "allow"

    def test_subconscious_vision_is_deny(self, tmp_path: Path) -> None:
        pm = _seeded_policy_db(tmp_path)
        assert pm._setting(_CHANNEL.SUBCONSCIOUS.value, "vision") == "deny"


# ---------------------------------------------------------------------------
# 2. Tool visibility — real config resolution via the real find_tools dispatch.
# ---------------------------------------------------------------------------

class TestVisionVisibility:

    def test_vision_is_a_delegate_tool(self) -> None:
        """Pin the constant the whole visibility split rests on: a silent edit
        that drops vision from DELEGATE_TOOLS trips this guard."""
        assert "vision" in DELEGATE_TOOLS

    def test_vision_selectable_on_user_channel(self) -> None:
        mp = _mp_for(UserConfig())
        assert "vision" not in mp.config.blocked

        result = _find_tools_on(mp, {"select": ["vision"]})

        assert "vision" in mp.active_tools, (
            f"vision delegate must be selectable on the user channel. "
            f"active_tools={mp.active_tools}"
        )
        # find_tools now returns a structured body: a successful select reports
        # nothing under not_found (vision was injected, not treated as unavailable).
        assert cast(dict[str, object], result)["not_found"] == [], (
            f"vision must not be reported unavailable on the user channel. result={result!r}"
        )

    def test_vision_blocked_on_dmn_background_channel(self) -> None:
        mp = _mp_for(DmnConfig())
        assert "vision" in mp.config.blocked

        result = _find_tools_on(mp, {"select": ["vision"]})

        assert "vision" not in mp.active_tools, (
            f"vision must be blocked on the DMN background channel. "
            f"active_tools={mp.active_tools}"
        )
        assert "not found or unavailable" in cast(str, result).lower()


# ---------------------------------------------------------------------------
# 3. Search index — real rebuilt abilities.sqlite + real registry resolution.
# ---------------------------------------------------------------------------

class TestVisionSearchIndex:

    def test_registry_resolves_vision_to_vision_ability(self) -> None:
        assert isinstance(AbilityRegistry.get("vision"), VisionAbility)

    def test_abilities_sqlite_contains_vision_row(self) -> None:
        db_path = FileMapperService.get_abilities_db_path()
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT name FROM abilities WHERE name = ?", ("vision",)
            ).fetchone()
        finally:
            conn.close()
        assert row is not None, (
            "abilities.sqlite has no 'vision' row — rebuild "
            "(python -m utils.build_ability_db) did not capture VisionAbility."
        )
