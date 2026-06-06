"""Feature tests — the ``vision`` delegate tool is fully registered (TKT-838).

Task 7 of the Vision Subagent feature wires an already-built ``VisionAbility``
into three production surfaces. These tests drive the REAL production hot paths
for all three — zero mocks, real DB, real seed file, real config resolution,
real search index — and assert the downstream effects:

1. Policy gate — the real ``PolicyManager.apply_seed()`` reads the real
   ``policy_defaults.json`` into a real (temp) DB exactly as ``run.py`` boots it,
   and the real ``_setting`` lookup returns the seeded vision settings:
       (chat, vision) -> allow · (external_agent, vision) -> allow ·
       (subconscious, vision) -> deny.
2. Tool visibility — the real ``FindToolsAbility.run({"select": ["vision"]})``
   dispatch (the same call ToolDispatcher makes) injects ``vision`` into a
   user-facing config's ``active_tools`` and is rejected on a background config,
   because ``vision`` is in ``DELEGATE_TOOLS`` (subtracted on DmnConfig) yet in
   ``DEFAULT_DISCOVERABLE`` (offered on UserConfig). The real subtraction runs —
   the test never re-implements it.
3. Search index — the rebuilt real ``abilities.sqlite`` carries a row for
   ``vision``, and the real ``AbilityRegistry`` resolves ``"vision"`` to
   ``VisionAbility``. This proves the index rebuild captured the new ability.
"""

import sqlite3

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

_CHANNEL = ProcessorConfig.POLICY_CHANNEL


# ---------------------------------------------------------------------------
# Shared real-stack helpers (mirror the channel-isolation feature suite).
# ---------------------------------------------------------------------------

def _mp_for(config) -> MessageProcessor:
    """A real MessageProcessor bound to a real channel config, seeded exactly
    as ``_setup`` seeds it (active_tools = config.always_available)."""
    mp = MessageProcessor("find a vision tool for me")
    mp.config = config
    mp.active_tools = list(config.always_available or [])
    return mp


def _find_tools_on(mp: MessageProcessor, params: dict) -> str:
    """Drive the real find_tools dispatch path: bind the invoking mp and run."""
    ability = FindToolsAbility()
    ability.mp = mp
    return ability.run(params)


def _seeded_policy_db(tmp_path) -> PolicyManager:
    """Boot a real DB through the exact ``run.py`` _init_database order
    (legacy copy → converge → apply_seed) so the real policy_defaults.json is
    seeded into a real ``policy`` table. Returns a PolicyManager over it."""
    db = DatabaseService(str(tmp_path / "vision_policy.db"))
    _migrate_legacy_policy_rules(db)                                  # no-op on fresh
    SchemaConvergenceService(db, embedding_dimensions=256).converge()  # creates policy
    PolicyManager(db).apply_seed()                                    # reads the JSON
    return PolicyManager(db)


# ---------------------------------------------------------------------------
# 1. Policy gate — real seed path, real lookup.
# ---------------------------------------------------------------------------

class TestVisionPolicyDefaults:

    def test_chat_vision_is_allow(self, tmp_path):
        pm = _seeded_policy_db(tmp_path)
        assert pm._setting(_CHANNEL.CHAT.value, "vision") == "allow"

    def test_external_agent_vision_is_allow(self, tmp_path):
        """Dylan's explicit decision: external_agent vision is allow even though
        external_agent web_search is deny — do not regress this to deny."""
        pm = _seeded_policy_db(tmp_path)
        assert pm._setting(_CHANNEL.EXTERNAL_AGENT.value, "vision") == "allow"

    def test_subconscious_vision_is_deny(self, tmp_path):
        pm = _seeded_policy_db(tmp_path)
        assert pm._setting(_CHANNEL.SUBCONSCIOUS.value, "vision") == "deny"


# ---------------------------------------------------------------------------
# 2. Tool visibility — real config resolution via the real find_tools dispatch.
# ---------------------------------------------------------------------------

class TestVisionVisibility:

    def test_vision_is_a_delegate_tool(self):
        """Pin the constant the whole visibility split rests on: a silent edit
        that drops vision from DELEGATE_TOOLS trips this guard."""
        assert "vision" in DELEGATE_TOOLS

    def test_vision_selectable_on_user_channel(self):
        """vision is in DEFAULT_DISCOVERABLE and NOT subtracted on UserConfig
        (UserConfig.blocked does not include DELEGATE_TOOLS), so the real
        find_tools select path injects it into active_tools."""
        mp = _mp_for(UserConfig())
        assert "vision" not in mp.config.blocked  # contract guard

        result = _find_tools_on(mp, {"select": ["vision"]})

        assert "vision" in mp.active_tools, (
            f"vision delegate must be selectable on the user channel. "
            f"active_tools={mp.active_tools}"
        )
        assert "not found or unavailable" not in result.lower()

    def test_vision_blocked_on_dmn_background_channel(self):
        """DmnConfig.blocked = DELEGATE_TOOLS | ... so vision (a delegate tool)
        is subtracted from discovery on this background loop — the real select
        path must reject it."""
        mp = _mp_for(DmnConfig())
        assert "vision" in mp.config.blocked  # contract guard

        result = _find_tools_on(mp, {"select": ["vision"]})

        assert "vision" not in mp.active_tools, (
            f"vision must be blocked on the DMN background channel. "
            f"active_tools={mp.active_tools}"
        )
        assert "not found or unavailable" in result.lower()


# ---------------------------------------------------------------------------
# 3. Search index — real rebuilt abilities.sqlite + real registry resolution.
# ---------------------------------------------------------------------------

class TestVisionSearchIndex:

    def test_registry_resolves_vision_to_vision_ability(self):
        assert isinstance(AbilityRegistry.get("vision"), VisionAbility)

    def test_abilities_sqlite_contains_vision_row(self):
        """The rebuilt search index must carry a ``vision`` row in the real
        ``abilities`` table — proof the rebuild captured the new ability."""
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
