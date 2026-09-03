"""Feature tests — the ``vision`` delegate tool is fully registered.

Policy defaults seeded by ``policy_defaults.json``:
    (chat, vision) -> allow · (external_agent, vision) -> allow ·
    (subconscious, vision) -> deny.

``vision`` is ``DISCOVERABLE=True``, so it lives in the single global discovery
roster (``AbilityRegistry.discoverable_names()``) and is selectable via
find_tools on every channel that carries find_tools — including the discovery
background channel. Channel containment for vision is the policy gate above
(subconscious → deny), NOT a discovery block. The policy gate and the discovery
roster are orthogonal.
"""

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import cast

import pytest

from abilities._registry import AbilityRegistry
from abilities.find_tools import FindToolsAbility
from abilities.vision import VisionAbility
from configs.channels import DiscoveryConfig, UserConfig
from contracts.params.find_tools_params_bag import FindToolsParamsBag
from configs.enums.policy_channel import PolicyChannel
from services.database import Database
from services.file_mapper_service import FileMapperService
from controllers.message_processor import MessageProcessor
from services.policy_manager import PolicyManager
from services.processor_config import ProcessorConfig
from services.versioned_database_service import VersionedDatabaseService
from tests._tool_result_harness import built

pytestmark = pytest.mark.unit

_CHANNEL = PolicyChannel


# ---------------------------------------------------------------------------
# Shared real-stack helpers (mirror the channel-isolation feature suite).
# ---------------------------------------------------------------------------

def _mp_for(config: ProcessorConfig) -> MessageProcessor:
    mp = MessageProcessor(config, raw_input="find a vision tool for me")
    mp.active_tools = list(config.always_available or [])
    return mp


def _find_tools_on(mp: MessageProcessor, params: dict[str, object]) -> Mapping[str, object]:
    ability = FindToolsAbility()
    ability.mp = mp
    return cast(Mapping[str, object], ability.run(built(FindToolsParamsBag.from_params(params))).body)


def _seeded_policy_db(tmp_path: Path) -> PolicyManager:
    VersionedDatabaseService().provision()  # creates policy
    PolicyManager().apply_seed()                                  # reads the JSON
    return PolicyManager()


# ---------------------------------------------------------------------------
# 1. Policy gate — real seed path, real lookup.
# ---------------------------------------------------------------------------

class TestVisionPolicyDefaults:

    @pytest.fixture(autouse=True)
    def _gateway_to_tmp_db(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        # provision()/apply_seed()/_setting() all resolve
        # their path through the Database gateway (FileMapperService.get_db_path).
        # Redirect the gateway to the same tmp file so the seed writes and the
        # _setting reads share one database, not the real chalie.db.
        monkeypatch.setattr(FileMapperService, "get_db_path", lambda *_: tmp_path / "vision_policy.db")
        Database.close()
        yield
        Database.close()

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

    def test_vision_is_globally_discoverable(self) -> None:
        """Pin the flag the whole visibility model now rests on: a silent edit
        that flips VisionAbility.DISCOVERABLE to False drops it from the global
        roster and trips this guard."""
        assert "vision" in AbilityRegistry.discoverable_names()

    def test_vision_selectable_on_user_channel(self, db: object) -> None:
        mp = _mp_for(UserConfig())

        result = _find_tools_on(mp, {"query": ["vision"]})

        assert "vision" in mp.active_tools, (
            f"vision delegate must be selectable on the user channel. "
            f"active_tools={mp.active_tools}"
        )
        # find_tools returns a structured body: an exact-name hit pins vision and
        # reports nothing under not_found (vision was injected, not unavailable).
        assert result["not_found"] == [], (
            f"vision must not be reported unavailable on the user channel. result={result!r}"
        )

    def test_vision_selectable_on_background_channel(self, db: object) -> None:
        """Discovery is global: the discovery background channel carries
        find_tools, and vision is DISCOVERABLE=True, so the research loop can
        discover and spawn the vision delegate. Whether a channel may RUN vision
        is the policy gate's job (asserted above), NOT a discovery block."""
        mp = _mp_for(DiscoveryConfig())

        result = _find_tools_on(mp, {"query": ["vision"]})

        assert "vision" in mp.active_tools, (
            f"vision must be discoverable on the discovery background channel. "
            f"active_tools={mp.active_tools}"
        )
        assert result["not_found"] == [], (
            f"vision must not be reported unavailable on the discovery channel. result={result!r}"
        )


# ---------------------------------------------------------------------------
# 3. Registration — vision is discoverable and its aliases resolve correctly.
# ---------------------------------------------------------------------------
class TestVisionSearchIndex:

    def test_vision_is_discoverable_and_aliases_resolve(self) -> None:
        """Pin the visibility model for vision: it must be in the global
        discoverable roster AND its SEARCHABLE_AS aliases must resolve to it
        via the alias map (e.g. "see image" -> "vision")."""
        assert "vision" in AbilityRegistry.discoverable_names()

        aliases = AbilityRegistry.discovery_aliases()
        for alias in VisionAbility.SEARCHABLE_AS:
            canonical = aliases.get(alias)
            assert canonical == "vision", (
                f"alias {alias!r} must resolve to 'vision', got {canonical!r}"
            )
