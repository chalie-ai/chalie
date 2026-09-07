"""
Feature tests for the ``user_device`` ability — the model's on-demand view of
the user's device, network, battery and display preferences.

Policy defaults seeded at boot: chat allow · subconscious allow ·
external_agent deny. The tool is discoverable (it rides the find_tools menu
with the tooltip "Returns info regarding user's device") and takes no
parameters. It answers from the persisted client heartbeat: exactly the
nested ``device``/``network``/``battery``/``preferences`` groups, never the
locale scalars or the raw coordinates, and a loud no-results when no
heartbeat has arrived yet.
"""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from tests._tool_result_harness import built

from abilities._registry import AbilityRegistry
from abilities._result import ToolResult
from abilities.find_tools import FindToolsAbility
from abilities.user_device import UserDeviceAbility
from configs.channels import UserConfig
from configs.enums.policy_channel import PolicyChannel
from contracts.params.user_device_params_bag import UserDeviceParamsBag
from services.database import Database
from services.file_mapper_service import FileMapperService
from services.policy_manager import PolicyManager
from services.telemetry_service import TelemetryService
from services.versioned_database_service import VersionedDatabaseService
from tests.test_vision_registration import _find_tools_on, _mp_for

pytestmark = pytest.mark.unit

NAME = UserDeviceAbility.NAME

_HEARTBEAT: dict[str, object] = {
    "timezone": "Europe/Malta",
    "locale": "en-GB",
    "language": "en-US",
    "currency": "EUR",
    "location": {"lat": 35.9, "lon": 14.5},
    "location_name": "Valletta, Malta",
    "saved_at": "2026-09-07T10:00:00+00:00",
    "device": {"class": "desktop", "platform": "macOS", "screen_w": 1728, "screen_h": 1117, "input": "mouse"},
    "network": {"effective_type": "4g", "save_data": False},
    "battery": {"level": 0.82, "charging": True},
    "preferences": {"color_scheme": "dark", "reduced_motion": False},
}


def _run() -> ToolResult:
    return UserDeviceAbility().run(built(UserDeviceParamsBag.from_params({})))


class TestUserDevicePolicyDefaults:
    @pytest.fixture(autouse=True)
    def _gateway_to_tmp_db(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        monkeypatch.setattr(FileMapperService, "get_db_path", lambda *_: tmp_path / "user_device_policy.db")
        Database.close()
        yield
        Database.close()

    @staticmethod
    def _seeded() -> PolicyManager:
        VersionedDatabaseService().provision()
        pm = PolicyManager()
        pm.apply_seed()
        return pm

    def test_chat_and_subconscious_allow(self) -> None:
        pm = self._seeded()
        assert pm._setting(PolicyChannel.CHAT.value, NAME) == "allow"
        assert pm._setting(PolicyChannel.SUBCONSCIOUS.value, NAME) == "allow"

    def test_external_agent_denied(self) -> None:
        pm = self._seeded()
        assert pm._setting(PolicyChannel.EXTERNAL_AGENT.value, NAME) == "deny"


class TestUserDeviceDiscovery:
    def test_discoverable_with_the_exact_tooltip_on_the_menu(self) -> None:
        assert NAME in AbilityRegistry.discoverable_names()
        assert f"- `{NAME}`: Returns info regarding user's device" in FindToolsAbility().get_summary()

    def test_aliases_route_to_the_tool(self) -> None:
        aliases = AbilityRegistry.discovery_aliases()
        for alias in ("device", "battery", "network", "screen"):
            assert aliases[alias] == NAME, alias

    def test_find_tools_loads_it_by_alias(self) -> None:
        mp = _mp_for(UserConfig())
        result = _find_tools_on(mp, {"query": ["battery"]})
        assert result["not_found"] == []
        assert NAME in mp.active_tools

    def test_takes_no_parameters(self) -> None:
        assert UserDeviceAbility().get_parameters() == {"type": "object", "properties": {}}


class TestUserDeviceRun:
    def test_returns_exactly_the_device_groups(self, db: sqlite3.Connection) -> None:
        TelemetryService.write(_HEARTBEAT)
        result = _run()
        assert result.status == "success"
        assert result.body == {
            "device": _HEARTBEAT["device"],
            "network": _HEARTBEAT["network"],
            "battery": _HEARTBEAT["battery"],
            "preferences": _HEARTBEAT["preferences"],
        }

    def test_partial_heartbeat_returns_only_present_groups(self, db: sqlite3.Connection) -> None:
        TelemetryService.write({"timezone": "Europe/Malta", "battery": {"level": 0.1, "charging": False}})
        result = _run()
        assert result.status == "success"
        assert result.body == {"battery": {"level": 0.1, "charging": False}}

    def test_no_heartbeat_is_a_loud_no_results(self, db: sqlite3.Connection) -> None:
        result = _run()
        assert result.status == "error"
        assert result.code == "no-results"
        assert result.hint
