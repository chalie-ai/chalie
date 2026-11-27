"""
Feature tests — the locale line at the bottom of every channel's system prompt.

The persisted client heartbeat used to render as a world-state block inside
the user message. It now reaches the model as exactly one line, last in the
system prompt on every channel:

    User is currently in {city}, {country}. The timezone is {CET|PST|...}.
    They prefer {language} and use the currency {currency}. Use this
    information to better tailor your response.

Pinned here: the exact text and its position (after the response-format
contract, at the very end); that it lands on the user channel, a delegate,
an external agent and a bare ProcessorConfig alike; that the timezone is the
current abbreviation (DST-aware) with the IANA key as the fallback for
numeric-offset zones; that a missing heartbeat field drops its clause; that
no heartbeat means no line (never a fabricated default); and that nothing
from the heartbeat rides the user message any more. Real PromptService over
real MessageProcessors, the real telemetry snapshot via the ``db`` fixture.
Zero mocks.
"""

from __future__ import annotations

import re
import sqlite3

import pytest

from configs.channels.code_agent import CodeAgentConfig
from configs.channels.external_agent import EAMPConfig
from configs.channels.user import UserConfig
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig
from services.telemetry_service import TelemetryService
from tests.helpers import make_stub_config

pytestmark = pytest.mark.unit

_TOKYO: dict[str, object] = {
    "timezone": "Asia/Tokyo", "language": "ja-JP", "currency": "JPY", "location_name": "Tokyo, Japan",
}
_TAIL = "Use this information to better tailor your response."
_TOKYO_LINE = (
    "User is currently in Tokyo, Japan. The timezone is JST. They prefer ja-JP "
    f"and use the currency JPY. {_TAIL}"
)


def _system_prompt(config: ProcessorConfig) -> str:
    return MessageProcessor(config, raw_input="what should I wear today").prompt_service.system_prompt()


def test_line_is_the_last_thing_in_the_user_channel_system_prompt(db: sqlite3.Connection) -> None:
    TelemetryService.write(_TOKYO)
    prompt = _system_prompt(UserConfig())
    assert prompt.endswith("\n\n" + _TOKYO_LINE), f"prompt tail={prompt[-300:]!r}"
    assert prompt.count(_TAIL) == 1


def test_line_follows_the_response_format_contract(db: sqlite3.Connection) -> None:
    TelemetryService.write(_TOKYO)
    prompt = _system_prompt(UserConfig())
    assert prompt.index("## Response format") < prompt.index(_TOKYO_LINE)


def test_line_lands_on_every_channel(db: sqlite3.Connection) -> None:
    TelemetryService.write(_TOKYO)
    configs: list[ProcessorConfig] = [
        UserConfig(),
        CodeAgentConfig(PolicyChannel.CHAT),
        EAMPConfig("reviewer", "sample-project", False),
        make_stub_config(channel="delegate:anything"),
    ]
    for config in configs:
        prompt = _system_prompt(config)
        assert prompt.endswith(_TOKYO_LINE), f"{type(config).__name__}: tail={prompt[-200:]!r}"


def test_heartbeat_no_longer_rides_the_user_message(db: sqlite3.Connection) -> None:
    TelemetryService.write(_TOKYO)
    user_prompt = MessageProcessor(UserConfig(), raw_input="anything").prompt_service.user_prompt()
    assert "Tokyo" not in user_prompt and "Asia/Tokyo" not in user_prompt, user_prompt


def test_timezone_is_the_current_abbreviation(db: sqlite3.Connection) -> None:
    TelemetryService.write({"timezone": "Europe/Malta"})
    assert re.search(r"The timezone is CES?T\.", _system_prompt(UserConfig()))


def test_numeric_offset_zone_falls_back_to_the_iana_key(db: sqlite3.Connection) -> None:
    TelemetryService.write({"timezone": "Asia/Dubai"})
    assert "The timezone is Asia/Dubai." in _system_prompt(UserConfig())


def test_missing_fields_drop_their_clause(db: sqlite3.Connection) -> None:
    TelemetryService.write({"timezone": "Asia/Tokyo", "currency": "JPY"})
    prompt = _system_prompt(UserConfig())
    assert prompt.endswith(f"\n\nThe timezone is JST. They use the currency JPY. {_TAIL}")
    assert "User is currently in" not in prompt and "They prefer" not in prompt


def test_no_heartbeat_means_no_line(db: sqlite3.Connection) -> None:
    prompt = _system_prompt(UserConfig())
    assert _TAIL not in prompt
    assert not prompt.endswith("\n\n")
