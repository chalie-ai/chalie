"""Feature tests for FilePermissionsAbility (abilities/file_permissions.py).

Real filesystem only — no mocks. Uses ``tmp_path`` so every test gets its own
sandbox directory and never touches user files.
"""

import json
import os
import stat

import pytest

from abilities.file_permissions import (
    FilePermissionsAbility,
    _format_octal,
    _parse_octal,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _parse_octal — accepts 3- or 4-digit octal strings, rejects everything else
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("755", 0o755),
    ("644", 0o644),
    ("600", 0o600),
    ("000", 0o000),
    ("777", 0o777),
    ("0755", 0o755),
    ("0000", 0o000),
    ("7777", 0o7777),
    ("1755", 0o1755),
])
def test_parse_octal_accepts_valid_strings(text, expected):
    assert _parse_octal(text) == expected


@pytest.mark.parametrize("text", [
    "",       # empty
    "75",     # too short
    "75555",  # too long
    "abc",    # not octal
    "789",    # 8/9 are not octal digits
    "7a5",    # mixed
    " 755 ",  # whitespace not stripped here — caller strips
    "-755",   # negative sign
    "0o755",  # python literal prefix
])
def test_parse_octal_rejects_invalid_strings(text):
    assert _parse_octal(text) is None


# ---------------------------------------------------------------------------
# _format_octal — masks to the low 4 octal digits
# ---------------------------------------------------------------------------


def test_format_octal_strips_file_type_bits():
    assert _format_octal(stat.S_IFREG | 0o644) == "0644"


def test_format_octal_preserves_special_bits():
    assert _format_octal(0o4755) == "4755"


# ---------------------------------------------------------------------------
# execute — happy path, returns before + after
# ---------------------------------------------------------------------------


def _result_payload(result: dict) -> dict:
    return json.loads(result["text"])


def test_execute_changes_permissions_and_reports_before_after(tmp_path):
    target = tmp_path / "script.sh"
    target.write_text("#!/bin/bash\necho hi\n")
    os.chmod(target, 0o644)

    result = FilePermissionsAbility().execute(
        "user", {"path": str(target), "permissions": "755"}, None,
    )
    payload = _result_payload(result)

    assert payload["status"] == "success"
    assert payload["path"] == str(target.resolve())
    assert payload["permissions_before"] == "0644"
    assert payload["permissions_after"] == "0755"
    assert (target.stat().st_mode & 0o7777) == 0o755


def test_execute_accepts_leading_zero_permissions(tmp_path):
    target = tmp_path / "config.yaml"
    target.write_text("k: v\n")
    os.chmod(target, 0o644)

    result = FilePermissionsAbility().execute(
        "user", {"path": str(target), "permissions": "0600"}, None,
    )
    assert _result_payload(result)["permissions_after"] == "0600"
    assert (target.stat().st_mode & 0o7777) == 0o600


def test_execute_works_on_directories(tmp_path):
    target = tmp_path / "subdir"
    target.mkdir()
    os.chmod(target, 0o755)

    result = FilePermissionsAbility().execute(
        "user", {"path": str(target), "permissions": "700"}, None,
    )
    assert _result_payload(result)["status"] == "success"
    assert (target.stat().st_mode & 0o7777) == 0o700


# ---------------------------------------------------------------------------
# execute — error contracts
# ---------------------------------------------------------------------------


def test_execute_rejects_missing_path():
    result = FilePermissionsAbility().execute("user", {"permissions": "755"}, None)
    assert _result_payload(result) == {"error": "path-required"}


def test_execute_rejects_missing_permissions(tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("x")
    result = FilePermissionsAbility().execute("user", {"path": str(target)}, None)
    assert _result_payload(result) == {"error": "permissions-required"}


def test_execute_rejects_invalid_octal(tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("x")
    result = FilePermissionsAbility().execute(
        "user", {"path": str(target), "permissions": "abc"}, None,
    )
    payload = _result_payload(result)
    assert payload["error"] == "invalid-permissions"
    assert payload["permissions"] == "abc"


def test_execute_rejects_path_not_found(tmp_path):
    missing = tmp_path / "does-not-exist.txt"
    result = FilePermissionsAbility().execute(
        "user", {"path": str(missing), "permissions": "644"}, None,
    )
    payload = _result_payload(result)
    assert payload["error"] == "path-not-found"
    assert payload["path"] == str(missing)


# ---------------------------------------------------------------------------
# Ability metadata — keep TKT-630 contract stable
# ---------------------------------------------------------------------------


def test_ability_metadata_matches_ticket_spec():
    cls = FilePermissionsAbility
    assert cls.NAME == "file_permissions"
    assert cls.TIMEOUT == 5
    assert cls.POLICY_CATEGORY == "Files"
    assert cls.POLICY_LABELS == {"": "Change file permissions"}
    schema = cls.INPUT_SCHEMA
    assert schema["required"] == ["path", "permissions"]
    assert set(schema["properties"]) == {"path", "permissions"}
    assert "action" not in schema["properties"]


def test_policy_defaults_register_file_permissions_with_correct_matrix():
    from services.policy_service import _build_defaults

    defaults = _build_defaults()
    assert "file_permissions" in defaults, "file_permissions missing from policy defaults"
    matrix = defaults["file_permissions"]
    assert matrix["chat"] == "ask"
    assert matrix["subagent"] == "ask"
    assert matrix["subconscious"] == "deny"
    assert matrix["external_agent"] == "deny"


def test_find_tools_guardrail_steers_to_file_permissions():
    from services.message_processor import MessageProcessor

    guardrail = MessageProcessor._FIND_TOOLS_GUARDRAILS.get("file_permissions", "")
    assert "chmod" in guardrail.lower()
    assert "bash" in guardrail.lower()
    assert "permissions" in guardrail.lower()


def test_file_permissions_in_discoverable_list():
    from services.message_processor import MessageProcessor

    assert "file_permissions" in MessageProcessor.DISCOVERABLE
