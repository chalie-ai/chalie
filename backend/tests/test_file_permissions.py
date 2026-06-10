"""Feature tests for FilePermissionsAbility (abilities/file_permissions.py).

Real filesystem only — no mocks. Uses ``tmp_path`` so every test gets its own
sandbox directory and never touches user files.

Pure-function tests for ``_parse_octal`` and ``_format_octal`` are permitted
under tester.md rule 5 (no collaborators, deterministic).
"""

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
# _parse_octal — pure function, no collaborators
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
# _format_octal — pure function, no collaborators
# ---------------------------------------------------------------------------


def test_format_octal_strips_file_type_bits():
    assert _format_octal(stat.S_IFREG | 0o644) == "0644"


def test_format_octal_preserves_special_bits():
    assert _format_octal(0o4755) == "4755"


# ---------------------------------------------------------------------------
# execute — real filesystem, real behavior
# ---------------------------------------------------------------------------


def _result_payload(result) -> dict:
    """``run()`` returns a ``ToolResult`` whose success ``body`` is the structured
    dict the dispatcher renders as compact JSON. The dispatcher adds the
    ``[file_permissions(...)]`` envelope around it."""
    return result.body


def test_execute_changes_permissions_and_reports_before_after(tmp_path):
    target = tmp_path / "script.sh"
    target.write_text("#!/bin/bash\necho hi\n")
    os.chmod(target, 0o644)

    result = FilePermissionsAbility().run({"path": str(target), "permissions": "755"})
    payload = _result_payload(result)

    assert result.status == "success"
    assert payload["path"] == str(target.resolve())
    assert payload["mode_octal"] == "0755"
    assert payload["mode_symbolic"] == "rwxr-xr-x"
    assert result.meta["previous"] == "0644"
    assert (target.stat().st_mode & 0o7777) == 0o755


def test_execute_accepts_leading_zero_permissions(tmp_path):
    target = tmp_path / "config.yaml"
    target.write_text("k: v\n")
    os.chmod(target, 0o644)

    result = FilePermissionsAbility().run({"path": str(target), "permissions": "0600"})
    assert _result_payload(result)["mode_octal"] == "0600"
    assert (target.stat().st_mode & 0o7777) == 0o600


def test_execute_accepts_symbolic_clause(tmp_path):
    target = tmp_path / "script.sh"
    target.write_text("#!/bin/bash\n")
    os.chmod(target, 0o644)

    result = FilePermissionsAbility().run({"path": str(target), "permissions": "+x"})
    assert result.status == "success"
    assert _result_payload(result)["mode_octal"] == "0755"
    assert (target.stat().st_mode & 0o7777) == 0o755


def test_execute_accepts_intent_keyword(tmp_path):
    target = tmp_path / "key.pem"
    target.write_text("x")
    os.chmod(target, 0o666)

    result = FilePermissionsAbility().run({"path": str(target), "permissions": "owner-only"})
    assert result.status == "success"
    assert _result_payload(result)["mode_octal"] == "0600"
    assert (target.stat().st_mode & 0o7777) == 0o600


def test_execute_works_on_directories(tmp_path):
    target = tmp_path / "subdir"
    target.mkdir()
    os.chmod(target, 0o755)

    result = FilePermissionsAbility().run({"path": str(target), "permissions": "700"})
    assert result.status == "success"
    assert (target.stat().st_mode & 0o7777) == 0o700


# ---------------------------------------------------------------------------
# execute — error paths (error-handling IS the feature here)
# ---------------------------------------------------------------------------


def test_execute_rejects_missing_path():
    result = FilePermissionsAbility().run({"permissions": "755"})
    assert result.status == "error"
    assert result.code == "missing-params"


def test_execute_rejects_missing_permissions(tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("x")
    result = FilePermissionsAbility().run({"path": str(target)})
    assert result.status == "error"
    assert result.code == "missing-params"


def test_execute_rejects_invalid_mode(tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("x")
    os.chmod(target, 0o644)
    result = FilePermissionsAbility().run({"path": str(target), "permissions": "abc"})
    assert result.status == "error"
    assert result.code == "invalid-mode"
    assert result.hint is not None
    # The file mode is unchanged when the value cannot be parsed.
    assert (target.stat().st_mode & 0o7777) == 0o644


def test_execute_rejects_path_not_found(tmp_path):
    missing = tmp_path / "does-not-exist.txt"
    result = FilePermissionsAbility().run({"path": str(missing), "permissions": "644"})
    assert result.status == "error"
    assert result.code == "path-not-found"
    assert str(missing) in result.body
