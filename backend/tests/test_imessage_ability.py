"""Feature tests: the iMessage tool texts the user, and only the user.

The real setup endpoint hands Messages a confirmation for the handle the user
enters and keeps the handle only when Messages takes it. The real dispatcher
and policy gate then decide whether a turn may send, and the real capability
hands the message to Messages for that handle.

Messages is driven through ``osascript``. A stand-in ``osascript`` sits first
on PATH: it records the arguments and the script it is given, then answers the
way the test sets it up (the script's result line, an error on stderr, an exit
code, or a delay). The real ``osascript`` never runs, so no message is sent.
The folder voice notes are copied into for Messages is redirected into
pytest's ``tmp_path``, so no test writes into the real Messages folder. The
vault, database, setup, status and disconnect endpoints, dispatcher and policy
rows are the real ones.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest
from flask.testing import FlaskClient

import capabilities
import capabilities.imessage_capability.capability as capability_module
import services.vault_service as vault_module
from abilities._registry import AbilityRegistry
from abilities.find_tools import FindToolsAbility
from abilities.imessage import ImessageAbility
from configs.channels.external_agent import EAMPConfig
from configs.channels.skill_suggestion import SkillSuggestionConfig
from configs.channels.user import UserConfig
from controllers.message_processor import MessageProcessor
from services.memory_store import MemoryStore
from services.processor_config import ProcessorConfig
from services.vault_service import _vault_state, get_vault_service

pytestmark = pytest.mark.unit

_on_a_mac = pytest.mark.skipif(sys.platform != "darwin", reason="iMessage works only on a Mac")
_not_on_a_mac = pytest.mark.skipif(sys.platform == "darwin", reason="proves what happens where Messages is missing")

_HANDLE = "+15551234567"
_OTHER_HANDLE = "ada@icloud.com"
_VAULT_PASSWORD = "test-password"
_FAILED = "Could not connect iMessage, please try again."
_AUTOMATION_HINT = (
    "Allow the app that runs Chalie (Terminal, or the Chalie app) to control Messages "
    "in System Settings → Privacy & Security → Automation."
)
_OK = "CHALIE_IMESSAGE\tok\n"
_NOT_AUTHORIZED = "CHALIE_IMESSAGE\tfailed\tbefore_send\t-1743\tNot authorized to send Apple events to Messages.\n"
_NO_BUDDY = 'CHALIE_IMESSAGE\tfailed\tbefore_send\t-1728\tMessages got an error: Can’t get buddy id "x".\n'
_SEND_BROKE = "CHALIE_IMESSAGE\tfailed\tafter_send\t-1708\tMessages got an error: The connection was lost.\n"
_VOICE_NOTE_BROKE = "CHALIE_IMESSAGE\tfailed\tafter_text\t-1708\tMessages got an error: The connection was lost.\n"
_FULL_DISK_ACCESS = (
    "Chalie cannot put the voice note where Messages can read it. Allow the app that runs "
    "Chalie (Terminal, or the Chalie app) Full Disk Access in System Settings → Privacy & "
    "Security → Full Disk Access. Nothing was sent."
)
_CLIP_BYTES = b"ID3\x04\x00 a spoken clip"

_STAND_IN = """\
#!{python}
import json
import sys
import time
from pathlib import Path

here = Path(__file__).resolve().parent
script = sys.stdin.read()
with open(here / "calls.jsonl", "a", encoding="utf-8") as calls:
    calls.write(json.dumps({{"argv": sys.argv[1:], "stdin": script}}) + "\\n")
answer = here / "answer.json"
plan = json.loads(answer.read_text(encoding="utf-8")) if answer.exists() else {{}}
time.sleep(plan.get("sleep", 0))
sys.stdout.write(plan.get("stdout", {ok!r}))
sys.stderr.write(plan.get("stderr", ""))
sys.exit(plan.get("exit", 0))
"""


# ── The stand-in osascript ─────────────────────────────────────────────────────


class _Osascript:
    """The stand-in ``osascript``: how it answers, and every call it got."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def answer(self, stdout: str = _OK, stderr: str = "", exit_code: int = 0, sleep: float = 0.0) -> None:
        plan = {"stdout": stdout, "stderr": stderr, "exit": exit_code, "sleep": sleep}
        (self.folder / "answer.json").write_text(json.dumps(plan), encoding="utf-8")

    def calls(self) -> list[dict[str, object]]:
        record = self.folder / "calls.jsonl"
        if not record.exists():
            return []
        lines = record.read_text(encoding="utf-8").splitlines()
        return [cast("dict[str, object]", json.loads(line)) for line in lines]

    def forget_calls(self) -> None:
        (self.folder / "calls.jsonl").unlink(missing_ok=True)


@pytest.fixture()
def osascript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Osascript:
    """The stand-in, first on PATH, so the real ``osascript`` cannot run."""
    folder = tmp_path / "bin"
    folder.mkdir()
    stand_in = folder / "osascript"
    stand_in.write_text(_STAND_IN.format(python=sys.executable, ok=_OK), encoding="utf-8")
    stand_in.chmod(0o755)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ.get('PATH', '')}")
    assert shutil.which("osascript") == str(stand_in), "the stand-in must be the osascript that runs"
    return _Osascript(folder)


@pytest.fixture(autouse=True)
def staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The folder voice notes are copied into for Messages, redirected so no
    test writes into the real Messages folder."""
    root = tmp_path / "Messages" / "Attachments" / "Chalie"
    monkeypatch.setattr(capability_module, "_STAGING_ROOT", root)
    return root


@pytest.fixture()
def clip(tmp_path: Path) -> Path:
    """An audio file, as text_to_speech leaves one in the temp folder."""
    path = tmp_path / "temp" / "chalie_tts_3f2a.mp3"
    path.parent.mkdir()
    path.write_bytes(_CLIP_BYTES)
    return path


@contextmanager
def _open_vault_and_own_registry() -> Iterator[None]:
    """An unlocked vault and a capability registry of this test's own."""
    _vault_state.dek = None
    vault_module._vault_service_instance = None
    vault = get_vault_service()
    vault.initialize(_VAULT_PASSWORD)
    vault.unlock(_VAULT_PASSWORD)
    saved_registry = capabilities._capabilities_cache
    capabilities._capabilities_cache = None
    try:
        yield
    finally:
        capabilities._capabilities_cache = saved_registry
        _vault_state.dek = None
        vault_module._vault_service_instance = None


@pytest.fixture()
def client(authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> Iterator[FlaskClient]:
    """The real app with an unlocked vault and a capability registry of its own."""
    with _open_vault_and_own_registry():
        yield authed_client[0]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _set_up(client: FlaskClient, handle: str) -> int:
    return client.post("/api/capabilities/setup/imessage", json={"handle": handle}).status_code


def _connect(client: FlaskClient, osascript: _Osascript, handle: str = _HANDLE) -> None:
    """The user sets up iMessage and Messages takes the confirmation. Then the
    calls setup made are forgotten, so a test counts only its own."""
    osascript.answer()
    assert _set_up(client, handle) == 204
    assert _card(client)["connected"] is True
    osascript.forget_calls()


def _status(client: FlaskClient) -> dict[str, object]:
    resp = client.get("/api/capabilities/status/imessage")
    assert resp.status_code == 200
    return cast("dict[str, dict[str, object]]", resp.get_json())["result"]


def _cards(client: FlaskClient) -> dict[str, dict[str, object]]:
    resp = client.get("/api/capabilities/all?limit=100")
    assert resp.status_code == 200
    rows = cast("dict[str, list[dict[str, object]]]", resp.get_json())["result"]
    return {str(item["id"]): item for item in rows}


def _card(client: FlaskClient) -> dict[str, object]:
    """The iMessage card's row in the capability list."""
    return _cards(client)["imessage"]


def _stored_rows(db: sqlite3.Connection) -> int:
    return int(db.execute("SELECT COUNT(*) FROM tool_configs WHERE tool_name = 'imessage'").fetchone()[0])


def _send(text: str | None, config: ProcessorConfig | None = None, audio: object = None) -> str:
    """One iMessage call through the real dispatcher and policy gate, as a turn
    on *config*'s channel (a chat turn by default) makes it, passing only the
    fields given. Returns the result the model reads."""
    mp = MessageProcessor(config or UserConfig(), raw_input="text me on iMessage")
    params: dict[str, object] = {}
    if text is not None:
        params["message"] = text
    if audio is not None:
        params["audio"] = audio
    return mp.dispatch_service.dispatch(ImessageAbility.NAME, params)


def _head(result: str) -> str:
    return result.split("\n", 1)[0]


def _argv(call: dict[str, object]) -> list[str]:
    return cast("list[str]", call["argv"])


# ── On a Mac ───────────────────────────────────────────────────────────────────


@_on_a_mac
def test_setup_hands_messages_the_confirmation_and_the_card_names_the_handle(
    client: FlaskClient, db: sqlite3.Connection, osascript: _Osascript,
) -> None:
    osascript.answer()

    assert _set_up(client, f"  {_HANDLE} ") == 204

    calls = osascript.calls()
    assert len(calls) == 1
    assert _argv(calls[0]) == [
        "-l", "AppleScript", "-", _HANDLE, "Chalie is connected. Its messages will come to this conversation.", "",
    ]
    assert "on run argv" in str(calls[0]["stdin"])
    card = _card(client)
    assert card["available"] is True and card["unavailable_reason"] is None
    assert card["connected"] is True
    assert card["status_note"] == f"Messages go to {_HANDLE}."
    status = _status(client)
    assert status["connected"] is True and status["pending"] is False and status["available"] is True
    assert _stored_rows(db) == 1
    detail = client.get("/api/capabilities/imessage")
    assert detail.status_code == 200
    config = cast("dict[str, dict[str, dict[str, object]]]", detail.get_json())["result"]["config"]
    assert config["handle"] == _HANDLE, "the Edit form starts from the handle in use"


@_on_a_mac
def test_a_setup_messages_refuses_keeps_nothing_and_the_card_says_why(
    client: FlaskClient, db: sqlite3.Connection, osascript: _Osascript,
) -> None:
    _connect(client, osascript)

    osascript.answer(stdout=_NO_BUDDY)
    assert _set_up(client, _OTHER_HANDLE) == 422

    assert [_argv(c)[3] for c in osascript.calls()] == [_OTHER_HANDLE]
    assert _stored_rows(db) == 0, "setting up again ends the old connection, and a failure keeps nothing"
    card = _card(client)
    assert card["connected"] is False
    assert card["status_note"] == (
        f'{_FAILED} Messages could not send the message: Messages got an error: Can’t get buddy id "x". '
        "(-1728). Nothing was sent."
    )
    assert "code=not-connected" in _head(_send("Is it set up?"))

    osascript.answer(stdout=_NOT_AUTHORIZED)
    assert _set_up(client, _HANDLE) == 422
    assert _status(client)["status_note"] == f"{_FAILED} {_AUTOMATION_HINT}"
    assert _stored_rows(db) == 0

    osascript.forget_calls()
    assert _set_up(client, "   ") == 422
    assert osascript.calls() == [], "a blank handle is never handed to Messages"
    assert _card(client)["status_note"] == (
        f"{_FAILED} Enter the phone number or Apple ID email you use for iMessage."
    )
    assert _stored_rows(db) == 0


@_on_a_mac
def test_the_handle_and_message_reach_the_script_as_arguments_exactly_as_written(
    client: FlaskClient, db: sqlite3.Connection, osascript: _Osascript,
) -> None:
    _connect(client, osascript)
    text = 'He said "hi" & left"; do shell script "touch /tmp/owned" --\\ \ttab\nline two ✅'

    result = _send(text)

    # The probe turn has no surface to ask on: an ``ask`` row would deny here.
    assert "status=success" in _head(result), result
    assert f"Handed to Messages for {_HANDLE}." in result
    assert "deliver" not in result.lower()
    calls = osascript.calls()
    assert len(calls) == 1
    assert _argv(calls[0]) == ["-l", "AppleScript", "-", _HANDLE, text, ""]
    script = str(calls[0]["stdin"])
    assert "do shell script" not in script and _HANDLE not in script, "nothing sent is ever part of the script"
    assert "item 1 of argv" in script and "item 2 of argv" in script and "item 3 of argv" in script
    assert db.execute("SELECT COUNT(*) FROM policy_blocked_log").fetchone()[0] == 0


@_on_a_mac
def test_messages_not_allowed_to_be_controlled_says_where_to_allow_it(
    client: FlaskClient, osascript: _Osascript,
) -> None:
    _connect(client, osascript)

    osascript.answer(stdout=_NOT_AUTHORIZED)
    from_the_script = _send("First try")
    osascript.answer(
        stdout="", stderr="execution error: Not authorized to send Apple events to Messages. (-1743)\n", exit_code=1,
    )
    from_osascript = _send("Second try")

    for result in (from_the_script, from_osascript):
        assert "status=error" in _head(result), result
        assert _AUTOMATION_HINT in result, result
    assert len(osascript.calls()) == 2


@_on_a_mac
def test_a_failed_send_says_whether_the_message_may_have_gone(
    client: FlaskClient, osascript: _Osascript,
) -> None:
    _connect(client, osascript)

    osascript.answer(stdout=_NO_BUDDY)
    before_sending = _send("Never sent")
    osascript.answer(stdout=_SEND_BROKE)
    while_sending = _send("Maybe sent")
    osascript.answer(stdout="", stderr="0:41: syntax error: Expected end of line. (-2741)\n", exit_code=1)
    no_result_line = _send("Unclear")

    for result in (before_sending, while_sending, no_result_line):
        assert "status=error" in _head(result), result
        assert "Handed to Messages" not in result
    assert "Nothing was sent." in before_sending and "may have been sent" not in before_sending
    assert "Can’t get buddy" in before_sending and "(-1728)" in before_sending
    assert "The connection was lost. (-1708). The message may have been sent." in while_sending
    assert "may have been sent" in no_result_line and "syntax error: Expected end of line" in no_result_line


@_on_a_mac
def test_messages_that_never_answers_is_given_up_on_in_bounded_time(
    client: FlaskClient, osascript: _Osascript, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _connect(client, osascript)
    monkeypatch.setattr(capability_module, "_SEND_TIMEOUT_SECONDS", 0.5)
    osascript.answer(sleep=30)

    started = time.monotonic()
    result = _send("Anyone there?")
    elapsed = time.monotonic() - started

    assert elapsed < 10, f"the send waited {elapsed:.1f}s"
    assert "status=error" in _head(result), result
    assert "Messages did not answer within 0.5 seconds, so the message may have been sent." in result
    assert len(osascript.calls()) == 1


@_on_a_mac
def test_sending_before_setup_or_after_disconnecting_says_so_and_hands_messages_nothing(
    client: FlaskClient, db: sqlite3.Connection, osascript: _Osascript,
) -> None:
    never_set_up = _send("Is anyone there?")
    assert _card(client)["status_note"] is None

    _connect(client, osascript)
    assert client.post("/api/capabilities/disconnect/imessage").status_code == 204
    after_disconnect = _send("Is anyone still there?")

    for result in (never_set_up, after_disconnect):
        assert "code=not-connected" in _head(result), result
        assert f"hint: {ImessageAbility.NOT_CONNECTED_HINT}" in result
    assert osascript.calls() == []
    assert _stored_rows(db) == 0
    card = _card(client)
    assert card["connected"] is False and card["status_note"] is None


@_on_a_mac
def test_an_outside_agent_or_a_background_process_cannot_send(
    client: FlaskClient, db: sqlite3.Connection, osascript: _Osascript,
) -> None:
    _connect(client, osascript)

    for config in (EAMPConfig("reviewer", "sample-project", False), SkillSuggestionConfig()):
        result = _send("Message from a process the user is not watching", config)
        assert "status=success" not in result, result

    assert osascript.calls() == []
    blocked = db.execute("SELECT action_id, context, reason FROM policy_blocked_log ORDER BY id").fetchall()
    assert [tuple(r) for r in blocked] == [
        ("imessage.send", "external_agent", "deny"),
        ("imessage.send", "subconscious", "deny"),
    ]


@_on_a_mac
def test_a_voice_note_is_copied_where_messages_can_read_it_and_sent_after_the_text(
    client: FlaskClient, osascript: _Osascript, staging: Path, clip: Path,
) -> None:
    _connect(client, osascript)

    first = _send("Here is the summary.", audio=str(clip))
    second = _send(None, audio=str(clip))
    third = _send("   ", audio=str(clip))

    for result in (first, second, third):
        assert "status=success" in _head(result), result
        assert f"Handed to Messages for {_HANDLE}." in result
    calls = osascript.calls()
    assert [_argv(c)[:5] for c in calls] == [["-l", "AppleScript", "-", _HANDLE, text] for text in (
        "Here is the summary.", "", "",
    )], "a voice note alone, or with a blank text, sends no text"
    copies = [Path(_argv(c)[5]) for c in calls]
    for copy in copies:
        assert copy.parent.parent == staging, copy
        assert re.fullmatch(r"[0-9a-f]{32}", copy.parent.name), copy
        assert copy.name == clip.name
        assert copy.read_bytes() == _CLIP_BYTES, "Messages is handed a copy of the clip, byte for byte"
    assert len({c.parent for c in copies}) == 3, "every send gets a folder of its own"
    assert clip.read_bytes() == _CLIP_BYTES, "the clip itself is left where it was"
    script = str(calls[0]["stdin"])
    assert str(clip) not in script and "POSIX file theFilePath as alias" in script
    assert script.index("send theMessage to targetBuddy") < script.index("send theFile to targetBuddy")


@_on_a_mac
def test_a_missing_clip_or_nothing_to_send_is_refused_and_messages_gets_nothing(
    client: FlaskClient, osascript: _Osascript, staging: Path, tmp_path: Path,
) -> None:
    _connect(client, osascript)
    gone = tmp_path / "temp" / "chalie_tts_gone.mp3"

    missing = _send("Listen to this", audio=str(gone))
    neither = _send(None)
    blank = _send("  \n ")
    not_text = _send(None, audio=["a.mp3"])

    assert "status=error" in _head(missing), missing
    assert f"There is no audio file at {gone}. Nothing was sent." in missing
    for result in (neither, blank):
        assert "code=missing-params" in _head(result), result
        assert "Pass 'message', 'audio', or both." in result
    assert "code=invalid-param" in _head(not_text), not_text
    assert osascript.calls() == []
    assert not staging.exists(), "nothing is copied for a send that is refused"


@_on_a_mac
def test_without_full_disk_access_a_voice_note_is_refused_and_nothing_is_sent(
    client: FlaskClient, osascript: _Osascript, staging: Path, clip: Path,
) -> None:
    _connect(client, osascript)
    attachments = staging.parent
    attachments.mkdir(parents=True)
    attachments.chmod(0o500)
    try:
        assert not os.access(attachments, os.W_OK), "the folder must refuse writes for this test to mean anything"
        result = _send("Here is the summary.", audio=str(clip))
    finally:
        attachments.chmod(0o700)

    assert "status=error" in _head(result), result
    assert _FULL_DISK_ACCESS in result
    assert osascript.calls() == [], "the text is not sent without its voice note"
    assert not staging.exists()


@_on_a_mac
def test_a_voice_note_that_fails_after_the_text_says_the_text_was_sent(
    client: FlaskClient, osascript: _Osascript, staging: Path, clip: Path,
) -> None:
    _connect(client, osascript)
    osascript.answer(stdout=_VOICE_NOTE_BROKE)

    result = _send("Here is the summary.", audio=str(clip))

    assert "status=error" in _head(result), result
    assert "Handed to Messages" not in result
    assert (
        "Messages failed while sending the voice note: Messages got an error: The connection was lost. (-1708). "
        "The text was sent; the voice note may not have been."
    ) in result
    copy = Path(_argv(osascript.calls()[0])[5])
    assert copy.read_bytes() == _CLIP_BYTES, "the copy is kept: Messages may still read it"


@_on_a_mac
def test_on_a_mac_find_tools_offers_imessage() -> None:
    assert ImessageAbility.NAME in AbilityRegistry.discoverable_names()
    assert AbilityRegistry.discovery_aliases().get(ImessageAbility.NAME) == ImessageAbility.NAME
    assert ImessageAbility.NAME in FindToolsAbility.tools_index()


# ── Anywhere else ──────────────────────────────────────────────────────────────


@_not_on_a_mac
def test_away_from_a_mac_imessage_is_shown_as_apple_only_and_never_reaches_messages(
    client: FlaskClient, db: sqlite3.Connection, osascript: _Osascript,
) -> None:
    cards = _cards(client)
    card = cards["imessage"]
    reason = str(card["unavailable_reason"])
    assert card["available"] is False
    assert "Apple-only" in reason and "Mac" in reason, reason
    assert card["connected"] is False
    others = [c for cap_id, c in cards.items() if cap_id != "imessage"]
    assert others and all(c["available"] is True and c["unavailable_reason"] is None for c in others)
    status = _status(client)
    assert status["available"] is False and status["unavailable_reason"] == reason

    resp = client.post("/api/capabilities/setup/imessage", json={"handle": _HANDLE})
    assert resp.status_code == 422
    assert cast("dict[str, object]", resp.get_json())["error"] == reason
    assert _stored_rows(db) == 0

    assert ImessageAbility.NAME not in AbilityRegistry.discoverable_names()
    assert ImessageAbility.NAME not in AbilityRegistry.discovery_aliases().values()
    assert ImessageAbility.NAME not in FindToolsAbility.tools_index()

    result = _send("Sent anyway")
    assert "status=error" in _head(result) and "code=apple-only" in _head(result), result
    assert reason in result
    assert osascript.calls() == []
