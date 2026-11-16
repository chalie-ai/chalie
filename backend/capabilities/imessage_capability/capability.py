"""iMessage capability — text the user through the Messages app on this Mac.

The user gives the phone number or Apple ID email they use for iMessage, the
handle. Setup hands Messages one confirmation for it and stores the handle only
when Messages takes it, so a setup that cannot send keeps nothing. Every send
goes to that one handle: there is no other recipient to choose. A send is a
text, a voice note made from an audio file, or both, the text first.

Messages is driven through ``osascript``. The script is fixed text read from
stdin; the handle, the message and the audio file's path reach it as arguments
(``on run argv``), so nothing the user or the model writes ever becomes part of
the script. Success means Messages accepted the message, which is all
AppleScript can tell; whether it arrived is not known.

Messages is sandboxed and reliably reads an attachment only from its own
folder, so an audio file is first copied into a folder of its own under
``~/Library/Messages/Attachments/Chalie``. Writing there needs Full Disk
Access for the app that runs Chalie. The copies are kept: the sent message in
the Mac's Messages history points at its copy.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from capabilities.base import AbstractCapability
from capabilities.imessage_capability.availability import APPLE_ONLY, AVAILABLE
from configs.enums.param_key import Keys
from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)

_MANIFEST_PATH = FileMapperService.get_capabilities_path("imessage_capability", "manifest.yaml")

_FAILED = "Could not connect iMessage, please try again."
_CONFIRMATION = "Chalie is connected. Its messages will come to this conversation."
_NO_HANDLE = "Enter the phone number or Apple ID email you use for iMessage."
_NOT_CONNECTED = "iMessage is not connected. Set it up under Capabilities in the Brain dashboard."
_AUTOMATION_HINT = (
    "Allow the app that runs Chalie (Terminal, or the Chalie app) to control Messages "
    "in System Settings → Privacy & Security → Automation."
)
_FULL_DISK_ACCESS_HINT = (
    "Chalie cannot put the voice note where Messages can read it. Allow the app that runs "
    "Chalie (Terminal, or the Chalie app) Full Disk Access in System Settings → Privacy & "
    "Security → Full Disk Access. Nothing was sent."
)

_K_HANDLE = "imessage:handle"

# Long enough for the first send, which waits while macOS asks the user
# whether the app that runs Chalie may control Messages.
_SEND_TIMEOUT_SECONDS = 60

# AppleScript's "not allowed to send Apple events" error.
_NOT_PERMITTED = "-1743"

_RESULT_TAG = "CHALIE_IMESSAGE"

# Inside Messages' own folder, which its sandbox lets it read.
_STAGING_ROOT = Path.home() / "Library" / "Messages" / "Attachments" / "Chalie"

# The service is picked by type: looking a chat up by its id fails on current
# macOS. An empty message or path means that part is not sent. The phase says
# how far the send got: nothing handed over, the first part being handed over,
# or the text taken and the voice note failing. The error number is written out
# by hand because some locales render its minus sign as a character that is
# not "-".
_SEND_SCRIPT = f"""\
on run argv
    set thePhase to "before_send"
    try
        set theHandle to item 1 of argv
        set theMessage to item 2 of argv
        set theFilePath to item 3 of argv
        if theFilePath is not "" then set theFile to POSIX file theFilePath as alias
        tell application "Messages"
            set targetService to first service whose service type is iMessage
            set targetBuddy to buddy theHandle of targetService
            if theMessage is not "" then
                set thePhase to "after_send"
                send theMessage to targetBuddy
                set thePhase to "after_text"
            end if
            if theFilePath is not "" then
                if thePhase is "before_send" then set thePhase to "after_send"
                send theFile to targetBuddy
            end if
        end tell
        return "{_RESULT_TAG}" & tab & "ok"
    on error errorMessage number errorNumber
        return "{_RESULT_TAG}" & tab & "failed" & tab & thePhase & tab & (my asciiNumber(errorNumber)) & tab & errorMessage
    end try
end run

on asciiNumber(n)
    if n < 0 then return "-" & ((0 - n) as text)
    return n as text
end asciiNumber
"""


def _hand_to_messages(handle: str, text: str, attachment: str = "") -> str | None:
    """Hand *text*, then the file at *attachment*, to Messages for *handle*;
    an empty one is not sent. ``None`` when Messages took them, otherwise why
    not, saying so whenever something may have been sent anyway."""
    if not AVAILABLE:
        return APPLE_ONLY
    try:
        done = subprocess.run(
            ["osascript", "-l", "AppleScript", "-", handle, text, attachment],
            input=_SEND_SCRIPT,
            capture_output=True,
            text=True,
            timeout=_SEND_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        return "osascript was not found, so Chalie cannot reach Messages. Nothing was sent."
    except subprocess.TimeoutExpired:
        return (
            f"Messages did not answer within {_SEND_TIMEOUT_SECONDS} seconds, so the message may "
            "have been sent. If macOS is asking whether the app that runs Chalie may control "
            "Messages, allow it."
        )
    return _outcome(done)


def _outcome(done: subprocess.CompletedProcess[str]) -> str | None:
    """Read the one result line the script returns."""
    line = done.stdout.strip()
    if not line.startswith(f"{_RESULT_TAG}\t"):
        detail = done.stderr.strip() or line or f"osascript exited with code {done.returncode}."
        if _NOT_PERMITTED in detail:
            return _AUTOMATION_HINT
        return f"Messages gave no clear answer, so the message may have been sent. {detail}"
    fields = line.split("\t", 4)
    if fields[1:] == ["ok"]:
        return None
    if len(fields) == 5 and fields[1] == "failed":
        _, _, phase, number, message = fields
        if number == _NOT_PERMITTED:
            return _AUTOMATION_HINT
        if phase == "before_send":
            return f"Messages could not send the message: {message} ({number}). Nothing was sent."
        if phase == "after_send":
            return f"Messages failed while sending: {message} ({number}). The message may have been sent."
        if phase == "after_text":
            return (
                f"Messages failed while sending the voice note: {message} ({number}). "
                "The text was sent; the voice note may not have been."
            )
    return "Messages gave an answer Chalie does not understand, so the message may have been sent."


def _stage(audio: str) -> str:
    """Copy *audio* into a new folder of its own under :data:`_STAGING_ROOT`,
    keeping its name, and return the copy's path."""
    folder = _STAGING_ROOT / uuid.uuid4().hex
    folder.mkdir(parents=True)
    return str(shutil.copy2(audio, folder / Path(audio).name))


class ImessageCapability(AbstractCapability):

    def __init__(self) -> None:
        super().__init__(_MANIFEST_PATH)
        self._failure: str | None = None

    def get_id(self) -> str:
        return "imessage"

    def unavailable_reason(self) -> str | None:
        return None if AVAILABLE else APPLE_ONLY

    # ── Lifecycle ────────────────────────────────────────────────────

    def configure(self, credentials: dict[str, object]) -> None:
        """Hand Messages the confirmation for the handle in *credentials*,
        then store the handle. Every setup starts afresh, so the old
        connection ends first; a failure keeps nothing and raises
        ``ValueError`` saying why."""
        self.disconnect()
        handle = str(credentials.get("handle") or "").strip()
        error = _hand_to_messages(handle, _CONFIRMATION) if handle else _NO_HANDLE
        if error:
            self._failure = f"{_FAILED} {error}"
            logger.warning("[imessage] Could not connect: %s", error)
            raise ValueError(error)
        self.store_credential(_K_HANDLE, handle)
        self._connected = True
        logger.info("[imessage] Connected.")

    def connect(self) -> bool:
        """Connected exactly when a handle is stored and this is a Mac.
        Nothing is sent, so a restart never depends on Messages. A locked
        vault raises ``VaultLockedError``, as for every capability."""
        self._connected = AVAILABLE and bool(self.load_credential(_K_HANDLE))
        return self._connected

    def disconnect(self) -> None:
        self._failure = None
        self._connected = False
        self.delete_credentials()
        logger.info("[imessage] Disconnected and the handle removed.")

    def status_note(self) -> str | None:
        """Why the last setup failed, or where messages go."""
        if self._failure:
            return self._failure
        handle = self.load_credential(_K_HANDLE) if self.is_connected() else None
        return f"Messages go to {handle}." if handle else None

    # ── Cognitive pipeline ───────────────────────────────────────────

    def ingest(self) -> list[object]:
        """Chalie reads no iMessage inbox."""
        return []

    def understand(self, items: list[object]) -> list[object]:
        return items

    def _do_monitor(self) -> None:
        """Nothing to poll: Chalie only sends."""
        return

    # ── Sending ──────────────────────────────────────────────────────

    def send(self, message: str, audio: str | None = None) -> dict[str, object]:
        """Hand *message*, then the audio file at *audio* as a voice note, to
        Messages for the user's handle; a blank message is not sent. A success
        names the handle in ``to``; a failure is an ``error`` text."""
        handle = self.load_credential(_K_HANDLE)
        if not handle:
            return {"error": _NOT_CONNECTED}
        attachment = ""
        if audio:
            if not os.path.isfile(audio):
                return {"error": f"There is no audio file at {audio}. Nothing was sent."}
            try:
                attachment = _stage(audio)
            except PermissionError:
                logger.warning("[imessage] Could not copy the voice note into Messages' folder.")
                return {"error": _FULL_DISK_ACCESS_HINT}
        error = _hand_to_messages(handle, message if message.strip() else "", attachment)
        if error:
            logger.warning("[imessage] Send failed: %s", error)
            return {"error": error}
        return {"to": handle}

    # ── Tool handlers ────────────────────────────────────────────────

    def _th_send(self, params: dict[str, object], telemetry: object = None) -> dict[str, object]:
        return self.send(str(params.get(Keys.message) or ""), audio=str(params.get(Keys.audio) or "") or None)

    def get_tools(self) -> list[dict[str, object]]:
        return [
            {"name": "send", "handler": self._th_send, "timeout": _SEND_TIMEOUT_SECONDS,
             "description": "Hand the user an iMessage through Messages: plain text, a voice note, or both.",
             "parameters": {
                 Keys.message: {"type": "string", "description": "The text to send."},
                 Keys.audio: {"type": "string", "description": "The full path of an audio file to send as a voice note."},
             }},
        ]
