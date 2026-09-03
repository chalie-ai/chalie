"""Feature tests for the raw-MIME attachment helpers in imap_handler.py.

Zero mocks. Every test constructs a real email.message.EmailMessage, serialises
it via .as_bytes(), and feeds the bytes straight into the production functions:
extract_attachments, extract_body, attachment_meta, build_outgoing_message and
_bodystructure_has_attachments. Path fixtures live in pytest tmp_path so there
are no stray files and no cross-test coupling.
"""

from __future__ import annotations

import email
import email.message
import email.policy
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from abilities._result import ToolResult

import pytest

from capabilities.mail_capability.imap_handler import (
    _bodystructure_has_attachments,
    _is_attachment,
    attachment_meta,
    build_outgoing_message,
    extract_attachments,
    extract_body,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_with_attachment(
    body: str = "hello",
    data: bytes = b"pdf",
    filename: str = "report.pdf",
    maintype: str = "application",
    subtype: str = "pdf",
) -> bytes:
    """Return a multipart EmailMessage with one text part and one attachment."""
    msg = email.message.EmailMessage()
    msg.set_content(body, subtype="plain", charset="utf-8")
    msg.add_attachment(
        data,
        maintype=maintype,
        subtype=subtype,
        filename=filename,
    )
    return msg.as_bytes()


def _make_body_only(body: str = "hello") -> bytes:
    msg = email.message.EmailMessage()
    msg.set_content(body, subtype="plain", charset="utf-8")
    return msg.as_bytes()


# ---------------------------------------------------------------------------
# 1. extract_attachments — basic single attachment
# ---------------------------------------------------------------------------


def test_extract_attachments_writes_one_file_and_returns_meta(tmp_path: "Path") -> None:
    raw = _make_with_attachment()
    dest = tmp_path / "downloads"
    results = extract_attachments(raw, str(dest))

    assert len(results) == 1
    r = results[0]
    assert r["filename"] == "report.pdf"
    assert r["size"] == 3
    assert r["mime"] == "application/pdf"
    assert os.path.dirname(cast(str, r["path"])) == str(dest)
    assert os.path.exists(cast(str, r["path"]))
    assert Path(cast(str, r["path"])).read_bytes() == b"pdf"


# ---------------------------------------------------------------------------
# 2. Hostile filename — no path traversal
# ---------------------------------------------------------------------------


def test_extract_attachments_sanitises_hostile_filename(tmp_path: "Path") -> None:
    raw = _make_with_attachment(filename="../../evil.txt")
    dest = tmp_path / "dest"
    results = extract_attachments(raw, str(dest))

    assert len(results) == 1
    r = results[0]
    assert os.path.dirname(cast(str, r["path"])) == str(dest)
    fn = cast(str, r["filename"])
    assert "/" not in fn
    assert ".." not in fn


# ---------------------------------------------------------------------------
# 3. Unicode-only name — safe_filename returns "", fallback naming
# ---------------------------------------------------------------------------


def test_extract_attachments_fallback_for_unicode_name(tmp_path: "Path") -> None:
    # Build a message with a Japanese-only filename (no extension) so
    # safe_filename returns "".  Using a bare name like "日本語.png" would
    # yield "png" because werkzeug preserves the extension, so we use
    # "日本語" to actually trigger the fallback path.
    msg = email.message.EmailMessage()
    msg.set_content("body", subtype="plain", charset="utf-8")
    msg.add_attachment(
        b"img",
        maintype="image",
        subtype="png",
        filename="日本語",
    )
    raw = msg.as_bytes()

    dest = tmp_path / "dest"
    results = extract_attachments(raw, str(dest))

    assert len(results) == 1
    r = results[0]
    fn = cast(str, r["filename"])
    # Filename is "attachment_0" + the guessed extension from image/png → .png
    assert fn.startswith("attachment_0")
    assert fn.endswith(".png")
    assert r["size"] == 3
    assert r["mime"] == "image/png"


# ---------------------------------------------------------------------------
# 4. Collision — two attachments named identically
# ---------------------------------------------------------------------------


def test_extract_attachments_handles_collision(tmp_path: "Path") -> None:
    msg = email.message.EmailMessage()
    msg.set_content("body", subtype="plain", charset="utf-8")
    msg.add_attachment(b"a", maintype="application", subtype="pdf", filename="report.pdf")
    msg.add_attachment(b"b", maintype="application", subtype="pdf", filename="report.pdf")
    raw = msg.as_bytes()

    dest = tmp_path / "dest"
    results = extract_attachments(raw, str(dest))

    assert len(results) == 2
    names = sorted(cast(str, r["filename"]) for r in results)
    assert names == ["report.pdf", "report_2.pdf"]


# ---------------------------------------------------------------------------
# 5. message/rfc822 — inner message is preserved as a whole
# ---------------------------------------------------------------------------


def test_extract_attachments_rfc822_inner_subject(tmp_path: "Path") -> None:
    from email.mime.message import MIMEMessage
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    inner = email.message.EmailMessage()
    inner["Subject"] = "DISTINCTIVE_SUBJECT_MARKER"
    inner.set_content("inner body", subtype="plain", charset="utf-8")

    outer = MIMEMultipart()
    outer.attach(MIMEText("outer body", "plain", "utf-8"))
    rfc822_part = MIMEMessage(inner)
    rfc822_part.add_header("Content-Disposition", "attachment", filename="inner.eml")
    outer.attach(rfc822_part)
    raw = outer.as_bytes()

    dest = tmp_path / "dest"
    results = extract_attachments(raw, str(dest))

    assert len(results) == 1
    saved = Path(cast(str, results[0]["path"])).read_bytes()
    inner_parsed = email.message_from_bytes(saved, policy=email.policy.default)
    assert inner_parsed["Subject"] == "DISTINCTIVE_SUBJECT_MARKER"


# ---------------------------------------------------------------------------
# 6. Inline image — still extracted as an attachment
# ---------------------------------------------------------------------------


def test_extract_attachments_inline_image(tmp_path: "Path") -> None:
    msg = email.message.EmailMessage()
    msg.set_content("body", subtype="plain", charset="utf-8")
    msg.add_attachment(
        b"imgdata",
        maintype="image",
        subtype="png",
        filename="pic.png",
        disposition="inline",
    )
    raw = msg.as_bytes()

    dest = tmp_path / "dest"
    results = extract_attachments(raw, str(dest))

    assert len(results) == 1
    r = results[0]
    assert r["filename"] == "pic.png"
    assert r["mime"] == "image/png"
    assert Path(cast(str, r["path"])).read_bytes() == b"imgdata"


# ---------------------------------------------------------------------------
# 7. extract_body regression — body only, not attachment content
# ---------------------------------------------------------------------------


def test_extract_body_does_not_leak_attachment_content() -> None:
    msg = email.message.EmailMessage()
    msg.set_content("the real body", subtype="plain", charset="utf-8")
    msg.add_attachment(b"LEAKED_MARKER", maintype="text", subtype="plain", filename="secret.txt")
    raw = msg.as_bytes()

    body = extract_body(raw)
    assert body == "the real body\n"
    assert "LEAKED_MARKER" not in body


# ---------------------------------------------------------------------------
# 8. attachment_meta — correct listing and empty case
# ---------------------------------------------------------------------------


def test_attachment_meta_on_message_with_attachment() -> None:
    raw = _make_with_attachment()
    meta = attachment_meta(raw)

    assert meta == [
        {"filename": "report.pdf", "size": 3, "mime": "application/pdf"},
    ]


def test_attachment_meta_on_body_only_message() -> None:
    raw = _make_body_only()
    assert attachment_meta(raw) == []


# ---------------------------------------------------------------------------
# 9. build_outgoing_message — headers, attachments, and no-attachment case
# ---------------------------------------------------------------------------


def test_build_outgoing_message_with_attachments(tmp_path: "Path") -> None:
    # Create real source files.
    png_path = str(tmp_path / "pic.png")
    Path(png_path).write_bytes(b"fake-png")
    noext_path = str(tmp_path / "noext")
    Path(noext_path).write_bytes(b"no-ext-data")

    params: dict[str, object] = {
        "to": "to@example.com",
        "subject": "Hello",
        "body": "the body text",
        "cc": "cc@example.com",
        "in_reply_to": "<reply-id@example.com>",
        "attachments": [png_path, noext_path],
    }
    msg = build_outgoing_message("from@example.com", params)

    # Headers round-trip.
    assert str(msg["To"]) == "to@example.com"
    assert str(msg["Subject"]) == "Hello"
    assert str(msg["Cc"]) == "cc@example.com"
    assert str(msg["In-Reply-To"]) == "<reply-id@example.com>"
    assert str(msg["References"]) == "<reply-id@example.com>"

    # Two attachment parts with correct content types and filenames.
    parts = list(msg.walk())
    att_parts = [p for p in parts if _is_attachment(p)]
    assert len(att_parts) == 2

    # Collect filenames and content types.
    names = sorted(cast(str, p.get_filename()) for p in att_parts)
    types = sorted(p.get_content_type() for p in att_parts)
    assert names == ["noext", "pic.png"]
    assert types == ["application/octet-stream", "image/png"]


def test_build_outgoing_message_without_attachments_is_singlepart_and_roundtrips() -> None:
    params: dict[str, object] = {
        "to": "to@example.com",
        "subject": "Solo",
        "body": "just the body",
    }
    msg = build_outgoing_message("from@example.com", params)

    assert not msg.is_multipart()
    # Round-trip: serialize and re-parse, body survives intact.
    round_trip = email.message_from_bytes(msg.as_bytes(), policy=email.policy.default)
    assert round_trip.get_content_type() == "text/plain"
    assert round_trip.get_content() == "just the body\n"


# ---------------------------------------------------------------------------
# 10. _bodystructure_has_attachments — true / false / None
# ---------------------------------------------------------------------------


def test_bodystructure_has_attachments_true_for_imapclient_tuple() -> None:
    bs = (
        ("text", ("plain", ("charset", "utf-8")), None, None, None, None, "500", None),
        ("attachment", ("filename", "x.pdf"), None, None, None, None, "1234", None),
    )
    assert _bodystructure_has_attachments(bs) is True


def test_bodystructure_has_attachments_false_for_text_only() -> None:
    bs = (
        ("text", ("plain", ("charset", "utf-8")), None, None, None, None, "500", None),
    )
    assert _bodystructure_has_attachments(bs) is False


def test_bodystructure_has_attachments_false_for_none() -> None:
    assert _bodystructure_has_attachments(None) is False


# ---------------------------------------------------------------------------
# 11. Guardrail ordering — attachment validation fires before the connected gate
# ---------------------------------------------------------------------------


def _run_send(attachments: object) -> "ToolResult":
    """Run a send with a valid envelope through the full ability entrypoint.

    The test environment has no mail credentials, so a result other than
    ``not-connected`` proves the attachment guardrail fired ahead of the gate."""
    from abilities.email import EmailAbility
    from contracts.params.capability_params_bag import CapabilityParamsBag
    from tests._tool_result_harness import built

    return EmailAbility().run(built(CapabilityParamsBag.from_params({
        "action": "send",
        "to": "user@example.com",
        "subject": "hi",
        "body": "hello",
        "attachments": attachments,
    })))


def test_missing_attachment_beats_the_connected_gate(tmp_path: "Path") -> None:
    missing = str(tmp_path / "nope.pdf")
    result = _run_send([missing])
    assert result.status == "error"
    assert result.code == "attachment-not-found"
    assert missing in cast(str, result.body)


def test_bare_string_attachment_is_coerced_to_a_list(tmp_path: "Path") -> None:
    missing = str(tmp_path / "solo.pdf")
    result = _run_send(missing)
    assert result.status == "error"
    assert result.code == "attachment-not-found"
    assert missing in cast(str, result.body)


def test_non_string_attachment_items_fail_loudly() -> None:
    result = _run_send([123])
    assert result.status == "error"
    assert result.code == "attachment-not-found"
    assert "must be a list of file paths" in cast(str, result.body)


def test_valid_attachment_paths_reach_the_connected_gate(tmp_path: "Path") -> None:
    real = tmp_path / "real.txt"
    real.write_bytes(b"data")
    result = _run_send([str(real)])
    assert result.status == "error"
    assert result.code == "not-connected"
