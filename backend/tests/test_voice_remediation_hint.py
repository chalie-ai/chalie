"""Feature test: voice remediation hints must point users at reinstalling
Chalie, never at the deleted Settings voice toggle or an installer-only
caveat.

Voice deps (kokoro-onnx, useful-moonshine-onnx, soundfile, noisereduce) and
model assets are unconditional base-install artifacts now — fetched once at
install time (installer/install.sh / the Docker build), never at boot or
runtime. There is no more enable/disable surface to point a broken install
at: the only real remedy left is reinstalling.
"""

import io
import sqlite3

import pytest
from flask.testing import FlaskClient


def _assert_reinstall_oriented(hint: str) -> None:
    assert "Settings" not in hint, (
        "the Settings voice toggle no longer exists — a hint pointing there "
        "sends the user to a dead UI surface; got %r" % hint
    )
    assert "reinstall" in hint.lower(), (
        "voice deps/models are unconditional base-install assets now — the "
        "only real remedy is reinstalling; got %r" % hint
    )


@pytest.mark.unit
def test_voice_remediation_hints_point_to_reinstall_never_settings(
    authed_client: tuple[FlaskClient, sqlite3.Connection, object],
) -> None:
    """/voice/health (and /voice/transcribe, when unavailable) must steer
    users to reinstall Chalie, never a Settings toggle that no longer exists."""
    client, _db, _store = authed_client

    # /voice/health always returns 200 JSON with status + (when not ready) a hint.
    health = client.get("/api/voice/health")
    assert health.status_code == 200
    body = health.get_json()
    assert body is not None
    hdata = body["result"]
    assert hdata.get("status") in ("ok", "loading", "unavailable")

    health_hint = hdata.get("hint")
    if health_hint:
        _assert_reinstall_oriented(health_hint)

    # When voice deps are missing, transcribe surfaces the same remediation.
    # Only probe it in the unavailable state so we never trigger a real (heavy)
    # model load just to read a hint. (Playback needs no model at all — it
    # serves a stored file — so it has no remediation surface to check.)
    if hdata.get("reason") == "deps_missing":
        stt = client.post(
            "/api/voice/transcribe",
            data={"file": (io.BytesIO(b"RIFF....WAVE"), "clip.wav")},
            content_type="multipart/form-data",
        )
        assert stt.status_code == 503
        sdata = stt.get_json()
        assert sdata is not None
        # The failure envelope carries one string, so the hint rides inside it.
        _assert_reinstall_oriented(sdata.get("error") or "")
