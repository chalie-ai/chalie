"""Feature test: voice remediation hints must point users at reinstalling
Chalie, never at the deleted Settings voice toggle or an installer-only
caveat.

Voice deps (kokoro-onnx, useful-moonshine-onnx, soundfile, noisereduce) and
model assets are unconditional base-install artifacts now — fetched once at
install time (installer/install.sh / the Docker build), never at boot or
runtime. There is no more enable/disable surface to point a broken install
at: the only real remedy left is reinstalling.
"""

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
    """/voice/health (and /voice/synthesize, when unavailable) must steer
    users to reinstall Chalie, never a Settings toggle that no longer exists."""
    client, _db, _store = authed_client

    # /voice/health always returns 200 JSON with status + (when not ready) a hint.
    health = client.get("/api/voice/health")
    assert health.status_code == 200
    hdata = health.get_json()
    assert hdata is not None
    assert hdata.get("status") in ("ok", "loading", "unavailable")

    health_hint = hdata.get("hint")
    if health_hint:
        _assert_reinstall_oriented(health_hint)

    # When voice is unavailable, the synthesize route surfaces the same
    # remediation. Only probe it in the unavailable state so we never trigger
    # a real (heavy) model load just to read a hint.
    if hdata.get("status") == "unavailable":
        synth = client.post(
            "/api/voice/synthesize",
            json={"text": "Hello."},
            content_type="application/json",
        )
        assert synth.status_code == 503
        sdata = synth.get_json()
        assert sdata is not None
        _assert_reinstall_oriented(sdata.get("hint") or "")
