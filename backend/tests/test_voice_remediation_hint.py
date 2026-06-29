"""Feature test: every user-facing "voice not ready" hint must point at Settings,
never the installer, across whatever readiness branch the environment exercises.
"""

import sqlite3

import pytest
from flask.testing import FlaskClient


def _no_installer(hint: str) -> None:
    assert "installer" not in (hint or "").lower(), (
        "voice remediation hint must not reference the installer "
        "(models download at runtime via Settings → RuntimeDepsService.enable_voice); "
        "got %r" % hint
    )


@pytest.mark.unit
def test_voice_remediation_points_to_settings_not_installer(authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
    """/voice/health (and /voice/synthesize) must steer users to Settings, not the installer."""
    client, _db, _store = authed_client

    # /voice/health always returns 200 JSON with status + (when not ready) a hint.
    health = client.get("/api/voice/health")
    assert health.status_code == 200
    hdata = health.get_json()
    assert hdata is not None
    assert hdata.get("status") in ("ok", "loading", "unavailable")

    health_hint = hdata.get("hint") or ""
    _no_installer(health_hint)
    if hdata.get("reason") == "models_missing":
        assert "Settings" in health_hint, (
            "models_missing remediation must point the user to enable voice in Settings; "
            "got %r" % health_hint
        )

    # When voice is unavailable, the synthesize route surfaces the same
    # remediation (via _loading_or_missing_response for models_missing, or the
    # deps-unavailable payload otherwise). Only probe it in the unavailable
    # state so we never trigger a real (heavy) model load just to read a hint.
    if hdata.get("status") == "unavailable":
        synth = client.post(
            "/api/voice/synthesize",
            json={"text": "Hello."},
            content_type="application/json",
        )
        assert synth.status_code == 503
        sdata = synth.get_json()
        assert sdata is not None
        synth_hint = sdata.get("hint") or ""
        _no_installer(synth_hint)
        if sdata.get("reason") == "models_missing":
            assert "Settings" in synth_hint, (
                "models_missing remediation must point the user to enable voice in Settings; "
                "got %r" % synth_hint
            )
