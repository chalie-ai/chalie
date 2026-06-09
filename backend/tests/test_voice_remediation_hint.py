"""Feature test: voice "not ready" remediation hints.

Voice ONNX models are no longer shipped by the installer. ``c3633e0c`` moved the
download into ``RuntimeDepsService``: turning voice on in Settings
(``PUT /api/voice-settings`` → ``RuntimeDepsService.enable_voice()``) background-
installs the voice deps and downloads the models into ``resources/voice-models/``.

So every user-facing "voice not ready" hint must point the user at that setting —
never at the (now voice-agnostic) installer. This drives the real Flask
``/voice/health`` and ``/voice/synthesize`` endpoints through the production app
+ real SQLite (the ``authed_client`` fixture) and asserts the remediation
contract on whatever readiness branch the environment exercises.

Environment note (same capability-gating as ``tests/integration/test_voice_api.py``):
the ``models_missing`` branch is only reachable when the voice deps are importable
but the model files are absent. Where deps are absent (CI, and any box that has
not enabled voice) the endpoints take the ``deps_missing`` branch instead — the
test still runs and still asserts the cross-cutting "no installer remediation"
invariant on that branch; the ``models_missing``-specific assertion fires
wherever that branch is live.
"""

import pytest


def _no_installer(hint: str) -> None:
    """A voice remediation hint must never tell the user to re-run the installer."""
    assert "installer" not in (hint or "").lower(), (
        "voice remediation hint must not reference the installer "
        "(models download at runtime via Settings → RuntimeDepsService.enable_voice); "
        "got %r" % hint
    )


@pytest.mark.unit
def test_voice_remediation_points_to_settings_not_installer(authed_client):
    """/voice/health (and /voice/synthesize) must steer users to Settings, not the installer."""
    client, _db, _store = authed_client

    # /voice/health always returns 200 JSON with status + (when not ready) a hint.
    health = client.get("/voice/health")
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
            "/voice/synthesize",
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
