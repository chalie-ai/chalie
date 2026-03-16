"""
Unit tests for the /api/signals blueprint.

All tests mock external dependencies (auth, WrapperAuthService,
WrapperRateLimiter, emit_reasoning_signal, MemoryClientService) so no
database or reasoning loop is required.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from flask import Flask

from api.signals import signals_bp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_app(wrapper_id=None, rate_limit_allowed=True):
    """Build a minimal Flask test app with the signals blueprint.

    Args:
        wrapper_id: ``None`` for cookie auth, a string for bearer auth.
        rate_limit_allowed: Whether the rate limiter returns True.

    Returns:
        Configured Flask app (not a test client — callers create their own).
    """
    app = Flask(__name__)
    app.register_blueprint(signals_bp)
    app.config["TESTING"] = True
    app.secret_key = "test-secret"
    return app


@pytest.fixture
def cookie_app():
    """App with cookie session auth (g.wrapper_id = None → '__chat_ui__')."""
    return _make_app(wrapper_id=None)


@pytest.fixture
def bearer_app():
    """Factory: returns a function that creates an app with a given wrapper_id."""
    def _factory(wrapper_id="wrp_test"):
        return _make_app(wrapper_id=wrapper_id)
    return _factory


from contextlib import contextmanager

@contextmanager
def _cookie_patches(rate_allowed=True, capabilities_ok=True):
    """Context manager that patches all external deps for cookie-auth paths."""
    from contextlib import ExitStack

    with ExitStack() as stack:
        stack.enter_context(patch("services.auth_session_service.validate_session", return_value=True))

        limiter_mock = MagicMock()
        limiter_mock.is_allowed.return_value = rate_allowed
        stack.enter_context(patch("api.signals._get_rate_limiter", return_value=limiter_mock))

        # Cookie auth → __chat_ui__ → always has capability
        stack.enter_context(patch("api.signals._check_signal_capability", return_value=capabilities_ok))

        emit_mock = MagicMock(return_value=True)
        stack.enter_context(patch("services.reasoning_loop_service.emit_reasoning_signal", emit_mock))

        store_mock = MagicMock()
        stack.enter_context(patch("services.memory_client.MemoryClientService.create_connection", return_value=store_mock))

        yield limiter_mock, emit_mock, store_mock


# ---------------------------------------------------------------------------
# POST /api/signals — single signal
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIngestSignal:
    def test_valid_signal_returns_202(self, cookie_app):
        with cookie_app.test_client() as client:
            with _cookie_patches() as (limiter, emit, store):
                    resp = client.post(
                        "/api/signals",
                        json={
                            "signal_type": "build_failed",
                            "content": "3 test failures in auth module",
                        },
                        content_type="application/json",
                    )
        assert resp.status_code == 202
        data = resp.get_json()
        assert data["ok"] is True
        assert "signal_id" in data
        assert len(data["signal_id"]) > 0

    def test_missing_signal_type_returns_400(self, cookie_app):
        with cookie_app.test_client() as client:
            with _cookie_patches():
                    resp = client.post(
                        "/api/signals",
                        json={"content": "some content"},
                        content_type="application/json",
                    )
        assert resp.status_code == 400
        assert "signal_type" in resp.get_json()["error"]

    def test_blank_signal_type_returns_400(self, cookie_app):
        with cookie_app.test_client() as client:
            with _cookie_patches():
                    resp = client.post(
                        "/api/signals",
                        json={"signal_type": "  ", "content": "x"},
                        content_type="application/json",
                    )
        assert resp.status_code == 400

    def test_missing_content_returns_400(self, cookie_app):
        with cookie_app.test_client() as client:
            with _cookie_patches():
                    resp = client.post(
                        "/api/signals",
                        json={"signal_type": "ctx"},
                        content_type="application/json",
                    )
        assert resp.status_code == 400
        assert "content" in resp.get_json()["error"]

    def test_blank_content_returns_400(self, cookie_app):
        with cookie_app.test_client() as client:
            with _cookie_patches():
                    resp = client.post(
                        "/api/signals",
                        json={"signal_type": "ctx", "content": ""},
                        content_type="application/json",
                    )
        assert resp.status_code == 400

    def test_invalid_activation_energy_type_returns_400(self, cookie_app):
        with cookie_app.test_client() as client:
            with _cookie_patches():
                    resp = client.post(
                        "/api/signals",
                        json={"signal_type": "ctx", "content": "x", "activation_energy": "high"},
                        content_type="application/json",
                    )
        assert resp.status_code == 400

    def test_activation_energy_out_of_range_returns_400(self, cookie_app):
        with cookie_app.test_client() as client:
            with _cookie_patches():
                    resp = client.post(
                        "/api/signals",
                        json={"signal_type": "ctx", "content": "x", "activation_energy": 1.5},
                        content_type="application/json",
                    )
        assert resp.status_code == 400

    def test_metadata_not_dict_returns_400(self, cookie_app):
        with cookie_app.test_client() as client:
            with _cookie_patches():
                    resp = client.post(
                        "/api/signals",
                        json={"signal_type": "ctx", "content": "x", "metadata": ["list"]},
                        content_type="application/json",
                    )
        assert resp.status_code == 400

    def test_unauthenticated_returns_401(self, cookie_app):
        with cookie_app.test_client() as client:
            with patch("services.auth_session_service.validate_session", return_value=False), \
                 patch("services.wrapper_auth_service.WrapperAuthService") as mock_cls:
                mock_cls.return_value.validate_bearer.return_value = None
                resp = client.post(
                    "/api/signals",
                    json={"signal_type": "ctx", "content": "x"},
                    content_type="application/json",
                )
        assert resp.status_code == 401

    def test_rate_limit_exceeded_returns_429(self, cookie_app):
        with cookie_app.test_client() as client:
            with _cookie_patches(rate_allowed=False):
                    resp = client.post(
                        "/api/signals",
                        json={"signal_type": "ctx", "content": "x"},
                        content_type="application/json",
                    )
        assert resp.status_code == 429

    def test_capability_denied_returns_403(self, cookie_app):
        with cookie_app.test_client() as client:
            with _cookie_patches(capabilities_ok=False):
                    resp = client.post(
                        "/api/signals",
                        json={"signal_type": "secret_signal", "content": "x"},
                        content_type="application/json",
                    )
        assert resp.status_code == 403

    def test_high_activation_energy_goes_to_priority_queue(self, cookie_app):
        """activation_energy >= 0.8 should be pushed to reasoning:priority."""
        with cookie_app.test_client() as client:
            with _cookie_patches() as (limiter, emit, store):
                    resp = client.post(
                        "/api/signals",
                        json={
                            "signal_type": "critical_event",
                            "content": "production down",
                            "activation_energy": 0.9,
                        },
                        content_type="application/json",
                    )
        assert resp.status_code == 202
        # emit_reasoning_signal should NOT have been called (priority path used)
        emit.assert_not_called()
        # Priority queue should have received the signal
        store.rpush.assert_called_once_with("reasoning:priority", pytest.approx(store.rpush.call_args[0][1], abs=1))

    def test_normal_activation_energy_uses_emit(self, cookie_app):
        """activation_energy < 0.8 should use emit_reasoning_signal."""
        with cookie_app.test_client() as client:
            with _cookie_patches() as (limiter, emit, store):
                    resp = client.post(
                        "/api/signals",
                        json={
                            "signal_type": "context_change",
                            "content": "user opened new file",
                            "activation_energy": 0.5,
                        },
                        content_type="application/json",
                    )
        assert resp.status_code == 202
        emit.assert_called_once()

    def test_default_activation_energy_is_0_5(self, cookie_app):
        """When activation_energy is omitted it defaults to 0.5."""
        with cookie_app.test_client() as client:
            with _cookie_patches() as (limiter, emit, store):
                    resp = client.post(
                        "/api/signals",
                        json={"signal_type": "ctx", "content": "x"},
                        content_type="application/json",
                    )
        assert resp.status_code == 202
        # 0.5 < 0.8 → normal emit path
        emit.assert_called_once()

    def test_full_valid_payload_accepted(self, cookie_app):
        """All optional fields should be accepted without error."""
        with cookie_app.test_client() as client:
            with _cookie_patches():
                    resp = client.post(
                        "/api/signals",
                        json={
                            "signal_type": "build_failed",
                            "source": "ci-pipeline",
                            "content": "3 test failures in auth module",
                            "topic": "development",
                            "activation_energy": 0.7,
                            "metadata": {"branch": "feature/auth"},
                        },
                        content_type="application/json",
                    )
        assert resp.status_code == 202


# ---------------------------------------------------------------------------
# Bearer auth capability checks
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBearerCapabilityCheck:
    def _make_bearer_app_and_client(self, wrapper_id, allowed_signals):
        """Build an app and test client with bearer auth returning wrapper_id."""
        app = Flask(__name__)
        app.register_blueprint(signals_bp)
        app.config["TESTING"] = True
        app.secret_key = "test-secret"

        bearer_svc = MagicMock()
        bearer_svc.validate_bearer.return_value = wrapper_id
        bearer_svc.get_wrapper.return_value = {
            "wrapper_id": wrapper_id,
            "capabilities": {"signals": allowed_signals},
            "permissions": {},
        }

        return app, bearer_svc

    def test_bearer_allowed_signal_type_returns_202(self):
        app, bearer_svc = self._make_bearer_app_and_client(
            "wrp_ci", ["build_failed", "test_passed"]
        )

        limiter_mock = MagicMock()
        limiter_mock.is_allowed.return_value = True
        emit_mock = MagicMock(return_value=True)
        store_mock = MagicMock()

        with patch("services.auth_session_service.validate_session", return_value=False), \
             patch("services.wrapper_auth_service.WrapperAuthService", return_value=bearer_svc), \
             patch("api.signals._get_wrapper_service", return_value=bearer_svc), \
             patch("api.signals._get_rate_limiter", return_value=limiter_mock), \
             patch("services.reasoning_loop_service.emit_reasoning_signal", emit_mock), \
             patch("services.memory_client.MemoryClientService.create_connection", return_value=store_mock):
            with app.test_client() as client:
                resp = client.post(
                    "/api/signals",
                    json={"signal_type": "build_failed", "content": "tests failed"},
                    headers={"Authorization": "Bearer fake_token"},
                    content_type="application/json",
                )
        assert resp.status_code == 202

    def test_bearer_disallowed_signal_type_returns_403(self):
        app, bearer_svc = self._make_bearer_app_and_client(
            "wrp_ci", ["build_failed"]  # does NOT include "deploy_triggered"
        )

        limiter_mock = MagicMock()
        limiter_mock.is_allowed.return_value = True

        with patch("services.auth_session_service.validate_session", return_value=False), \
             patch("services.wrapper_auth_service.WrapperAuthService", return_value=bearer_svc), \
             patch("api.signals._get_wrapper_service", return_value=bearer_svc), \
             patch("api.signals._get_rate_limiter", return_value=limiter_mock):
            with app.test_client() as client:
                resp = client.post(
                    "/api/signals",
                    json={"signal_type": "deploy_triggered", "content": "deployed to prod"},
                    headers={"Authorization": "Bearer fake_token"},
                    content_type="application/json",
                )
        assert resp.status_code == 403

    def test_bearer_empty_signals_list_denies_all(self):
        app, bearer_svc = self._make_bearer_app_and_client("wrp_empty", [])

        limiter_mock = MagicMock()
        limiter_mock.is_allowed.return_value = True

        with patch("services.auth_session_service.validate_session", return_value=False), \
             patch("services.wrapper_auth_service.WrapperAuthService", return_value=bearer_svc), \
             patch("api.signals._get_wrapper_service", return_value=bearer_svc), \
             patch("api.signals._get_rate_limiter", return_value=limiter_mock):
            with app.test_client() as client:
                resp = client.post(
                    "/api/signals",
                    json={"signal_type": "anything", "content": "x"},
                    headers={"Authorization": "Bearer fake_token"},
                    content_type="application/json",
                )
        assert resp.status_code == 403

    def test_bearer_revoked_wrapper_returns_403(self):
        """A revoked / unknown wrapper should be denied capability."""
        app = Flask(__name__)
        app.register_blueprint(signals_bp)
        app.config["TESTING"] = True
        app.secret_key = "test-secret"

        auth_svc = MagicMock()
        auth_svc.validate_bearer.return_value = "wrp_gone"
        # get_wrapper returns None → wrapper revoked or not found
        wrapper_svc = MagicMock()
        wrapper_svc.get_wrapper.return_value = None

        limiter_mock = MagicMock()
        limiter_mock.is_allowed.return_value = True

        with patch("services.auth_session_service.validate_session", return_value=False), \
             patch("services.wrapper_auth_service.WrapperAuthService", return_value=auth_svc), \
             patch("api.signals._get_wrapper_service", return_value=wrapper_svc), \
             patch("api.signals._get_rate_limiter", return_value=limiter_mock):
            with app.test_client() as client:
                resp = client.post(
                    "/api/signals",
                    json={"signal_type": "ctx", "content": "x"},
                    headers={"Authorization": "Bearer fake_token"},
                    content_type="application/json",
                )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# POST /api/signals/batch
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIngestSignalsBatch:
    def test_valid_batch_returns_200_all_accepted(self, cookie_app):
        signals = [
            {"signal_type": "build_failed", "content": "err1"},
            {"signal_type": "build_failed", "content": "err2"},
        ]
        with cookie_app.test_client() as client:
            with _cookie_patches():
                    resp = client.post(
                        "/api/signals/batch",
                        json=signals,
                        content_type="application/json",
                    )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["accepted"] == 2
        assert data["rejected"] == 0
        assert data["errors"] == []

    def test_empty_batch_returns_200_zero_accepted(self, cookie_app):
        with cookie_app.test_client() as client:
            with _cookie_patches():
                    resp = client.post(
                        "/api/signals/batch",
                        json=[],
                        content_type="application/json",
                    )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["accepted"] == 0
        assert data["rejected"] == 0

    def test_batch_with_invalid_items_reports_errors(self, cookie_app):
        signals = [
            {"signal_type": "ok", "content": "valid"},       # index 0: valid
            {"content": "missing signal_type"},               # index 1: invalid
            {"signal_type": "also_ok", "content": "valid"},  # index 2: valid
        ]
        with cookie_app.test_client() as client:
            with _cookie_patches():
                    resp = client.post(
                        "/api/signals/batch",
                        json=signals,
                        content_type="application/json",
                    )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["accepted"] == 2
        assert data["rejected"] == 1
        assert len(data["errors"]) == 1
        assert data["errors"][0]["index"] == 1

    def test_batch_not_array_returns_400(self, cookie_app):
        with cookie_app.test_client() as client:
            with _cookie_patches():
                    resp = client.post(
                        "/api/signals/batch",
                        json={"signal_type": "x", "content": "x"},
                        content_type="application/json",
                    )
        assert resp.status_code == 400

    def test_batch_exceeds_max_returns_400(self, cookie_app):
        signals = [{"signal_type": "x", "content": "c"}] * 51
        with cookie_app.test_client() as client:
            with _cookie_patches():
                    resp = client.post(
                        "/api/signals/batch",
                        json=signals,
                        content_type="application/json",
                    )
        assert resp.status_code == 400

    def test_batch_at_exactly_50_is_accepted(self, cookie_app):
        signals = [{"signal_type": "x", "content": "c"}] * 50
        with cookie_app.test_client() as client:
            with _cookie_patches():
                    resp = client.post(
                        "/api/signals/batch",
                        json=signals,
                        content_type="application/json",
                    )
        assert resp.status_code == 200
        assert resp.get_json()["accepted"] == 50

    def test_batch_rate_limit_rejects_subsequent_items(self, cookie_app):
        """Once the rate limit is hit, remaining items in batch are rejected."""
        signals = [{"signal_type": "x", "content": "c"}] * 3

        # Allow first 2 calls, deny the 3rd
        limiter_mock = MagicMock()
        limiter_mock.is_allowed.side_effect = [True, True, False]

        with cookie_app.test_client() as client:
            with patch("services.auth_session_service.validate_session", return_value=True), \
                 patch("api.signals._get_rate_limiter", return_value=limiter_mock), \
                 patch("api.signals._check_signal_capability", return_value=True), \
                 patch("services.reasoning_loop_service.emit_reasoning_signal", return_value=True), \
                 patch("services.memory_client.MemoryClientService.create_connection", return_value=MagicMock()):
                resp = client.post(
                    "/api/signals/batch",
                    json=signals,
                    content_type="application/json",
                )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["accepted"] == 2
        assert data["rejected"] == 1
        assert data["errors"][0]["index"] == 2
        assert "rate limit" in data["errors"][0]["error"]

    def test_batch_unauthenticated_returns_401(self, cookie_app):
        with cookie_app.test_client() as client:
            with patch("services.auth_session_service.validate_session", return_value=False), \
                 patch("services.wrapper_auth_service.WrapperAuthService") as mock_cls:
                mock_cls.return_value.validate_bearer.return_value = None
                resp = client.post(
                    "/api/signals/batch",
                    json=[{"signal_type": "x", "content": "c"}],
                    content_type="application/json",
                )
        assert resp.status_code == 401

    def test_batch_valid_signals_emitted_even_with_errors(self, cookie_app):
        """Valid signals in the batch should be emitted regardless of other failures."""
        signals = [
            {"signal_type": "ok", "content": "valid"},
            {"content": "no signal_type"},  # invalid
        ]
        emit_mock = MagicMock(return_value=True)
        with cookie_app.test_client() as client:
            with patch("services.auth_session_service.validate_session", return_value=True), \
                 patch("api.signals._get_rate_limiter", return_value=MagicMock(is_allowed=MagicMock(return_value=True))), \
                 patch("api.signals._check_signal_capability", return_value=True), \
                 patch("services.reasoning_loop_service.emit_reasoning_signal", emit_mock), \
                 patch("services.memory_client.MemoryClientService.create_connection", return_value=MagicMock()):
                resp = client.post(
                    "/api/signals/batch",
                    json=signals,
                    content_type="application/json",
                )
        assert resp.get_json()["accepted"] == 1
        emit_mock.assert_called_once()


# ---------------------------------------------------------------------------
# _validate_signal unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestValidateSignal:
    def test_minimal_valid_payload(self):
        from api.signals import _validate_signal
        result, err = _validate_signal({"signal_type": "ctx", "content": "hello"})
        assert err is None
        assert result["signal_type"] == "ctx"
        assert result["content"] == "hello"
        assert result["activation_energy"] == 0.5

    def test_defaults_applied(self):
        from api.signals import _validate_signal
        result, _ = _validate_signal({"signal_type": "x", "content": "y"})
        assert result["source"] is None
        assert result["topic"] is None
        assert result["metadata"] is None

    def test_not_a_dict_returns_error(self):
        from api.signals import _validate_signal
        result, err = _validate_signal("a string")
        assert result is None
        assert "JSON object" in err

    def test_activation_energy_boundary_0(self):
        from api.signals import _validate_signal
        result, err = _validate_signal({"signal_type": "x", "content": "y", "activation_energy": 0.0})
        assert err is None
        assert result["activation_energy"] == 0.0

    def test_activation_energy_boundary_1(self):
        from api.signals import _validate_signal
        result, err = _validate_signal({"signal_type": "x", "content": "y", "activation_energy": 1.0})
        assert err is None
        assert result["activation_energy"] == 1.0

    def test_activation_energy_negative_fails(self):
        from api.signals import _validate_signal
        _, err = _validate_signal({"signal_type": "x", "content": "y", "activation_energy": -0.1})
        assert err is not None

    def test_source_trimmed(self):
        from api.signals import _validate_signal
        result, _ = _validate_signal({"signal_type": "x", "content": "y", "source": "  ci  "})
        assert result["source"] == "ci"

    def test_empty_source_becomes_none(self):
        from api.signals import _validate_signal
        result, _ = _validate_signal({"signal_type": "x", "content": "y", "source": ""})
        assert result["source"] is None

    def test_metadata_dict_accepted(self):
        from api.signals import _validate_signal
        result, err = _validate_signal({
            "signal_type": "x", "content": "y", "metadata": {"branch": "main"}
        })
        assert err is None
        assert result["metadata"] == {"branch": "main"}

    def test_metadata_none_accepted(self):
        from api.signals import _validate_signal
        result, err = _validate_signal({"signal_type": "x", "content": "y", "metadata": None})
        assert err is None
        assert result["metadata"] is None


# ---------------------------------------------------------------------------
# ReasoningSignal wrapper_id field
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestReasoningSignalWrapperId:
    def test_wrapper_id_field_defaults_to_none(self):
        from services.reasoning_loop_service import ReasoningSignal
        sig = ReasoningSignal(signal_type="ctx", source="test")
        assert sig.wrapper_id is None

    def test_wrapper_id_survives_json_roundtrip(self):
        from services.reasoning_loop_service import ReasoningSignal
        sig = ReasoningSignal(
            signal_type="ctx",
            source="ci",
            wrapper_id="wrp_abc123",
        )
        recovered = ReasoningSignal.from_json(sig.to_json())
        assert recovered.wrapper_id == "wrp_abc123"

    def test_wrapper_id_none_survives_json_roundtrip(self):
        from services.reasoning_loop_service import ReasoningSignal
        sig = ReasoningSignal(signal_type="ctx", source="internal")
        recovered = ReasoningSignal.from_json(sig.to_json())
        assert recovered.wrapper_id is None
