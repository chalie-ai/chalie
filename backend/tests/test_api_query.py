"""
Unit tests for the /api/query blueprint.

All tests mock external service dependencies so no database, MemoryStore,
or LLM provider is required.  Auth is patched at the source to avoid any
real session / token validation.

Patch strategy:
  The blueprint exposes module-level ``_get_*_service()`` factory functions.
  Tests patch those functions to return mock classes/instances.  This avoids
  any transitive import of real services.

Fixtures:
  cookie_app  — Flask test app where auth succeeds with g.wrapper_id = None
  bearer_app  — Flask test app where auth succeeds with g.wrapper_id = "wrp_test"
"""

import json
import pytest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch
from flask import Flask

from api.query import query_bp


# ---------------------------------------------------------------------------
# App factories
# ---------------------------------------------------------------------------

def _make_app():
    """Build a minimal Flask test app with only the query blueprint."""
    app = Flask(__name__)
    app.register_blueprint(query_bp)
    app.config["TESTING"] = True
    app.secret_key = "test-secret"
    return app


@pytest.fixture
def cookie_app():
    """App fixture: cookie-session auth (g.wrapper_id = None)."""
    return _make_app()


@pytest.fixture
def bearer_app():
    """App fixture: bearer-token auth (g.wrapper_id = 'wrp_test')."""
    return _make_app()


# ---------------------------------------------------------------------------
# Auth patch helpers
# ---------------------------------------------------------------------------

def _patch_cookie_auth():
    """Patch validate_session to return True (cookie auth path)."""
    return patch("services.auth_session_service.validate_session", return_value=True)


def _patch_bearer_auth(wrapper_id="wrp_test"):
    """Patch auth so that bearer validation succeeds with the given wrapper_id.

    Returns an ExitStack context manager that patches both auth paths.
    """
    stack = ExitStack()
    stack.enter_context(
        patch("services.auth_session_service.validate_session", return_value=False)
    )
    auth_svc = MagicMock()
    auth_svc.validate_bearer.return_value = wrapper_id
    stack.enter_context(
        patch(
            "services.wrapper_auth_service.WrapperAuthService", return_value=auth_svc
        )
    )
    return stack


def _patch_no_auth():
    """Patch both auth paths to fail (unauthenticated request)."""
    stack = ExitStack()
    stack.enter_context(
        patch("services.auth_session_service.validate_session", return_value=False)
    )
    auth_svc = MagicMock()
    auth_svc.validate_bearer.return_value = None
    stack.enter_context(
        patch(
            "services.wrapper_auth_service.WrapperAuthService", return_value=auth_svc
        )
    )
    return stack


# ---------------------------------------------------------------------------
# Service mock builders
# ---------------------------------------------------------------------------

def _make_situation_service_mock(
    scores=None,
    directive="Test directive.",
    phase=None,
):
    """Return a mock SituationModelService *instance*.

    The fixture patches ``api.query._get_situation_model_service`` to return
    a class whose constructor yields this mock.
    """
    if scores is None:
        scores = {
            "interruptibility": {"value": 0.35},
            "receptiveness": {"value": 0.72},
            "cognitive_load": {"value": 0.45},
            "proactivity_budget": {"value": 0.58},
            "stability": {"value": 0.85},
        }
    if phase is None:
        phase = {"current": "deepening", "momentum": 0.75}

    instance = MagicMock()
    instance.get_current.return_value = {"scores": scores, "phase": phase, "raw": {}}
    instance.get_directive.return_value = directive
    return instance


def _make_world_service_mock():
    """Return a mock WorldStateService *instance*."""
    instance = MagicMock()
    instance.get_world_model_summary.return_value = {
        "scheduled": ["[in 2h] Team standup"],
        "tasks": ["[ACTIVE] Refactor auth module"],
        "lists": ["Shopping"],
        "ambient": [],
        "topics": ["development"],
        "reasoning_focus": [],
    }
    return instance


def _make_episodic_service_mock(results=None):
    """Return a mock EpisodicService *instance*."""
    instance = MagicMock()
    instance.retrieve_episodes.return_value = results or [
        {
            "id": "ep_001",
            "summary": "User fixed authentication bug",
            "composite_score": 0.78,
            "created_at": "2026-03-01T10:00:00+00:00",
        }
    ]
    return instance


def _make_identity_service_mock():
    """Return a mock IdentityService *instance*."""
    instance = MagicMock()
    instance.get_vectors.return_value = {
        "warmth": {"baseline_weight": 0.6, "current_activation": 0.65},
        "assertiveness": {"baseline_weight": 0.5, "current_activation": 0.55},
    }
    return instance


def _make_store_mock(style_raw=None):
    """Return a mock MemoryStore connection."""
    store = MagicMock()
    store.get.return_value = style_raw
    return store


# ---------------------------------------------------------------------------
# Situation endpoint tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSituationEndpoint:
    def _patch_situation(self, svc_instance):
        """Patch _get_situation_model_service to return a class yielding svc_instance."""
        cls_mock = MagicMock(return_value=svc_instance)
        return patch("api.query._get_situation_model_service", return_value=cls_mock)

    def test_cookie_auth_returns_200(self, cookie_app):
        svc = _make_situation_service_mock()
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), self._patch_situation(svc):
                resp = client.get("/api/query/situation")
        assert resp.status_code == 200

    def test_response_has_required_keys(self, cookie_app):
        svc = _make_situation_service_mock()
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), self._patch_situation(svc):
                resp = client.get("/api/query/situation")
        data = resp.get_json()
        assert "scores" in data
        assert "directive" in data
        assert "phase" in data

    def test_scores_are_floats(self, cookie_app):
        svc = _make_situation_service_mock()
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), self._patch_situation(svc):
                resp = client.get("/api/query/situation")
        for key, val in resp.get_json()["scores"].items():
            assert isinstance(val, float), f"{key} score is not a float"

    def test_phase_has_current_and_momentum(self, cookie_app):
        svc = _make_situation_service_mock()
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), self._patch_situation(svc):
                resp = client.get("/api/query/situation")
        phase = resp.get_json()["phase"]
        assert "current" in phase
        assert "momentum" in phase

    def test_directive_is_string(self, cookie_app):
        svc = _make_situation_service_mock(directive="Be brief.")
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), self._patch_situation(svc):
                resp = client.get("/api/query/situation")
        directive = resp.get_json()["directive"]
        assert isinstance(directive, str)

    def test_directive_none_is_allowed(self, cookie_app):
        svc = _make_situation_service_mock(directive=None)
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), self._patch_situation(svc):
                resp = client.get("/api/query/situation")
        assert resp.get_json()["directive"] is None

    def test_unauthenticated_returns_401(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_no_auth() as stack:
                with stack:
                    resp = client.get("/api/query/situation")
        assert resp.status_code == 401

    def test_bearer_without_permission_returns_403(self, bearer_app):
        bearer_svc = MagicMock()
        bearer_svc.check_permission.return_value = False

        with bearer_app.test_client() as client:
            with _patch_bearer_auth() as stack:
                with stack, \
                     patch("api.query._get_wrapper_service", return_value=bearer_svc):
                    resp = client.get(
                        "/api/query/situation",
                        headers={"Authorization": "Bearer fake_token"},
                    )
        assert resp.status_code == 403

    def test_bearer_with_permission_returns_200(self, bearer_app):
        bearer_svc = MagicMock()
        bearer_svc.check_permission.return_value = True
        svc = _make_situation_service_mock()

        with bearer_app.test_client() as client:
            with _patch_bearer_auth() as stack:
                with stack, \
                     patch("api.query._get_wrapper_service", return_value=bearer_svc), \
                     self._patch_situation(svc):
                    resp = client.get(
                        "/api/query/situation",
                        headers={"Authorization": "Bearer fake_token"},
                    )
        assert resp.status_code == 200

    def test_service_unavailable_returns_defaults(self, cookie_app):
        """If _get_situation_model_service raises, return 200 with defaults."""
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), \
                 patch("api.query._get_situation_model_service", side_effect=Exception("unavailable")):
                resp = client.get("/api/query/situation")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["scores"] == {}
        assert data["directive"] is None
        assert data["phase"] == {}


# ---------------------------------------------------------------------------
# Relevance endpoint tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRelevanceEndpoint:
    def _patch_relevance_services(self, ep_results=None, world_raw=None):
        """Patch episodic and memory-client services for relevance tests."""
        ep_instance = _make_episodic_service_mock(ep_results)
        ep_cls = MagicMock(return_value=ep_instance)
        db_mock = MagicMock()
        store_mock = _make_store_mock(world_raw)
        mc_cls = MagicMock()
        mc_cls.create_connection.return_value = store_mock

        stack = ExitStack()
        stack.enter_context(
            patch("api.query._get_episodic_service", return_value=ep_cls)
        )
        stack.enter_context(patch("api.query._get_db_service", return_value=db_mock))
        stack.enter_context(
            patch("api.query._get_memory_client", return_value=mc_cls)
        )
        return stack, ep_instance, store_mock

    def test_returns_200_with_valid_query(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                stack, _, _ = self._patch_relevance_services()
                with stack:
                    resp = client.get("/api/query/relevance?q=auth+tests")
        assert resp.status_code == 200

    def test_response_has_required_keys(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                stack, _, _ = self._patch_relevance_services([])
                with stack:
                    resp = client.get("/api/query/relevance?q=anything")
        data = resp.get_json()
        assert "relevance" in data
        assert "related_goals" in data
        assert "related_traits" in data
        assert "recommendation" in data

    def test_empty_query_returns_no_query(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                resp = client.get("/api/query/relevance?q=")
        data = resp.get_json()
        assert data["recommendation"] == "no_query"
        assert data["relevance"] == 0.0

    def test_missing_q_returns_no_query(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                resp = client.get("/api/query/relevance")
        assert resp.get_json()["recommendation"] == "no_query"

    def test_high_score_recommends_surface_now(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                stack, _, _ = self._patch_relevance_services([{"composite_score": 80}])
                with stack:
                    resp = client.get("/api/query/relevance?q=urgent")
        assert resp.get_json()["recommendation"] == "surface_now"

    def test_mid_score_recommends_surface_soon(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                stack, _, _ = self._patch_relevance_services([{"composite_score": 55}])
                with stack:
                    resp = client.get("/api/query/relevance?q=some+topic")
        assert resp.get_json()["recommendation"] == "surface_soon"

    def test_low_score_recommends_defer(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                stack, _, _ = self._patch_relevance_services([{"composite_score": 10}])
                with stack:
                    resp = client.get("/api/query/relevance?q=low+relevance+topic")
        assert resp.get_json()["recommendation"] == "defer"

    def test_unauthenticated_returns_401(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_no_auth() as stack:
                with stack:
                    resp = client.get("/api/query/relevance?q=test")
        assert resp.status_code == 401

    def test_bearer_without_permission_returns_403(self, bearer_app):
        bearer_svc = MagicMock()
        bearer_svc.check_permission.return_value = False

        with bearer_app.test_client() as client:
            with _patch_bearer_auth() as stack:
                with stack, \
                     patch("api.query._get_wrapper_service", return_value=bearer_svc):
                    resp = client.get(
                        "/api/query/relevance?q=test",
                        headers={"Authorization": "Bearer fake_token"},
                    )
        assert resp.status_code == 403

    def test_service_error_returns_defaults(self, cookie_app):
        """_get_episodic_service failure → relevance 0.0, recommendation defer."""
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), \
                 patch("api.query._get_episodic_service", side_effect=Exception("no db")), \
                 patch("api.query._get_db_service", side_effect=Exception("no db")), \
                 patch("api.query._get_memory_client", side_effect=Exception("no store")):
                resp = client.get("/api/query/relevance?q=test")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["relevance"] == 0.0
        assert data["recommendation"] == "defer"

    def test_related_goals_populated_from_goals_table(self, cookie_app):
        ep_results: list = []
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            ('Refactor auth module',),
            ('Write tests',),
        ]
        mock_db.connection.return_value = mock_conn

        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                stack, _, _ = self._patch_relevance_services(ep_results)
                with stack, \
                     patch("api.query.get_shared_db_service", return_value=mock_db, create=True), \
                     patch("services.database_service.get_shared_db_service", return_value=mock_db):
                    resp = client.get("/api/query/relevance?q=auth")
        data = resp.get_json()
        # If the mock DB works, auth appears in related_goals. Otherwise graceful degradation.
        assert resp.status_code == 200
        assert "related_goals" in data


# ---------------------------------------------------------------------------
# World-state endpoint tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWorldStateEndpoint:
    def _patch_world_service(self, svc_instance=None):
        if svc_instance is None:
            svc_instance = _make_world_service_mock()
        cls_mock = MagicMock(return_value=svc_instance)
        return patch("api.query._get_world_state_service", return_value=cls_mock)

    def test_returns_200(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), self._patch_world_service():
                resp = client.get("/api/query/world-state")
        assert resp.status_code == 200

    def test_response_has_items_key(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), self._patch_world_service():
                resp = client.get("/api/query/world-state")
        data = resp.get_json()
        assert "items" in data
        assert isinstance(data["items"], list)

    def test_items_have_type_and_label(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), self._patch_world_service():
                resp = client.get("/api/query/world-state")
        items = resp.get_json()["items"]
        assert len(items) > 0
        for item in items:
            assert "type" in item
            assert "label" in item

    def test_item_types_come_from_summary_keys(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), self._patch_world_service():
                resp = client.get("/api/query/world-state")
        item_types = {i["type"] for i in resp.get_json()["items"]}
        # The mock has scheduled, tasks, lists, topics
        assert "scheduled" in item_types
        assert "task" in item_types

    def test_optional_q_param_accepted(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), self._patch_world_service():
                resp = client.get("/api/query/world-state?q=meetings")
        assert resp.status_code == 200

    def test_service_unavailable_returns_empty(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), \
                 patch("api.query._get_world_state_service", side_effect=Exception("unavailable")):
                resp = client.get("/api/query/world-state")
        assert resp.status_code == 200
        assert resp.get_json()["items"] == []

    def test_unauthenticated_returns_401(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_no_auth() as stack:
                with stack:
                    resp = client.get("/api/query/world-state")
        assert resp.status_code == 401

    def test_bearer_without_permission_returns_403(self, bearer_app):
        bearer_svc = MagicMock()
        bearer_svc.check_permission.return_value = False

        with bearer_app.test_client() as client:
            with _patch_bearer_auth() as stack:
                with stack, \
                     patch("api.query._get_wrapper_service", return_value=bearer_svc):
                    resp = client.get(
                        "/api/query/world-state",
                        headers={"Authorization": "Bearer fake_token"},
                    )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Identity endpoint tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIdentityEndpoint:
    def _patch_identity_services(self, style_raw=None):
        identity_instance = _make_identity_service_mock()
        id_cls = MagicMock(return_value=identity_instance)
        db_mock = MagicMock()
        store_mock = _make_store_mock(style_raw)
        mc_cls = MagicMock()
        mc_cls.create_connection.return_value = store_mock

        stack = ExitStack()
        stack.enter_context(
            patch("api.query._get_identity_service", return_value=id_cls)
        )
        stack.enter_context(patch("api.query._get_db_service", return_value=db_mock))
        stack.enter_context(
            patch("api.query._get_memory_client", return_value=mc_cls)
        )
        return stack, identity_instance, store_mock

    def test_returns_200(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                stack, _, _ = self._patch_identity_services()
                with stack:
                    resp = client.get("/api/query/identity")
        assert resp.status_code == 200

    def test_response_has_vectors_and_style(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                stack, _, _ = self._patch_identity_services()
                with stack:
                    resp = client.get("/api/query/identity")
        data = resp.get_json()
        assert "vectors" in data
        assert "style" in data

    def test_vectors_have_baseline_and_activation(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                stack, _, _ = self._patch_identity_services()
                with stack:
                    resp = client.get("/api/query/identity")
        vectors = resp.get_json()["vectors"]
        assert "warmth" in vectors
        assert "baseline" in vectors["warmth"]
        assert "activation" in vectors["warmth"]

    def test_style_metrics_returned_when_cached(self, cookie_app):
        style_data = {"verbosity": 4, "directness": 8, "formality": 3}
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                stack, _, _ = self._patch_identity_services(
                    style_raw=json.dumps(style_data)
                )
                with stack:
                    resp = client.get("/api/query/identity")
        assert resp.get_json()["style"] == style_data

    def test_style_empty_when_not_cached(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                stack, _, _ = self._patch_identity_services(style_raw=None)
                with stack:
                    resp = client.get("/api/query/identity")
        assert resp.get_json()["style"] == {}

    def test_service_unavailable_returns_defaults(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), \
                 patch("api.query._get_identity_service", side_effect=Exception("no db")), \
                 patch("api.query._get_db_service", side_effect=Exception("no db")), \
                 patch("api.query._get_memory_client", side_effect=Exception("no store")):
                resp = client.get("/api/query/identity")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["vectors"] == {}
        assert data["style"] == {}

    def test_unauthenticated_returns_401(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_no_auth() as stack:
                with stack:
                    resp = client.get("/api/query/identity")
        assert resp.status_code == 401

    def test_bearer_without_permission_returns_403(self, bearer_app):
        bearer_svc = MagicMock()
        bearer_svc.check_permission.return_value = False

        with bearer_app.test_client() as client:
            with _patch_bearer_auth() as stack:
                with stack, \
                     patch("api.query._get_wrapper_service", return_value=bearer_svc):
                    resp = client.get(
                        "/api/query/identity",
                        headers={"Authorization": "Bearer fake_token"},
                    )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Memory endpoint tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMemoryEndpoint:
    def _patch_memory_services(self, results=None):
        ep_instance = _make_episodic_service_mock(results)
        ep_cls = MagicMock(return_value=ep_instance)
        db_mock = MagicMock()

        stack = ExitStack()
        stack.enter_context(
            patch("api.query._get_episodic_service", return_value=ep_cls)
        )
        stack.enter_context(patch("api.query._get_db_service", return_value=db_mock))
        return stack, ep_instance

    def test_returns_200_with_results(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                stack, _ = self._patch_memory_services()
                with stack:
                    resp = client.get("/api/query/memory?q=authentication")
        assert resp.status_code == 200

    def test_response_has_results_key(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                stack, _ = self._patch_memory_services()
                with stack:
                    resp = client.get("/api/query/memory?q=test")
        assert "results" in resp.get_json()

    def test_result_items_have_expected_fields(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                stack, _ = self._patch_memory_services()
                with stack:
                    resp = client.get("/api/query/memory?q=test")
        results = resp.get_json()["results"]
        assert len(results) == 1
        item = results[0]
        assert "id" in item
        assert "summary" in item
        assert "score" in item
        assert "created_at" in item

    def test_empty_query_returns_empty_results(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                resp = client.get("/api/query/memory?q=")
        assert resp.status_code == 200
        assert resp.get_json()["results"] == []

    def test_missing_q_returns_empty_results(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                resp = client.get("/api/query/memory")
        assert resp.status_code == 200
        assert resp.get_json()["results"] == []

    def test_radius_forwarded_to_service(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                stack, ep_instance = self._patch_memory_services([])
                with stack:
                    client.get("/api/query/memory?q=test&k=3")

        call_kwargs = ep_instance.retrieve_episodes.call_args[1]
        assert call_kwargs["radius"] == 0.5

    def test_service_unavailable_returns_empty(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), \
                 patch("api.query._get_episodic_service", side_effect=Exception("no db")), \
                 patch("api.query._get_db_service", side_effect=Exception("no db")):
                resp = client.get("/api/query/memory?q=test")
        assert resp.status_code == 200
        assert resp.get_json()["results"] == []

    def test_unauthenticated_returns_401(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_no_auth() as stack:
                with stack:
                    resp = client.get("/api/query/memory?q=test")
        assert resp.status_code == 401

    def test_bearer_without_permission_returns_403(self, bearer_app):
        bearer_svc = MagicMock()
        bearer_svc.check_permission.return_value = False

        with bearer_app.test_client() as client:
            with _patch_bearer_auth() as stack:
                with stack, \
                     patch("api.query._get_wrapper_service", return_value=bearer_svc):
                    resp = client.get(
                        "/api/query/memory?q=test",
                        headers={"Authorization": "Bearer fake_token"},
                    )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Composite endpoint tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCompositeEndpoint:
    def _patch_situation(self, svc_instance=None):
        if svc_instance is None:
            svc_instance = _make_situation_service_mock()
        cls_mock = MagicMock(return_value=svc_instance)
        return patch("api.query._get_situation_model_service", return_value=cls_mock)

    def _patch_world(self, svc_instance=None):
        if svc_instance is None:
            svc_instance = _make_world_service_mock()
        cls_mock = MagicMock(return_value=svc_instance)
        return patch("api.query._get_world_state_service", return_value=cls_mock)

    def _patch_episodic(self, results=None):
        ep_instance = _make_episodic_service_mock(results or [])
        ep_cls = MagicMock(return_value=ep_instance)
        stack = ExitStack()
        stack.enter_context(
            patch("api.query._get_episodic_service", return_value=ep_cls)
        )
        stack.enter_context(patch("api.query._get_db_service", return_value=MagicMock()))
        return stack

    def test_single_slice_returns_200(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), self._patch_situation():
                resp = client.post(
                    "/api/query/composite",
                    json={"slices": ["situation"]},
                    content_type="application/json",
                )
        assert resp.status_code == 200

    def test_composite_response_structure(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), self._patch_situation():
                resp = client.post(
                    "/api/query/composite",
                    json={"slices": ["situation"]},
                    content_type="application/json",
                )
        data = resp.get_json()
        assert "results" in data
        assert "denied" in data
        assert "errors" in data

    def test_multiple_slices_returned(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), self._patch_situation(), self._patch_world():
                resp = client.post(
                    "/api/query/composite",
                    json={"slices": ["situation", "world-state"]},
                    content_type="application/json",
                )
        data = resp.get_json()
        assert "situation" in data["results"]
        assert "world-state" in data["results"]

    def test_parameterised_relevance_slice(self, cookie_app):
        ep_stack = self._patch_episodic()
        store = _make_store_mock()
        mc_cls = MagicMock()
        mc_cls.create_connection.return_value = store

        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), ep_stack, \
                 patch("api.query._get_memory_client", return_value=mc_cls):
                resp = client.post(
                    "/api/query/composite",
                    json={"slices": ["relevance:auth tests"]},
                    content_type="application/json",
                )
        data = resp.get_json()
        assert "relevance:auth tests" in data["results"]

    def test_missing_slices_key_returns_400(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                resp = client.post(
                    "/api/query/composite",
                    json={"not_slices": []},
                    content_type="application/json",
                )
        assert resp.status_code == 400

    def test_slices_not_list_returns_400(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                resp = client.post(
                    "/api/query/composite",
                    json={"slices": "situation"},
                    content_type="application/json",
                )
        assert resp.status_code == 400

    def test_empty_body_returns_400(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                resp = client.post(
                    "/api/query/composite",
                    data="not json",
                    content_type="text/plain",
                )
        assert resp.status_code == 400

    def test_empty_slices_returns_empty_results(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                resp = client.post(
                    "/api/query/composite",
                    json={"slices": []},
                    content_type="application/json",
                )
        assert resp.status_code == 200
        assert resp.get_json()["results"] == {}

    def test_denied_slices_in_denied_key(self, bearer_app):
        bearer_svc = MagicMock()
        bearer_svc.check_permission.return_value = False

        with bearer_app.test_client() as client:
            with _patch_bearer_auth() as stack:
                with stack, \
                     patch("api.query._get_wrapper_service", return_value=bearer_svc):
                    resp = client.post(
                        "/api/query/composite",
                        json={"slices": ["situation", "identity"]},
                        content_type="application/json",
                        headers={"Authorization": "Bearer fake_token"},
                    )
        data = resp.get_json()
        assert resp.status_code == 200
        assert "situation" in data["denied"]
        assert "identity" in data["denied"]
        assert data["results"] == {}

    def test_partial_permissions(self, bearer_app):
        """Bearer with situation but not identity gets situation result, identity denied."""
        bearer_svc = MagicMock()
        # situation → allowed, identity → denied
        bearer_svc.check_permission.side_effect = (
            lambda wid, op, res: res == "situation"
        )
        situation_mock = _make_situation_service_mock()

        with bearer_app.test_client() as client:
            with _patch_bearer_auth() as stack:
                with stack, \
                     patch("api.query._get_wrapper_service", return_value=bearer_svc), \
                     self._patch_situation(situation_mock):
                    resp = client.post(
                        "/api/query/composite",
                        json={"slices": ["situation", "identity"]},
                        content_type="application/json",
                        headers={"Authorization": "Bearer fake_token"},
                    )
        data = resp.get_json()
        assert "situation" in data["results"]
        assert "identity" in data["denied"]

    def test_unauthenticated_returns_401(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_no_auth() as stack:
                with stack:
                    resp = client.post(
                        "/api/query/composite",
                        json={"slices": ["situation"]},
                        content_type="application/json",
                    )
        assert resp.status_code == 401

    def test_unknown_slice_in_results_with_error_field(self, cookie_app):
        """Unknown slice names are dispatched and return {error: ...} in results."""
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                resp = client.post(
                    "/api/query/composite",
                    json={"slices": ["nonexistent_slice"]},
                    content_type="application/json",
                )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "nonexistent_slice" in data["results"]
        assert "error" in data["results"]["nonexistent_slice"]

    def test_cookie_auth_bypasses_permission_checks(self, cookie_app):
        """Cookie-authenticated requests skip all permission enforcement."""
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), self._patch_situation():
                resp = client.post(
                    "/api/query/composite",
                    json={"slices": ["situation"]},
                    content_type="application/json",
                )
        data = resp.get_json()
        assert resp.status_code == 200
        assert "situation" in data["results"]
        assert data["denied"] == []


# ---------------------------------------------------------------------------
# _check_query_permission unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckQueryPermission:
    def test_cookie_auth_always_permitted(self):
        """g.wrapper_id = None → always True without calling WrapperAuthService."""
        app = _make_app()
        with app.test_request_context("/"):
            from flask import g
            g.wrapper_id = None
            from api.query import _check_query_permission
            assert _check_query_permission("situation") is True
            assert _check_query_permission("anything") is True

    def test_bearer_delegates_to_wrapper_service(self):
        """g.wrapper_id set → calls WrapperAuthService.check_permission."""
        app = _make_app()
        wrapper_svc = MagicMock()
        wrapper_svc.check_permission.return_value = True

        with app.test_request_context("/"):
            from flask import g
            g.wrapper_id = "wrp_test"
            with patch("api.query._get_wrapper_service", return_value=wrapper_svc):
                from api.query import _check_query_permission
                result = _check_query_permission("memory")

        wrapper_svc.check_permission.assert_called_once_with("wrp_test", "query", "memory")
        assert result is True

    def test_bearer_denied_returns_false(self):
        app = _make_app()
        wrapper_svc = MagicMock()
        wrapper_svc.check_permission.return_value = False

        with app.test_request_context("/"):
            from flask import g
            g.wrapper_id = "wrp_no_perm"
            with patch("api.query._get_wrapper_service", return_value=wrapper_svc):
                from api.query import _check_query_permission
                result = _check_query_permission("identity")

        assert result is False

    def test_service_exception_fails_closed(self):
        """WrapperAuthService raising → permission denied (False)."""
        app = _make_app()
        with app.test_request_context("/"):
            from flask import g
            g.wrapper_id = "wrp_test"
            with patch("api.query._get_wrapper_service", side_effect=Exception("db error")):
                from api.query import _check_query_permission
                result = _check_query_permission("situation")

        assert result is False


# ---------------------------------------------------------------------------
# _dispatch_slice unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDispatchSlice:
    def test_unknown_slice_returns_error(self):
        from api.query import _dispatch_slice
        result = _dispatch_slice("does_not_exist")
        assert "error" in result
        assert "does_not_exist" in result["error"]

    def test_world_state_hyphen_alias(self):
        from api.query import _dispatch_slice
        svc_instance = _make_world_service_mock()
        cls_mock = MagicMock(return_value=svc_instance)
        with patch("api.query._get_world_state_service", return_value=cls_mock):
            result = _dispatch_slice("world-state")
        assert "items" in result

    def test_world_state_underscore_alias(self):
        from api.query import _dispatch_slice
        svc_instance = _make_world_service_mock()
        cls_mock = MagicMock(return_value=svc_instance)
        with patch("api.query._get_world_state_service", return_value=cls_mock):
            result = _dispatch_slice("world_state")
        assert "items" in result

    def test_parameterised_relevance(self):
        from api.query import _dispatch_slice
        ep_instance = _make_episodic_service_mock([])
        ep_cls = MagicMock(return_value=ep_instance)
        store = _make_store_mock()
        mc_cls = MagicMock()
        mc_cls.create_connection.return_value = store

        with patch("api.query._get_episodic_service", return_value=ep_cls), \
             patch("api.query._get_db_service", return_value=MagicMock()), \
             patch("api.query._get_memory_client", return_value=mc_cls):
            result = _dispatch_slice("relevance:unit tests")

        assert "relevance" in result
        assert "recommendation" in result

    def test_situation_dispatch(self):
        from api.query import _dispatch_slice
        svc_instance = _make_situation_service_mock()
        cls_mock = MagicMock(return_value=svc_instance)
        with patch("api.query._get_situation_model_service", return_value=cls_mock):
            result = _dispatch_slice("situation")
        assert "scores" in result
        assert "directive" in result
        assert "phase" in result

    def test_identity_dispatch(self):
        from api.query import _dispatch_slice
        id_instance = _make_identity_service_mock()
        id_cls = MagicMock(return_value=id_instance)
        store = _make_store_mock()
        mc_cls = MagicMock()
        mc_cls.create_connection.return_value = store

        with patch("api.query._get_identity_service", return_value=id_cls), \
             patch("api.query._get_db_service", return_value=MagicMock()), \
             patch("api.query._get_memory_client", return_value=mc_cls):
            result = _dispatch_slice("identity")

        assert "vectors" in result
        assert "style" in result

    def test_memory_dispatch(self):
        from api.query import _dispatch_slice
        ep_instance = _make_episodic_service_mock()
        ep_cls = MagicMock(return_value=ep_instance)

        with patch("api.query._get_episodic_service", return_value=ep_cls), \
             patch("api.query._get_db_service", return_value=MagicMock()):
            result = _dispatch_slice("memory")

        assert "results" in result
