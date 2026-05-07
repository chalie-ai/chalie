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

import secrets

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
    app.secret_key = secrets.token_hex(16)
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

def _make_retrieval_module_mock(results=None):
    """Return a mock episodic_retrieval_service *module*.

    api/query.py calls _get_retrieval_module().retrieve(query_text=...) —
    the retrieval module is a plain module, not a class instance.
    """
    mod = MagicMock()
    mod.retrieve.return_value = results or [
        {
            "id": "ep_001",
            "summary": "User fixed authentication bug",
            "composite_score": 0.78,
            "created_at": "2026-03-01T10:00:00+00:00",
        }
    ]
    return mod


# Keep the old name as an alias so test helpers that reference it still work.
def _make_episodic_service_mock(results=None):
    """Alias for _make_retrieval_module_mock — kept for test-helper back-compat."""
    return _make_retrieval_module_mock(results)


def _make_store_mock(style_raw=None):
    """Return a mock MemoryStore connection."""
    store = MagicMock()
    store.get.return_value = style_raw
    return store


# ---------------------------------------------------------------------------
# Relevance endpoint tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRelevanceEndpoint:
    def _patch_relevance_services(self, ep_results=None, world_raw=None):
        """Patch episodic retrieval module and memory-client services for relevance tests.

        api/query.py uses _get_retrieval_module().retrieve() — not EpisodicService.
        """
        retrieval_mod = _make_retrieval_module_mock(ep_results)
        db_mock = MagicMock()
        store_mock = _make_store_mock(world_raw)
        mc_cls = MagicMock()
        mc_cls.create_connection.return_value = store_mock

        stack = ExitStack()
        stack.enter_context(
            patch("api.query._get_retrieval_module", return_value=retrieval_mod)
        )
        stack.enter_context(patch("api.query._get_db_service", return_value=db_mock))
        stack.enter_context(
            patch("api.query._get_memory_client", return_value=mc_cls)
        )
        return stack, retrieval_mod, store_mock

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
        assert "related_traits" in data
        assert "recommendation" in data

    def test_empty_query_returns_no_query(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                resp = client.get("/api/query/relevance?q=")
        data = resp.get_json()
        assert data["recommendation"] == "no_query"
        assert data["relevance"] == pytest.approx(0.0, abs=1e-9)

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
        """_get_retrieval_module failure → relevance 0.0, recommendation defer."""
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), \
                 patch("api.query._get_retrieval_module", side_effect=Exception("no db")), \
                 patch("api.query._get_db_service", side_effect=Exception("no db")), \
                 patch("api.query._get_memory_client", side_effect=Exception("no store")):
                resp = client.get("/api/query/relevance?q=test")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["relevance"] == pytest.approx(0.0, abs=1e-9)
        assert data["recommendation"] == "defer"

    def test_related_traits_present_and_empty_list(self, cookie_app):
        """related_traits key is always present and is a list (goals table removed)."""
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                stack, _, _ = self._patch_relevance_services([])
                with stack:
                    resp = client.get("/api/query/relevance?q=auth")
        data = resp.get_json()
        assert resp.status_code == 200
        assert "related_traits" in data
        assert isinstance(data["related_traits"], list)


# ---------------------------------------------------------------------------
# Identity endpoint tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Memory endpoint tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMemoryEndpoint:
    def _patch_memory_services(self, results=None):
        """Patch episodic retrieval module for memory endpoint tests.

        api/query.py uses _get_retrieval_module().retrieve() — not EpisodicService.
        """
        retrieval_mod = _make_retrieval_module_mock(results)
        db_mock = MagicMock()

        stack = ExitStack()
        stack.enter_context(
            patch("api.query._get_retrieval_module", return_value=retrieval_mod)
        )
        stack.enter_context(patch("api.query._get_db_service", return_value=db_mock))
        return stack, retrieval_mod

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
        """retrieve() is called with radius=0.5 (hardcoded in _slice_memory)."""
        with cookie_app.test_client() as client:
            with _patch_cookie_auth():
                stack, retrieval_mod = self._patch_memory_services([])
                with stack:
                    client.get("/api/query/memory?q=test&k=3")

        # api/query._slice_memory calls retrieval_mod.retrieve(query_text=..., channel=None, radius=0.5)
        call_kwargs = retrieval_mod.retrieve.call_args[1]
        assert call_kwargs["radius"] == pytest.approx(0.5)

    def test_service_unavailable_returns_empty(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), \
                 patch("api.query._get_retrieval_module", side_effect=Exception("no db")), \
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
    def _patch_episodic(self, results=None):
        retrieval_mod = _make_retrieval_module_mock(results or [])
        stack = ExitStack()
        stack.enter_context(
            patch("api.query._get_retrieval_module", return_value=retrieval_mod)
        )
        stack.enter_context(patch("api.query._get_db_service", return_value=MagicMock()))
        return stack

    def test_single_slice_returns_200(self, cookie_app):
        ep_stack = self._patch_episodic()
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), ep_stack:
                resp = client.post(
                    "/api/query/composite",
                    json={"slices": ["memory"]},
                    content_type="application/json",
                )
        assert resp.status_code == 200

    def test_composite_response_structure(self, cookie_app):
        ep_stack = self._patch_episodic()
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), ep_stack:
                resp = client.post(
                    "/api/query/composite",
                    json={"slices": ["memory"]},
                    content_type="application/json",
                )
        data = resp.get_json()
        assert "results" in data
        assert "denied" in data
        assert "errors" in data

    def test_memory_slice_returned(self, cookie_app):
        ep_stack = self._patch_episodic()
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), ep_stack:
                resp = client.post(
                    "/api/query/composite",
                    json={"slices": ["memory"]},
                    content_type="application/json",
                )
        data = resp.get_json()
        assert "memory" in data["results"]

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
                    json={"slices": "relevance"},
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
                        json={"slices": ["relevance", "memory"]},
                        content_type="application/json",
                        headers={"Authorization": "Bearer fake_token"},
                    )
        data = resp.get_json()
        assert resp.status_code == 200
        assert "relevance" in data["denied"]
        assert "memory" in data["denied"]
        assert data["results"] == {}

    def test_partial_permissions(self, bearer_app):
        """Bearer with memory but not relevance gets memory result, relevance denied."""
        bearer_svc = MagicMock()
        # memory → allowed, relevance → denied
        bearer_svc.check_permission.side_effect = (
            lambda wid, op, res: res == "memory"
        )
        ep_stack = self._patch_episodic()

        with bearer_app.test_client() as client:
            with _patch_bearer_auth() as stack:
                with stack, \
                     patch("api.query._get_wrapper_service", return_value=bearer_svc), \
                     ep_stack:
                    resp = client.post(
                        "/api/query/composite",
                        json={"slices": ["memory", "relevance"]},
                        content_type="application/json",
                        headers={"Authorization": "Bearer fake_token"},
                    )
        data = resp.get_json()
        assert "memory" in data["results"]
        assert "relevance" in data["denied"]

    def test_unauthenticated_returns_401(self, cookie_app):
        with cookie_app.test_client() as client:
            with _patch_no_auth() as stack:
                with stack:
                    resp = client.post(
                        "/api/query/composite",
                        json={"slices": ["memory"]},
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
        ep_stack = self._patch_episodic()
        with cookie_app.test_client() as client:
            with _patch_cookie_auth(), ep_stack:
                resp = client.post(
                    "/api/query/composite",
                    json={"slices": ["memory"]},
                    content_type="application/json",
                )
        data = resp.get_json()
        assert resp.status_code == 200
        assert "memory" in data["results"]
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
                result = _check_query_permission("memory")

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

    def test_parameterised_relevance(self):
        from api.query import _dispatch_slice
        retrieval_mod = _make_retrieval_module_mock([])
        store = _make_store_mock()
        mc_cls = MagicMock()
        mc_cls.create_connection.return_value = store

        with patch("api.query._get_retrieval_module", return_value=retrieval_mod), \
             patch("api.query._get_db_service", return_value=MagicMock()), \
             patch("api.query._get_memory_client", return_value=mc_cls):
            result = _dispatch_slice("relevance:unit tests")

        assert "relevance" in result
        assert "recommendation" in result

    def test_memory_dispatch(self):
        from api.query import _dispatch_slice
        retrieval_mod = _make_retrieval_module_mock()

        with patch("api.query._get_retrieval_module", return_value=retrieval_mod), \
             patch("api.query._get_db_service", return_value=MagicMock()):
            result = _dispatch_slice("memory")

        assert "results" in result
