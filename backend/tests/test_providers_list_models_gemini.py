"""Tests for the Gemini branch of the providers ``/list-models`` endpoint.

Covers ``_fetch_gemini_models`` (parse/filter/auth) and the endpoint routing
that previously returned ``Unsupported platform 'gemini'``.
"""

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from api.providers import _fetch_gemini_models, providers_bp


def _resp(status=200, json_body=None):
    r = MagicMock()
    r.status_code = status
    r.ok = 200 <= status < 300
    r.json.return_value = json_body or {}
    return r


@pytest.mark.unit
class TestFetchGeminiModels:
    def test_requires_api_key(self):
        models, err = _fetch_gemini_models("")
        assert models is None
        assert err == "API key is required"

    def test_parses_filters_and_strips_prefix(self):
        body = {
            "models": [
                {
                    "name": "models/gemini-2.5-flash",
                    "displayName": "Gemini 2.5 Flash",
                    "supportedGenerationMethods": ["generateContent", "countTokens"],
                },
                {
                    "name": "models/embedding-001",
                    "displayName": "Embedding 001",
                    "supportedGenerationMethods": ["embedContent"],
                },
            ]
        }
        with patch("api.providers.req.get", return_value=_resp(200, body)) as g:
            models, err = _fetch_gemini_models("key-123")

        assert err is None
        # embedding-001 filtered out (no generateContent); prefix stripped.
        assert models == [{"id": "gemini-2.5-flash", "display_name": "Gemini 2.5 Flash"}]
        # Key travels in the header, never the URL/query.
        _, kwargs = g.call_args
        assert kwargs["headers"]["x-goog-api-key"] == "key-123"
        assert "key-123" not in g.call_args[0][0]

    def test_invalid_key_400(self):
        body = {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key."}}
        with patch("api.providers.req.get", return_value=_resp(400, body)):
            models, err = _fetch_gemini_models("bad")
        assert models is None
        assert err == "Invalid API key"

    def test_invalid_key_403(self):
        with patch("api.providers.req.get", return_value=_resp(403, {})):
            models, err = _fetch_gemini_models("bad")
        assert models is None
        assert err == "Invalid API key"


@pytest.mark.unit
class TestListModelsGeminiRoute:
    @pytest.fixture
    def client(self):
        app = Flask(__name__)
        app.register_blueprint(providers_bp)
        app.config["TESTING"] = True
        return app.test_client()

    @pytest.fixture(autouse=True)
    def bypass_auth(self):
        with patch("services.auth_session_service.validate_session", return_value=True):
            yield

    def test_gemini_no_longer_unsupported(self, client):
        body = {
            "models": [
                {
                    "name": "models/gemini-2.5-pro",
                    "displayName": "Gemini 2.5 Pro",
                    "supportedGenerationMethods": ["generateContent"],
                }
            ]
        }
        with patch("api.providers.req.get", return_value=_resp(200, body)):
            res = client.post("/providers/list-models", json={"platform": "gemini", "api_key": "k"})

        assert res.status_code == 200
        data = res.get_json()
        assert data["error"] is None
        assert data["models"] == [{"id": "gemini-2.5-pro", "display_name": "Gemini 2.5 Pro"}]
