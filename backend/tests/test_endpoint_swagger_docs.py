"""Feature test: the Endpoint/Action base contract auto-generates Swagger docs.

Every route flask-restx serves through :class:`api.endpoint.Endpoint` /
:class:`api.action.Action` is documented purely from each subclass's
``request_dto`` / ``response_dto`` declarations (see ``Endpoint._document`` /
``Endpoint._document_method``) — no per-route hand-written swagger decorators
exist anywhere in ``api/endpoints`` or ``api/actions``. This suite proves the
generated ``/swagger.json`` actually reflects that contract for a
representative slice of real, migrated routes:

  * success responses are wrapped in the shared envelope models
    (``{DTO}Envelope`` / ``{DTO}ListingEnvelope``, both requiring the fields
    the real ``Response.single``/``Response.listing`` wire shapes carry);
  * every non-2xx except 401 is the shared ``ErrorEnvelope``; 401 is the bare
    ``AuthError`` shape ``require_auth`` actually emits (``{"error": str}``,
    never envelope-wrapped) and is documented on every verb, since auth runs
    before dispatch ever reaches a handler;
  * a POST with a declared ``request_dto`` documents its body parameter and a
    422 branch, in addition to any declared success codes;
  * a handler that is implemented but declares no response DTO documents a
    bare 204;
  * a handler that resolves to the inherited default (never overridden by the
    concrete subclass) documents only 401 plus the uniform 405;
  * per-handler ``DocumentedResponse.extras`` surface the real non-2xx
    statuses a handler emits beyond the structurally-guaranteed set.

Real production path, zero mocks: the real ``create_app()`` Flask app (which
walks the real ``api/endpoints`` and ``api/actions`` packages and builds each
namespace's real swagger doc), a real test client hitting the real
``/swagger.json`` route flask-restx serves, and the real ``db`` fixture so app
construction (which reads ``Setting`` for CORS config) succeeds against a real
SQLite file.
"""

from __future__ import annotations

import sqlite3
from typing import cast

import pytest
from flask.testing import FlaskClient

pytestmark = pytest.mark.unit

_MIGRATED_PREFIXES = ("/api/lists", "/api/skills", "/api/policies", "/api/wrappers")
"""Endpoint groups migrated onto the Endpoint/Action base contract as of this
test's writing — the prefixes the base-contract swagger bridge is responsible
for documenting."""


@pytest.fixture
def swagger_spec(db: sqlite3.Connection) -> dict[str, object]:
    """The real, fully-built OpenAPI spec served by a real app instance."""
    from api import create_app
    app = create_app()
    app.config["TESTING"] = True
    client: FlaskClient = app.test_client()
    resp = client.get("/swagger.json")
    assert resp.status_code == 200
    spec = resp.get_json()
    assert spec is not None
    return cast("dict[str, object]", spec)


def _at(node: object, *keys: str) -> dict[str, object]:
    """Walk nested spec dicts by key; loud assert on a missing key or shape."""
    for key in keys:
        assert isinstance(node, dict) and key in node, f"spec node missing {key!r}"
        node = node[key]
    assert isinstance(node, dict), f"spec node at {keys!r} is not an object"
    return cast("dict[str, object]", node)


def _strs(node: dict[str, object], key: str) -> list[str]:
    """A spec node's string-array member (e.g. a schema's ``required`` list)."""
    value = node[key]
    assert isinstance(value, list), f"spec member {key!r} is not an array"
    return cast("list[str]", value)


def _ref(node: object, *keys: str) -> str:
    """The ``$ref`` target string at the end of a key walk."""
    ref = _at(node, *keys)["$ref"]
    assert isinstance(ref, str)
    return ref


def _refs(node: object) -> list[str]:
    """Collect every ``$ref`` target string reachable from a JSON-schema node."""
    found: list[str] = []
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            found.append(ref)
        for value in node.values():
            found.extend(_refs(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_refs(item))
    return found


class TestSwaggerSpecServesRealDefinitions:
    def test_swagger_json_is_valid_and_has_definitions(self, swagger_spec: dict[str, object]) -> None:
        assert "paths" in swagger_spec
        definitions = _at(swagger_spec, "definitions")
        assert definitions
        # The shared error envelope is registered once and reused everywhere,
        # so it must exist regardless of which namespaces happened to register it.
        assert "ErrorEnvelope" in definitions


class TestListingRouteDocumentsPaginatedEnvelope:
    def test_get_all_lists_documents_listing_envelope_with_resolvable_item_ref(
        self, swagger_spec: dict[str, object]
    ) -> None:
        get_op = _at(swagger_spec, "paths", "/api/lists/all", "get")
        schema_ref = _ref(get_op, "responses", "200", "schema")
        assert schema_ref == "#/definitions/ListResponseListingEnvelope"

        envelope = _at(swagger_spec, "definitions", "ListResponseListingEnvelope")
        assert set(_strs(envelope, "required")) == {"success", "result", "pagination"}
        pagination = _at(envelope, "properties", "pagination")
        assert set(_strs(pagination, "required")) == {"page", "limit", "total"}

        item_ref = _ref(envelope, "properties", "result", "items")
        item_def_name = item_ref.rsplit("/", 1)[-1]
        assert item_def_name in _at(swagger_spec, "definitions")


class TestSingleResourceRouteDocumentsEnvelopeAndErrors:
    def test_get_list_by_id_documents_envelope_success_and_uniform_errors(
        self, swagger_spec: dict[str, object]
    ) -> None:
        responses = _at(swagger_spec, "paths", "/api/lists/{id}", "get", "responses")

        assert _ref(responses, "200", "schema") == "#/definitions/ListResponseEnvelope"
        envelope = _at(swagger_spec, "definitions", "ListResponseEnvelope")
        assert set(_strs(envelope, "required")) == {"success", "result"}

        for status in ("404", "403", "400"):
            assert _ref(responses, status, "schema") == "#/definitions/ErrorEnvelope"

        # 401 is deliberately NOT the envelope: require_auth rejects with a
        # bare {"error": str} body before dispatch, documented as AuthError.
        assert _ref(responses, "401", "schema") == "#/definitions/AuthError"
        auth_error = _at(swagger_spec, "definitions", "AuthError")
        assert _strs(auth_error, "required") == ["error"]
        assert set(_at(auth_error, "properties")) == {"error"}


def _body_params(post_op: dict[str, object]) -> list[dict[str, object]]:
    """The operation's body parameters (the ``in: body`` entries)."""
    params = cast("list[dict[str, object]]", post_op.get("parameters", []))
    return [p for p in params if p.get("in") == "body"]


class TestPostWithRequestDtoDocumentsExpectAnd422:
    def test_post_list_documents_body_param_and_create_update_and_422(
        self, swagger_spec: dict[str, object]
    ) -> None:
        post_op = _at(swagger_spec, "paths", "/api/lists/{id}", "post")

        body_params = _body_params(post_op)
        assert len(body_params) == 1
        assert _ref(body_params[0], "schema") == "#/definitions/ListRequest"
        assert "ListRequest" in _at(swagger_spec, "definitions")

        responses = _at(post_op, "responses")
        assert _ref(responses, "422", "schema") == "#/definitions/ErrorEnvelope"
        # A handler that supports both create (id=-1) and update documents both codes.
        assert _ref(responses, "200", "schema") == "#/definitions/ListResponseEnvelope"
        assert _ref(responses, "201", "schema") == "#/definitions/ListResponseEnvelope"


class TestImplementedHandlerWithoutResponseDtoDocuments204:
    def test_policies_upsert_documents_204_with_no_declared_dto(
        self, swagger_spec: dict[str, object]
    ) -> None:
        # PoliciesEndpoint.post upserts a single cell and returns ("", 204); it
        # is implemented but declares no entry in response_dto, so the bridge
        # must fall back to a bare 204 rather than inventing a success schema.
        post_op = _at(swagger_spec, "paths", "/api/policies/{id}", "post")
        responses = _at(post_op, "responses")
        assert responses["204"] == {"description": "No Content"}
        assert "200" not in responses
        assert "201" not in responses
        # The request body is still documented — POST still validates through request_dto.
        assert _ref(_body_params(post_op)[0], "schema") == "#/definitions/PolicyUpsertRequest"


class TestUnimplementedVerbDocuments401And405Only:
    def test_policies_get_by_id_is_401_and_405_only(self, swagger_spec: dict[str, object]) -> None:
        # PoliciesEndpoint never overrides get() (only get_all/post), so it
        # resolves to the inherited Endpoint.get default and must be documented
        # as 401 (require_auth wraps even the never-dispatched stubs) plus the
        # bare 405 — no success shape, no 400/403/404 branches that imply a
        # real handler runs.
        responses = _at(swagger_spec, "paths", "/api/policies/{id}", "get", "responses")
        assert set(responses) == {"401", "405"}
        assert _ref(responses, "405", "schema") == "#/definitions/ErrorEnvelope"
        assert _ref(responses, "401", "schema") == "#/definitions/AuthError"


class TestDeclaredExtrasDocumentRealHandlerStatuses:
    def test_skill_copy_post_documents_conflict_unprocessable_and_unavailable(
        self, swagger_spec: dict[str, object]
    ) -> None:
        # SkillCopy.post really emits 409 (duplicate copy title), 422 (only
        # curated skills can be copied — semantic, no request_dto involved) and
        # 503 (skills database unavailable), all via the envelope builders, so
        # its DocumentedResponse.extras must surface each against ErrorEnvelope.
        responses = _at(swagger_spec, "paths", "/api/skills/copy/{id}", "post", "responses")
        for status in ("409", "422", "503"):
            assert _ref(responses, status, "schema") == "#/definitions/ErrorEnvelope"
        assert _ref(responses, "200", "schema") == "#/definitions/SkillResponseEnvelope"


class TestEveryRefInMigratedNamespacesResolves:
    def test_all_refs_under_migrated_prefixes_resolve_to_a_definition(
        self, swagger_spec: dict[str, object]
    ) -> None:
        definitions = _at(swagger_spec, "definitions")
        migrated_paths = {
            path: item
            for path, item in _at(swagger_spec, "paths").items()
            if path.startswith(_MIGRATED_PREFIXES)
        }
        assert migrated_paths, "expected at least one migrated route in the spec"

        unresolved: list[str] = []
        for path, path_item in migrated_paths.items():
            for ref in _refs(path_item):
                if not ref.startswith("#/definitions/"):
                    unresolved.append(f"{path}: non-definitions ref {ref}")
                    continue
                name = ref.rsplit("/", 1)[-1]
                if name not in definitions:
                    unresolved.append(f"{path}: dangling ref {ref}")

        assert not unresolved, "\n".join(unresolved)
