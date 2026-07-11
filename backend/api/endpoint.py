"""Endpoint — abstract base owning ALL route generation, auth, and error envelopes.

Subclass :class:`Endpoint` to migrate a CRUD endpoint group into ``api/endpoints/``.
The base builds the Flask-RESTx wiring (Namespace, Resources, routes, auth) from
``slug()`` alone; subclasses hold pure controller logic and zero route strings.

Routes generated per subclass:

- ``GET    /api/{slug}/all``   → :meth:`Endpoint.get_all`
- ``GET    /api/{slug}/<id>``  → :meth:`Endpoint.get`
- ``POST   /api/{slug}/<id>``  → :meth:`Endpoint.post` (id ``-1`` = create)
- ``DELETE /api/{slug}/<id>``  → :meth:`Endpoint.delete`

Every method has a concrete 405 default, so a subclass overrides only what it
supports. Failures are raised as :class:`EndpointError` subclasses and mapped
onto the uniform error envelope here — handlers never hand-build error bodies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from flask import g, request
from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource
from pydantic import ValidationError

from exceptions import EndpointError, ForbiddenError
from .auth import require_auth
from .dto.openapi import register_auth_error, register_dto, register_envelope, register_error_envelope
from .request import Request
from .response import Response

_DECLARABLE_HANDLERS = frozenset({"get_all", "get", "post", "delete"})
"""Handler names :attr:`Endpoint.response_dto` may key on — the same set
:meth:`Endpoint.namespace`/:meth:`Action.namespace` dispatch to."""


@dataclass(frozen=True)
class DocumentedResponse:
    """Declares one handler's documented success shape for the swagger bridge.

    ``dto`` is the :class:`Response` subclass the envelope's ``result`` wraps;
    ``None`` documents ``204 No Content`` (the base contract's own success
    shape for a handler with nothing to declare — e.g. ``delete``) while still
    letting :attr:`extras` attach genuinely-emitted error statuses. ``listing``
    selects the paginated-collection envelope (``result`` an array plus a
    ``pagination`` block) over the single-resource envelope. ``extras`` are
    ``(status, description)`` pairs for real non-2xx statuses a handler emits
    beyond the structurally-guaranteed 400/403/404/422/405 — each documented
    against the uniform error envelope; only declare a status a handler
    actually returns. ``not_found`` (default ``True``) documents the
    structural 404 on an id-addressed ``get``/``delete`` handler; declare
    ``False`` only on a handler that provably never emits 404 — e.g. an
    ack-style handler that always 200s, or an idempotent delete that 204s
    even when the row is already gone. Purely declarative — never consulted
    at request time, only when :meth:`Endpoint.namespace` builds the swagger
    doc.
    """

    dto: type[Response] | None = None
    listing: bool = False
    extras: tuple[tuple[int, str], ...] = ()
    not_found: bool = True


class Endpoint(ABC):
    """Base for every CRUD endpoint group; subclasses hold controller logic only.

    Subclasses must be constructible with no arguments — auto-discovery
    instantiates them bare (see ``api._register_namespaces``).
    """

    id_type: ClassVar[type[int] | type[str]] = str
    """Type the captured ``<id>`` segment is coerced to before reaching handlers."""

    request_dto: ClassVar[type[Request] | None] = None
    """Inbound DTO the POST body validates through; ``None`` = no body expected."""

    max_limit: ClassVar[int] = 100
    """Upper bound the ``limit`` pagination arg is clamped to."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset()
    """Handler names (``get_all``/``get``/``post``/``delete``) restricted to human
    cookie sessions — a bearer wrapper gets an enveloped 403. Declared where a
    wrapper must never act on its own behalf (e.g. minting wrapper tokens)."""

    _post_may_create: ClassVar[bool] = True
    """Internal swagger plumbing: whether POST has a create path that can
    return 201. :class:`Action` defaults it to ``False`` (most verbs act on
    existing resources); a creating verb overrides back to ``True``."""

    response_dto: ClassVar[Mapping[str, DocumentedResponse]] = {}
    """Per-handler documented success shape, keyed by handler name
    (``get_all``/``get``/``post``/``delete``). A handler implemented but absent
    from this mapping is documented as ``204 No Content``, matching the base
    contract's ``"", 204`` convention. A key naming a handler this subclass
    does not override raises at namespace-build time — such a declaration can
    never render, so it is a bug, not a default. Declaring entries here is the
    only swagger wiring a subclass ever does — :meth:`namespace` does the rest."""

    @abstractmethod
    def slug(self) -> str:
        """URL segment this endpoint group lives under (``/api/{slug}/...``)."""

    def get_all(self, page: int = 1, limit: int = 20) -> ResponseReturnValue:
        """Default: listing not supported."""
        return self.not_allowed()

    def get(self, id: int | str) -> ResponseReturnValue:
        """Default: fetch not supported."""
        return self.not_allowed()

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Default: create/update not supported.

        Overrides must branch on :meth:`is_create` — never on a literal, since
        the sentinel arrives as ``-1`` (int) or ``"-1"`` (str) depending on
        :attr:`id_type`.
        """
        return self.not_allowed()

    def delete(self, id: int | str) -> ResponseReturnValue:
        """Default: delete not supported."""
        return self.not_allowed()

    def not_allowed(self) -> ResponseReturnValue:
        """The uniform 405 envelope every unimplemented method falls back to."""
        return Response.failure("Method not allowed"), 405

    def is_create(self, id: int | str) -> bool:
        """True when the id is the create sentinel, regardless of :attr:`id_type`."""
        return id in (-1, "-1")

    def namespace(self) -> Namespace:
        """Build the fully-wired Namespace for this endpoint group."""
        ns = Namespace(
            self.slug(),
            path=f"/api/{self.slug()}",
            description=f"{self.slug()} endpoint group",
        )
        endpoint = self

        class AllResource(Resource):
            """Generated listing route: GET /api/{slug}/all (other verbs 405 enveloped)."""

            @require_auth
            def get(self) -> ResponseReturnValue:
                return endpoint._handle_all()

            @require_auth
            def post(self) -> ResponseReturnValue:
                return endpoint.not_allowed()

            @require_auth
            def put(self) -> ResponseReturnValue:
                return endpoint.not_allowed()

            @require_auth
            def delete(self) -> ResponseReturnValue:
                return endpoint.not_allowed()

        class ItemResource(Resource):
            """Generated item routes: GET/POST/DELETE /api/{slug}/<id>."""

            @require_auth
            def get(self, id: str) -> ResponseReturnValue:
                return endpoint._handle_get(id)

            @require_auth
            def post(self, id: str) -> ResponseReturnValue:
                return endpoint._handle_post(id)

            @require_auth
            def put(self, id: str) -> ResponseReturnValue:
                return endpoint.not_allowed()

            @require_auth
            def delete(self, id: str) -> ResponseReturnValue:
                return endpoint._handle_delete(id)

        # Explicit endpoint names: the generated Resource classes share their
        # class name across every subclass, and Flask requires globally unique
        # endpoint names — the slug disambiguates.
        ns.route("/all", endpoint=f"{self.slug()}_all")(AllResource)
        ns.route("/<string:id>", endpoint=f"{self.slug()}_item")(ItemResource)
        self._document(
            ns,
            [
                (AllResource, {"get": "get_all", "post": None, "put": None, "delete": None}),
                (ItemResource, {"get": "get", "post": "post", "put": None, "delete": "delete"}),
            ],
        )
        return ns

    # ── swagger documentation (shared with Action) ─────────────────────────

    def _document(
        self,
        ns: Namespace,
        resources: Sequence[tuple[type[Resource], Mapping[str, str | None]]],
    ) -> None:
        """Register every declared DTO and attach flask-restx response/expect docs.

        ``resources`` pairs each generated Resource class with a map from HTTP
        verb (``get``/``post``/``put``/``delete``) to the logical handler name
        it dispatches to (``None`` when the verb is hardcoded to
        :meth:`not_allowed`, e.g. every ``put``). Subclasses never call this —
        it is the sole swagger wiring point, driven entirely by
        :attr:`request_dto` and :attr:`response_dto`.
        """
        for declared_handler in self.response_dto:
            if (
                declared_handler not in _DECLARABLE_HANDLERS
                or getattr(type(self), declared_handler) is getattr(Endpoint, declared_handler)
            ):
                raise ValueError(
                    f"{type(self).__name__}.response_dto declares {declared_handler!r}, which this "
                    "class does not override — a declaration that can never render is a bug, "
                    "not a default"
                )

        if self.request_dto is not None:
            register_dto(ns, self.request_dto)

        dto_classes: list[type[Response]] = []
        seen: set[str] = set()
        for doc in self.response_dto.values():
            if doc.dto is not None and doc.dto.__name__ not in seen:
                dto_classes.append(doc.dto)
                seen.add(doc.dto.__name__)
        if dto_classes:
            register_dto(ns, *dto_classes)

        error_model = ns.models[register_error_envelope(ns)]
        auth_error_model = ns.models[register_auth_error(ns)]

        for resource, verb_map in resources:
            for http_verb, handler_name in verb_map.items():
                self._document_method(ns, resource, http_verb, handler_name, error_model, auth_error_model)

    def _document_method(
        self,
        ns: Namespace,
        resource: type[Resource],
        http_verb: str,
        handler_name: str | None,
        error_model: object,
        auth_error_model: object,
    ) -> None:
        """Attach swagger response/expect metadata to one generated Resource method.

        Every verb documents 401 first — ``require_auth`` wraps every
        generated method (see :meth:`namespace`), so authentication failure
        structurally precedes dispatch, including the ``put``/unimplemented
        stubs that never reach a handler at all. A ``handler_name`` that
        resolves to the inherited :class:`Endpoint` default (never overridden
        by this subclass) documents only the uniform 405 on top of that;
        an implemented handler documents the structurally-guaranteed error
        codes plus its success shape and any declared :attr:`DocumentedResponse.extras`
        from :attr:`response_dto` (``204`` when the handler declares no DTO).
        """
        func = getattr(resource, http_verb)
        func = ns.response(401, "Authentication required", model=auth_error_model)(func)

        if handler_name is None or getattr(type(self), handler_name) is getattr(Endpoint, handler_name):
            setattr(resource, http_verb, ns.response(405, "Method not allowed", model=error_model)(func))
            return

        func = ns.response(400, "Invalid request", model=error_model)(func)
        func = ns.response(403, "Forbidden", model=error_model)(func)

        doc = self.response_dto.get(handler_name)
        if handler_name in ("get", "delete") and (doc is None or doc.not_found):
            func = ns.response(404, "Not found", model=error_model)(func)
        if handler_name == "post" and self.request_dto is not None:
            func = ns.expect(ns.models[self.request_dto.__name__])(func)
            func = ns.response(422, "Invalid request body", model=error_model)(func)

        if doc is None or doc.dto is None:
            func = ns.response(204, "No Content")(func)
        else:
            envelope_model = ns.models[register_envelope(ns, doc.dto, listing=doc.listing)]
            func = ns.response(200, "OK", model=envelope_model)(func)
            if handler_name == "post" and self._post_may_create:
                func = ns.response(201, "Created", model=envelope_model)(func)

        if doc is not None:
            for status, description in doc.extras:
                func = ns.response(status, description, model=error_model)(func)

        setattr(resource, http_verb, func)

    # ── dispatch plumbing (shared with Action) ────────────────────────────

    def _handle_all(self) -> ResponseReturnValue:
        """Parse, clamp, and dispatch pagination args to :meth:`get_all`.

        Handlers receive sanitized values (``page >= 1``,
        ``1 <= limit <= max_limit``) and must not re-validate them.
        """
        try:
            page = int(request.args.get("page", "1"))
            limit = int(request.args.get("limit", "20"))
        except ValueError:
            return Response.failure("Invalid pagination"), 400
        page = max(1, page)
        limit = min(max(1, limit), self.max_limit)

        def dispatch() -> ResponseReturnValue:
            self._require_cookie("get_all")
            return self.get_all(page, limit)

        return self._guard(dispatch)

    def _handle_get(self, raw_id: str) -> ResponseReturnValue:
        """Coerce the id and dispatch to :meth:`get`."""

        def dispatch() -> ResponseReturnValue:
            self._require_cookie("get")
            return self.get(self._coerce_id(raw_id))

        return self._guard(dispatch)

    def _handle_post(self, raw_id: str) -> ResponseReturnValue:
        """Coerce the id, validate the body, and dispatch to :meth:`post`."""

        def dispatch() -> ResponseReturnValue:
            self._require_cookie("post")
            return self.post(self._coerce_id(raw_id), self._read_body())

        return self._guard(dispatch)

    def _handle_delete(self, raw_id: str) -> ResponseReturnValue:
        """Coerce the id and dispatch to :meth:`delete`."""

        def dispatch() -> ResponseReturnValue:
            self._require_cookie("delete")
            return self.delete(self._coerce_id(raw_id))

        return self._guard(dispatch)

    def _require_cookie(self, method: str) -> None:
        """Reject bearer-wrapper callers on methods declared cookie-only."""
        if method in self.cookie_only_methods and getattr(g, "wrapper_id", None) is not None:
            raise ForbiddenError("This action requires cookie session auth")

    def _guard(self, call: Callable[[], ResponseReturnValue]) -> ResponseReturnValue:
        """Run a handler call, mapping typed failures onto the error envelope."""
        try:
            return call()
        except EndpointError as exc:
            return Response.failure(str(exc)), exc.status
        except ValidationError as exc:
            # Compact field:message pairs — never the raw pydantic dump, which
            # leaks input values and internal error URLs to the client.
            details = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            )
            return Response.failure(f"Invalid request body — {details}"), 422

    def _coerce_id(self, raw: str) -> int | str:
        """Coerce the captured string id to the declared :attr:`id_type`."""
        if self.id_type is int:
            try:
                return int(raw)
            except ValueError as exc:
                raise EndpointError("Invalid id") from exc
        return raw

    def _read_body(self) -> Request | None:
        """Validate the JSON body through :attr:`request_dto` (``None`` when undeclared)."""
        if self.request_dto is None:
            return None
        body = request.get_json(silent=True)
        if body is None:
            # A non-empty body that failed to parse must fail loudly — silently
            # validating {} would misreport the cause as missing fields.
            if request.get_data():
                raise EndpointError(
                    "Request body must be valid JSON (Content-Type: application/json)"
                )
            body = {}
        return self.request_dto.model_validate(body)
