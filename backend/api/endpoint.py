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
from collections.abc import Callable
from typing import ClassVar

from flask import g, request
from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource
from pydantic import ValidationError

from .auth import require_auth
from .request import Request
from .response import Response


class EndpointError(Exception):
    """Typed endpoint failure; the base maps it onto the error envelope with its status."""

    status: ClassVar[int] = 400


class NotFoundError(EndpointError):
    """Requested resource does not exist."""

    status: ClassVar[int] = 404


class ForbiddenError(EndpointError):
    """Authenticated, but not permitted to perform this action."""

    status: ClassVar[int] = 403


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
        return ns

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
