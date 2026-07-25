"""Action — abstract base for verb-shaped endpoints, aligned with the CRUD contract.

Subclass :class:`Action` to migrate a non-CRUD operation into
``api/actions/{slug}/{verb}.py``. Routes generated per subclass:

- ``/api/{slug}/{verb}``       → dispatched with the create sentinel id ``-1``
- ``/api/{slug}/{verb}/<id>``  → dispatched with the captured id

Nested resources are actions too (e.g. list items = ``GET /lists/items/<list_id>``).
Most actions implement a single HTTP method; the rest 405 via the inherited
defaults. ``all`` is a reserved verb — it collides with the CRUD listing route.
"""

from __future__ import annotations

import re
from abc import abstractmethod
from typing import ClassVar

from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from .auth import require_auth
from .endpoint import Endpoint

_VERB_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
"""Kebab-case only — underscores would let a verb forge another route's
generated Flask endpoint name (``{slug}__{verb}`` separators below)."""


class Action(Endpoint):
    """Base for every verb endpoint; subclasses hold controller logic only."""

    _post_may_create: ClassVar[bool] = False
    """Most Action verbs act on existing resources and never 201; the few
    that genuinely create (skill copy, list add-item) override to ``True``."""

    id_converter: ClassVar[str] = "string"
    """Flask URL converter for the ``<id>`` segment. Override to ``"path"``
    when the id itself contains slashes (e.g. a file path relative to a
    root) — a ``string`` converter stops at the first ``/``."""

    @abstractmethod
    def verb(self) -> str:
        """URL segment of the operation (``/api/{slug}/{verb}/...``)."""

    def namespace(self) -> Namespace:
        """Build the fully-wired Namespace for this action's verb routes only."""
        verb = self.verb()
        if verb == "all":
            raise ValueError(
                f'Action verb "all" is reserved — it collides with /api/{self.slug()}/all'
            )
        if not _VERB_PATTERN.fullmatch(verb):
            raise ValueError(
                f'Action verb "{verb}" must be kebab-case ([a-z0-9-], no leading/trailing dash)'
            )
        ns = Namespace(
            self.slug(),
            path=f"/api/{self.slug()}",
            description=f"{self.slug()} {verb} action",
        )
        action = self

        class VerbResource(Resource):
            """Generated id-less routes — dispatched with the ``-1`` sentinel.

            For POST that means create; for GET/DELETE the handler decides what
            an untargeted call means (usually 404 via the base defaults).
            """

            @require_auth
            def get(self) -> ResponseReturnValue:
                return action._handle_get("-1")

            @require_auth
            def post(self) -> ResponseReturnValue:
                return action._handle_post("-1")

            @require_auth
            def put(self) -> ResponseReturnValue:
                return action.not_allowed()

            @require_auth
            def delete(self) -> ResponseReturnValue:
                return action._handle_delete("-1")

        class VerbItemResource(Resource):
            """Generated id-addressed routes: GET/POST/DELETE /api/{slug}/{verb}/<id>."""

            @require_auth
            def get(self, id: str) -> ResponseReturnValue:
                return action._handle_get(id)

            @require_auth
            def post(self, id: str) -> ResponseReturnValue:
                return action._handle_post(id)

            @require_auth
            def put(self, id: str) -> ResponseReturnValue:
                return action.not_allowed()

            @require_auth
            def delete(self, id: str) -> ResponseReturnValue:
                return action._handle_delete(id)

        # Flask requires globally unique endpoint names. Double-underscore
        # separators + kebab-only verbs make these disjoint from the CRUD
        # names ({slug}_all / {slug}_item) and from every other verb — a verb
        # named "item" or "x-item" can never forge another route's name.
        ns.route(f"/{verb}", endpoint=f"{self.slug()}__{verb}")(VerbResource)
        ns.route(f"/{verb}/<{self.id_converter}:id>", endpoint=f"{self.slug()}__{verb}__item")(VerbItemResource)
        verb_map: dict[str, str | None] = {"get": "get", "post": "post", "put": None, "delete": "delete"}
        self._document(ns, [(VerbResource, verb_map), (VerbItemResource, verb_map)])
        return ns
