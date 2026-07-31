"""``GET /api/threads/batch`` — many turn blocks in one round-trip.

The feed hydrates a screenful of expanded turns on load; fetching each one
separately would cost a request per turn. This is a pure concatenation of the
single-turn read over the requested ids, so a block from here and a block from
``GET /api/threads/<id>`` are the same shape.
"""

from __future__ import annotations

from typing import ClassVar

from flask import request
from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.response.threads import ThreadBatch, TurnBlock
from api.thread_scope import ThreadScope
from services.turn_serializer_service import get_service as _serializer


class ThreadsBatch(Action):
    """Batch turn-block read for the conversation feed."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"get"})
    response_dto = {"get": DocumentedResponse(ThreadBatch, not_found=False)}

    def get(self, id: int | str) -> ResponseReturnValue:
        """Fetch every requested turn's full block at once.

        Ids arrive as repeatable ``id[]`` query params rather than in the path —
        a batch has no single subject — so the captured id is the base's unused
        ``-1`` sentinel. Non-numeric ids are ignored rather than rejected: the
        surface builds this list from whatever it currently has on screen.

        Args:
            id: Unused — the batch is addressed by the ``id[]`` query params.

        Returns:
            ThreadBatch carrying one TurnBlock per resolvable id.
        """
        scope = ThreadScope.from_query()
        turn_ids = [int(t) for t in request.args.getlist("id[]") if t.isdigit()]
        blocks = [
            TurnBlock.model_validate(_serializer().serialize(scope.channel, turn_id, scope.type))
            for turn_id in turn_ids
        ]
        return ThreadBatch(blocks=blocks).single()
