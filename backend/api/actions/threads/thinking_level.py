"""``GET /api/threads/thinking-level/<turn_id>`` — a thread's current thinking level.

The composer restores the level a thread was last sent with, so reopening a
thread does not silently drop back to ``auto``. The value is read from the
transcript input row itself — the single place it is persisted — rather than
mirrored into any client-side or in-memory copy.
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.response.threads import ThinkingLevel
from api.thread_scope import ThreadScope
from models.transcript import Transcript


class ThreadsThinkingLevel(Action):
    """The thinking level persisted on a thread's input row."""

    id_type: ClassVar[type[int] | type[str]] = int
    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"get"})
    response_dto = {"get": DocumentedResponse(ThinkingLevel, not_found=False)}

    _DEFAULT = "auto"
    """What a thread with no stored level means — the model decides per turn."""

    def get(self, id: int | str) -> ResponseReturnValue:
        """Read the thread's thinking level.

        ``-1`` (a new thread, not yet written) spans every turn of the channel,
        so the composer opens pre-set to whatever the surface last used; a real
        id reads that turn's own row. A thread that has never carried one reads
        as ``auto``, never 404 — the absence is a value, not a missing resource.

        Args:
            id: Turn id; ``-1`` reads the channel's most recent level.

        Returns:
            ThinkingLevel — the literal stored value (``auto``/``medium``/``high``).
        """
        scope = ThreadScope.from_query()
        turn_id = cast("int", id)
        level = Transcript.latest_thinking_level(scope.channel, turn_id if turn_id != -1 else None)
        return ThinkingLevel(level=level or self._DEFAULT).single()
