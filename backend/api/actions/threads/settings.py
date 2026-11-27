"""``/api/threads/settings/<turn_id>`` — the model and thinking level a thread runs on.

Covers:
- GET  → the settings the thread's next message runs on. ``-1`` reads the
  channel's latest thread — the main chat box's thread; on a channel with no
  thread yet it reads the default with a null ``turn_id``, and there is
  nothing to change until the first message makes one.
- POST → change a thread's settings with a plain update of its row; they apply
  from its next message. ``-1`` changes the channel's latest thread.
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request import Request
from api.request.thread_settings import ThreadSettingsChange
from api.response.threads import ThreadSettings
from api.thread_scope import ThreadScope
from exceptions import NotFoundError
from models.thread import Thread
from services.provider_cache_service import ProviderCacheService


class ThreadsSettings(Action):
    """Reads and changes the model and thinking level a thread runs on."""

    id_type: ClassVar[type[int] | type[str]] = int
    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"get", "post"})
    request_dto: ClassVar[type[Request] | None] = ThreadSettingsChange
    response_dto = {
        "get": DocumentedResponse(ThreadSettings),
        "post": DocumentedResponse(ThreadSettings, extras=((404, "Thread not found"),)),
    }

    def get(self, id: int | str) -> ResponseReturnValue:
        """Read the settings the thread's next message runs on: its model, or
        the default when that is unset or no longer configured, and its level."""
        channel = ThreadScope.from_query().channel
        turn_id = cast("int", id)
        if self.is_create(turn_id):
            return self._body(Thread.latest(channel))
        thread = Thread.of(channel, turn_id)
        if thread is None:
            raise NotFoundError("Thread not found")
        return self._body(thread)

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Change the thread's settings, from its next message on."""
        channel = ThreadScope.from_query().channel
        change = cast("ThreadSettingsChange", data)
        turn_id = cast("int", id)
        if self.is_create(turn_id):
            latest = Thread.latest(channel)
            if latest is None:
                raise NotFoundError("Thread not found")
            turn_id = latest.turn_id
        if not Thread.set_settings(channel, turn_id, change.provider_id, change.thinking_level):
            raise NotFoundError("Thread not found")
        return self._body(Thread.of(channel, turn_id))

    @staticmethod
    def _body(thread: Thread | None) -> ResponseReturnValue:
        """The settings *thread*'s next message runs on; the default model at
        ``auto`` with a null ``turn_id`` when there is no thread."""
        provider = ProviderCacheService.chat_provider(thread.provider_id if thread else None)
        return ThreadSettings(
            provider_id=cast("int | None", provider.get("id")),
            thinking_level=thread.thinking_level if thread else "auto",
            turn_id=thread.turn_id if thread else None,
        ).single()
