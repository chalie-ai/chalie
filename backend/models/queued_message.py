# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Active-record row for the ``queued_messages`` table — a message delivered
to a channel whose turn was still running, waiting for that turn to end.

Rows are read oldest first (by ``id``) and removed as each one opens its turn.
Holds no mp and calls no service.
"""

from __future__ import annotations

from typing import ClassVar

from models.model import Model


class QueuedMessage(Model):
    """One message waiting for its channel's next turn."""

    __columns__: ClassVar[tuple[str, ...]] = ("id", "channel", "text", "tool_call_id", "created_at")

    @classmethod
    def get_table(cls) -> str:
        return "queued_messages"

    channel: str
    text: str
    tool_call_id: int | None
    created_at: str

    @classmethod
    def oldest(cls, channel: str) -> QueuedMessage | None:
        """The message that has waited longest on ``channel``, or ``None``."""
        return cls.filter("channel", channel).order_by("id ASC").first()

    @classmethod
    def channels(cls) -> set[str]:
        """Every channel holding at least one waiting message."""
        return {str(c) for c in cls.all().pluck("channel")}
