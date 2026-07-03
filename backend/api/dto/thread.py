"""DTOs for the thread feed and per-turn blocks — the /api/thread(s) surface.

A turn (thread) is many transcript rows sharing one ``turn_id``. The feed lists
collapsed thread summaries; a block is one turn's full rows plus its collapsed
metadata. Internal recency/row-count keys from the query layer are never exposed.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field

from .base import DTO
from .message import Message


class ThreadFeedQuery(DTO):
    """Query params for GET /api/threads.

    A non-empty ``q`` switches the feed into search (the handler caps the page);
    otherwise ``limit``/``offset`` paginate. Present-but-invalid values are 422.
    """

    q: str = ""
    limit: int = Field(default=20, ge=1, le=120)
    offset: int = Field(default=0, ge=0)
    type: str = "user"


class ThreadSummary(DTO):
    """One collapsed feed row: the thread's id, last-activity timestamp, opener
    preview, row count, one-sentence gist, and whether the turn is still working.
    The internal recency sort key (the latest transcript row id) drives ordering
    server-side and is not exposed."""

    turn_id: int | None = None
    last_activity_at: str | None = None
    preview: str
    row_count: int
    gist: str | None = None
    working: bool = False


class ThreadFeed(DTO):
    """GET /api/threads body — a page of collapsed thread summaries."""

    threads: list[ThreadSummary]
    has_more: bool
    threads_returned: int


class TurnBlock(DTO):
    """One turn's full block — the single read, every batch entry and the WS
    refetch all return this shape, so one fetch fully determines a turn's render.
    Rows past the turn's settle0 carry ``thread_message`` on the message."""

    turn_id: int
    gist: str | None = None
    preview: str
    last_activity_at: str | None = None
    working: bool
    duration_ms: int
    messages: list[Message]


class ThreadBatch(DTO):
    """GET /api/threads/batch body — many turn blocks in one round-trip."""

    blocks: list[TurnBlock]


class ThreadSendRequest(DTO):
    """multipart/form-data text fields for the thread send endpoints.

    Files arrive as repeatable ``files`` parts read directly via ``request.files``
    (one DTO field can't hold many uploads), so this model ignores the merged file
    key rather than forbidding it. ``turn_id`` is path-only, never in the body.
    """

    model_config = ConfigDict(extra="ignore")

    text: str
    type: str = "user"
