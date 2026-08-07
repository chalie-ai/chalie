"""Scheduler endpoint — migrated from the legacy ``scheduler`` namespace.

CRUD over ``scheduled_items``, a prompt-only, dumb-cron table: one row per
schedule, forever. There is no ``item_type``/``due_at``/``status``/
``group_id`` — a stateless poller fires any enabled row whose ``start_at``
floor has passed and whose cron fields match the current wall-clock minute.
``id`` is the SQLite auto-increment PK and doubles as the schedule's
``turn_id`` on the ``'schedule'`` channel.

Covers the CRUD routes the legacy module exposed for the item collection:
- GET    /api/scheduler/all   → get_all (paginated listing, newest first)
- GET    /api/scheduler/<id>  → get
- POST   /api/scheduler/<id>  → post (create when id=-1, update otherwise)
- DELETE /api/scheduler/<id>  → delete (hard delete — no soft-cancel state)

The legacy PUT /api/scheduler/<id> is mapped to POST with id != -1 (the same
contract reshape as every other migrated CRUD group). Delete is NOT
idempotent — a missing id 404s, matching the legacy resource exactly (unlike
``providers``' idempotent delete). ``/turns`` moves to an Action verb under
``api/actions/scheduler/``.

Persistence is owned by :class:`~models.scheduled_item.ScheduledItem` (the
sole home of ``scheduled_items`` + ``scheduled_items_vec`` SQL). The only
service touchpoint is the fire-and-forget ``embed_scheduled_item`` thread on
create.

Every implemented handler is cookie-only (``cookie_only_methods``), matching
the legacy module's blanket ``require_session`` on every route.
"""

from __future__ import annotations

import logging
import threading
from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.endpoint import DocumentedResponse, Endpoint
from exceptions import NotFoundError
from api.request import Request
from api.request.scheduler_item import SchedulerItemRequest
from api.response.scheduler_item import SchedulerItem
from models.scheduled_item import ScheduledItem
from services.database import Database
from services.locale_service import LocaleService
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

_ITEM_NOT_FOUND = "Scheduled item not found"


def _embed(item_id: int, message: str) -> None:
    """Fire-and-forget embedding of a freshly created item (non-fatal on failure)."""
    try:
        from services.scheduler_service import embed_scheduled_item
        embed_scheduled_item(item_id, message)
    except Exception as exc:
        logger.warning("[SCHEDULER API] Embedding failed (non-fatal): %s", exc)


class Scheduler(Endpoint):
    """CRUD endpoint for scheduled prompts."""

    id_type: ClassVar[type[int] | type[str]] = int
    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"get_all", "get", "post", "delete"})
    request_dto: ClassVar[type[Request] | None] = SchedulerItemRequest
    response_dto = {
        "get_all": DocumentedResponse(SchedulerItem, listing=True),
        "get": DocumentedResponse(SchedulerItem),
        "post": DocumentedResponse(SchedulerItem, extras=((404, _ITEM_NOT_FOUND),)),
        "delete": DocumentedResponse(),
    }

    def get_all(self, page: int = 1, limit: int = 20) -> ResponseReturnValue:
        """List every live schedule, newest first (``created_at`` DESC)."""
        items = [SchedulerItem.from_model(item) for item in ScheduledItem.recent()]
        total = len(items)
        start = (page - 1) * limit
        return SchedulerItem.listing(items[start : start + limit], page=page, limit=limit, total=total)

    def get(self, id: int | str) -> ResponseReturnValue:
        """Fetch one scheduled item by id."""
        item = ScheduledItem.get(cast(int, id))
        if item is None:
            raise NotFoundError(_ITEM_NOT_FOUND)
        return SchedulerItem.from_model(item).single()

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Create (id -1) or update a scheduled item.

        Raises:
            NotFoundError: item id not found on update.

        Returns:
            SchedulerItem, 201 on create / 200 on update.
        """
        dto = cast(SchedulerItemRequest, data)
        if self.is_create(id):
            return self._create(dto)
        return self._update(cast(int, id), dto)

    def delete(self, id: int | str) -> ResponseReturnValue:
        """Cancel a scheduled item — a hard delete (no soft-cancel state).

        Raises:
            NotFoundError: item id not found. Unlike ``providers``, this is
                NOT idempotent, matching the legacy resource exactly.

        Returns:
            204 No Content on success (no body).
        """
        item_id = cast(int, id)
        item = ScheduledItem.get(item_id)
        if item is None:
            raise NotFoundError(_ITEM_NOT_FOUND)
        # Drop the main row and its vec embedding in one transaction so a
        # cancelled schedule never orphans its embedding (delete_embedding
        # rides in this transaction — rowid == id under INTEGER PRIMARY KEY).
        with Database.transaction():
            item.delete()
            ScheduledItem.delete_embedding(item_id)
        return "", 204

    def _create(self, dto: SchedulerItemRequest) -> ResponseReturnValue:
        start_at_utc = LocaleService.parse_local(dto.start_at) if dto.start_at else utc_now()
        item = ScheduledItem.create(
            message=dto.message,
            start_at=start_at_utc.isoformat(),
            cron_minute=dto.minute,
            cron_hour=dto.hour,
            cron_dom=dto.day,
            cron_month=dto.month,
            cron_dow=dto.weekday,
            enabled=1 if dto.enabled else 0,
            channel=dto.channel,
            created_by_session=None,
        )  # INSERT + 1-1 gist seed, atomic; id autoincrements, created_at defaults in SQL
        item_id = cast(int, item.id)
        # Re-read so created_at (SQL default) is populated for the response.
        saved = ScheduledItem.get(item_id)

        threading.Thread(
            target=_embed, args=(item_id, dto.message),
            daemon=True, name="scheduler-embed",
        ).start()
        return SchedulerItem.from_model(cast(ScheduledItem, saved)).single(), 201

    def _update(self, item_id: int, dto: SchedulerItemRequest) -> ResponseReturnValue:
        item = ScheduledItem.get(item_id)
        if item is None:
            raise NotFoundError(_ITEM_NOT_FOUND)
        # Anchor: the caller's new start_at when supplied, else the row's
        # existing one (a plain enabled-toggle omits it) — a future-dated
        # start_at is never re-floored to now.
        item.start_at = LocaleService.parse_local(dto.start_at).isoformat() if dto.start_at else item.start_at
        item.message = dto.message
        item.cron_minute = dto.minute
        item.cron_hour = dto.hour
        item.cron_dom = dto.day
        item.cron_month = dto.month
        item.cron_dow = dto.weekday
        item.enabled = 1 if dto.enabled else 0
        item.save()  # UPDATE (id set); channel/created_by_session/created_at unchanged
        return SchedulerItem.from_model(item).single()
