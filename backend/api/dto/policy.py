"""DTOs for the policies namespace — per-action permission control.

A policy is a flat ``(channel, permission, setting)`` triple surfaced to the
Brain UI. The read shape (:class:`Policy`) carries the display enrichment
(``label``/``group``) added by ``_tag_display_rows`` in the namespace, not by a
serializer. :class:`PolicyUpsert` is the inbound single-cell upsert body;
``channel``/``setting`` are validated against the service's
``VALID_CHANNELS``/``VALID_SETTINGS`` inside ``upsert()`` (which returns 0 rows
for invalid input — a no-op 204), so only presence/type live on the DTO.
"""

from __future__ import annotations

from pydantic import Field

from .base import DTO


class Policy(DTO):
    """Read shape for one policy triple, enriched for the Brain UI."""

    channel: str
    permission: str
    setting: str
    label: str
    group: str | None = None


class PolicyUpsert(DTO):
    """Inbound body for a single-cell policy upsert."""

    channel: str
    permission: str = Field(..., min_length=1, max_length=200)
    setting: str