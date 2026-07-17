"""Request DTOs for the skills module.

One DTO serves both create and update — every field is optional so a single
class can validate both request shapes. The legacy ``SkillCreate``/
``SkillUpdate`` validators (``backend/api/dto/skill.py``) are merged here:

- ``title``/``tags``: stripped only (create semantics) — a blank ``tags``
  stays ``""`` (explicit clear on update; ``None`` means "no change").
- ``use_for``/``content``: stripped, then blank collapses to ``None``
  (update "keep existing" semantics). This also serves create correctly —
  the endpoint's create branch treats ``None`` as "required field missing".
"""

from __future__ import annotations

from pydantic import Field, field_validator

from .request import Request


class SkillRequest(Request):
    """Inbound body for skill create/update.

    All fields are optional: the endpoint's ``post()`` enforces
    ``title``/``use_for``/``content`` are required on create, and treats
    ``None`` as "leave unchanged" on update (``title`` is never updatable).
    """

    title: str | None = Field(default=None, min_length=1, max_length=200)
    use_for: str | None = Field(default=None, min_length=1)
    content: str | None = Field(default=None, min_length=1)
    tags: str | None = None

    @field_validator("title", "tags", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("use_for", "content", mode="before")
    @classmethod
    def _blank_keeps_old(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        return stripped or None
