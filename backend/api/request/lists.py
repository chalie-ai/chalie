"""Request DTOs for the lists module.

Only the routes that consume JSON bodies are declared here. Each class mirrors
the legacy DTOs' validation rules (field lengths, defaults) so handlers stay
thin.
"""

from __future__ import annotations

from pydantic import Field

from .request import Request


class ListRequest(Request):
    """Inbound body for list create/update.

    Both fields are optional — ``None`` means "leave unchanged" on update,
    while the endpoint's ``post()`` enforces ``name`` is required on create
    and defaults ``list_type`` to ``"checklist"`` when omitted.
    """

    name: str | None = Field(default=None, min_length=1, max_length=200)
    list_type: str | None = None


class ItemRequest(Request):
    """Inbound body for list-item add/update.

    Every field is optional — ``None`` means "leave unchanged" on update,
    while the action's ``post()`` enforces ``content`` is required on add.
    """

    content: str | None = Field(default=None, min_length=1, max_length=500)
    checked: bool | None = None
    position: int | None = None
