"""Pydantic DTO boundary layer — the typed HTTP request/response contract.

Re-exports the public surface every namespace consumes:

- :class:`DTO` — base for every request/response data object.
- :class:`Error` — universal non-2xx body and 422 validation payload.
- :func:`expects` / :func:`responds` — boundary decorators.
- :func:`register_dto` — swagger bridge that lifts DTOs into the flask-restx ``Api``.
"""

from __future__ import annotations

from .base import DTO
from .boundary import expects, responds
from .error import Error
from .openapi import register_dto

__all__ = ["DTO", "Error", "expects", "responds", "register_dto"]
