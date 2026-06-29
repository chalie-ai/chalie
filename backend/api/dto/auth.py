"""Auth-namespace DTOs — /auth HTTP boundary types.

``RegisterRequest``/``LoginRequest`` are swagger-documentation only; runtime
validation stays hand-rolled in the handlers (preserves exact 400 bodies).
``VaultResult`` is swagger-documentation only; register/login/logout return
a Flask ``Response`` directly so the Set-Cookie header is attached correctly.
``password`` is WRITE-ONLY and appears on NO response DTO.
"""

from __future__ import annotations

from pydantic import Field

from .base import DTO


class AuthStatus(DTO):
    """GET /auth/status — public auth state snapshot."""

    has_master_account: bool
    has_providers: bool
    has_session: bool
    vault_state: str
    has_vision_provider: bool
    internal_dev: bool


class Username(DTO):
    """GET /auth/username — master account login credential."""

    username: str


class RegisterRequest(DTO):
    """POST /auth/register — swagger docs only; runtime validation is hand-rolled."""

    username: str
    password: str = Field(min_length=8)


class LoginRequest(DTO):
    """POST /auth/login — swagger docs only; runtime validation is hand-rolled."""

    username: str
    password: str


class VaultResult(DTO):
    """Successful register/login/logout shape — swagger docs only.

    Actual responses are Flask ``Response`` objects (cookie side-effect).
    """

    ok: bool
    vault_state: str
