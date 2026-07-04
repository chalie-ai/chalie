"""Auth-namespace DTOs — /api/auth HTTP boundary types.

``RegisterRequest``/``LoginRequest`` validate the inbound body at the HTTP
boundary via ``@expects`` (uniform 422 on malformed input). ``VaultResult`` is
swagger-documentation only; register/login/logout return a Flask ``Response``
directly so the Set-Cookie header is attached correctly. ``password`` is
WRITE-ONLY and appears on NO response DTO.
"""

from __future__ import annotations

from pydantic import Field

from .base import DTO


class AuthStatus(DTO):
    """GET /api/auth/status — public auth state snapshot."""

    has_master_account: bool
    has_providers: bool
    has_session: bool
    vault_state: str
    has_vision_provider: bool
    internal_dev: bool


class Username(DTO):
    """GET /api/auth/username — master account login credential."""

    username: str


class RegisterRequest(DTO):
    """POST /api/auth/register — parsed and validated at the HTTP boundary via @expects."""

    username: str = Field(min_length=1)
    password: str = Field(min_length=8)


class LoginRequest(DTO):
    """POST /api/auth/login — parsed and validated at the HTTP boundary via @expects."""

    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class VaultResult(DTO):
    """Successful register/login/logout shape — swagger docs only.

    Actual responses are Flask ``Response`` objects (cookie side-effect).
    """

    ok: bool
    vault_state: str
