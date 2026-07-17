"""Network/TLS DTOs — /api/system/network HTTP boundary types.

The deployment domain drives CORS; ``ssl_enabled`` plus the uploaded cert/key
drive HTTPS serving and ``Secure`` session cookies. Applying either restarts the
process — the supervisor rebinds the socket and re-resolves CORS/TLS at boot.
"""

from __future__ import annotations

from pydantic import Field

from .base import DTO
from .boundary import File


class NetworkState(DTO):
    """GET /api/system/network — current deployment domain + TLS state."""

    deployment_domain: str
    ssl_enabled: bool
    ssl_cert_present: bool


class NetworkUpdate(DTO):
    """PUT /api/system/network — multipart form; cert/key omitted keeps the existing pair."""

    # host[:port] characters only — blocks scheme/path/regex metacharacters from
    # reaching the CORS allowlist (flask-cors treats metacharacter origins as regex).
    deployment_domain: str = Field(default="", max_length=253, pattern=r"^[A-Za-z0-9.\-:]*$")
    ssl_enabled: bool = False
    ssl_cert: File | None = None
    ssl_key: File | None = None


class NetworkUpdateResult(DTO):
    """Ack for a network/TLS write — the server is restarting to apply the change."""

    ssl_enabled: bool
    restarting: bool
