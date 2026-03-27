"""
CalDAV provider registry.

Defines pre-configured server endpoints for each supported CalDAV provider and
exports :func:`resolve_provider` for looking up a provider by name.

Provider entries
----------------
Each value in :data:`PROVIDERS` is a ``dict`` with the following keys:

``url`` (str)
    Base URL of the CalDAV server (absolute for hosted providers, path
    suffix for self-hosted ones such as Nextcloud/Synology/Radicale —
    must be combined with the user-supplied ``server_url``).

``principal_path_template`` (str)
    URL path template for the user's principal resource.  ``{username}`` is
    replaced at connection time with the authenticated user's name.

``requires_app_password`` (bool)
    ``True`` if the provider requires an app-specific password rather than the
    user's primary account password (e.g. Google, Apple, Fastmail).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, dict] = {
    "google": {
        "url": "https://apidata.googleusercontent.com/caldav/v2/",
        "principal_path_template": "/caldav/v2/{username}/user",
        "requires_app_password": True,
    },
    "apple": {
        "url": "https://caldav.icloud.com/",
        "principal_path_template": "/{username}/principal/",
        "requires_app_password": True,
    },
    "fastmail": {
        "url": "https://caldav.fastmail.com/dav/",
        "principal_path_template": "/dav/principals/user/{username}/",
        "requires_app_password": True,
    },
    "nextcloud": {
        "url": "/remote.php/dav/",
        "principal_path_template": "/remote.php/dav/principals/users/{username}/",
        "requires_app_password": False,
        "requires_server_url": True,
    },
    "synology": {
        "url": "/caldav/",
        "principal_path_template": "/caldav/{username}/",
        "requires_app_password": False,
        "requires_server_url": True,
    },
    "radicale": {
        "url": "/",
        "principal_path_template": "/{username}/",
        "requires_app_password": False,
        "requires_server_url": True,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_provider(name: str) -> dict | None:
    """Return the provider configuration for *name*, or ``None`` if unknown.

    Lookup is case-insensitive; the *name* argument is lower-cased before
    the dict is consulted so that ``resolve_provider('Google')`` and
    ``resolve_provider('google')`` both succeed.

    Args:
        name: Provider identifier, e.g. ``"google"``, ``"apple"``,
            ``"fastmail"``, ``"nextcloud"``, ``"synology"``, or
            ``"radicale"``.

    Returns:
        dict | None: Provider configuration dict with keys ``url``,
        ``principal_path_template``, and ``requires_app_password``, or
        ``None`` if no matching provider is registered.

    Examples:
        >>> from capabilities.caldav_capability.providers import resolve_provider
        >>> p = resolve_provider("google")
        >>> p["url"].startswith("https://")
        True
        >>> resolve_provider("nonexistent") is None
        True
    """
    return PROVIDERS.get(name.lower())
