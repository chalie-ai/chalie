"""One outbound HTTP stack with named profiles.

Three abilities used to each carry their own fetch code: ``read`` (a browser-like
``requests.Session``), ``web_download`` (a streaming ``requests.get``) — both
URL roles since folded into ``web_fetch`` — and ``programming_docs_search``
(a ``urllib.request`` opener with a bot UA). They duplicated headers and
timeouts.

This module is the single place those concerns live:

* Three header profiles (plain dicts): ``BROWSER`` (impersonates Chrome, for
  article/page reads), ``API`` (a polite identified bot, for JSON/HTML doc
  APIs), and ``DOWNLOAD`` (minimal headers for binary file pulls).
* :func:`fetch_text` — GET a URL and return the decoded text.
* :func:`stream_to_file` — GET a URL and stream the body to a path in fixed-size
  chunks (never buffering the whole file in memory).

Any host is reachable — public, private, or on the local network.
"""

from __future__ import annotations

import os

import requests

from exceptions import DownloadTooLarge

# Default chunk size for streamed downloads (bytes).
DEFAULT_CHUNK_SIZE = 8192

# Default request timeout (seconds) when a caller does not specify one.
DEFAULT_TIMEOUT = 15

#: Impersonates a current Chrome on macOS — used by ``read`` so article and page
#: servers that gate on a real browser UA serve their content.
BROWSER: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8,"
        "application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "max-age=0",
    "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

#: A politely identified bot — used against documentation search APIs that prefer
#: an honest, attributable User-Agent over browser impersonation.
API: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (compatible; ChalieBot/1.0; +https://chalie.ai)",
    "Accept": "text/html,application/xhtml+xml,application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

#: Minimal headers for pulling binary files; no Accept gymnastics.
DOWNLOAD: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (compatible; ChalieBot/1.0; +https://chalie.ai)",
}


def fetch_text(
    url: str,
    *,
    profile: dict[str, str] = BROWSER,
    timeout: float = DEFAULT_TIMEOUT,
    verify: bool = False,
    allow_redirects: bool = True,
) -> str:
    """GET *url* and return the decoded response text.

    Raises :class:`requests.RequestException` (incl. ``raise_for_status``) on any
    HTTP failure — errors bubble so callers surface them, never swallow.
    """
    text, _ = fetch_page(
        url,
        profile=profile,
        timeout=timeout,
        verify=verify,
        allow_redirects=allow_redirects,
    )
    return text


def fetch_page(
    url: str,
    *,
    profile: dict[str, str] = BROWSER,
    timeout: float = DEFAULT_TIMEOUT,
    verify: bool = False,
    allow_redirects: bool = True,
) -> tuple[str, str]:
    """GET *url*; return ``(text, content_type)``.

    The bare ``Content-Type`` header — lower-cased, charset suffix stripped (the
    part before any ``;``) — is returned alongside the decoded body so a caller
    can decide whether the body is HTML to extract or plain text to pass through
    verbatim (the ``read`` ability's ``.diff`` / ``.patch`` case). An empty string
    is returned when the server sends no ``Content-Type``.

    Raises :class:`requests.RequestException` (incl. ``raise_for_status``) on any
    HTTP failure — errors bubble so callers surface them, never swallow.
    """
    with requests.Session() as session:
        session.headers.update(profile)
        response = session.get(
            url, timeout=timeout, allow_redirects=allow_redirects, verify=verify
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
        return response.text, content_type


def stream_to_file(
    url: str,
    dest_path: str,
    *,
    profile: dict[str, str] = DOWNLOAD,
    timeout: float = DEFAULT_TIMEOUT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_bytes: int | None = None,
) -> tuple[int, str]:
    """GET *url* and stream the body to *dest_path*.

    When *max_bytes* is set: the ``Content-Length`` header is checked upfront so
    an obviously-oversized download is aborted before the body is pulled, and the
    running byte count is also checked during streaming so a server that
    under-declares (or omits) its length is still stopped. On over-cap the
    partial file is removed and :class:`DownloadTooLarge` is raised — never a
    silent truncation.

    Raises :class:`DownloadTooLarge` when the cap is exceeded and
    :class:`requests.RequestException` on any HTTP failure.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    response = requests.get(
        url, stream=True, timeout=timeout, headers=profile
    )
    with response:
        response.raise_for_status()

        if max_bytes is not None:
            declared = response.headers.get("Content-Length")
            if declared is not None and declared.isdigit() and int(declared) > max_bytes:
                raise DownloadTooLarge(max_bytes)

        content_type = (
            response.headers.get("Content-Type", "").split(";")[0].strip().lower()
        )

        written = 0
        try:
            with open(dest_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if max_bytes is not None and written > max_bytes:
                        raise DownloadTooLarge(max_bytes)
                    fh.write(chunk)
        except (DownloadTooLarge, OSError):
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise

        return written, content_type
