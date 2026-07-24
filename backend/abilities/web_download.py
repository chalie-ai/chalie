"""WebDownloadAbility — stream a file from the internet to a temporary path.

Pulls a URL's body to a unique temp path (for later reading or processing) and
returns the path, byte count, and content type so the model can hand it to
``document.upload`` or a file tool.

Security:
  - SSRF guard: a SINGLE gate in ``services.web_fetch`` blocks requests to
    private/internal IP ranges (resolved, not string-matched) before any socket
    opens. The ability catches :class:`FetchBlocked` and maps it to
    ``code=blocked-url``. There is no second guard in this module.
  - Scheme check: ``file:`` / ``data:`` URLs are refused inline (network-free,
    catches ``data:`` URLs urlparse-cheaply) with the same ``code=blocked-url``.

Result contract: every return is a :class:`abilities._result.ToolResult` built
only via ``ok()`` / ``err()``; the dispatcher renders the wire envelope. Success
body is the structured dict ``{"path", "bytes", "content_type"}`` (rendered as
compact JSON) with the enforced size cap declared in ``meta`` (``max_bytes``);
errors carry a stable kebab-case ``code`` (``blocked-url`` / ``too-large`` /
``download-failed``) so a weak model can self-correct without re-reading the
schema. A too-large download is an ERROR — never silently clipped.
"""

import logging
import os
import tempfile
import uuid
from typing import ClassVar
from urllib.parse import urlparse

import requests

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from contracts.params.web_download_params_bag import WebDownloadParamsBag
from exceptions import DownloadTooLarge, FetchBlocked
from services.web_fetch import DOWNLOAD, stream_to_file

logger = logging.getLogger(__name__)

_DOWNLOAD_DIR = os.path.join(tempfile.gettempdir(), "chalie_downloads")
_BLOCKED_SCHEMES = frozenset({"file", "data"})


class WebDownloadAbility(Ability[WebDownloadParamsBag]):
    #: The download size cap (100 MB). Enforced by ``stream_to_file`` during the
    #: stream and declared in ``meta`` on success; an over-cap pull is a
    #: ``too-large`` error with the partial file removed.
    _MAX_DOWNLOAD_BYTES: ClassVar[int] = 100 * 1024 * 1024

    #: Action-less tool — the ``""`` key drives the dispatcher's ACTION_REQUIRED
    #: pre-gate to reject a missing/blank url with ``code=missing-params`` BEFORE
    #: run() (instead of run() returning a status=success "url required" prose).
    #: Runs AFTER seam key-healing, so a model's ``uri``/``source``/etc. has already
    #: been canonicalised to ``url`` via ``VARIANTS[Keys.url]`` before this fires.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.url,)}

    # The typed input contract: the dispatch seam builds the bag via
    # WebDownloadParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = WebDownloadParamsBag

    def get_name(self) -> str:
        return "web_download"

    def get_summary(self) -> str:
        return (
            "Download a file from the internet to a temporary location for later "
            "reading or processing. Use in conjunction with the `read` tool to then "
            "get the file's contents into context."
        )

    def get_examples(self) -> list[str]:
        return [
            "download this PDF so I can read it",
            "fetch the CSV file from this URL",
            "download the image at this link",
            "grab that JSON file from the API",
            "save this webpage as a file",
            "pull down the spreadsheet from that URL",
        ]

    def get_search_tooltip(self) -> str:
        return "File download from URL"

    def get_follow_up(self, tr: ToolResult) -> str:
        """Read the downloaded file's content straight from its path."""
        path = tr.body.get("path") if isinstance(tr.body, dict) else None
        if not path:
            return ""
        return f"File downloaded. Use the `read({path})` tool to fetch its content."

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.url: {
                "type": "string",
                "description": "URL of the file to download",
            },
            Keys.timeout: {
                "type": "number",
                "description": "Download timeout in minutes (default: 15, max: 120)",
            },
        },
        "required": [Keys.url],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: WebDownloadParamsBag) -> ToolResult:
        url = params.url

        scheme = urlparse(url).scheme
        if scheme in _BLOCKED_SCHEMES:
            return ToolResult.err(
                f"The '{scheme}:' URL scheme cannot be downloaded.",
                code="blocked-url",
                hint="provide an http(s) URL to a file",
                source=url,
            )

        timeout_sec = params.timeout * 60.0

        dest_path = _build_dest_path(url)
        try:
            bytes_written, content_type = stream_to_file(
                url,
                dest_path,
                profile=DOWNLOAD,
                timeout=timeout_sec,
                max_bytes=self._MAX_DOWNLOAD_BYTES,
            )
        except FetchBlocked:
            # The single SSRF gate in web_fetch refused this host.
            return ToolResult.err(
                "This URL resolves to a private or internal address and was blocked.",
                code="blocked-url",
                source=url,
            )
        except DownloadTooLarge as e:
            return ToolResult.err(
                f"The file exceeds the {e.max_bytes}-byte download cap and was not saved.",
                code="too-large",
                hint="download a smaller file or a specific part of it",
                max_bytes=e.max_bytes,
                source=url,
            )
        except requests.RequestException as e:
            return ToolResult.err(
                f"Could not download the file: {str(e)[:150]}",
                code="download-failed",
                hint="check the URL is reachable and try again",
                source=url,
            )

        return ToolResult.ok(
            {"path": dest_path, "bytes": bytes_written, "content_type": content_type},
            source=url,
            max_bytes=self._MAX_DOWNLOAD_BYTES,
        )


def _build_dest_path(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    filename = os.path.basename(path) if path else "download"
    if not filename:
        filename = "download"
    return os.path.join(_DOWNLOAD_DIR, uuid.uuid4().hex, filename)
