import logging
import os
import tempfile
import uuid
from typing import ClassVar
from urllib.parse import urlparse

import requests

from abilities._ability import Ability
from abilities._result import ToolResult
from abilities._ssrf import is_private_url

logger = logging.getLogger(__name__)

_DOWNLOAD_DIR = os.path.join(tempfile.gettempdir(), "chalie_downloads")
_BLOCKED_SCHEMES = frozenset({"file", "data"})
_DEFAULT_TIMEOUT_MIN = 15
_MAX_TIMEOUT_MIN = 120
_CHUNK_SIZE = 8192


class WebDownloadAbility(Ability):
    def get_name(self) -> str:
        return "web_download"

    def get_summary(self) -> str:
        return "Download a file from the internet to a temporary location for later reading or processing."

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

    _PARAMETERS: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL of the file to download",
            },
            "timeout": {
                "type": "number",
                "description": "Download timeout in minutes (default: 15, max: 120)",
            },
        },
        "required": ["url"],
    }

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    def run(self, params: dict) -> ToolResult:
        url = params.get("url", "").strip()
        if not url:
            return ToolResult.ok("url parameter is required")

        error = _validate_url(url)
        if error:
            return ToolResult.ok(error)

        timeout_min = params.get("timeout", _DEFAULT_TIMEOUT_MIN)
        try:
            timeout_min = float(timeout_min)
        except (TypeError, ValueError):
            timeout_min = _DEFAULT_TIMEOUT_MIN
        timeout_sec = max(1, min(timeout_min, _MAX_TIMEOUT_MIN)) * 60

        dest_path = _build_dest_path(url)
        try:
            _download(url, dest_path, timeout_sec)
        except Exception as e:
            logger.exception("[WEB_DOWNLOAD] url=%r: %s", url, e)
            return ToolResult.ok(str(e))

        return ToolResult.ok(dest_path)


def _validate_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme in _BLOCKED_SCHEMES:
        return f"URL scheme '{parsed.scheme}' is blocked"
    if is_private_url(url):
        return "private or internal URL blocked"
    return None


def _build_dest_path(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    filename = os.path.basename(path) if path else "download"
    if not filename:
        filename = "download"
    return os.path.join(_DOWNLOAD_DIR, uuid.uuid4().hex, filename)


def _download(url: str, dest_path: str, timeout_sec: float) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    response = requests.get(url, stream=True, timeout=timeout_sec)
    with response:
        response.raise_for_status()
        try:
            with open(dest_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
        except Exception:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise
