import logging
import os
import tempfile
import uuid
from typing import ClassVar
from urllib.parse import urlparse

import requests

from abilities._ability import Ability
from abilities._ssrf import is_private_url

logger = logging.getLogger(__name__)

_DOWNLOAD_DIR = os.path.join(tempfile.gettempdir(), "chalie_downloads")
_BLOCKED_SCHEMES = frozenset({"file", "data"})
_DEFAULT_TIMEOUT_MIN = 15
_MAX_TIMEOUT_MIN = 120
_CHUNK_SIZE = 8192


class WebDownloadAbility(Ability):
    NAME = "web_download"
    SUMMARY = "Download a file from the internet to a temporary location for later reading or processing."
    SEARCH_TOOLTIP = "File download from URL"
    EXAMPLES: ClassVar[list[str]] = [
        "download this PDF so I can read it",
        "fetch the CSV file from this URL",
        "download the image at this link",
        "grab that JSON file from the API",
        "save this webpage as a file",
        "pull down the spreadsheet from that URL",
    ]
    INPUT_SCHEMA: ClassVar[dict] = {
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

    def run(self, params: dict) -> str:
        url = params.get("url", "").strip()
        if not url:
            return "url parameter is required"

        error = _validate_url(url)
        if error:
            return error

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
            return str(e)

        return dest_path


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
