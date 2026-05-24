import logging
import os
import ssl
import time
import uuid
from typing import ClassVar
from urllib.parse import urlparse

import requests
import urllib3

from abilities._base import Ability
from abilities._ssrf import is_private_url

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

_DOWNLOAD_DIR = "/tmp/chalie_downloads"
_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
_MAX_REDIRECTS = 5
_CONNECT_TIMEOUT = 30
_READ_TIMEOUT = 60
_CHUNK_SIZE = 8192
_RETRY_DELAYS = [1, 3]

_ALLOWED_SCHEMES = frozenset({"http", "https"})

_BLOCKED_PATH_PREFIXES = ("/etc", "/proc", "/dev", "/sys", "/var/run", "/boot")


class WebDownloadAbility(Ability):
    NAME = "web_download"
    SUMMARY = "Download a file from the internet to a temporary location for later reading or processing."
    SEARCH_TOOLTIP = "File download from URL"
    POLICY_CATEGORY = "Web"
    POLICY_LABELS = {
        "tmp": "Download to temp directory",
        "system": "Download to filesystem path",
    }
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
            "action": {
                "type": "string",
                "enum": ["tmp", "system"],
                "description": (
                    "By default downloads to /tmp. To download to a "
                    "specific path, set action to 'system' and supply 'path'."
                ),
            },
            "url": {
                "type": "string",
                "description": "URL of the file to download",
            },
            "path": {
                "type": "string",
                "description": "Absolute destination path. Required when action is 'system', ignored for 'tmp'.",
            },
        },
        "required": ["url"],
    }
    TIMEOUT = 120

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> str:
        action = params.get("action", "tmp")
        url = params.get("url", "").strip()
        if not url:
            return "There was an error downloading this file: url parameter is required"

        error = _validate_url(url)
        if error:
            return f"There was an error downloading this file: {error}"

        if action == "system":
            path = params.get("path", "").strip()
            if not path:
                return "There was an error downloading this file: 'path' is required when action is 'system'"
            error = _validate_destination(path)
            if error:
                return f"There was an error downloading this file: {error}"
            dest_path = path
        else:
            dest_path = _resolve_destination(url)

        try:
            _perform_download(url, dest_path)
        except Exception as e:
            logger.exception("[WEB_DOWNLOAD] Failed to download url=%r: %s", url, e)
            return f"There was an error downloading this file: {e}"

        return f"File downloaded at {dest_path}, use the 'read' tool to view it"


def _validate_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return f"unsupported URL scheme '{parsed.scheme}' — only http and https are allowed"
    if is_private_url(url):
        return "private or internal URL blocked"
    return None


def _validate_destination(destination: str) -> str | None:
    real = os.path.realpath(destination)
    if any(real.startswith(p + os.sep) or real == p for p in _BLOCKED_PATH_PREFIXES):
        return f"destination path '{destination}' is blocked — system directory"
    return None


def _resolve_destination(url: str) -> str:
    filename = _extract_filename(url)
    return os.path.join(_DOWNLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")


def _extract_filename(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    name = os.path.basename(path) if path else ""
    return name if name else "download"


def _extract_filename_from_headers(headers: dict) -> str | None:
    cd = headers.get("Content-Disposition", "")
    if not cd:
        return None
    for part in cd.split(";"):
        part = part.strip()
        if part.lower().startswith("filename="):
            name = part[9:].strip().strip('"').strip("'")
            return os.path.basename(name) if name else None
    return None


def _perform_download(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    last_error: Exception | None = None
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            _stream_to_file(url, dest_path)
            return
        except requests.ConnectionError as e:
            last_error = e
            logger.debug("[WEB_DOWNLOAD] Connection error on attempt %d: %s", attempt + 1, e)
        except requests.ReadTimeout:
            raise
        except requests.RequestException:
            raise

    raise last_error  # type: ignore[misc]


def _stream_to_file(url: str, dest_path: str) -> None:
    with requests.Session() as session:
        session.max_redirects = _MAX_REDIRECTS
        try:
            response = session.get(
                url, stream=True, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT), verify=True,
            )
        except (ssl.SSLError, requests.exceptions.SSLError):
            logger.debug("[WEB_DOWNLOAD] SSL verification failed for %s, retrying without verify", url)
            response = session.get(
                url, stream=True, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT), verify=False,
            )

        with response:
            response.raise_for_status()

            content_length = response.headers.get("Content-Length")
            if content_length:
                cl = int(content_length)
                if cl < 0 or cl > _MAX_FILE_SIZE:
                    raise ValueError(f"file too large or invalid Content-Length: {content_length}")

            # Use Content-Disposition filename when downloading to tmp (UUID-prefixed path)
            if dest_path.startswith(_DOWNLOAD_DIR):
                cd_name = _extract_filename_from_headers(response.headers)
                if cd_name:
                    prefix = dest_path.rsplit("_", 1)[0]
                    dest_path = f"{prefix}_{cd_name}"

            accumulated = 0
            try:
                with open(dest_path, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                        if not chunk:
                            continue
                        accumulated += len(chunk)
                        if accumulated > _MAX_FILE_SIZE:
                            raise ValueError(f"file too large: exceeded {_MAX_FILE_SIZE} bytes during download")
                        fh.write(chunk)
            except Exception:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                raise
