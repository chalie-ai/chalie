import json
import logging
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Correlation-ID context variable
# ---------------------------------------------------------------------------

_correlation_id: ContextVar[Optional[str]] = ContextVar("_correlation_id", default=None)


def get_correlation_id() -> Optional[str]:
    return _correlation_id.get()


def set_correlation_id(value: str) -> None:
    _correlation_id.set(value)


# ---------------------------------------------------------------------------
# Custom JSON formatter (zero external dependencies)
# ---------------------------------------------------------------------------


class _ChalieJsonFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "service": record.name.split(".")[0] if record.name else "unknown",
            "message": record.getMessage(),
        }

        cid = _correlation_id.get()
        if cid is not None:
            entry["correlation_id"] = cid

        if record.exc_info and record.exc_info[1] is not None:
            entry["exc_info"] = traceback.format_exception(*record.exc_info)

        return json.dumps(entry, default=str)


# ---------------------------------------------------------------------------
# Logger facade
# ---------------------------------------------------------------------------


class Logger:
    @staticmethod
    def start() -> None:
        root = logging.getLogger()

        if root.handlers:
            return

        formatter = _ChalieJsonFormatter()

        # api/system.py:observability_errors reads this file to serve the brain dashboard error log.
        file_handler = logging.FileHandler("/tmp/chalie.log")
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        root.setLevel(logging.INFO)
        root.addHandler(file_handler)
        root.addHandler(stream_handler)

        # Werkzeug logs every request at INFO. Frontend polling fires
        # /health, /scheduler every 30–60s —
        # that's per-request CPU + log noise for no value. WARNING keeps real
        # errors (404/500) visible without the per-request stream.
        logging.getLogger("werkzeug").setLevel(logging.WARNING)

    @staticmethod
    def info(message: str) -> None:
        logging.info(message)

    @staticmethod
    def debug(message: str) -> None:
        logging.debug(message)

    @staticmethod
    def warning(message: str) -> None:
        logging.warning(message)

    @staticmethod
    def error(message: str) -> None:
        logging.error(message)

    @staticmethod
    def critical(message: str) -> None:
        logging.critical(message)
