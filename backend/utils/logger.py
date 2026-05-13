"""
Structured JSON logging facade for the Chalie backend.

Wraps Python's standard ``logging`` module with:

- :class:`_ChalieJsonFormatter` — a zero-dependency JSON formatter so every
  log line is machine-parseable JSON containing ``timestamp``, ``level``,
  ``logger``, ``message``, and ``service`` fields.
- A :data:`_correlation_id` :class:`~contextvars.ContextVar` that propagates a
  request-scoped identifier into every log record emitted within that context.
- Module-level :func:`get_correlation_id` / :func:`set_correlation_id` helpers
  for callers that manage request boundaries.
- A static :class:`Logger` facade that preserves the existing call-site API
  (``Logger.info``, ``Logger.debug``, etc.) so no other module needs changing.

Typical startup usage::

    # In run.py or consumer.py — once, before anything else logs
    from utils.logger import Logger
    Logger.start()

Per-request usage (WebSocket / REST)::

    from utils.logger import set_correlation_id
    set_correlation_id(str(uuid.uuid4()))
"""

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
    """Return the correlation ID active in the current execution context."""
    return _correlation_id.get()


def set_correlation_id(value: str) -> None:
    """Bind a correlation ID to the current execution context."""
    _correlation_id.set(value)


# ---------------------------------------------------------------------------
# Custom JSON formatter (zero external dependencies)
# ---------------------------------------------------------------------------


class _ChalieJsonFormatter(logging.Formatter):
    """JSON log formatter that emits one JSON object per log line.

    Fields emitted: ``timestamp``, ``level``, ``logger``, ``service``,
    ``message``, and optionally ``correlation_id`` and ``exc_info``.
    """

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
    """Static facade over the standard ``logging`` module.

    Call :meth:`start` once at process startup before issuing any log messages.
    """

    @staticmethod
    def start() -> None:
        """Initialise the root logger with JSON-formatted output handlers."""
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
        # /health, /system/observability/tasks, /scheduler every 30–60s —
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
