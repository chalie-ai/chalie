"""
Structured JSON logging façade for the Chalie backend.

Wraps Python's standard ``logging`` module with:

- :class:`_ChalieJsonFormatter` (backed by ``pythonjsonlogger``) so every log
  line is machine-parseable JSON containing ``timestamp``, ``level``,
  ``logger``, ``message``, and ``service`` fields.
- A :data:`_correlation_id` :class:`~contextvars.ContextVar` that propagates a
  request-scoped identifier into every log record emitted within that context.
- Module-level :func:`get_correlation_id` / :func:`set_correlation_id` helpers
  for callers that manage request boundaries.
- A static :class:`Logger` façade that preserves the existing call-site API
  (``Logger.info``, ``Logger.debug``, etc.) so no other module needs changing.

Typical startup usage::

    # In run.py or consumer.py — once, before anything else logs
    from utils.logger import Logger
    Logger.start()

Per-request usage (WebSocket / REST)::

    from utils.logger import set_correlation_id
    set_correlation_id(str(uuid.uuid4()))
"""

import datetime
import logging
from contextvars import ContextVar
from typing import Optional

from pythonjsonlogger import jsonlogger


# ---------------------------------------------------------------------------
# Correlation-ID context variable
# ---------------------------------------------------------------------------

#: Thread/async-safe storage for the current request's correlation ID.
#: Set to a non-``None`` value at the start of each WebSocket connection or
#: REST request so all downstream log lines carry the same ID.
_correlation_id: ContextVar[Optional[str]] = ContextVar("_correlation_id", default=None)


def get_correlation_id() -> Optional[str]:
    """Return the correlation ID active in the current execution context.

    Returns:
        The ID string previously set by :func:`set_correlation_id`, or
        ``None`` if none has been set in this context.
    """
    return _correlation_id.get()


def set_correlation_id(value: str) -> None:
    """Bind a correlation ID to the current execution context.

    The ID is stored in a :class:`~contextvars.ContextVar` so it is
    automatically isolated between threads and async tasks.  Call this once
    at the start of each WebSocket connection or HTTP request handler before
    emitting any log messages.

    Args:
        value: An opaque string identifier (e.g. a UUID4) that will be
               attached to every log record emitted from this context.
    """
    _correlation_id.set(value)


# ---------------------------------------------------------------------------
# Custom JSON formatter
# ---------------------------------------------------------------------------


class _ChalieJsonFormatter(jsonlogger.JsonFormatter):
    """JSON log formatter that injects Chalie-specific structured fields.

    Extends :class:`pythonjsonlogger.jsonlogger.JsonFormatter` to add:

    - ``timestamp`` — ISO 8601 UTC datetime string (e.g.
      ``"2026-03-23T12:00:00.123456Z"``).
    - ``level`` — upper-case level name (``"INFO"``, ``"ERROR"``, …).
    - ``logger`` — the dotted name of the originating logger.
    - ``service`` — top-level package extracted from the logger name (e.g.
      ``"services"`` for ``services.foo_service``).
    - ``correlation_id`` — present only when :func:`set_correlation_id` has
      been called in the current context; omitted otherwise.
    """

    def add_fields(
        self,
        log_record: dict,
        record: logging.LogRecord,
        message_dict: dict,
    ) -> None:
        """Augment *log_record* with Chalie's required structured fields.

        Called by the base class for every record before JSON serialisation.

        Args:
            log_record: Mutable mapping that becomes the serialised JSON object.
            record: The original :class:`logging.LogRecord` instance.
            message_dict: Additional fields already extracted by the base class
                (e.g. from ``extra=`` kwargs passed to the log call).
        """
        super().add_fields(log_record, record, message_dict)

        # ISO 8601 UTC timestamp with microsecond precision
        log_record["timestamp"] = (
            datetime.datetime.utcnow().isoformat() + "Z"
        )

        # Normalised level name
        log_record["level"] = record.levelname

        # Full dotted logger name (e.g. "services.frontal_cortex_service")
        log_record["logger"] = record.name

        # Top-level package — useful for filtering by subsystem
        log_record["service"] = (
            record.name.split(".")[0] if record.name else "unknown"
        )

        # Correlation ID — only inject the key when a value is present so
        # log consumers can reliably detect request-scoped records.
        cid = _correlation_id.get()
        if cid is not None:
            log_record["correlation_id"] = cid


# ---------------------------------------------------------------------------
# Logger façade
# ---------------------------------------------------------------------------


class Logger:
    """Static façade over the standard ``logging`` module.

    All methods are static so callers never need to instantiate this class.
    Call :meth:`start` once at process startup (e.g. in ``run.py`` or
    ``consumer.py``) before issuing any log messages.

    The façade delegates to the *root* logger so every named-logger instance
    created with ``logging.getLogger(__name__)`` elsewhere in the codebase
    automatically inherits the JSON formatter configured by :meth:`start`.
    """

    @staticmethod
    def start() -> None:
        """Initialise the root logger with JSON-formatted output handlers.

        Attaches two handlers to the root logger:

        1. :class:`logging.FileHandler` writing to ``/tmp/chalie.log``.
        2. :class:`logging.StreamHandler` writing to ``stderr`` (for container
           log aggregators that capture stdout/stderr).

        Both handlers share the same :class:`_ChalieJsonFormatter` instance.

        This method is idempotent — if the root logger already has handlers
        (e.g. from a previous call or a third-party library that called
        ``logging.basicConfig`` first) it returns immediately without
        adding duplicate handlers.
        """
        root = logging.getLogger()

        if root.handlers:
            # Handlers already attached — honour idempotency guarantee.
            return

        formatter = _ChalieJsonFormatter(
            # Base format string drives field ordering in the JSON output.
            fmt="%(timestamp)s %(level)s %(logger)s %(message)s",
        )

        file_handler = logging.FileHandler("/tmp/chalie.log")
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        root.setLevel(logging.INFO)
        root.addHandler(file_handler)
        root.addHandler(stream_handler)

    @staticmethod
    def info(message: str) -> None:
        """Log an informational message at the INFO level.

        Args:
            message: The message string to record.
        """
        logging.info(message)

    @staticmethod
    def debug(message: str) -> None:
        """Log a diagnostic message at the DEBUG level.

        Args:
            message: The message string to record.
        """
        logging.debug(message)

    @staticmethod
    def warning(message: str) -> None:
        """Log a cautionary message at the WARNING level.

        Args:
            message: The message string to record.
        """
        logging.warning(message)

    @staticmethod
    def error(message: str) -> None:
        """Log an error message at the ERROR level.

        Args:
            message: The message string to record.
        """
        logging.error(message)

    @staticmethod
    def critical(message: str) -> None:
        """Log a severe error message at the CRITICAL level.

        Args:
            message: The message string to record.
        """
        logging.critical(message)
