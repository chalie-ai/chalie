import json
import logging
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Union


# ---------------------------------------------------------------------------
# Correlation-ID context variable
# ---------------------------------------------------------------------------

_correlation_id: ContextVar[str | None] = ContextVar("_correlation_id", default=None)



# ---------------------------------------------------------------------------
# Custom JSON formatter (zero external dependencies)
# ---------------------------------------------------------------------------


class _ChalieJsonFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Union[str, list[str]]] = {
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
# Count-capped file handler — keeps /tmp/chalie.log a rolling window of the
# most recent LOG_FILE_MAX_ENTRIES lines. Single-writer (see api/system.py).
# ---------------------------------------------------------------------------

# api/system.py:observability_errors reads this file to serve the brain dashboard error log.
LOG_FILE_PATH = "/tmp/chalie.log"
LOG_FILE_MAX_ENTRIES = 20_000
# Overshoot allowed before one amortised rewrite. Batching keeps steady-state
# logging O(1) per record instead of an O(n) full-file rewrite on every line.
_LOG_TRIM_BATCH = 2_000
_LOG_TRIM_AT = LOG_FILE_MAX_ENTRIES + _LOG_TRIM_BATCH


class _CountCappedFileHandler(logging.FileHandler):
    """FileHandler that keeps only the most recent LOG_FILE_MAX_ENTRIES lines.

    Counts lines as they are emitted and, once the file overshoots the cap by
    _LOG_TRIM_BATCH, rewrites it keeping the newest LOG_FILE_MAX_ENTRIES lines.
    Each log record is exactly one line (_ChalieJsonFormatter), so line == entry.
    """

    def __init__(self, filename: str) -> None:
        self._line_count = self._count_lines(filename)
        super().__init__(filename)

    @staticmethod
    def _count_lines(filename: str) -> int:
        try:
            with open(filename, "r", errors="replace") as fh:
                return sum(1 for _ in fh)
        except OSError:
            return 0

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self._line_count += 1
        if self._line_count > _LOG_TRIM_AT:
            self._trim()

    def _trim(self) -> None:
        # Rewrite the same inode in "w" mode: O_TRUNC resets it to the kept tail,
        # and the handler's own append stream resumes writing past it (O_APPEND
        # recomputes the offset), so the live stream needs no reopen.
        self.acquire()
        try:
            self.flush()
            with open(self.baseFilename, "r", errors="replace") as fh:
                kept = fh.readlines()[-LOG_FILE_MAX_ENTRIES:]
            with open(self.baseFilename, "w") as fh:
                fh.writelines(kept)
            self._line_count = len(kept)
        finally:
            self.release()


# ---------------------------------------------------------------------------
# Logger facade
# ---------------------------------------------------------------------------


class Logger:
    @staticmethod
    def get_correlation_id() -> str | None:
        return _correlation_id.get()

    @staticmethod
    def set_correlation_id(value: str) -> None:
        _correlation_id.set(value)

    @staticmethod
    def start() -> None:
        root = logging.getLogger()

        if root.handlers:
            return

        formatter = _ChalieJsonFormatter()

        file_handler = _CountCappedFileHandler(LOG_FILE_PATH)
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
