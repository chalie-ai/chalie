"""
Write Queue Service — Serialised SQLite write dispatcher.

All SQLite write operations (INSERT / UPDATE / DELETE) that would otherwise
race under concurrent callers are funnelled through a single background thread
that drains a :class:`queue.Queue`.  This eliminates ``SQLITE_BUSY`` under
high write concurrency while keeping call-sites simple.

Three access patterns are supported:

- **Fire-and-forget** (:meth:`WriteQueueService.submit`) — caller returns
  immediately; the write happens asynchronously.
- **Synchronous** (:meth:`WriteQueueService.submit_sync`) — caller blocks
  until the callable completes and receives its return value.
- **Transactional** (:meth:`WriteQueueService.submit_transaction`) — multiple
  callables share a single SQLite connection/transaction; the whole batch is
  rolled back on any failure.

Typical usage::

    from services.write_queue_service import get_write_queue

    wq = get_write_queue()
    wq.submit(db.execute, "INSERT INTO foo (bar) VALUES (?)", ("baz",))
    row_id = wq.submit_sync(db.execute, "INSERT INTO foo (bar) VALUES (?)", ("qux",))

Design notes
------------
- Internal implementation uses :class:`queue.Queue` (unbounded) and a single
  daemon thread with a per-name lock pattern, simplified to one shared thread.
- The background thread obtains its own thread-local SQLite connection from
  :class:`~services.database_service.DatabaseService`, keeping it separate
  from callers' connections.
- For :meth:`submit_transaction`, each callable in *fns* receives an open
  ``sqlite3.Connection`` as its sole positional argument.  The caller is
  responsible for using that connection (e.g. ``conn.cursor().execute(…)``).
"""

import logging
import queue
import threading
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Item types placed on the internal queue ────────────────────────────────

_TYPE_SINGLE = "single"
_TYPE_TRANSACTION = "transaction"


class _QueueItem:
    """Internal container for a single work item placed on the write queue.

    Attributes:
        item_type: One of the ``_TYPE_*`` module-level constants.
        fn: Callable to execute for ``_TYPE_SINGLE`` items.
        args: Positional arguments forwarded to *fn*.
        kwargs: Keyword arguments forwarded to *fn*.
        fns: List of callables for ``_TYPE_TRANSACTION`` items.
        done_event: :class:`threading.Event` set by the worker when execution
            completes.  ``None`` for fire-and-forget items.
        result: Mutable one-element list used to pass the callable's return
            value (or raised exception) back to a waiting caller.
    """

    __slots__ = ("item_type", "fn", "args", "kwargs", "fns", "done_event", "result")

    def __init__(
        self,
        item_type: str,
        fn: Optional[Callable] = None,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        fns: Optional[List[Callable]] = None,
        done_event: Optional[threading.Event] = None,
        result: Optional[List] = None,
    ) -> None:
        """Initialise a queue work item.

        Args:
            item_type: Execution strategy — ``_TYPE_SINGLE`` for a single
                callable, ``_TYPE_TRANSACTION`` for a multi-callable batch.
            fn: Callable invoked for single items.  Ignored for transactions.
            args: Positional arguments forwarded to *fn* (single items only).
            kwargs: Keyword arguments forwarded to *fn* (single items only).
                Defaults to an empty dict when ``None``.
            fns: Ordered list of callables for transaction items.  Each
                callable receives an open ``sqlite3.Connection`` as its sole
                argument.
            done_event: :class:`threading.Event` signalled by the worker on
                completion.  ``None`` means fire-and-forget.
            result: One-element list whose first slot receives either the
                callable's return value or a raised :class:`Exception`.
                ``None`` for fire-and-forget items.
        """
        self.item_type = item_type
        self.fn = fn
        self.args = args
        self.kwargs = kwargs if kwargs is not None else {}
        self.fns = fns or []
        self.done_event = done_event
        self.result = result  # [value] or [exception_instance] after execution


class WriteQueueService:
    """Serialised SQLite write dispatcher backed by a single daemon thread.

    All public *submit* methods are thread-safe.  The constructor starts the
    background worker immediately; no further setup is required.

    Attributes:
        _queue: Unbounded :class:`queue.Queue` that holds pending work items.
        _thread: Daemon :class:`threading.Thread` draining the queue.
        _processed: Running count of successfully completed write operations.
        _errors: Running count of write operations that raised an exception.
        _stats_lock: :class:`threading.Lock` protecting the counters above.
    """

    def __init__(self) -> None:
        """Initialise the service and start the background drain thread.

        The drain thread is started as a *daemon* so it does not prevent the
        Python interpreter from exiting when the main thread finishes.
        """
        self._queue: queue.Queue = queue.Queue()
        self._processed: int = 0
        self._errors: int = 0
        self._stats_lock = threading.Lock()

        self._thread = threading.Thread(
            target=self._drain,
            name="write-queue-worker",
            daemon=True,
        )
        self._thread.start()
        logger.info("[WriteQueue] Background drain thread started")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> None:
        """Enqueue a callable for fire-and-forget execution in the background.

        Returns immediately without blocking.  The callable is executed by the
        background drain thread in FIFO order.  Any exception raised by *fn*
        is logged but does not propagate to the caller.

        Args:
            fn: Callable to execute asynchronously.
            *args: Positional arguments forwarded to *fn*.
            **kwargs: Keyword arguments forwarded to *fn*.
        """
        item = _QueueItem(
            item_type=_TYPE_SINGLE,
            fn=fn,
            args=args,
            kwargs=kwargs,
        )
        self._queue.put(item)

    def submit_sync(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Enqueue a callable and block until it completes, returning its result.

        The callable is executed by the background drain thread.  If *fn*
        raises an exception that exception is re-raised in the calling thread.

        Args:
            fn: Callable to execute.
            *args: Positional arguments forwarded to *fn*.
            **kwargs: Keyword arguments forwarded to *fn*.

        Returns:
            Whatever *fn* returns.

        Raises:
            Exception: Any exception raised by *fn* is propagated to the
                calling thread verbatim.
        """
        done_event = threading.Event()
        result: List[Any] = [None]  # result[0] = value or exception

        item = _QueueItem(
            item_type=_TYPE_SINGLE,
            fn=fn,
            args=args,
            kwargs=kwargs,
            done_event=done_event,
            result=result,
        )
        self._queue.put(item)
        done_event.wait()

        if isinstance(result[0], BaseException):
            raise result[0]
        return result[0]

    def submit_transaction(self, fns: List[Callable]) -> None:
        """Enqueue a batch of callables to execute inside a single transaction.

        All callables in *fns* are invoked sequentially within a single SQLite
        connection/transaction.  If any callable raises an exception the entire
        transaction is rolled back and the exception is logged.

        Each callable in *fns* must accept exactly one positional argument: an
        open ``sqlite3.Connection``.  The caller is responsible for using that
        connection to issue SQL (e.g. ``conn.cursor().execute(…)``).

        The method blocks until the transaction completes (commit or rollback).

        Args:
            fns: Ordered list of callables, each receiving an open
                ``sqlite3.Connection`` as its sole argument.

        Raises:
            Exception: The exception from the failing callable is re-raised in
                the calling thread after the transaction has been rolled back.
        """
        if not fns:
            return

        done_event = threading.Event()
        result: List[Any] = [None]

        item = _QueueItem(
            item_type=_TYPE_TRANSACTION,
            fns=fns,
            done_event=done_event,
            result=result,
        )
        self._queue.put(item)
        done_event.wait()

        if isinstance(result[0], BaseException):
            raise result[0]

    def get_stats(self) -> Dict[str, int]:
        """Return a snapshot of the queue's runtime statistics.

        The snapshot is consistent within the method call but may lag real-time
        by one work item due to the lock-free queue size read.

        Returns:
            A dict with three integer keys:

            - ``queue_size`` — number of items currently waiting in the queue
              (does not include the item currently being executed, if any).
            - ``processed`` — total number of items successfully completed
              since the service was created.
            - ``errors`` — total number of items that raised an exception since
              the service was created.
        """
        with self._stats_lock:
            return {
                "queue_size": self._queue.qsize(),
                "processed": self._processed,
                "errors": self._errors,
            }

    # ------------------------------------------------------------------
    # Internal drain loop
    # ------------------------------------------------------------------

    def _drain(self) -> None:
        """Drain the work queue in an infinite loop (runs in the daemon thread).

        Blocks on :meth:`queue.Queue.get` waiting for the next item, then
        dispatches to :meth:`_execute_single` or :meth:`_execute_transaction`
        depending on the item type.  After each dispatch the item's
        ``result[0]`` is inspected: a :class:`BaseException` instance means the
        write failed (``_errors`` incremented); any other value means success
        (``_processed`` incremented).  Counter updates are protected by
        ``_stats_lock``.

        This method never returns; the daemon thread exits when the process
        terminates.
        """
        logger.debug("[WriteQueue] Drain loop started")
        while True:
            item: _QueueItem = self._queue.get()
            try:
                if item.item_type == _TYPE_TRANSACTION:
                    self._execute_transaction(item)
                else:
                    self._execute_single(item)

                # Determine success/failure from stored result (avoids
                # double-logging that would occur if sub-methods re-raised)
                failed = (
                    item.result is not None
                    and isinstance(item.result[0], BaseException)
                )
                with self._stats_lock:
                    if failed:
                        self._errors += 1
                    else:
                        self._processed += 1

            except Exception:
                # Unexpected error from outside the try/except in execute_*
                # methods — should not happen in normal operation.
                with self._stats_lock:
                    self._errors += 1
                logger.error(
                    "[WriteQueue] Unexpected exception in drain loop",
                    exc_info=True,
                )
                if item.done_event is not None and not item.done_event.is_set():
                    item.done_event.set()
            finally:
                self._queue.task_done()

    def _execute_single(self, item: _QueueItem) -> None:
        """Execute a single callable work item.

        Calls ``item.fn(*item.args, **item.kwargs)`` and stores the return
        value or raised exception in ``item.result[0]``, then signals
        ``item.done_event`` so a blocked :meth:`submit_sync` caller can
        unblock.  The exception is *not* re-raised here; the drain loop
        inspects ``item.result[0]`` to update success/error counters, and
        :meth:`submit_sync` re-raises in the calling thread.

        Args:
            item: A :class:`_QueueItem` with ``item_type == _TYPE_SINGLE``.
        """
        return_value: Any = None
        exc_caught: Optional[Exception] = None

        try:
            return_value = item.fn(*item.args, **item.kwargs)
        except Exception as exc:
            logger.error(
                f"[WriteQueue] Single item failed: {exc}",
                exc_info=True,
            )
            exc_caught = exc
        finally:
            if item.result is not None:
                item.result[0] = exc_caught if exc_caught is not None else return_value
            if item.done_event is not None:
                item.done_event.set()

    def _execute_transaction(self, item: _QueueItem) -> None:
        """Execute multiple callables atomically within a single SQLite transaction.

        Opens a new ``sqlite3.Connection`` (independent of any thread-local
        connection managed by :class:`~services.database_service.DatabaseService`)
        and manually manages ``BEGIN``/``COMMIT``/``ROLLBACK`` so that all
        callables in ``item.fns`` share the same transaction context.

        Each callable in ``item.fns`` is called with the connection as its sole
        positional argument.  On success the transaction is committed; on any
        failure it is rolled back and the exception is stored for re-raising in
        the caller's thread.

        Args:
            item: A :class:`_QueueItem` with ``item_type == _TYPE_TRANSACTION``.

        Note:
            Exceptions are *not* re-raised here.  The exception instance is
            stored in ``item.result[0]`` and ``item.done_event`` is set; the
            drain loop inspects ``item.result[0]`` for error accounting, and
            :meth:`submit_transaction` re-raises in the calling thread.
        """
        import sqlite3

        from services.database_service import get_db_path

        exc_to_raise: Optional[Exception] = None

        try:
            conn = sqlite3.connect(get_db_path(), timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=15000")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.isolation_level = None  # manual transaction control
            conn.execute("BEGIN")

            try:
                for fn in item.fns:
                    fn(conn)
                conn.execute("COMMIT")
                logger.debug(
                    f"[WriteQueue] Transaction committed ({len(item.fns)} callables)"
                )
            except Exception as exc:
                logger.error(
                    f"[WriteQueue] Transaction failed, rolling back: {exc}",
                    exc_info=True,
                )
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    logger.warning("[WriteQueue] ROLLBACK failed", exc_info=True)
                exc_to_raise = exc
            finally:
                try:
                    conn.close()
                except Exception:
                    logger.warning(
                        "[WriteQueue] Failed to close transaction connection",
                        exc_info=True,
                    )

        except Exception as exc:
            logger.error(
                f"[WriteQueue] Failed to open transaction connection: {exc}",
                exc_info=True,
            )
            exc_to_raise = exc

        if item.result is not None:
            item.result[0] = exc_to_raise
        if item.done_event is not None:
            item.done_event.set()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_write_queue_instance: Optional[WriteQueueService] = None
_write_queue_lock = threading.Lock()


def get_write_queue() -> WriteQueueService:
    """Return the process-wide :class:`WriteQueueService` singleton.

    Creates the instance on first call using a double-checked lock pattern so
    it is safe to call from any thread without external synchronisation.

    Returns:
        The singleton :class:`WriteQueueService` instance with its background
        drain thread already running.
    """
    global _write_queue_instance
    if _write_queue_instance is None:
        with _write_queue_lock:
            if _write_queue_instance is None:
                _write_queue_instance = WriteQueueService()
                logger.info("[WriteQueue] Singleton created")
    return _write_queue_instance
