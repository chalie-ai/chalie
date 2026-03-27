"""
Capability Scheduler Service — periodic ingestion loop for all connected capability plugins.

Discovers all capability plugins via :func:`capabilities.load_capabilities`, round-robins
over connected ones, calls ``ingest()`` on each, and injects schedule hints into
:class:`WorldStateService` via ``notify_external_signal()``.

Implements per-capability exponential backoff: on consecutive failures the capability is
skipped until its back-off window expires. Tracks last-sync timestamps in ``tool_configs``
for consumption by the status API.

Follows the same worker pattern as :func:`services.decay_engine_service.decay_engine_worker`.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class CapabilitySchedulerService:
    """Periodic scheduler that drives ingestion for all connected capabilities.

    Implements per-capability exponential back-off so a single failing capability
    does not stall the others.  Back-off windows follow ``min(30, 2^n)`` minutes
    where *n* is the number of consecutive failures.

    Attributes:
        interval_minutes: Seconds (converted internally to a sleep duration) between
            full scheduling cycles.
        _failure_counts: Mapping of capability id → consecutive failure count.
        _next_retry: Mapping of capability id → earliest ``time.time()`` at which the
            capability should be retried after a back-off.
    """

    def __init__(self, interval_minutes: int = 15) -> None:
        """Initialise the scheduler.

        Args:
            interval_minutes: How many minutes to sleep between full sync cycles.
                Defaults to 15.
        """
        self.interval_minutes: int = interval_minutes
        self._failure_counts: dict[str, int] = {}
        self._next_retry: dict[str, float] = {}

        logger.info(
            "[CAPABILITY SCHEDULER] Initialised (interval=%d min)", interval_minutes
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sleep_interruptible(self, seconds: float) -> None:
        """Sleep for *seconds* total, using 1-second increments.

        Using 1-second increments allows a ``KeyboardInterrupt`` to surface
        promptly rather than waiting for the full sleep duration to expire.

        Args:
            seconds: Total duration in seconds to sleep.
        """
        elapsed = 0.0
        while elapsed < seconds:
            time.sleep(1)
            elapsed += 1.0

    def _backoff_seconds(self, cap_id: str) -> float:
        """Return the back-off delay in seconds for a capability after a failure.

        Follows ``min(30, 2^n)`` minutes where *n* is the current consecutive
        failure count (before incrementing).

        Args:
            cap_id: Capability identifier.

        Returns:
            Number of seconds to wait before the next retry attempt.
        """
        n = self._failure_counts.get(cap_id, 0)
        backoff_minutes = min(30, 2 ** n)
        return backoff_minutes * 60.0

    def _record_success(self, cap_id: str) -> None:
        """Reset backoff state for *cap_id* after a successful ingest.

        Args:
            cap_id: Capability identifier.
        """
        self._failure_counts.pop(cap_id, None)
        self._next_retry.pop(cap_id, None)

    def _record_failure(self, cap_id: str) -> None:
        """Increment the failure counter and schedule the next retry for *cap_id*.

        Args:
            cap_id: Capability identifier.
        """
        n = self._failure_counts.get(cap_id, 0)
        self._failure_counts[cap_id] = n + 1
        delay = self._backoff_seconds(cap_id)
        self._next_retry[cap_id] = time.time() + delay
        logger.warning(
            "[CAPABILITY SCHEDULER] Capability '%s' failed (consecutive=%d); "
            "back-off %.0f s",
            cap_id,
            n + 1,
            delay,
        )

    def _is_in_backoff(self, cap_id: str) -> bool:
        """Return True if *cap_id* is currently within its back-off window.

        Args:
            cap_id: Capability identifier.

        Returns:
            True if the capability should be skipped this cycle.
        """
        retry_at = self._next_retry.get(cap_id)
        if retry_at is None:
            return False
        return time.time() < retry_at

    def _store_last_sync(self, cap_id: str) -> None:
        """Persist the current UTC timestamp as ``{cap_id}:last_sync_at`` in tool_configs.

        Uses :class:`services.tool_config_service.ToolConfigService` with a shared
        database connection.  Failure is non-fatal and only logged at DEBUG level.

        Args:
            cap_id: Capability identifier; used as both the ``tool_name`` and as part
                of the config key.
        """
        try:
            from services.database_service import get_shared_db_service
            from services.tool_config_service import ToolConfigService
            from services.time_utils import utc_now

            db = get_shared_db_service()
            try:
                svc = ToolConfigService(db)
                svc.set_tool_config(cap_id, {f"{cap_id}:last_sync_at": utc_now().isoformat()})
            finally:
                db.close_pool()
        except Exception as exc:
            logger.debug(
                "[CAPABILITY SCHEDULER] Could not persist last_sync_at for '%s': %s",
                cap_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _run_cycle(self) -> None:
        """Execute one full ingestion cycle across all connected capabilities.

        Loads the current set of capabilities via :func:`capabilities.load_capabilities`,
        filters to those that are connected and not in backoff, calls ``ingest()`` on
        each, and records success or failure.  Each capability is isolated: an exception
        from one never prevents the others from running.
        """
        try:
            from capabilities import load_capabilities
        except Exception as exc:
            logger.error(
                "[CAPABILITY SCHEDULER] Failed to import load_capabilities: %s", exc
            )
            return

        try:
            all_caps = load_capabilities()
        except Exception as exc:
            logger.error(
                "[CAPABILITY SCHEDULER] load_capabilities() raised: %s", exc, exc_info=True
            )
            return

        connected = {
            cap_id: cap
            for cap_id, cap in all_caps.items()
            if cap.is_connected()
        }

        if not connected:
            logger.debug("[CAPABILITY SCHEDULER] No connected capabilities; skipping cycle")
            return

        logger.info(
            "[CAPABILITY SCHEDULER] Cycle start — %d connected capability(ies): %s",
            len(connected),
            list(connected.keys()),
        )

        for cap_id, cap in connected.items():
            if self._is_in_backoff(cap_id):
                remaining = self._next_retry[cap_id] - time.time()
                logger.debug(
                    "[CAPABILITY SCHEDULER] Skipping '%s' — back-off for %.0f s more",
                    cap_id,
                    remaining,
                )
                continue

            try:
                logger.debug("[CAPABILITY SCHEDULER] Ingesting '%s' …", cap_id)
                cap.ingest()
                self._record_success(cap_id)
                self._store_last_sync(cap_id)
                logger.info("[CAPABILITY SCHEDULER] Ingest complete for '%s'", cap_id)
            except Exception as exc:
                logger.error(
                    "[CAPABILITY SCHEDULER] Ingest failed for '%s': %s",
                    cap_id,
                    exc,
                    exc_info=True,
                )
                self._record_failure(cap_id)

    def run(self, shared_state: Optional[dict] = None) -> None:
        """Main service loop — runs capability ingestion cycles indefinitely.

        Sleeps for ``interval_minutes * 60`` seconds between cycles using
        1-second increments so a ``KeyboardInterrupt`` is handled promptly.

        Args:
            shared_state: Optional shared-state dict injected by the worker
                manager (unused internally; reserved for future cross-worker
                co-ordination).
        """
        logger.info("[CAPABILITY SCHEDULER] Service started")

        while True:
            try:
                self._run_cycle()
                self._sleep_interruptible(self.interval_minutes * 60)

            except KeyboardInterrupt:
                logger.info("[CAPABILITY SCHEDULER] Shutdown requested; exiting loop")
                break
            except Exception as exc:
                logger.error(
                    "[CAPABILITY SCHEDULER] Unexpected error in main loop: %s",
                    exc,
                    exc_info=True,
                )
                logger.info("[CAPABILITY SCHEDULER] Waiting 60 s before retry …")
                self._sleep_interruptible(60)


# ---------------------------------------------------------------------------
# Module-level worker function — follows the same pattern as decay_engine_worker
# ---------------------------------------------------------------------------


def capability_scheduler_worker(shared_state=None):
    """Module-level entry point for worker registration via :class:`WorkerManager`.

    Instantiates :class:`CapabilitySchedulerService` with the default interval
    and delegates to its :meth:`~CapabilitySchedulerService.run` method.  This
    function signature is intentionally identical to other workers such as
    :func:`services.decay_engine_service.decay_engine_worker`.

    Args:
        shared_state: Optional dict passed through by the worker manager.  May
            be ``None``; forwarded unchanged to
            :meth:`CapabilitySchedulerService.run`.
    """
    service = CapabilitySchedulerService()
    service.run(shared_state)
