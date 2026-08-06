# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests for the cron runner's dispatch core (``_poll_and_dispatch``).

The runner unifies the old ``scheduler_service`` poll loop and the
``subconscious_worker`` tick: every wall-clock minute it iterates ``cron.JOBS``,
asks each job ``should_run()``, and dispatches the runnable ones on their own
daemon threads under a non-reentrant per-job lock. This exercises that dispatch
contract with REAL ``ScheduledJob`` subclasses — never a ``Mock`` — swapping only
the ``JOBS`` registry (the runner reads it via ``from cron import JOBS`` at call
time, so patching ``cron.JOBS`` installs the test fleet).

The contract under test, one property per test:
  * a job whose gate says yes fires; a job whose gate says no never fires;
  * one job raising in ``should_run`` never starves the jobs after it;
  * one job raising in ``execute`` is isolated AND its lock is released (the next
    tick can re-run it — no permanent wedge);
  * a job whose lock is already held (still running from a prior tick) is skipped
    for this minute rather than run re-entrantly.

Threads are synchronised on a real ``threading.Event`` per job (``.wait(timeout)``
returns the instant the job signals — never a blind ``sleep``); the negative
assertions use a short bounded wait for a signal that must never arrive.
"""

import threading
from unittest.mock import patch

import pytest

from cron.base import ScheduledJob
from cron.runner import CronRunner

pytestmark = pytest.mark.unit

_JOBS_REGISTRY = "cron.JOBS"
_FIRED_TIMEOUT = 5.0
_NOT_FIRED_TIMEOUT = 0.3


class _RecordingJob(ScheduledJob):
    """A real job that records each ``execute`` and can be told whether to gate.

    ``runnable`` drives ``should_run``; ``fired`` is set from inside the actual
    ``_run`` work hook, so a set event proves the runner drove the whole
    ``execute`` → ``_run`` path on a background thread.
    """

    def __init__(self, name: str, *, runnable: bool = True) -> None:
        super().__init__()
        self.name = name
        self._runnable = runnable
        self.fired = threading.Event()
        self.run_count = 0

    def should_run(self) -> bool:
        return self._runnable

    def _run(self) -> str:
        self.run_count += 1
        self.fired.set()
        return "ran"


class _RaisingGateJob(_RecordingJob):
    """Blows up in ``should_run`` — the runner must log and move on."""

    def should_run(self) -> bool:
        raise RuntimeError("gate boom")


class _RaisingRunJob(_RecordingJob):
    """Fires (gate says yes) but blows up inside the work — the runner must
    isolate it and still release the lock in ``_run_job``'s ``finally``."""

    def _run(self) -> str:
        self.fired.set()
        raise RuntimeError("work boom")


def test_runnable_job_fires_and_non_runnable_is_skipped() -> None:
    runnable = _RecordingJob("runnable", runnable=True)
    skipped = _RecordingJob("skipped", runnable=False)

    with patch(_JOBS_REGISTRY, (runnable, skipped)):
        CronRunner._poll_and_dispatch()

    assert runnable.fired.wait(_FIRED_TIMEOUT), "a job whose gate says yes must fire"
    assert not skipped.fired.wait(_NOT_FIRED_TIMEOUT), "a job whose gate says no must never fire"
    assert runnable.run_count == 1


def test_raising_should_run_does_not_starve_later_jobs() -> None:
    bad_gate = _RaisingGateJob("bad_gate")
    after = _RecordingJob("after", runnable=True)

    # bad_gate is first in the registry: if its raising gate broke the loop, the
    # job after it would never be reached.
    with patch(_JOBS_REGISTRY, (bad_gate, after)):
        CronRunner._poll_and_dispatch()

    assert after.fired.wait(_FIRED_TIMEOUT), (
        "a job raising in should_run must not stop the runner reaching later jobs"
    )
    assert not bad_gate.fired.wait(_NOT_FIRED_TIMEOUT), "the raising-gate job never runs"


def test_raising_execute_is_isolated_and_releases_lock() -> None:
    bad_run = _RaisingRunJob("bad_run")
    after = _RecordingJob("after", runnable=True)

    with patch(_JOBS_REGISTRY, (bad_run, after)):
        CronRunner._poll_and_dispatch()

    assert bad_run.fired.wait(_FIRED_TIMEOUT), "the job entered its work before raising"
    assert after.fired.wait(_FIRED_TIMEOUT), "a raising execute must not starve later jobs"

    # The lock must be free again — a wedged lock would silently skip the job
    # forever. Joining the job's own thread guarantees ``_run_job``'s finally ran.
    for t in threading.enumerate():
        if t.name == f"cron-{bad_run.name}":
            t.join(_FIRED_TIMEOUT)
    assert bad_run._lock.acquire(blocking=False), (
        "the lock must be released after a raising execute so the next tick can re-run the job"
    )
    bad_run._lock.release()


def test_already_running_job_is_skipped_this_tick() -> None:
    busy = _RecordingJob("busy", runnable=True)
    # Simulate a still-in-flight prior tick: hold the job's lock so this tick's
    # non-reentrant acquire fails and the job is skipped rather than re-entered.
    assert busy._lock.acquire(blocking=False)

    try:
        with patch(_JOBS_REGISTRY, (busy,)):
            CronRunner._poll_and_dispatch()
        assert not busy.fired.wait(_NOT_FIRED_TIMEOUT), (
            "a job whose lock is already held must be skipped, never run re-entrantly"
        )
        assert busy.run_count == 0
    finally:
        busy._lock.release()
