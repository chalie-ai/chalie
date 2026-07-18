"""Response DTO for the scheduler turns action (``/api/scheduler/turns``).

One active prompt-schedule thread, collapsed to its ``turn_id`` (§13.5) — the
scheduler dock lists these, each a growing, replyable thread on the
``schedule`` channel.
"""

from __future__ import annotations

from .response import Response


class SchedulerTurn(Response):
    """One active prompt-schedule thread.

    ``turn_id`` is the schedule's own ``id`` (no separate stored turn key).
    ``preview`` is the prompt; ``gist`` its generated one-line label (``None``
    until the generation this listing kicks off lands, so the caller falls back
    to ``preview``); ``minute``/``hour``/``day``/``month``/``weekday`` are the
    schedule's crontab field expressions (``*`` = every).
    """

    turn_id: int
    gist: str | None = None
    preview: str
    minute: str = "*"
    hour: str = "*"
    day: str = "*"
    month: str = "*"
    weekday: str = "*"
