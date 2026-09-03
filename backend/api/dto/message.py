"""DTO for one transcript row in the conversation feed."""

from __future__ import annotations

from .attachment import Attachment
from .base import DTO
from .chip import Chip
from .segment import Segment


class Thinking(DTO):
    """Chain-of-thought traces stored for a transcript row.

    Present only when the row has one or more ``transcript_thinking`` rows;
    absent / ``None`` otherwise. Traces are in row order; ``duration_ms`` and
    ``tokens`` are the sums across all traces anchored to that row.
    """

    traces: list[str]
    duration_ms: int
    tokens: int


class Message(DTO):
    """One transcript row.

    Role-conditional: ``attachments`` appears on user rows only; ``segments`` on
    non-user rows only. ``tool_calls`` appears on WHATEVER row anchors them —
    including the user input row (a step-1 tool-only call anchors there before
    any assistant text is written). ``timestamp`` is a
    pre-formatted locale string, not a datetime — a display string carrying no
    year, so it is never parseable back into a date. ``day`` is the same
    instant as the user's local calendar day (``YYYY-MM-DD``), the machine-
    readable key the conversation spine groups its date dividers by; it exists
    so the client never has to parse ``timestamp``. ``thread_message`` is set on
    rows past the turn's settle0 — the fork-reply continuation (a turn that has
    any is a thread; the main spine renders only through settle0).
    ``settled`` projects the transcript column on assistant rows — the client
    needs it to place the speaker button on the turn's final reply and nowhere
    else. ``voice_state`` carries that row's speech pre-synthesis outcome
    (``ready`` once the audio is stored, ``failed`` once the pipeline gave up);
    ``None`` on a settled row means no outcome is recorded yet, so the first
    speaker press starts the pipeline through the playback route. ``thinking``
    carries that row's stored chain-of-thought traces (traces in row order,
    summed ``duration_ms`` and ``tokens``); absent on rows with no thinking
    rows.
    """

    id: str
    role: str
    content: str
    timestamp: str
    day: str
    turn_id: int | None = None
    attachments: list[Attachment] | None = None
    tool_calls: list[Chip] | None = None
    segments: list[Segment] | None = None
    thread_message: bool | None = None
    settled: bool | None = None
    voice_state: str | None = None
    thinking: Thinking | None = None
