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
    a model call's tool calls on that provider call's own assistant row (empty
    content when the call carried no prose), the turn-zero memory seed on the
    user input row. ``timestamp`` is a
    pre-formatted locale string, not a datetime — a display string carrying no
    year, so it is never parseable back into a date. ``day`` is the same
    instant as the user's local calendar day (``YYYY-MM-DD``), the machine-
    readable key the conversation spine groups its date dividers by; it exists
    so the client never has to parse ``timestamp``. ``created_at`` is the row's
    creation instant as ISO-8601 UTC with its offset — the raw time the client
    needs to time one exchange (the turn-level ``duration_ms``
    spans the whole turn, so it cannot). ``thread_message`` is set on
    rows from the turn's second user row onward — the fork-reply continuation
    (a turn that has any is a thread; the main spine renders only the opener).
    ``settled`` projects the transcript column on assistant rows — set only on
    each exchange's final reply, where the client places the speaker button,
    and nowhere else. ``voice_state`` carries that row's speech pre-synthesis outcome
    (``ready`` once the audio is stored, ``failed`` once the pipeline gave up);
    ``None`` on a settled row means no outcome is recorded yet, so the first
    speaker press starts the pipeline through the playback route. ``thinking``
    carries that row's stored chain-of-thought traces (traces in row order,
    summed ``duration_ms`` and ``tokens``); absent on rows with no thinking
    rows. ``joined`` projects the transcript column on user rows — true on a
    message sent into the turn while it was working, which a cancel keeps.
    """

    id: str
    role: str
    content: str
    timestamp: str
    day: str
    created_at: str
    turn_id: int | None = None
    attachments: list[Attachment] | None = None
    tool_calls: list[Chip] | None = None
    segments: list[Segment] | None = None
    thread_message: bool | None = None
    settled: bool | None = None
    joined: bool | None = None
    voice_state: str | None = None
    thinking: Thinking | None = None
