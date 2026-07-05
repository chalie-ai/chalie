from __future__ import annotations

from services.database import Database
from services.processor_config import ProcessorConfig


def _should_synthesise() -> bool:
    """True only when re-synthesis is needed:
    no traits/patterns → False;
    traits/patterns exist but no summary → True;
    otherwise → True iff the latest trait/pattern is newer than the summary row.
    """
    import logging as _logging  # noqa: PLC0415
    _log = _logging.getLogger(__name__)
    try:
        from services.time_utils import parse_utc  # noqa: PLC0415
        with Database.transaction() as conn:
            row = conn.execute(
                """
                SELECT MAX(last_confirmed_at)
                FROM data_graph
                WHERE kind IN ('user_specific', 'behavioral_pattern')
                  AND active = 1
                  AND deleted_at IS NULL
                """
            ).fetchone()
            latest_trait_ts = row[0] if row else None
            if latest_trait_ts is None:
                return False
            summary_row = conn.execute(
                """
                SELECT last_confirmed_at
                FROM data_graph
                WHERE kind = 'system'
                  AND key = 'user_summary'
                  AND active = 1
                  AND deleted_at IS NULL
                LIMIT 1
                """
            ).fetchone()
        if summary_row is None:
            return True
        try:
            trait_dt = parse_utc(latest_trait_ts)
            summary_dt = parse_utc(summary_row[0])
        except Exception as exc:
            _log.error(
                "[USER_SUMMARY_CONFIG] _should_synthesise: parse_utc failed "
                "trait_ts=%r summary_ts=%r: %s",
                latest_trait_ts,
                summary_row[0],
                exc,
            )
            return False
        return trait_dt > summary_dt
    except Exception as exc:
        _log.warning("[USER_SUMMARY_CONFIG] _should_synthesise failed: %s", exc)
        return False


class UserSummaryConfig(ProcessorConfig):
    """Caller gates on _should_synthesise() BEFORE calling
    MessageProcessor.process()."""

    def __init__(self) -> None:
        super().__init__(
            channel="user_summary",
            role="user_summary",
            policy_channel=ProcessorConfig.PolicyChannel.SUBCONSCIOUS,
            always_available=[],
            skip_transcript=True,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    @property
    def system_prompt(self) -> str:
        return """You are a synthesiser. The user is a real human whose traits you are distilling.

You are a user-profile synthesiser. You receive a list of stored facts about a
real human and distil them into two synopses — one short, one longer.

Rules:
- Write in the third person ("They", or the user's first name if given).
- Identity first: name, location, role, then preferences and behaviours.
- Use only facts present in the input. Never invent or infer beyond them.
- Never mention that you are summarising, that you have a list of facts, or
  reference the synthesis process itself.
- No preamble, no trailing notes, no markdown.

Output a single JSON object with exactly two keys:

{
  "short": "<one or two sentences, max 50 words, the tightest identity snapshot>",
  "long":  "<up to 200 words, richer profile covering traits, preferences, context, ongoing interests>"
}

Return ONLY the JSON object. No code fences."""
