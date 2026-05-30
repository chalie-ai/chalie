"""
CompactionAbility — internal continuity-compaction tool.

``INTERNAL = True``: never surfaced to the LLM. It is excluded from every
processor's tool list, from ``find_tools`` / ``abilities.sqlite``, and from the
policy UI (enforced by ``Ability.__init_subclass__``, ``AbilityRegistry``, and
``utils/build_ability_db``). The model can never emit a ``compaction`` action.

Compaction stays automatic: ``MessageProcessor._run_full_compaction()`` invokes
this ability directly (the same internal-invoke pattern ``UserMessageProcessor``
uses for ``memory``) when the ACT loop overflows the provider context window.
Routing compaction through the ability + tool-event machinery gives it the same
ACT-trail pill and audit logging every other tool call gets — the wrapper emits
``act_tool_start`` / ``act_tool_end`` around this call.

This ability owns the continuity-compaction execution: read transcript entries
since the watermark, build the continuity-first LLM envelope, dispatch
``ContinuityCompactionProcessor``, parse ``<summary>``, and write the append-only
``tool_calls`` audit row (``ephemeral=0``). It is stateless — the turn state it
needs (channel, exclude_id, transcript_id) arrives via ``params``.
"""

import json
import logging
from typing import ClassVar

from abilities._base import Ability
from services.time_utils import utc_now

logger = logging.getLogger(__name__)


class CompactionAbility(Ability):
    """Internal tool that runs a full continuity compaction for one channel.

    Returns a structured dict (never an LLM-facing string — it is invoked
    internally, not dispatched):

        {
          'status':   'success' | 'failure',
          'summary':  str | None,            # <summary> body on success
          'watermark': int,                  # new watermark (advanced on success)
          '_metrics': MetricsAccumulator | None,  # compaction LLM usage to merge
        }
    """

    NAME: ClassVar[str] = 'compaction'
    INTERNAL: ClassVar[bool] = True
    INPUT_SCHEMA: ClassVar[dict] = {
        'type': 'object',
        'properties': {
            'exclude_id': {
                'type': ['integer', 'null'],
                'description': (
                    "Transcript row id to exclude from the compaction — the "
                    "current, not-yet-answered turn. None compacts everything "
                    "since the watermark."
                ),
            },
            'transcript_id': {
                'type': ['integer', 'null'],
                'description': (
                    "Transcript row id the durable audit tool_calls row is "
                    "linked to. None skips the audit write."
                ),
            },
        },
        'required': [],
    }

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        del telemetry  # compaction is location-agnostic
        from services import compaction_persistence
        from services.message_processor import (
            _build_compaction_input,
            _COMPACTION_FAILURE_FMT,
            _extract_compaction_summary,
            _format_compaction_entry,
        )

        exclude_id = params.get('exclude_id')
        transcript_id = params.get('transcript_id')

        prior = compaction_persistence.get_compaction(channel)
        watermark = prior['compacted_up_to_id'] if prior else 0
        prev_text = (prior.get('compacted_text') or '').strip() if prior else ''

        all_entries = list(compaction_persistence.get_entries_since(channel, watermark))

        # Filter out the current turn's input row so the LLM does not see an
        # incomplete user message (question not yet answered).
        if exclude_id is not None:
            entries = [e for e in all_entries if e.get('id') != exclude_id]
        else:
            entries = all_entries

        # Nothing to compact — bail before hitting the LLM.
        if not entries and not prev_text:
            logger.warning(
                "[COMPACTION] %s: compaction invoked with no entries and no prior "
                "checkpoint — skipping LLM call",
                channel,
            )
            return self._fail(watermark, metrics=None)

        rendered = [_format_compaction_entry(e) for e in entries]
        compaction_input = _build_compaction_input(prev_text, rendered)
        in_chars = len(compaction_input)

        raw_output, metrics = self._dispatch_llm(
            channel, compaction_input, watermark, transcript_id
        )
        if raw_output is None:
            return self._fail(watermark, metrics=metrics)

        if not raw_output:
            reason = "LLM returned empty output"
            logger.warning(_COMPACTION_FAILURE_FMT, channel, reason)
            self._write_audit_row(
                transcript_id, watermark=watermark, status='failure',
                summary='', reason=reason,
            )
            return self._fail(watermark, metrics=metrics)

        summary = _extract_compaction_summary(raw_output)
        if not summary:
            reason = "no <summary> tags in LLM output"
            logger.warning(_COMPACTION_FAILURE_FMT, channel, reason)
            self._write_audit_row(
                transcript_id, watermark=watermark, status='failure',
                summary='', reason=reason,
            )
            return self._fail(watermark, metrics=metrics)

        # Watermark advances to the highest ID fed to the LLM.
        new_watermark = max((e.get('id', 0) for e in entries), default=watermark)
        self._write_audit_row(
            transcript_id, watermark=new_watermark, status='success', summary=summary,
        )
        logger.info(
            "[COMPACTION] %s: continuity success — in=%d chars, out=%d chars, "
            "watermark %d→%d",
            channel, in_chars, len(summary), watermark, new_watermark,
        )
        return {
            'status': 'success',
            'summary': summary,
            'watermark': new_watermark,
            '_metrics': metrics,
        }

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fail(watermark: int, *, metrics) -> dict:
        return {
            'status': 'failure',
            'summary': None,
            'watermark': watermark,
            '_metrics': metrics,
        }

    def _dispatch_llm(self, channel, compaction_input, watermark, transcript_id):
        """Invoke ContinuityCompactionProcessor; return (raw_output, metrics).

        On PayloadTooLargeError / generic exception writes a failure audit row
        and returns (None, metrics) so the caller can bail. Always returns the
        processor's metrics so the parent can merge compaction LLM usage even
        when the call fails. Never raises.
        """
        from services.compaction_message_processor import ContinuityCompactionProcessor
        from services.llm_service import PayloadTooLargeError
        from services.message_processor import _COMPACTION_FAILURE_FMT

        proc = ContinuityCompactionProcessor(raw_input=compaction_input)
        metrics = getattr(proc, '_metrics', None)
        try:
            raw_output = (proc.send() or '').strip()
            return raw_output, metrics
        except PayloadTooLargeError as exc:
            reason = f"compaction LLM hit HTTP 413: {exc}"
            logger.error(_COMPACTION_FAILURE_FMT, channel, reason)
            self._write_audit_row(
                transcript_id, watermark=watermark, status='failure',
                summary='', reason=reason,
            )
            return None, metrics
        except Exception as exc:
            reason = f"LLM error: {exc}"
            logger.error(_COMPACTION_FAILURE_FMT, channel, reason)
            self._write_audit_row(
                transcript_id, watermark=watermark, status='failure',
                summary='', reason=reason,
            )
            return None, metrics

    @staticmethod
    def _write_audit_row(
        transcript_id: 'int | None',
        *,
        watermark: int,
        status: str,
        summary: str,
        reason: str = '',
    ) -> None:
        """Write an append-only ``tool_calls`` audit row for a compaction attempt.

        Success rows (``status='success'``) are picked up by the canonical
        ``compaction_persistence.get_compaction()`` lookup. Failure rows are
        stored for traceability but invisible to the lookup (filtered by
        ``json_extract(params, '$.status') = 'success'``).

        Never raises — persistence failure is logged and swallowed.
        """
        if transcript_id is None:
            return
        try:
            from services.database_service import get_shared_db_service
            row_params: dict = {'compacted_up_to_id': watermark, 'status': status}
            if reason:
                row_params['reason'] = reason
            db = get_shared_db_service()
            with db.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO tool_calls
                        (transcript_id, tool_name, params, result, ephemeral, created_at)
                    VALUES (?, 'compaction', ?, ?, 0, ?)
                    """,
                    (
                        transcript_id,
                        json.dumps(row_params),
                        summary,
                        utc_now().isoformat(),
                    ),
                )
        except Exception as exc:
            logger.warning(
                "[COMPACTION] failed to write audit row (status=%s): %s",
                status, exc,
            )
