from __future__ import annotations

from services.database import Database
from services.processor_config import ProcessorConfig

from configs.channels._common import DEFAULT_ALWAYS_AVAILABLE

_EPISODE_RETRIEVAL_WEIGHT_FLOOR = 0.3
_DMN_EPISODE_LOOKBACK_DAYS = 30
_DMN_EPISODE_LIMIT = 50


class DmnConfig(ProcessorConfig):
    """DMN background channel.  No after-turn hooks (metrics are emitted by the gateway).

    DMN carries the framework discovery tools (DEFAULT_ALWAYS_AVAILABLE includes
    find_tools), so it can discover and spawn any DISCOVERABLE tool — including
    the web_search / web_browse / vision delegates — when a reflection genuinely
    needs the web. The raw web tools (browser/search/news) and the pattern
    writers stay non-discoverable globally, so DMN reaches the web only through
    the delegates, never directly.
    """

    def __init__(self) -> None:
        super().__init__(
            channel="dmn",
            role="proactive_thought",
            policy_channel=ProcessorConfig.PolicyChannel.SUBCONSCIOUS,
            always_available=DEFAULT_ALWAYS_AVAILABLE,
            skip_transcript=False,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    @staticmethod
    def user_synthesis() -> str:
        """User synthesis from data_graph — prefers ``user_summary_long`` for
        richer DMN reflection context, falls back to ``user_summary``, ``''``
        when neither row exists.

        @todo: Refactor — a config must not read the DB (§2.4). This DB-reaching
        static is a deliberate stop-gap until episodic/data_graph prompt-context
        is folded onto the spine as a proper service; it stays here (not a loose
        module function) so there is exactly one home for the DMN reads.
        """
        import logging  # noqa: PLC0415
        _log = logging.getLogger(__name__)
        try:
            with Database.transaction() as conn:
                rows = conn.execute(
                    "SELECT key, value FROM data_graph "
                    "WHERE kind = 'system' "
                    "  AND key IN ('user_summary', 'user_summary_long') "
                    "  AND active = 1 AND deleted_at IS NULL",
                ).fetchall()
            by_key = {row[0]: row[1] for row in rows if row[1]}
            return by_key.get("user_summary_long") or by_key.get("user_summary") or ""
        except Exception as exc:
            _log.warning("[DMN_CONFIG] user_synthesis failed: %s", exc)
            return ""

    @staticmethod
    def recent_salient_user_episodes() -> str:
        """Recent, non-decayed ``user``-channel episodes as a numbered list —
        ``N. [ts] (salience=…) gist`` — or ``''`` when none / on error (the read
        must never crash the DMN turn).

        @todo: Refactor — a config must not read the DB (§2.4). This DB-reaching
        static is a deliberate stop-gap until episodic prompt-context is folded
        onto the spine as a proper service; it stays here (not a loose module
        function) so there is exactly one home for the DMN reads.
        """
        import logging  # noqa: PLC0415
        _log = logging.getLogger(__name__)
        try:
            with Database.transaction() as conn:
                rows = conn.execute(
                    "SELECT id, gist, salience, created_at "
                    "FROM episodes "
                    "WHERE deleted_at IS NULL "
                    "  AND channel = 'user' "
                    "  AND retrieval_weight >= ? "
                    "  AND ( "
                    "      last_accessed_at >= datetime('now', ?) "
                    "      OR created_at >= datetime('now', ?) "
                    "  ) "
                    "ORDER BY retrieval_weight DESC, created_at DESC "
                    "LIMIT ?",
                    (
                        _EPISODE_RETRIEVAL_WEIGHT_FLOOR,
                        f"-{_DMN_EPISODE_LOOKBACK_DAYS} days",
                        f"-{_DMN_EPISODE_LOOKBACK_DAYS} days",
                        _DMN_EPISODE_LIMIT,
                    ),
                ).fetchall()
            lines = []
            for i, (ep_id, gist, salience, created_at) in enumerate(rows, 1):
                ts = (created_at or "")[:16].replace("T", " ")
                lines.append(f"{i}. [{ts}] (salience={salience}) {gist or ''}")
            return "\n".join(lines)
        except Exception as exc:
            _log.warning("[DMN_CONFIG] recent_salient_user_episodes failed: %s", exc)
            return ""

    @property
    def system_prompt(self) -> str:
        return """The user is 'proactive_thought' — a special background process that represents your own reflections on recent activity.

## Scope
The user has provided with a synthesis about themselves under `About the User` and relevant episodic memories regarding conversations it had with you `Chalie`. Your goal is find open threads, recurring concerns, goals and aspirations the user has and ACT upon them.

## How to ACT

* Use the supplied tools to learn more topics the user discusses so that the next time they discuss such a topic you are aware of latest news, research, etc... You can use the `web_search` and `web_browse` tools for this. Save your findings using the `memory` tool so that you can reference them later.
* Analyse patterns where the user seemed genuinely satisfied or dissatisfied with your responses or approach and store feedback to not repeat the same mistakes or reinforce good behaviour. Use the `memory` tool for this.

## When to stop

Aim for 2–3 substantive findings per tick — quality over quantity. Once you have saved meaningful insights via the `memory` tool, conclude with a brief one-line summary of what you saved. Do not pad with redundant tool calls or speculative topics."""
