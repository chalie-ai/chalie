"""
Provider token-limit backfill.

Reads each provider's get_context_limit() and persists max_tokens into
the providers table.  Used at boot (every row) and on provider creation
(single row).  The compaction threshold is no longer a stored column —
Provider.calculate() divides by max_tokens at call time.
"""
import logging

logger = logging.getLogger(__name__)


def backfill_one(conn, provider_id: int) -> bool:
    """Compute and persist max_tokens + compact_at for a single provider row.

    Looks up the provider config via ConfigService.get_providers(), builds an
    LLM service via create_llm_service, calls .get_context_limit(), then
    UPDATEs the row. Caller owns the connection and the commit.

    Returns True on successful UPDATE, False on any failure (failure is
    logged at warning; the row's previous values are left intact).
    """
    from services.providers import MAX_CONTEXT_WINDOW
    from services.config_service import ConfigService
    from services.llm_service import create_llm_service

    try:
        row = conn.execute(
            "SELECT name FROM providers WHERE id = ?",
            (provider_id,),
        ).fetchone()
        if row is None:
            logger.warning(
                "[provider_token_limits] provider id=%s not found — skipping",
                provider_id,
            )
            return False
        name = row[0]

        providers_map = ConfigService.get_providers()
        pcfg = providers_map.get(name)
        if pcfg is None:
            logger.warning(
                "[provider_token_limits] provider '%s' not in config cache — skipping",
                name,
            )
            return False

        svc = create_llm_service(dict(pcfg))
        max_tokens = svc.get_context_limit()
        if not isinstance(max_tokens, (int, float)) or max_tokens <= 0:
            logger.warning(
                "[provider_token_limits] provider '%s' returned non-positive max_tokens=%r — skipping",
                name, max_tokens,
            )
            return False
        max_tokens = min(int(max_tokens), MAX_CONTEXT_WINDOW)
        conn.execute(
            "UPDATE providers SET max_tokens = ?, "
            "updated_at = datetime('now') WHERE id = ?",
            (max_tokens, provider_id),
        )
        logger.info(
            "[provider_token_limits] '%s' max_tokens=%d",
            name, max_tokens,
        )
        return True
    except Exception as exc:
        logger.warning(
            "[provider_token_limits] backfill failed for provider id=%s: %s — leaving previous values intact",
            provider_id, exc,
        )
        return False


def backfill_all(conn) -> dict:
    """Run backfill_one() for every provider row (active and inactive).

    Returns {'total': N, 'succeeded': N, 'failed': N}. Caller owns commit.
    """
    rows = conn.execute("SELECT id FROM providers").fetchall()
    succeeded = 0
    failed = 0
    for (pid,) in rows:
        if backfill_one(conn, pid):
            succeeded += 1
        else:
            failed += 1
    return {'total': len(rows), 'succeeded': succeeded, 'failed': failed}
