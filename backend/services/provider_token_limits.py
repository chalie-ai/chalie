import logging
import sqlite3

logger = logging.getLogger(__name__)


def backfill_one(conn: sqlite3.Connection, provider_id: int) -> bool:
    """Failure leaves the row's previous values intact."""
    from services.providers import MAX_CONTEXT_WINDOW
    from services.config_service import ConfigService
    from services.llm_clients.factory import build_client

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

        svc = build_client(dict(pcfg))
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


def backfill_all(conn: sqlite3.Connection) -> dict[str, int]:
    """Returns {'total': N, 'succeeded': N, 'failed': N}. Caller owns commit."""
    rows = conn.execute("SELECT id FROM providers").fetchall()
    succeeded = 0
    failed = 0
    for (pid,) in rows:
        if backfill_one(conn, pid):
            succeeded += 1
        else:
            failed += 1
    return {'total': len(rows), 'succeeded': succeeded, 'failed': failed}
