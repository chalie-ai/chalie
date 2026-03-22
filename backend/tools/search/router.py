"""
Router — semantic query routing via example embeddings.

Routes incoming queries to the best provider(s) by:
1. Embedding the query
2. k-NN search against pre-embedded example queries
3. Top-3 mean scoring per provider (avoids dilution)
4. Fixed-threshold gap detection (0.10 from top score)

The embedding cache (providers_cache.sqlite) is auto-generated on first
use from the text-only examples in providers.sqlite. Regenerated when the
embedding model changes.
"""

import hashlib
import logging
import os
import sqlite3
import threading

from services.embedding_utils import pack_embedding
import time

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
_PROVIDERS_DB = os.path.join(_DATA_DIR, 'providers.sqlite')
_CACHE_DB = os.path.join(_DATA_DIR, 'providers_cache.sqlite')

_GAP_THRESHOLD = 0.10   # fixed threshold from top score
_MAX_PROVIDERS = 3       # maximum providers to route to
_TOP_K = 30              # k-NN search depth
_TOP_N_MEAN = 3          # top-N examples per provider for scoring
_MIN_SCORE = 0.50        # minimum score to consider (below = routing miss)

_cache_ready = False
_cache_lock = threading.Lock()


# ── Public API ───────────────────────────────────────────────────────────────

def route_query(query: str) -> list:
    """
    Route a query to the best provider(s).

    Returns list of dicts: [{"name": str, "score": float}, ...]
    sorted by score descending. Empty list on failure.
    """
    _ensure_cache()

    try:
        from services.embedding_service import EmbeddingService
        emb_service = EmbeddingService()
        query_embedding = emb_service.generate_embedding(query)
    except Exception as e:
        logger.warning(f'[SEARCH] embedding failed for routing: {e}')
        return []

    blob = pack_embedding(query_embedding)

    try:
        conn = sqlite3.connect(_CACHE_DB)
        conn.enable_load_extension(True)
        try:
            import sqlite_vec
            sqlite_vec.load(conn)
        except Exception:
            conn.load_extension('vec0')
        conn.row_factory = sqlite3.Row

        # k-NN search against example embeddings
        rows = conn.execute(
            """
            SELECT rowid, distance
            FROM example_embeddings
            WHERE embedding MATCH ?
            AND k = ?
            ORDER BY distance
            """,
            (blob, _TOP_K),
        ).fetchall()
        conn.close()

        if not rows:
            logger.info(f'[SEARCH] no embedding matches for query="{query}"')
            return []

        # Map rowid → provider_id from the providers db
        rowid_to_provider = _get_example_provider_map()

        # Group by provider, collect scores (convert distance → similarity)
        provider_scores: dict[str, list] = {}
        for row in rows:
            rowid = row['rowid']
            distance = row['distance']
            similarity = max(0.0, 1.0 - distance / 2.0)

            provider_name = rowid_to_provider.get(rowid)
            if not provider_name:
                continue

            if provider_name not in provider_scores:
                provider_scores[provider_name] = []
            provider_scores[provider_name].append(similarity)

        if not provider_scores:
            return []

        # Top-3 mean per provider
        ranked = []
        for name, scores in provider_scores.items():
            top_n = sorted(scores, reverse=True)[:_TOP_N_MEAN]
            mean_score = sum(top_n) / len(top_n)
            ranked.append({'name': name, 'score': round(mean_score, 4)})

        ranked.sort(key=lambda x: x['score'], reverse=True)

        # Gap detection
        top_score = ranked[0]['score']

        if top_score < _MIN_SCORE:
            # Routing miss — log for future example tuning
            logger.info(
                f'[SEARCH] routing_miss query="{query}" '
                f'top_score={top_score:.3f} fallback=ddg'
            )
            return []

        threshold = top_score - _GAP_THRESHOLD
        selected = [p for p in ranked if p['score'] >= threshold]
        selected = selected[:_MAX_PROVIDERS]

        names = [p['name'] for p in selected]
        score_parts = ', '.join(f'{p["name"]}: {p["score"]:.3f}' for p in selected)
        logger.info(
            f'[SEARCH] routed query="{query}" to {names} scores={{{score_parts}}}'
        )

        return selected

    except Exception as e:
        logger.warning(f'[SEARCH] routing failed: {e}')
        return []


# ── Example → provider mapping ───────────────────────────────────────────────

_example_map_cache: dict | None = None


def _get_example_provider_map() -> dict:
    """
    Build rowid → provider_name mapping from providers.sqlite.
    Cached in memory after first call.
    """
    global _example_map_cache
    if _example_map_cache is not None:
        return _example_map_cache

    mapping = {}
    try:
        conn = sqlite3.connect(_PROVIDERS_DB)
        rows = conn.execute(
            """
            SELECT pe.id, p.name
            FROM provider_examples pe
            JOIN providers p ON p.id = pe.provider_id
            """
        ).fetchall()
        conn.close()

        for row in rows:
            mapping[row[0]] = row[1]

    except Exception as e:
        logger.warning(f'[SEARCH] failed to load example map: {e}')

    _example_map_cache = mapping
    return mapping


# ── Cache generation ─────────────────────────────────────────────────────────

def _ensure_cache() -> None:
    """Ensure the embedding cache exists and is up-to-date."""
    global _cache_ready

    if _cache_ready:
        return

    with _cache_lock:
        if _cache_ready:
            return

        try:
            current_hash = _get_model_hash()

            if os.path.exists(_CACHE_DB):
                stored_hash = _get_stored_hash()
                if stored_hash == current_hash:
                    _cache_ready = True
                    return
                logger.info('[SEARCH] embedding model changed, regenerating cache')

            _generate_cache(current_hash)
            _cache_ready = True

        except Exception as e:
            logger.error(f'[SEARCH] cache generation failed: {e}')
            raise


def _get_model_hash() -> str:
    """Get a hash identifying the current embedding model."""
    try:
        from services.embedding_service import EmbeddingService
        emb = EmbeddingService()
        # Generate a test embedding to verify model is loaded
        test = emb.generate_embedding("test")
        # Hash based on model name + dimension
        model_id = getattr(emb, 'model_name', 'unknown')
        return hashlib.sha256(f'{model_id}:{len(test)}'.encode()).hexdigest()[:16]
    except Exception:
        return 'unknown'


def _get_stored_hash() -> str | None:
    """Get the stored model hash from cache db."""
    try:
        conn = sqlite3.connect(_CACHE_DB)
        row = conn.execute(
            "SELECT value FROM cache_meta WHERE key = 'embedding_model_hash'"
        ).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def _generate_cache(model_hash: str) -> None:
    """Generate embedding cache from provider examples."""
    from services.embedding_service import EmbeddingService
    emb_service = EmbeddingService()

    # Load all examples
    conn_src = sqlite3.connect(_PROVIDERS_DB)
    examples = conn_src.execute(
        'SELECT id, example_query FROM provider_examples ORDER BY id'
    ).fetchall()
    conn_src.close()

    if not examples:
        logger.warning('[SEARCH] no provider examples found, skipping cache generation')
        return

    logger.info(f'[SEARCH] generating embedding cache for {len(examples)} examples...')
    t0 = time.time()

    # Remove old cache
    if os.path.exists(_CACHE_DB):
        os.remove(_CACHE_DB)

    # Create cache db with sqlite-vec
    conn = sqlite3.connect(_CACHE_DB)
    conn.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
    except Exception:
        # Try loading the extension directly
        conn.load_extension('vec0')

    conn.execute("""
        CREATE TABLE cache_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    # Detect embedding dimension from first embedding
    first_embedding = emb_service.generate_embedding(examples[0][1])
    dim = len(first_embedding)

    conn.execute(f"""
        CREATE VIRTUAL TABLE example_embeddings USING vec0(
            embedding float[{dim}]
        )
    """)

    # Embed all examples in batches
    batch_size = 50
    inserted = 0
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        for example_id, query_text in batch:
            try:
                embedding = emb_service.generate_embedding(query_text)
                blob = pack_embedding(embedding)
                conn.execute(
                    'INSERT INTO example_embeddings (rowid, embedding) VALUES (?, ?)',
                    (example_id, blob),
                )
                inserted += 1
            except Exception as e:
                logger.debug(f'[SEARCH] embedding failed for example {example_id}: {e}')

        conn.commit()
        if (i + batch_size) % 200 == 0:
            logger.info(f'[SEARCH] embedded {min(i + batch_size, len(examples))}/{len(examples)}...')

    # Store model hash
    conn.execute(
        "INSERT INTO cache_meta (key, value) VALUES ('embedding_model_hash', ?)",
        (model_hash,),
    )
    conn.commit()
    conn.close()

    elapsed = time.time() - t0
    logger.info(
        f'[SEARCH] cache generated: {inserted}/{len(examples)} examples '
        f'embedded in {elapsed:.1f}s'
    )
