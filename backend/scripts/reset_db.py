"""
Dev utility — clears all cognitive data tables in the SQLite database.

Usage:
    cd backend && python scripts/reset_db.py
"""

import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.database_service import get_shared_db_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Tables that hold user/cognitive data (not schema or config)
DATA_TABLES = [
    'episodes',
    'cortex_iterations',
    'semantic_concepts',
    'semantic_relationships',
    'interaction_log',
    'routing_decisions',
    'user_traits',
    'procedural_memory',
    'topics',
    'threads',
    'scheduled_items',
    'autobiography',
    'curiosity_threads',
    'lists',
    'list_items',
    'list_events',
    'documents',
    'document_chunks',
    'persistent_tasks',
    'place_fingerprints',
    'moments',
    'tool_performance_metrics',
]


def clear_database():
    """Delete all rows from cognitive data tables."""
    db = get_shared_db_service()

    with db.connection() as conn:
        cursor = conn.cursor()
        cleared = 0
        for table in DATA_TABLES:
            try:
                cursor.execute(f"DELETE FROM {table}")
                count = cursor.rowcount
                if count > 0:
                    logger.info(f"  Cleared {count} rows from {table}")
                    cleared += 1
            except Exception as e:
                logger.warning(f"  Skipped {table}: {e}")
        conn.commit()
        cursor.close()

    logger.info(f"Done — cleared {cleared}/{len(DATA_TABLES)} tables")


if __name__ == "__main__":
    confirm = input("This will DELETE all cognitive data. Type 'yes' to confirm: ")
    if confirm.strip().lower() == 'yes':
        clear_database()
    else:
        print("Aborted.")
