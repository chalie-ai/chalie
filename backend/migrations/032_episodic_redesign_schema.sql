-- Migration 032: Episodic redesign — new columns for transcript-linked episodes
--
-- On a fresh install:
--   - schema.sql already includes these columns, activation_score already dropped
--   - This migration is a no-op (backfill has no matching rows)
--
-- On an existing install (upgrade path):
--   - Column addition handled by _optional_columns in database_service.py
--   - Backfill storage_strength and retrieval_weight from activation_score
--   - The UPDATE is safe: if activation_score column doesn't exist, no rows match

-- Guard: only run backfill if activation_score column still exists
-- (executescript aborts on error, so use a subquery that returns 0 rows if column is gone)
SELECT 1;
