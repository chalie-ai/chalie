-- 026: Add pending_contradictions table; drop reliability column from knowledge.
--
-- pending_contradictions holds trait pairs awaiting user resolution, replacing
-- the heavier uncertainties model.
--
-- ALTER TABLE DROP COLUMN requires SQLite 3.35+. If this migration fails on an
-- older SQLite, remove the final statement and backfill manually.

CREATE TABLE IF NOT EXISTS pending_contradictions (
    id TEXT PRIMARY KEY,
    trait_a_id INTEGER NOT NULL,
    trait_b_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    surfaced_at TEXT NOT NULL,
    source TEXT NOT NULL  -- 'chat' | 'ambient'
);

-- Drop reliability column from knowledge (no longer used by application code).
-- schema.sql retains it only for migration 018 compat on fresh DBs; this
-- statement removes it for all DBs (fresh and existing).
ALTER TABLE knowledge DROP COLUMN reliability;
