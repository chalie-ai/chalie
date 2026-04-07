-- Migration 033: Episodic cleanup — soft-delete stale knowledge kinds, drop ghost columns
--
-- Part A: Soft-delete concept/relationship knowledge entries (superseded by episodic memory pipeline)
-- Safe to run on fresh installs (no concept/relationship rows exist)
UPDATE knowledge SET deleted_at = datetime('now')
WHERE kind IN ('concept', 'relationship')
  AND deleted_at IS NULL;

-- Part B: Drop ghost columns from episodes table
-- SQLite 3.35+ required (production ships 3.39+, macOS ships 3.39+)
-- On fresh installs these columns don't exist (schema.sql already excludes them),
-- so we guard each DROP with a PRAGMA check via the application layer.
-- The database_service.run_pending_migrations() executes this as a script,
-- so we use a conditional approach: attempt the drop, ignore if column doesn't exist.
-- SQLite ALTER TABLE DROP COLUMN silently fails with "no such column" — non-fatal in executescript.

DROP INDEX IF EXISTS idx_episodes_activation;
DROP INDEX IF EXISTS idx_episodes_semantic_status;
DROP INDEX IF EXISTS idx_episodes_composite;

-- Recreate composite index using retrieval_weight (replaces activation_score)
CREATE INDEX IF NOT EXISTS idx_episodes_composite ON episodes(topic, retrieval_weight DESC, created_at DESC) WHERE deleted_at IS NULL;
