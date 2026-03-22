-- Migration 006: Add goals_vec virtual table and is_muted column on goals
--
-- goals_vec enables KNN embedding search over goals (replaces O(n) linear scan).
-- is_muted is a fast SQL-filterable column so get_actionable_goals() can exclude
-- muted goals without deserializing outcome_feedback JSON for every row.
--
-- On a fresh install:
--   - goals table created by schema.sql with is_muted already present
--   - goals_vec created by schema.sql and schema_service._create_vec_tables()
--   Both CREATE statements below are idempotent (IF NOT EXISTS).
--
-- On an existing install (upgrade path):
--   - goals_vec may not exist → CREATE VIRTUAL TABLE adds it
--   - is_muted column may not exist → handled via _optional_columns in
--     database_service.run_pending_migrations() (idempotent ALTER TABLE)

CREATE VIRTUAL TABLE IF NOT EXISTS goals_vec USING vec0(embedding float[768]);

-- Note: is_muted column addition is handled idempotently via _optional_columns
-- in database_service.py rather than here, because SQLite lacks
-- "ALTER TABLE … ADD COLUMN IF NOT EXISTS".
SELECT 1;
