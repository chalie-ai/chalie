-- Migration 029: Add keywords column to tool_capability_profiles
--
-- On a fresh install:
--   - schema.sql already includes the keywords column
--   - This migration is a no-op (SELECT 1 below)
--
-- On an existing install (upgrade path):
--   - Column addition is handled idempotently via _optional_columns in
--     database_service.py (SQLite lacks ALTER TABLE ... ADD COLUMN IF NOT EXISTS)
SELECT 1;
