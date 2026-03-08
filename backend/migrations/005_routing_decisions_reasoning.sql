-- Migration 005: add reasoning column to routing_decisions
-- Stores triage effort annotation e.g. "[effort:trivial]"
--
-- This column is now declared in schema.sql (added during the same commit).
-- On a fresh install ALTER TABLE would fail with "duplicate column name".
-- The column is added idempotently via _optional_columns in
-- database_service.run_pending_migrations() instead (checks PRAGMA table_info
-- before issuing ALTER TABLE — safe for both fresh installs and upgrades).
SELECT 1;
