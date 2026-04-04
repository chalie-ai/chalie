-- 009: Strip dead anti-nag columns from the uncertainties table.
--
-- Originally rebuilt the uncertainties table without surfaced_count and
-- last_surfaced_at columns. Migration 025 drops the entire uncertainties
-- table, making this rebuild redundant. On fresh installs the table was
-- always empty, so the rebuild was a no-op anyway.
--
-- Now a permanent no-op. Kept for schema_migrations compatibility.
SELECT 1;
