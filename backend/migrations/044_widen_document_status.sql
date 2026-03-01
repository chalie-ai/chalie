-- Migration 044: Widen documents.status from VARCHAR(20) to VARCHAR(30)
-- 'awaiting_confirmation' is 21 characters, exceeds previous limit.

ALTER TABLE documents ALTER COLUMN status TYPE VARCHAR(30);
