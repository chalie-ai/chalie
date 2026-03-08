-- Migration 003: Add missing index on moments.exchange_id.
--
-- Originally added idx_moments_exchange to the moments table, but moments
-- was dropped entirely by migration 004 and removed from schema.sql.
-- On a fresh install the moments table never exists, so the original
-- CREATE INDEX statement would crash.
--
-- This migration is now a no-op. The index (and the table) are cleaned up
-- by migration 004 using DROP INDEX/TABLE IF EXISTS.
SELECT 1;
