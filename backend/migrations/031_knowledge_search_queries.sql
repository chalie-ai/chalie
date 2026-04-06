-- Migration 031: Add search_queries column to knowledge table.
--
-- Stores doc2query-generated search queries as a JSON array string.
-- Used by knowledge_fts for improved semantic retrieval coverage.
--
-- Column is added idempotently via _optional_columns in database_service.py
-- so that fresh databases (which already have it from schema.sql) do not fail.
-- This migration is intentionally a no-op to preserve migration chain ordering.
SELECT 1;
