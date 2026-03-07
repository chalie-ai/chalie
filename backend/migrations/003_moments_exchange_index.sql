-- Migration 003: Add missing index on moments.exchange_id.
--
-- The moments table was missing an index on exchange_id, causing full table
-- scans when looking up moments by their source exchange. Other tables with
-- exchange_id (cortex_iterations, interaction_log) already had this index.

CREATE INDEX IF NOT EXISTS idx_moments_exchange
    ON moments(exchange_id) WHERE exchange_id IS NOT NULL;
