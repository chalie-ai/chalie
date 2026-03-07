-- Migration 004: Drop the unused moments table and its companion vec table.
--
-- Moments were refactored to be stored as documents with source_type='moment'
-- in the documents table. The dedicated moments table was never written to
-- after this change and has been empty since. The moments_vec virtual table
-- was its sqlite-vec companion and is also unused.
--
-- All moment data (create, list, search, forget) flows through MomentService
-- which reads and writes the documents table exclusively.

DROP INDEX IF EXISTS idx_moments_exchange;
DROP INDEX IF EXISTS idx_moments_topic;
DROP INDEX IF EXISTS idx_moments_enriching;
DROP INDEX IF EXISTS idx_moments_active;
DROP TABLE IF EXISTS moments;

-- Drop the unused sqlite-vec companion table.
DROP TABLE IF EXISTS moments_vec;
