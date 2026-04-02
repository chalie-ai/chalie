-- 025: Drop uncertainties table and its indexes.
--
-- The uncertainties table (epistemic confidence tracking) is replaced by the
-- pending_contradictions table introduced in migration 026.

DROP TABLE IF EXISTS uncertainties;
DROP INDEX IF EXISTS idx_uncertainties_state;
DROP INDEX IF EXISTS idx_uncertainties_memory_a;
DROP INDEX IF EXISTS idx_uncertainties_memory_b;
DROP INDEX IF EXISTS idx_uncertainties_severity;
