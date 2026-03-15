-- 009: Strip dead anti-nag columns from the uncertainties table.
--
-- mark_surfaced() and downgrade_overexposed() were fully implemented but never
-- wired into the response pipeline. The surfaced state, surfaced_count, and
-- last_surfaced_at columns are dead code. Uncertainty states simplify to
-- open | resolved.
--
-- SQLite < 3.35 doesn't support DROP COLUMN, so we rebuild the table.

-- Migrate any stale 'surfaced' rows to 'open' before dropping the state value.
UPDATE uncertainties SET state = 'open' WHERE state = 'surfaced';

-- Rebuild uncertainties without surfaced_count and last_surfaced_at.
CREATE TABLE IF NOT EXISTS uncertainties_new (
    id TEXT PRIMARY KEY,
    memory_a_type TEXT NOT NULL,
    memory_a_id TEXT NOT NULL,
    memory_b_type TEXT,
    memory_b_id TEXT,
    uncertainty_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    detection_context TEXT NOT NULL,
    reasoning TEXT,
    temporal_signal INTEGER DEFAULT 0,
    surface_context TEXT,
    state TEXT NOT NULL DEFAULT 'open',
    resolution_strategy TEXT,
    resolution_detail TEXT,
    resolved_at TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

INSERT INTO uncertainties_new
    (id, memory_a_type, memory_a_id, memory_b_type, memory_b_id,
     uncertainty_type, severity, detection_context, reasoning,
     temporal_signal, surface_context, state, resolution_strategy,
     resolution_detail, resolved_at, created_at)
SELECT
    id, memory_a_type, memory_a_id, memory_b_type, memory_b_id,
    uncertainty_type, severity, detection_context, reasoning,
    temporal_signal, surface_context, state, resolution_strategy,
    resolution_detail, resolved_at, created_at
FROM uncertainties;

DROP TABLE IF EXISTS uncertainties;
ALTER TABLE uncertainties_new RENAME TO uncertainties;

CREATE INDEX IF NOT EXISTS idx_uncertainties_state ON uncertainties(state);
CREATE INDEX IF NOT EXISTS idx_uncertainties_memory_a ON uncertainties(memory_a_type, memory_a_id);
CREATE INDEX IF NOT EXISTS idx_uncertainties_memory_b ON uncertainties(memory_b_type, memory_b_id);
CREATE INDEX IF NOT EXISTS idx_uncertainties_severity ON uncertainties(severity, state);
