-- Migration 036: Replace goals with curiosity_threads
-- Curiosity threads are self-directed explorations seeded from cognitive drift.

DROP TABLE IF EXISTS goals;

CREATE TABLE IF NOT EXISTS curiosity_threads (
    id                  TEXT PRIMARY KEY,
    title               TEXT NOT NULL,
    rationale           TEXT,
    thread_type         TEXT NOT NULL CHECK (thread_type IN ('learning', 'behavioral')),
    status              TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'dormant', 'abandoned')),
    seed_topic          TEXT,
    learning_notes      JSONB NOT NULL DEFAULT '[]',
    last_explored_at    TIMESTAMPTZ,
    exploration_count   INTEGER NOT NULL DEFAULT 0,
    last_surfaced_at    TIMESTAMPTZ,
    engagement_score    FLOAT NOT NULL DEFAULT 0.5,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_curiosity_threads_status ON curiosity_threads(status);
CREATE INDEX IF NOT EXISTS idx_curiosity_threads_explore
    ON curiosity_threads(status, last_explored_at NULLS FIRST)
    WHERE status = 'active';
