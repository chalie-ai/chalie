-- Migration 027: Create goals table for persistent directional goals
CREATE TABLE IF NOT EXISTS goals (
    id              TEXT        PRIMARY KEY,
    user_id         TEXT        NOT NULL DEFAULT 'primary',
    title           TEXT        NOT NULL CHECK (char_length(title) <= 200),
    description     TEXT,
    status          TEXT        NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active','progressing','achieved','abandoned','dormant')),
    priority        INTEGER     DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    source          TEXT        DEFAULT 'inferred'
                    CHECK (source IN ('explicit','inferred','autobiography')),
    progress_notes  JSONB       DEFAULT '[]',
    related_topics  TEXT[]      DEFAULT '{}',
    last_mentioned  TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_goals_user_status ON goals(user_id, status);
CREATE INDEX IF NOT EXISTS idx_goals_active_priority ON goals(user_id, priority DESC) WHERE status IN ('active','progressing');
