CREATE TABLE IF NOT EXISTS autobiography (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT        NOT NULL DEFAULT 'primary',
    version         INTEGER     NOT NULL DEFAULT 1,
    narrative       TEXT        NOT NULL,
    section_hashes  JSONB       NOT NULL DEFAULT '{}',
    episode_cursor  TIMESTAMPTZ,
    episodes_since  INTEGER     NOT NULL DEFAULT 0,
    synthesis_model TEXT,
    synthesis_ms    INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, version)
);
CREATE INDEX IF NOT EXISTS idx_autobiography_user_version
    ON autobiography(user_id, version DESC);
