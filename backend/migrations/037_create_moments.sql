-- Moments — pinned message bookmarks with LLM-enriched context.

CREATE TABLE IF NOT EXISTS moments (
    id                  TEXT        PRIMARY KEY,
    user_id             TEXT        NOT NULL DEFAULT 'primary',
    title               TEXT,
    message_text        TEXT        NOT NULL,
    exchange_id         TEXT,
    topic               TEXT,
    thread_id           TEXT,
    gists               JSONB       NOT NULL DEFAULT '[]'::jsonb,
    summary             TEXT,
    embedding           vector(768),
    status              TEXT        NOT NULL DEFAULT 'enriching'
                        CHECK (status IN ('enriching', 'sealed', 'forgotten')),
    pinned_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sealed_at           TIMESTAMPTZ,
    last_enriched_at    TIMESTAMPTZ,
    metadata            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at          TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_moments_user_active
    ON moments (user_id, pinned_at DESC) WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_moments_enriching
    ON moments (status, pinned_at) WHERE status = 'enriching' AND deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_moments_embedding
    ON moments USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);

CREATE INDEX IF NOT EXISTS idx_moments_topic
    ON moments (topic, pinned_at DESC) WHERE deleted_at IS NULL;
