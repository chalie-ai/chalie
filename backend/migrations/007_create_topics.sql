-- Migration 007: Create topics table for deterministic topic classification
-- Topics are semantic attractors in embedding space

CREATE TABLE IF NOT EXISTS topics (
    topic_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    message_count INT NOT NULL DEFAULT 0,

    -- Semantic attractor: running average of message embeddings (L2-normalized)
    rolling_embedding vector(768) NOT NULL,

    -- Salience: running average of message salience scores
    avg_salience FLOAT NOT NULL DEFAULT 0.5,

    -- Metadata for future extensibility
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Index for vector similarity search (cosine distance)
CREATE INDEX IF NOT EXISTS idx_topics_embedding ON topics
USING ivfflat (rolling_embedding vector_cosine_ops)
WITH (lists = 100);

-- Index for name lookup
CREATE INDEX IF NOT EXISTS idx_topics_name ON topics(name);

-- Index for temporal queries
CREATE INDEX IF NOT EXISTS idx_topics_last_updated ON topics(last_updated DESC);
