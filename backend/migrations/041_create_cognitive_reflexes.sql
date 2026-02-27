-- Cognitive Reflexes: learned fast-path clusters via semantic abstraction.
-- Each row represents a reflex cluster — a rolling-average centroid of similar
-- queries that didn't benefit from the full cognitive pipeline.

CREATE TABLE IF NOT EXISTS cognitive_reflexes (
    id              SERIAL PRIMARY KEY,
    embedding       vector(768) NOT NULL,       -- rolling average centroid (L2-normalized)
    sample_queries  TEXT[] DEFAULT '{}',         -- last 5 queries for observability
    times_seen      INT DEFAULT 1,              -- total queries merged into this cluster
    times_unnecessary INT DEFAULT 0,            -- times full pipeline added no value
    times_activated INT DEFAULT 0,              -- times fast path was used
    times_succeeded INT DEFAULT 0,              -- fast path not corrected
    times_failed    INT DEFAULT 0,              -- fast path corrected by user
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_seen       TIMESTAMPTZ DEFAULT NOW(),
    last_activated  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_cognitive_reflexes_embedding
    ON cognitive_reflexes USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
