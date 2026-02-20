-- Migration 010: Upgrade topic embeddings from 384 to 768 dimensions
-- Full embeddinggemma precision for better discriminative power
-- Drops existing topics (dimension mismatch requires re-embedding)

DROP INDEX IF EXISTS idx_topics_embedding;
TRUNCATE TABLE topics;

ALTER TABLE topics
    ALTER COLUMN rolling_embedding TYPE vector(768);

-- Recreate index for new dimensions
CREATE INDEX IF NOT EXISTS idx_topics_embedding ON topics
USING ivfflat (rolling_embedding vector_cosine_ops)
WITH (lists = 100);
