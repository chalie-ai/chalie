-- Migration 004: Create semantic memory tables
-- Date: 2026-02-09
-- Description: Implements semantic memory (concepts, relationships, schemas)

-- Enable vector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Table 1: semantic_concepts
CREATE TABLE IF NOT EXISTS semantic_concepts (
    -- Identity
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Core concept data
    concept_name TEXT NOT NULL,
    concept_type TEXT NOT NULL,
    definition TEXT NOT NULL,

    -- Semantic embedding (256-dim embeddinggemma)
    embedding vector(256),

    -- Abstraction metadata
    abstraction_level INTEGER DEFAULT 3,
    domain TEXT,

    -- Strength & activation (human-like memory dynamics)
    strength FLOAT DEFAULT 1.0,
    activation_score FLOAT DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    consolidation_count INTEGER DEFAULT 0,

    -- Confidence & verification
    confidence FLOAT DEFAULT 0.5,
    source_episodes JSONB DEFAULT '[]'::jsonb,
    verification_status TEXT DEFAULT 'unverified',

    -- Context & examples
    context_constraints JSONB DEFAULT '{}'::jsonb,
    examples JSONB DEFAULT '[]'::jsonb,

    -- Temporal tracking
    first_learned_at TIMESTAMP DEFAULT NOW(),
    last_accessed_at TIMESTAMP DEFAULT NOW(),
    last_reinforced_at TIMESTAMP DEFAULT NOW(),

    -- Decay resistance (mimics human salience for concepts)
    utility_score FLOAT DEFAULT 0.5,
    decay_resistance FLOAT DEFAULT 0.5,

    -- Soft delete
    deleted_at TIMESTAMP,

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for semantic_concepts
CREATE INDEX IF NOT EXISTS idx_concepts_embedding ON semantic_concepts USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_concepts_name ON semantic_concepts (concept_name);
CREATE INDEX IF NOT EXISTS idx_concepts_type ON semantic_concepts (concept_type);
CREATE INDEX IF NOT EXISTS idx_concepts_domain ON semantic_concepts (domain);
CREATE INDEX IF NOT EXISTS idx_concepts_strength ON semantic_concepts (strength DESC);
CREATE INDEX IF NOT EXISTS idx_concepts_activation ON semantic_concepts (activation_score DESC);
CREATE INDEX IF NOT EXISTS idx_concepts_deleted ON semantic_concepts (deleted_at) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_concepts_source_episodes ON semantic_concepts USING gin (source_episodes);

COMMENT ON TABLE semantic_concepts IS 'Semantic memory: general knowledge concepts extracted from episodes';
COMMENT ON COLUMN semantic_concepts.strength IS 'Concept strength (1.0-10.0): strengthened by repeated exposure';
COMMENT ON COLUMN semantic_concepts.decay_resistance IS 'Decay resistance (0.5-1.0): based on consolidation_count';
COMMENT ON COLUMN semantic_concepts.utility_score IS 'Utility score (0-1): frequency + recency + relationships + diversity';
COMMENT ON COLUMN semantic_concepts.embedding IS '256-dim embeddinggemma vector for fuzzy semantic search';

-- Table 2: semantic_relationships
CREATE TABLE IF NOT EXISTS semantic_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Relationship structure
    source_concept_id UUID NOT NULL REFERENCES semantic_concepts(id),
    target_concept_id UUID NOT NULL REFERENCES semantic_concepts(id),
    relationship_type TEXT NOT NULL,

    -- Relationship strength (spreading activation weight)
    strength FLOAT DEFAULT 0.5,
    bidirectional BOOLEAN DEFAULT false,

    -- Evidence
    source_episodes JSONB DEFAULT '[]'::jsonb,
    confidence FLOAT DEFAULT 0.5,

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    deleted_at TIMESTAMP,

    -- Prevent duplicate relationships
    UNIQUE(source_concept_id, target_concept_id, relationship_type)
);

-- Indexes for semantic_relationships
CREATE INDEX IF NOT EXISTS idx_relationships_source ON semantic_relationships (source_concept_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON semantic_relationships (target_concept_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON semantic_relationships (relationship_type);
CREATE INDEX IF NOT EXISTS idx_relationships_strength ON semantic_relationships (strength DESC);

COMMENT ON TABLE semantic_relationships IS 'Relationships between concepts for spreading activation';
COMMENT ON COLUMN semantic_relationships.strength IS 'Relationship strength (0-1): weight for spreading activation';
COMMENT ON COLUMN semantic_relationships.bidirectional IS 'Can activation spread both ways?';

-- Table 3: semantic_schemas
CREATE TABLE IF NOT EXISTS semantic_schemas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Schema identity
    schema_name TEXT NOT NULL UNIQUE,
    description TEXT,

    -- Schema structure
    core_concepts JSONB NOT NULL,
    relationships JSONB DEFAULT '[]'::jsonb,

    -- Schema metadata
    activation_count INTEGER DEFAULT 0,
    last_activated_at TIMESTAMP,

    -- Source tracking
    learned_from_episodes JSONB DEFAULT '[]'::jsonb,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for semantic_schemas
CREATE INDEX IF NOT EXISTS idx_schemas_name ON semantic_schemas (schema_name);
CREATE INDEX IF NOT EXISTS idx_schemas_activation ON semantic_schemas (activation_count DESC);

COMMENT ON TABLE semantic_schemas IS 'Mental frameworks: patterns of co-occurring concepts';
COMMENT ON COLUMN semantic_schemas.core_concepts IS 'JSONB array of concept IDs that form this schema';
