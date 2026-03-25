-- Unified knowledge store — merges user_traits, procedural_memory,
-- semantic_concepts, and semantic_relationships into a single table
-- with kind-based discrimination.  Idempotent (IF NOT EXISTS + INSERT OR IGNORE).

-- ────────────────────────────────────────────────────────────────
-- 1. Create the table + indexes + FTS
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS knowledge (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    entity      TEXT NOT NULL DEFAULT 'user',
    key         TEXT NOT NULL,
    value       TEXT,
    data        TEXT,
    decay_class TEXT NOT NULL DEFAULT 'standard',
    confidence  REAL NOT NULL DEFAULT 0.5,
    reliability TEXT NOT NULL DEFAULT 'reliable',
    source      TEXT,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    last_accessed_at TEXT,
    deleted_at  TEXT,
    UNIQUE(entity, key)
);

CREATE INDEX IF NOT EXISTS idx_knowledge_kind ON knowledge(kind);
CREATE INDEX IF NOT EXISTS idx_knowledge_entity ON knowledge(entity);
CREATE INDEX IF NOT EXISTS idx_knowledge_key ON knowledge(key);
CREATE INDEX IF NOT EXISTS idx_knowledge_confidence ON knowledge(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_decay_class ON knowledge(decay_class);
CREATE INDEX IF NOT EXISTS idx_knowledge_deleted ON knowledge(deleted_at) WHERE deleted_at IS NULL;

CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    key, value, kind, entity, content='knowledge', content_rowid='rowid'
);

-- ────────────────────────────────────────────────────────────────
-- 2. Migrate user_traits → knowledge (kind=trait)
-- ────────────────────────────────────────────────────────────────
INSERT OR IGNORE INTO knowledge (kind, entity, key, value, data, decay_class, confidence, reliability, source, evidence_count, created_at, updated_at)
SELECT
    'trait', 'user', trait_key, trait_value,
    json_object('category', category, 'last_conflict_at', last_conflict_at),
    CASE category
        WHEN 'core' THEN 'permanent'
        WHEN 'behavioral' THEN 'slow'
        ELSE 'standard'
    END,
    confidence, COALESCE(reliability, 'reliable'), 'migrated:user_traits',
    reinforcement_count, created_at, updated_at
FROM user_traits;

-- ────────────────────────────────────────────────────────────────
-- 3. Migrate procedural_memory → knowledge (kind=procedure)
-- ────────────────────────────────────────────────────────────────
INSERT OR IGNORE INTO knowledge (kind, entity, key, value, data, decay_class, confidence, source, evidence_count, created_at, updated_at)
SELECT
    'procedure', 'chalie', action_name,
    printf('%.0f%% success over %d attempts', success_rate * 100, total_attempts),
    json_object('success_rate', success_rate, 'weight', weight, 'avg_reward', avg_reward,
                'reward_history', json(reward_history), 'context_stats', json(context_stats)),
    'slow',
    MIN(1.0, success_rate + 0.1),
    'migrated:procedural_memory', total_attempts, created_at, updated_at
FROM procedural_memory;

-- ────────────────────────────────────────────────────────────────
-- 4. Migrate semantic_concepts → knowledge (kind=concept)
-- ────────────────────────────────────────────────────────────────
INSERT OR IGNORE INTO knowledge (kind, entity, key, value, data, decay_class, confidence, reliability, source, evidence_count, created_at, updated_at, last_accessed_at)
SELECT
    'concept', 'chalie', concept_name, definition,
    json_object('domain', domain, 'concept_type', concept_type, 'strength', strength,
                'decay_resistance', decay_resistance, 'abstraction_level', abstraction_level,
                'source_episodes', json(source_episodes), 'examples', json(examples),
                'verification_status', verification_status, 'activation_score', activation_score,
                'utility_score', utility_score, 'context_constraints', json(context_constraints)),
    CASE
        WHEN COALESCE(decay_resistance, 0.5) > 0.8 THEN 'slow'
        WHEN COALESCE(decay_resistance, 0.5) < 0.3 THEN 'fast'
        ELSE 'standard'
    END,
    confidence, COALESCE(reliability, 'reliable'), 'migrated:semantic_concepts',
    CASE WHEN consolidation_count > 0 THEN consolidation_count ELSE 1 END,
    created_at, updated_at, last_accessed_at
FROM semantic_concepts WHERE deleted_at IS NULL;

-- ────────────────────────────────────────────────────────────────
-- 5. Migrate semantic_relationships → knowledge (kind=relationship)
-- ────────────────────────────────────────────────────────────────
INSERT OR IGNORE INTO knowledge (kind, entity, key, value, data, decay_class, confidence, source, evidence_count, created_at, updated_at)
SELECT
    'relationship', 'chalie',
    sc1.concept_name || ':' || sr.relationship_type || ':' || sc2.concept_name,
    sr.relationship_type,
    json_object('source_key', sc1.concept_name, 'target_key', sc2.concept_name,
                'relationship_type', sr.relationship_type, 'bidirectional', sr.bidirectional,
                'strength', sr.strength, 'source_episodes', json(sr.source_episodes)),
    'standard',
    sr.confidence, 'migrated:semantic_relationships', 1,
    sr.created_at, sr.updated_at
FROM semantic_relationships sr
JOIN semantic_concepts sc1 ON sr.source_concept_id = sc1.id
JOIN semantic_concepts sc2 ON sr.target_concept_id = sc2.id
WHERE sr.deleted_at IS NULL AND sc1.deleted_at IS NULL AND sc2.deleted_at IS NULL;

-- ────────────────────────────────────────────────────────────────
-- 6. Rebuild FTS index
-- ────────────────────────────────────────────────────────────────
INSERT INTO knowledge_fts(knowledge_fts) VALUES('rebuild');
