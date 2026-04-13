-- Chalie: Consolidated SQLite Schema
-- Replaces 44 PostgreSQL migration files with a single SQLite-dialect schema.
-- Uses: sqlite-vec for vector search, FTS5 for full-text search.
-- Conventions:
--   - JSONB → TEXT (store JSON as text)
--   - TEXT[] → TEXT (store as JSON array)
--   - SERIAL → INTEGER PRIMARY KEY AUTOINCREMENT
--   - TIMESTAMPTZ → TEXT (ISO-8601 strings)
--   - gen_random_uuid() → application-side uuid4()
--   - vector(768) → companion _vec virtual tables via sqlite-vec

-- ────────────────────────────────────────────────────────────────
-- EPISODES — narrative memory with decay
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    intent TEXT NOT NULL,                     -- JSONB stored as TEXT
    context TEXT NOT NULL,                    -- JSONB stored as TEXT
    action TEXT NOT NULL,
    emotion TEXT NOT NULL,                    -- JSONB stored as TEXT
    outcome TEXT NOT NULL,
    gist TEXT NOT NULL,
    salience INTEGER NOT NULL CHECK (salience BETWEEN 1 AND 10),
    channel TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    last_accessed_at TEXT,
    access_count INTEGER DEFAULT 0,
    deleted_at TEXT,
    salience_factors TEXT DEFAULT '{}',       -- JSONB
    open_loops TEXT DEFAULT '[]',             -- JSONB
    transcript_ids TEXT DEFAULT '[]',         -- JSONB: list of transcript.id values this episode covers
    transcript_id_start INTEGER,              -- lowest transcript.id in this episode's range
    transcript_id_end INTEGER,                -- highest transcript.id in this episode's range
    entities TEXT DEFAULT '[]',               -- JSONB: people, places, orgs, products mentioned
    goal_tags TEXT DEFAULT '[]',              -- JSONB: active goal tags detected
    emotional_valence REAL,                   -- -1.0 (negative) to 1.0 (positive)
    emotional_arousal REAL,                   -- 0.0 (calm) to 1.0 (intense) — drives consolidation strength
    consolidated_from TEXT DEFAULT '[]',      -- JSONB: episode IDs this was consolidated from
    storage_strength REAL DEFAULT 1.0,        -- encoding strength at storage time
    retrieval_weight REAL DEFAULT 1.0         -- current retrieval priority weight
);

CREATE INDEX IF NOT EXISTS idx_episodes_channel ON episodes(channel) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_episodes_composite ON episodes(channel, retrieval_weight DESC, created_at DESC) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_episodes_intent_type ON episodes(json_extract(intent, '$.type'));
CREATE INDEX IF NOT EXISTS idx_episodes_transcript_range ON episodes(transcript_id_start, transcript_id_end);
CREATE INDEX IF NOT EXISTS idx_episodes_retrieval_weight ON episodes(retrieval_weight DESC);

-- FTS5 for full-text search on episodes (replaces GIN tsvector indexes)
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    gist, action, content='episodes', content_rowid='rowid'
);

-- ────────────────────────────────────────────────────────────────
-- CORTEX ITERATIONS — ACT loop audit trail
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cortex_iterations (
    id TEXT PRIMARY KEY,
    channel TEXT NOT NULL,
    exchange_id TEXT,
    session_id TEXT,
    loop_id TEXT NOT NULL,
    iteration_number INTEGER NOT NULL,
    started_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT,
    execution_time_ms REAL,
    chosen_mode TEXT,
    chosen_confidence REAL,
    alternative_paths TEXT,                   -- JSONB
    iteration_cost REAL,
    diminishing_cost REAL,
    uncertainty_cost REAL,
    action_base_cost REAL,
    total_cost REAL,
    cumulative_cost REAL,
    efficiency_score REAL,
    expected_confidence_gain REAL,
    task_value REAL,
    future_leverage REAL,
    effort_estimate TEXT,
    effort_multiplier REAL,
    iteration_penalty REAL,
    exploration_bonus REAL,
    net_value REAL,
    decision_override INTEGER,               -- BOOLEAN
    overridden_mode TEXT,
    termination_reason TEXT,
    actions_executed TEXT,                    -- JSONB
    action_count INTEGER,
    action_success_count INTEGER,
    frontal_cortex_response TEXT,             -- JSONB
    config_snapshot TEXT,                     -- JSONB
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cortex_iterations_loop ON cortex_iterations(loop_id, iteration_number);
CREATE INDEX IF NOT EXISTS idx_cortex_iterations_channel ON cortex_iterations(channel, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cortex_iterations_exchange ON cortex_iterations(exchange_id);

-- semantic_concepts, semantic_relationships removed — replaced by unified knowledge table.
-- semantic_schemas table removed — never used by any service.

-- ────────────────────────────────────────────────────────────────
-- INTERACTION LOG — append-only audit trail
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS interaction_log (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    channel TEXT,
    exchange_id TEXT,
    session_id TEXT,
    source TEXT,
    payload TEXT NOT NULL DEFAULT '{}',       -- JSONB
    metadata TEXT DEFAULT '{}',              -- JSONB
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_interaction_log_channel_created ON interaction_log(channel, created_at);
CREATE INDEX IF NOT EXISTS idx_interaction_log_event_type_created ON interaction_log(event_type, created_at);
CREATE INDEX IF NOT EXISTS idx_interaction_log_session_created ON interaction_log(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_interaction_log_exchange ON interaction_log(exchange_id);

-- procedural_memory removed — replaced by unified knowledge table.

-- ────────────────────────────────────────────────────────────────
-- KNOWLEDGE — unified knowledge store (traits, procedures, concepts, relationships)
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
    reliability TEXT NOT NULL DEFAULT 'reliable',  -- dropped by migration 026; kept here for migration compat
    source      TEXT,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    last_accessed_at TEXT,
    deleted_at  TEXT,
    search_queries TEXT DEFAULT NULL,
    UNIQUE(entity, key)
);

CREATE INDEX IF NOT EXISTS idx_knowledge_kind ON knowledge(kind);
CREATE INDEX IF NOT EXISTS idx_knowledge_entity ON knowledge(entity);
CREATE INDEX IF NOT EXISTS idx_knowledge_key ON knowledge(key);
CREATE INDEX IF NOT EXISTS idx_knowledge_confidence ON knowledge(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_decay_class ON knowledge(decay_class);
CREATE INDEX IF NOT EXISTS idx_knowledge_deleted ON knowledge(deleted_at) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_knowledge_kind_entity_active ON knowledge(kind, entity) WHERE deleted_at IS NULL;

CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    key, value, kind, entity, search_queries, content='knowledge', content_rowid='rowid',
    tokenize='porter unicode61'
);

-- topics table removed — dropped by migration 035_channel_migration.sql

-- identity_vectors + identity_events removed — personality dimensions ripped out.

-- threads table removed — dropped by migration 035_channel_migration.sql
-- thread_exchanges table removed — dropped by migration 035_channel_migration.sql

-- ────────────────────────────────────────────────────────────────
-- AUTH SESSIONS — durable session storage (survives restarts)
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS auth_sessions (
    token TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires ON auth_sessions(expires_at);

-- ────────────────────────────────────────────────────────────────
-- VAULT CONFIG — envelope encryption (AES-256-GCM)
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS vault_config (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    kdf_salt        BLOB    NOT NULL,
    kdf_algorithm   TEXT    NOT NULL DEFAULT 'pbkdf2_sha256',
    kdf_iterations  INTEGER NOT NULL DEFAULT 600000,
    wrapped_dek     BLOB    NOT NULL,
    dek_nonce       BLOB    NOT NULL,
    created_at      TEXT,
    updated_at      TEXT
);

-- ────────────────────────────────────────────────────────────────
-- INTERFACES — external interface registry (bluetooth-style pairing)
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS interfaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    host TEXT NOT NULL,
    port INTEGER NOT NULL,
    signal_token_hash TEXT,
    status TEXT NOT NULL DEFAULT 'offline',
    capabilities_hash TEXT,
    last_seen_at TEXT,
    paired_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS interface_tools (
    interface_id TEXT NOT NULL REFERENCES interfaces(id) ON DELETE CASCADE,
    tool_name TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    registered_at TEXT NOT NULL,
    PRIMARY KEY (interface_id, tool_name)
);

CREATE TABLE IF NOT EXISTS interface_pairing_keys (
    key_hash TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used_at TEXT
);

-- ────────────────────────────────────────────────────────────────
-- TOOL CONFIGS — per-tool key-value configuration
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tool_configs (
    id TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    config_key TEXT NOT NULL,
    config_value TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(tool_name, config_key)
);

CREATE INDEX IF NOT EXISTS idx_tool_configs_tool ON tool_configs(tool_name);

-- ────────────────────────────────────────────────────────────────
-- PROVIDERS — LLM provider configuration
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS providers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    platform TEXT NOT NULL,
    model TEXT NOT NULL,                     -- default model
    models TEXT,                             -- JSON array of available models (NULL = [model])
    host TEXT,
    api_key BLOB,                            -- encrypted storage
    dimensions INTEGER,
    timeout INTEGER DEFAULT 120,
    is_active INTEGER DEFAULT 1,             -- BOOLEAN
    supports_vision INTEGER DEFAULT 0,       -- BOOLEAN
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS job_provider_assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_name TEXT UNIQUE NOT NULL,
    provider_id INTEGER NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
    model TEXT,                              -- model override (NULL = use provider default)
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_providers_name ON providers(name);
CREATE INDEX IF NOT EXISTS idx_providers_platform ON providers(platform);
CREATE INDEX IF NOT EXISTS idx_job_assignments_job ON job_provider_assignments(job_name);

-- ────────────────────────────────────────────────────────────────
-- SETTINGS — application-wide configuration
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    value TEXT,
    value_type TEXT DEFAULT 'string',
    description TEXT,
    is_sensitive INTEGER NOT NULL DEFAULT 0, -- BOOLEAN
    encrypted_value BLOB,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_settings_key ON settings(key);

INSERT OR IGNORE INTO settings (key, value_type, description, is_sensitive)
VALUES ('api_key', 'string', 'REST API authentication key (auto-generated on first startup if not set)', 1);

-- ────────────────────────────────────────────────────────────────
-- MASTER ACCOUNT
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS master_account (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

-- ────────────────────────────────────────────────────────────────
-- SCHEDULED ITEMS
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS scheduled_items (
    id TEXT PRIMARY KEY,
    item_type TEXT NOT NULL DEFAULT 'notification',
    message TEXT NOT NULL,
    due_at TEXT NOT NULL,
    recurrence TEXT,
    window_start TEXT,
    window_end TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    channel TEXT,
    created_by_session TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_fired_at TEXT,
    group_id TEXT,
    is_prompt INTEGER DEFAULT 0,             -- BOOLEAN
    source TEXT,                              -- origin: 'caldav', 'imap', 'system', NULL = user-created
    external_uid TEXT,                        -- dedup key for external sources
    metadata TEXT DEFAULT '{}',               -- JSON blob (location, attendees, etc.)
    hidden INTEGER DEFAULT 0                  -- hide from user-facing list/API
);

CREATE INDEX IF NOT EXISTS idx_scheduled_items_pending ON scheduled_items(due_at) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_scheduled_items_group_id ON scheduled_items(group_id, due_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_scheduled_items_external_uid ON scheduled_items(external_uid);


-- ────────────────────────────────────────────────────────────────
-- LISTS
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS lists (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    list_type TEXT NOT NULL DEFAULT 'checklist',
    metadata TEXT NOT NULL DEFAULT '{}',      -- JSONB
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    deleted_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_lists_name_unique
    ON lists(name COLLATE NOCASE)
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_lists_active ON lists(created_at DESC) WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS list_items (
    id TEXT PRIMARY KEY,
    list_id TEXT NOT NULL REFERENCES lists(id),
    content TEXT NOT NULL,
    checked INTEGER NOT NULL DEFAULT 0,      -- BOOLEAN
    position INTEGER NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}',      -- JSONB
    added_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    removed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_list_items_active ON list_items(list_id) WHERE removed_at IS NULL;

CREATE TABLE IF NOT EXISTS list_events (
    id TEXT PRIMARY KEY,
    list_id TEXT NOT NULL REFERENCES lists(id),
    event_type TEXT NOT NULL,
    item_content TEXT,
    details TEXT NOT NULL DEFAULT '{}',       -- JSONB
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_list_events_list ON list_events(list_id, created_at DESC);

-- ────────────────────────────────────────────────────────────────
-- TOOL CAPABILITY PROFILES
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tool_capability_profiles (
    id TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL UNIQUE,
    tool_type TEXT NOT NULL DEFAULT 'tool',
    short_summary TEXT NOT NULL,
    full_profile TEXT NOT NULL,
    usage_scenarios TEXT NOT NULL DEFAULT '[]',    -- JSONB
    anti_scenarios TEXT NOT NULL DEFAULT '[]',     -- JSONB
    complementary_skills TEXT DEFAULT '[]',        -- JSONB
    triage_triggers TEXT DEFAULT '[]',             -- JSONB
    manifest_hash TEXT,
    enrichment_episode_ids TEXT DEFAULT '[]',      -- JSONB
    enrichment_count INTEGER DEFAULT 0,
    last_enriched_at TEXT,
    avg_latency_ms REAL DEFAULT 0,
    cost_tier TEXT DEFAULT 'free',
    reliability_score REAL DEFAULT 1.0,
    domain TEXT DEFAULT 'Other',
    effort TEXT DEFAULT 'moderate',
    skill_category TEXT,                           -- e.g. 'memory', 'cognition', 'productivity'
    descriptor TEXT,                                -- compact discovery label: 'name (synonym1, synonym2, ...)'
    keywords TEXT DEFAULT '',                       -- comma-separated search keywords for 2-axis scoring
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tcp_tool_name ON tool_capability_profiles(tool_name);

-- ────────────────────────────────────────────────────────────────
-- TOOL PERFORMANCE METRICS
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tool_performance_metrics (
    id TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    exchange_id TEXT,
    invocation_success INTEGER NOT NULL,      -- BOOLEAN
    latency_ms REAL,
    cost_estimate REAL DEFAULT 0,
    user_correction INTEGER DEFAULT 0,        -- BOOLEAN
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tpm_tool_created ON tool_performance_metrics(tool_name, created_at DESC);

-- ────────────────────────────────────────────────────────────────
-- USER TOOL PREFERENCES
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_tool_preferences (
    id TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL UNIQUE,
    usage_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    explicit_preference REAL DEFAULT 0,
    implicit_preference REAL DEFAULT 0,
    last_used_at TEXT,
    updated_at TEXT DEFAULT (datetime('now'))
);


-- NOTE: moments are stored in the knowledge table as kind='moment' (parent)
-- and kind='moment_context' (enrichment context rows). The old dedicated
-- moments table and documents-table path (source_type='moment') are removed.

-- ────────────────────────────────────────────────────────────────
-- PLACE FINGERPRINTS — learned place patterns
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS place_fingerprints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint_hash TEXT UNIQUE NOT NULL,
    device_class TEXT NOT NULL,
    hour_bucket INTEGER NOT NULL,
    location_hash TEXT,
    connection_type TEXT,
    place_label TEXT NOT NULL,
    count INTEGER DEFAULT 1,
    last_seen_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_place_fp_hash ON place_fingerprints(fingerprint_hash);

-- persistent_tasks table removed — replaced by goal_pursuit skill + GoalPursuitProcessor.
DROP TABLE IF EXISTS persistent_tasks;
DROP TABLE IF EXISTS persistent_tasks_vec;

-- cognitive_reflexes table removed — CognitiveReflexService removed.
DROP TABLE IF EXISTS cognitive_reflexes;
DROP TABLE IF EXISTS cognitive_reflexes_vec;
-- triage_calibration_events table removed — TriageCalibrationService removed.
DROP TABLE IF EXISTS triage_calibration_events;

-- WATCHED FOLDERS — monitored filesystem directories
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS watched_folders (
    id TEXT PRIMARY KEY,
    folder_path TEXT NOT NULL UNIQUE,
    label TEXT,
    source_type TEXT DEFAULT 'filesystem',
    enabled INTEGER DEFAULT 1,
    file_patterns TEXT DEFAULT '["*"]',
    ignore_patterns TEXT DEFAULT '[".git","node_modules","__pycache__","build","dist",".DS_Store","Thumbs.db"]',
    recursive INTEGER DEFAULT 1,
    scan_interval INTEGER DEFAULT 300,
    last_scan_at TEXT,
    last_scan_files INTEGER DEFAULT 0,
    last_scan_error TEXT,
    source_config TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_watched_folders_enabled
    ON watched_folders(enabled) WHERE enabled = 1;


-- ────────────────────────────────────────────────────────────────
-- DOCUMENTS — document metadata + chunks
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    original_name TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    file_size_bytes INTEGER,
    file_path TEXT NOT NULL,
    file_hash TEXT,
    page_count INTEGER,
    status TEXT DEFAULT 'pending',
    error_message TEXT,
    chunk_count INTEGER DEFAULT 0,
    source_type TEXT DEFAULT 'upload',
    watched_folder_id TEXT REFERENCES watched_folders(id),
    tags TEXT DEFAULT '[]',                  -- JSON array (was TEXT[])
    summary TEXT,
    extracted_metadata TEXT DEFAULT '{}',    -- JSONB
    supersedes_id TEXT REFERENCES documents(id),
    clean_text TEXT,
    language TEXT,
    fingerprint TEXT,
    doc_category TEXT,
    doc_project TEXT,
    doc_date TEXT,
    meta_locked INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    deleted_at TEXT,
    purge_after TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_deleted ON documents(deleted_at) WHERE deleted_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_purge ON documents(purge_after) WHERE purge_after IS NOT NULL;
-- idx_documents_watched_folder created by migration 001_watched_folders.sql
CREATE INDEX IF NOT EXISTS idx_documents_folder_pending ON documents(watched_folder_id, status) WHERE deleted_at IS NULL AND watched_folder_id IS NOT NULL;

-- FTS5 for document search
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    original_name, summary, clean_text, content='documents', content_rowid='rowid'
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    page_number INTEGER,
    section_title TEXT,
    token_count INTEGER
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_id ON document_chunks(document_id);

-- FTS5 for chunk search (porter stemming: "temperatures" matches "temperature")
CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
    content, section_title, content='document_chunks', content_rowid='id',
    tokenize='porter unicode61'
);

-- ────────────────────────────────────────────────────────────────
-- SCHEMA VERSION
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO schema_version (version) VALUES (1);

-- ────────────────────────────────────────────────────────────────
-- SCHEMA MIGRATIONS TRACKING
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS schema_migrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT UNIQUE NOT NULL,
    applied_at TEXT DEFAULT (datetime('now'))
);

-- uncertainties table removed — dropped by migration 025, replaced by
-- pending_contradictions. Migration 009 is now a no-op so this table
-- no longer needs to exist in schema.sql for the migration chain.

-- ────────────────────────────────────────────────────────────────
-- PENDING CONTRADICTIONS — trait contradictions awaiting user resolution
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pending_contradictions (
    id TEXT PRIMARY KEY,
    trait_a_id INTEGER NOT NULL,
    trait_b_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    surfaced_at TEXT NOT NULL,
    source TEXT NOT NULL  -- 'chat' | 'ambient'
);

-- ────────────────────────────────────────────────────────────────
-- GOALS — persistent goal lifecycle (stated, inferred, emergent, developmental)
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,                -- natural language goal description
    type TEXT NOT NULL DEFAULT 'emergent'
        CHECK (type IN ('stated', 'inferred', 'emergent', 'developmental')),
    status TEXT NOT NULL DEFAULT 'candidate'
        CHECK (status IN ('candidate', 'strengthening', 'actionable', 'active', 'completed', 'decayed', 'evolved')),
    salience REAL NOT NULL DEFAULT 0.0,       -- 0.0–1.0 current strength
    confidence REAL NOT NULL DEFAULT 0.1,     -- 0.0–1.0 prediction confidence
    commitment REAL DEFAULT 0.0,              -- 0.0–1.0 commitment level
    evidence_count INTEGER DEFAULT 0,         -- number of supporting signals
    outcome_feedback TEXT DEFAULT '[]',       -- JSON array: engagement history
    is_muted INTEGER DEFAULT 0,               -- fast column: 1 if user has muted this goal
    urgency REAL DEFAULT 0.0,                 -- 0.0–1.0 time pressure
    timescale TEXT DEFAULT 'medium_term',     -- immediate, short_term, medium_term, long_term
    strategy TEXT,                            -- generated approach for actionable goals
    strategy_hash INTEGER,                    -- hash mod 10000 for tracking strategy versions
    parent_motives TEXT DEFAULT '[]',         -- JSON: aligned CORE_MOTIVES
    identity_links TEXT DEFAULT '[]',         -- JSON: aligned identity dimensions
    lineage_parent_id TEXT,                   -- parent goal for hierarchy
    channel TEXT,                             -- associated conversation channel (optional)
    last_reinforced_at TEXT,                  -- when goal last received evidence
    last_acted_at TEXT,                       -- when goal was last acted upon
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    derived_from TEXT DEFAULT '[]'         -- JSON array of episode IDs that formed this goal
);

CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);
CREATE INDEX IF NOT EXISTS idx_goals_salience ON goals(salience, status);
CREATE INDEX IF NOT EXISTS idx_goals_type ON goals(type, status);
CREATE INDEX IF NOT EXISTS idx_goals_active_salience ON goals(salience DESC) WHERE status IN ('candidate', 'strengthening', 'actionable');

-- ────────────────────────────────────────────────────────────────
-- GOAL EVIDENCE — accumulated signals supporting goals
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS goal_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
    signal_type TEXT NOT NULL,                -- 'topic_mention', 'trait_update', 'explicit', etc.
    content TEXT NOT NULL,                    -- the raw signal text
    source TEXT,                              -- which service emitted the signal
    strength REAL NOT NULL DEFAULT 0.5,       -- 0.0–1.0 how strongly this supports the goal
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_goal_evidence_goal_id ON goal_evidence(goal_id);
CREATE INDEX IF NOT EXISTS idx_goal_evidence_type ON goal_evidence(signal_type);

-- ────────────────────────────────────────────────────────────────
-- WORLD STATE VECTOR TABLES — salience-based retrieval
-- ────────────────────────────────────────────────────────────────
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_vec USING vec0(embedding float[768]);
CREATE VIRTUAL TABLE IF NOT EXISTS tool_capability_profiles_vec USING vec0(embedding float[768]);
CREATE VIRTUAL TABLE IF NOT EXISTS documents_vec USING vec0(embedding float[768]);
CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_vec USING vec0(embedding float[768]);
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_vec USING vec0(embedding float[768]);
CREATE VIRTUAL TABLE IF NOT EXISTS scheduled_items_vec USING vec0(embedding float[768]);
CREATE VIRTUAL TABLE IF NOT EXISTS lists_vec USING vec0(embedding float[768]);
CREATE VIRTUAL TABLE IF NOT EXISTS goals_vec USING vec0(embedding float[768]);

-- ────────────────────────────────────────────────────────────────
-- WRAPPER TOKENS — bearer auth for external programmatic access
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS wrapper_tokens (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    wrapper_id TEXT NOT NULL UNIQUE,
    capabilities TEXT NOT NULL DEFAULT '{}',
    permissions TEXT NOT NULL DEFAULT '{}',
    metadata TEXT NOT NULL DEFAULT '{}',
    last_seen_at TEXT,
    created_at TEXT NOT NULL,
    revoked_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_wrapper_tokens_hash
    ON wrapper_tokens(token_hash)
    WHERE revoked_at IS NULL;

-- ────────────────────────────────────────────────────────────────
-- LLM_CALL_LOG — per-call token usage and latency tracking
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS llm_call_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_name TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    tokens_input INTEGER NOT NULL DEFAULT 0,
    tokens_output INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_llm_call_log_job_created
    ON llm_call_log (job_name, created_at);

-- ────────────────────────────────────────────────────────────────
-- MEMORY RECALL LOG — telemetry for dynamic memory radius tuning
-- One row per memory recall call (seed or llm-driven). Written by
-- memory_skill after EpisodicService.retrieve_episodes returns.
-- Consumed by the meta-harness (nightly tests) to tune the 8
-- radius constants in memory_skill.py — see
-- /Volumes/llm/chalie-plans/v0.3.2/memory-dynamic-radius.md.
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memory_recall_log (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at               TEXT NOT NULL DEFAULT (datetime('now')),
    turn_uid                 TEXT NOT NULL,
    transcript_id            INTEGER,
    channel                  TEXT NOT NULL,
    caller                   TEXT NOT NULL CHECK(caller IN ('seed', 'llm_recall')),
    query                    TEXT NOT NULL,
    query_embedding_hash     TEXT NOT NULL,
    input_radius             REAL NOT NULL,
    narrow_factor            REAL NOT NULL DEFAULT 1.0,
    expand_factor            REAL NOT NULL DEFAULT 1.0,
    adaptive_shrink_divisor  REAL NOT NULL DEFAULT 1.0,
    effective_radius         REAL NOT NULL,
    episode_count            INTEGER NOT NULL DEFAULT 0,
    vector_candidates        INTEGER NOT NULL DEFAULT 0,
    fts_candidates           INTEGER NOT NULL DEFAULT 0,
    survivors_after_radius   INTEGER NOT NULL DEFAULT 0,
    final_rrf_count          INTEGER NOT NULL DEFAULT 0,
    top_distances            TEXT
);

CREATE INDEX IF NOT EXISTS idx_memory_recall_log_turn
    ON memory_recall_log (turn_uid, id);
CREATE INDEX IF NOT EXISTS idx_memory_recall_log_caller
    ON memory_recall_log (caller, created_at DESC);

-- ────────────────────────────────────────────────────────────────
-- TRANSCRIPT — persistent, channel-scoped conversation record
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS transcript (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    channel     TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    tool_call_id TEXT,
    tool_name   TEXT,
    internal    INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_transcript_channel ON transcript(channel, created_at);
CREATE INDEX IF NOT EXISTS idx_transcript_channel_created_desc ON transcript(channel, created_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS transcript_vec USING vec0(
    embedding float[768]
);

-- ────────────────────────────────────────────────────────────────
-- COMPACTIONS — incremental conversation summarization
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS compactions (
    channel             TEXT PRIMARY KEY,
    compacted_text      TEXT NOT NULL,
    compacted_up_to_id  INTEGER NOT NULL,
    token_count         INTEGER DEFAULT 0,
    updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
    overflow_content    TEXT,
    FOREIGN KEY (compacted_up_to_id) REFERENCES transcript(id)
);

-- ────────────────────────────────────────────────────────────────
-- TOOL CALLS — audit log of every tool invocation per transcript turn
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tool_calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript_id INTEGER NOT NULL,
    tool_name     TEXT NOT NULL,
    params        TEXT DEFAULT '{}',
    result        TEXT DEFAULT '',
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    ephemeral     INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (transcript_id) REFERENCES transcript(id)
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_transcript ON tool_calls(transcript_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_created ON tool_calls(created_at DESC);

-- ────────────────────────────────────────────────────────────────
-- BROWSER SNAPSHOTS — page monitoring change detection
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS browser_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id   INTEGER NOT NULL,
    snapshot_key TEXT NOT NULL,
    url          TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    content_text TEXT NOT NULL,
    captured_at  TEXT NOT NULL,
    UNIQUE(account_id, snapshot_key)
);

-- ────────────────────────────────────────────────────────────────
-- BROWSER CREDENTIALS — encrypted credential vault
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS browser_credentials (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id      INTEGER NOT NULL,
    domain          TEXT NOT NULL,
    label           TEXT NOT NULL,
    credential_type TEXT NOT NULL,
    encrypted_data  TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    last_used_at    TEXT,
    UNIQUE(account_id, domain, label)
);

-- ────────────────────────────────────────────────────────────────
-- DATA GRAPH — research-informed knowledge graph (replaces knowledge)
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS data_graph (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    -- CHECK constraint removed: Python validates kind via VALID_KINDS in data_graph_service.py.
    -- To be restored when SchemaConvergence handles constraint changes (v0.5.0 TODO).
    kind              TEXT NOT NULL,
    key               TEXT NOT NULL,
    value             TEXT,
    storage_strength  REAL NOT NULL DEFAULT 0.5,
    retrieval_weight  REAL NOT NULL DEFAULT 1.0,
    salience_score    REAL NOT NULL DEFAULT 0.0,
    evidence_count    INTEGER NOT NULL DEFAULT 1,
    first_seen_at     TEXT NOT NULL DEFAULT (datetime('now')),
    last_confirmed_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_accessed_at  TEXT,
    source            TEXT,
    deleted_at        TEXT,
    active            INTEGER NOT NULL DEFAULT 1,
    search_queries    TEXT DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_data_graph_kind         ON data_graph(kind);
CREATE INDEX IF NOT EXISTS idx_data_graph_key          ON data_graph(key);
CREATE INDEX IF NOT EXISTS idx_data_graph_retrieval    ON data_graph(retrieval_weight DESC);
CREATE INDEX IF NOT EXISTS idx_data_graph_active       ON data_graph(kind, active) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_data_graph_confirmed    ON data_graph(last_confirmed_at);

CREATE VIRTUAL TABLE IF NOT EXISTS data_graph_fts USING fts5(
    key, value, kind, search_queries,
    content='data_graph', content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS data_graph_key_vec   USING vec0(embedding float[768]);
CREATE VIRTUAL TABLE IF NOT EXISTS data_graph_value_vec USING vec0(embedding float[768]);

-- ────────────────────────────────────────────────────────────────
-- DATA GRAPH EDGES — typed join table for graph traversal
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS data_graph_edges (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    from_id          INTEGER NOT NULL REFERENCES data_graph(id) ON DELETE CASCADE,
    to_id            INTEGER NOT NULL REFERENCES data_graph(id) ON DELETE CASCADE,
    edge_type        TEXT NOT NULL DEFAULT 'related',
    strength         REAL NOT NULL DEFAULT 1.0,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    last_accessed_at TEXT,
    UNIQUE (from_id, to_id, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_data_graph_edges_from ON data_graph_edges(from_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_data_graph_edges_to   ON data_graph_edges(to_id, edge_type);
