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
    gist TEXT NOT NULL,
    salience INTEGER NOT NULL CHECK (salience BETWEEN 1 AND 10),
    channel TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    last_accessed_at TEXT,
    access_count INTEGER DEFAULT 0,
    deleted_at TEXT,
    transcript_ids TEXT DEFAULT '[]',         -- JSONB: list of transcript.id values this episode covers
    transcript_id_start INTEGER,              -- lowest transcript.id in this episode's range
    transcript_id_end INTEGER,                -- highest transcript.id in this episode's range
    emotional_valence REAL,                   -- -1.0 (negative) to 1.0 (positive)
    emotional_arousal REAL,                   -- 0.0 (calm) to 1.0 (intense) — drives consolidation strength
    consolidated_from TEXT DEFAULT '[]',      -- JSONB: episode IDs this was consolidated from
    consolidated_into TEXT,                   -- back-pointer to super-episode id (UUID, FK-ish to episodes.id)
    storage_strength REAL DEFAULT 1.0,        -- encoding strength at storage time
    retrieval_weight REAL DEFAULT 1.0         -- current retrieval priority weight
);

CREATE INDEX IF NOT EXISTS idx_episodes_channel ON episodes(channel) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_episodes_composite ON episodes(channel, retrieval_weight DESC, created_at DESC) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_episodes_transcript_range ON episodes(transcript_id_start, transcript_id_end);
CREATE INDEX IF NOT EXISTS idx_episodes_retrieval_weight ON episodes(retrieval_weight DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_consolidated_into ON episodes(consolidated_into) WHERE consolidated_into IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_episodes_apex ON episodes(retrieval_weight DESC, created_at DESC) WHERE deleted_at IS NULL AND consolidated_into IS NULL;

-- FTS5 for full-text search on episodes (replaces GIN tsvector indexes)
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    gist, content='episodes', content_rowid='rowid'
);

-- cortex_iterations removed — CortexIterationService not wired into runtime.

-- semantic_concepts, semantic_relationships removed — replaced by data_graph table.
-- semantic_schemas table removed — never used by any service.

-- procedural_memory removed — replaced by data_graph table.

-- knowledge table removed — all trait/concept/procedure storage routes through DataGraphService.

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
-- VAULT — envelope-encryption key store
-- vault_config: singleton row (id MUST equal 1)
--   Holds all KDF parameters and the KEK-wrapped DEK used by
--   VaultService for AES-256-GCM envelope encryption.
-- vault_secrets: optional centralised secret store (reserved for
--   future use; not required by Phase 1).
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS vault_config (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    kdf_salt        BLOB    NOT NULL,
    kdf_algorithm   TEXT    NOT NULL DEFAULT 'pbkdf2_sha256',
    kdf_iterations  INTEGER NOT NULL DEFAULT 600000,
    wrapped_dek     BLOB    NOT NULL,
    dek_nonce       BLOB    NOT NULL,
    created_at      TEXT,
    updated_at      TEXT,
    reinitialized_at TEXT
);

CREATE TABLE IF NOT EXISTS vault_secrets (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    scope                 TEXT    NOT NULL,
    scope_ref             TEXT    NOT NULL,
    encrypted_value       BLOB    NOT NULL,
    nonce                 BLOB    NOT NULL,
    migrated_from_fernet  INTEGER DEFAULT 0,
    created_at            TEXT
);

CREATE INDEX IF NOT EXISTS idx_vault_secrets_scope     ON vault_secrets(scope);
CREATE INDEX IF NOT EXISTS idx_vault_secrets_scope_ref ON vault_secrets(scope, scope_ref);

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
    model TEXT NOT NULL,                     -- single model for this provider
    host TEXT,
    api_key BLOB,                            -- encrypted storage
    dimensions INTEGER,
    timeout INTEGER DEFAULT 120,
    is_active INTEGER DEFAULT 1,             -- BOOLEAN
    supports_vision INTEGER DEFAULT 0,       -- BOOLEAN
    max_tokens INTEGER,
    compact_at INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_providers_name ON providers(name);
CREATE INDEX IF NOT EXISTS idx_providers_platform ON providers(platform);

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

INSERT OR IGNORE INTO settings (key, value_type, description, is_sensitive)
VALUES ('selected_provider_id', 'int', 'ID of the active LLM provider used for all cognitive jobs', 0);

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

-- list_events + idx_list_events_list removed — list history was a pointless
-- LLM-facing feature; SchemaConvergenceService DROPs both on next boot.

-- tool_capability_profiles + idx_tcp_tool_name removed — profiles replaced
-- by abilities.sqlite (ability registry). SchemaConvergenceService auto-DROPs
-- both the table and its index on next boot.

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


-- NOTE: moments are stored in data_graph as kind='moment' (parent)
-- and kind='moment_context' (enrichment context rows). The old dedicated
-- moments table and documents-table path (source_type='moment') are removed.

-- Note: tables that disappear from this schema are dropped automatically by
-- SchemaConvergenceService on the next boot.  No explicit DROP statements
-- needed here.  Past examples: cortex_iterations, persistent_tasks,
-- cognitive_reflexes, triage_calibration_events, document_chunks,
-- knowledge, knowledge_vec, knowledge_fts, scheduled_items_vec, lists_vec,
-- goals_vec, place_fingerprints, ambient_inferences, situation_states.

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


-- ────────────────────────────────────────────────────────────────
-- SCHEMA VERSION
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO schema_version (version) VALUES (1);

-- schema_migrations table removed — SchemaConvergenceService is now the
-- single source of truth.  schema.sql declares the desired shape and the
-- service converges the live DB to match.  Numbered migration files are
-- gone; legacy schema_migrations rows are auto-dropped on the next boot.

-- ────────────────────────────────────────────────────────────────
-- CONCEPT LUT MISSES — keys that didn't match the concept LUT
-- Rows accumulate as the LUT is encountered at runtime; used to
-- identify canonical key candidates for future LUT expansion.
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS concept_lut_misses (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,
    key        TEXT NOT NULL,
    value_preview TEXT,
    count      INTEGER NOT NULL DEFAULT 1,
    first_seen TEXT NOT NULL,
    last_seen  TEXT NOT NULL,
    UNIQUE(kind, key)
);
CREATE INDEX IF NOT EXISTS idx_lut_misses_kind ON concept_lut_misses(kind, count DESC);

-- ────────────────────────────────────────────────────────────────
-- WORLD STATE VECTOR TABLES — salience-based retrieval
-- ────────────────────────────────────────────────────────────────
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_vec USING vec0(embedding float[768]);
-- tool_capability_profiles_vec removed — replaced by abilities.sqlite.
CREATE VIRTUAL TABLE IF NOT EXISTS documents_vec USING vec0(embedding float[768]);
CREATE VIRTUAL TABLE IF NOT EXISTS scheduled_items_vec USING vec0(embedding float[768]);
CREATE VIRTUAL TABLE IF NOT EXISTS lists_vec USING vec0(embedding float[768]);

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
    tokens_cache_read INTEGER NOT NULL DEFAULT 0,
    tokens_cache_create INTEGER NOT NULL DEFAULT 0,
    tokens_thinking INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    usage_class TEXT NOT NULL DEFAULT 'chat',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_llm_call_log_job_created
    ON llm_call_log (job_name, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_call_log_usage_class
    ON llm_call_log (usage_class, created_at);

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
    deliberation_score REAL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    xml_migrated INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_transcript_channel ON transcript(channel, created_at);
CREATE INDEX IF NOT EXISTS idx_transcript_channel_created_desc ON transcript(channel, created_at DESC);

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

-- ────────────────────────────────────────────────────────────────
-- EXPANDED SEMANTIC — variant query strings + embeddings for KNN recall
-- Populated by SearchExpanderService (search_expander_service.py) after
-- every knowledge / data_graph write. Each row is one doc2query variant
-- whose embedding lives in the companion vec0 table, keyed by this row's id.
-- Callers: data_graph_service.recall() joins
--          expanded_semantic_vec → expanded_semantic to surface variant hits.
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS expanded_semantic (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    relates_to_table TEXT NOT NULL,   -- 'data_graph'
    related_to_id   INTEGER NOT NULL, -- rowid of the source row
    str             TEXT NOT NULL     -- the variant query string
);

CREATE INDEX IF NOT EXISTS idx_expanded_semantic_lookup
    ON expanded_semantic(relates_to_table, related_to_id);

-- One vec row per expanded_semantic row; rowid matches expanded_semantic.id.
CREATE VIRTUAL TABLE IF NOT EXISTS expanded_semantic_vec USING vec0(embedding float[768]);

-- Cascade: DELETE data_graph row → purge its expanded_semantic rows.
CREATE TRIGGER IF NOT EXISTS expanded_semantic_cascade_data_graph
    AFTER DELETE ON data_graph BEGIN
    DELETE FROM expanded_semantic
        WHERE relates_to_table = 'data_graph' AND related_to_id = OLD.id;
END;

-- Cascade: DELETE expanded_semantic row → purge its vec row.
CREATE TRIGGER IF NOT EXISTS expanded_semantic_vec_sync
    AFTER DELETE ON expanded_semantic BEGIN
    DELETE FROM expanded_semantic_vec WHERE rowid = OLD.id;
END;

-- ============================================================================
-- POLICY — per-action permission control (allow / ask / deny).
-- ============================================================================
CREATE TABLE IF NOT EXISTS policy_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_id TEXT NOT NULL,
    context TEXT NOT NULL,
    state TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(action_id, context)
);

CREATE TABLE IF NOT EXISTS policy_blocked_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_id TEXT NOT NULL,
    context TEXT NOT NULL,
    reason TEXT NOT NULL,
    params_json TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_policy_blocked_log_created ON policy_blocked_log(created_at DESC);

-- ============================================================================
-- TELEMETRY — flat key/value store for the latest client heartbeat.
-- ============================================================================
-- Populated by /health POST. The frontend (heartbeat.js) is the source of
-- truth for which keys are collected; the backend persists whatever is sent.
-- Nested payload keys are flattened with dots, e.g.
--   {"device": {"name": "iPhone"}}  →  key='device.name', value='iPhone'
-- One row per key; UPSERT on every push. WorldState renders by reading the
-- whole table and grouping by the top-level prefix.
CREATE TABLE IF NOT EXISTS telemetry (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- ────────────────────────────────────────────────────────────────
-- QUICK TIP STATE — tracks per-tip display/dismiss/retire state
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS quick_tip_state (
    tip_id        TEXT    NOT NULL,
    shown_count   INTEGER NOT NULL DEFAULT 0,
    dismissed     INTEGER NOT NULL DEFAULT 0,
    retired       INTEGER NOT NULL DEFAULT 0,
    last_shown_at TEXT,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    PRIMARY KEY (tip_id)
);
