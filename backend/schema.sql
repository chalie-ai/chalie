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
    freshness INTEGER NOT NULL CHECK (freshness BETWEEN 1 AND 10),
    topic TEXT NOT NULL,
    exchange_id TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    last_accessed_at TEXT,
    access_count INTEGER DEFAULT 0,
    deleted_at TEXT,
    activation_score REAL DEFAULT 1.0,
    salience_factors TEXT DEFAULT '{}',       -- JSONB
    open_loops TEXT DEFAULT '[]',             -- JSONB
    semantic_consolidation_status TEXT
);

CREATE INDEX IF NOT EXISTS idx_episodes_topic ON episodes(topic) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_episodes_activation ON episodes(activation_score DESC) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_episodes_composite ON episodes(topic, activation_score DESC, created_at DESC) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_episodes_intent_type ON episodes(json_extract(intent, '$.type'));
CREATE INDEX IF NOT EXISTS idx_episodes_semantic_status ON episodes(semantic_consolidation_status);

-- FTS5 for full-text search on episodes (replaces GIN tsvector indexes)
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    gist, action, content='episodes', content_rowid='rowid'
);

-- ────────────────────────────────────────────────────────────────
-- CORTEX ITERATIONS — ACT loop audit trail
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cortex_iterations (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
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
CREATE INDEX IF NOT EXISTS idx_cortex_iterations_topic ON cortex_iterations(topic, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cortex_iterations_exchange ON cortex_iterations(exchange_id);

-- ────────────────────────────────────────────────────────────────
-- SEMANTIC CONCEPTS — knowledge nodes
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS semantic_concepts (
    id TEXT PRIMARY KEY,
    concept_name TEXT NOT NULL,
    concept_type TEXT NOT NULL,
    definition TEXT NOT NULL,
    abstraction_level INTEGER DEFAULT 3,
    domain TEXT,
    strength REAL DEFAULT 1.0,
    activation_score REAL DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    consolidation_count INTEGER DEFAULT 0,
    confidence REAL DEFAULT 0.5,
    source_episodes TEXT DEFAULT '[]',        -- JSONB
    verification_status TEXT DEFAULT 'unverified',
    context_constraints TEXT DEFAULT '{}',    -- JSONB
    examples TEXT DEFAULT '[]',              -- JSONB
    first_learned_at TEXT DEFAULT (datetime('now')),
    last_accessed_at TEXT DEFAULT (datetime('now')),
    last_reinforced_at TEXT DEFAULT (datetime('now')),
    utility_score REAL DEFAULT 0.5,
    decay_resistance REAL DEFAULT 0.5,
    deleted_at TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_concepts_name ON semantic_concepts(concept_name);
CREATE INDEX IF NOT EXISTS idx_concepts_type ON semantic_concepts(concept_type);
CREATE INDEX IF NOT EXISTS idx_concepts_domain ON semantic_concepts(domain);
CREATE INDEX IF NOT EXISTS idx_concepts_strength ON semantic_concepts(strength DESC);
CREATE INDEX IF NOT EXISTS idx_concepts_activation ON semantic_concepts(activation_score DESC);
CREATE INDEX IF NOT EXISTS idx_concepts_deleted ON semantic_concepts(deleted_at) WHERE deleted_at IS NULL;

-- ────────────────────────────────────────────────────────────────
-- SEMANTIC RELATIONSHIPS
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS semantic_relationships (
    id TEXT PRIMARY KEY,
    source_concept_id TEXT NOT NULL REFERENCES semantic_concepts(id),
    target_concept_id TEXT NOT NULL REFERENCES semantic_concepts(id),
    relationship_type TEXT NOT NULL,
    strength REAL DEFAULT 0.5,
    bidirectional INTEGER DEFAULT 0,         -- BOOLEAN
    source_episodes TEXT DEFAULT '[]',       -- JSONB
    confidence REAL DEFAULT 0.5,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    deleted_at TEXT,
    UNIQUE(source_concept_id, target_concept_id, relationship_type)
);

CREATE INDEX IF NOT EXISTS idx_relationships_source ON semantic_relationships(source_concept_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON semantic_relationships(target_concept_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON semantic_relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_relationships_strength ON semantic_relationships(strength DESC);

-- ────────────────────────────────────────────────────────────────
-- SEMANTIC SCHEMAS — mental frameworks
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS semantic_schemas (
    id TEXT PRIMARY KEY,
    schema_name TEXT NOT NULL UNIQUE,
    description TEXT,
    core_concepts TEXT NOT NULL,              -- JSONB
    relationships TEXT DEFAULT '[]',         -- JSONB
    activation_count INTEGER DEFAULT 0,
    last_activated_at TEXT,
    learned_from_episodes TEXT DEFAULT '[]', -- JSONB
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_schemas_name ON semantic_schemas(schema_name);
CREATE INDEX IF NOT EXISTS idx_schemas_activation ON semantic_schemas(activation_count DESC);

-- ────────────────────────────────────────────────────────────────
-- INTERACTION LOG — append-only audit trail
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS interaction_log (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    topic TEXT,
    exchange_id TEXT,
    session_id TEXT,
    source TEXT,
    thread_id TEXT,
    payload TEXT NOT NULL DEFAULT '{}',       -- JSONB
    metadata TEXT DEFAULT '{}',              -- JSONB
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_interaction_log_topic_created ON interaction_log(topic, created_at);
CREATE INDEX IF NOT EXISTS idx_interaction_log_event_type_created ON interaction_log(event_type, created_at);
CREATE INDEX IF NOT EXISTS idx_interaction_log_session_created ON interaction_log(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_interaction_log_exchange ON interaction_log(exchange_id);

-- ────────────────────────────────────────────────────────────────
-- PROCEDURAL MEMORY — policy weights
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS procedural_memory (
    id TEXT PRIMARY KEY,
    action_name TEXT NOT NULL UNIQUE,
    total_attempts INTEGER DEFAULT 0,
    total_successes INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    avg_reward REAL DEFAULT 0.0,
    weight REAL DEFAULT 1.0,
    reward_history TEXT DEFAULT '[]',        -- JSONB
    context_stats TEXT DEFAULT '{}',         -- JSONB
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_procedural_action_name ON procedural_memory(action_name);
CREATE INDEX IF NOT EXISTS idx_procedural_weight ON procedural_memory(weight DESC);

-- ────────────────────────────────────────────────────────────────
-- TOPICS — semantic attractors
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS topics (
    topic_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_updated TEXT NOT NULL DEFAULT (datetime('now')),
    message_count INTEGER NOT NULL DEFAULT 0,
    avg_salience REAL NOT NULL DEFAULT 0.5,
    metadata TEXT DEFAULT '{}'               -- JSONB
);

CREATE INDEX IF NOT EXISTS idx_topics_name ON topics(name);
CREATE INDEX IF NOT EXISTS idx_topics_last_updated ON topics(last_updated DESC);

-- ────────────────────────────────────────────────────────────────
-- ROUTING DECISIONS — mode router audit trail
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS routing_decisions (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    exchange_id TEXT,
    selected_mode TEXT NOT NULL,
    router_confidence REAL,
    scores TEXT NOT NULL,                     -- JSONB
    tiebreaker_used INTEGER DEFAULT 0,       -- BOOLEAN
    tiebreaker_candidates TEXT,              -- JSONB
    margin REAL,
    effective_margin REAL,
    signal_snapshot TEXT NOT NULL,            -- JSONB
    weight_snapshot TEXT,                     -- JSONB
    routing_time_ms REAL,
    feedback TEXT,                            -- JSONB
    reflection TEXT,                          -- JSONB
    previous_mode TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_routing_decisions_topic ON routing_decisions(topic, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_routing_decisions_mode ON routing_decisions(selected_mode, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_routing_decisions_unreflected ON routing_decisions(created_at) WHERE reflection IS NULL;

-- ────────────────────────────────────────────────────────────────
-- IDENTITY VECTORS — 6 personality dimensions
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS identity_vectors (
    id TEXT PRIMARY KEY,
    vector_name TEXT NOT NULL UNIQUE,
    baseline_weight REAL NOT NULL DEFAULT 0.5,
    current_activation REAL NOT NULL DEFAULT 0.5,
    plasticity_rate REAL NOT NULL DEFAULT 0.05,
    inertia_rate REAL NOT NULL DEFAULT 0.1,
    min_cap REAL NOT NULL DEFAULT 0.2,
    max_cap REAL NOT NULL DEFAULT 0.8,
    reinforcement_count INTEGER DEFAULT 0,
    signal_history TEXT DEFAULT '[]',         -- JSONB
    baseline_drift_today REAL DEFAULT 0,
    drift_window_start TEXT DEFAULT (datetime('now')),
    created_at TEXT DEFAULT (datetime('now')),
    last_updated_at TEXT DEFAULT (datetime('now'))
);

-- Seed default archetype
INSERT OR IGNORE INTO identity_vectors (id, vector_name, baseline_weight, current_activation, plasticity_rate, inertia_rate, min_cap, max_cap)
VALUES
    ('iv-curiosity',           'curiosity',           0.7, 0.7, 0.05, 0.10, 0.3, 0.9),
    ('iv-assertiveness',       'assertiveness',       0.6, 0.6, 0.04, 0.10, 0.3, 0.8),
    ('iv-warmth',              'warmth',              0.6, 0.6, 0.05, 0.10, 0.3, 0.8),
    ('iv-playfulness',         'playfulness',         0.4, 0.4, 0.04, 0.10, 0.2, 0.7),
    ('iv-skepticism',          'skepticism',          0.5, 0.5, 0.03, 0.10, 0.2, 0.7),
    ('iv-emotional_intensity', 'emotional_intensity', 0.4, 0.4, 0.02, 0.15, 0.2, 0.6);

-- Identity event log
CREATE TABLE IF NOT EXISTS identity_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vector_name TEXT NOT NULL,
    old_activation REAL NOT NULL,
    new_activation REAL NOT NULL,
    signal_source TEXT NOT NULL,
    signal_value REAL,
    topic TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_identity_events_time ON identity_events(created_at);
CREATE INDEX IF NOT EXISTS idx_identity_events_vector ON identity_events(vector_name, created_at);

-- ────────────────────────────────────────────────────────────────
-- USER TRAITS — per-user trait memory
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_traits (
    id TEXT PRIMARY KEY,
    user_id TEXT DEFAULT 'primary',
    trait_key TEXT NOT NULL,
    trait_value TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    confidence REAL DEFAULT 0.5,
    source TEXT DEFAULT 'inferred',
    is_literal INTEGER DEFAULT 1,            -- BOOLEAN
    reinforcement_count INTEGER DEFAULT 1,
    last_reinforced_at TEXT DEFAULT (datetime('now')),
    last_conflict_at TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(user_id, trait_key)
);

CREATE INDEX IF NOT EXISTS idx_user_traits_user ON user_traits(user_id);
CREATE INDEX IF NOT EXISTS idx_user_traits_category ON user_traits(user_id, category);
CREATE INDEX IF NOT EXISTS idx_user_traits_confidence ON user_traits(user_id, confidence);

-- ────────────────────────────────────────────────────────────────
-- MESSAGE CYCLES — processing unit tracking
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS message_cycles (
    cycle_id TEXT PRIMARY KEY,
    parent_cycle_id TEXT REFERENCES message_cycles(cycle_id),
    root_cycle_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    cycle_type TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT,
    intent TEXT,                              -- JSONB
    metadata TEXT DEFAULT '{}',              -- JSONB
    status TEXT DEFAULT 'pending',
    depth INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_cycles_parent ON message_cycles(parent_cycle_id);
CREATE INDEX IF NOT EXISTS idx_cycles_root ON message_cycles(root_cycle_id);
CREATE INDEX IF NOT EXISTS idx_cycles_topic_created ON message_cycles(topic, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cycles_status_type ON message_cycles(status, cycle_type);

-- ────────────────────────────────────────────────────────────────
-- THREADS — conversation threads
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    platform TEXT NOT NULL DEFAULT 'unknown',
    state TEXT NOT NULL DEFAULT 'active',
    current_topic TEXT,
    topic_history TEXT DEFAULT '[]',          -- JSONB
    exchange_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    last_activity TEXT DEFAULT (datetime('now')),
    expired_at TEXT,
    summary TEXT
);

CREATE INDEX IF NOT EXISTS idx_threads_user_channel ON threads(user_id, channel_id);
CREATE INDEX IF NOT EXISTS idx_threads_state ON threads(state);

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
    model TEXT NOT NULL,
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
    topic TEXT,
    created_by_session TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_fired_at TEXT,
    group_id TEXT,
    is_prompt INTEGER DEFAULT 0              -- BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_scheduled_items_pending ON scheduled_items(due_at) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_scheduled_items_group_id ON scheduled_items(group_id, due_at DESC);

-- ────────────────────────────────────────────────────────────────
-- AUTOBIOGRAPHY — user narrative synthesis
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS autobiography (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'primary',
    version INTEGER NOT NULL DEFAULT 1,
    narrative TEXT NOT NULL,
    section_hashes TEXT NOT NULL DEFAULT '{}',  -- JSONB
    episode_cursor TEXT,
    episodes_since INTEGER NOT NULL DEFAULT 0,
    synthesis_model TEXT,
    synthesis_ms INTEGER,
    delta_summary TEXT,                        -- JSONB
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, version)
);

CREATE INDEX IF NOT EXISTS idx_autobiography_user_version ON autobiography(user_id, version DESC);

-- ────────────────────────────────────────────────────────────────
-- LISTS
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS lists (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'primary',
    name TEXT NOT NULL,
    list_type TEXT NOT NULL DEFAULT 'checklist',
    metadata TEXT NOT NULL DEFAULT '{}',      -- JSONB
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    deleted_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_lists_user_name_unique
    ON lists(user_id, name COLLATE NOCASE)
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_lists_user_active ON lists(user_id) WHERE deleted_at IS NULL;

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
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tcp_tool_name ON tool_capability_profiles(tool_name);

-- ────────────────────────────────────────────────────────────────
-- TRIAGE CALIBRATION EVENTS
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS triage_calibration_events (
    id TEXT PRIMARY KEY,
    exchange_id TEXT,
    topic TEXT,
    triage_branch TEXT NOT NULL,
    triage_mode TEXT NOT NULL,
    tool_selected TEXT,                       -- JSON array (was TEXT[])
    confidence_internal REAL,
    confidence_tool_need REAL,
    reasoning TEXT,
    freshness_risk REAL,
    decision_entropy REAL,
    self_eval_override INTEGER DEFAULT 0,     -- BOOLEAN
    self_eval_reason TEXT,
    outcome_mode TEXT,
    outcome_tools_used TEXT,                  -- JSON array (was TEXT[])
    outcome_tool_success INTEGER,             -- BOOLEAN
    outcome_latency_ms REAL,
    tool_abstention INTEGER DEFAULT 0,        -- BOOLEAN
    signal_rephrase INTEGER DEFAULT 0,        -- BOOLEAN
    signal_correction INTEGER DEFAULT 0,      -- BOOLEAN
    signal_explicit_lookup INTEGER DEFAULT 0, -- BOOLEAN
    signal_abandonment INTEGER DEFAULT 0,     -- BOOLEAN
    correctness_label TEXT,
    correctness_score REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tce_created ON triage_calibration_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tce_topic ON triage_calibration_events(topic, created_at DESC);

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
    follow_up_confusion INTEGER DEFAULT 0,    -- BOOLEAN
    result_used_in_response INTEGER DEFAULT 1,-- BOOLEAN
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tpm_tool_created ON tool_performance_metrics(tool_name, created_at DESC);

-- ────────────────────────────────────────────────────────────────
-- USER TOOL PREFERENCES
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_tool_preferences (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'default',
    tool_name TEXT NOT NULL,
    usage_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    explicit_preference REAL DEFAULT 0,
    implicit_preference REAL DEFAULT 0,
    last_used_at TEXT,
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(user_id, tool_name)
);

CREATE INDEX IF NOT EXISTS idx_utp_user ON user_tool_preferences(user_id);

-- ────────────────────────────────────────────────────────────────
-- CURIOSITY THREADS
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS curiosity_threads (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    rationale TEXT,
    thread_type TEXT NOT NULL CHECK (thread_type IN ('learning', 'behavioral')),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'dormant', 'abandoned')),
    seed_topic TEXT,
    learning_notes TEXT NOT NULL DEFAULT '[]',  -- JSONB
    last_explored_at TEXT,
    exploration_count INTEGER NOT NULL DEFAULT 0,
    last_surfaced_at TEXT,
    engagement_score REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_curiosity_threads_status ON curiosity_threads(status);
CREATE INDEX IF NOT EXISTS idx_curiosity_threads_explore
    ON curiosity_threads(status, last_explored_at)
    WHERE status = 'active';

-- ────────────────────────────────────────────────────────────────
-- MOMENTS — pinned message bookmarks
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS moments (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'primary',
    title TEXT,
    message_text TEXT NOT NULL,
    exchange_id TEXT,
    topic TEXT,
    thread_id TEXT,
    gists TEXT NOT NULL DEFAULT '[]',         -- JSONB
    summary TEXT,
    status TEXT NOT NULL DEFAULT 'enriching'
        CHECK (status IN ('enriching', 'sealed', 'forgotten')),
    pinned_at TEXT NOT NULL DEFAULT (datetime('now')),
    sealed_at TEXT,
    last_enriched_at TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',      -- JSONB
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    deleted_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_moments_user_active
    ON moments(user_id, pinned_at DESC) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_moments_enriching
    ON moments(status, pinned_at) WHERE status = 'enriching' AND deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_moments_topic
    ON moments(topic, pinned_at DESC) WHERE deleted_at IS NULL;

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

-- ────────────────────────────────────────────────────────────────
-- PERSISTENT TASKS — multi-session ACT work
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS persistent_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER REFERENCES master_account(id),
    thread_id TEXT REFERENCES threads(thread_id),
    goal TEXT NOT NULL,
    scope TEXT,
    status TEXT DEFAULT 'proposed',
    priority INTEGER DEFAULT 5,
    progress TEXT DEFAULT '{}',              -- JSONB
    result TEXT,
    result_artifact TEXT,                    -- JSONB
    iterations_used INTEGER DEFAULT 0,
    max_iterations INTEGER DEFAULT 20,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    expires_at TEXT DEFAULT (datetime('now', '+14 days')),
    deadline TEXT,
    next_run_after TEXT,
    fatigue_budget REAL DEFAULT 15.0
);

CREATE INDEX IF NOT EXISTS idx_persistent_tasks_status ON persistent_tasks(account_id, status);
CREATE INDEX IF NOT EXISTS idx_persistent_tasks_next_run ON persistent_tasks(status, next_run_after);

-- ────────────────────────────────────────────────────────────────
-- COGNITIVE REFLEXES — learned fast-path clusters
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cognitive_reflexes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_queries TEXT DEFAULT '[]',         -- JSON array (was TEXT[])
    times_seen INTEGER DEFAULT 1,
    times_unnecessary INTEGER DEFAULT 0,
    times_activated INTEGER DEFAULT 0,
    times_succeeded INTEGER DEFAULT 0,
    times_failed INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    last_seen TEXT DEFAULT (datetime('now')),
    last_activated TEXT
);

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
    tags TEXT DEFAULT '[]',                  -- JSON array (was TEXT[])
    summary TEXT,
    extracted_metadata TEXT DEFAULT '{}',    -- JSONB
    supersedes_id TEXT REFERENCES documents(id),
    clean_text TEXT,
    language TEXT,
    fingerprint TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    deleted_at TEXT,
    purge_after TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_deleted ON documents(deleted_at) WHERE deleted_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_purge ON documents(purge_after) WHERE purge_after IS NOT NULL;

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

-- FTS5 for chunk search
CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
    content, section_title, content='document_chunks', content_rowid='id'
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
