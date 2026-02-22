# Data Schemas

## Redis Topics (configurations)
The Redis configuration is defined in `configs/connections.json`.  Current topic queue names:
```json
{
  "chat_history": "llm-chat",
  "prompt_queue": "prompt-queue",
  "memory_chunker": "memory-chunker-queue",
  "episodic_memory": "episodic-memory-queue"
}
```
```
"topics": {
    "chat_history": "llm-chat",
    "prompt_queue": "prompt-queue",
    "memory_chunker": "memory-chunker-queue",
    "episodic_memory": "episodic-memory-queue"
}
```
Each key maps to a queue name that the worker services consume from.

## PostgreSQL – Episodes Table
The `episodic_storage_service.py` inserts rows into the `episodes` table. The table schema (partial) is:
```sql
CREATE TABLE episodes (
    id UUID PRIMARY KEY,
    intent JSONB NOT NULL,
    context JSONB NOT NULL,
    action TEXT NOT NULL,
    emotion JSONB NOT NULL,
    outcome TEXT NOT NULL,
    gist TEXT NOT NULL,
    salience FLOAT NOT NULL,
    freshness FLOAT NOT NULL,
    embedding vector(768),
    topic TEXT NOT NULL,
    exchange_id TEXT,
    activation_score FLOAT DEFAULT 1.0,
    salience_factors JSONB NOT NULL,
    open_loops JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_accessed_at TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    deleted_at TIMESTAMP
);
```

### Field Descriptions

**Core Episode Content:**
- `intent` (JSONB) – Intent structure: `{"type": "exploration|...", "direction": "open-ended|..."}`
- `context` (JSONB) – Context: `{"situational": "...", "conversational": "...", "constraints": [...]}`
- `action` (TEXT) – Concrete actions taken or discussed
- `emotion` (JSONB) – Emotion: `{"type": "...", "valence": "positive|...", "intensity": "low|...", "arc": "..."}`
- `outcome` (TEXT) – Result or conclusion
- `gist` (TEXT) – Concise 1-3 sentence summary

**Salience & Retrieval:**
- `salience` (FLOAT) – Computed [0.1,1.0]: Base = `0.4·novelty + 0.4·emotional + 0.2·commitment`, then `×1.25` if unresolved. **Reconsolidation**: Retrieved episodes get boosted by 0.2 (configurable)
- `freshness` (FLOAT) – Initially salience; dynamic at retrieval: `e^(-decay × (1-salience) × hours)`
- `salience_factors` (JSONB) – LLM factors (0-3 scale): `{"novelty": 2, "emotional": 2, "commitment": 1, "unresolved": true}`
- `open_loops` (JSONB) – Array of unresolved items: `["Question X remains unresolved", ...]`

**Search & Tracking:**
- `embedding` (vector) – 768-dim semantic vector for similarity search
- `topic` (TEXT) – Conversation topic
- `activation_score` (FLOAT) – ACT-R activation score
- `last_accessed_at` (TIMESTAMP) – Last retrieval time (for freshness decay)
- `access_count` (INTEGER) – Number of accesses

## PostgreSQL – Threads Table

Conversations are stored as Redis-backed threads with metadata in PostgreSQL.

**Threads Table:**
```sql
CREATE TABLE threads (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    title TEXT,
    status TEXT DEFAULT 'active',  -- active, archived, deleted
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_message_at TIMESTAMP,
    expires_at TIMESTAMP
);
```

**Fields**:
- `id`: Unique thread identifier
- `topic`: Active conversation topic (used for memory scoping)
- `metadata`: Thread state including confidence, classifier info
- `expires_at`: Thread expiry time (managed by thread_expiry_service)

Conversation history is stored in Redis via `ThreadConversationService` with a 24h TTL. Thread metadata is persisted in PostgreSQL for durability.

## PostgreSQL – Lists Tables

Three tables provide deterministic list management with full history (`list_service.py`).

**`lists`** — List containers
```sql
CREATE TABLE lists (
    id          TEXT        PRIMARY KEY,           -- 8-char hex
    user_id     TEXT        NOT NULL DEFAULT 'primary',
    name        TEXT        NOT NULL,
    list_type   TEXT        NOT NULL DEFAULT 'checklist',
    metadata    JSONB       NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at  TIMESTAMPTZ                        -- soft delete
);
-- Unique name per user (case-insensitive, active lists only)
CREATE UNIQUE INDEX idx_lists_user_name_unique ON lists (user_id, lower(name)) WHERE deleted_at IS NULL;
```

**`list_items`** — Items within lists
```sql
CREATE TABLE list_items (
    id          TEXT        PRIMARY KEY,           -- 8-char hex
    list_id     TEXT        NOT NULL REFERENCES lists(id),
    content     TEXT        NOT NULL,
    checked     BOOLEAN     NOT NULL DEFAULT FALSE,
    position    INTEGER     NOT NULL DEFAULT 0,
    metadata    JSONB       NOT NULL DEFAULT '{}',
    added_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    removed_at  TIMESTAMPTZ                        -- soft delete (preserves history)
);
```

**`list_events`** — Audit log for history queries
```sql
CREATE TABLE list_events (
    id           TEXT        PRIMARY KEY,          -- 8-char hex
    list_id      TEXT        NOT NULL REFERENCES lists(id),
    event_type   TEXT        NOT NULL,             -- item_added, item_removed, item_checked, item_unchecked, list_created, list_cleared, list_deleted, list_renamed
    item_content TEXT,
    details      JSONB       NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

**Design notes:**
- Items are soft-deleted (`removed_at`) rather than hard-deleted — history is preserved for temporal reasoning
- Re-adding a previously removed item restores the original row (clears `removed_at`) instead of inserting a duplicate
- `lists.updated_at` is touched on every item mutation for recency-based resolution
- Name resolution is case-insensitive; `list_service._resolve_list()` tries exact ID first, then name match
- Context injection: `list_service.get_lists_for_prompt()` formats a compact summary injected as `{{active_lists}}` into all four mode prompts (RESPOND, ACT, CLARIFY, ACKNOWLEDGE)

## PostgreSQL – Additional Core Tables

### Scheduled Items
```sql
CREATE TABLE scheduled_items (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    scheduled_for TIMESTAMP NOT NULL,
    recurrence TEXT,  -- none, daily, weekly, monthly
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Goals
```sql
CREATE TABLE goals (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'active',  -- active, completed, abandoned
    progress FLOAT DEFAULT 0.0,
    target_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Autobiography
```sql
CREATE TABLE autobiography (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    sections JSONB,  -- identity, values, patterns, etc.
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Routing Decisions (Audit Trail)
```sql
CREATE TABLE routing_decisions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    selected_mode TEXT NOT NULL,
    scores JSONB NOT NULL,
    signals JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Semantic Concepts & Relationships
```sql
CREATE TABLE semantic_concepts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    definition TEXT,
    embedding vector(768),
    strength FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE semantic_relationships (
    id TEXT PRIMARY KEY,
    concept_a_id TEXT NOT NULL REFERENCES semantic_concepts(id),
    concept_b_id TEXT NOT NULL REFERENCES semantic_concepts(id),
    relationship_type TEXT NOT NULL,
    strength FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### User Traits
```sql
CREATE TABLE user_traits (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    decay_rate FLOAT DEFAULT 0.05,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Providers & Tool Configs
```sql
CREATE TABLE providers (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    type TEXT NOT NULL,  -- ollama, openai, anthropic, gemini
    endpoint TEXT,
    config JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE job_provider_assignments (
    id TEXT PRIMARY KEY,
    job_name TEXT NOT NULL,
    provider_id TEXT NOT NULL REFERENCES providers(id),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE tool_configs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    config JSONB,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Master Account
```sql
CREATE TABLE master_account (
    id TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    encryption_key TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Triage Calibration
```sql
CREATE TABLE triage_calibration (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    score FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```
