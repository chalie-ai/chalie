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

## PostgreSQL – Conversations & Messages Tables

Conversations are stored as relational data in PostgreSQL (not files).

**Conversations Table:**
```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id),
    topic TEXT NOT NULL,
    title TEXT,
    status TEXT DEFAULT 'active',  -- active, archived, deleted
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    thread_id TEXT  -- Maps to conversation thread
);
```

**Messages Table:**
```sql
CREATE TABLE messages (
    id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    mode TEXT,  -- RESPOND, ACT, CLARIFY, ACKNOWLEDGE
    metadata JSONB,  -- routing_score, tool_calls, etc.
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Fields**:
- `topic`: Normalized conversation topic (used for memory scoping)
- `role`: Message author ('user' or 'assistant/chalie')
- `content`: Full message text
- `mode`: Cognitive mode used for Chalie's response
- `metadata`: Additional context (routing decisions, tool invocations, etc.)

These schemas are referenced by the services and workers throughout the codebase.
