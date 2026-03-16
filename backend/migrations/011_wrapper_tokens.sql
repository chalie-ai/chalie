-- Wrapper authentication tokens for external programmatic access.
-- External wrappers (IDE, trading terminal, etc.) authenticate via bearer tokens
-- instead of cookie sessions.  Token hashes are stored, never raw tokens.

CREATE TABLE IF NOT EXISTS wrapper_tokens (
    id TEXT PRIMARY KEY,                        -- uuid4
    name TEXT NOT NULL,                         -- "IDE Wrapper", "Trading Bot"
    token_hash TEXT NOT NULL UNIQUE,            -- SHA-256 of bearer token
    wrapper_id TEXT NOT NULL UNIQUE,            -- stable wrapper identifier
    capabilities TEXT NOT NULL DEFAULT '{}',   -- JSON: {signals:[], intents:[]}
    permissions TEXT NOT NULL DEFAULT '{}',    -- JSON: {query:[], update:[], broadcast:bool}
    metadata TEXT NOT NULL DEFAULT '{}',       -- JSON: {version, description}
    last_seen_at TEXT,
    created_at TEXT NOT NULL,
    revoked_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_wrapper_tokens_hash
    ON wrapper_tokens(token_hash)
    WHERE revoked_at IS NULL;
