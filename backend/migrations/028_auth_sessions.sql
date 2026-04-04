-- Durable auth sessions — survive MemoryStore wipes (container restart).
CREATE TABLE IF NOT EXISTS auth_sessions (
    token TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires ON auth_sessions(expires_at);
