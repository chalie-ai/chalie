CREATE TABLE IF NOT EXISTS user_tool_preferences (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             TEXT NOT NULL DEFAULT 'default',
    tool_name           TEXT NOT NULL,
    usage_count         INTEGER DEFAULT 0,
    success_count       INTEGER DEFAULT 0,
    explicit_preference FLOAT DEFAULT 0,
    implicit_preference FLOAT DEFAULT 0,
    last_used_at        TIMESTAMPTZ,
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, tool_name)
);

CREATE INDEX IF NOT EXISTS idx_utp_user ON user_tool_preferences(user_id);
