-- Tool configs: per-tool key-value configuration storage (credentials, endpoints, etc.)
CREATE TABLE IF NOT EXISTS tool_configs (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tool_name    TEXT NOT NULL,
    config_key   TEXT NOT NULL,
    config_value TEXT NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(tool_name, config_key)
);

CREATE INDEX IF NOT EXISTS idx_tool_configs_tool ON tool_configs(tool_name);
