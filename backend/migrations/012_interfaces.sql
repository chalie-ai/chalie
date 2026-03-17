-- Interface registry — external interfaces that pair with Chalie via
-- bluetooth-style handshake. Each interface exposes capabilities (tools)
-- that get registered in the ACT loop.

-- Interface registry
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

-- Track which tools came from which interface
CREATE TABLE IF NOT EXISTS interface_tools (
    interface_id TEXT NOT NULL REFERENCES interfaces(id) ON DELETE CASCADE,
    tool_name TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    registered_at TEXT NOT NULL,
    PRIMARY KEY (interface_id, tool_name)
);

-- Pairing keys (short-lived, one-time use)
CREATE TABLE IF NOT EXISTS interface_pairing_keys (
    key_hash TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used_at TEXT
);
