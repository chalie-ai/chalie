-- Migration 028: Create lists tables for deterministic list management
CREATE TABLE IF NOT EXISTS lists (
    id          TEXT        PRIMARY KEY,
    user_id     TEXT        NOT NULL DEFAULT 'primary',
    name        TEXT        NOT NULL,
    list_type   TEXT        NOT NULL DEFAULT 'checklist',
    metadata    JSONB       NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at  TIMESTAMPTZ
);

-- Unique list name per user (case-insensitive, active only)
CREATE UNIQUE INDEX IF NOT EXISTS idx_lists_user_name_unique
    ON lists (user_id, lower(name))
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_lists_user_active
    ON lists (user_id)
    WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS list_items (
    id          TEXT        PRIMARY KEY,
    list_id     TEXT        NOT NULL REFERENCES lists(id),
    content     TEXT        NOT NULL,
    checked     BOOLEAN     NOT NULL DEFAULT FALSE,
    position    INTEGER     NOT NULL DEFAULT 0,
    metadata    JSONB       NOT NULL DEFAULT '{}',
    added_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    removed_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_list_items_active
    ON list_items (list_id)
    WHERE removed_at IS NULL;

CREATE TABLE IF NOT EXISTS list_events (
    id           TEXT        PRIMARY KEY,
    list_id      TEXT        NOT NULL REFERENCES lists(id),
    event_type   TEXT        NOT NULL,
    item_content TEXT,
    details      JSONB       NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_list_events_list
    ON list_events (list_id, created_at DESC);
