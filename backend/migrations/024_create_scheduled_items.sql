CREATE TABLE IF NOT EXISTS scheduled_items (
    id                  TEXT        PRIMARY KEY,
    item_type           TEXT        NOT NULL DEFAULT 'reminder',
    message             TEXT        NOT NULL CHECK (char_length(message) <= 1000),
    due_at              TIMESTAMPTZ NOT NULL,
    recurrence          TEXT,
    window_start        TEXT,       -- HH:MM — for hourly recurrence window (e.g. '09:00')
    window_end          TEXT,       -- HH:MM — for hourly recurrence window (e.g. '17:00')
    status              TEXT        NOT NULL DEFAULT 'pending',
    topic               TEXT,
    created_by_session  TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_fired_at       TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_scheduled_items_pending
    ON scheduled_items(due_at)
    WHERE status = 'pending';
