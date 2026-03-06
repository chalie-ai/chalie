-- Watched folders: monitored filesystem directories for automatic document ingestion.

CREATE TABLE IF NOT EXISTS watched_folders (
    id TEXT PRIMARY KEY,
    folder_path TEXT NOT NULL UNIQUE,
    label TEXT,
    source_type TEXT DEFAULT 'filesystem',
    enabled INTEGER DEFAULT 1,
    file_patterns TEXT DEFAULT '["*"]',
    ignore_patterns TEXT DEFAULT '[".git","node_modules","__pycache__","build","dist",".DS_Store","Thumbs.db"]',
    recursive INTEGER DEFAULT 1,
    scan_interval INTEGER DEFAULT 300,
    last_scan_at TEXT,
    last_scan_files INTEGER DEFAULT 0,
    last_scan_error TEXT,
    source_config TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_watched_folders_enabled
    ON watched_folders(enabled) WHERE enabled = 1;

-- idx_documents_watched_folder is created by the idempotent column migration
-- in database_service.py (after watched_folder_id column is added to documents).
