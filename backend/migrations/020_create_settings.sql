-- 020_create_settings.sql
-- Create settings table for application-wide configuration

CREATE TABLE IF NOT EXISTS settings (
    id SERIAL PRIMARY KEY,
    key VARCHAR(255) UNIQUE NOT NULL,
    value TEXT,
    value_type VARCHAR(50) DEFAULT 'string',
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_settings_key ON settings(key);

-- Insert default API key if not present
-- The API key will be auto-generated on first startup if not provided
INSERT INTO settings (key, value_type, description)
VALUES ('api_key', 'string', 'REST API authentication key (auto-generated on first startup if not set)')
ON CONFLICT (key) DO NOTHING;
