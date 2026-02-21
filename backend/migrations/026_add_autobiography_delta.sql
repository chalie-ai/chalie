-- Migration 026: Add delta_summary column to autobiography table
ALTER TABLE autobiography ADD COLUMN IF NOT EXISTS delta_summary JSONB DEFAULT NULL;
