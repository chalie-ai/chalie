-- Migration 043: Add vision capability flag to providers
-- Enables camera OCR and scanned PDF extraction via vision-capable LLMs

ALTER TABLE providers ADD COLUMN IF NOT EXISTS supports_vision BOOLEAN DEFAULT FALSE;
