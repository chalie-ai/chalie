-- Migration 005: add reasoning column to routing_decisions
-- Stores triage effort annotation e.g. "[effort:trivial]"
ALTER TABLE routing_decisions ADD COLUMN reasoning TEXT;
