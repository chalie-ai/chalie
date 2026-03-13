-- 006: Simplify user_traits — originally removed source/is_literal and collapsed categories.
--
-- This migration is now a no-op. The schema changes are folded into migration 002
-- (which must work on both fresh installs and upgrades). On fresh installs, schema.sql
-- already creates user_traits without source/is_literal. On existing installs that
-- already ran the original 006, this no-op is harmless.
SELECT 1;
