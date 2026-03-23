-- Backfill topic_transcript from thread_exchanges for pre-transcript instances.
-- Guard: only runs if topic_transcript is empty and thread_exchanges has data.
-- Uses a subquery so the NOT EXISTS check is evaluated once for the entire INSERT.

INSERT INTO topic_transcript (topic, role, content, created_at)
SELECT topic, role, content, created_at FROM (
    SELECT topic, 'user' AS role, prompt_message AS content, created_at
    FROM thread_exchanges
    WHERE prompt_message IS NOT NULL AND prompt_message != ''
    UNION ALL
    SELECT topic, 'assistant' AS role, response_message AS content, created_at
    FROM thread_exchanges
    WHERE response_message IS NOT NULL AND response_message != ''
)
WHERE NOT EXISTS (SELECT 1 FROM topic_transcript LIMIT 1)
ORDER BY created_at;
