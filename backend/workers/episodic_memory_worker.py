"""
Episodic Memory Worker — Utility functions and goal emergence detection.
"""

import json
import logging
import math
import re


def _extract_json(text: str) -> str:
    """
    Extract JSON from text, handling markdown code fences.

    Strips markdown fences (```json ... ``` or ``` ... ```), handles commentary
    before/after JSON, and multiple fenced blocks (takes first).
    """
    text = text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    return match.group(1).strip() if match else text


def _safe_json_load(text: str) -> dict | None:
    """
    Safely load JSON with graceful fallback for parse errors.

    Extracts JSON from markdown fences and attempts parsing.
    On failure, logs error and returns None instead of crashing.
    """
    cleaned = _extract_json(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logging.error("[EPISODIC] Failed to parse JSON from LLM output")
        logging.debug(f"[EPISODIC] Raw output: {cleaned[:500]}")
        return None


def _check_goal_emergence(episode_id: str, gist: str, embedding: list):
    """
    Check if a newly created episode clusters with existing episodes,
    suggesting an emergent goal.

    Flow:
    1. KNN search episode embedding against episodes_vec (top 10)
    2. Apply temporal decay + salience weighting
    3. If cluster found, check if an existing goal already covers these episodes
       - Yes → reinforce that goal + append this episode to derived_from
       - No → if cluster >= 2 episodes, run tiny LLM to assess goal emergence
    """
    if not embedding or not gist:
        return

    from services.embedding_utils import pack_embedding
    from services.database_service import DatabaseService
    from services.time_utils import utc_now, parse_utc

    db = DatabaseService()
    packed = pack_embedding(embedding)

    # 1. KNN search against episodes_vec
    with db.connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT e.id, e.gist, e.salience, e.created_at, v.distance
            FROM episodes_vec v
            JOIN episodes e ON e.id = v.rowid
            WHERE v.embedding MATCH ? AND k = 10
              AND e.deleted_at IS NULL
              AND e.id != ?
            ORDER BY v.distance
        """, (packed, episode_id))
        rows = cursor.fetchall()
        cursor.close()

    if not rows:
        return

    # 2. Score with temporal decay and salience
    now = utc_now()
    scored = []
    SIMILARITY_THRESHOLD = 0.60

    for row in rows:
        ep_id, ep_gist, salience, created_at_str, distance = row
        similarity = max(0.0, 1.0 - (distance or 1.0))
        if similarity < SIMILARITY_THRESHOLD:
            continue

        try:
            created_at = parse_utc(created_at_str)
            days_old = (now - created_at).total_seconds() / 86400
        except Exception:
            days_old = 7.0

        # Temporal decay: 14-day half-life
        decay = math.exp(-days_old / 14.0)
        sal_norm = (salience or 5) / 10.0
        score = similarity * sal_norm * decay

        scored.append({
            'id': ep_id,
            'gist': ep_gist,
            'similarity': similarity,
            'score': score,
        })

    if not scored:
        return

    scored.sort(key=lambda x: x['score'], reverse=True)
    cluster = scored[:5]
    cluster_episode_ids = [ep['id'] for ep in cluster]

    # 3. Check if an existing goal already covers any of these episodes
    from services.goal_ecology_service import GoalEcologyService
    ecology = GoalEcologyService()

    existing_goals = ecology.find_goals_by_episode_ids(cluster_episode_ids)
    if existing_goals:
        goal = existing_goals[0]
        ecology.add_evidence(
            goal_id=goal['id'],
            signal_type='episode_cluster',
            content=gist[:500],
            source=f'episode:{episode_id}',
            strength=0.8,
        )
        ecology.append_derived_from(goal['id'], episode_id)
        logging.info(
            f"[EPISODIC] Reinforced goal {goal['id'][:8]} with episode {episode_id[:8]}"
        )
        return

    # 4. No existing goal covers this cluster — need >= 2 episodes to synthesize
    if len(cluster) < 2:
        return

    from services.background_llm_queue import create_background_llm_proxy

    gists = [f"- {ep['gist']}" for ep in cluster]
    gists.append(f"- {gist}")
    gists_text = "\n".join(gists)

    llm = create_background_llm_proxy("goal-emergence")
    response = llm.send_message(
        system_prompt=(
            "You analyze episode clusters for emergent goals. "
            "An episode is a narrative summary of a conversation segment. "
            "Respond with EXACTLY this format:\n"
            "GOAL: <one-sentence goal description>\n"
            "CONFIDENCE: <low|medium|high>\n\n"
            "If no coherent goal emerges, respond with:\n"
            "GOAL: none\nCONFIDENCE: low"
        ),
        user_message=(
            f"Do these episodes suggest an emerging goal?\n\n{gists_text}"
        ),
    )

    if not response or not response.text:
        return

    text = response.text.strip()
    goal_line = None
    confidence = 'low'

    for line in text.split('\n'):
        line = line.strip()
        if line.upper().startswith('GOAL:'):
            goal_line = line[5:].strip()
        elif line.upper().startswith('CONFIDENCE:'):
            confidence = line[11:].strip().lower()

    if not goal_line or goal_line.lower() == 'none' or confidence == 'low':
        return

    derived_from_ids = cluster_episode_ids + [episode_id]
    goal = ecology.create_goal(
        description=goal_line,
        type='emergent',
        derived_from=derived_from_ids,
    )

    logging.info(
        f"[EPISODIC] Created emergent goal '{goal_line[:60]}' "
        f"(confidence={confidence}, episodes={len(derived_from_ids)})"
    )
