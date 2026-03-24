"""
Goal-Autobiography Bridge — Bidirectional integration.

Direction B: Autobiography → Goals
Scans autobiography narrative to derive:
  - parent_motives: which CORE_MOTIVES align with the goal
  - identity_links: which identity vector dimensions are relevant

Called:
  1. On goal creation (create_goal post-hook)
  2. On autobiography synthesis completion (post-synthesis hook)
  3. During drift engine idle cycle (periodic refresh)
"""

import json
import logging
from typing import List, Dict, Optional

from services.time_utils import utc_now

logger = logging.getLogger(__name__)
LOG_PREFIX = "[GOAL-AUTOBIO BRIDGE]"

IDENTITY_DIMS = ['curiosity', 'assertiveness', 'warmth', 'playfulness', 'skepticism', 'emotional_intensity']

# Sentence-length motive descriptions yield more reliable embedding similarity than
# single words because sentence embeddings encode richer semantic context.
# Threshold is lower (0.35 vs 0.5) because sentence-to-sentence cosine similarity
# has a smaller absolute range than word-to-word similarity.
MOTIVE_SENTENCES = {
    'coherence': 'Consistency, making sense of the world, resolving contradictions',
    'truthfulness': 'Honesty, accuracy, integrity, and factual correctness',
    'competence_growth': 'Building skills, learning, mastery, and professional growth',
    'relationship_preservation': 'Social connection, community, relationships, and being part of a group',
    'initiative': 'Making independent choices, self-direction, taking action proactively',
}
MOTIVE_SIM_THRESHOLD = 0.35


def compute_goal_alignment(
    goal_description: str,
    autobiography_narrative: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Compute parent_motives and identity_links for a goal.

    Uses embedding similarity between goal description and:
      - CORE_MOTIVE keywords
      - Identity vector dimension names
      - Autobiography Values & Goals / Long Term Themes sections

    Deterministic embedding comparison, no LLM.

    Returns:
        {'parent_motives': [...], 'identity_links': [...]}
    """
    parent_motives = []
    identity_links = []

    if not autobiography_narrative:
        try:
            from services.autobiography_service import AutobiographyService
            from services.database_service import get_shared_db_service
            auto = AutobiographyService(get_shared_db_service())
            narrative = auto.get_current_narrative()
            autobiography_narrative = narrative.get('narrative', '') if narrative else ''
        except Exception:
            return {'parent_motives': [], 'identity_links': []}

    try:
        from services.embedding_service import get_embedding_service
        import numpy as np

        emb_service = get_embedding_service()
        goal_emb = emb_service.generate_embedding_np(goal_description)

        # Match against CORE_MOTIVES using sentence-level embeddings.
        # Sentences encode richer semantic context than single words, giving
        # more reliable similarity scores. Threshold is lower (MOTIVE_SIM_THRESHOLD)
        # because sentence-to-sentence cosine similarity has a smaller absolute range.
        for motive_name, motive_sentence in MOTIVE_SENTENCES.items():
            motive_emb = emb_service.generate_embedding_np(motive_sentence)
            sim = float(np.dot(goal_emb, motive_emb) / (
                np.linalg.norm(goal_emb) * np.linalg.norm(motive_emb) + 1e-8
            ))
            if sim > MOTIVE_SIM_THRESHOLD:
                parent_motives.append(motive_name)

        # Match against identity vector dimensions
        for dim in IDENTITY_DIMS:
            dim_emb = emb_service.generate_embedding_np(dim.replace('_', ' '))
            sim = float(np.dot(goal_emb, dim_emb) / (
                np.linalg.norm(goal_emb) * np.linalg.norm(dim_emb) + 1e-8
            ))
            if sim > 0.45:
                identity_links.append(dim)

        # Scan autobiography for value alignment
        if autobiography_narrative and autobiography_narrative.strip():
            from services.autobiography_delta_service import _extract_sections
            sections = _extract_sections(autobiography_narrative)
            values_text = sections.get('values_and_goals', '')
            themes_text = sections.get('long_term_themes', '')
            combined = f"{values_text} {themes_text}".strip()

            if combined:
                combined_emb = emb_service.generate_embedding_np(combined[:2000])
                alignment_score = float(np.dot(goal_emb, combined_emb) / (
                    np.linalg.norm(goal_emb) * np.linalg.norm(combined_emb) + 1e-8
                ))
                if alignment_score > 0.6 and 'coherence' not in parent_motives:
                    parent_motives.append('coherence')

    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Alignment computation failed: {e}")

    return {
        'parent_motives': parent_motives[:3],
        'identity_links': identity_links[:3],
    }


def refresh_all_goals() -> int:
    """
    Refresh parent_motives and identity_links for all active goals.

    Called post-synthesis and during drift idle cycle.

    Returns:
        Number of goals updated
    """
    try:
        from services.goal_ecology_service import GoalEcologyService
        from services.autobiography_service import AutobiographyService
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        auto = AutobiographyService(db)
        narrative_data = auto.get_current_narrative()
        narrative = narrative_data.get('narrative', '') if narrative_data else ''

        if not narrative:
            return 0

        ecology = GoalEcologyService()
        goals = ecology.get_active_stack(limit=20)

        updated = 0
        for g in goals:
            try:
                alignment = compute_goal_alignment(g['description'], narrative)

                if alignment['parent_motives'] or alignment['identity_links']:
                    ecology.update_goal(g['id'], {})  # trigger updated_at
                    with db.connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            UPDATE goals
                            SET parent_motives = ?, identity_links = ?, updated_at = ?
                            WHERE id = ?
                        """, (
                            json.dumps(alignment['parent_motives']),
                            json.dumps(alignment['identity_links']),
                            utc_now().isoformat(),
                            g['id'],
                        ))
                        cursor.close()
                    # Recalculate salience (now with populated motives/links)
                    ecology.recalculate_salience(g['id'])
                    updated += 1
            except Exception as ge:
                logger.debug(f"{LOG_PREFIX} Failed to refresh goal {g.get('id', '?')[:8]}: {ge}")

        if updated:
            logger.info(f"{LOG_PREFIX} Refreshed alignment for {updated} goals")
        return updated

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} refresh_all_goals failed: {e}")
        return 0
