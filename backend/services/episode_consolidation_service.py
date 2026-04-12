# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Episode Consolidation Service — creates super episodes from related episode clusters.

Runs periodically (called from decay engine). Finds clusters of 3-5 similar episodes
that haven't been consolidated, synthesises them into a new "super episode" via LLM,
and stores the result with consolidated_from referencing source episode IDs.

Source episodes are NEVER modified or deleted. The super episode sits above them in the graph.
Super episodes can themselves be consolidated further (graph, not tree).
"""

import json
import logging
import re
import uuid
from typing import Optional

from services.database_service import DatabaseService
from services.embedding_utils import pack_embedding

logger = logging.getLogger(__name__)


def _safe_json_list(value) -> list:
    if isinstance(value, list):
        return value
    try:
        result = json.loads(value) if value else []
        return result if isinstance(result, list) else []
    except Exception:
        return []


def _extract_json(text: str) -> str:
    text = text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    return match.group(1).strip() if match else text


class EpisodeConsolidationService:
    def __init__(self, db_service: DatabaseService, config: dict = None):
        self.db_service = db_service
        self.config = config or {}
        self._llm = None
        self._prompt_template = None

    def _get_llm(self):
        if self._llm is None:
            from services.config_service import ConfigService
            from services.llm_service import create_llm_service
            agent_config = ConfigService.resolve_agent_config("episode-consolidation")
            self._llm = create_llm_service(agent_config)
            self._prompt_template = ConfigService.get_agent_prompt("episode-consolidation")
        return self._llm

    def run_consolidation_cycle(self) -> int:
        """
        Find and consolidate episode clusters. Returns number of super episodes created.

        Steps:
        1. Fetch all unconsolidated episodes (not in any consolidated_from list)
           with retrieval_weight > 0.1
        2. For each, find similar episodes via vector KNN (cosine distance < 0.3)
        3. When 3-5 related episodes form a cluster: LLM synthesises into super episode
        4. Store super episode with consolidated_from = [source episode IDs]
        5. Source episodes remain untouched
        """
        try:
            consolidated_ids = self._fetch_already_consolidated_ids()
            candidates = self._fetch_consolidation_candidates(consolidated_ids)

            if not candidates:
                return 0

            created = 0
            processed_ids: set = set()

            for episode in candidates:
                episode_id = episode['id']
                if episode_id in processed_ids:
                    continue

                embedding = self._fetch_episode_embedding(episode['rowid'])
                if embedding is None:
                    continue

                similar = self._find_similar_episodes(
                    embedding, episode_id, consolidated_ids | processed_ids
                )

                if len(similar) < 2:
                    continue

                cluster = [episode] + similar
                cluster = cluster[:5]

                if len(cluster) < 3:
                    continue

                super_id = self._synthesise_and_store(cluster)
                if super_id:
                    created += 1
                    for ep in cluster:
                        processed_ids.add(ep['id'])

            if created > 0:
                logger.info(f"[CONSOLIDATION] Created {created} super episode(s)")

            return created

        except Exception as e:
            logger.error(f"[CONSOLIDATION] Cycle failed: {e}", exc_info=True)
            return 0

    def _fetch_already_consolidated_ids(self) -> set:
        """Return the set of episode IDs already referenced in any consolidated_from list."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT consolidated_from
                    FROM episodes
                    WHERE consolidated_from IS NOT NULL
                      AND consolidated_from != '[]'
                      AND deleted_at IS NULL
                """)
                rows = cursor.fetchall()
                cursor.close()

            ids: set = set()
            for (cf_json,) in rows:
                ids.update(_safe_json_list(cf_json))
            return ids
        except Exception as e:
            logger.error(f"[CONSOLIDATION] Failed to fetch consolidated IDs: {e}")
            return set()

    def _fetch_consolidation_candidates(self, exclude_ids: set) -> list:
        """Fetch unconsolidated episodes with retrieval_weight > 0.1."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, rowid, gist, context, intent, outcome, open_loops,
                           entities, goal_tags, emotional_valence, emotional_arousal,
                           salience, storage_strength, retrieval_weight
                    FROM episodes
                    WHERE deleted_at IS NULL
                      AND retrieval_weight > 0.1
                    ORDER BY retrieval_weight DESC
                """)
                rows = cursor.fetchall()
                cursor.close()

            result = []
            for row in rows:
                ep_id = str(row[0])
                if ep_id not in exclude_ids:
                    result.append({
                        'id': ep_id,
                        'rowid': row[1],
                        'gist': row[2] or '',
                        'context': row[3] or '',
                        'intent': row[4] or '',
                        'outcome': row[5] or '',
                        'open_loops': _safe_json_list(row[6]),
                        'entities': _safe_json_list(row[7]),
                        'goal_tags': _safe_json_list(row[8]),
                        'emotional_valence': row[9],
                        'emotional_arousal': row[10],
                        'salience': row[11] or 1.0,
                        'storage_strength': row[12] or 1.0,
                        'retrieval_weight': row[13] or 1.0,
                    })
            return result

        except Exception as e:
            logger.error(f"[CONSOLIDATION] Failed to fetch candidates: {e}")
            return []

    def _fetch_episode_embedding(self, rowid: int) -> Optional[bytes]:
        """Return the packed embedding bytes for an episode rowid, or None."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT embedding FROM episodes_vec WHERE rowid = ?", (rowid,)
                )
                row = cursor.fetchone()
                cursor.close()
                return row[0] if row else None
        except Exception as e:
            logger.warning(f"[CONSOLIDATION] Failed to fetch embedding for rowid {rowid}: {e}")
            return None

    def _find_similar_episodes(self, embedding: bytes, exclude_id: str,
                               exclude_ids: set) -> list:
        """KNN search for similar episodes (distance < 0.3), excluding given IDs."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT e.id, e.rowid, e.gist, e.context, e.intent, e.outcome,
                           e.open_loops, e.entities, e.goal_tags,
                           e.emotional_valence, e.emotional_arousal,
                           e.salience, e.storage_strength, e.retrieval_weight,
                           v.distance
                    FROM episodes e
                    JOIN episodes_vec v ON v.rowid = e.rowid
                    WHERE v.embedding MATCH ? AND k = 10
                      AND e.deleted_at IS NULL
                      AND e.retrieval_weight > 0.1
                    ORDER BY v.distance
                """, (embedding,))
                rows = cursor.fetchall()
                cursor.close()

            results = []
            for row in rows:
                ep_id = str(row[0])
                distance = row[14]
                if ep_id == exclude_id:
                    continue
                if ep_id in exclude_ids:
                    continue
                if distance is not None and distance >= 0.3:
                    continue
                results.append({
                    'id': ep_id,
                    'rowid': row[1],
                    'gist': row[2] or '',
                    'context': row[3] or '',
                    'intent': row[4] or '',
                    'outcome': row[5] or '',
                    'open_loops': _safe_json_list(row[6]),
                    'entities': _safe_json_list(row[7]),
                    'goal_tags': _safe_json_list(row[8]),
                    'emotional_valence': row[9],
                    'emotional_arousal': row[10],
                    'salience': row[11] or 1.0,
                    'storage_strength': row[12] or 1.0,
                    'retrieval_weight': row[13] or 1.0,
                })
            return results

        except Exception as e:
            logger.error(f"[CONSOLIDATION] KNN search failed: {e}")
            return []

    def _synthesise_and_store(self, cluster: list) -> Optional[str]:
        """LLM-synthesise cluster into a super episode and persist it. Returns new episode ID."""
        try:
            llm = self._get_llm()
        except Exception as e:
            logger.error(f"[CONSOLIDATION] Failed to initialise LLM: {e}")
            return None

        if not self._prompt_template:
            logger.error("[CONSOLIDATION] No prompt template loaded")
            return None

        source_text = self._format_cluster(cluster)
        prompt = self._prompt_template.replace('{{source_episodes}}', source_text)

        try:
            response = llm.send_message("", prompt)
        except Exception as e:
            logger.error(f"[CONSOLIDATION] LLM call failed: {e}")
            return None

        raw = response.text if response else ""
        if not raw:
            logger.warning("[CONSOLIDATION] LLM returned empty response")
            return None

        try:
            parsed = json.loads(_extract_json(raw))
        except json.JSONDecodeError:
            logger.error(f"[CONSOLIDATION] Failed to parse LLM JSON: {raw[:300]}")
            return None

        return self._store_super_episode(parsed, cluster)

    def _format_cluster(self, cluster: list) -> str:
        parts = []
        for i, ep in enumerate(cluster, 1):
            intent = ep.get('intent', '')
            if isinstance(intent, str):
                try:
                    intent_obj = json.loads(intent)
                    intent = intent_obj.get('direction', str(intent_obj))
                except Exception:
                    pass

            parts.append(
                f"Episode {i}:\n"
                f"  Gist: {ep['gist']}\n"
                f"  Context: {ep['context']}\n"
                f"  Intent: {intent}\n"
                f"  Outcome: {ep['outcome']}\n"
                f"  Open loops: {json.dumps(ep['open_loops'])}\n"
                f"  Entities: {json.dumps(ep['entities'])}\n"
                f"  Goal tags: {json.dumps(ep['goal_tags'])}\n"
                f"  Emotional valence: {ep.get('emotional_valence')}\n"
                f"  Emotional arousal: {ep.get('emotional_arousal')}"
            )
        return "\n\n".join(parts)

    def _store_super_episode(self, data: dict, cluster: list) -> Optional[str]:
        """Persist the synthesised super episode. Returns UUID or None on failure."""
        source_ids = [ep['id'] for ep in cluster]

        storage_strength = min(10.0, sum(ep.get('storage_strength', 1.0) for ep in cluster))
        max_salience = max(ep.get('salience', 1.0) for ep in cluster)
        salience = min(10, max_salience + 1)

        merged_entities = list({
            e for ep in cluster for e in ep.get('entities', [])
        })
        merged_goal_tags = list({
            t for ep in cluster for t in ep.get('goal_tags', [])
        })

        intent = data.get('intent', {'type': 'reflection', 'direction': 'consolidated memory'})
        context = data.get('context', '')
        action = data.get('action', '')
        emotion = data.get('emotion', {'valence': 'neutral', 'intensity': 'low'})
        outcome = data.get('outcome', '')
        gist = data.get('gist', '')
        salience_factors = data.get('salience_factors', {})
        open_loops = data.get('open_loops', [])
        emotional_valence = data.get('emotional_valence')
        emotional_arousal = data.get('emotional_arousal')

        if not gist:
            logger.warning("[CONSOLIDATION] LLM returned no gist — skipping")
            return None

        episode_id = str(uuid.uuid4())

        try:
            from services.embedding_service import get_embedding_service
            emb_service = get_embedding_service()
            embedding = emb_service.generate_embedding(gist)
        except Exception as e:
            logger.warning(f"[CONSOLIDATION] Embedding generation failed: {e}")
            embedding = None

        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO episodes (
                        id, intent, context, action, emotion, outcome, gist,
                        salience, channel,
                        salience_factors, open_loops,
                        transcript_ids, entities, goal_tags,
                        emotional_valence, emotional_arousal,
                        consolidated_from, storage_strength, retrieval_weight
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    episode_id,
                    json.dumps(intent),
                    json.dumps(context),
                    action,
                    json.dumps(emotion),
                    outcome,
                    gist,
                    salience,
                    'consolidated',
                    json.dumps(salience_factors),
                    json.dumps(open_loops),
                    json.dumps([]),
                    json.dumps(merged_entities),
                    json.dumps(merged_goal_tags),
                    emotional_valence,
                    emotional_arousal,
                    json.dumps(source_ids),
                    storage_strength,
                    1.0,
                ))

                if embedding is not None:
                    blob = pack_embedding(embedding)
                    if blob:
                        cursor.execute(
                            "SELECT rowid FROM episodes WHERE id = ?", (episode_id,)
                        )
                        row = cursor.fetchone()
                        if row:
                            cursor.execute(
                                "INSERT OR REPLACE INTO episodes_vec(rowid, embedding) VALUES (?, ?)",
                                (row[0], blob)
                            )

                # Sync FTS index (external-content table requires explicit insert)
                fts_row = cursor.execute(
                    "SELECT rowid FROM episodes WHERE id = ?", (episode_id,)
                ).fetchone()
                if fts_row:
                    conn.execute(
                        "INSERT INTO episodes_fts(rowid, gist, action) VALUES (?, ?, ?)",
                        (fts_row[0], gist, action),
                    )

                cursor.close()

            logger.info(
                f"[CONSOLIDATION] Super episode {episode_id} created from "
                f"{len(source_ids)} sources: {source_ids}"
            )
            return episode_id

        except Exception as e:
            logger.error(f"[CONSOLIDATION] Failed to store super episode: {e}")
            return None
