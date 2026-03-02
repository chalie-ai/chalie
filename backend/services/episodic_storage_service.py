"""
Episodic Storage Service - CRUD operations for episodes.
Responsibility: Storage layer only (SRP).
"""

import json
import uuid
from datetime import datetime
from typing import Optional
from services.database_service import DatabaseService
import logging


class EpisodicStorageService:
    """Manages episode storage and retrieval operations."""

    def __init__(self, database_service: DatabaseService):
        self.db_service = database_service

    def store_episode(self, episode_data: dict) -> str:
        """
        Store a new episode in the database.

        Returns:
            UUID of the created episode
        """
        required_fields = ['intent', 'context', 'action', 'emotion', 'outcome',
                          'gist', 'salience', 'freshness', 'topic']
        for field in required_fields:
            if field not in episode_data:
                raise ValueError(f"Missing required field: {field}")

        try:
            episode_id = str(uuid.uuid4())
            embedding = episode_data.get('embedding')

            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO episodes (
                        id, intent, context, action, emotion, outcome, gist,
                        salience, freshness, topic, exchange_id,
                        activation_score, salience_factors, open_loops
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    episode_id,
                    json.dumps(episode_data['intent']),
                    json.dumps(episode_data['context']),
                    episode_data['action'],
                    json.dumps(episode_data['emotion']),
                    episode_data['outcome'],
                    episode_data['gist'],
                    episode_data['salience'],
                    episode_data['freshness'],
                    episode_data['topic'],
                    episode_data.get('exchange_id'),
                    1.0,
                    json.dumps(episode_data.get('salience_factors', {})),
                    json.dumps(episode_data.get('open_loops', []))
                ))

                # Insert embedding into vec table if available
                if embedding is not None:
                    self._store_embedding(conn, episode_id, embedding)

                cursor.close()

                logging.info(f"Stored episode {episode_id} for topic '{episode_data['topic']}'")

                # Notify curiosity pursuit service for conversational reinforcement
                try:
                    from services.curiosity_pursuit_service import CuriosityPursuitService
                    CuriosityPursuitService().on_new_episode(episode_data)
                except Exception:
                    pass  # Non-fatal

                return episode_id

        except Exception as e:
            logging.error(f"Failed to store episode: {e}")
            raise

    def _store_embedding(self, conn, episode_id: str, embedding):
        """Store embedding in the companion vec table."""
        try:
            import struct
            if isinstance(embedding, (list, tuple)):
                blob = struct.pack(f'{len(embedding)}f', *embedding)
            elif isinstance(embedding, bytes):
                blob = embedding
            else:
                blob = embedding

            # Get the rowid of the episode for vec table linking
            cursor = conn.cursor()
            cursor.execute("SELECT rowid FROM episodes WHERE id = ?", (episode_id,))
            row = cursor.fetchone()
            if row:
                rowid = row[0]
                cursor.execute(
                    "INSERT OR REPLACE INTO episodes_vec(rowid, embedding) VALUES (?, ?)",
                    (rowid, blob)
                )
            cursor.close()
        except Exception as e:
            logging.warning(f"Failed to store episode embedding: {e}")

    def update_episode(self, episode_id: str, updates: dict) -> bool:
        """Update an existing episode."""
        if not updates:
            return True

        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                set_clauses = []
                values = []
                embedding = None

                for key, value in updates.items():
                    if key == 'embedding':
                        embedding = value
                        continue
                    if key in ['intent', 'context', 'emotion', 'salience_factors', 'open_loops']:
                        set_clauses.append(f"{key} = ?")
                        values.append(json.dumps(value))
                    else:
                        set_clauses.append(f"{key} = ?")
                        values.append(value)

                set_clauses.append("updated_at = datetime('now')")
                values.append(episode_id)

                query = f"UPDATE episodes SET {', '.join(set_clauses)} WHERE id = ?"
                cursor.execute(query, values)

                rows_updated = cursor.rowcount

                # Update embedding if provided
                if embedding is not None:
                    self._store_embedding(conn, episode_id, embedding)

                cursor.close()

                logging.info(f"Updated episode {episode_id}")
                return rows_updated > 0

        except Exception as e:
            logging.error(f"Failed to update episode: {e}")
            return False

    def soft_delete_episode(self, episode_id: str) -> bool:
        """Soft delete an episode (set deleted_at timestamp)."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE episodes
                    SET deleted_at = datetime('now')
                    WHERE id = ? AND deleted_at IS NULL
                """, (episode_id,))

                rows_deleted = cursor.rowcount
                cursor.close()

                if rows_deleted > 0:
                    logging.info(f"Soft deleted episode {episode_id}")
                    return True
                else:
                    logging.warning(f"Episode {episode_id} not found or already deleted")
                    return False

        except Exception as e:
            logging.error(f"Failed to soft delete episode: {e}")
            return False

    def get_episode_by_id(self, episode_id: str) -> Optional[dict]:
        """Retrieve an episode by ID."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT id, intent, context, action, emotion, outcome, gist,
                           salience, freshness, topic, exchange_id,
                           created_at, updated_at, last_accessed_at, access_count,
                           activation_score, salience_factors, open_loops
                    FROM episodes
                    WHERE id = ? AND deleted_at IS NULL
                """, (episode_id,))

                row = cursor.fetchone()
                cursor.close()

                if not row:
                    return None

                # Update access tracking
                self._update_activation_score(episode_id)

                episode = {
                    'id': str(row[0]),
                    'intent': row[1],
                    'context': row[2],
                    'action': row[3],
                    'emotion': row[4],
                    'outcome': row[5],
                    'gist': row[6],
                    'salience': row[7],
                    'freshness': row[8],
                    'topic': row[9],
                    'exchange_id': row[10],
                    'created_at': row[11],
                    'updated_at': row[12],
                    'last_accessed_at': row[13],
                    'access_count': row[14],
                    'activation_score': row[15],
                    'salience_factors': row[16] if len(row) > 16 else {},
                    'open_loops': row[17] if len(row) > 17 else []
                }

                return episode

        except Exception as e:
            logging.error(f"Failed to get episode by ID: {e}")
            return None

    def _update_activation_score(self, episode_id: str):
        """
        Update activation score based on access frequency and recency.
        Follows ACT-R memory activation model.
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                # Increment access count and update last_accessed_at
                cursor.execute("""
                    UPDATE episodes
                    SET access_count = access_count + 1,
                        last_accessed_at = datetime('now')
                    WHERE id = ?
                """, (episode_id,))

                # Recalculate activation score
                cursor.execute("""
                    UPDATE episodes
                    SET activation_score = 1.0
                        + (access_count * 0.1)
                        + CASE
                            WHEN last_accessed_at IS NOT NULL THEN
                                (1.0 / (1.0 + (CAST(strftime('%s', 'now') AS REAL) - CAST(strftime('%s', last_accessed_at) AS REAL)) / 86400.0))
                            ELSE 0
                          END
                    WHERE id = ?
                """, (episode_id,))

                cursor.close()

        except Exception as e:
            logging.error(f"Failed to update activation score: {e}")
