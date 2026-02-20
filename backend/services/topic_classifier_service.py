"""
Deterministic topic classification using sentence embeddings.

Topics are semantic attractors in embedding space.
Classification is structural cognition, not stochastic generation.
"""

import json
import logging
import time
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

from services.database_service import DatabaseService, get_merged_db_config
from services.config_service import ConfigService
from services.embedding_service import EmbeddingService


logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768

# Singleton embedding service
_EMBED_SERVICE = None


def _get_embed_service():
    """Lazy-init singleton EmbeddingService."""
    global _EMBED_SERVICE
    if _EMBED_SERVICE is None:
        _EMBED_SERVICE = EmbeddingService()
        logger.info("[TOPIC CLASSIFIER] EmbeddingService initialized")
    return _EMBED_SERVICE


def generate_embedding(text: str) -> np.ndarray:
    """
    Generate embedding via unified EmbeddingService (sentence-transformers, 768-dim).

    Returns:
        numpy array of shape (768,) - L2-normalized embedding
    """
    try:
        service = _get_embed_service()
        return service.generate_embedding_np(text)
    except Exception as e:
        logger.error(f"[TOPIC CLASSIFIER] Embedding generation failed: {e}")
        return np.zeros(EMBEDDING_DIM)


class TopicClassifierService:
    """
    Deterministic topic classification using embeddings.

    Cognitive Parameters (adaptive - tuned every 24h):
    - SWITCH_THRESHOLD: Semantic boundary for topic switching
    - DECAY_CONSTANT: Working memory duration (personality parameter)
    - Weights: Balance between semantic, temporal, and salience signals
    """

    # Default cognitive parameters (overridden by adaptive tuner)
    DEFAULT_SWITCH_THRESHOLD = 0.65    # Semantic boundary (raised to reduce over-clamping)
    DEFAULT_DECAY_CONSTANT = 300       # Working memory duration (5 min)
    TOP_K_CANDIDATES = 5                # Consider top-k similar topics for switch scoring
    ROLLING_AVG_CAP = 20                 # Stop updating embedding after N messages (prevents centroid dilution)

    # Default switch scoring weights
    DEFAULT_W_SEMANTIC = 0.6           # Primary signal
    DEFAULT_W_FRESHNESS = 0.3          # Recency bias
    DEFAULT_W_SALIENCE = 0.1           # Tie-breaker
    RECENCY_BONUS = 0.15               # Similarity boost for recently-active topic

    def __init__(self):
        self.db = DatabaseService(get_merged_db_config())

        # Load adaptive weights (or use defaults)
        self._load_weights()

        # Topic naming is now deterministic (keyword extraction, no LLM)

    def _load_weights(self):
        """
        Load regulated parameters from stability regulator.

        Falls back to defaults if regulator hasn't run yet.
        """
        try:
            from services.topic_stability_regulator_service import TopicStabilityRegulator
            regulator = TopicStabilityRegulator()
            weights = regulator.get_current_parameters()

            self.SWITCH_THRESHOLD = weights['switch_threshold']
            self.DECAY_CONSTANT = weights['decay_constant']
            self.W_SEMANTIC = weights['w_semantic']
            self.W_FRESHNESS = weights['w_freshness']
            self.W_SALIENCE = weights['w_salience']

            logger.info(
                f"[TOPIC CLASSIFIER] Loaded adaptive weights: "
                f"threshold={self.SWITCH_THRESHOLD:.3f}, "
                f"decay={self.DECAY_CONSTANT:.0f}s, "
                f"w_sem={self.W_SEMANTIC:.2f}, "
                f"w_fresh={self.W_FRESHNESS:.2f}, "
                f"w_sal={self.W_SALIENCE:.2f}"
            )

        except Exception as e:
            logger.warning(f"[TOPIC CLASSIFIER] Could not load adaptive weights: {e}")
            logger.info("[TOPIC CLASSIFIER] Using default weights")

            self.SWITCH_THRESHOLD = self.DEFAULT_SWITCH_THRESHOLD
            self.DECAY_CONSTANT = self.DEFAULT_DECAY_CONSTANT
            self.W_SEMANTIC = self.DEFAULT_W_SEMANTIC
            self.W_FRESHNESS = self.DEFAULT_W_FRESHNESS
            self.W_SALIENCE = self.DEFAULT_W_SALIENCE

    def classify(self, message_text: str, recent_topic: str = None) -> Dict:
        """
        Classify message into existing topic or create new one.

        Two-stage process:
        1. Check if ANY topic exceeds similarity threshold
        2. If yes: rank top-k candidates by switch_score
        3. If no: create new topic

        Args:
            message_text: User's message
            recent_topic: Name of the recently-active topic (gets similarity bonus)

        Returns:
            {
                'topic': str,              # Topic name
                'confidence': float,       # Similarity to chosen topic (0-1)
                'switch_score': float,     # Combined switching cost
                'is_new_topic': bool,      # Whether a new topic was created
                'classification_time': float
            }
        """
        start_time = time.time()

        # Generate embedding for current message (already L2-normalized)
        t0 = time.time()
        current_embedding = generate_embedding(message_text)
        embedding_time = time.time() - t0

        # Estimate salience (cheap heuristic)
        current_salience = self._estimate_salience(message_text)

        # Fetch all existing topics
        t1 = time.time()
        existing_topics = self._fetch_all_topics()
        fetch_time = time.time() - t1

        logger.info(f"[TOPIC CLASSIFIER] Timing: embed={embedding_time:.3f}s, fetch={fetch_time:.3f}s")

        if not existing_topics:
            # No topics exist - create first one
            topic_name = self._generate_topic_name(message_text)
            self._create_topic(topic_name, current_embedding, current_salience)

            classification_time = time.time() - start_time
            logger.info(
                f"[TOPIC CLASSIFIER] Created first topic '{topic_name}' "
                f"in {classification_time:.3f}s"
            )

            return {
                'topic': topic_name,
                'confidence': 1.0,
                'switch_score': 0.0,
                'is_new_topic': True,
                'classification_time': classification_time
            }

        # Stage 1: Calculate similarities to all topics
        t2 = time.time()
        similarities = []
        for topic in existing_topics:
            topic_embedding = np.array(topic['rolling_embedding'])
            # Both vectors are L2-normalized, so dot product = cosine similarity
            similarity = float(np.dot(current_embedding, topic_embedding))
            similarities.append((topic, similarity))
        similarity_time = time.time() - t2

        # Apply recency bonus to the recently-active topic
        if recent_topic:
            similarities = [
                (topic, sim + self.RECENCY_BONUS if topic['name'] == recent_topic else sim)
                for topic, sim in similarities
            ]

        # Check threshold: Is ANY topic plausible?
        best_similarity = max(sim for _, sim in similarities)

        logger.info(f"[TOPIC CLASSIFIER] Calculated {len(similarities)} similarities in {similarity_time:.3f}s"
                     f"{f' (recency bonus applied to {recent_topic})' if recent_topic else ''}")

        if best_similarity < self.SWITCH_THRESHOLD:
            # No plausible match - create new topic
            topic_name = self._generate_topic_name(message_text)
            self._create_topic(topic_name, current_embedding, current_salience)

            classification_time = time.time() - start_time
            logger.info(
                f"[TOPIC CLASSIFIER] New topic '{topic_name}' "
                f"(best_sim={best_similarity:.3f} < threshold={self.SWITCH_THRESHOLD}) "
                f"in {classification_time:.3f}s"
            )

            return {
                'topic': topic_name,
                'confidence': best_similarity,
                'switch_score': 1.0 - best_similarity,
                'is_new_topic': True,
                'classification_time': classification_time
            }

        # Stage 2: Rank top-k candidates by switch_score
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:self.TOP_K_CANDIDATES]

        # Calculate switch scores for top-k
        now = datetime.now(timezone.utc)
        scored_candidates = []

        for topic, similarity in top_k:
            # Freshness penalty: how long since last update?
            time_delta = (now - topic['last_updated']).total_seconds()
            freshness_penalty = np.tanh(time_delta / self.DECAY_CONSTANT)

            # Salience difference: is message importance aligned?
            salience_diff = abs(current_salience - topic['avg_salience'])

            # Switch score: lower is better
            switch_score = (
                self.W_SEMANTIC * (1 - similarity) +
                self.W_FRESHNESS * freshness_penalty +
                self.W_SALIENCE * salience_diff
            )

            scored_candidates.append((topic, similarity, switch_score))

        # Pick candidate with LOWEST switch_score
        best_topic, best_sim, best_switch_score = min(
            scored_candidates,
            key=lambda x: x[2]
        )

        # Update topic with new message (capped to prevent centroid dilution)
        self._update_topic(
            best_topic['name'],
            current_embedding,
            current_salience,
            best_topic['message_count']
        )


        classification_time = time.time() - start_time

        logger.info(
            f"[TOPIC CLASSIFIER] Matched topic '{best_topic['name']}' "
            f"(sim={best_sim:.3f}, switch={best_switch_score:.3f}) "
            f"in {classification_time:.3f}s"
        )

        return {
            'topic': best_topic['name'],
            'confidence': best_sim,
            'switch_score': best_switch_score,
            'is_new_topic': False,
            'classification_time': classification_time
        }

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """
        L2-normalize embedding vector.

        Critical for cosine similarity stability.
        Averaging drifts magnitude without normalization.
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def _estimate_salience(self, text: str) -> float:
        """
        Cheap heuristic for message salience.

        This is a tie-breaker, not a core signal.
        Don't over-engineer.
        """
        score = 0.5  # Baseline

        # Length (longer = more substantial)
        if len(text) > 100:
            score += 0.2
        elif len(text) > 50:
            score += 0.1

        # Questions (engagement signal)
        if '?' in text:
            score += 0.15

        # Imperatives (urgency keywords)
        urgency_words = ['urgent', 'asap', 'immediately', 'help', 'error', 'bug', 'broken', 'crash']
        if any(word in text.lower() for word in urgency_words):
            score += 0.15

        return min(1.0, score)  # Cap at 1.0

    def _fetch_all_topics(self) -> List[Dict]:
        """
        Fetch all topics from database.

        Returns:
            List of topic dicts with rolling_embedding as numpy array
        """
        query = """
            SELECT name, rolling_embedding, avg_salience, last_updated, message_count
            FROM topics
            ORDER BY last_updated DESC
        """

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()

            topics = []
            for row in rows:
                # pgvector may return as string or list depending on psycopg2 version
                embedding = row[1]
                if isinstance(embedding, str):
                    # Parse string representation: '[1.0, 2.0, ...]'
                    embedding = json.loads(embedding)

                topics.append({
                    'name': row[0],
                    'rolling_embedding': embedding,
                    'avg_salience': row[2],
                    'last_updated': row[3],
                    'message_count': row[4]
                })

            return topics

    # Common English stop words for keyword extraction
    STOP_WORDS = frozenset([
        'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your', 'he', 'she', 'it',
        'they', 'them', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
        'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
        'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
        'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
        'by', 'for', 'with', 'about', 'against', 'between', 'through', 'during',
        'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
        'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
        'just', 'don', 'should', 'now', 'tell', 'me', 'please', 'could', 'would',
        'like', 'know', 'think', 'want', 'need', 'help', 'let', 'make', 'get',
        'really', 'also', 'something', 'anything', 'everything', 'much', 'many',
    ])

    def _generate_topic_name(self, message_text: str) -> str:
        """
        Generate topic name using keyword extraction (no LLM).

        Deterministic, instant (~0ms), and reliable.
        Extracts 2-4 content words from the message.
        """
        import re

        # Clean and tokenize
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', message_text.lower())
        words = text.split()

        # Filter stop words, keep content words
        content_words = [w for w in words if w not in self.STOP_WORDS and len(w) > 1]

        if not content_words:
            return f"topic-{int(time.time())}"

        # Take first 2-4 content words
        topic_words = content_words[:min(4, max(2, len(content_words)))]
        topic_name = '-'.join(topic_words)

        logger.info(f"[TOPIC CLASSIFIER] Generated name '{topic_name}' (keyword extraction)")
        return topic_name

    def _create_topic(self, name: str, embedding: np.ndarray, salience: float):
        """
        Create new topic in database.

        Args:
            name: Topic name
            embedding: L2-normalized embedding (768-dim)
            salience: Initial salience score
        """
        embedding_list = embedding.tolist()

        query = """
            INSERT INTO topics (name, rolling_embedding, avg_salience, message_count)
            VALUES (%s, %s, %s, 1)
            ON CONFLICT (name) DO NOTHING
        """

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (name, embedding_list, salience))
            conn.commit()

        logger.info(f"[TOPIC CLASSIFIER] Created topic '{name}'")

    def _update_topic(self, name: str, new_embedding: np.ndarray, new_salience: float, old_count: int):
        """
        Update topic with new message using running average.

        Formula:
            avg = (old * n + new) / (n + 1)
            avg = avg / ||avg||  # L2-normalize to keep cosine stable

        Args:
            name: Topic name
            new_embedding: New message embedding (already normalized)
            new_salience: New message salience
            old_count: Previous message count
        """
        # Fetch current rolling embedding
        query_fetch = "SELECT rolling_embedding, avg_salience FROM topics WHERE name = %s"

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query_fetch, (name,))
            row = cursor.fetchone()

            if not row:
                logger.error(f"[TOPIC CLASSIFIER] Topic '{name}' not found for update")
                return

            # Handle string or list from pgvector
            embedding_data = row[0]
            if isinstance(embedding_data, str):
                embedding_data = json.loads(embedding_data)
            old_embedding = np.array(embedding_data)
            old_salience = row[1]

            # Running average for embedding (capped to prevent centroid dilution)
            n = min(old_count, self.ROLLING_AVG_CAP)
            new_avg_embedding = (old_embedding * n + new_embedding) / (n + 1)

            # L2-normalize to keep cosine similarity stable
            new_avg_embedding = self._normalize(new_avg_embedding)

            # Running average for salience (same cap)
            new_avg_salience = (old_salience * n + new_salience) / (n + 1)

            # Update database
            query_update = """
                UPDATE topics
                SET rolling_embedding = %s,
                    avg_salience = %s,
                    last_updated = NOW(),
                    message_count = message_count + 1
                WHERE name = %s
            """

            cursor.execute(query_update, (
                new_avg_embedding.tolist(),
                new_avg_salience,
                name
            ))
            conn.commit()

        logger.info(f"[TOPIC CLASSIFIER] Updated topic '{name}' (count: {old_count} -> {old_count + 1})")
