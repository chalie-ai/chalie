"""
ContradictionClassifierService — classification of memory conflicts.

Core question: "Can these two memories both be true simultaneously?"
Output: classification enum + confidence + recommended resolution.

Used by:
  - Ingestion detection subprocess (digest_worker)
  - Drift RECONCILE action (ReconcileAction)
  - Semantic consolidation post-check (SemanticConsolidationService)

Classifications (STATIC — changing these requires retraining the ONNX model):
  A: temporal_change   — old belief replaced by new one (job switch, relocation, etc.)
  B: true_contradiction — cannot both be true simultaneously
  C: context_dependent  — both true but in different contexts
  D: figurative         — one memory is non-literal (hyperbole, humor)
  E: compatible         — no conflict; they can coexist (likes Honda + likes Toyota)

ONNX MODEL CONTRACT
===================
This service has a companion ONNX classifier (trained in training/data/tasks/contradiction/).
The ONNX model is the primary classifier; the LLM path (_classify_pair_llm) is the
fallback when the ONNX model is unavailable or confidence is below threshold.

The ONNX model was trained on a specific input format and signal contract. Changes to
any of the following REQUIRE retraining the model:

  1. Input JSON field names: text_a, text_b, type_a, type_b, age_a_days, age_b_days,
     established_a, established_b
  2. Memory type vocabulary: incoming, trait, concept, episode
  3. The 5 classification labels and their A-E letter mapping
  4. The _is_established() thresholds and logic
  5. The prompt suffix format ("Options: A: ... Answer:")

See training/data/tasks/contradiction/SIGNALS.md for the full signal contract.
"""

import json
import logging
import re
import struct
import time
from typing import Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "[CONTRADICTION]"

# Similarity threshold above which we consider two memories topically related
_SIMILARITY_THRESHOLD = 0.75

# Maximum time to spend on ingestion detection (ms). Skip if exceeded.
_INGESTION_TIMEOUT_MS = 600

# Maximum candidate pairs to classify per ingestion call
_MAX_PAIRS_PER_INGESTION = 3

# Anti-duplicate: how many recently-created uncertainties to check before
# creating a new one for the same pair
_RECENT_UNCERTAINTY_WINDOW_DAYS = 7

_JSON_FENCE_RE = re.compile(r'```(?:json)?\s*\n?(.*?)\n?\s*```', re.DOTALL)


def _is_established(memory_type: str, meta: dict) -> bool:
    """
    Collapse confidence/reinforcement/access signals into a single boolean.

    A memory is "established" when there's strong evidence it's not noise:
      - trait: reinforced 3+ times (user has said this repeatedly)
      - concept: confidence >= 0.8 OR accessed 5+ times (well-grounded knowledge)
      - episode: always False (singular narrative events, never "established")
      - incoming: always False (just arrived, unverified)

    !! ONNX CONTRACT — RETRAINING REQUIRED IF CHANGED !!
    The ONNX contradiction classifier was trained with these exact thresholds.
    The model learned correlations between established=true/false and classification
    outcomes. Changing thresholds (e.g. reinforcement_count >= 5 instead of >= 3)
    will silently degrade accuracy without raising any error.

    To retrain: see training/data/tasks/contradiction/SIGNALS.md
    Mirrored in: training/data/tasks/contradiction/__init__.py::_synth_established()
    """
    if memory_type in ('incoming', 'episode'):
        return False
    if memory_type == 'trait':
        return meta.get('reinforcement_count', 1) >= 3
    if memory_type == 'concept':
        return meta.get('confidence', 0.5) >= 0.8 or meta.get('access_count', 0) >= 5
    return False


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith('{'):
        return json.loads(text)
    m = _JSON_FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1).strip())
    start = text.find('{')
    if start != -1:
        return json.loads(text[start:])
    raise json.JSONDecodeError("No JSON object found", text, 0)


def _unpack_embedding(blob) -> Optional[list]:
    """Unpack sqlite-vec binary blob to float list."""
    if blob is None:
        return None
    if isinstance(blob, (list, tuple)):
        return list(blob)
    if isinstance(blob, bytes):
        n = len(blob) // 4
        if n == 0:
            return None
        return list(struct.unpack(f'{n}f', blob))
    return None


def _cosine_similarity(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


_CLASSIFIER_SYSTEM_PROMPT = """You are an epistemic consistency checker for a cognitive memory system.

Given two memory records, decide whether they are contradictory.

Core question: **Can these both be true simultaneously for the same person?**
- "I work at Google" + "I work at Apple" → true_contradiction (you can't work at both as primary)
- "I like Honda" + "I like Toyota" → compatible (you can like both brands)
- "I worked at Google" + "I work at Apple" → temporal_change (career evolution)
- "I hate mornings" + "woke up at 5am to train" → context_dependent (both plausible)
- "I'm dead inside" + "I have strong emotions" → figurative (first is hyperbole)

Output JSON only:
{
  "classification": "temporal_change|true_contradiction|context_dependent|figurative|compatible",
  "confidence": 0.0-1.0,
  "temporal_signal": true|false,
  "reasoning": "one sentence explanation",
  "surface_context": "describe when in conversation this should be raised, or null if auto-resolve",
  "recommended_resolution": "auto_supersede|flag_response|background_queue|ignore"
}

Rules:
- When in doubt, prefer "compatible" over "true_contradiction"
- "temporal_change" requires clear evidence that one state was replaced by another
- "surface_context" should describe conversational context (e.g. "user discusses career or job changes"), not be generic
- Only set surface_context when classification is true_contradiction or context_dependent"""


class ContradictionClassifierService:
    """
    LLM-based contradiction detection across memory pairs.

    Constructor takes an optional db_service. When provided, ingestion
    detection and drift reconciliation use the DB for vector search.
    """

    def __init__(self, db_service=None):
        self.db = db_service

    # ── Public API ──────────────────────────────────────────────────────────

    def check_ingestion(self, text: str) -> Optional[dict]:
        """
        Run ingestion-time contradiction detection.

        Embeds the user message, vector-searches traits + concepts,
        and classifies any high-similarity divergent pairs found.

        Returns:
            dict with keys {classification, memory_a, memory_b, confidence,
                            temporal_signal, surface_context, reasoning}
            or None if no contradiction found or detection skipped.
        """
        if self.db is None:
            return None

        start = time.time()

        try:
            from services.embedding_service import get_embedding_service
            emb_service = get_embedding_service()
            embedding = emb_service.generate_embedding(text)
            if embedding is None:
                return None
        except Exception as e:
            logger.info(f"{LOG_PREFIX} Embedding failed for ingestion check: {e}")
            return None

        elapsed_ms = (time.time() - start) * 1000
        if elapsed_ms > _INGESTION_TIMEOUT_MS:
            logger.info(f"{LOG_PREFIX} Ingestion timeout after embedding ({elapsed_ms:.0f}ms)")
            return None

        pairs = self._find_candidate_pairs_ingestion(embedding, text, start)
        if not pairs:
            return None

        for mem_a, mem_b in pairs[:_MAX_PAIRS_PER_INGESTION]:
            elapsed_ms = (time.time() - start) * 1000
            if elapsed_ms > _INGESTION_TIMEOUT_MS:
                logger.info(f"{LOG_PREFIX} Ingestion timeout before classification")
                return None

            result = self._classify_pair_llm(
                mem_a['text'], mem_b['text'],
                context_hint=text,
                meta_a=mem_a.get('meta', {}),
                meta_b=mem_b.get('meta', {}),
            )
            if result is None:
                continue

            classification = result.get('classification', 'compatible')
            if classification == 'compatible' or classification == 'figurative':
                continue

            return {
                'classification': classification,
                'confidence': result.get('confidence', 0.5),
                'temporal_signal': result.get('temporal_signal', False),
                'reasoning': result.get('reasoning', ''),
                'surface_context': result.get('surface_context'),
                'recommended_resolution': result.get('recommended_resolution', 'background_queue'),
                'memory_a': mem_a,
                'memory_b': mem_b,
            }

        return None

    def check_concept_conflict(
        self,
        concept_name: str,
        concept_definition: str,
        existing: dict,
    ) -> Optional[dict]:
        """
        Check if a new concept conflicts with an existing one.

        Used by SemanticConsolidationService before storing a new concept.

        Returns:
            Classification dict (same schema as check_ingestion) or None.
        """
        meta_b = {
            'source': 'consolidation_existing',
            'confidence': existing.get('confidence', 0.5),
            'access_count': existing.get('access_count', 0),
            'created_at': existing.get('created_at'),
        }
        meta_b['established'] = _is_established('concept', meta_b)
        result = self._classify_pair_llm(
            f"{concept_name}: {concept_definition}",
            f"{existing.get('concept_name', '')}: {existing.get('definition', '')}",
            context_hint=None,
            meta_a={'source': 'consolidation_new', 'established': False},
            meta_b=meta_b,
        )
        if result is None:
            return None

        classification = result.get('classification', 'compatible')
        if classification in ('compatible', 'figurative'):
            return None

        return {
            'classification': classification,
            'confidence': result.get('confidence', 0.5),
            'temporal_signal': result.get('temporal_signal', False),
            'reasoning': result.get('reasoning', ''),
            'surface_context': result.get('surface_context'),
            'recommended_resolution': result.get('recommended_resolution', 'background_queue'),
            'memory_a': {'type': 'concept', 'id': None, 'text': f"{concept_name}: {concept_definition}"},
            'memory_b': {'type': 'concept', 'id': existing.get('id'), 'text': f"{existing.get('concept_name', '')}: {existing.get('definition', '')}"},
        }

    def reconcile_memory_batch(self, memories: list) -> list:
        """
        Cross-store contradiction sweep for drift RECONCILE action.

        Takes a list of memory dicts (trait/concept/episode), runs pairwise
        vector search within the batch, and classifies candidate pairs.

        Returns:
            list of classification dicts (same schema as check_ingestion)
        """
        results = []
        seen_pairs = set()

        # Build embedding map for fast comparison
        embedding_map = {}
        for mem in memories:
            emb = mem.get('embedding')
            if emb:
                embedding_map[mem['id']] = emb

        # Compare each memory against others by embedding similarity
        mem_list = [m for m in memories if m.get('id') in embedding_map]
        for i, mem_a in enumerate(mem_list):
            for mem_b in mem_list[i + 1:]:
                pair_key = frozenset({mem_a['id'], mem_b['id']})
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                sim = _cosine_similarity(
                    embedding_map[mem_a['id']],
                    embedding_map[mem_b['id']],
                )
                if sim < _SIMILARITY_THRESHOLD:
                    continue

                result = self._classify_pair_llm(
                    mem_a['text'], mem_b['text'],
                    context_hint=None,
                    meta_a=mem_a.get('meta', {}),
                    meta_b=mem_b.get('meta', {}),
                )
                if result is None:
                    continue

                classification = result.get('classification', 'compatible')
                if classification in ('compatible', 'figurative'):
                    continue

                results.append({
                    'classification': classification,
                    'confidence': result.get('confidence', 0.5),
                    'temporal_signal': result.get('temporal_signal', False),
                    'reasoning': result.get('reasoning', ''),
                    'surface_context': result.get('surface_context'),
                    'recommended_resolution': result.get('recommended_resolution', 'background_queue'),
                    'memory_a': mem_a,
                    'memory_b': mem_b,
                })

        return results

    # ── ONNX wrapper (primary path) ────────────────────────────────────────

    # Label mapping: ONNX model outputs single letters, callers expect class names.
    #
    # !! STATIC CONTRACT — must match CLASS_LABELS in
    # !! training/data/tasks/contradiction/__init__.py
    # !! and the Options line in _build_onnx_input().
    _ONNX_LABEL_TO_CLASS = {
        'A': 'temporal_change',
        'B': 'true_contradiction',
        'C': 'context_dependent',
        'D': 'figurative',
        'E': 'compatible',
    }

    # Memory type mapping: the ONNX model only knows these 4 types.
    # Any source/type not in this set must be mapped before inference.
    #
    # !! STATIC CONTRACT — must match MEMORY_TYPES in
    # !! training/data/tasks/contradiction/__init__.py
    _ONNX_KNOWN_TYPES = frozenset(['incoming', 'trait', 'concept', 'episode'])

    # Source-to-type mapping for values the backend uses internally but
    # the ONNX model has never seen.
    _SOURCE_TYPE_MAP = {
        'consolidation_new': 'concept',
        'consolidation_existing': 'concept',
    }

    # Minimum ONNX confidence to trust the classification.
    # Below this, fall back to LLM for higher-quality classification.
    _ONNX_CONFIDENCE_THRESHOLD = 0.80

    def _classify_pair_onnx(
        self,
        text_a: str,
        text_b: str,
        meta_a: dict,
        meta_b: dict,
    ) -> Optional[dict]:
        """
        Classify a memory pair using the ONNX contradiction model.

        Returns a dict with {classification, confidence, temporal_signal,
        recommended_resolution} or None if the model is unavailable or
        confidence is below threshold.

        !! WRAPPER CONTRACT — RETRAINING REQUIRED IF CHANGED !!
        =====================================================
        This method translates between the backend's internal representation
        and the ONNX model's trained input format. The following elements
        are FROZEN and must not be modified without retraining:

        1. INPUT FORMAT: The _build_onnx_input() method must produce the exact
           same format as training/data/tasks/contradiction/__init__.py::_format_input().
           This includes JSON field names, key order, the Options line, and the
           "Answer:" suffix.

        2. LABEL MAPPING: _ONNX_LABEL_TO_CLASS must match CLASS_LABELS in the
           training task. The model outputs A/B/C/D/E; this dict maps to names.

        3. TYPE VOCABULARY: The model only knows 'incoming', 'trait', 'concept',
           'episode'. Any other type_a/type_b value MUST be mapped via
           _SOURCE_TYPE_MAP or default to 'episode'.

        4. ESTABLISHED SIGNAL: Computed by _is_established() which has frozen
           thresholds. See that function's docstring.

        SAFE TO CHANGE (no retraining):
        - _ONNX_CONFIDENCE_THRESHOLD (post-model gating)
        - The deterministic resolution/temporal_signal inference below
        - Adding new entries to _SOURCE_TYPE_MAP (maps TO existing types)
        """
        try:
            from services.onnx_inference_service import get_onnx_inference_service
            svc = get_onnx_inference_service()

            input_text = self._build_onnx_input(text_a, text_b, meta_a, meta_b)
            label, confidence = svc.predict("contradiction", input_text)

            if label is None:
                return None

            if confidence < self._ONNX_CONFIDENCE_THRESHOLD:
                logger.info(
                    f"{LOG_PREFIX} ONNX confidence {confidence:.3f} below "
                    f"threshold {self._ONNX_CONFIDENCE_THRESHOLD} — using LLM"
                )
                return None

            classification = self._ONNX_LABEL_TO_CLASS.get(label, 'compatible')

            # Deterministic post-classification signals (not predicted by model)
            temporal_signal = classification == 'temporal_change'
            if classification == 'temporal_change':
                resolution = 'auto_supersede'
            elif classification in ('true_contradiction', 'context_dependent'):
                resolution = 'flag_response'
            else:
                resolution = 'ignore'

            logger.info(
                f"{LOG_PREFIX} ONNX classification: {classification} "
                f"(confidence={confidence:.3f}, resolution={resolution})"
            )

            return {
                'classification': classification,
                'confidence': confidence,
                'temporal_signal': temporal_signal,
                'reasoning': f'ONNX classifier ({confidence:.2f})',
                'surface_context': None,
                'recommended_resolution': resolution,
            }
        except Exception as e:
            logger.info(f"{LOG_PREFIX} ONNX classification failed: {e}")
            return None

    def _build_onnx_input(
        self,
        text_a: str,
        text_b: str,
        meta_a: dict,
        meta_b: dict,
    ) -> str:
        """
        Build the ONNX model input string from a memory pair.

        !! STATIC CONTRACT — RETRAINING REQUIRED IF CHANGED !!
        This method MUST produce output identical to:
            training/data/tasks/contradiction/__init__.py::_format_input()

        The format is:
            {JSON payload}
            Options: A: temporal_change | B: true_contradiction | C: context_dependent | D: figurative | E: compatible
            Answer:

        JSON fields (exact names, no extras):
            text_a, text_b, type_a, type_b, age_a_days, age_b_days,
            established_a, established_b
        """
        from services.time_utils import utc_now, parse_utc

        # Resolve memory types — map internal source types to model vocabulary
        type_a = meta_a.get('source', meta_a.get('type', 'incoming'))
        type_b = meta_b.get('source', meta_b.get('type', 'incoming'))
        type_a = self._SOURCE_TYPE_MAP.get(type_a, type_a)
        type_b = self._SOURCE_TYPE_MAP.get(type_b, type_b)
        # Final guard: unknown types default to 'episode'
        if type_a not in self._ONNX_KNOWN_TYPES:
            type_a = 'episode'
        if type_b not in self._ONNX_KNOWN_TYPES:
            type_b = 'episode'

        # Compute age in days
        now = utc_now()
        age_a = 0
        age_b = 0
        if meta_a.get('created_at'):
            try:
                age_a = (now - parse_utc(meta_a['created_at'])).days
            except Exception:
                pass
        if meta_b.get('created_at'):
            try:
                age_b = (now - parse_utc(meta_b['created_at'])).days
            except Exception:
                pass

        # Compute established signal
        established_a = meta_a.get('established', _is_established(type_a, meta_a))
        established_b = meta_b.get('established', _is_established(type_b, meta_b))

        # Build JSON payload — must match _format_input() exactly:
        # separators=(',', ':') for compact JSON, no spaces
        payload = json.dumps({
            "text_a": text_a,
            "text_b": text_b,
            "type_a": type_a,
            "type_b": type_b,
            "age_a_days": age_a,
            "age_b_days": age_b,
            "established_a": established_a,
            "established_b": established_b,
        }, separators=(',', ':'))

        return (
            f"{payload}\n"
            "Options: A: temporal_change | B: true_contradiction | "
            "C: context_dependent | D: figurative | E: compatible\n"
            "Answer:"
        )

    # ── LLM fallback ───────────────────────────────────────────────────────

    def _classify_pair_llm(
        self,
        text_a: str,
        text_b: str,
        context_hint: Optional[str],
        meta_a: dict,
        meta_b: dict,
    ) -> Optional[dict]:
        """
        Classify a memory pair — ONNX primary, LLM fallback.

        Flow:
          1. Try ONNX model (< 5ms, no context_hint needed)
          2. If ONNX unavailable or confidence < 0.80 → fall through to LLM
          3. LLM provides richer output (reasoning, surface_context, resolution)

        Returns parsed JSON dict or None on failure.
        """
        # ── ONNX primary path ──
        onnx_result = self._classify_pair_onnx(text_a, text_b, meta_a, meta_b)
        if onnx_result is not None:
            return onnx_result

        # ── LLM fallback ──
        user_parts = [
            f"Memory A: {text_a}",
            f"Memory B: {text_b}",
        ]
        if meta_a:
            user_parts.append(f"Memory A metadata: {json.dumps(meta_a)}")
        if meta_b:
            user_parts.append(f"Memory B metadata: {json.dumps(meta_b)}")
        if context_hint:
            user_parts.append(f"Current conversation context: {context_hint[:300]}")

        user_message = "\n\n".join(user_parts)

        try:
            from services.config_service import ConfigService
            from services.llm_service import create_llm_service
            # Use triage-tier config (fast, lightweight)
            try:
                config = ConfigService.resolve_agent_config('cognitive-triage')
            except Exception:
                config = ConfigService.resolve_agent_config('frontal-cortex')

            llm = create_llm_service(config)
            response = llm.send_message(_CLASSIFIER_SYSTEM_PROMPT, user_message)
            return _extract_json(response.text)
        except Exception as e:
            logger.info(f"{LOG_PREFIX} LLM classification failed: {e}")
            return None

    def _find_candidate_pairs_ingestion(
        self,
        embedding: list,
        text: str,
        start_time: float,
    ) -> list:
        """
        Vector-search traits and concepts for high-similarity matches to the
        user message. Returns list of (mem_a, mem_b) pairs where mem_a is
        the incoming statement and mem_b is the matched memory.
        """
        if self.db is None:
            return []

        try:
            from services.user_trait_service import _pack_embedding
            packed = _pack_embedding(embedding)
            if packed is None:
                return []
        except Exception:
            return []

        pairs = []
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                # Search traits
                try:
                    cursor.execute("""
                        SELECT t.id, t.trait_key, t.trait_value, t.confidence,
                               t.source, t.reinforcement_count, t.reliability,
                               t.created_at
                        FROM user_traits_vec v
                        JOIN user_traits t ON t.rowid = v.rowid
                        WHERE v.embedding MATCH ? AND k = 5
                        ORDER BY v.distance
                    """, (packed,))
                    trait_rows = cursor.fetchall()
                    for row in trait_rows:
                        mem_text = f"{row[1]}: {row[2]}"
                        meta = {
                            'confidence': row[3],
                            'source': row[4],
                            'reinforcement_count': row[5],
                            'reliability': row[6] or 'reliable',
                            'created_at': row[7],
                        }
                        meta['established'] = _is_established('trait', meta)
                        pairs.append((
                            {'type': 'incoming', 'id': None, 'text': text},
                            {
                                'type': 'trait',
                                'id': row[0],
                                'text': mem_text,
                                'meta': meta,
                            }
                        ))
                except Exception as e:
                    logger.debug(f"{LOG_PREFIX} Trait vector search failed: {e}")

                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > _INGESTION_TIMEOUT_MS:
                    cursor.close()
                    return pairs

                # Search concepts
                try:
                    cursor.execute("""
                        SELECT sc.id, sc.concept_name, sc.definition, sc.confidence,
                               sc.access_count, sc.reliability, sc.created_at
                        FROM concepts_vec v
                        JOIN semantic_concepts sc ON sc.id = v.id
                        WHERE v.embedding MATCH ? AND k = 5
                          AND sc.deleted_at IS NULL
                        ORDER BY v.distance
                    """, (packed,))
                    concept_rows = cursor.fetchall()
                    for row in concept_rows:
                        mem_text = f"{row[1]}: {row[2]}"
                        meta = {
                            'confidence': row[3],
                            'access_count': row[4],
                            'reliability': row[5] or 'reliable',
                            'created_at': row[6],
                        }
                        meta['established'] = _is_established('concept', meta)
                        pairs.append((
                            {'type': 'incoming', 'id': None, 'text': text},
                            {
                                'type': 'concept',
                                'id': row[0],
                                'text': mem_text,
                                'meta': meta,
                            }
                        ))
                except Exception as e:
                    logger.debug(f"{LOG_PREFIX} Concept vector search failed: {e}")

                cursor.close()
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Candidate pair search failed: {e}")

        return pairs

    def sample_memories_for_reconcile(self, n_traits: int = 5, n_concepts: int = 5) -> list:
        """
        Sample recent high-confidence traits and concepts for drift reconciliation.

        Returns list of memory dicts with keys:
            {id, type, text, embedding, meta}
        """
        if self.db is None:
            return []

        memories = []
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                # Sample top traits by confidence
                cursor.execute("""
                    SELECT t.id, t.trait_key, t.trait_value, t.confidence,
                           t.source, t.reinforcement_count, t.reliability,
                           v.embedding, t.created_at
                    FROM user_traits t
                    LEFT JOIN user_traits_vec v ON v.rowid = t.rowid
                    WHERE t.confidence > 0.3
                    ORDER BY t.confidence DESC, t.reinforcement_count DESC
                    LIMIT ?
                """, (n_traits,))
                for row in cursor.fetchall():
                    meta = {
                        'confidence': row[3],
                        'source': row[4],
                        'reinforcement_count': row[5],
                        'reliability': row[6] or 'reliable',
                        'created_at': row[8],
                    }
                    meta['established'] = _is_established('trait', meta)
                    memories.append({
                        'id': row[0],
                        'type': 'trait',
                        'text': f"{row[1]}: {row[2]}",
                        'embedding': _unpack_embedding(row[7]),
                        'meta': meta,
                    })

                # Sample top concepts by strength
                cursor.execute("""
                    SELECT sc.id, sc.concept_name, sc.definition, sc.confidence,
                           sc.access_count, sc.reliability,
                           v.embedding, sc.created_at
                    FROM semantic_concepts sc
                    LEFT JOIN concepts_vec v ON v.id = sc.id
                    WHERE sc.deleted_at IS NULL AND sc.confidence > 0.3
                    ORDER BY sc.access_count DESC, sc.strength DESC
                    LIMIT ?
                """, (n_concepts,))
                for row in cursor.fetchall():
                    meta = {
                        'confidence': row[3],
                        'access_count': row[4],
                        'reliability': row[5] or 'reliable',
                        'created_at': row[7],
                    }
                    meta['established'] = _is_established('concept', meta)
                    memories.append({
                        'id': row[0],
                        'type': 'concept',
                        'text': f"{row[1]}: {row[2]}",
                        'embedding': _unpack_embedding(row[6]),
                        'meta': meta,
                    })

                cursor.close()
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} sample_memories_for_reconcile failed: {e}")

        return memories

    def pair_already_tracked(self, id_a: str, id_b: str) -> bool:
        """Return True if this pair already has an open uncertainty record."""
        if self.db is None or not id_a or not id_b:
            return False
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 1 FROM uncertainties
                    WHERE state IN ('open', 'surfaced')
                      AND created_at > datetime('now', '-7 days')
                      AND (
                          (memory_a_id = ? AND memory_b_id = ?)
                          OR (memory_a_id = ? AND memory_b_id = ?)
                      )
                    LIMIT 1
                """, (id_a, id_b, id_b, id_a))
                row = cursor.fetchone()
                cursor.close()
                return row is not None
        except Exception:
            return False
