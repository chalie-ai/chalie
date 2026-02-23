import json
import re
import time
from services import ConfigService
from services.llm_service import create_llm_service
from services.world_state_service import WorldStateService
from services.thread_conversation_service import ThreadConversationService
from services.prompt_queue import enqueue_episodic_memory
from services.gist_storage_service import GistStorageService
from services.fact_store_service import FactStoreService
import logging


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response that may contain leading text or code fences."""
    # First try: strip code fences
    match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if match:
        logging.debug("[memory_chunker] Stripped code fence from LLM response")
        return match.group(1).strip()
    # Second try: find JSON object by scanning for first { and last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        logging.debug("[memory_chunker] Extracted JSON by boundary scan")
        return text[start:end + 1]
    return text


def load_config():
    """Load memory chunker configuration."""
    return {
        'config': ConfigService.resolve_agent_config("memory-chunker"),
        'prompt': ConfigService.get_agent_prompt("memory-chunker")
    }


def load_existing_gists(topic: str, min_confidence: int = 7) -> list:
    """Load existing gists for context injection from Redis."""
    config = ConfigService.resolve_agent_config("memory-chunker")
    gist_storage = GistStorageService(
        attention_span_minutes=config.get('attention_span_minutes', 30),
        min_confidence=min_confidence,
        max_gists=config.get('max_gists', 8),
    )
    return gist_storage.get_latest_gists(topic)


def generate_memory_chunk(topic: str, prompt_message: str, response_message: str, config: dict, prompt_template: str, thread_id: str = None) -> dict:
    """Generate memory chunk using LLM."""
    # Get world state
    world_state_service = WorldStateService()
    world_state = world_state_service.get_world_state(topic, thread_id=thread_id)

    # Load existing gists
    min_gist_confidence = config.get('min_gist_confidence', 7)
    try:
        existing_gists = load_existing_gists(thread_id or topic, min_gist_confidence)
    except Exception as e:
        logging.warning(f"[memory_chunker] Context loading failed, proceeding without gists: {e}")
        existing_gists = []

    # Format gists section
    gists_section = ""
    if existing_gists:
        gists_section = "\n# Existing Gists\n"
        for gist in existing_gists:
            content = gist.get('content', '')
            confidence = gist.get('confidence', 0)
            gists_section += f"- {content} (confidence: {confidence})\n"

    # Inject world state and gists into prompt template
    system_prompt = prompt_template.replace('{{world_state}}', world_state + gists_section)

    # Build user message with exchange context (supports single messages)
    if prompt_message and response_message:
        user_message = f"#Prompt\n{prompt_message}\n\n#Response\n{response_message}"
    elif prompt_message:
        user_message = f"#Message (user)\n{prompt_message}"
    elif response_message:
        user_message = f"#Message (assistant)\n{response_message}"
    else:
        return {'gists': [], 'scope': 'none', 'emotion': {}}

    # Send to LLM
    llm = create_llm_service(config)
    response = llm.send_message(system_prompt, user_message).text

    # Parse JSON response
    extracted = _extract_json(response)
    try:
        memory_chunk = json.loads(extracted)
    except json.JSONDecodeError:
        logging.error(f"[memory_chunker] JSON parse failed for topic '{topic}'")
        logging.debug(f"[memory_chunker] Raw LLM response: {response[:1000]}")
        raise

    return memory_chunk


def _compute_emotion_signals(memory_chunk: dict) -> dict:
    """
    Map memory chunker emotion scores to per-vector emotion signals.
    Returns {vector_name: signal_strength} where signal_strength is -1.0 to 1.0.
    """
    signals = {}
    emotion = memory_chunk.get('emotion', {})
    user_emotion = emotion.get('user', {})
    scope = memory_chunk.get('scope', {})

    # High user joy → reinforce warmth and playfulness
    joy = user_emotion.get('joy', 0) / 10.0
    if joy > 0.3:
        signals['warmth'] = joy * 0.5
        signals['playfulness'] = joy * 0.3

    # High user surprise → reinforce curiosity
    surprise = user_emotion.get('surprise', 0) / 10.0
    if surprise > 0.3:
        signals['curiosity'] = surprise * 0.4

    # High user anger/disgust → reduce assertiveness, increase warmth
    anger = user_emotion.get('anger', 0) / 10.0
    disgust = user_emotion.get('disgust', 0) / 10.0
    negative = max(anger, disgust)
    if negative > 0.3:
        signals['assertiveness'] = -negative * 0.3
        signals['warmth'] = signals.get('warmth', 0) + negative * 0.2

    # High intent + confidence → reinforce assertiveness
    intent = scope.get('intent', 0) / 10.0
    confidence = scope.get('confidence', 0) / 10.0
    if intent > 0.5 and confidence > 0.5:
        signals['assertiveness'] = signals.get('assertiveness', 0) + 0.2

    # High emotion scope → reinforce emotional_intensity (dampened)
    emotion_scope = scope.get('emotion', 0) / 10.0
    if emotion_scope > 0.4:
        signals['emotional_intensity'] = emotion_scope * 0.2

    return signals


def _compute_reward_signals(reward_value: float, vectors: dict) -> dict:
    """
    State-aware reward reinforcement.
    On positive reward: reinforce vectors whose activation > baseline (what worked).
    On negative reward: dampen vectors whose activation > baseline (what didn't work).
    """
    signals = {}

    if abs(reward_value) < 0.1:
        return signals

    for name, v in vectors.items():
        deviation = v['current_activation'] - v['baseline_weight']
        if abs(deviation) < 0.05:
            continue

        # Reinforce or dampen the deviation direction
        signals[name] = reward_value * 0.3 * (1.0 if deviation > 0 else -1.0)

    return signals


def _extract_and_store_traits(memory_chunk: dict, metadata: dict):
    """
    Extract user traits from memory chunk and store them.

    Args:
        memory_chunk: The LLM-generated memory chunk (may contain user_traits)
        metadata: Job metadata (may contain user_id for speaker detection)
    """
    traits = memory_chunk.get('user_traits', [])
    if not traits or not isinstance(traits, list):
        return

    from services.user_trait_service import UserTraitService
    from services.database_service import get_shared_db_service

    db_service = get_shared_db_service()
    trait_service = UserTraitService(db_service)
    speaker_confidence = trait_service.get_speaker_confidence(metadata)

    stored = 0
    for trait in traits:
        key = trait.get('key')
        value = trait.get('value')
        if not key or not value:
            continue

        # Normalize confidence from 0-10 scale to 0-1
        raw_confidence = trait.get('confidence', 5)
        confidence = max(0.0, min(1.0, raw_confidence / 10.0))

        success = trait_service.store_trait(
            trait_key=key.lower().strip(),
            trait_value=value.strip(),
            confidence=confidence,
            category=trait.get('category', 'general'),
            source=trait.get('source', 'inferred'),
            is_literal=trait.get('is_literal', True),
            speaker_confidence=speaker_confidence,
        )
        if success:
            stored += 1

    if stored > 0:
        logging.info(f"[memory_chunker] Stored {stored}/{len(traits)} user traits")


def _extract_and_store_communication_style(memory_chunk: dict, metadata: dict):
    """
    Extract communication style from memory chunk and store as a single structured trait.

    Stores one 'communication_style' trait per exchange as JSON value containing all
    dimensions. Merges with existing via EMA (0.3 * observed + 0.7 * existing) per dim.
    Skips if confidence < 3.

    Args:
        memory_chunk: The LLM-generated memory chunk (may contain communication_style)
        metadata: Job metadata
    """
    import json as _json

    comm_style = memory_chunk.get('communication_style', {})
    if not comm_style or not isinstance(comm_style, dict):
        return

    confidence_raw = comm_style.get('confidence', 0)
    if confidence_raw < 3:
        return

    # Normalize confidence from 0-10 to 0-1
    confidence = max(0.0, min(1.0, confidence_raw / 10.0))

    dimensions = {
        k: comm_style[k]
        for k in ('verbosity', 'directness', 'formality', 'abstraction_level',
                  'emotional_valence', 'certainty_level', 'challenge_appetite',
                  'depth_preference', 'pacing')
        if k in comm_style and isinstance(comm_style[k], (int, float)) and comm_style[k] > 0
    }
    if not dimensions:
        return

    from services.user_trait_service import UserTraitService
    from services.database_service import get_shared_db_service

    db_service = get_shared_db_service()
    trait_service = UserTraitService(db_service)

    # Load existing style for EMA merge
    existing = trait_service.get_communication_style()
    observation_count = existing.get('_observation_count', 0) if existing else 0

    # Use faster convergence during cold start (first 5 observations)
    ema_weight = 0.5 if observation_count < 5 else 0.3

    if existing:
        merged = {}
        for dim, val in dimensions.items():
            old_val = existing.get(dim, val)
            merged[dim] = round(ema_weight * val + (1 - ema_weight) * old_val, 2)
    else:
        merged = {k: round(float(v), 2) for k, v in dimensions.items()}

    # Track observation count to manage cold-start EMA weight
    merged['_observation_count'] = observation_count + 1

    trait_service.store_trait(
        trait_key='communication_style',
        trait_value=_json.dumps(merged),
        confidence=confidence,
        category='communication_style',
        source='inferred',
        is_literal=True,
    )
    logging.info(f"[memory_chunker] Stored communication_style: {merged} (confidence={confidence:.2f}, obs={merged['_observation_count']})")


# Micro-preference detection patterns: (regex, trait_key)
_MICRO_PREF_PATTERNS = [
    (r'\b(bullet|list|points)\s*(please|format|version|form)?\b', 'prefers_bullet_format'),
    (r'\b(short|brief|quick)\s*(version|answer|summary|response)?\b', 'prefers_concise'),
    (r'\b(too\s+much|too\s+long|too\s+verbose|too\s+detailed)\b', 'prefers_concise'),
    (r'\b(detail|explain\s+more|elaborate|expand|deeper|more\s+depth)\b', 'prefers_depth'),
    (r'\btoo\s+(short|brief|vague)\b', 'prefers_depth'),
    (r'\b(challenge\s+me|push\s+back|devil.s\s+advocate|stress.test|counterpoint)\b', 'enjoys_challenge'),
]


def _extract_and_store_micro_preferences(prompt_message: str):
    """
    Detect explicit format/style requests in the user's message and store them
    as micro-preference traits.

    No LLM call — pure regex on the user's message text.
    Uses existing store_trait() reinforcement mechanism for repeated signals.

    Args:
        prompt_message: The user's raw message text from the exchange.
    """
    import re as _re

    if not prompt_message or not isinstance(prompt_message, str):
        return

    text = prompt_message.lower()

    matched = []
    for pattern, trait_key in _MICRO_PREF_PATTERNS:
        if _re.search(pattern, text):
            matched.append(trait_key)

    if not matched:
        return

    try:
        from services.user_trait_service import UserTraitService
        from services.database_service import get_shared_db_service

        db_service = get_shared_db_service()
        trait_service = UserTraitService(db_service)

        for trait_key in matched:
            trait_service.store_trait(
                trait_key=trait_key,
                trait_value='true',
                confidence=0.7,
                category='micro_preference',
                source='explicit',
                is_literal=True,
            )

        logging.info(f"[memory_chunker] Stored micro-preferences: {matched}")
    except Exception as e:
        logging.warning(f"[memory_chunker] Micro-preference storage failed: {e}")


_CHALLENGE_POSITIVE_PATTERNS = [
    r'\b(but what about|on the other hand|interesting point|good point|hmm.{0,10}but)\b',
    r'\b(what if|why not|how would|let.s explore|dig deeper|go deeper)\b',
    r'\b(fair enough|you.re right.{0,10}but|I see.{0,10}however|that.s true.{0,10}but)\b',
]
_CHALLENGE_NEGATIVE_PATTERNS = [
    r'\b(whatever|fine|ok move on|let.s just|anyway|never\s*mind)\b',
    r'\b(that.s not what I|you.re missing the point|no[,.]|just stop)\b',
    r'\bthat.s (too much|not helpful|off track|beside the point)\b',
]
_CHALLENGE_ASSISTANT_INDICATORS = [
    r'\b(on the other hand|devil.s advocate|counterpoint|have you considered|push back|challenge)\b',
    r'\b(however|but consider|what if instead|the flip side|alternative view)\b',
]


def _detect_challenge_reaction(prompt_message: str, response_message: str):
    """
    Detect how the user reacted to challenge in the previous response and
    update challenge_tolerance accordingly.

    Fires only when the assistant's response contains challenge indicators.
    No LLM call — regex pattern matching on exchange text.

    Args:
        prompt_message: The user's message (reaction to check)
        response_message: The assistant's previous response (checked for challenge content)
    """
    import re as _re

    if not prompt_message or not response_message:
        return

    # Only fire if assistant's response contained challenge content
    response_lower = response_message.lower()
    had_challenge = any(
        _re.search(p, response_lower)
        for p in _CHALLENGE_ASSISTANT_INDICATORS
    )
    if not had_challenge:
        return

    # Classify user's reaction
    user_lower = prompt_message.lower()
    is_positive = any(_re.search(p, user_lower) for p in _CHALLENGE_POSITIVE_PATTERNS)
    is_negative = any(_re.search(p, user_lower) for p in _CHALLENGE_NEGATIVE_PATTERNS)

    if not is_positive and not is_negative:
        return  # Neutral — no update

    try:
        from services.user_trait_service import UserTraitService
        from services.database_service import get_shared_db_service
        import json as _json

        db_service = get_shared_db_service()
        trait_service = UserTraitService(db_service)

        # Read existing tolerance
        existing_rows = None
        try:
            with db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT trait_value FROM user_traits WHERE user_id = 'primary' "
                    "AND trait_key = 'challenge_tolerance' LIMIT 1"
                )
                row = cursor.fetchone()
                cursor.close()
                existing_rows = float(row[0]) if row else None
        except Exception:
            pass

        current = existing_rows if existing_rows is not None else 5.0

        # Apply slow EMA update (0.2 new + 0.8 old)
        if is_positive:
            new_val = round(0.2 * min(current + 1.5, 10.0) + 0.8 * current, 2)
            direction = 'positive'
        else:
            new_val = round(0.2 * max(current - 1.5, 1.0) + 0.8 * current, 2)
            direction = 'negative'

        trait_service.store_trait(
            trait_key='challenge_tolerance',
            trait_value=str(new_val),
            confidence=0.6,
            category='micro_preference',
            source='inferred',
            is_literal=True,
        )
        logging.info(f"[memory_chunker] challenge_tolerance updated: {current} → {new_val} ({direction} reaction)")
    except Exception as e:
        logging.warning(f"[memory_chunker] Challenge reaction detection failed: {e}")


def _apply_identity_reinforcement(topic: str, memory_chunk: dict):
    """Apply dual-channel identity reinforcement after chunk extraction."""
    from services.redis_client import RedisClientService

    emotion_signals = _compute_emotion_signals(memory_chunk)

    # Read last reward signal from Redis (1-exchange lag)
    redis_conn = RedisClientService.create_connection()
    reward_raw = redis_conn.get(f"identity_reward:{topic}")
    last_reward = float(reward_raw) if reward_raw else 0.0

    # Need vectors for state-aware reward computation
    from services.identity_service import IdentityService
    from services.database_service import get_shared_db_service

    db_service = get_shared_db_service()
    identity_service = IdentityService(db_service)
    vectors = identity_service.get_vectors()
    reward_signals = _compute_reward_signals(last_reward, vectors)

    # Merge: union of all vectors touched by either channel
    all_vectors = set(list(emotion_signals.keys()) + list(reward_signals.keys()))

    for vector_name in all_vectors:
        emotion_val = emotion_signals.get(vector_name, 0.0)
        reward_val = reward_signals.get(vector_name, 0.0)
        identity_service.update_activation(vector_name, emotion_val, reward_val, topic=topic)

    if all_vectors:
        logging.info(f"[memory_chunker] Identity reinforcement: {len(all_vectors)} vectors updated for topic '{topic}'")


def memory_chunker_worker(job_data: dict) -> str:
    """
    Worker function that generates memory chunks for conversation exchanges.

    Args:
        job_data: Dict with topic, exchange_id, prompt_message, response_message

    Returns:
        str: Status message
    """
    import signal

    # Hard timeout to prevent infinite hangs
    def timeout_handler(signum, frame):
        raise TimeoutError("Memory chunker job exceeded hard timeout")

    # Set signal alarm for 300 seconds (5 minutes) - well under RQ 600s job timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)

    try:
        topic = job_data['topic']
        exchange_id = job_data['exchange_id']
        thread_id = job_data.get('thread_id') or None
        prompt_message = job_data.get('prompt_message', '')
        response_message = job_data.get('response_message', '')

        # Support single-message encoding (per-message cycle architecture)
        is_single_message = not prompt_message or not response_message
        msg_type = 'single' if is_single_message else 'pair'

        logging.info(f"log [memory_chunker]: Processing topic '{topic}' ({msg_type}, exchange {exchange_id[:8]}...)")

        # Load config and prompt
        config_data = load_config()

        # Generate memory chunk
        try:
            start_time = time.time()
            memory_chunk = generate_memory_chunk(
                topic,
                prompt_message,
                response_message,
                config_data['config'],
                config_data['prompt'],
                thread_id=thread_id,
            )
            generation_time = time.time() - start_time

            # Add memory chunk to specific exchange
            if thread_id:
                thread_conv_service = ThreadConversationService()
                stored = thread_conv_service.add_memory_chunk(thread_id, exchange_id, memory_chunk)
                if not stored:
                    logging.error(f"[memory_chunker] Exchange {exchange_id[:8]} not found in thread {thread_id} — memory chunk lost")
                    raise RuntimeError(f"Exchange {exchange_id[:8]} not found in thread {thread_id}")

            # Store gists in Redis with TTL and confidence filtering
            gists = memory_chunk.get('gists', [])
            if gists:
                config = config_data['config']
                attention_span_minutes = config.get('attention_span_minutes', 30)
                min_confidence = config.get('min_gist_confidence', 7)
                max_gists = config.get('max_gists', 8)

                gist_storage = GistStorageService(
                    attention_span_minutes=attention_span_minutes,
                    min_confidence=min_confidence,
                    max_gists=max_gists,
                    similarity_threshold=config.get('gist_similarity_threshold', 0.7),
                    max_per_type=config.get('max_gists_per_type', 2)
                )

                stored_count = gist_storage.store_gists(
                    topic=topic,
                    gists=gists,
                    prompt=prompt_message,
                    response=response_message
                )

                logging.info(f"log [memory_chunker]: Stored {stored_count}/{len(gists)} gists in Redis for topic '{topic}'")

            # Identity reinforcement: dual-channel (emotion + reward) → update vectors
            try:
                _apply_identity_reinforcement(topic, memory_chunk)
            except TimeoutError:
                raise
            except Exception as e:
                logging.warning(f"[memory_chunker] Identity reinforcement failed: {e}")

            # Enqueue episodic memory job for this topic (will check readiness in worker)
            logging.info(f"log [memory_chunker]: Enqueueing episodic memory job for topic '{topic}'")
            enqueue_episodic_memory({'topic': topic, 'thread_id': thread_id or ''})

            # Extract and store facts from memory chunk (same LLM call, no second LLM needed)
            try:
                facts = memory_chunk.get('facts', [])
                if facts:
                    fact_store_config = ConfigService.get_agent_config("fact-store")
                    min_confidence = fact_store_config.get('min_confidence', 0.5)
                    fact_store = FactStoreService(
                        ttl_minutes=fact_store_config.get('ttl_minutes', 1440),
                        max_facts_per_topic=fact_store_config.get('max_facts_per_topic', 50)
                    )
                    stored_facts = 0
                    for fact in facts:
                        key = fact.get('key')
                        value = fact.get('value')
                        # Normalize from 0-10 scale to 0.0-1.0
                        raw_confidence = fact.get('confidence', 5)
                        confidence = max(0.0, min(1.0, raw_confidence / 10.0))
                        if key and value and confidence >= min_confidence:
                            fact_store.store_fact(
                                topic=topic, key=key, value=value,
                                confidence=confidence, source=exchange_id
                            )
                            stored_facts += 1
                    if stored_facts > 0:
                        logging.info(f"[memory_chunker] Stored {stored_facts}/{len(facts)} facts for '{topic}'")
            except TimeoutError:
                raise
            except Exception as e:
                logging.warning(f"[memory_chunker] Fact storage failed: {e}")

            # Extract and store user traits from the same memory chunk
            try:
                metadata = job_data.get('metadata', {})
                _extract_and_store_traits(memory_chunk, metadata)
            except TimeoutError:
                raise
            except Exception as e:
                logging.warning(f"[memory_chunker] Trait extraction failed: {e}")

            # Extract and store communication style from the same memory chunk
            try:
                metadata = job_data.get('metadata', {})
                _extract_and_store_communication_style(memory_chunk, metadata)
            except TimeoutError:
                raise
            except Exception as e:
                logging.warning(f"[memory_chunker] Communication style extraction failed: {e}")

            # Detect explicit micro-preferences from user's message (no LLM call)
            try:
                _extract_and_store_micro_preferences(prompt_message)
            except TimeoutError:
                raise
            except Exception as e:
                logging.warning(f"[memory_chunker] Micro-preference extraction failed: {e}")

            # Detect challenge reaction and update challenge_tolerance (no LLM call)
            try:
                _detect_challenge_reaction(prompt_message, response_message)
            except TimeoutError:
                raise
            except Exception as e:
                logging.warning(f"[memory_chunker] Challenge reaction detection failed: {e}")

            return f"Topic '{topic}' | Memory chunk generated in {generation_time:.2f}s"

        except json.JSONDecodeError as e:
            logging.error(f"[memory_chunker] Invalid JSON from LLM for topic '{topic}': {e}")
            raise
        except Exception as e:
            return f"Topic '{topic}' | ERROR: {str(e)}"
    finally:
        # Cancel alarm
        signal.alarm(0)
